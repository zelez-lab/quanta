//! emit_op() — match dispatcher over KernelOp variants.

use crate::*;
use std::collections::{BTreeMap, HashMap};

use super::helpers::*;

/// `(exp_bits, mant_bits)` for an fp8 scalar type, else `None`.
fn fp8_dims(ty: &ScalarType) -> Option<(u32, u32)> {
    match ty {
        ScalarType::FP8E5M2 => Some((5, 2)),
        ScalarType::FP8E4M3 => Some((4, 3)),
        _ => None,
    }
}

/// Destination lvalue for a register write.
///
/// The KernelOp contract is mutable-register semantics: a register may be
/// written more than once (loop-carried accumulators, `Copy` reassignment)
/// or written inside a Branch arm / Loop body and read after the merge.
/// Registers flagged by the `reg_mutability` pre-pass are declared once at
/// function entry (see `kernel::emit`), so every write is a plain
/// assignment; re-emitting `<type> rN = ...` would be a C++ redefinition.
/// Single-def registers keep the inline typed declaration — pure SSA.
fn dst_lv(mutable: &BTreeMap<u32, ScalarType>, ty: &str, reg: u32) -> String {
    if mutable.contains_key(&reg) {
        format!("r{}", reg)
    } else {
        format!("{} r{}", ty, reg)
    }
}

pub(super) fn emit_op(
    out: &mut String,
    op: &KernelOp,
    indent: usize,
    names: &HashMap<u32, String>,
    int_consts: &HashMap<u32, i64>,
    mutable: &BTreeMap<u32, ScalarType>,
) {
    let pad = "    ".repeat(indent);
    match op {
        KernelOp::Const { dst, value } => {
            let (ty, val) = const_msl(value);
            out.push_str(&format!(
                "{}{} = {};\n",
                pad,
                dst_lv(mutable, ty, dst.0),
                val
            ));
        }
        KernelOp::QuarkId { dst } => out.push_str(&format!(
            "{}{} = _quark_id;\n",
            pad,
            dst_lv(mutable, "uint", dst.0)
        )),
        KernelOp::ProtonId { dst } => out.push_str(&format!(
            "{}{} = _proton_id;\n",
            pad,
            dst_lv(mutable, "uint", dst.0)
        )),
        KernelOp::NucleusId { dst } => out.push_str(&format!(
            "{}{} = _nucleus_id;\n",
            pad,
            dst_lv(mutable, "uint", dst.0)
        )),
        KernelOp::ProtonSize { dst } => out.push_str(&format!(
            "{}{} = _proton_size;\n",
            pad,
            dst_lv(mutable, "uint", dst.0)
        )),
        KernelOp::QuarkCount { dst } => out.push_str(&format!(
            "{}{} = _nucleus_id * _proton_size + _proton_size;\n",
            pad,
            dst_lv(mutable, "uint", dst.0)
        )),
        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            let elem = if index.0 == u32::MAX {
                n.to_string()
            } else {
                format!("{}[r{}]", n, index.0)
            };
            if matches!(ty, ScalarType::BF16) {
                // bf16 storage is `uint` (one per word); unpack to float.
                out.push_str(&format!(
                    "{}{} = as_type<float>({} << 16);\n",
                    pad,
                    dst_lv(mutable, "float", dst.0),
                    elem
                ));
            } else if let Some((eb, mb)) = fp8_dims(ty) {
                // fp8 storage is `uint` (one byte per word); unpack to float.
                let tag = crate::dtype_codegen::fp8_tag(eb, mb);
                out.push_str(&format!(
                    "{}{} = qa_fp8_{}_unpack({});\n",
                    pad,
                    dst_lv(mutable, "float", dst.0),
                    tag,
                    elem
                ));
            } else if matches!(ty, ScalarType::I4) {
                // int4 PackedU32: element i lives in word i/8, nibble i%8;
                // load sign-extends the 4-bit value: ((n ^ 8) - 8).
                // Brace-scope the `_sh`/`_n` temps when the destination is a
                // hoisted mutable register so a re-load of the same register
                // does not redefine them.
                let (open, close) = if mutable.contains_key(&dst.0) {
                    ("{ ", " }")
                } else {
                    ("", "")
                };
                out.push_str(&format!(
                    "{pad}{open}uint r{d}_sh = (r{i} % 8u) * 4u; \
                     uint r{d}_n = ({n}[r{i} / 8u] >> r{d}_sh) & 0xFu; \
                     {lv} = int((r{d}_n ^ 0x8u) - 0x8u);{close}\n",
                    pad = pad,
                    open = open,
                    close = close,
                    d = dst.0,
                    i = index.0,
                    n = n,
                    lv = dst_lv(mutable, "int", dst.0)
                ));
            } else {
                out.push_str(&format!(
                    "{}{} = {};\n",
                    pad,
                    dst_lv(mutable, ty.msl_name(), dst.0),
                    elem
                ));
            }
        }
        KernelOp::Store {
            field,
            index,
            src,
            ty,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            if matches!(ty, ScalarType::BF16) {
                // Pack float → bf16 (round-to-nearest-even), store as uint.
                out.push_str(&format!(
                    "{}uint r{}_b = as_type<uint>(r{}); \
                     {}[r{}] = (r{}_b + 0x7fffu + ((r{}_b >> 16) & 1u)) >> 16;\n",
                    pad, src.0, src.0, n, index.0, src.0, src.0
                ));
            } else if let Some((eb, mb)) = fp8_dims(ty) {
                // Pack float → fp8 (round-to-nearest-even), store as uint.
                let tag = crate::dtype_codegen::fp8_tag(eb, mb);
                out.push_str(&format!(
                    "{}{}[r{}] = qa_fp8_{}_pack(r{});\n",
                    pad, n, index.0, tag, src.0
                ));
            } else if matches!(ty, ScalarType::I4) {
                // int4 PackedU32: write nibble i%8 of word i/8 (read-modify-
                // write; single-quark in the op-matrix).
                out.push_str(&format!(
                    "{pad}uint r{s}_w = r{i} / 8u; \
                     uint r{s}_sh = (r{i} % 8u) * 4u; \
                     {n}[r{s}_w] = ({n}[r{s}_w] & ~(0xFu << r{s}_sh)) \
                     | ((uint(r{s}) & 0xFu) << r{s}_sh);\n",
                    pad = pad,
                    s = src.0,
                    i = index.0,
                    n = n
                ));
            } else {
                out.push_str(&format!("{}{}[r{}] = r{};\n", pad, n, index.0, src.0));
            }
        }
        KernelOp::BinOp { dst, a, b, op, ty } => {
            match op {
                BinOp::Rotl => {
                    // MSL: `rotate(x, k)` rotates left by k positions.
                    out.push_str(&format!(
                        "{}{} = rotate(r{}, r{});\n",
                        pad,
                        dst_lv(mutable, ty.msl_name(), dst.0),
                        a.0,
                        b.0
                    ));
                }
                BinOp::Rotr => {
                    // MSL doesn't have a rotate-right built-in. Rewrite
                    // as rotate-left by (width - k mod width).
                    let width: u32 = match ty {
                        ScalarType::U8 | ScalarType::I8 => 8,
                        ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
                        ScalarType::U32
                        | ScalarType::I32
                        | ScalarType::F32
                        | ScalarType::BF16
                        | ScalarType::FP8E5M2
                        | ScalarType::FP8E4M3
                        | ScalarType::I4 => 32,
                        ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
                        ScalarType::Bool => 1,
                    };
                    out.push_str(&format!(
                        "{}{} = rotate(r{}, ({}) - (r{} % {}));\n",
                        pad,
                        dst_lv(mutable, ty.msl_name(), dst.0),
                        a.0,
                        width,
                        b.0,
                        width
                    ));
                }
                BinOp::Shr | BinOp::Shl => {
                    // Cast operands to the BinOp's `ty` before shifting so
                    // the shift's signedness matches the IR's intent.
                    // Without this, an unsigned shift fed by a signed
                    // register (e.g. `u32 r = signed_xor >> 8`) compiles
                    // in MSL as an arithmetic shift in `int` and then
                    // assigns to `uint`, producing sign-extended garbage
                    // for the half of lanes where bit 31 is set. The
                    // WASM-route translator types every i32 XOR/AND/OR
                    // result as I32 regardless of source signedness, so
                    // this case is common.
                    let o = binop_str(op);
                    let t = ty.msl_name();
                    out.push_str(&format!(
                        "{}{} = ({})r{} {} ({})r{};\n",
                        pad,
                        dst_lv(mutable, t, dst.0),
                        t,
                        a.0,
                        o,
                        t,
                        b.0
                    ));
                }
                BinOp::SatAdd => {
                    // MSL has no saturating-add operator; expand to an
                    // overflow check. Wrapping-add then detect overflow
                    // via `_sum < a` (unsigned). Saturation literal must
                    // match the operand width — u32 saturates to
                    // 0xFFFFFFFFu, u64 to 0xFFFFFFFFFFFFFFFFul, etc.
                    // Brace-scope the `_sum` temp when the destination is
                    // a hoisted mutable register so a re-write of the same
                    // register does not redefine it.
                    let t = ty.msl_name();
                    let max_lit = unsigned_max_lit_msl(ty);
                    let (open, close) = if mutable.contains_key(&dst.0) {
                        ("{ ", " }")
                    } else {
                        ("", "")
                    };
                    out.push_str(&format!(
                        "{}{}{} _sum_{} = r{} + r{}; {} = (_sum_{} < r{}) ? ({}){} : _sum_{};{}\n",
                        pad,
                        open,
                        t,
                        dst.0,
                        a.0,
                        b.0,
                        dst_lv(mutable, t, dst.0),
                        dst.0,
                        a.0,
                        t,
                        max_lit,
                        dst.0,
                        close,
                    ));
                }
                BinOp::SatSub => {
                    // MSL has no saturating-sub operator; clamp to 0 on
                    // underflow (unsigned semantics). Mirrors AOT path.
                    let t = ty.msl_name();
                    out.push_str(&format!(
                        "{}{} = (r{} < r{}) ? ({})0 : r{} - r{};\n",
                        pad,
                        dst_lv(mutable, t, dst.0),
                        a.0,
                        b.0,
                        t,
                        a.0,
                        b.0,
                    ));
                }
                _ => {
                    let o = binop_str(op);
                    out.push_str(&format!(
                        "{}{} = r{} {} r{};\n",
                        pad,
                        dst_lv(mutable, ty.msl_name(), dst.0),
                        a.0,
                        o,
                        b.0
                    ));
                }
            }
        }
        KernelOp::Cmp { dst, a, b, op, ty } => {
            let o = cmpop_str(op);
            // Cast both operands to the comparison type for integer
            // compares. Registers feeding a Cmp can carry mixed
            // int/uint MSL declarations (the lowerer's Copy ops retype
            // freely — SSA ints are canonically unsigned); C's usual
            // arithmetic conversions would then promote to `uint` and
            // silently turn an `I32` signed compare into an unsigned
            // one (observed: the i32 min tree-reduce returning the
            // smallest *positive* value on Metal).
            match ty {
                ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::I4
                | ScalarType::U8
                | ScalarType::U16
                | ScalarType::U32
                | ScalarType::U64 => {
                    let c = ty.msl_name();
                    out.push_str(&format!(
                        "{}{} = (({})r{} {} ({})r{});\n",
                        pad,
                        dst_lv(mutable, "bool", dst.0),
                        c,
                        a.0,
                        o,
                        c,
                        b.0
                    ));
                }
                _ => {
                    out.push_str(&format!(
                        "{}{} = (r{} {} r{});\n",
                        pad,
                        dst_lv(mutable, "bool", dst.0),
                        a.0,
                        o,
                        b.0
                    ));
                }
            }
        }
        KernelOp::UnaryOp { dst, a, op, ty } => {
            let o = match op {
                UnaryOp::Neg => "-",
                UnaryOp::BitNot => "~",
                UnaryOp::LogicalNot => "!",
            };
            out.push_str(&format!(
                "{}{} = {}r{};\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                o,
                a.0
            ));
        }
        KernelOp::Cast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}{} = ({})r{};\n",
                pad,
                dst_lv(mutable, to.msl_name(), dst.0),
                to.msl_name(),
                src.0
            ));
        }
        KernelOp::MathCall {
            dst,
            func,
            args,
            ty,
        } => {
            let f = math_fn_str(func);
            let a: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}{} = {}({});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                f,
                a.join(", ")
            ));
        }
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            out.push_str(&format!("{}if (r{}) {{\n", pad, cond.0));
            for op in then_ops {
                emit_op(out, op, indent + 1, names, int_consts, mutable);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_op(out, op, indent + 1, names, int_consts, mutable);
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            // T1405: when `count` was defined by a Const op with a
            // small positive value, emit `#pragma clang loop unroll(full)`
            // so the Metal compiler (Clang-based) fully unrolls. Mirrors
            // the SPIR-V emitter's LOOP_CONTROL_UNROLL hint. Threshold
            // shared via `crate::const_analysis::should_unroll_loop_count`.
            let count_val = int_consts.get(&count.0).copied();
            if crate::const_analysis::should_unroll_loop_count(count_val) {
                out.push_str(&format!("{}#pragma clang loop unroll(full)\n", pad));
            }
            // A mutable iter_reg (e.g. re-written in the body) was hoisted
            // to a function-entry declaration; don't re-declare it in the
            // `for` header.
            if mutable.contains_key(&iter_reg.0) {
                out.push_str(&format!(
                    "{}for (r{} = 0u; r{} < r{}; r{}++) {{\n",
                    pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0
                ));
            } else {
                out.push_str(&format!(
                    "{}for (uint r{} = 0; r{} < r{}; r{}++) {{\n",
                    pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0
                ));
            }
            for op in body {
                emit_op(out, op, indent + 1, names, int_consts, mutable);
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Copy { dst, src, ty } => {
            // Copy is the canonical reassignment op (the wasm-route
            // lowering's local.set): its dst is usually a hoisted mutable
            // register (plain assignment). A single-def Copy dst still
            // needs its typed declaration.
            out.push_str(&format!(
                "{}{} = r{};\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        // Per-tensor symmetric quantize: q = clamp(int(rint(x/scale)), lo, hi).
        // MSL `rint` is round-half-to-even (MSL `round` is half-away-from-
        // zero); this matches the CPU oracle + WGSL/SPIR-V.
        KernelOp::Quantize {
            dst,
            src,
            scale,
            scheme,
            ..
        } => {
            let (lo, hi) = scheme.value.range();
            out.push_str(&format!(
                "{}{} = clamp(int(rint(r{} / r{})), {}, {});\n",
                pad,
                dst_lv(mutable, "int", dst.0),
                src.0,
                scale.0,
                lo,
                hi
            ));
        }
        // Per-tensor symmetric dequantize: dq = scale * float(q).
        KernelOp::Dequantize {
            dst, src, scale, ..
        } => {
            out.push_str(&format!(
                "{}{} = r{} * float(r{});\n",
                pad,
                dst_lv(mutable, "float", dst.0),
                scale.0,
                src.0
            ));
        }
        KernelOp::Break => out.push_str(&format!("{}break;\n", pad)),
        KernelOp::Barrier => out.push_str(&format!(
            "{}threadgroup_barrier(mem_flags::mem_threadgroup);\n",
            pad
        )),
        // MSL: atomic_thread_fence with an explicit memory_order. Scope
        // is mem_device because Fence is meant for inter-quark visibility
        // of storage-buffer writes, not just the threadgroup.
        KernelOp::Fence { order } => out.push_str(&format!(
            "{}atomic_thread_fence(mem_flags::mem_device, {});\n",
            pad,
            match order {
                crate::MemoryOrder::Relaxed => "memory_order_relaxed",
                crate::MemoryOrder::Acquire => "memory_order_acquire",
                crate::MemoryOrder::Release => "memory_order_release",
                crate::MemoryOrder::AcqRel => "memory_order_acq_rel",
                crate::MemoryOrder::SeqCst => "memory_order_seq_cst",
            }
        )),
        KernelOp::SharedDecl { id, ty, count } => {
            out.push_str(&format!(
                "{}threadgroup {} shared_{}[{}];\n",
                pad,
                ty.msl_name(),
                id,
                count
            ));
        }
        KernelOp::SharedLoad { dst, id, index, ty } => {
            out.push_str(&format!(
                "{}{} = shared_{}[r{}];\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                id,
                index.0
            ));
        }
        KernelOp::SharedStore { id, index, src, .. } => {
            out.push_str(&format!(
                "{}shared_{}[r{}] = r{};\n",
                pad, id, index.0, src.0
            ));
        }
        // Metal restricts `device`-address-space atomics to
        // `memory_order_relaxed` (per MSL spec — the per-op ordering
        // overload is gated to `threadgroup` storage). Stronger
        // orderings on the IR's AtomicOp must be expressed via a
        // surrounding `KernelOp::Fence`. Storage-buffer atomics on
        // Metal are therefore always emitted as relaxed regardless of
        // the IR's `order` field.
        KernelOp::AtomicOp {
            dst,
            field,
            index,
            val,
            op,
            ty,
            order: _,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            let f = atomic_fn_str(op);
            out.push_str(&format!(
                "{}{} = {}((device atomic_{}*)&{}[r{}], r{}, memory_order_relaxed);\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                f,
                ty.msl_name(),
                n,
                index.0,
                val.0
            ));
        }
        // Threadgroup-storage atomic. MSL spec permits per-op
        // `memory_order` on `threadgroup` atomics (unlike `device`
        // atomics which Metal pins to relaxed), but for simplicity we
        // emit relaxed here too — any stronger ordering the user
        // requires is expressed via a surrounding `KernelOp::Fence`.
        // The `(threadgroup atomic_T*)` cast reinterprets the bare
        // `threadgroup T*` element pointer; Apple's MSL accepts this
        // reinterpretation as long as the underlying storage is
        // appropriately aligned (true for the 4-byte u32/i32 element
        // types we expose).
        KernelOp::SharedAtomicOp {
            dst,
            slot,
            index,
            val,
            op,
            ty,
            order: _,
        } => {
            let f = atomic_fn_str(op);
            out.push_str(&format!(
                "{}{} = {}((threadgroup atomic_{}*)&shared_{}[r{}], r{}, memory_order_relaxed);\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                f,
                ty.msl_name(),
                slot,
                index.0,
                val.0
            ));
        }
        KernelOp::WaveShuffle {
            dst,
            src,
            lane_delta,
            ty,
        } => {
            out.push_str(&format!(
                "{}{} = simd_shuffle_xor(r{}, r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0,
                lane_delta.0
            ));
        }
        KernelOp::WaveBallot { dst, predicate } => {
            // Metal's `simd_ballot` returns `metal::simd_vote`, an
            // opaque struct convertible to `uint64_t` via cast.
            // We take the low 32 bits — Quanta's subgroups never
            // exceed 64 lanes, and the wire format is u32.
            out.push_str(&format!(
                "{}{} = (uint)((uint64_t)simd_ballot(r{} != 0));\n",
                pad,
                dst_lv(mutable, "uint", dst.0),
                predicate.0
            ));
        }
        KernelOp::WaveAny { dst, predicate } => {
            out.push_str(&format!(
                "{}{} = uint(simd_any(r{} != 0));\n",
                pad,
                dst_lv(mutable, "uint", dst.0),
                predicate.0
            ));
        }
        KernelOp::WaveAll { dst, predicate } => {
            out.push_str(&format!(
                "{}{} = uint(simd_all(r{} != 0));\n",
                pad,
                dst_lv(mutable, "uint", dst.0),
                predicate.0
            ));
        }
        KernelOp::DeviceCall {
            dst,
            func_name,
            args,
            ty,
        } => {
            let a: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}{} = {}({});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                func_name,
                a.join(", ")
            ));
        }
        KernelOp::TextureSample2D {
            dst,
            texture,
            x,
            y,
            ty,
        } => {
            out.push_str(&format!(
                "{}{} = tex_{}.sample(samp_{}, float2(r{}, r{}));\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                texture,
                texture,
                x.0,
                y.0
            ));
        }
        KernelOp::TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ty,
        } => {
            out.push_str(&format!(
                "{}{} = tex_{}.sample(samp_{}, float3(r{}, r{}, r{}));\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                texture,
                texture,
                x.0,
                y.0,
                z.0
            ));
        }
        KernelOp::TextureWrite2D {
            texture,
            x,
            y,
            value,
            ..
        } => {
            out.push_str(&format!(
                "{}tex_{}.write(r{}, uint2(r{}, r{}));\n",
                pad, texture, value.0, x.0, y.0
            ));
        }
        KernelOp::TextureSize {
            dst_w,
            dst_h,
            texture,
        } => {
            out.push_str(&format!(
                "{}{} = tex_{}.get_width();\n",
                pad,
                dst_lv(mutable, "uint", dst_w.0),
                texture
            ));
            out.push_str(&format!(
                "{}{} = tex_{}.get_height();\n",
                pad,
                dst_lv(mutable, "uint", dst_h.0),
                texture
            ));
        }
        // Vector-typed registers are never demoted by the pre-pass (no
        // scalar slot type), so the inline declaration is always correct.
        KernelOp::VecConstruct {
            dst,
            components,
            ty,
        } => {
            let n = components.len();
            let comps: Vec<String> = components.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}{}{} r{} = {}{}({});\n",
                pad,
                ty.msl_name(),
                n,
                dst.0,
                ty.msl_name(),
                n,
                comps.join(", ")
            ));
        }
        KernelOp::VecExtract {
            dst,
            vec,
            component,
            ty,
        } => {
            let swizzle = match component {
                0 => "x",
                1 => "y",
                2 => "z",
                _ => "w",
            };
            out.push_str(&format!(
                "{}{} = r{}.{};\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                vec.0,
                swizzle
            ));
        }
        KernelOp::MatMul { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} = r{} * r{};\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                a.0,
                b.0
            ));
        }
        KernelOp::Bitcast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}{} = as_type<{}>(r{});\n",
                pad,
                dst_lv(mutable, to.msl_name(), dst.0),
                to.msl_name(),
                src.0
            ));
        }
        KernelOp::CountTrailingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = ctz(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::CountLeadingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = clz(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::PopCount { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = popcount(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::Dot { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} = dot(r{}, r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                a.0,
                b.0
            ));
        }
        KernelOp::SubgroupReduceAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = simd_sum(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::SubgroupReduceMin { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = simd_min(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::SubgroupReduceMax { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = simd_max(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = simd_prefix_exclusive_sum(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} = simd_prefix_inclusive_sum(r{});\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                src.0
            ));
        }
        KernelOp::TextureLoad2D {
            dst,
            texture,
            x,
            y,
            ty,
        } => {
            out.push_str(&format!(
                "{}{} = tex_{}.read(uint2(r{}, r{}));\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                texture,
                x.0,
                y.0
            ));
        }
        KernelOp::SubgroupSize { dst } => {
            out.push_str(&format!(
                "{}{} = _simd_width;\n",
                pad,
                dst_lv(mutable, "uint", dst.0)
            ));
        }
        KernelOp::SharedDeclDyn { id, ty } => {
            out.push_str(&format!(
                "{}/* dynamic shared_{}: threadgroup {}[] — size set at dispatch */\n",
                pad,
                id,
                ty.msl_name(),
            ));
            out.push_str(&format!(
                "{}threadgroup {}* shared_{} = (threadgroup {}*)_dynamic_shared;\n",
                pad,
                ty.msl_name(),
                id,
                ty.msl_name(),
            ));
        }
        KernelOp::DebugPrint { src, ty } => {
            let val_expr = match ty {
                ScalarType::F32 => format!("as_type<uint>(r{})", src.0),
                ScalarType::U32 => format!("r{}", src.0),
                ScalarType::I32 => format!("as_type<uint>(r{})", src.0),
                _ => format!("uint(r{})", src.0),
            };
            out.push_str(&format!(
                "{}{{ uint _dbg_off = atomic_fetch_add_explicit((device atomic_uint*)&_debug_buf[0], 2u, memory_order_relaxed); ",
                pad,
            ));
            out.push_str(&format!(
                "if (_dbg_off + 2u < 16384u) {{ _debug_buf[_dbg_off + 1u] = _quark_id; _debug_buf[_dbg_off + 2u] = {}; }} }}\n",
                val_expr,
            ));
        }
        KernelOp::Dispatch { .. } => {
            out.push_str(&format!(
                "{}/* error: dynamic parallelism not supported in MSL */\n",
                pad
            ));
        }
        // Metal-on-device atomic CAS is RELAXED-only (same constraint
        // documented on AtomicOp above). Stronger orderings on `order`
        // must be expressed via a surrounding KernelOp::Fence.
        KernelOp::AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ty,
            success_order: _,
            failure_order: _,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            // Brace-scope the `_expected` temp when the destination is a
            // hoisted mutable register so a re-CAS into the same register
            // does not redefine it.
            let (open, close) = if mutable.contains_key(&dst.0) {
                ("{ ", " }")
            } else {
                ("", "")
            };
            out.push_str(&format!(
                "{}{}{} r{}_expected = r{};\n",
                pad,
                open,
                ty.msl_name(),
                dst.0,
                expected.0
            ));
            out.push_str(&format!(
                "{}atomic_compare_exchange_weak_explicit((device atomic_{}*)&{}[r{}], &r{}_expected, r{}, memory_order_relaxed, memory_order_relaxed);\n",
                pad, ty.msl_name(), n, index.0, dst.0, desired.0
            ));
            out.push_str(&format!(
                "{}{} = r{}_expected;{}\n",
                pad,
                dst_lv(mutable, ty.msl_name(), dst.0),
                dst.0,
                close
            ));
        }
        // simdgroup_matrix registers are excluded from the mutable-register
        // hoist in kernel::emit (their MSL type isn't a scalar slot); the
        // dst == c in-place arm below already handles the loop-carried
        // accumulate case.
        KernelOp::CooperativeMMA {
            dst,
            a,
            b,
            c,
            ty,
            m,
            n,
            ..
        } => {
            // D = A·B + C over subgroup-scoped simdgroup_matrix fragments.
            // When dst == c (the loop-carried accumulate case) update the
            // fragment in place — re-declaring would shadow the accumulator
            // declared outside the loop and read it uninitialised.
            let t = simd_matrix_elem(ty);
            if dst.0 == c.0 {
                out.push_str(&format!(
                    "{pad}simdgroup_multiply_accumulate(r{d}, r{a}, r{b}, r{d});\n",
                    pad = pad,
                    d = dst.0,
                    a = a.0,
                    b = b.0
                ));
            } else {
                out.push_str(&format!(
                    "{pad}simdgroup_matrix<{t}, {m}, {n}> r{d}; \
                     simdgroup_multiply_accumulate(r{d}, r{a}, r{b}, r{c});\n",
                    pad = pad,
                    t = t,
                    m = m,
                    n = n,
                    d = dst.0,
                    a = a.0,
                    b = b.0,
                    c = c.0
                ));
            }
        }
        KernelOp::CooperativeMatrixLoad {
            dst,
            field,
            index,
            stride,
            frag,
            from_shared,
            m,
            n,
            k,
            ty,
        } => {
            // Declare a simdgroup_matrix of the fragment's shape and load it
            // from the tile's top-left element (`index`) with row `stride`. A is
            // m×k, B is k×n, the accumulator is m×n. `from_shared` selects the
            // threadgroup-shared array `shared_<field>` instead of a buffer.
            let t = simd_matrix_elem(ty);
            let (rows, cols) = match frag {
                crate::MatrixFrag::A => (m, k),
                crate::MatrixFrag::B => (k, n),
                crate::MatrixFrag::Accumulator => (m, n),
            };
            let base = if *from_shared {
                format!("shared_{}", field)
            } else {
                names
                    .get(field)
                    .map(|s| s.as_str())
                    .unwrap_or("field")
                    .to_string()
            };
            out.push_str(&format!(
                "{pad}simdgroup_matrix<{t}, {rows}, {cols}> r{d}; \
                 simdgroup_load(r{d}, &{base}[r{i}], r{s});\n",
                pad = pad,
                t = t,
                rows = rows,
                cols = cols,
                d = dst.0,
                base = base,
                i = index.0,
                s = stride.0
            ));
        }
        KernelOp::CooperativeMatrixStore {
            field,
            index,
            stride,
            src,
            ..
        } => {
            let buf = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!(
                "{pad}simdgroup_store(r{s}, &{buf}[r{i}], r{st});\n",
                pad = pad,
                s = src.0,
                buf = buf,
                i = index.0,
                st = stride.0
            ));
        }
    }
}

/// MSL `simdgroup_matrix` element type — only `half`/`float` are valid
/// fragment element types; everything else falls back to `float`.
fn simd_matrix_elem(ty: &ScalarType) -> &'static str {
    match ty {
        ScalarType::F16 => "half",
        _ => "float",
    }
}

/// Return the MSL literal for the maximum value of an unsigned
/// integer type, used to saturate `SatAdd` on overflow. Signed
/// types fall back to their unsigned-max bit pattern via the cast
/// in the emitter so the same literal works for both — narrow ints
/// pick the width below.
fn unsigned_max_lit_msl(ty: &ScalarType) -> &'static str {
    match ty {
        ScalarType::U8 | ScalarType::I8 => "0xFFu",
        ScalarType::U16 | ScalarType::I16 => "0xFFFFu",
        ScalarType::U32 | ScalarType::I32 => "0xFFFFFFFFu",
        ScalarType::U64 | ScalarType::I64 => "0xFFFFFFFFFFFFFFFFul",
        // SatAdd is integer-only; floats/bools should never reach here.
        _ => "0u",
    }
}

//! KernelOp → MSL statement emission.
//!
//! Single recursive function that pattern-matches every `KernelOp` variant
//! and appends the corresponding MSL text.

use std::collections::HashMap;

use quanta_ir::*;

use super::helpers::{atomic_fn_str, binop_str, cmpop_str, const_msl, math_fn_str};

pub(crate) fn emit_op(
    out: &mut String,
    op: &KernelOp,
    indent: usize,
    names: &HashMap<u32, String>,
    int_consts: &HashMap<u32, i64>,
) {
    let pad = "    ".repeat(indent);
    match op {
        KernelOp::Const { dst, value } => {
            let (ty, val) = const_msl(value);
            out.push_str(&format!("{}{} r{} = {};\n", pad, ty, dst.0, val));
        }
        KernelOp::QuarkId { dst } => {
            out.push_str(&format!("{}uint r{} = _quark_id;\n", pad, dst.0))
        }
        KernelOp::ProtonId { dst } => {
            out.push_str(&format!("{}uint r{} = _proton_id;\n", pad, dst.0))
        }
        KernelOp::NucleusId { dst } => {
            out.push_str(&format!("{}uint r{} = _nucleus_id;\n", pad, dst.0))
        }
        KernelOp::ProtonSize { dst } => {
            out.push_str(&format!("{}uint r{} = _proton_size;\n", pad, dst.0))
        }
        KernelOp::QuarkCount { dst } => out.push_str(&format!(
            "{}uint r{} = _nucleus_id * _proton_size + _proton_size;\n",
            pad, dst.0
        )),
        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            if index.0 == u32::MAX {
                out.push_str(&format!("{}{} r{} = {};\n", pad, ty.msl_name(), dst.0, n));
            } else {
                out.push_str(&format!(
                    "{}{} r{} = {}[r{}];\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    n,
                    index.0
                ));
            }
        }
        KernelOp::Store {
            field, index, src, ..
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!("{}{}[r{}] = r{};\n", pad, n, index.0, src.0));
        }
        KernelOp::BinOp { dst, a, b, op, ty } => {
            if matches!(op, BinOp::SatAdd | BinOp::SatSub) {
                if matches!(op, BinOp::SatAdd) {
                    // Unsigned sat add: sum < a means overflow
                    out.push_str(&format!(
                        "{}{} _sum = r{} + r{}; {} r{} = (_sum < r{}) ? ({})0xFFFFFFFFu : _sum;\n",
                        pad,
                        ty.msl_name(),
                        a.0,
                        b.0,
                        ty.msl_name(),
                        dst.0,
                        a.0,
                        ty.msl_name()
                    ));
                } else {
                    // Unsigned sat sub: a < b means underflow
                    out.push_str(&format!(
                        "{}{} r{} = (r{} < r{}) ? ({})0 : r{} - r{};\n",
                        pad,
                        ty.msl_name(),
                        dst.0,
                        a.0,
                        b.0,
                        ty.msl_name(),
                        a.0,
                        b.0
                    ));
                }
            } else if matches!(op, BinOp::Rotl) {
                // MSL: `rotate(x, k)` rotates left by k bits.
                out.push_str(&format!(
                    "{}{} r{} = rotate(r{}, r{});\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    b.0,
                ));
            } else if matches!(op, BinOp::Rotr) {
                // No native rotate-right; rewrite as rotate-left by
                // (width - k mod width).
                let width: u32 = match ty {
                    ScalarType::U8 | ScalarType::I8 => 8,
                    ScalarType::U16 | ScalarType::I16 | ScalarType::F16 => 16,
                    ScalarType::U32
                    | ScalarType::I32
                    | ScalarType::F32
                    | ScalarType::BF16
                    | ScalarType::FP8E5M2
                    | ScalarType::FP8E4M3 => 32,
                    ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 64,
                    ScalarType::Bool => 1,
                };
                out.push_str(&format!(
                    "{}{} r{} = rotate(r{}, ({}) - (r{} % {}));\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    width,
                    b.0,
                    width,
                ));
            } else if matches!(op, BinOp::Shr | BinOp::Shl) {
                // Cast operands to the BinOp's `ty` before shifting so the
                // shift's signedness matches the IR's intent.
                //
                // Without this, an unsigned shift fed by a signed register
                // — e.g. `u32 r = r_xor_signed >> 8` — compiles in MSL as
                // an arithmetic (sign-propagating) shift in `int`, then
                // gets assigned to `uint`. For shifted-then-scaled patterns
                // like `((u32) >> 8) * (1.0 / 2^24)` the half of lanes
                // with bit 31 set silently produces values around 255
                // instead of in [0, 1). The WASM-route translator types
                // every i32 XOR/AND/OR result as I32 regardless of the
                // source language's signedness, so this case is common.
                let o = binop_str(op);
                let t = ty.msl_name();
                out.push_str(&format!(
                    "{}{} r{} = ({})r{} {} ({})r{};\n",
                    pad, t, dst.0, t, a.0, o, t, b.0
                ));
            } else {
                let o = binop_str(op);
                out.push_str(&format!(
                    "{}{} r{} = r{} {} r{};\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    o,
                    b.0
                ));
            }
        }
        KernelOp::Cmp { dst, a, b, op, .. } => {
            let o = cmpop_str(op);
            out.push_str(&format!(
                "{}bool r{} = (r{} {} r{});\n",
                pad, dst.0, a.0, o, b.0
            ));
        }
        KernelOp::UnaryOp { dst, a, op, ty } => {
            let o = match op {
                UnaryOp::Neg => "-",
                UnaryOp::BitNot => "~",
                UnaryOp::LogicalNot => "!",
            };
            out.push_str(&format!(
                "{}{} r{} = {}r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                o,
                a.0
            ));
        }
        KernelOp::Cast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}{} r{} = ({})r{};\n",
                pad,
                to.msl_name(),
                dst.0,
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
                "{}{} r{} = {}({});\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                emit_op(out, op, indent + 1, names, int_consts);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_op(out, op, indent + 1, names, int_consts);
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            // T1405 parity with quanta-ir MSL emitter: unroll hint for
            // small known iteration counts.
            let count_val = int_consts.get(&count.0).copied();
            if quanta_ir::const_analysis::should_unroll_loop_count(count_val) {
                out.push_str(&format!("{}#pragma clang loop unroll(full)\n", pad));
            }
            out.push_str(&format!(
                "{}for (uint r{} = 0; r{} < r{}; r{}++) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0
            ));
            for op in body {
                emit_op(out, op, indent + 1, names, int_consts);
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Copy { dst, src, .. } => {
            out.push_str(&format!("{}r{} = r{};\n", pad, dst.0, src.0));
        }
        KernelOp::Break => out.push_str(&format!("{}break;\n", pad)),
        KernelOp::Barrier => out.push_str(&format!(
            "{}threadgroup_barrier(mem_flags::mem_threadgroup);\n",
            pad
        )),
        KernelOp::Fence { order: _ } => out.push_str(&format!(
            // MSL doesn't expose `atomic_thread_fence` at the
            // `kernel` scope under -std=metal3.1 (even with
            // `#include <metal_atomic>`). The correct device-wide
            // ordering primitive is `threadgroup_barrier` with
            // `mem_flags::mem_device`, which both flushes pending
            // writes and acts as a memory-order barrier.
            //
            // Memory orderings other than relaxed are subsumed
            // by the barrier's release+acquire semantics; the
            // IR's `order` field is informational here.
            "{}threadgroup_barrier(mem_flags::mem_device);\n",
            pad,
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
                "{}{} r{} = shared_{}[r{}];\n",
                pad,
                ty.msl_name(),
                dst.0,
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
            // Metal device atomics: `memory_order_relaxed` is the
            // only ordering MSL accepts on `device atomic_*`
            // pointers. Strongerorders are silent-rejected by
            // xcrun metal. Clamp every IR ordering down to relaxed
            // here — see the matching note on Fence above.
            let mo = "memory_order_relaxed";
            out.push_str(&format!(
                "{}{} r{} = {}((device atomic_{}*)&{}[r{}], r{}, {});\n",
                pad,
                ty.msl_name(),
                dst.0,
                f,
                ty.msl_name(),
                n,
                index.0,
                val.0,
                mo
            ));
        }
        // Threadgroup-storage atomic. MSL permits per-op
        // `memory_order` on `threadgroup` atomics (unlike `device`
        // atomics which Metal pins to relaxed), but we emit relaxed
        // here too for consistency — stronger orderings expressed
        // via surrounding `Fence`. Underlying storage is the same
        // `threadgroup T shared_<slot>[N]` declaration emitted by
        // SharedDecl; the cast is what Apple's MSL accepts.
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
                "{}{} r{} = {}((threadgroup atomic_{}*)&shared_{}[r{}], r{}, memory_order_relaxed);\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}{} r{} = simd_shuffle_xor(r{}, r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}uint r{} = (uint)((uint64_t)simd_ballot(r{} != 0));\n",
                pad, dst.0, predicate.0
            ));
        }
        KernelOp::WaveAny { dst, predicate } => {
            out.push_str(&format!(
                "{}uint r{} = uint(simd_any(r{} != 0));\n",
                pad, dst.0, predicate.0
            ));
        }
        KernelOp::WaveAll { dst, predicate } => {
            out.push_str(&format!(
                "{}uint r{} = uint(simd_all(r{} != 0));\n",
                pad, dst.0, predicate.0
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
                "{}{} r{} = {}({});\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}{} r{} = tex_{}.sample(samp_{}, float2(r{}, r{})).x;\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}{} r{} = tex_{}.sample(samp_{}, float3(r{}, r{}, r{})).x;\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}uint r{} = tex_{}.get_width();\n",
                pad, dst_w.0, texture
            ));
            out.push_str(&format!(
                "{}uint r{} = tex_{}.get_height();\n",
                pad, dst_h.0, texture
            ));
        }
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
                "{}{} r{} = r{}.{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                vec.0,
                swizzle
            ));
        }
        KernelOp::MatMul { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} r{} = r{} * r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                a.0,
                b.0
            ));
        }
        KernelOp::Bitcast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}{} r{} = as_type<{}>(r{});\n",
                pad,
                to.msl_name(),
                dst.0,
                to.msl_name(),
                src.0
            ));
        }
        KernelOp::CountTrailingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = ctz(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::CountLeadingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = clz(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::PopCount { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = popcount(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::Dot { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} r{} = dot(r{}, r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                a.0,
                b.0
            ));
        }
        KernelOp::SubgroupReduceAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::SubgroupReduceMin { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_min(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::SubgroupReduceMax { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_max(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_prefix_exclusive_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_prefix_inclusive_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
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
                "{}{} r{} = tex_{}.read(uint2(r{}, r{})).x;\n",
                pad,
                ty.msl_name(),
                dst.0,
                texture,
                x.0,
                y.0
            ));
        }
        KernelOp::SubgroupSize { dst } => {
            out.push_str(&format!("{}uint r{} = _simd_width;\n", pad, dst.0));
        }
        KernelOp::SharedDeclDyn { id, ty } => {
            // Dynamic shared memory: declared as a threadgroup pointer.
            // The actual size is set at dispatch via setThreadgroupMemoryLength.
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
            // Debug buffer approach: atomic append (value, thread_id) to _debug_buf.
            // _debug_buf[0] = current write offset (in uint units).
            // _debug_buf[1..] = pairs of (thread_id, value_as_uint).
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
        // Metal device atomic CAS is RELAXED-only; both orderings ignored.
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
            out.push_str(&format!(
                "{}{} r{}_expected = r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                expected.0
            ));
            out.push_str(&format!(
                "{}atomic_compare_exchange_weak_explicit((device atomic_{}*)&{}[r{}], &r{}_expected, r{}, memory_order_relaxed, memory_order_relaxed);\n",
                pad, ty.msl_name(), n, index.0, dst.0, desired.0
            ));
            out.push_str(&format!(
                "{}{} r{} = r{}_expected;\n",
                pad,
                ty.msl_name(),
                dst.0,
                dst.0
            ));
        }
        KernelOp::CooperativeMMA {
            dst,
            a: _,
            b: _,
            c,
            ty,
            m,
            n: _,
            k,
            ..
        } => {
            let t = match ty {
                ScalarType::F16 => "half",
                _ => "float",
            };
            out.push_str(&format!(
                "{}simdgroup_matrix<{}, {}, {}> _mc = r{}; simdgroup_multiply_accumulate(_mc, _mc, _mc, _mc); {} r{} = _mc;\n",
                pad, t, m, k, c.0, ty.msl_name(), dst.0
            ));
        }
    }
}

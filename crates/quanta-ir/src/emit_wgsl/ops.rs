//! emit_op() — match dispatcher over [`KernelOp`] variants.
//!
//! Mirrors the build-time emitter in `quanta-compiler/src/emit_wgsl/ops.rs`
//! op-for-op so that T1001 (cross-emitter exhaustiveness) holds for the JIT
//! path as well. Differences are documented inline; the JIT version closes
//! gaps where the build-time emitter previously emitted
//! `/* unsupported */`.

use crate::*;
use std::collections::{HashMap, HashSet};

use super::helpers::*;

pub(super) fn emit_op(
    out: &mut String,
    op: &KernelOp,
    indent: usize,
    names: &HashMap<u32, String>,
    atomic_fields: &HashSet<u32>,
    atomic_shared: &HashSet<u32>,
) {
    let pad = "    ".repeat(indent);
    match op {
        KernelOp::Const { dst, value } => {
            out.push_str(&format!("{}let r{} = {};\n", pad, dst.0, const_wgsl(value)));
        }
        KernelOp::QuarkId { dst } => out.push_str(&format!("{}let r{} = _quark_id;\n", pad, dst.0)),
        KernelOp::ProtonId { dst } => {
            out.push_str(&format!("{}let r{} = _proton_id;\n", pad, dst.0))
        }
        KernelOp::NucleusId { dst } => {
            out.push_str(&format!("{}let r{} = _nucleus_id;\n", pad, dst.0))
        }
        KernelOp::ProtonSize { dst } => {
            out.push_str(&format!("{}let r{} = _proton_size;\n", pad, dst.0))
        }
        KernelOp::QuarkCount { dst } => {
            out.push_str(&format!("{}let r{} = _quark_count;\n", pad, dst.0));
        }
        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            if atomic_fields.contains(field) {
                out.push_str(&format!(
                    "{}let r{}: {} = atomicLoad(&{}[r{}]);\n",
                    pad,
                    dst.0,
                    ty.wgsl_name(),
                    n,
                    index.0
                ));
            } else if matches!(ty, ScalarType::BF16) {
                // bf16 storage is u32 (one per word); unpack to f32.
                out.push_str(&format!(
                    "{}let r{}: f32 = bitcast<f32>({}[r{}] << 16u);\n",
                    pad, dst.0, n, index.0
                ));
            } else {
                out.push_str(&format!(
                    "{}let r{}: {} = {}[r{}];\n",
                    pad,
                    dst.0,
                    ty.wgsl_name(),
                    n,
                    index.0
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
            if atomic_fields.contains(field) {
                out.push_str(&format!(
                    "{}atomicStore(&{}[r{}], r{});\n",
                    pad, n, index.0, src.0
                ));
            } else if matches!(ty, ScalarType::BF16) {
                // Pack f32 → bf16 (round-to-nearest-even), store as u32.
                out.push_str(&format!(
                    "{}let r{}_b: u32 = bitcast<u32>(r{}); \
                     {}[r{}] = (r{}_b + 0x7fffu + ((r{}_b >> 16u) & 1u)) >> 16u;\n",
                    pad, src.0, src.0, n, index.0, src.0, src.0
                ));
            } else {
                out.push_str(&format!("{}{}[r{}] = r{};\n", pad, n, index.0, src.0));
            }
        }
        KernelOp::BinOp { dst, a, b, op, ty } => {
            binop_wgsl(out, &pad, dst.0, a.0, b.0, op, ty);
        }
        KernelOp::Cmp { dst, a, b, op, .. } => {
            let o = cmpop_str(op);
            out.push_str(&format!(
                "{}let r{} = (r{} {} r{});\n",
                pad, dst.0, a.0, o, b.0
            ));
        }
        KernelOp::UnaryOp { dst, a, op, ty } => {
            let tyw = ty.wgsl_name();
            // WGSL forbids unary `-` on unsigned integers. Negate as a
            // two's-complement subtraction from zero, which is correct
            // (and wrapping) for both signed and unsigned ints; floats
            // keep the unary minus.
            let is_uint = matches!(
                ty,
                ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64
            );
            let expr = match op {
                UnaryOp::Neg if is_uint => format!("({tyw}(0) - r{})", a.0),
                UnaryOp::Neg => format!("-r{}", a.0),
                UnaryOp::BitNot => format!("~r{}", a.0),
                UnaryOp::LogicalNot => format!("!r{}", a.0),
            };
            out.push_str(&format!("{}let r{}: {} = {};\n", pad, dst.0, tyw, expr));
        }
        KernelOp::Cast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}let r{} = {}(r{});\n",
                pad,
                dst.0,
                to.wgsl_name(),
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
                "{}let r{}: {} = {}({});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                f,
                a.join(", ")
            ));
        }
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            out.push_str(&format!("{}if r{} {{\n", pad, cond.0));
            for op in then_ops {
                emit_op(out, op, indent + 1, names, atomic_fields, atomic_shared);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_op(out, op, indent + 1, names, atomic_fields, atomic_shared);
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            // T1405 parity note: WGSL has no standard unroll annotation
            // (Naga/Tint compile WGSL to SPIR-V/MSL and apply their own
            // unrolling there). The SPIR-V emitter still sets
            // LOOP_CONTROL_UNROLL on the lowered representation when the
            // count is a small const — so kernels that go WGSL → SPIR-V
            // → Vulkan still benefit. See `emit_spirv/ops.rs` and
            // `emit_msl/ops.rs` for the explicit hint sites.
            out.push_str(&format!(
                "{}for (var r{}: u32 = 0u; r{} < r{}; r{} = r{} + 1u) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0, iter_reg.0
            ));
            for op in body {
                emit_op(out, op, indent + 1, names, atomic_fields, atomic_shared);
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Copy { dst, src, .. } => {
            out.push_str(&format!("{}r{} = r{};\n", pad, dst.0, src.0));
        }
        KernelOp::Break => out.push_str(&format!("{}break;\n", pad)),
        KernelOp::Barrier => out.push_str(&format!("{}workgroupBarrier();\n", pad)),
        // WGSL has no per-op memory ordering — atomicAdd / atomicLoad are
        // SeqCst by spec. The closest equivalent to a non-Relaxed fence is
        // `storageBarrier()`, which forces visibility of storage-buffer
        // writes across the workgroup. Relaxed is a no-op.
        KernelOp::Fence { order } => match order {
            crate::MemoryOrder::Relaxed => {}
            _ => out.push_str(&format!("{}storageBarrier();\n", pad)),
        },
        KernelOp::SharedDecl { .. } | KernelOp::SharedDeclDyn { .. } => {
            // Already emitted at module scope by collect_shared_decls.
        }
        KernelOp::SharedLoad { dst, id, index, ty } => {
            // Slots wrapped in atomic<T> (any SharedAtomicOp touched
            // them) forbid bare reads — go through atomicLoad.
            if atomic_shared.contains(id) {
                out.push_str(&format!(
                    "{}let r{}: {} = atomicLoad(&shared_{}[r{}]);\n",
                    pad,
                    dst.0,
                    ty.wgsl_name(),
                    id,
                    index.0
                ));
            } else {
                out.push_str(&format!(
                    "{}let r{}: {} = shared_{}[r{}];\n",
                    pad,
                    dst.0,
                    ty.wgsl_name(),
                    id,
                    index.0
                ));
            }
        }
        KernelOp::SharedStore { id, index, src, .. } => {
            if atomic_shared.contains(id) {
                out.push_str(&format!(
                    "{}atomicStore(&shared_{}[r{}], r{});\n",
                    pad, id, index.0, src.0
                ));
            } else {
                out.push_str(&format!(
                    "{}shared_{}[r{}] = r{};\n",
                    pad, id, index.0, src.0
                ));
            }
        }
        // WGSL atomics are SeqCst by spec — there's no per-op `order`
        // parameter on `atomicAdd` / `atomicLoad` / etc. Stronger orderings
        // than requested are always sound; this lane therefore ignores
        // `order` and emits the standard atomic call.
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
                "{}let r{}: {} = {}(&{}[r{}], r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                f,
                n,
                index.0,
                val.0
            ));
        }
        // Shared-memory atomic. The slot is declared
        // `var<workgroup> shared_N: array<atomic<T>, ...>` by
        // collect_shared_decls (driven by the same atomic_shared
        // set), so the standard atomic builtins apply directly.
        // WGSL atomics are SeqCst by spec — `order` is ignored,
        // stronger-than-requested is always sound (same stance as
        // the buffer AtomicOp arm above).
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
                "{}let r{}: {} = {}(&shared_{}[r{}], r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                f,
                slot,
                index.0,
                val.0
            ));
        }
        // WGSL atomicCompareExchangeWeak is SeqCst by spec; both
        // `success_order` and `failure_order` are ignored here, same
        // as AtomicOp above.
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
                "{}let r{}: {} = atomicCompareExchangeWeak(&{}[r{}], r{}, r{}).old_value;\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                n,
                index.0,
                expected.0,
                desired.0
            ));
        }
        KernelOp::WaveShuffle {
            dst,
            src,
            lane_delta,
            ty,
        } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupShuffleXor(r{}, r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0,
                lane_delta.0
            ));
        }
        KernelOp::WaveBallot { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = subgroupBallot(r{} != 0u).x;\n",
                pad, dst.0, predicate.0
            ));
        }
        KernelOp::WaveAny { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = select(0u, 1u, subgroupAny(r{} != 0u));\n",
                pad, dst.0, predicate.0
            ));
        }
        KernelOp::WaveAll { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = select(0u, 1u, subgroupAll(r{} != 0u));\n",
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
                "{}let r{}: {} = {}({});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                func_name,
                a.join(", ")
            ));
        }
        KernelOp::TextureSample2D {
            dst, texture, x, y, ..
        } => {
            let n = names.get(texture).map(|s| s.as_str()).unwrap_or("tex");
            out.push_str(&format!(
                "{}let r{} = textureSampleLevel({}, samp_{}, vec2<f32>(r{}, r{}), 0.0);\n",
                pad, dst.0, n, texture, x.0, y.0
            ));
        }
        KernelOp::TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ..
        } => {
            let n = names.get(texture).map(|s| s.as_str()).unwrap_or("tex");
            out.push_str(&format!(
                "{}let r{} = textureSampleLevel({}, samp_{}, vec3<f32>(r{}, r{}, r{}), 0.0);\n",
                pad, dst.0, n, texture, x.0, y.0, z.0
            ));
        }
        KernelOp::TextureWrite2D {
            texture,
            x,
            y,
            value,
            ..
        } => {
            let n = names.get(texture).map(|s| s.as_str()).unwrap_or("tex");
            out.push_str(&format!(
                "{}textureStore({}, vec2<i32>(i32(r{}), i32(r{})), r{});\n",
                pad, n, x.0, y.0, value.0
            ));
        }
        KernelOp::TextureSize {
            dst_w,
            dst_h,
            texture,
        } => {
            let n = names.get(texture).map(|s| s.as_str()).unwrap_or("tex");
            out.push_str(&format!(
                "{}let _dim_{} = textureDimensions({});\n",
                pad, texture, n
            ));
            out.push_str(&format!("{}let r{} = _dim_{}.x;\n", pad, dst_w.0, texture));
            out.push_str(&format!("{}let r{} = _dim_{}.y;\n", pad, dst_h.0, texture));
        }
        KernelOp::VecConstruct {
            dst,
            components,
            ty,
        } => {
            let n = components.len();
            let comps: Vec<String> = components.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}let r{} = vec{}<{}>({});\n",
                pad,
                dst.0,
                n,
                ty.wgsl_name(),
                comps.join(", ")
            ));
        }
        KernelOp::VecExtract {
            dst,
            vec,
            component,
            ..
        } => {
            let swizzle = match component {
                0 => "x",
                1 => "y",
                2 => "z",
                _ => "w",
            };
            out.push_str(&format!(
                "{}let r{} = r{}.{};\n",
                pad, dst.0, vec.0, swizzle
            ));
        }
        KernelOp::MatMul { dst, a, b, .. } => {
            out.push_str(&format!("{}let r{} = r{} * r{};\n", pad, dst.0, a.0, b.0));
        }
        KernelOp::Bitcast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}let r{} = bitcast<{}>(r{});\n",
                pad,
                dst.0,
                to.wgsl_name(),
                src.0
            ));
        }
        KernelOp::CountTrailingZeros { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countTrailingZeros(r{});\n",
                pad, dst.0, src.0
            ));
        }
        KernelOp::CountLeadingZeros { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countLeadingZeros(r{});\n",
                pad, dst.0, src.0
            ));
        }
        KernelOp::PopCount { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countOneBits(r{});\n",
                pad, dst.0, src.0
            ));
        }
        KernelOp::Dot { dst, a, b, .. } => {
            out.push_str(&format!(
                "{}let r{} = dot(r{}, r{});\n",
                pad, dst.0, a.0, b.0
            ));
        }
        KernelOp::SubgroupReduceAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupAdd(r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0
            ));
        }
        KernelOp::SubgroupReduceMin { dst, src, ty } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupMin(r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0
            ));
        }
        KernelOp::SubgroupReduceMax { dst, src, ty } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupMax(r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0
            ));
        }
        KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupExclusiveAdd(r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0
            ));
        }
        KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}let r{}: {} = subgroupInclusiveAdd(r{});\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                src.0
            ));
        }
        KernelOp::TextureLoad2D {
            dst, texture, x, y, ..
        } => {
            let n = names.get(texture).map(|s| s.as_str()).unwrap_or("tex");
            out.push_str(&format!(
                "{}let r{} = textureLoad({}, vec2<i32>(i32(r{}), i32(r{})), 0);\n",
                pad, dst.0, n, x.0, y.0
            ));
        }
        KernelOp::SubgroupSize { dst } => {
            out.push_str(&format!("{}let r{} = subgroupSize();\n", pad, dst.0));
        }
        KernelOp::DebugPrint { src, .. } => {
            // WGSL has no print primitive. Emit a comment so the op is
            // visible in generated source for debuggers (DevTools shows the
            // shader); a real implementation can route through a debug
            // storage buffer once one is wired up by the WebGPU driver.
            out.push_str(&format!("{}// gpu_print: r{}\n", pad, src.0));
        }
        KernelOp::Dispatch { .. } => {
            // Dynamic parallelism is not supported by WebGPU. Surface as a
            // comment so the kernel still validates; the host-side dispatch
            // path must enqueue the child wave instead.
            out.push_str(&format!(
                "{}// dynamic-parallelism dispatch lowered host-side\n",
                pad
            ));
        }
        KernelOp::CooperativeMMA {
            dst, a, b, c, ty, ..
        } => {
            // WGSL has no native cooperative matrix; emit an equivalent
            // scalar lowering so the op compiles. Hardware acceleration will
            // be added when WebGPU exposes the matrix extension.
            out.push_str(&format!(
                "{}var r{}: {} = r{} * r{} + r{};\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                a.0,
                b.0,
                c.0
            ));
        }
    }
}

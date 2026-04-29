//! emit_op() — match dispatcher over KernelOp variants.

use crate::*;
use std::collections::HashMap;

use super::helpers::*;

pub(super) fn emit_op(
    out: &mut String,
    op: &KernelOp,
    indent: usize,
    names: &HashMap<u32, String>,
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
                emit_op(out, op, indent + 1, names);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_op(out, op, indent + 1, names);
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            out.push_str(&format!(
                "{}for (uint r{} = 0; r{} < r{}; r{}++) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0
            ));
            for op in body {
                emit_op(out, op, indent + 1, names);
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
                "{}{} r{} = {}((device atomic_{}*)&{}[r{}], r{}, memory_order_relaxed);\n",
                pad,
                ty.msl_name(),
                dst.0,
                f,
                ty.msl_name(),
                n,
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
            out.push_str(&format!(
                "{}uint r{} = simd_ballot(r{} != 0).x;\n",
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
                "{}{} r{} = tex_{}.sample(samp_{}, float2(r{}, r{}));\n",
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
                "{}{} r{} = tex_{}.sample(samp_{}, float3(r{}, r{}, r{}));\n",
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
                "{}{} r{} = tex_{}.read(uint2(r{}, r{}));\n",
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
        KernelOp::AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ty,
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
        KernelOp::CooperativeMMA { dst, ty, .. } => {
            // Cooperative matrix multiply-add not yet supported in MSL; placeholder zero.
            out.push_str(&format!("{}{} r{} = 0;\n", pad, ty.msl_name(), dst.0));
        }
    }
}

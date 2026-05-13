//! Built-in MSL emitter (kept for Phase 4 JIT migration).

#![allow(dead_code)]

use quanta_ir::{KernelDef, KernelOp, KernelParam};
use std::collections::HashMap;

fn emit_msl(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Emit device helper functions
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_msl_fallback(src));
        out.push('\n');
    }

    out.push_str(&format!("kernel void {}(\n", kernel.name));

    let mut param_lines = Vec::new();
    for param in &kernel.params {
        match param {
            KernelParam::FieldRead {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    device const {}* {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
            }
            KernelParam::FieldWrite {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    device {}* {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
            }
            KernelParam::Constant {
                name,
                slot,
                scalar_type,
            } => {
                param_lines.push(format!(
                    "    constant {}& {} [[buffer({})]]",
                    scalar_type.msl_name(),
                    name,
                    slot
                ));
            }
            KernelParam::Texture2DRead { name, slot, .. } => {
                param_lines.push(format!(
                    "    texture2d<float, access::sample> {} [[texture({})]]",
                    name, slot
                ));
            }
            KernelParam::Texture2DWrite { name, slot, .. } => {
                param_lines.push(format!(
                    "    texture2d<float, access::write> {} [[texture({})]]",
                    name, slot
                ));
            }
            KernelParam::Texture3DRead { name, slot, .. } => {
                param_lines.push(format!(
                    "    texture3d<float, access::sample> {} [[texture({})]]",
                    name, slot
                ));
            }
        }
    }
    param_lines.push("    uint _quark_id [[thread_position_in_grid]]".to_string());
    param_lines.push("    uint _proton_id [[thread_position_in_threadgroup]]".to_string());
    param_lines.push("    uint _nucleus_id [[threadgroup_position_in_grid]]".to_string());
    param_lines.push("    uint _proton_size [[threads_per_threadgroup]]".to_string());

    out.push_str(&param_lines.join(",\n"));
    out.push_str("\n) {\n");

    // Build slot → name map for field references in ops
    let mut slot_names: HashMap<u32, String> = HashMap::new();
    for param in &kernel.params {
        match param {
            KernelParam::FieldRead { name, slot, .. }
            | KernelParam::FieldWrite { name, slot, .. }
            | KernelParam::Constant { name, slot, .. } => {
                slot_names.insert(*slot, name.clone());
            }
            _ => {}
        }
    }

    // Emit body from KernelOps, or fall back to raw source translation
    if kernel.body.is_empty() {
        if let Some(ref src) = kernel.body_source {
            out.push_str(&translate_body_to_msl(src));
        }
    } else {
        for op in &kernel.body {
            emit_msl_op(&mut out, op, 1, &slot_names);
        }
    }

    out.push_str("}\n");
    Ok(out)
}

fn emit_msl_op(
    out: &mut String,
    op: &quanta_ir::KernelOp,
    indent: usize,
    names: &HashMap<u32, String>,
) {
    let pad = "    ".repeat(indent);
    use quanta_ir::KernelOp::*;
    match op {
        Const { dst, value } => {
            let (ty, val) = const_to_msl(value);
            out.push_str(&format!("{}{} r{} = {};\n", pad, ty, dst.0, val));
        }
        QuarkId { dst } => {
            out.push_str(&format!("{}uint r{} = _quark_id;\n", pad, dst.0));
        }
        QuarkCount { dst } => {
            out.push_str(&format!(
                "{}uint r{} = _nucleus_id * _proton_size + _proton_size;\n",
                pad, dst.0
            ));
        }
        ProtonId { dst } => {
            out.push_str(&format!("{}uint r{} = _proton_id;\n", pad, dst.0));
        }
        NucleusId { dst } => {
            out.push_str(&format!("{}uint r{} = _nucleus_id;\n", pad, dst.0));
        }
        ProtonSize { dst } => {
            out.push_str(&format!("{}uint r{} = _proton_size;\n", pad, dst.0));
        }
        Load {
            dst,
            field,
            index,
            ty,
        } => {
            let fname = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            if index.0 == u32::MAX {
                // Push constant — direct reference
                out.push_str(&format!(
                    "{}{} r{} = {};\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    fname
                ));
            } else {
                out.push_str(&format!(
                    "{}{} r{} = {}[r{}];\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    fname,
                    index.0
                ));
            }
        }
        Store {
            field, index, src, ..
        } => {
            let fname = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!("{}{}[r{}] = r{};\n", pad, fname, index.0, src.0));
        }
        BinOp { dst, a, b, op, ty } => {
            let op_str = match op {
                quanta_ir::BinOp::Add => "+",
                quanta_ir::BinOp::Sub => "-",
                quanta_ir::BinOp::Mul => "*",
                quanta_ir::BinOp::Div => "/",
                quanta_ir::BinOp::Rem => "%",
                quanta_ir::BinOp::BitAnd => "&",
                quanta_ir::BinOp::BitOr => "|",
                quanta_ir::BinOp::BitXor => "^",
                quanta_ir::BinOp::Shl => "<<",
                quanta_ir::BinOp::Shr => ">>",
                quanta_ir::BinOp::SatAdd => "+", // handled via clamp below
                quanta_ir::BinOp::SatSub => "-",
                // Rotates emitted via the `rotate(...)` MSL builtin
                // in the special-case branch below; this arm is
                // never read for them.
                quanta_ir::BinOp::Rotl | quanta_ir::BinOp::Rotr => "",
            };
            if matches!(op, quanta_ir::BinOp::SatAdd) {
                out.push_str(&format!(
                    "{}{} _s = r{} + r{}; {} r{} = (_s < r{}) ? ({})0xFFFFFFFFu : _s;\n",
                    pad,
                    ty.msl_name(),
                    a.0,
                    b.0,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    ty.msl_name()
                ));
            } else if matches!(op, quanta_ir::BinOp::SatSub) {
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
            } else if matches!(op, quanta_ir::BinOp::Rotl) {
                out.push_str(&format!(
                    "{}{} r{} = rotate(r{}, r{});\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    b.0,
                ));
            } else if matches!(op, quanta_ir::BinOp::Rotr) {
                let width: u32 = match ty {
                    quanta_ir::ScalarType::U8 | quanta_ir::ScalarType::I8 => 8,
                    quanta_ir::ScalarType::U16
                    | quanta_ir::ScalarType::I16
                    | quanta_ir::ScalarType::F16 => 16,
                    quanta_ir::ScalarType::U32
                    | quanta_ir::ScalarType::I32
                    | quanta_ir::ScalarType::F32 => 32,
                    quanta_ir::ScalarType::U64
                    | quanta_ir::ScalarType::I64
                    | quanta_ir::ScalarType::F64 => 64,
                    quanta_ir::ScalarType::Bool => 1,
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
            } else {
                out.push_str(&format!(
                    "{}{} r{} = r{} {} r{};\n",
                    pad,
                    ty.msl_name(),
                    dst.0,
                    a.0,
                    op_str,
                    b.0
                ));
            }
        }
        Cmp { dst, a, b, op, .. } => {
            let op_str = match op {
                quanta_ir::CmpOp::Eq => "==",
                quanta_ir::CmpOp::Ne => "!=",
                quanta_ir::CmpOp::Lt => "<",
                quanta_ir::CmpOp::Le => "<=",
                quanta_ir::CmpOp::Gt => ">",
                quanta_ir::CmpOp::Ge => ">=",
            };
            out.push_str(&format!(
                "{}bool r{} = (r{} {} r{});\n",
                pad, dst.0, a.0, op_str, b.0
            ));
        }
        Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            out.push_str(&format!("{}if (r{}) {{\n", pad, cond.0));
            for op in then_ops {
                emit_msl_op(out, op, indent + 1, names);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_msl_op(out, op, indent + 1, names);
                }
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        Loop {
            count,
            iter_reg,
            body,
        } => {
            out.push_str(&format!(
                "{}for (uint r{} = 0; r{} < r{}; r{}++) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0
            ));
            for op in body {
                emit_msl_op(out, op, indent + 1, names);
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        MathCall {
            dst,
            func,
            args,
            ty,
        } => {
            let fn_name = math_fn_msl(func);
            let arg_strs: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}{} r{} = {}({});\n",
                pad,
                ty.msl_name(),
                dst.0,
                fn_name,
                arg_strs.join(", ")
            ));
        }
        Cast {
            dst,
            src,
            from: _,
            to,
        } => {
            out.push_str(&format!(
                "{}{} r{} = ({})r{};\n",
                pad,
                to.msl_name(),
                dst.0,
                to.msl_name(),
                src.0
            ));
        }
        Copy { dst, src, .. } => {
            out.push_str(&format!("{}r{} = r{};\n", pad, dst.0, src.0));
        }
        Break => {
            out.push_str(&format!("{}break;\n", pad));
        }
        Barrier => {
            out.push_str(&format!(
                "{}threadgroup_barrier(mem_flags::mem_threadgroup);\n",
                pad
            ));
        }
        Fence { order } => out.push_str(&format!(
            "{}atomic_thread_fence(mem_flags::mem_device, {});\n",
            pad,
            match order {
                quanta_ir::MemoryOrder::Relaxed => "memory_order_relaxed",
                quanta_ir::MemoryOrder::Acquire => "memory_order_acquire",
                quanta_ir::MemoryOrder::Release => "memory_order_release",
                quanta_ir::MemoryOrder::AcqRel => "memory_order_acq_rel",
                quanta_ir::MemoryOrder::SeqCst => "memory_order_seq_cst",
            }
        )),
        SharedDecl { id, ty, count } => {
            out.push_str(&format!(
                "{}threadgroup {} shared_{}[{}];\n",
                pad,
                ty.msl_name(),
                id,
                count
            ));
        }
        SharedLoad { dst, id, index, ty } => {
            out.push_str(&format!(
                "{}{} r{} = shared_{}[r{}];\n",
                pad,
                ty.msl_name(),
                dst.0,
                id,
                index.0
            ));
        }
        SharedStore { id, index, src, .. } => {
            out.push_str(&format!(
                "{}shared_{}[r{}] = r{};\n",
                pad, id, index.0, src.0
            ));
        }
        UnaryOp { dst, a, op, ty } => {
            let op_str = match op {
                quanta_ir::UnaryOp::Neg => "-",
                quanta_ir::UnaryOp::BitNot => "~",
                quanta_ir::UnaryOp::LogicalNot => "!",
            };
            out.push_str(&format!(
                "{}{} r{} = {}r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                op_str,
                a.0
            ));
        }
        // Metal restricts device-address-space atomics to
        // memory_order_relaxed (see emit_msl/ops.rs in quanta-ir for
        // the same constraint). Stronger orderings on `order` must be
        // expressed via a surrounding KernelOp::Fence.
        AtomicOp {
            dst,
            field,
            index,
            val,
            op,
            ty,
            order: _,
        } => {
            let fn_name = match op {
                quanta_ir::AtomicOp::Add => "atomic_fetch_add_explicit",
                quanta_ir::AtomicOp::Sub => "atomic_fetch_sub_explicit",
                quanta_ir::AtomicOp::Min => "atomic_fetch_min_explicit",
                quanta_ir::AtomicOp::Max => "atomic_fetch_max_explicit",
                quanta_ir::AtomicOp::And => "atomic_fetch_and_explicit",
                quanta_ir::AtomicOp::Or => "atomic_fetch_or_explicit",
                quanta_ir::AtomicOp::Xor => "atomic_fetch_xor_explicit",
                quanta_ir::AtomicOp::Exchange => "atomic_exchange_explicit",
                quanta_ir::AtomicOp::CompareExchange => "atomic_compare_exchange_weak_explicit",
            };
            let fname = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!(
                "{}{} r{} = {}((device atomic_{}*)&{}[r{}], r{}, memory_order_relaxed);\n",
                pad,
                ty.msl_name(),
                dst.0,
                fn_name,
                ty.msl_name(),
                fname,
                index.0,
                val.0
            ));
        }
        WaveShuffle {
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
        WaveBallot { dst, predicate } => {
            out.push_str(&format!(
                "{}uint r{} = simd_ballot(r{} != 0).x;\n",
                pad, dst.0, predicate.0
            ));
        }
        WaveAny { dst, predicate } => {
            out.push_str(&format!(
                "{}uint r{} = uint(simd_any(r{} != 0));\n",
                pad, dst.0, predicate.0
            ));
        }
        WaveAll { dst, predicate } => {
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
            let arg_strs: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}{} r{} = {}({});\n",
                pad,
                ty.msl_name(),
                dst.0,
                func_name,
                arg_strs.join(", ")
            ));
        }
        TextureSample2D {
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
        TextureSample3D {
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
        TextureWrite2D {
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
        TextureSize {
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
        VecConstruct {
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
        VecExtract {
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
        MatMul { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} r{} = r{} * r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                a.0,
                b.0
            ));
        }
        Bitcast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}{} r{} = as_type<{}>(r{});\n",
                pad,
                to.msl_name(),
                dst.0,
                to.msl_name(),
                src.0
            ));
        }
        CountTrailingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = ctz(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        CountLeadingZeros { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = clz(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        PopCount { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = popcount(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        Dot { dst, a, b, ty, .. } => {
            out.push_str(&format!(
                "{}{} r{} = dot(r{}, r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                a.0,
                b.0
            ));
        }
        SubgroupReduceAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        SubgroupReduceMin { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_min(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        SubgroupReduceMax { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_max(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        SubgroupExclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_prefix_exclusive_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        SubgroupInclusiveAdd { dst, src, ty } => {
            out.push_str(&format!(
                "{}{} r{} = simd_prefix_inclusive_sum(r{});\n",
                pad,
                ty.msl_name(),
                dst.0,
                src.0
            ));
        }
        TextureLoad2D {
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
        Dispatch { .. } => {
            out.push_str(&format!(
                "{}/* error: dynamic parallelism not supported in MSL */\n",
                pad
            ));
        }
        AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ty,
            success_order: _,
            failure_order: _,
        } => {
            let fname = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!(
                "{}{} r{}_expected = r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                expected.0
            ));
            out.push_str(&format!(
                "{}atomic_compare_exchange_weak_explicit((device atomic_{}*)&{}[r{}], &r{}_expected, r{}, memory_order_relaxed, memory_order_relaxed);\n",
                pad, ty.msl_name(), fname, index.0, dst.0, desired.0
            ));
            out.push_str(&format!(
                "{}{} r{} = r{}_expected;\n",
                pad,
                ty.msl_name(),
                dst.0,
                dst.0
            ));
        }
        SubgroupSize { dst } => {
            out.push_str(&format!("{}uint r{} = _simd_width;\n", pad, dst.0));
        }
        SharedDeclDyn { id, ty } => {
            out.push_str(&format!(
                "{}/* dynamic shared_{}: threadgroup {}[] */\n",
                pad,
                id,
                ty.msl_name(),
            ));
        }
        DebugPrint { src, ty } => {
            let val_expr = match ty {
                quanta_ir::ScalarType::F32 => format!("as_type<uint>(r{})", src.0),
                quanta_ir::ScalarType::U32 => format!("r{}", src.0),
                _ => format!("uint(r{})", src.0),
            };
            out.push_str(&format!("{}/* gpu_print: {} */\n", pad, val_expr,));
        }
        CooperativeMMA {
            dst, a, b, c, ty, ..
        } => {
            out.push_str(&format!(
                "{}{} r{} = r{} * r{} + r{};\n",
                pad,
                ty.msl_name(),
                dst.0,
                a.0,
                b.0,
                c.0
            ));
        }
    }
}

fn const_to_msl(value: &quanta_ir::ConstValue) -> (&'static str, String) {
    match value {
        quanta_ir::ConstValue::F32(v) => ("float", format!("{:.6}", v)),
        quanta_ir::ConstValue::F64(v) => ("double", format!("{:.6}", v)),
        quanta_ir::ConstValue::U32(v) => ("uint", format!("{}u", v)),
        quanta_ir::ConstValue::U64(v) => ("ulong", format!("{}ul", v)),
        quanta_ir::ConstValue::I32(v) => ("int", format!("{}", v)),
        quanta_ir::ConstValue::I64(v) => ("long", format!("{}l", v)),
        quanta_ir::ConstValue::Bool(v) => ("bool", format!("{}", v)),
        quanta_ir::ConstValue::F16(v) => (
            "half",
            format!("(half){}", f32::from_bits((*v as u32) << 16)),
        ),
    }
}

fn math_fn_msl(func: &quanta_ir::MathFn) -> &'static str {
    match func {
        quanta_ir::MathFn::Sin => "sin",
        quanta_ir::MathFn::Cos => "cos",
        quanta_ir::MathFn::Tan => "tan",
        quanta_ir::MathFn::Asin => "asin",
        quanta_ir::MathFn::Acos => "acos",
        quanta_ir::MathFn::Atan => "atan",
        quanta_ir::MathFn::Atan2 => "atan2",
        quanta_ir::MathFn::Sqrt => "sqrt",
        quanta_ir::MathFn::Rsqrt => "rsqrt",
        quanta_ir::MathFn::Exp => "exp",
        quanta_ir::MathFn::Exp2 => "exp2",
        quanta_ir::MathFn::Log => "log",
        quanta_ir::MathFn::Log2 => "log2",
        quanta_ir::MathFn::Pow => "pow",
        quanta_ir::MathFn::Abs => "abs",
        quanta_ir::MathFn::Min => "min",
        quanta_ir::MathFn::Max => "max",
        quanta_ir::MathFn::Clamp => "clamp",
        quanta_ir::MathFn::Floor => "floor",
        quanta_ir::MathFn::Ceil => "ceil",
        quanta_ir::MathFn::Round => "round",
        quanta_ir::MathFn::Fma => "fma",
    }
}

fn translate_body_to_msl(rust_source: &str) -> String {
    rust_source
        .replace("quark_id ()", "_quark_id")
        .replace("quark_id()", "_quark_id")
        .replace("proton_id ()", "_proton_id")
        .replace("proton_id()", "_proton_id")
        .replace("nucleus_id ()", "_nucleus_id")
        .replace("nucleus_id()", "_nucleus_id")
        .replace("proton_size ()", "_proton_size")
        .replace("proton_size()", "_proton_size")
        .replace("let mut ", "auto ")
        .replace("let ", "auto ")
        .replace(" as f32", "")
        .replace(" as u32", "")
        .replace(" as i32", "")
        .replace(
            "barrier ()",
            "threadgroup_barrier(mem_flags::mem_threadgroup)",
        )
        .replace(
            "barrier()",
            "threadgroup_barrier(mem_flags::mem_threadgroup)",
        )
}

/// Translate a Rust device function source to MSL (fallback emitter).
fn translate_device_fn_to_msl_fallback(rust_source: &str) -> String {
    let type_map: &[(&str, &str)] = &[
        ("f32", "float"),
        ("f64", "double"),
        ("u32", "uint"),
        ("u64", "ulong"),
        ("i32", "int"),
        ("i64", "long"),
        ("bool", "bool"),
    ];

    let mut s = rust_source.to_string();

    // Extract return type and replace arrow
    let mut ret_msl = "void";
    for &(rust_ty, msl_ty) in type_map {
        let arrow = format!("-> {}", rust_ty);
        if s.contains(&arrow) {
            ret_msl = msl_ty;
            s = s.replace(&arrow, "");
            break;
        }
    }

    // Replace "fn name" with "inline <ret_type> name"
    if let Some(pos) = s.find("fn ") {
        s = format!("{}inline {} {}", &s[..pos], ret_msl, &s[pos + 3..]);
    }

    // Replace parameter types
    for &(rust_ty, msl_ty) in type_map {
        let param_pattern = format!(": {}", rust_ty);
        let param_replacement = format!(": {}", msl_ty);
        s = s.replace(&param_pattern, &param_replacement);
    }

    // Body translations
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace(" as i32", "");

    s
}

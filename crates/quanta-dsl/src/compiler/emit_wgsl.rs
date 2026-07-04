//! Built-in WGSL emitter (kept for Phase 4 JIT migration).

#![allow(dead_code)]

use quanta_ir::{KernelDef, KernelOp, KernelParam};

fn emit_wgsl(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();

    // Emit device helper functions
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_wgsl_fallback(src));
        out.push('\n');
    }

    for param in &kernel.params {
        match param {
            KernelParam::FieldRead {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read> {}: array<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
            }
            KernelParam::FieldWrite {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read_write> {}: array<{}>;\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
            }
            KernelParam::Constant {
                name,
                slot,
                scalar_type,
            } => {
                out.push_str(&format!(
                    "@group(0) @binding({}) var<uniform> {}: {};\n",
                    slot,
                    name,
                    scalar_type.wgsl_name()
                ));
            }
            _ => {}
        }
    }

    // Pre-pass: collect SharedDecl ops and emit them as module-level declarations
    if !kernel.body.is_empty() {
        collect_shared_decls_wgsl(&mut out, &kernel.body);
    }

    out.push_str(&format!(
        "\n@compute @workgroup_size(64)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n",
        kernel.name
    ));
    out.push_str("    let _quark_id = gid.x;\n");

    if kernel.body.is_empty() {
        if let Some(ref src) = kernel.body_source {
            out.push_str(&translate_body_to_wgsl(src));
        }
    } else {
        for op in &kernel.body {
            emit_wgsl_op(&mut out, op, 1);
        }
    }

    out.push_str("}\n");
    Ok(out)
}

/// Recursively collect SharedDecl ops from the kernel body and emit them
/// as WGSL module-level `var<workgroup>` declarations.
fn collect_shared_decls_wgsl(out: &mut String, ops: &[quanta_ir::KernelOp]) {
    use quanta_ir::KernelOp::*;
    for op in ops {
        match op {
            SharedDecl { id, ty, count } => {
                out.push_str(&format!(
                    "var<workgroup> shared_{}: array<{}, {}>;\n",
                    id,
                    ty.wgsl_name(),
                    count
                ));
            }
            Branch {
                then_ops, else_ops, ..
            } => {
                collect_shared_decls_wgsl(out, then_ops);
                collect_shared_decls_wgsl(out, else_ops);
            }
            Loop { body, .. } => {
                collect_shared_decls_wgsl(out, body);
            }
            _ => {}
        }
    }
}

fn emit_wgsl_op(out: &mut String, op: &quanta_ir::KernelOp, indent: usize) {
    let pad = "    ".repeat(indent);
    use quanta_ir::KernelOp::*;
    match op {
        Const { dst, value } => {
            let val = const_to_wgsl(value);
            out.push_str(&format!("{}let r{} = {};\n", pad, dst.0, val));
        }
        QuarkId { dst } => {
            out.push_str(&format!("{}let r{} = _quark_id;\n", pad, dst.0));
        }
        Load {
            dst, field, index, ..
        } => {
            out.push_str(&format!(
                "{}let r{} = field_{}[r{}];\n",
                pad, dst.0, field, index.0
            ));
        }
        Store {
            field, index, src, ..
        } => {
            out.push_str(&format!(
                "{}field_{}[r{}] = r{};\n",
                pad, field, index.0, src.0
            ));
        }
        BinOp { dst, a, b, op, .. } => {
            let op_str = match op {
                quanta_ir::BinOp::Add => "+",
                quanta_ir::BinOp::Sub => "-",
                quanta_ir::BinOp::Mul => "*",
                quanta_ir::BinOp::Div => "/",
                quanta_ir::BinOp::Rem => "%",
                _ => "/* unsupported op */",
            };
            out.push_str(&format!(
                "{}let r{} = r{} {} r{};\n",
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
                emit_wgsl_op(out, op, indent + 1);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_wgsl_op(out, op, indent + 1);
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
                "{}for (var r{}: u32 = 0u; r{} < r{}; r{} = r{} + 1u) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0, iter_reg.0
            ));
            for op in body {
                emit_wgsl_op(out, op, indent + 1);
            }
            out.push_str(&format!("{}}}\n", pad));
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
                "{}let r{} = (r{} {} r{});\n",
                pad, dst.0, a.0, op_str, b.0
            ));
        }
        Cast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}let r{} = {}(r{});\n",
                pad,
                dst.0,
                to.wgsl_name(),
                src.0
            ));
        }
        MathCall {
            dst, func, args, ..
        } => {
            let f = match func {
                quanta_ir::MathFn::Sin => "sin",
                quanta_ir::MathFn::Cos => "cos",
                quanta_ir::MathFn::Sqrt => "sqrt",
                quanta_ir::MathFn::Abs => "abs",
                quanta_ir::MathFn::Min => "min",
                quanta_ir::MathFn::Max => "max",
                quanta_ir::MathFn::Floor => "floor",
                quanta_ir::MathFn::Ceil => "ceil",
                quanta_ir::MathFn::Round => "round",
                quanta_ir::MathFn::Exp => "exp",
                quanta_ir::MathFn::Log => "log",
                quanta_ir::MathFn::Pow => "pow",
                quanta_ir::MathFn::Clamp => "clamp",
                quanta_ir::MathFn::Fma => "fma",
                quanta_ir::MathFn::Rsqrt => "inverseSqrt",
                _ => "/* unsupported */",
            };
            let a: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}let r{} = {}({});\n",
                pad,
                dst.0,
                f,
                a.join(", ")
            ));
        }
        UnaryOp { dst, a, op, .. } => {
            let op_str = match op {
                quanta_ir::UnaryOp::Neg => "-",
                quanta_ir::UnaryOp::BitNot => "~",
                quanta_ir::UnaryOp::LogicalNot => "!",
            };
            out.push_str(&format!("{}let r{} = {}r{};\n", pad, dst.0, op_str, a.0));
        }
        Copy { dst, src, .. } => {
            out.push_str(&format!("{}r{} = r{};\n", pad, dst.0, src.0));
        }
        Quantize { .. } | Dequantize { .. } => {
            out.push_str(&format!("{}/* quantize: lowering pending */\n", pad));
        }
        Break => {
            out.push_str(&format!("{}break;\n", pad));
        }
        Barrier => {
            out.push_str(&format!("{}workgroupBarrier();\n", pad));
        }
        Fence { order } => match order {
            quanta_ir::MemoryOrder::Relaxed => {}
            _ => out.push_str(&format!("{}storageBarrier();\n", pad)),
        },
        SharedDecl { .. } => {
            // Already emitted at module level by collect_shared_decls_wgsl
        }
        SharedLoad { dst, id, index, .. } => {
            out.push_str(&format!(
                "{}let r{} = shared_{}[r{}];\n",
                pad, dst.0, id, index.0
            ));
        }
        SharedStore { id, index, src, .. } => {
            out.push_str(&format!(
                "{}shared_{}[r{}] = r{};\n",
                pad, id, index.0, src.0
            ));
        }
        WaveShuffle {
            dst,
            src,
            lane_delta,
            ..
        } => {
            out.push_str(&format!(
                "{}let r{} = subgroupShuffleXor(r{}, r{});\n",
                pad, dst.0, src.0, lane_delta.0
            ));
        }
        WaveBallot { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = subgroupBallot(r{} != 0u);\n",
                pad, dst.0, predicate.0
            ));
        }
        WaveAny { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = select(0u, 1u, subgroupAny(r{} != 0u));\n",
                pad, dst.0, predicate.0
            ));
        }
        WaveAll { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = select(0u, 1u, subgroupAll(r{} != 0u));\n",
                pad, dst.0, predicate.0
            ));
        }
        KernelOp::DeviceCall {
            dst,
            func_name,
            args,
            ..
        } => {
            let arg_strs: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}let r{} = {}({});\n",
                pad,
                dst.0,
                func_name,
                arg_strs.join(", ")
            ));
        }
        ProtonId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // proton\n", pad, dst.0));
        }
        NucleusId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // group\n", pad, dst.0));
        }
        QuarkCount { dst } => {
            out.push_str(&format!(
                "{}let r{} = gid.x; // total quark count unavailable in WGSL\n",
                pad, dst.0
            ));
        }
        ProtonSize { dst } => {
            out.push_str(&format!("{}let r{} = 64u; // workgroup_size\n", pad, dst.0));
        }
        AtomicOp {
            dst,
            field,
            index,
            val,
            op,
            ..
        } => {
            let f = match op {
                quanta_ir::AtomicOp::Add => "atomicAdd",
                quanta_ir::AtomicOp::Sub => "atomicSub",
                quanta_ir::AtomicOp::Min => "atomicMin",
                quanta_ir::AtomicOp::Max => "atomicMax",
                quanta_ir::AtomicOp::And => "atomicAnd",
                quanta_ir::AtomicOp::Or => "atomicOr",
                quanta_ir::AtomicOp::Xor => "atomicXor",
                quanta_ir::AtomicOp::Exchange => "atomicExchange",
                quanta_ir::AtomicOp::CompareExchange => "atomicCompareExchangeWeak",
            };
            out.push_str(&format!(
                "{}let r{} = {}(&field_{}[r{}], r{});\n",
                pad, dst.0, f, field, index.0, val.0
            ));
        }
        // Shared-memory atomic on WebGPU: blocked on shared-decl
        // `atomic<T>` decoration. See emit_wgsl/ops.rs in quanta-ir.
        SharedAtomicOp { .. } => {
            out.push_str(&format!(
                "{}/* unsupported: SharedAtomicOp — WGSL requires atomic<T> decoration */\n",
                pad
            ));
        }
        AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ..
        } => {
            out.push_str(&format!(
                "{}let r{} = atomicCompareExchangeWeak(&field_{}[r{}], r{}, r{}).old_value;\n",
                pad, dst.0, field, index.0, expected.0, desired.0
            ));
        }
        TextureSample2D {
            dst, texture, x, y, ..
        } => {
            out.push_str(&format!(
                "{}let r{} = textureSample(tex_{}, samp_{}, vec2<f32>(r{}, r{}));\n",
                pad, dst.0, texture, texture, x.0, y.0
            ));
        }
        TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ..
        } => {
            out.push_str(&format!(
                "{}let r{} = textureSample(tex_{}, samp_{}, vec3<f32>(r{}, r{}, r{}));\n",
                pad, dst.0, texture, texture, x.0, y.0, z.0
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
                "{}textureStore(tex_{}, vec2<i32>(r{}, r{}), r{});\n",
                pad, texture, x.0, y.0, value.0
            ));
        }
        TextureSize {
            dst_w,
            dst_h,
            texture,
        } => {
            out.push_str(&format!(
                "{}let _dim_{} = textureDimensions(tex_{});\n",
                pad, texture, texture
            ));
            out.push_str(&format!("{}let r{} = _dim_{}.x;\n", pad, dst_w.0, texture));
            out.push_str(&format!("{}let r{} = _dim_{}.y;\n", pad, dst_h.0, texture));
        }
        VecConstruct {
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
        VecExtract {
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
        MatMul { dst, a, b, .. } => {
            out.push_str(&format!("{}let r{} = r{} * r{};\n", pad, dst.0, a.0, b.0));
        }
        Bitcast { dst, src, to, .. } => {
            out.push_str(&format!(
                "{}let r{} = bitcast<{}>(r{});\n",
                pad,
                dst.0,
                to.wgsl_name(),
                src.0
            ));
        }
        CountTrailingZeros { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countTrailingZeros(r{});\n",
                pad, dst.0, src.0
            ));
        }
        CountLeadingZeros { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countLeadingZeros(r{});\n",
                pad, dst.0, src.0
            ));
        }
        PopCount { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = countOneBits(r{});\n",
                pad, dst.0, src.0
            ));
        }
        Dot { dst, a, b, .. } => {
            out.push_str(&format!(
                "{}let r{} = dot(r{}, r{});\n",
                pad, dst.0, a.0, b.0
            ));
        }
        SubgroupReduceAdd { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = subgroupAdd(r{});\n",
                pad, dst.0, src.0
            ));
        }
        SubgroupReduceMin { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = subgroupMin(r{});\n",
                pad, dst.0, src.0
            ));
        }
        SubgroupReduceMax { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = subgroupMax(r{});\n",
                pad, dst.0, src.0
            ));
        }
        SubgroupExclusiveAdd { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = subgroupExclusiveAdd(r{});\n",
                pad, dst.0, src.0
            ));
        }
        SubgroupInclusiveAdd { dst, src, .. } => {
            out.push_str(&format!(
                "{}let r{} = subgroupInclusiveAdd(r{});\n",
                pad, dst.0, src.0
            ));
        }
        TextureLoad2D {
            dst, texture, x, y, ..
        } => {
            out.push_str(&format!(
                "{}let r{} = textureLoad(tex_{}, vec2<i32>(r{}, r{}), 0);\n",
                pad, dst.0, texture, x.0, y.0
            ));
        }
        Dispatch { .. } => {
            out.push_str(&format!(
                "{}// error: dynamic parallelism not supported in WGSL\n",
                pad
            ));
        }
        SubgroupSize { dst } => {
            out.push_str(&format!("{}let r{} = subgroup_size;\n", pad, dst.0));
        }
        SharedDeclDyn { id, ty } => {
            out.push_str(&format!(
                "{}// dynamic shared_{}: {} — size set at dispatch\n",
                pad,
                id,
                ty.wgsl_name(),
            ));
        }
        DebugPrint { src, .. } => {
            out.push_str(&format!("{}// gpu_print: r{}\n", pad, src.0,));
        }
        CooperativeMMA {
            dst, a, b, c, ty, ..
        } => {
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
        CooperativeMatrixLoad { dst, ty, .. } => {
            out.push_str(&format!(
                "{}var r{}: {} = {}(0);\n",
                pad,
                dst.0,
                ty.wgsl_name(),
                ty.wgsl_name()
            ));
        }
        CooperativeMatrixStore { .. } => {}
    }
}

fn translate_body_to_wgsl(rust_source: &str) -> String {
    rust_source
        .replace("quark_id ()", "_quark_id")
        .replace("quark_id()", "_quark_id")
        .replace("proton_id ()", "gid.x")
        .replace("proton_id()", "gid.x")
        .replace("let mut ", "var ")
        .replace(" as f32", "")
        .replace(" as u32", "")
}

fn const_to_wgsl(value: &quanta_ir::ConstValue) -> String {
    match value {
        quanta_ir::ConstValue::F32(v) => format!("{}f", v),
        quanta_ir::ConstValue::U32(v) => format!("{}u", v),
        quanta_ir::ConstValue::I32(v) => format!("{}i", v),
        quanta_ir::ConstValue::Bool(v) => format!("{}", v),
        _ => "/* unsupported const */".to_string(),
    }
}

/// Translate a Rust device function source to WGSL (fallback emitter).
fn translate_device_fn_to_wgsl_fallback(rust_source: &str) -> String {
    // WGSL uses `fn name(...) -> type` — similar to Rust syntax
    let mut s = rust_source.to_string();
    s = s.replace("let mut ", "var ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s
}

//! KernelDef → WebGPU Shading Language.

use quanta_ir::*;
use std::collections::HashMap;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();

    // Emit device helper functions (from inner fn definitions)
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_wgsl(src));
        out.push('\n');
    }

    let mut slot_names: HashMap<u32, String> = HashMap::new();

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
                slot_names.insert(*slot, name.clone());
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
                slot_names.insert(*slot, name.clone());
            }
            KernelParam::Constant {
                name,
                slot,
                scalar_type,
            } => {
                slot_names.insert(*slot, name.clone());
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

    out.push_str(&format!(
        "\n@compute @workgroup_size(64)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let _quark_id = gid.x;\n",
        kernel.name));

    for op in &kernel.body {
        emit_op(&mut out, op, 1, &slot_names);
    }

    out.push_str("}\n");
    Ok(out)
}

fn emit_op(out: &mut String, op: &KernelOp, indent: usize, names: &HashMap<u32, String>) {
    let pad = "    ".repeat(indent);
    match op {
        KernelOp::Const { dst, value } => {
            out.push_str(&format!("{}let r{} = {};\n", pad, dst.0, const_wgsl(value)));
        }
        KernelOp::QuarkId { dst } => out.push_str(&format!("{}let r{} = _quark_id;\n", pad, dst.0)),
        KernelOp::LocalId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // local\n", pad, dst.0))
        }
        KernelOp::GroupId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // group\n", pad, dst.0))
        }
        KernelOp::Load {
            dst, field, index, ..
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!("{}let r{} = {}[r{}];\n", pad, dst.0, n, index.0));
        }
        KernelOp::Store {
            field, index, src, ..
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!("{}{}[r{}] = r{};\n", pad, n, index.0, src.0));
        }
        KernelOp::BinOp { dst, a, b, op, .. } => {
            let o = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Rem => "%",
                _ => "/* unsupported */",
            };
            out.push_str(&format!(
                "{}let r{} = r{} {} r{};\n",
                pad, dst.0, a.0, o, b.0
            ));
        }
        KernelOp::Cmp { dst, a, b, op, .. } => {
            let o = match op {
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
            };
            out.push_str(&format!(
                "{}let r{} = (r{} {} r{});\n",
                pad, dst.0, a.0, o, b.0
            ));
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
            dst, func, args, ..
        } => {
            let f = match func {
                MathFn::Sin => "sin",
                MathFn::Cos => "cos",
                MathFn::Sqrt => "sqrt",
                MathFn::Abs => "abs",
                MathFn::Min => "min",
                MathFn::Max => "max",
                MathFn::Floor => "floor",
                MathFn::Ceil => "ceil",
                MathFn::Round => "round",
                MathFn::Exp => "exp",
                MathFn::Log => "log",
                MathFn::Pow => "pow",
                MathFn::Clamp => "clamp",
                MathFn::Fma => "fma",
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
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            out.push_str(&format!("{}if r{} {{\n", pad, cond.0));
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
                "{}for (var r{}: u32 = 0u; r{} < r{}; r{} = r{} + 1u) {{\n",
                pad, iter_reg.0, iter_reg.0, count.0, iter_reg.0, iter_reg.0
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
        KernelOp::Barrier => out.push_str(&format!("{}workgroupBarrier();\n", pad)),
        KernelOp::DeviceCall {
            dst,
            func_name,
            args,
            ..
        } => {
            let a: Vec<String> = args.iter().map(|r| format!("r{}", r.0)).collect();
            out.push_str(&format!(
                "{}let r{} = {}({});\n",
                pad,
                dst.0,
                func_name,
                a.join(", ")
            ));
        }
        KernelOp::QuarkCount { dst } => {
            out.push_str(&format!(
                "{}let r{} = gid.x; // total quark count unavailable in WGSL\n",
                pad, dst.0
            ));
        }
        KernelOp::GroupSize { dst } => {
            out.push_str(&format!("{}let r{} = 64u; // workgroup_size\n", pad, dst.0));
        }
        KernelOp::UnaryOp { dst, a, op, .. } => {
            let o = match op {
                UnaryOp::Neg => "-",
                UnaryOp::BitNot => "~",
                UnaryOp::LogicalNot => "!",
            };
            out.push_str(&format!("{}let r{} = {}r{};\n", pad, dst.0, o, a.0));
        }
        KernelOp::SharedDecl { .. } => {
            // WGSL shared memory must be at module scope -- emit separately.
        }
        KernelOp::SharedLoad { dst, id, index, .. } => {
            out.push_str(&format!(
                "{}let r{} = shared_{}[r{}];\n",
                pad, dst.0, id, index.0
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
            ..
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            let f = match op {
                AtomicOp::Add => "atomicAdd",
                AtomicOp::Sub => "atomicSub",
                AtomicOp::Min => "atomicMin",
                AtomicOp::Max => "atomicMax",
                AtomicOp::And => "atomicAnd",
                AtomicOp::Or => "atomicOr",
                AtomicOp::Xor => "atomicXor",
                AtomicOp::Exchange => "atomicExchange",
                AtomicOp::CompareExchange => "atomicCompareExchangeWeak",
            };
            out.push_str(&format!(
                "{}let r{} = {}(&{}[r{}], r{});\n",
                pad, dst.0, f, n, index.0, val.0
            ));
        }
        KernelOp::AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ..
        } => {
            let n = names.get(field).map(|s| s.as_str()).unwrap_or("field");
            out.push_str(&format!(
                "{}let r{} = atomicCompareExchangeWeak(&{}[r{}], r{}, r{}).old_value;\n",
                pad, dst.0, n, index.0, expected.0, desired.0
            ));
        }
        KernelOp::WaveShuffle {
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
        KernelOp::WaveBallot { dst, predicate } => {
            out.push_str(&format!(
                "{}let r{} = subgroupBallot(r{} != 0u);\n",
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
        KernelOp::TextureSample2D {
            dst, texture, x, y, ..
        } => {
            out.push_str(&format!(
                "{}let r{} = textureSample(tex_{}, samp_{}, vec2<f32>(r{}, r{}));\n",
                pad, dst.0, texture, texture, x.0, y.0
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
            out.push_str(&format!(
                "{}let r{} = textureSample(tex_{}, samp_{}, vec3<f32>(r{}, r{}, r{}));\n",
                pad, dst.0, texture, texture, x.0, y.0, z.0
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
                "{}textureStore(tex_{}, vec2<i32>(r{}, r{}), r{});\n",
                pad, texture, x.0, y.0, value.0
            ));
        }
        KernelOp::TextureSize {
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
        KernelOp::Dispatch { .. } => {
            out.push_str(&format!(
                "{}// error: dynamic parallelism not supported in WGSL\n",
                pad
            ));
        }
    }
}

fn const_wgsl(v: &ConstValue) -> String {
    match v {
        ConstValue::F32(x) => format!("{}f", x),
        ConstValue::U32(x) => format!("{}u", x),
        ConstValue::I32(x) => format!("{}i", x),
        ConstValue::Bool(x) => format!("{}", x),
        _ => "/* unsupported const */".to_string(),
    }
}

/// Translate a Rust device function source to WGSL.
/// WGSL uses `fn name(...) -> type` — same syntax as Rust for function
/// signatures, so only body-level translations are needed.
fn translate_device_fn_to_wgsl(rust_source: &str) -> String {
    let mut s = rust_source.to_string();
    s = s.replace("let mut ", "var ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s
}

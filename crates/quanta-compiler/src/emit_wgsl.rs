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
        _ => out.push_str(&format!("{}// TODO: {:?}\n", pad, op)),
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

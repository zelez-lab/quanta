//! KernelDef → WebGPU Shading Language.

use quanta_ir::*;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();

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

    out.push_str(&format!(
        "\n@compute @workgroup_size(64)\nfn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{\n    let _quark_id = gid.x;\n",
        kernel.name));

    for op in &kernel.body {
        emit_op(&mut out, op, 1);
    }

    out.push_str("}\n");
    Ok(out)
}

fn emit_op(out: &mut String, op: &KernelOp, indent: usize) {
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
            // TODO: use param names
            out.push_str(&format!(
                "{}let r{} = field_{}[r{}];\n",
                pad, dst.0, field, index.0
            ));
        }
        KernelOp::Store {
            field, index, src, ..
        } => {
            out.push_str(&format!(
                "{}field_{}[r{}] = r{};\n",
                pad, field, index.0, src.0
            ));
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
                emit_op(out, op, indent + 1);
            }
            if !else_ops.is_empty() {
                out.push_str(&format!("{}}} else {{\n", pad));
                for op in else_ops {
                    emit_op(out, op, indent + 1);
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
                emit_op(out, op, indent + 1);
            }
            out.push_str(&format!("{}}}\n", pad));
        }
        KernelOp::Barrier => out.push_str(&format!("{}workgroupBarrier();\n", pad)),
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

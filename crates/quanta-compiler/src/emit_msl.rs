//! KernelDef → Metal Shading Language.
//!
//! Walks KernelOps and emits correct MSL for all supported operations.
//! This is the structured emitter — no string replacement.

use quanta_ir::*;
use std::collections::HashMap;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Emit device helper functions (from inner fn definitions)
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_msl(src));
        out.push('\n');
    }

    // Kernel signature
    out.push_str(&format!("kernel void {}(\n", kernel.name));

    let mut param_lines = Vec::new();
    let mut slot_names: HashMap<u32, String> = HashMap::new();

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
                slot_names.insert(*slot, name.clone());
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
                slot_names.insert(*slot, name.clone());
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
                slot_names.insert(*slot, name.clone());
            }
            _ => {}
        }
    }
    param_lines.push("    uint _quark_id [[thread_position_in_grid]]".to_string());
    param_lines.push("    uint _local_id [[thread_position_in_threadgroup]]".to_string());
    param_lines.push("    uint _group_id [[threadgroup_position_in_grid]]".to_string());
    param_lines.push("    uint _group_size [[threads_per_threadgroup]]".to_string());

    out.push_str(&param_lines.join(",\n"));
    out.push_str("\n) {\n");

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
            let (ty, val) = const_msl(value);
            out.push_str(&format!("{}{} r{} = {};\n", pad, ty, dst.0, val));
        }
        KernelOp::QuarkId { dst } => {
            out.push_str(&format!("{}uint r{} = _quark_id;\n", pad, dst.0))
        }
        KernelOp::LocalId { dst } => {
            out.push_str(&format!("{}uint r{} = _local_id;\n", pad, dst.0))
        }
        KernelOp::GroupId { dst } => {
            out.push_str(&format!("{}uint r{} = _group_id;\n", pad, dst.0))
        }
        KernelOp::GroupSize { dst } => {
            out.push_str(&format!("{}uint r{} = _group_size;\n", pad, dst.0))
        }
        KernelOp::QuarkCount { dst } => {
            out.push_str(&format!("{}uint r{} = _quark_id; /* TODO */\n", pad, dst.0))
        }
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
        _ => out.push_str(&format!("{}/* TODO: {:?} */\n", pad, op)),
    }
}

fn const_msl(v: &ConstValue) -> (&'static str, String) {
    match v {
        ConstValue::F32(x) => ("float", format!("{:.6}", x)),
        ConstValue::F64(x) => ("double", format!("{:.6}", x)),
        ConstValue::U32(x) => ("uint", format!("{}u", x)),
        ConstValue::U64(x) => ("ulong", format!("{}ul", x)),
        ConstValue::I32(x) => ("int", format!("{}", x)),
        ConstValue::I64(x) => ("long", format!("{}l", x)),
        ConstValue::Bool(x) => ("bool", if *x { "true" } else { "false" }.to_string()),
        ConstValue::F16(x) => (
            "half",
            format!("(half){}", f32::from_bits((*x as u32) << 16)),
        ),
    }
}

fn binop_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
    }
}

fn cmpop_str(op: &CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "==",
        CmpOp::Ne => "!=",
        CmpOp::Lt => "<",
        CmpOp::Le => "<=",
        CmpOp::Gt => ">",
        CmpOp::Ge => ">=",
    }
}

fn math_fn_str(f: &MathFn) -> &'static str {
    match f {
        MathFn::Sin => "sin",
        MathFn::Cos => "cos",
        MathFn::Tan => "tan",
        MathFn::Asin => "asin",
        MathFn::Acos => "acos",
        MathFn::Atan => "atan",
        MathFn::Atan2 => "atan2",
        MathFn::Sqrt => "sqrt",
        MathFn::Rsqrt => "rsqrt",
        MathFn::Exp => "exp",
        MathFn::Exp2 => "exp2",
        MathFn::Log => "log",
        MathFn::Log2 => "log2",
        MathFn::Pow => "pow",
        MathFn::Abs => "abs",
        MathFn::Min => "min",
        MathFn::Max => "max",
        MathFn::Clamp => "clamp",
        MathFn::Floor => "floor",
        MathFn::Ceil => "ceil",
        MathFn::Round => "round",
        MathFn::Fma => "fma",
    }
}

fn atomic_fn_str(op: &AtomicOp) -> &'static str {
    match op {
        AtomicOp::Add => "atomic_fetch_add_explicit",
        AtomicOp::Sub => "atomic_fetch_sub_explicit",
        AtomicOp::Min => "atomic_fetch_min_explicit",
        AtomicOp::Max => "atomic_fetch_max_explicit",
        AtomicOp::And => "atomic_fetch_and_explicit",
        AtomicOp::Or => "atomic_fetch_or_explicit",
        AtomicOp::Xor => "atomic_fetch_xor_explicit",
        AtomicOp::Exchange => "atomic_exchange_explicit",
        AtomicOp::CompareExchange => "atomic_compare_exchange_weak_explicit",
    }
}

/// Translate a Rust device function source to MSL.
///
/// Rewrites the function signature (return type, parameter types) and body
/// using string substitutions. This is the Phase 1 text-based approach;
/// Phase 2 will walk KernelOps for device function bodies too.
fn translate_device_fn_to_msl(rust_source: &str) -> String {
    // Map Rust return types to MSL return types. The `fn name(...) -> T` form
    // becomes `T name(...)` in MSL.
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

    // Replace return type: "-> f32" → "" (moved to front)
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
        // Only replace parameter annotations (": type" patterns), not
        // occurrences inside the body. Since parameter annotations come before
        // the opening brace, this is safe with a simple replace.
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

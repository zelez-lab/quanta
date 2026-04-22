//! KernelDef → Metal Shading Language.
//!
//! Walks KernelOps and emits correct MSL for all supported operations.
//! This is the structured emitter — no string replacement.

use crate::*;
use std::collections::HashMap;

pub fn emit(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Emit device helper functions (from inner fn definitions)
    for src in &kernel.device_sources {
        out.push_str(&translate_device_fn_to_msl(src));
        out.push('\n');
    }

    // Kernel signature with max_total_threads_per_threadgroup attribute
    let max_threads =
        kernel.workgroup_size[0] * kernel.workgroup_size[1] * kernel.workgroup_size[2];
    out.push_str(&format!(
        "[[max_total_threads_per_threadgroup({})]]\nkernel void {}(\n",
        max_threads, kernel.name
    ));

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
    // Check if kernel uses debug print — if so, add a debug buffer parameter
    let uses_debug_print = kernel
        .body
        .iter()
        .any(|op| matches!(op, KernelOp::DebugPrint { .. }));
    if uses_debug_print {
        param_lines.push("    device uint* _debug_buf [[buffer(30)]]".to_string());
    }

    param_lines.push("    uint _quark_id [[thread_position_in_grid]]".to_string());
    param_lines.push("    uint _local_id [[thread_position_in_threadgroup]]".to_string());
    param_lines.push("    uint _group_id [[threadgroup_position_in_grid]]".to_string());
    param_lines.push("    uint _group_size [[threads_per_threadgroup]]".to_string());
    param_lines.push("    uint _simd_width [[threads_per_simdgroup]]".to_string());

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
        KernelOp::QuarkCount { dst } => out.push_str(&format!(
            "{}uint r{} = _group_id * _group_size + _group_size;\n",
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

// ── Vertex/Fragment shader MSL emitters ────────────────────────────────────

fn shader_type_msl(ty: crate::ShaderType) -> &'static str {
    match ty {
        crate::ShaderType::F32 => "float",
        crate::ShaderType::Vec2 => "float2",
        crate::ShaderType::Vec3 => "float3",
        crate::ShaderType::Vec4 => "float4",
        crate::ShaderType::Mat4 => "float4x4",
        crate::ShaderType::Mat3 => "float3x3",
    }
}

/// Translate a Rust-like shader body to MSL (basic string substitutions).
///
/// Handles both hand-written source and tokenized source (proc_macro2
/// tokenizes `Vec4::new` as `Vec4 :: new` with spaces around `::`) .
fn translate_shader_body(src: &str) -> String {
    let mut s = src.to_string();
    // Strip outer braces from block expression
    let trimmed = s.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        s = trimmed[1..trimmed.len() - 1].to_string();
    }
    // Handle tokenized form (spaces around ::)
    s = s.replace("Vec4 :: new(", "float4(");
    s = s.replace("Vec3 :: new(", "float3(");
    s = s.replace("Vec2 :: new(", "float2(");
    // Handle direct source form
    s = s.replace("Vec4::new(", "float4(");
    s = s.replace("Vec3::new(", "float3(");
    s = s.replace("Vec2::new(", "float2(");
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");
    s
}

/// Wrap the last expression of a body as a return statement.
fn indent_and_return(body: &str) -> String {
    let lines: Vec<&str> = body.trim().lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if i == lines.len() - 1 && !trimmed.is_empty() {
            if !trimmed.ends_with(';') && !trimmed.starts_with("return") {
                out.push_str(&format!("    return {};\n", trimmed));
            } else if !trimmed.contains("return") {
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=') {
                    out.push_str(&format!("    return {};\n", without_semi));
                } else {
                    out.push_str(&format!("    {}\n", trimmed));
                }
            } else {
                out.push_str(&format!("    {}\n", trimmed));
            }
        } else {
            out.push_str(&format!("    {}\n", trimmed));
        }
    }
    out
}

/// Emit MSL for a vertex shader.
///
/// Metal requires vertex attributes to be passed via a struct with `[[stage_in]]`.
/// Uniform parameters use `[[buffer(N)]]` bindings.
pub fn emit_vertex_shader(shader: &crate::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let attr_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Emit vertex input struct with [[attribute(N)]] decorations
    if !attr_params.is_empty() {
        out.push_str(&format!("struct {}_VertexIn {{\n", shader.name));
        for (i, p) in attr_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[attribute({})]];\n",
                shader_type_msl(p.ty),
                p.name,
                i,
            ));
        }
        out.push_str("};\n\n");
    }

    // Build parameter list
    let mut param_lines = Vec::new();
    if !attr_params.is_empty() {
        param_lines.push(format!("    {}_VertexIn in [[stage_in]]", shader.name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}& {} [[buffer({})]]",
            shader_type_msl(p.ty),
            p.name,
            i,
        ));
    }

    out.push_str(&format!(
        "vertex {} {}(\n{}\n) {{\n",
        shader_type_msl(shader.return_type),
        shader.name,
        param_lines.join(",\n"),
    ));

    // Unpack struct members into local variables
    for p in &attr_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            shader_type_msl(p.ty),
            p.name,
            p.name,
        ));
    }

    let body = translate_shader_body(&shader.body_source);
    out.push_str(&indent_and_return(&body));
    out.push_str("}\n");
    Ok(out)
}

/// Emit MSL for a fragment shader.
pub fn emit_fragment_shader(shader: &crate::ShaderDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    let stage_in_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&crate::ShaderParam> =
        shader.params.iter().filter(|p| p.is_uniform).collect();

    // Stage-in struct for interpolated inputs
    if !stage_in_params.is_empty() {
        out.push_str(&format!("struct {}_Input {{\n", shader.name));
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[user(loc{})]];\n",
                shader_type_msl(p.ty),
                p.name,
                i,
            ));
        }
        out.push_str("};\n\n");
    }

    let mut param_lines = Vec::new();
    if !stage_in_params.is_empty() {
        param_lines.push(format!("    {}_Input in [[stage_in]]", shader.name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}& {} [[buffer({})]]",
            shader_type_msl(p.ty),
            p.name,
            i,
        ));
    }

    out.push_str(&format!(
        "fragment {} {}(\n{}\n) {{\n",
        shader_type_msl(shader.return_type),
        shader.name,
        param_lines.join(",\n"),
    ));

    // Unpack stage_in members
    for p in &stage_in_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            shader_type_msl(p.ty),
            p.name,
            p.name,
        ));
    }

    let body = translate_shader_body(&shader.body_source);
    out.push_str(&indent_and_return(&body));
    out.push_str("}\n");
    Ok(out)
}

//! Call the quanta-compiler binary or use built-in emitters.

use quanta_ir::{CompilerOutput, KernelDef, KernelParam};
use std::collections::HashMap;

/// Compile a KernelDef to all available targets.
///
/// Strategy:
/// 1. Try quanta-compiler binary (supports LLVM targets + MSL + WGSL)
/// 2. Fall back to built-in MSL/WGSL emitters (no LLVM targets)
pub fn compile_kernel(kernel: &KernelDef) -> Result<CompilerOutput, String> {
    // Try calling the compiler binary for full output (AMD + NVIDIA + MSL + WGSL)
    if let Some(output) = try_compiler_binary(kernel) {
        return Ok(output);
    }

    // Fallback: built-in MSL/WGSL emitters (no LLVM → no AMD/NVIDIA)
    let msl = emit_msl(kernel)?;
    let wgsl = emit_wgsl(kernel)?;

    Ok(CompilerOutput {
        amd: None,
        nvidia: None,
        msl: Some(msl),
        wgsl: Some(wgsl),
        llvm_ir: None,
    })
}

/// Try to find and call the quanta-compiler binary.
fn try_compiler_binary(kernel: &KernelDef) -> Option<CompilerOutput> {
    let binary = find_compiler_binary()?;
    // Compiler binary found

    // Serialize KernelDef to bincode
    let input = quanta_ir::serialize_kernel(kernel).ok()?;

    // Call the binary: stdin = KernelDef, stdout = CompilerOutput
    let result = std::process::Command::new(&binary)
        .arg("--targets")
        .arg("nvptx,amdgpu")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = result.ok()?;

    // Write input
    use std::io::Write;
    child.stdin.take()?.write_all(&input).ok()?;

    // Read output
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[quanta] compiler failed: {}", stderr);
        return None;
    }

    let result = quanta_ir::deserialize_output(&output.stdout);
    if let Err(ref e) = result {
        eprintln!("[quanta] deserialize error: {}", e);
    }
    result.ok()
}

/// Find the quanta-compiler binary.
/// Search order:
/// 1. QUANTA_COMPILER env var
/// 2. ../quanta-compiler/target/release/quanta-compiler (development)
/// 3. ../quanta-compiler/target/debug/quanta-compiler (development)
/// 4. quanta-compiler in PATH
fn find_compiler_binary() -> Option<String> {
    // 1. Environment variable
    if let Ok(path) = std::env::var("QUANTA_COMPILER")
        && std::path::Path::new(&path).exists()
    {
        return Some(path);
    }

    // 2. Development: workspace target directory
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    for sub in &[
        "target/release/quanta-compiler",
        "../target/release/quanta-compiler",
        "../../target/release/quanta-compiler",
        "target/debug/quanta-compiler",
        "../target/debug/quanta-compiler",
        "../../target/debug/quanta-compiler",
    ] {
        let path = std::path::PathBuf::from(&manifest_dir).join(sub);
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    // 4. PATH
    if let Ok(output) = std::process::Command::new("which")
        .arg("quanta-compiler")
        .output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(path);
        }
    }

    None // Fall back to built-in emitters
}

// ============================================================================
// Built-in MSL emitter (fallback when quanta-compiler not available)
// ============================================================================

fn emit_msl(kernel: &KernelDef) -> Result<String, String> {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");
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
            _ => {} // textures — TODO
        }
    }
    param_lines.push("    uint _quark_id [[thread_position_in_grid]]".to_string());
    param_lines.push("    uint _local_id [[thread_position_in_threadgroup]]".to_string());
    param_lines.push("    uint _group_id [[threadgroup_position_in_grid]]".to_string());
    param_lines.push("    uint _group_size [[threads_per_threadgroup]]".to_string());

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
                "{}uint r{} = _quark_id; /* TODO: total count */\n",
                pad, dst.0
            ));
        }
        LocalId { dst } => {
            out.push_str(&format!("{}uint r{} = _local_id;\n", pad, dst.0));
        }
        GroupId { dst } => {
            out.push_str(&format!("{}uint r{} = _group_id;\n", pad, dst.0));
        }
        GroupSize { dst } => {
            out.push_str(&format!("{}uint r{} = _group_size;\n", pad, dst.0));
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
            };
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
        AtomicOp {
            dst,
            field,
            index,
            val,
            op,
            ty,
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
        _ => {
            out.push_str(&format!("{}/* TODO: {:?} */\n", pad, op));
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

// ============================================================================
// Built-in WGSL emitter (fallback)
// ============================================================================

fn emit_wgsl(kernel: &KernelDef) -> Result<String, String> {
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
        Break => {
            out.push_str(&format!("{}break;\n", pad));
        }
        _ => {
            out.push_str(&format!("{}// TODO: {:?}\n", pad, op));
        }
    }
}

// ============================================================================
// String-based body translators (Phase 1 fallback — replaced by KernelOps in Phase 2)
// ============================================================================

fn translate_body_to_msl(rust_source: &str) -> String {
    rust_source
        .replace("quark_id ()", "_quark_id")
        .replace("quark_id()", "_quark_id")
        .replace("local_id ()", "_local_id")
        .replace("local_id()", "_local_id")
        .replace("group_id ()", "_group_id")
        .replace("group_id()", "_group_id")
        .replace("group_size ()", "_group_size")
        .replace("group_size()", "_group_size")
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

fn translate_body_to_wgsl(rust_source: &str) -> String {
    rust_source
        .replace("quark_id ()", "_quark_id")
        .replace("quark_id()", "_quark_id")
        .replace("local_id ()", "gid.x")
        .replace("local_id()", "gid.x")
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

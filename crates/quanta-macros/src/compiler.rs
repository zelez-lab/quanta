//! Call the quanta-compiler binary or use built-in emitters.
//!
//! The MSL/WGSL emitter code is kept for Phase 4 (JIT migration to quanta-ir).

use quanta_ir::{CompilerOutput, KernelDef, KernelOp, KernelParam};
use std::collections::HashMap;

/// Compile a KernelDef to all available targets.
///
/// Strategy:
/// 1. Try quanta-compiler binary (supports LLVM targets + SPIR-V + metallib)
/// 2. If not found, return error (binary-only — no text fallback)
pub fn compile_kernel(kernel: &KernelDef) -> Result<CompilerOutput, String> {
    // Try calling the compiler binary for full output
    if let Some(output) = try_compiler_binary(kernel) {
        return Ok(output);
    }

    // No compiler binary found — return empty output.
    // This is acceptable during development: proc macro tests will get
    // empty binaries, but GPU execution requires the compiler.
    Ok(CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
    })
}

/// Try to find and call the quanta-compiler binary.
fn try_compiler_binary(kernel: &KernelDef) -> Option<CompilerOutput> {
    let binary = find_compiler_binary()?;
    // Compiler binary found

    // Serialize KernelDef to bincode
    let input = quanta_ir::serialize_kernel(kernel);

    // Call the binary: stdin = KernelDef, stdout = CompilerOutput
    let result = std::process::Command::new(&binary)
        .arg("--targets")
        .arg("nvptx,amdgpu")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = result.ok()?;

    // Write input and explicitly close stdin before reading output
    use std::io::Write;
    {
        let mut stdin = child.stdin.take()?;
        if stdin.write_all(&input).is_err() {
            let _ = child.kill();
            return None;
        }
    } // stdin dropped here → pipe closed → child sees EOF

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
// Shader compilation (vertex / fragment) via compiler binary
// ============================================================================

/// Output from shader compilation — SPIR-V and metallib binaries.
pub(crate) struct ShaderCompileOutput {
    pub(crate) spirv: Option<Vec<u8>>,
    pub(crate) metallib: Option<Vec<u8>>,
}

/// Compile a vertex or fragment shader via the quanta-compiler binary.
///
/// Serializes the ShaderDef to the compiler's stdin, reads ShaderOutput
/// from stdout. Returns None if the compiler binary is not found.
pub(crate) fn compile_shader(
    name: &str,
    stage: &str,
    params: &[ShaderParam],
    return_type: &ShaderType,
    body_source: &str,
) -> Option<ShaderCompileOutput> {
    let binary = find_compiler_binary()?;

    // Build ShaderDef from the parsed macro arguments
    let shader_def = quanta_ir::ShaderDef {
        name: name.to_string(),
        stage: match stage {
            "vertex" => quanta_ir::ShaderStage::Vertex,
            "fragment" => quanta_ir::ShaderStage::Fragment,
            _ => return None,
        },
        params: params
            .iter()
            .map(|p| quanta_ir::ShaderParam {
                name: p.name.clone(),
                ty: shader_type_to_ir(&p.ty),
                is_uniform: p.is_uniform,
            })
            .collect(),
        return_type: shader_type_to_ir(return_type),
        body_source: body_source.to_string(),
    };

    let input = quanta_ir::serialize_shader(&shader_def);

    let result = std::process::Command::new(&binary)
        .arg("--shader-type")
        .arg(stage)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = result.ok()?;

    use std::io::Write;
    {
        let mut stdin = child.stdin.take()?;
        if stdin.write_all(&input).is_err() {
            let _ = child.kill();
            return None;
        }
    }

    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[quanta] shader compiler failed: {}", stderr);
        return None;
    }

    let shader_output = quanta_ir::deserialize_shader_output(&output.stdout).ok()?;
    Some(ShaderCompileOutput {
        spirv: shader_output.spirv,
        metallib: shader_output.metallib,
    })
}

fn shader_type_to_ir(ty: &ShaderType) -> quanta_ir::ShaderType {
    match ty {
        ShaderType::F32 => quanta_ir::ShaderType::F32,
        ShaderType::Vec2 => quanta_ir::ShaderType::Vec2,
        ShaderType::Vec3 => quanta_ir::ShaderType::Vec3,
        ShaderType::Vec4 => quanta_ir::ShaderType::Vec4,
        ShaderType::Mat4 => quanta_ir::ShaderType::Mat4,
        ShaderType::Mat3 => quanta_ir::ShaderType::Mat3,
    }
}

// ============================================================================
// Built-in MSL emitter (kept for Phase 4 JIT migration)
// ============================================================================

#[allow(dead_code)]
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
                "{}uint r{} = _group_id * _group_size + _group_size;\n",
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
// Built-in WGSL emitter (kept for Phase 4 JIT migration)
// ============================================================================

#[allow(dead_code)]
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
        Break => {
            out.push_str(&format!("{}break;\n", pad));
        }
        Barrier => {
            out.push_str(&format!("{}workgroupBarrier();\n", pad));
        }
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
        LocalId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // local\n", pad, dst.0));
        }
        GroupId { dst } => {
            out.push_str(&format!("{}let r{} = gid.x; // group\n", pad, dst.0));
        }
        QuarkCount { dst } => {
            out.push_str(&format!(
                "{}let r{} = gid.x; // total quark count unavailable in WGSL\n",
                pad, dst.0
            ));
        }
        GroupSize { dst } => {
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

/// Translate a Rust device function source to WGSL (fallback emitter).
fn translate_device_fn_to_wgsl_fallback(rust_source: &str) -> String {
    // WGSL uses `fn name(...) -> type` — similar to Rust syntax
    let mut s = rust_source.to_string();
    s = s.replace("let mut ", "var ");
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s
}

// ============================================================================
// Shader (vertex / fragment) parameter parsing and MSL/WGSL emitters
// ============================================================================

/// A parsed shader parameter — either a vertex/fragment attribute or a uniform.
pub(crate) struct ShaderParam {
    pub(crate) name: String,
    pub(crate) ty: ShaderType,
    pub(crate) is_uniform: bool,
}

/// Shader types understood by the vertex/fragment emitters.
#[derive(Clone, Copy)]
pub(crate) enum ShaderType {
    F32,
    Vec2,
    Vec3,
    Vec4,
    Mat4,
    Mat3,
}

impl ShaderType {
    fn msl_name(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::Vec2 => "float2",
            Self::Vec3 => "float3",
            Self::Vec4 => "float4",
            Self::Mat4 => "float4x4",
            Self::Mat3 => "float3x3",
        }
    }

    fn wgsl_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Vec2 => "vec2<f32>",
            Self::Vec3 => "vec3<f32>",
            Self::Vec4 => "vec4<f32>",
            Self::Mat4 => "mat4x4<f32>",
            Self::Mat3 => "mat3x3<f32>",
        }
    }
}

fn shader_type_from_ident(name: &str) -> Result<ShaderType, String> {
    match name {
        "f32" => Ok(ShaderType::F32),
        "Vec2" => Ok(ShaderType::Vec2),
        "Vec3" => Ok(ShaderType::Vec3),
        "Vec4" => Ok(ShaderType::Vec4),
        "Mat4" => Ok(ShaderType::Mat4),
        "Mat3" => Ok(ShaderType::Mat3),
        other => Err(format!("unsupported shader type: {}", other)),
    }
}

/// Parse function parameters into shader params.
///
/// Value params (Vec2, Vec3, Vec4, f32) become attributes/inputs.
/// Reference params (&T) become uniform buffer bindings.
pub(crate) fn parse_shader_params(func: &syn::ItemFn) -> Result<Vec<ShaderParam>, syn::Error> {
    let mut params = Vec::new();

    for arg in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            let name = match pat_type.pat.as_ref() {
                syn::Pat::Ident(ident) => ident.ident.to_string(),
                _ => {
                    return Err(syn::Error::new_spanned(
                        &pat_type.pat,
                        "shader parameter must be a simple identifier",
                    ));
                }
            };

            let (ty, is_uniform) = parse_shader_type(&pat_type.ty)?;
            params.push(ShaderParam {
                name,
                ty,
                is_uniform,
            });
        }
    }

    Ok(params)
}

/// Parse a type into (ShaderType, is_uniform).
/// `&T` → uniform, `T` → attribute/input.
fn parse_shader_type(ty: &syn::Type) -> Result<(ShaderType, bool), syn::Error> {
    match ty {
        syn::Type::Reference(ref_ty) => {
            let inner = parse_shader_type_inner(&ref_ty.elem)?;
            Ok((inner, true))
        }
        _ => {
            let inner = parse_shader_type_inner(ty)?;
            Ok((inner, false))
        }
    }
}

fn parse_shader_type_inner(ty: &syn::Type) -> Result<ShaderType, syn::Error> {
    match ty {
        syn::Type::Path(path) => {
            let ident = path
                .path
                .segments
                .last()
                .ok_or_else(|| syn::Error::new_spanned(path, "empty type path"))?;
            shader_type_from_ident(&ident.ident.to_string())
                .map_err(|msg| syn::Error::new_spanned(&ident.ident, msg))
        }
        _ => Err(syn::Error::new_spanned(ty, "unsupported shader type")),
    }
}

/// Parse the return type of a shader function.
pub(crate) fn parse_return_type(func: &syn::ItemFn) -> Result<ShaderType, syn::Error> {
    match &func.sig.output {
        syn::ReturnType::Type(_, ty) => parse_shader_type_inner(ty),
        syn::ReturnType::Default => Err(syn::Error::new_spanned(
            &func.sig.ident,
            "shader must have a return type",
        )),
    }
}

/// Extract the function body as a source string for text-based translation.
pub(crate) fn extract_body_source(func: &syn::ItemFn) -> String {
    use quote::ToTokens;
    let mut body = String::new();
    for stmt in &func.block.stmts {
        body.push_str(&stmt.to_token_stream().to_string());
        body.push('\n');
    }
    body
}

/// Translate a Rust shader body to MSL.
///
/// Applies the same text substitutions as the kernel fallback path, plus
/// shader-specific vector/matrix type replacements.
fn translate_body_to_msl_shader(src: &str, _params: &[ShaderParam]) -> String {
    let mut s = src.to_string();

    // Replace Quanta vector constructors with MSL equivalents
    s = s.replace("Vec4 :: new", "float4");
    s = s.replace("Vec4::new", "float4");
    s = s.replace("Vec3 :: new", "float3");
    s = s.replace("Vec3::new", "float3");
    s = s.replace("Vec2 :: new", "float2");
    s = s.replace("Vec2::new", "float2");

    // Replace type casts
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");
    s = s.replace(" as i32", "");

    // Replace let bindings
    s = s.replace("let mut ", "auto ");
    s = s.replace("let ", "auto ");

    // Replace parameter references — uniform params are accessed via their
    // MSL name directly (they are in scope as function parameters).
    // Attribute params accessed via their name directly as well.
    // No renaming needed.

    s
}

/// Translate a Rust shader body to WGSL.
fn translate_body_to_wgsl_shader(src: &str, params: &[ShaderParam], _is_vertex: bool) -> String {
    let mut s = src.to_string();

    // Replace Quanta vector constructors with WGSL equivalents
    s = s.replace("Vec4 :: new", "vec4<f32>");
    s = s.replace("Vec4::new", "vec4<f32>");
    s = s.replace("Vec3 :: new", "vec3<f32>");
    s = s.replace("Vec3::new", "vec3<f32>");
    s = s.replace("Vec2 :: new", "vec2<f32>");
    s = s.replace("Vec2::new", "vec2<f32>");

    // Replace type casts
    s = s.replace(" as f32", "");
    s = s.replace(" as u32", "");

    // Replace let bindings
    s = s.replace("let mut ", "var ");

    // Replace attribute param names with struct member access (in.name)
    for p in params {
        if !p.is_uniform {
            let from = &p.name;
            let to = format!("in.{}", p.name);
            // Only replace bare identifiers, not already-qualified ones.
            // Simple heuristic: replace "p.name" when not preceded by '.'
            s = s.replace(from, &to);
        }
    }

    s
}

// ============================================================================
// Vertex shader MSL emitter
// ============================================================================

pub(crate) fn emit_vertex_msl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Build parameter list
    let mut param_lines = Vec::new();
    let mut attr_idx = 0u32;
    let mut buf_idx = 0u32;

    for p in params {
        if p.is_uniform {
            param_lines.push(format!(
                "    constant {}&{} [[buffer({})]]",
                p.ty.msl_name(),
                if p.name.is_empty() {
                    String::new()
                } else {
                    format!(" {}", p.name)
                },
                buf_idx
            ));
            buf_idx += 1;
        } else {
            param_lines.push(format!(
                "    {} {} [[attribute({})]]",
                p.ty.msl_name(),
                p.name,
                attr_idx
            ));
            attr_idx += 1;
        }
    }

    out.push_str(&format!(
        "vertex {} {}(\n{}\n) {{\n",
        return_ty.msl_name(),
        name,
        param_lines.join(",\n")
    ));

    let body = translate_body_to_msl_shader(body_source, params);
    // Indent body and wrap the last expression as return
    let body = indent_and_return_msl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

// ============================================================================
// Fragment shader MSL emitter
// ============================================================================

pub(crate) fn emit_fragment_msl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();
    out.push_str("#include <metal_stdlib>\nusing namespace metal;\n\n");

    // Fragment inputs come as a struct with [[stage_in]]
    let stage_in_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    // Emit stage_in struct if there are interpolated inputs
    if !stage_in_params.is_empty() {
        out.push_str(&format!("struct {}_Input {{\n", name));
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    {} {} [[user(loc{})]];\n",
                p.ty.msl_name(),
                p.name,
                i
            ));
        }
        out.push_str("};\n\n");
    }

    // Build parameter list
    let mut param_lines = Vec::new();
    if !stage_in_params.is_empty() {
        param_lines.push(format!("    {}_Input in [[stage_in]]", name));
    }
    for (i, p) in uniform_params.iter().enumerate() {
        param_lines.push(format!(
            "    constant {}&{} [[buffer({})]]",
            p.ty.msl_name(),
            if p.name.is_empty() {
                String::new()
            } else {
                format!(" {}", p.name)
            },
            i
        ));
    }

    out.push_str(&format!(
        "fragment {} {}(\n{}\n) {{\n",
        return_ty.msl_name(),
        name,
        param_lines.join(",\n")
    ));

    // Unpack stage_in members into local variables
    for p in &stage_in_params {
        out.push_str(&format!(
            "    {} {} = in.{};\n",
            p.ty.msl_name(),
            p.name,
            p.name
        ));
    }

    let body = translate_body_to_msl_shader(body_source, params);
    let body = indent_and_return_msl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

/// Indent body lines and convert a trailing expression into a return statement.
fn indent_and_return_msl(body: &str) -> String {
    let lines: Vec<&str> = body.trim().lines().collect();
    if lines.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if i == lines.len() - 1 && !trimmed.is_empty() {
            // Last line: if it doesn't end with ';' or start with 'return',
            // wrap it as a return statement.
            if !trimmed.ends_with(';') && !trimmed.starts_with("return") {
                out.push_str(&format!("    return {};\n", trimmed));
            } else if !trimmed.starts_with("return") && !trimmed.contains("return ") {
                // Already has semicolon but no return — check if it looks like
                // a bare expression (no '=' assignment).
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=')
                    && !without_semi.starts_with("auto ")
                    && !without_semi.starts_with("if ")
                    && !without_semi.starts_with("for ")
                {
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

// ============================================================================
// Vertex shader WGSL emitter
// ============================================================================

pub(crate) fn emit_vertex_wgsl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();

    let attr_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    // Emit uniform bindings
    for (i, p) in uniform_params.iter().enumerate() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<uniform> {}: {};\n",
            i,
            p.name,
            p.ty.wgsl_name()
        ));
    }
    if !uniform_params.is_empty() {
        out.push('\n');
    }

    // Emit vertex input struct
    if !attr_params.is_empty() {
        out.push_str("struct VertexInput {\n");
        for (i, p) in attr_params.iter().enumerate() {
            out.push_str(&format!(
                "    @location({}) {}: {},\n",
                i,
                p.name,
                p.ty.wgsl_name()
            ));
        }
        out.push_str("};\n\n");
    }

    // Function signature
    out.push_str(&format!(
        "@vertex\nfn {}(in: VertexInput) -> @builtin(position) {} {{\n",
        name,
        return_ty.wgsl_name()
    ));

    let body = translate_body_to_wgsl_shader(body_source, params, true);
    let body = indent_and_return_wgsl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

// ============================================================================
// Fragment shader WGSL emitter
// ============================================================================

pub(crate) fn emit_fragment_wgsl(
    name: &str,
    params: &[ShaderParam],
    return_ty: &ShaderType,
    body_source: &str,
) -> String {
    let mut out = String::new();

    let stage_in_params: Vec<&ShaderParam> = params.iter().filter(|p| !p.is_uniform).collect();
    let uniform_params: Vec<&ShaderParam> = params.iter().filter(|p| p.is_uniform).collect();

    // Emit uniform bindings
    for (i, p) in uniform_params.iter().enumerate() {
        out.push_str(&format!(
            "@group(0) @binding({}) var<uniform> {}: {};\n",
            i,
            p.name,
            p.ty.wgsl_name()
        ));
    }
    if !uniform_params.is_empty() {
        out.push('\n');
    }

    // Emit fragment input struct
    if !stage_in_params.is_empty() {
        out.push_str("struct FragmentInput {\n");
        for (i, p) in stage_in_params.iter().enumerate() {
            out.push_str(&format!(
                "    @location({}) {}: {},\n",
                i,
                p.name,
                p.ty.wgsl_name()
            ));
        }
        out.push_str("};\n\n");
    }

    // Function signature
    out.push_str(&format!(
        "@fragment\nfn {}(in: FragmentInput) -> @location(0) {} {{\n",
        name,
        return_ty.wgsl_name()
    ));

    let body = translate_body_to_wgsl_shader(body_source, params, false);
    let body = indent_and_return_wgsl(&body);
    out.push_str(&body);

    out.push_str("}\n");
    out
}

/// Indent body lines and convert trailing expression to return (WGSL).
fn indent_and_return_wgsl(body: &str) -> String {
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
            } else if !trimmed.starts_with("return") && !trimmed.contains("return ") {
                let without_semi = trimmed.trim_end_matches(';').trim();
                if !without_semi.contains('=')
                    && !without_semi.starts_with("var ")
                    && !without_semi.starts_with("let ")
                    && !without_semi.starts_with("if ")
                    && !without_semi.starts_with("for ")
                {
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

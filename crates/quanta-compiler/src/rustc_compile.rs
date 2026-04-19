//! Compile Rust kernel source via rustc → LLVM IR → retarget to GPU.
//!
//! This replaces the custom AST parser for LLVM targets.
//! rustc handles ALL Rust syntax — no manual pattern matching needed.
//!
//! Flow:
//!   1. Receive Rust source + parameter metadata
//!   2. Generate temp crate with GPU builtin stubs
//!   3. Call rustc --emit=llvm-ir
//!   4. Read LLVM IR text
//!   5. Retarget: replace stubs with GPU intrinsics, fix address spaces
//!   6. Return retargeted LLVM IR for optimization + emission

use quanta_ir::KernelParam;
use std::process::Command;

/// Compile a Rust kernel function to LLVM IR via rustc.
pub fn rust_to_llvm_ir(
    kernel_name: &str,
    params: &[KernelParam],
    rust_body: &str,
) -> Result<String, String> {
    let tmp_dir = std::env::temp_dir().join("quanta_rustc");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("mkdir: {}", e))?;

    let src_path = tmp_dir.join(format!("{}.rs", kernel_name));
    let ir_path = tmp_dir.join(format!("{}.ll", kernel_name));

    // Generate the Rust source file
    let source = generate_kernel_source(kernel_name, params, rust_body);
    std::fs::write(&src_path, &source).map_err(|e| format!("write: {}", e))?;

    // Call rustc
    let output = Command::new("rustc")
        .arg("--edition=2024")
        .arg("--emit=llvm-ir")
        .arg("--crate-type=lib")
        .arg("-C")
        .arg("opt-level=0") // we optimize later with our LLVM
        .arg("-o")
        .arg(&ir_path)
        .arg(&src_path)
        .output()
        .map_err(|e| format!("rustc exec: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("rustc failed:\n{}", stderr));
    }

    // Read LLVM IR
    let ir = std::fs::read_to_string(&ir_path).map_err(|e| format!("read IR: {}", e))?;

    // Retarget for GPU
    let retargeted = retarget_for_gpu(kernel_name, &ir)?;

    // Cleanup
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&ir_path);

    Ok(retargeted)
}

/// Generate a Rust source file with GPU builtin stubs + the kernel function.
fn generate_kernel_source(kernel_name: &str, params: &[KernelParam], rust_body: &str) -> String {
    let mut src = String::new();

    src.push_str("#![no_std]\n");
    src.push_str("#![allow(unused, non_snake_case)]\n\n");

    // Core trait impls needed for no_std float math
    src.push_str("extern crate core;\n");
    src.push_str("use core::panic::PanicInfo;\n");
    src.push_str("#[panic_handler]\nfn panic(_: &PanicInfo) -> ! { loop {} }\n\n");

    // GPU builtin stubs — these become `call @quark_id()` in LLVM IR
    // which we later replace with actual GPU intrinsics
    src.push_str("unsafe extern \"C\" {\n");
    src.push_str("    fn quark_id() -> u32;\n");
    src.push_str("    fn quark_count() -> u32;\n");
    src.push_str("    fn local_id() -> u32;\n");
    src.push_str("    fn group_id() -> u32;\n");
    src.push_str("    fn group_size() -> u32;\n");
    src.push_str("    fn barrier();\n");
    src.push_str("    fn atomic_add(ptr: *mut u32, val: u32) -> u32;\n");
    src.push_str("}\n\n");

    // Math function stubs for no_std
    src.push_str("mod math {\n");
    src.push_str("    unsafe extern \"C\" {\n");
    src.push_str("        pub fn sinf(x: f32) -> f32;\n");
    src.push_str("        pub fn cosf(x: f32) -> f32;\n");
    src.push_str("        pub fn sqrtf(x: f32) -> f32;\n");
    src.push_str("        pub fn fabsf(x: f32) -> f32;\n");
    src.push_str("    }\n");
    src.push_str("}\n\n");

    // Trait to provide .sin(), .cos(), .sqrt(), .abs() on f32
    src.push_str("trait GpuFloat {\n");
    src.push_str("    fn sin(self) -> f32;\n");
    src.push_str("    fn cos(self) -> f32;\n");
    src.push_str("    fn sqrt(self) -> f32;\n");
    src.push_str("    fn abs(self) -> f32;\n");
    src.push_str("}\n");
    src.push_str("impl GpuFloat for f32 {\n");
    src.push_str("    fn sin(self) -> f32 { unsafe { math::sinf(self) } }\n");
    src.push_str("    fn cos(self) -> f32 { unsafe { math::cosf(self) } }\n");
    src.push_str("    fn sqrt(self) -> f32 { unsafe { math::sqrtf(self) } }\n");
    src.push_str("    fn abs(self) -> f32 { unsafe { math::fabsf(self) } }\n");
    src.push_str("}\n\n");

    // The kernel function
    src.push_str("#[unsafe(no_mangle)]\n");
    src.push_str(&format!("pub unsafe fn {}(", kernel_name));

    // Parameters
    let mut param_strs = Vec::new();
    for param in params {
        match param {
            KernelParam::FieldRead {
                name, scalar_type, ..
            } => {
                let ty = scalar_type_to_rust(scalar_type);
                param_strs.push(format!("{}: *const {}", name, ty));
            }
            KernelParam::FieldWrite {
                name, scalar_type, ..
            } => {
                let ty = scalar_type_to_rust(scalar_type);
                param_strs.push(format!("{}: *mut {}", name, ty));
            }
            KernelParam::Constant {
                name, scalar_type, ..
            } => {
                let ty = scalar_type_to_rust(scalar_type);
                param_strs.push(format!("{}: {}", name, ty));
            }
            _ => {}
        }
    }
    src.push_str(&param_strs.join(", "));
    src.push_str(") {\n");
    src.push_str("unsafe {\n");
    src.push_str(rust_body);
    src.push_str("\n}\n}\n");

    src
}

/// Retarget LLVM IR from host CPU to GPU.
fn retarget_for_gpu(_kernel_name: &str, ir: &str) -> Result<String, String> {
    let mut out = ir.to_string();

    // Replace target triple (will be set properly by our LLVM compilation)
    // Just remove the host triple for now
    if let Some(start) = out.find("target triple = \"")
        && let Some(end) = out[start..].find('\n')
    {
        out.replace_range(
            start..start + end,
            "target triple = \"nvptx64-nvidia-cuda\"",
        );
    }

    // Replace GPU builtin stubs with markers
    // These get replaced with actual intrinsics when we load into inkwell
    out = out.replace("call i32 @quark_id()", "call i32 @__quanta_quark_id()");
    out = out.replace("call i32 @local_id()", "call i32 @__quanta_local_id()");
    out = out.replace("call i32 @group_id()", "call i32 @__quanta_group_id()");
    out = out.replace("call void @barrier()", "call void @__quanta_barrier()");

    // Replace math functions with LLVM intrinsics
    out = out.replace("call float @sinf(", "call float @llvm.sin.f32(");
    out = out.replace("call float @cosf(", "call float @llvm.cos.f32(");
    out = out.replace("call float @sqrtf(", "call float @llvm.sqrt.f32(");
    out = out.replace("call float @fabsf(", "call float @llvm.fabs.f32(");

    Ok(out)
}

fn scalar_type_to_rust(ty: &quanta_ir::ScalarType) -> &'static str {
    match ty {
        quanta_ir::ScalarType::F32 => "f32",
        quanta_ir::ScalarType::F64 => "f64",
        quanta_ir::ScalarType::U8 => "u8",
        quanta_ir::ScalarType::U16 => "u16",
        quanta_ir::ScalarType::U32 => "u32",
        quanta_ir::ScalarType::U64 => "u64",
        quanta_ir::ScalarType::I8 => "i8",
        quanta_ir::ScalarType::I16 => "i16",
        quanta_ir::ScalarType::I32 => "i32",
        quanta_ir::ScalarType::I64 => "i64",
        quanta_ir::ScalarType::Bool => "bool",
        quanta_ir::ScalarType::F16 => "f32", // no f16 in Rust, use f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quanta_ir::ScalarType;

    #[test]
    fn test_generate_source() {
        let params = vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ];

        let body = r#"
    let i = quark_id() as usize;
    *result.add(i) = *a.add(i) + *b.add(i);
"#;

        let source = generate_kernel_source("vector_add", &params, body);
        assert!(source.contains("pub unsafe fn vector_add("));
        assert!(source.contains("a: *const f32"));
        assert!(source.contains("result: *mut f32"));
        assert!(source.contains("quark_id"));
    }
}

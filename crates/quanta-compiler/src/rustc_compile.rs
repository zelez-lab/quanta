//! Compile Rust kernel source via rustc → LLVM IR → retarget to GPU.
//!
//! rustc handles ALL Rust syntax. No custom AST parser needed.
//!
//! Flow:
//!   1. Wrap user's kernel body in a temp crate with GpuSlice wrappers
//!   2. Call rustc --emit=llvm-ir
//!   3. Retarget IR: strip panics, rewrite address spaces, replace intrinsics
//!   4. Return clean GPU-ready LLVM IR

use quanta_ir::KernelParam;
use std::process::Command;

/// Compile a Rust kernel function body to GPU-ready LLVM IR via rustc.
///
/// `rust_body` is the original function body as written by the user,
/// using `a[i]` syntax on slice-like parameters and `quark_id()` etc.
/// `device_sources` contains Rust source for inner device functions that
/// should be included in the generated crate so rustc can resolve calls.
pub fn rust_to_llvm_ir(
    kernel_name: &str,
    params: &[KernelParam],
    rust_body: &str,
) -> Result<String, String> {
    rust_to_llvm_ir_with_devices(kernel_name, params, rust_body, &[])
}

/// Full version with device function support.
pub fn rust_to_llvm_ir_with_devices(
    kernel_name: &str,
    params: &[KernelParam],
    rust_body: &str,
    device_sources: &[String],
) -> Result<String, String> {
    let tmp_dir = std::env::temp_dir().join("quanta_rustc");
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("mkdir: {}", e))?;

    let src_path = tmp_dir.join(format!("{}.rs", kernel_name));
    let ir_path = tmp_dir.join(format!("{}.ll", kernel_name));

    let source = generate_kernel_source(kernel_name, params, rust_body, device_sources);
    std::fs::write(&src_path, &source).map_err(|e| format!("write: {}", e))?;

    let output = Command::new("rustc")
        .arg("--edition=2024")
        .arg("--emit=llvm-ir")
        .arg("--crate-type=lib")
        .arg("-C")
        .arg("opt-level=0")
        .arg("-o")
        .arg(&ir_path)
        .arg(&src_path)
        .output()
        .map_err(|e| format!("rustc exec: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("rustc failed:\n{}", stderr));
    }

    let ir = std::fs::read_to_string(&ir_path).map_err(|e| format!("read IR: {}", e))?;
    let retargeted = retarget_for_gpu(kernel_name, &ir)?;

    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&ir_path);

    Ok(retargeted)
}

/// Generate the Rust source file that rustc will compile.
///
/// Key: GpuSlice<T> wrapper provides `a[i]` indexing on raw pointers,
/// so the user's kernel body works unchanged.
fn generate_kernel_source(
    kernel_name: &str,
    params: &[KernelParam],
    rust_body: &str,
    device_sources: &[String],
) -> String {
    let mut src = String::new();

    src.push_str("#![no_std]\n");
    src.push_str("#![allow(unused, non_snake_case)]\n\n");

    // Panic handler (no_std requirement — will be stripped from IR)
    src.push_str("use core::panic::PanicInfo;\n");
    src.push_str("#[panic_handler]\nfn panic(_: &PanicInfo) -> ! { loop {} }\n\n");

    // === GpuSlice: indexable wrapper over raw pointers ===
    // This lets the user write `a[i]` in their kernel body
    src.push_str("
#[repr(transparent)]
#[derive(Copy, Clone)]
struct GpuSlice<T: Copy>(*const T);

#[repr(transparent)]
#[derive(Copy, Clone)]
struct GpuSliceMut<T: Copy>(*mut T);

impl<T: Copy> GpuSlice<T> {
    #[inline(always)]
    fn get(&self, i: u32) -> T { unsafe { *self.0.add(i as usize) } }
}

impl<T: Copy> GpuSliceMut<T> {
    #[inline(always)]
    fn get(&self, i: u32) -> T { unsafe { *self.0.add(i as usize) } }
    #[inline(always)]
    fn set(&self, i: u32, val: T) { unsafe { *(self.0.add(i as usize) as *mut T) = val } }
}

impl<T: Copy> core::ops::Index<u32> for GpuSlice<T> {
    type Output = T;
    fn index(&self, i: u32) -> &T { unsafe { &*self.0.add(i as usize) } }
}

impl<T: Copy> core::ops::Index<u32> for GpuSliceMut<T> {
    type Output = T;
    fn index(&self, i: u32) -> &T { unsafe { &*self.0.add(i as usize) } }
}

impl<T: Copy> core::ops::IndexMut<u32> for GpuSliceMut<T> {
    fn index_mut(&mut self, i: u32) -> &mut T { unsafe { &mut *(self.0.add(i as usize) as *mut T) } }
}
");

    // === GPU builtin stubs ===
    src.push_str(
        "
unsafe extern \"C\" {
    fn __quanta_quark_id() -> u32;
    fn __quanta_quark_count() -> u32;
    fn __quanta_proton_id() -> u32;
    fn __quanta_nucleus_id() -> u32;
    fn __quanta_proton_size() -> u32;
    fn __quanta_barrier();
}

#[inline(always)]
fn quark_id() -> u32 { unsafe { __quanta_quark_id() } }
#[inline(always)]
fn quark_count() -> u32 { unsafe { __quanta_quark_count() } }
#[inline(always)]
fn proton_id() -> u32 { unsafe { __quanta_proton_id() } }
#[inline(always)]
fn nucleus_id() -> u32 { unsafe { __quanta_nucleus_id() } }
#[inline(always)]
fn proton_size() -> u32 { unsafe { __quanta_proton_size() } }
#[inline(always)]
fn barrier() { unsafe { __quanta_barrier() } }
",
    );

    // === Math stubs (extern C → replaced with LLVM intrinsics in retargeting) ===
    src.push_str(
        "
unsafe extern \"C\" {
    fn sinf(x: f32) -> f32;
    fn cosf(x: f32) -> f32;
    fn sqrtf(x: f32) -> f32;
    fn fabsf(x: f32) -> f32;
    fn expf(x: f32) -> f32;
    fn logf(x: f32) -> f32;
    fn powf(x: f32, y: f32) -> f32;
    fn floorf(x: f32) -> f32;
    fn ceilf(x: f32) -> f32;
}

trait GpuFloat {
    fn sin(self) -> f32;
    fn cos(self) -> f32;
    fn sqrt(self) -> f32;
    fn abs(self) -> f32;
    fn exp(self) -> f32;
    fn log(self) -> f32;
    fn powf(self, y: f32) -> f32;
    fn floor(self) -> f32;
    fn ceil(self) -> f32;
}

impl GpuFloat for f32 {
    #[inline(always)] fn sin(self) -> f32 { unsafe { sinf(self) } }
    #[inline(always)] fn cos(self) -> f32 { unsafe { cosf(self) } }
    #[inline(always)] fn sqrt(self) -> f32 { unsafe { sqrtf(self) } }
    #[inline(always)] fn abs(self) -> f32 { unsafe { fabsf(self) } }
    #[inline(always)] fn exp(self) -> f32 { unsafe { expf(self) } }
    #[inline(always)] fn log(self) -> f32 { unsafe { logf(self) } }
    #[inline(always)] fn powf(self, y: f32) -> f32 { unsafe { powf(self, y) } }
    #[inline(always)] fn floor(self) -> f32 { unsafe { floorf(self) } }
    #[inline(always)] fn ceil(self) -> f32 { unsafe { ceilf(self) } }
}
",
    );

    // === Device functions (inner helper functions from the kernel body) ===
    for device_src in device_sources {
        // Emit as module-level functions. The Rust source from the proc macro
        // is already valid Rust — just prefix with #[inline(always)].
        src.push_str("#[inline(always)]\n");
        src.push_str(device_src);
        src.push_str("\n\n");
    }

    // === The kernel function ===
    src.push_str("#[unsafe(no_mangle)]\n");
    src.push_str(&format!("pub unsafe fn {}(", kernel_name));

    // Parameters: &[T] → GpuSlice<T>, &mut [T] → GpuSliceMut<T>
    let mut param_strs = Vec::new();
    for param in params {
        match param {
            KernelParam::FieldRead {
                name, scalar_type, ..
            } => {
                let ty = scalar_type_to_rust(scalar_type);
                param_strs.push(format!("mut {}: GpuSlice<{}>", name, ty));
            }
            KernelParam::FieldWrite {
                name, scalar_type, ..
            } => {
                let ty = scalar_type_to_rust(scalar_type);
                param_strs.push(format!("mut {}: GpuSliceMut<{}>", name, ty));
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
    src.push_str(rust_body);
    src.push_str("\n}\n");

    src
}

/// Retarget LLVM IR from host CPU to GPU.
///
/// 1. Strip panic paths and unreachable code
/// 2. Replace GPU builtin stubs (already named __quanta_*)
/// 3. Math intrinsics are already LLVM intrinsics (sinf32 → llvm.sin.f32)
/// 4. Add kernel metadata
fn retarget_for_gpu(kernel_name: &str, ir: &str) -> Result<String, String> {
    let mut lines: Vec<String> = Vec::new();
    let mut skip_panic_fn = false;

    for line in ir.lines() {
        // Skip panic-related functions entirely
        if line.starts_with("define")
            && (line.contains("panic")
                || line.contains("rust_begin_unwind")
                || line.contains("__rust_alloc_error_handler"))
        {
            skip_panic_fn = true;
            continue;
        }
        if skip_panic_fn {
            if line == "}" {
                skip_panic_fn = false;
            }
            continue;
        }

        // Skip panic calls within the kernel
        if line.contains("call void @_ZN4core9panicking")
            || line.contains("call void @_ZN4core6option13unwrap_failed")
        {
            continue;
        }

        // Replace unreachable after stripped panic calls with ret void
        if line.trim() == "unreachable" {
            lines.push("  ret void".to_string());
            continue;
        }

        // Skip panic-related global constants
        if line.starts_with("@alloc_") && line.contains("panic") {
            continue;
        }

        lines.push(line.to_string());
    }

    let mut out = lines.join("\n");

    // Replace target triple
    if let Some(start) = out.find("target triple = \"")
        && let Some(end) = out[start..].find('\n')
    {
        out.replace_range(
            start..start + end,
            "target triple = \"nvptx64-nvidia-cuda\"",
        );
    }

    // Replace math extern stubs with LLVM intrinsics
    out = out.replace("call float @sinf(", "call float @llvm.sin.f32(");
    out = out.replace("call float @cosf(", "call float @llvm.cos.f32(");
    out = out.replace("call float @sqrtf(", "call float @llvm.sqrt.f32(");
    out = out.replace("call float @fabsf(", "call float @llvm.fabs.f32(");
    out = out.replace("call float @expf(", "call float @llvm.exp.f32(");
    out = out.replace("call float @logf(", "call float @llvm.log.f32(");
    out = out.replace("call float @powf(", "call float @llvm.pow.f32(");
    out = out.replace("call float @floorf(", "call float @llvm.floor.f32(");
    out = out.replace("call float @ceilf(", "call float @llvm.ceil.f32(");

    // Add kernel entry metadata at the end
    out.push_str(&format!(
        "\n!nvvm.annotations = !{{!quanta_kernel}}\n!quanta_kernel = !{{ptr @{}, !\"kernel\", i32 1}}\n",
        kernel_name
    ));

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
        quanta_ir::ScalarType::I4 => "i32",
        quanta_ir::ScalarType::Bool => "bool",
        quanta_ir::ScalarType::F16 => "f32",
        quanta_ir::ScalarType::BF16 => "f32",
        quanta_ir::ScalarType::FP8E5M2 | quanta_ir::ScalarType::FP8E4M3 => "f32",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quanta_ir::ScalarType;

    #[test]
    fn test_generate_source_with_gpu_slice() {
        let params = vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ];

        let body = "    let i = quark_id();\n    result[i] = a[i] + 1.0;";
        let source = generate_kernel_source("test_kernel", &params, body, &[]);

        // Verify GpuSlice wrapper is present
        assert!(source.contains("struct GpuSlice"));
        assert!(source.contains("struct GpuSliceMut"));
        // Verify parameter types use GpuSlice
        assert!(source.contains("a: GpuSlice<f32>"));
        assert!(source.contains("result: GpuSliceMut<f32>"));
        // Verify builtin stubs
        assert!(source.contains("fn __quanta_quark_id() -> u32"));
        // Verify user body is included
        assert!(source.contains("result[i] = a[i] + 1.0"));
    }
}

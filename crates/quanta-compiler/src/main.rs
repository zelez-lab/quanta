//! Quanta kernel compiler — reads KernelDef, emits GPU code.
//!
//! Usage:
//!   echo '<bincode>' | quanta-compiler --targets nvptx,amdgpu
//!   quanta-compiler --test-ir    # test with a built-in vector_add kernel

mod emit_msl;
mod emit_spirv;
#[allow(dead_code)]
mod emit_wgsl;
mod metallib;
mod pipeline;
mod rustc_compile;
mod shader_pipeline;
mod targets;
mod to_llvm;

use targets::GpuTarget;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--test-ir") {
        pipeline::test_ir();
        return;
    }

    if args.iter().any(|a| a == "--test-ptx") {
        pipeline::test_ptx();
        return;
    }

    if args.iter().any(|a| a == "--test-amd") {
        pipeline::test_amd();
        return;
    }

    if args.iter().any(|a| a == "--test-complex") {
        pipeline::test_complex();
        return;
    }

    if args.iter().any(|a| a == "--test-spirv") {
        pipeline::test_spirv();
        return;
    }

    if args.iter().any(|a| a == "--test-rustc") {
        pipeline::test_rustc();
        return;
    }

    // Shader mode: --shader-type vertex|fragment
    if let Some(shader_type) = shader_pipeline::parse_shader_type(&args) {
        shader_pipeline::compile_shader(shader_type);
        return;
    }

    // LLVM subprocess mode: compile a single target, write raw binary to stdout.
    // Used by the parent process to isolate LLVM fatal errors (abort on fsin etc.)
    if let Some(pos) = args.iter().position(|a| a == "--llvm-only")
        && let Some(target_name) = args.get(pos + 1)
    {
        let target = match target_name.as_str() {
            "nvptx" => GpuTarget::Nvptx,
            "amdgpu" => GpuTarget::Amdgpu,
            _ => std::process::exit(1),
        };
        pipeline::llvm_only(target);
        return;
    }

    // Normal mode: read KernelDef from stdin, emit results to stdout
    pipeline::compile_kernel(&args);
}

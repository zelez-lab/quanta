//! Shader compilation pipeline (vertex/fragment).

use crate::{emit_msl, emit_spirv, emit_wgsl, metallib};

/// Parse `--shader-type vertex|fragment` from CLI args.
pub fn parse_shader_type(args: &[String]) -> Option<&str> {
    for (i, arg) in args.iter().enumerate() {
        if arg == "--shader-type" {
            return args.get(i + 1).map(|s| s.as_str());
        }
    }
    None
}

/// Compile a vertex or fragment shader.
///
/// Reads a serialized ShaderDef from stdin, emits SPIR-V and metallib,
/// writes a serialized ShaderOutput to stdout.
pub fn compile_shader(stage: &str) {
    let mut input = Vec::new();
    std::io::Read::read_to_end(&mut std::io::stdin(), &mut input).unwrap();
    let shader: quanta_ir::ShaderDef = quanta_ir::deserialize_shader(&input).unwrap();

    let mut output = quanta_ir::ShaderOutput {
        spirv: None,
        metallib: None,
        wgsl: None,
    };

    // Emit SPIR-V
    let spirv_result = match stage {
        "vertex" => emit_spirv::emit_vertex(&shader),
        "fragment" => emit_spirv::emit_fragment(&shader),
        other => {
            eprintln!("[quanta] unknown shader type: {}", other);
            std::process::exit(1);
        }
    };
    // Unlike compute kernels, render shaders have no JIT fallback at
    // dispatch time — a missing binary means the shader can never run
    // on that backend. Fail the build instead of shipping a partial
    // artifact that panics at pipeline creation.
    match spirv_result {
        Ok(spirv) => output.spirv = Some(spirv),
        Err(e) => {
            eprintln!("[quanta] SPIR-V shader emitter error: {}", e);
            std::process::exit(1);
        }
    }

    // Emit MSL and compile to metallib
    let msl_result = match stage {
        "vertex" => emit_msl::emit_vertex_shader(&shader),
        "fragment" => emit_msl::emit_fragment_shader(&shader),
        _ => unreachable!(),
    };
    let msl = match msl_result {
        Ok(msl) => msl,
        Err(e) => {
            eprintln!("[quanta] MSL shader emitter error: {}", e);
            std::process::exit(1);
        }
    };
    match metallib::compile_msl_to_metallib(&msl) {
        Ok(bytes) => output.metallib = bytes,
        Err(e) => {
            eprintln!("[quanta] metallib error: {}", e);
            std::process::exit(1);
        }
    }

    // Emit WGSL — soft failure: WebGPU is not a supported render
    // target yet, so a WGSL gap must not block Metal/Vulkan shaders.
    let wgsl_result = match stage {
        "vertex" => emit_wgsl::emit_vertex_shader(&shader),
        "fragment" => emit_wgsl::emit_fragment_shader(&shader),
        _ => unreachable!(),
    };
    match wgsl_result {
        Ok(wgsl) => output.wgsl = Some(wgsl),
        Err(e) => eprintln!("[quanta] WGSL shader emitter warning: {}", e),
    }

    let out_bytes = quanta_ir::serialize_shader_output(&output);
    std::io::Write::write_all(&mut std::io::stdout(), &out_bytes).unwrap();
}

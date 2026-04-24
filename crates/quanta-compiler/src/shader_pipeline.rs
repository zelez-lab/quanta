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
    match spirv_result {
        Ok(spirv) => output.spirv = Some(spirv),
        Err(e) => eprintln!("[quanta] SPIR-V shader emitter error: {}", e),
    }

    // Emit MSL and compile to metallib
    let msl_result = match stage {
        "vertex" => emit_msl::emit_vertex_shader(&shader),
        "fragment" => emit_msl::emit_fragment_shader(&shader),
        _ => unreachable!(),
    };
    if let Ok(msl) = msl_result {
        match metallib::compile_msl_to_metallib(&msl) {
            Ok(bytes) => output.metallib = bytes,
            Err(e) => eprintln!("[quanta] metallib error: {}", e),
        }
    }

    // Emit WGSL
    let wgsl_result = match stage {
        "vertex" => emit_wgsl::emit_vertex_shader(&shader),
        "fragment" => emit_wgsl::emit_fragment_shader(&shader),
        _ => unreachable!(),
    };
    if let Ok(wgsl) = wgsl_result {
        output.wgsl = Some(wgsl);
    }

    let out_bytes = quanta_ir::serialize_shader_output(&output);
    std::io::Write::write_all(&mut std::io::stdout(), &out_bytes).unwrap();
}

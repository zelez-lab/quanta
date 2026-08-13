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
        metallib_ios: None,
        metallib_ios_sim: None,
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
    match metallib::compile_msl_to_metallib_variants(&msl) {
        Ok(variants) => attach_metal_artifacts(&mut output, variants, msl),
        Err(e) => {
            eprintln!("[quanta] metallib error: {}", e);
            std::process::exit(1);
        }
    }

    // Emit WGSL — soft failure by design: the WGSL emitter is at construct
    // parity with MSL/SPIR-V, so this warning should never fire for the
    // documented grammar; if a gap ever reappears it must not block the
    // Metal/Vulkan binaries of the same shader.
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

/// Attach the Metal artifacts to the output. A host WITH the Apple
/// toolchain ships real metallibs; a host WITHOUT one (ToolAbsent →
/// every variant `None`) ships the MSL SOURCE as the Metal artifact
/// instead — the Metal driver's binary loader sniffs the `MTLB` magic
/// and compiles non-MTLB bytes as source at pipeline creation, so a
/// bundle built on any host runs on Apple targets (first pipeline
/// creation pays a one-time driver compile; `apple_metallib`'s
/// iOS→macOS field fallback makes the source serve every Apple
/// platform). A toolchain FAILURE on an Apple host never reaches here
/// — that path stays `Err`/Fatal, not a silent source fallback.
fn attach_metal_artifacts(
    output: &mut quanta_ir::ShaderOutput,
    variants: metallib::MetallibVariants,
    msl: String,
) {
    output.metallib = variants.macos.or_else(|| Some(msl.into_bytes()));
    output.metallib_ios = variants.ios;
    output.metallib_ios_sim = variants.ios_sim;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_output() -> quanta_ir::ShaderOutput {
        quanta_ir::ShaderOutput {
            spirv: None,
            metallib: None,
            metallib_ios: None,
            metallib_ios_sim: None,
            wgsl: None,
        }
    }

    #[test]
    fn toolchain_absent_ships_msl_source_as_the_metal_artifact() {
        let mut out = empty_output();
        let variants = metallib::MetallibVariants {
            macos: None,
            ios: None,
            ios_sim: None,
        };
        attach_metal_artifacts(&mut out, variants, "using namespace metal;".into());
        // The source rides the metallib field; the driver sniffs MTLB
        // magic and compiles non-MTLB bytes as source.
        assert_eq!(
            out.metallib.as_deref(),
            Some(b"using namespace metal;" as &[u8])
        );
        assert!(out.metallib_ios.is_none() && out.metallib_ios_sim.is_none());
    }

    #[test]
    fn real_metallib_wins_over_the_source_fallback() {
        let mut out = empty_output();
        let variants = metallib::MetallibVariants {
            macos: Some(b"MTLB\x01".to_vec()),
            ios: None,
            ios_sim: None,
        };
        attach_metal_artifacts(&mut out, variants, "using namespace metal;".into());
        assert_eq!(out.metallib.as_deref(), Some(b"MTLB\x01" as &[u8]));
    }
}

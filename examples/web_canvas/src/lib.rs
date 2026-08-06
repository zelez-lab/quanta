//! Canvas-presentation smoke test for step 096.
//!
//! Runs the **public uniform frame loop** — the same shape as
//! `examples/native_window.rs` on macOS — against a browser canvas:
//! `init_async` → `create_surface(SurfaceTarget::Canvas { .. })` →
//! `acquire` → `gpu.render(frame.texture())` → `present`. Four full
//! loop iterations: a clear-only frame, a clear + centered triangle,
//! the same scene at 4x MSAA through `.msaa(4)`/`.msaa_resolve()`
//! with the canvas frame as the resolve destination, and finally the
//! same triangle again through a pipeline built from **runtime-emitted
//! render-DSL WGSL with NAMED entry points** — the seam no
//! compile-time check covers (naga validates a module in isolation;
//! only a real `CreateRenderPipeline` checks the descriptor's
//! `entryPoint` against the module — dija's R8). The Playwright
//! harness asserts what the compositor actually shows from an element
//! screenshot, so the last (DSL-emitted) frame is the asserted one.
//!
//! ## Build
//!
//! ```sh
//! quanta build web web_canvas
//! ```

#![cfg(target_arch = "wasm32")]

use quanta::webgpu::spawn_local;
use quanta::{Color, PipelineDesc, RenderGpu as _, ShaderSource, SurfaceConfig, SurfaceTarget};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn quanta_complete_bytes(task: u32, ptr: *const u8, len: usize);
    fn quanta_complete_err(task: u32, ptr: *const u8, len: usize);
}

const TRIANGLE_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Centered triangle — the red clear stays visible in the corners.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>( 0.0, -0.8),
        vec2<f32>(-0.8,  0.8),
        vec2<f32>( 0.8,  0.8),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.2, 0.4, 0.9, 1.0);
}
"#;

async fn run(canvas: u32) -> Result<Vec<u8>, String> {
    if !quanta::webgpu::available() {
        return Err("navigator.gpu is undefined".into());
    }

    let gpu = quanta::webgpu::init_async()
        .await
        .map_err(|e| format!("init_async: {e:?}"))?;
    if !gpu.supports_surface_present() {
        return Err("supports_surface_present() is false".into());
    }

    let mut surface = gpu
        .create_surface(
            &SurfaceTarget::Canvas { canvas },
            &SurfaceConfig::new(128, 128),
        )
        .map_err(|e| format!("create_surface: {e:?}"))?;

    // The portable pattern: type the pipeline from the NEGOTIATED
    // format, never from the config preference.
    let format = surface.format().map_err(|e| format!("format: {e:?}"))?;
    let pipeline = gpu
        .pipeline(
            &PipelineDesc::new(ShaderSource::Combined(TRIANGLE_WGSL.as_bytes()))
                .with_entries("vertex_main", "fragment_main")
                .with_color_formats(vec![format]),
        )
        .map_err(|e| format!("pipeline: {e:?}"))?;

    // Frame 1 — clear only. Proves a present retires the frame so the
    // next acquire is allowed.
    let frame = surface.acquire().map_err(|e| format!("acquire 1: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 1: {e:?}"))?
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pulse()
        .map_err(|e| format!("pulse 1: {e:?}"))?;
    frame.present().map_err(|e| format!("present 1: {e:?}"))?;

    // Frame 2 — clear + triangle. Proves the plain single-sample pass.
    let frame = surface.acquire().map_err(|e| format!("acquire 2: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 2: {e:?}"))?
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pipeline(&pipeline)
        .draw(3)
        .pulse()
        .map_err(|e| format!("pulse 2: {e:?}"))?;
    frame.present().map_err(|e| format!("present 2: {e:?}"))?;

    // Frame 3 — the same scene at 4x MSAA through the builder-managed
    // pooled intermediate, resolved into the acquired canvas frame at
    // pass end (WebGPU's native resolveTarget). This is the dija main
    // pass shape; the compositor shows this frame.
    let msaa_pipeline = gpu
        .pipeline(
            &PipelineDesc::new(ShaderSource::Combined(TRIANGLE_WGSL.as_bytes()))
                .with_entries("vertex_main", "fragment_main")
                .with_color_formats(vec![format])
                .with_sample_count(4),
        )
        .map_err(|e| format!("msaa pipeline: {e:?}"))?;
    let frame = surface.acquire().map_err(|e| format!("acquire 3: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 3: {e:?}"))?
        .msaa(4)
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pipeline(&msaa_pipeline)
        .draw(3)
        .msaa_resolve()
        .pulse()
        .map_err(|e| format!("pulse 3 (msaa resolve): {e:?}"))?;
    frame.present().map_err(|e| format!("present 3: {e:?}"))?;

    // Frame 4 — the same triangle through a pipeline built from
    // RUNTIME-EMITTED render-DSL WGSL with NAMED entry points. This is
    // the R8 seam: `ShaderBinary.entry_point` carries the shader's real
    // fn name into `GPURenderPipelineDescriptor.entryPoint`, and only a
    // live `CreateRenderPipeline` validates that name against the
    // module — naga and every compile-time check pass a `fn main`
    // module that fails here. The pair also exercises the R7 shapes in
    // a REAL pipeline: `vertex_id()` quad synthesis with NO vertex
    // buffer (and no input struct), plus a flat u32 varying branched on
    // with `== 1u32` in the fragment. This is the frame the compositor
    // shows and the harness screenshots.
    let (vs_wgsl, fs_wgsl) = emit_dsl_pair().map_err(|e| format!("dsl emit: {e}"))?;
    let dsl_pipeline = gpu
        .pipeline(
            &PipelineDesc::new(ShaderSource::Stages {
                vertex: vs_wgsl.as_bytes(),
                fragment: fs_wgsl.as_bytes(),
            })
            .with_entries("canvas_dsl_vertex", "canvas_dsl_fragment")
            .with_color_formats(vec![format]),
        )
        .map_err(|e| format!("dsl pipeline (named entries): {e:?}"))?;
    let frame = surface.acquire().map_err(|e| format!("acquire 4: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 4: {e:?}"))?
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pipeline(&dsl_pipeline)
        .draw(3)
        .pulse()
        .map_err(|e| format!("pulse 4 (dsl): {e:?}"))?;
    frame.present().map_err(|e| format!("present 4: {e:?}"))?;

    // Hand the negotiated format back so the page can report it.
    Ok(format!("{format:?}").into_bytes())
}

/// Emit the frame-4 shader pair through the REAL render-DSL WGSL emitter
/// at runtime (the same emitter the compiler embeds in `ShaderBinary.wgsl`
/// at build time). Bodies are in the token-spaced wire form the macro
/// ships. The vertex synthesizes the same centered triangle as
/// `TRIANGLE_WGSL` from `vertex_id()` — no vertex buffer, no input
/// struct — and forwards a flat u32 varying the fragment branches on.
fn emit_dsl_pair() -> Result<(String, String), String> {
    use quanta_ir::{ShaderDef, ShaderStage, ShaderType, ShaderVaryings, VaryingField};

    let varyings = |binding: Option<&str>| ShaderVaryings {
        struct_name: "DslVary".to_string(),
        position: "clip".to_string(),
        fields: vec![VaryingField {
            name: "shade".to_string(),
            ty: ShaderType::U32,
        }],
        binding: binding.map(str::to_string),
    };

    let vertex = ShaderDef {
        name: "canvas_dsl_vertex".to_string(),
        stage: ShaderStage::Vertex,
        params: vec![],
        return_type: ShaderType::Vec4,
        body_source: "{ let vid = vertex_id ( ) ; \
                       let mut x = 0.0 ; let mut y = - 0.8 ; \
                       if vid == 1u32 { x = - 0.8 ; y = 0.8 ; } else { } \
                       if vid == 2u32 { x = 0.8 ; y = 0.8 ; } else { } \
                       DslVary { clip : Vec4 :: new ( x , y , 0.0 , 1.0 ) , shade : 1u32 } }"
            .to_string(),
        varyings: Some(varyings(None)),
    };
    let fragment = ShaderDef {
        name: "canvas_dsl_fragment".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![],
        return_type: ShaderType::Vec4,
        body_source: "{ let c = if s . shade == 1u32 { 0.9 } else { 0.1 } ; \
                       Vec4 :: new ( 0.2 , 0.4 , c , 1.0 ) }"
            .to_string(),
        varyings: Some(varyings(Some("s"))),
    };

    Ok((
        quanta_ir::emit_wgsl::emit_vertex_shader(&vertex)?,
        quanta_ir::emit_wgsl::emit_fragment_shader(&fragment)?,
    ))
}

/// Smoke-test entry. The page registers its canvas with the glue and
/// calls `wasm.web_canvas_run(task, canvasHandle)`; the result string
/// (the negotiated format) is delivered via `quanta_complete_bytes`.
#[unsafe(no_mangle)]
pub extern "C" fn web_canvas_run(task: u32, canvas: u32) {
    spawn_local(async move {
        match run(canvas).await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}

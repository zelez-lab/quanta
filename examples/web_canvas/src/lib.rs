//! Canvas-presentation smoke test for step 096.
//!
//! Runs the **public uniform frame loop** — the same shape as
//! `examples/native_window.rs` on macOS — against a browser canvas:
//! `init_async` → `create_surface(SurfaceTarget::Canvas { .. })` →
//! `acquire` → `gpu.render(frame.texture())` → `present`. Five full
//! loop iterations: a clear-only frame, a clear + centered triangle,
//! the same scene at 4x MSAA through `.msaa(4)`/`.msaa_resolve()`
//! with the canvas frame as the resolve destination, the same
//! triangle through a pipeline built from **runtime-emitted
//! render-DSL WGSL with NAMED entry points** — the seam no
//! compile-time check covers (naga validates a module in isolation;
//! only a real `CreateRenderPipeline` checks the descriptor's
//! `entryPoint` against the module — dija's R8) — and finally the
//! full render-lane buffer plumbing (dija's R9): a `Field`-fed vertex
//! buffer, a uniform `Field` in the SAME bind group as a sampled
//! texture, and the driver's default-sampler fallback. The Playwright
//! harness asserts what the compositor actually shows from an element
//! screenshot, so the last (mixed-bind-group) frame is the asserted
//! one.
//!
//! ## Build
//!
//! ```sh
//! quanta build web web_canvas
//! ```

#![cfg(target_arch = "wasm32")]

use quanta::webgpu::{complete_bytes, complete_err, spawn_local};
use quanta::{
    AttributeFormat, Color, Format, PipelineDesc, RenderGpu as _, ShaderSource, StepMode,
    SurfaceConfig, SurfaceTarget, TextureDesc, TextureUsage, VertexAttribute, VertexLayout,
};

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

    // Frame 5 — the same triangle through the FULL render-lane buffer
    // plumbing (dija's R9 shape): a `Field`-fed VERTEX buffer (WebGPU
    // enforces buffer usage at creation — the natives don't — so this
    // frame is the seam that proves a plain `gpu.field()` can feed a
    // vertex fetch), a plain-`Field` UNIFORM in the SAME bind group as a
    // sampled texture (the mixed uniform+texture group is exactly the
    // shape whose buffer entry the driver used to drop), and NO explicit
    // sampler — the driver's linear-clamp default-sampler fallback fills
    // the pair's `@binding(8+2s+1)` slot, mirroring the Vulkan arm. The
    // shader pair is runtime-emitted DSL again; a 1x1 texture of the
    // triangle blue times a white tint keeps the pixels — and the
    // harness's screenshot asserts — identical to every earlier frame.
    let (vs5, fs5) = emit_dsl_textured_pair().map_err(|e| format!("dsl5 emit: {e}"))?;
    let verts = gpu
        .field::<[f32; 2]>(3)
        .map_err(|e| format!("vertex field: {e:?}"))?;
    verts
        .write(&[[0.0, -0.8], [-0.8, 0.8], [0.8, 0.8]])
        .map_err(|e| format!("vertex write: {e:?}"))?;
    let tint = gpu
        .field::<[f32; 4]>(1)
        .map_err(|e| format!("tint field: {e:?}"))?;
    tint.write(&[[1.0, 1.0, 1.0, 1.0]])
        .map_err(|e| format!("tint write: {e:?}"))?;
    let blue = gpu
        .create_texture(
            &TextureDesc::new(1, 1, Format::RGBA8).with_usage(TextureUsage::SHADER_READ),
        )
        .map_err(|e| format!("blue texture: {e:?}"))?;
    blue.write(&[51, 102, 229, 255])
        .map_err(|e| format!("blue write: {e:?}"))?;
    let layouts = [VertexLayout {
        stride: 8,
        step: StepMode::Vertex,
        attributes: vec![VertexAttribute {
            location: 0,
            offset: 0,
            format: AttributeFormat::Float2,
        }],
    }];
    let mixed_pipeline = gpu
        .pipeline(
            &PipelineDesc::new(ShaderSource::Stages {
                vertex: vs5.as_bytes(),
                fragment: fs5.as_bytes(),
            })
            .with_entries("canvas_mixed_vertex", "canvas_mixed_fragment")
            .with_vertex_layouts(&layouts)
            .with_color_formats(vec![format]),
        )
        .map_err(|e| format!("mixed pipeline: {e:?}"))?;
    let frame = surface.acquire().map_err(|e| format!("acquire 5: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 5: {e:?}"))?
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pipeline(&mixed_pipeline)
        .vertices(0, &verts)
        .uniform(0, &tint)
        .texture(0, &blue)
        .draw(3)
        .pulse()
        .map_err(|e| format!("pulse 5 (mixed bind group): {e:?}"))?;
    frame.present().map_err(|e| format!("present 5: {e:?}"))?;

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

/// Emit the frame-5 shader pair: a vertex fed from a real vertex buffer
/// (`@location(0) pos: vec2<f32>` — the `Field`-backed fetch), and a
/// fragment whose bind group MIXES a `&Vec4` uniform (binding 0) with a
/// sampled texture (bindings 8/9) — the R9 shape. No explicit sampler is
/// ever set; the driver's default-sampler fallback supplies binding 9.
fn emit_dsl_textured_pair() -> Result<(String, String), String> {
    use quanta_ir::{
        ShaderDef, ShaderParam, ShaderStage, ShaderType, ShaderVaryings, VaryingField,
    };

    let varyings = |binding: Option<&str>| ShaderVaryings {
        struct_name: "MixedVary".to_string(),
        position: "clip".to_string(),
        fields: vec![VaryingField {
            name: "uv".to_string(),
            ty: ShaderType::Vec2,
        }],
        binding: binding.map(str::to_string),
    };

    let vertex = ShaderDef {
        name: "canvas_mixed_vertex".to_string(),
        stage: ShaderStage::Vertex,
        params: vec![ShaderParam {
            name: "pos".to_string(),
            ty: ShaderType::Vec2,
            is_uniform: false,
            is_slice: false,
        }],
        return_type: ShaderType::Vec4,
        body_source: "{ MixedVary { clip : Vec4 :: new ( pos . x , pos . y , 0.0 , 1.0 ) , \
                       uv : Vec2 :: new ( ( pos . x + 1.0 ) * 0.5 , ( pos . y + 1.0 ) * 0.5 ) } }"
            .to_string(),
        varyings: Some(varyings(None)),
    };
    let fragment = ShaderDef {
        name: "canvas_mixed_fragment".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![ShaderParam {
            name: "tint".to_string(),
            ty: ShaderType::Vec4,
            is_uniform: true,
            is_slice: false,
        }],
        return_type: ShaderType::Vec4,
        body_source: "{ sample(0 , s . uv ) * * tint }".to_string(),
        varyings: Some(varyings(Some("s"))),
    };

    Ok((
        quanta_ir::emit_wgsl::emit_vertex_shader(&vertex)?,
        quanta_ir::emit_wgsl::emit_fragment_shader(&fragment)?,
    ))
}

/// Smoke-test entry. The page registers its canvas with the glue and
/// calls `wasm.web_canvas_run(task, canvasHandle)`; the result string
/// (the negotiated format) is delivered via `complete_bytes`.
#[unsafe(no_mangle)]
pub extern "C" fn web_canvas_run(task: u32, canvas: u32) {
    spawn_local(async move {
        match run(canvas).await {
            Ok(bytes) => complete_bytes(task, &bytes),
            Err(msg) => complete_err(task, &msg),
        }
    });
}

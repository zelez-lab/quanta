//! Render-path smoke test for step 050 + B⁰.
//!
//! Creates a 16×16 offscreen render target, runs a clear+draw render
//! pass, and reads the texels back. Validates `pipeline_create`,
//! `render_begin/end`, and `texture_read_async` end-to-end without
//! `wasm-bindgen`. Result is reported via `quanta_complete_bytes`
//! / `quanta_complete_err`.
//!
//! ## Build
//!
//! ```sh
//! quanta build web web_triangle
//! ```

#![cfg(target_arch = "wasm32")]

use quanta::webgpu::spawn_local;
use quanta::{
    Color, Format, GpuDevice as _, PipelineDesc, RenderPass, SamplerDesc, ShaderSource,
    TextureDesc, TextureUsage,
};

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
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
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

async fn run() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    let target = dev
        .texture_create(
            &TextureDesc::new(16, 16, Format::RGBA8)
                .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
        )
        .map_err(|e| format!("texture_create: {:?}", e))?;

    let _sampler = dev
        .sampler_create(&SamplerDesc::default())
        .map_err(|e| format!("sampler_create: {:?}", e))?;

    let pipeline = dev
        .pipeline_create(
            &PipelineDesc::new(ShaderSource::Combined(TRIANGLE_WGSL.as_bytes()))
                .with_entries("vertex_main", "fragment_main")
                .with_color_formats(vec![Format::RGBA8]),
        )
        .map_err(|e| format!("pipeline_create: {:?}", e))?;

    let mut pass: RenderPass = dev
        .render_begin(&target)
        .map_err(|e| format!("render_begin: {:?}", e))?;
    pass.clear(Color::rgba(1.0, 0.0, 0.0, 1.0));
    pass.set_pipeline(&pipeline);
    pass.draw(3);
    let _pulse = dev
        .render_end(pass)
        .map_err(|e| format!("render_end: {:?}", e))?;

    dev.texture_read_async(&target)
        .await
        .map_err(|e| format!("texture_read_async: {:?}", e))
}

/// Smoke-test entry. JS-side harness calls
/// `wasm.web_triangle_run(task)`; result delivered via
/// `quanta_complete_bytes` / `quanta_complete_err`.
#[unsafe(no_mangle)]
pub extern "C" fn web_triangle_run(task: u32) {
    spawn_local(async move {
        match run().await {
            Ok(bytes) => unsafe {
                quanta_complete_bytes(task, bytes.as_ptr(), bytes.len());
            },
            Err(msg) => unsafe {
                quanta_complete_err(task, msg.as_ptr(), msg.len());
            },
        }
    });
}

//! Render-path smoke test for step 050.
//!
//! Builds the simplest possible end-to-end Quanta-on-WebGPU render demo:
//! create a tiny offscreen render target, run a clear-only render pass,
//! and read the texels back. Validates `pipeline_create`,
//! `render_begin/end`, and `texture_read_async` end-to-end.
//!
//! Run from a real browser tab — there's no headless harness, by design.
//!
//! ## Build
//!
//! ```sh
//! cargo build --target wasm32-unknown-unknown --release -p web-triangle
//! wasm-bindgen --target web --out-dir examples/web_triangle/pkg \
//!     target/wasm32-unknown-unknown/release/web_triangle.wasm
//! ```
//!
//! Serve `examples/web_triangle/index.html` over HTTPS (or
//! http://localhost) and open in a WebGPU-capable browser.

#![cfg(target_arch = "wasm32")]

use quanta::{
    AddressMode, Color, Filter, Format, GpuDevice as _, PipelineDesc, RenderPass, SamplerDesc,
    TextureDesc, TextureUsage,
};
use wasm_bindgen::prelude::*;

/// Trivial WGSL pipeline: full-screen triangle that outputs a constant color.
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

#[wasm_bindgen]
pub async fn run_triangle() -> Result<Vec<u8>, JsValue> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| JsValue::from_str(&format!("new_async: {:?}", e)))?;

    // 1. Render target.
    let target = dev
        .texture_create(&TextureDesc {
            width: 16,
            height: 16,
            format: Format::RGBA8,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
        .map_err(|e| JsValue::from_str(&format!("texture_create: {:?}", e)))?;

    // 2. Sampler — created so the smoke test exercises sampler_create end-to-end
    //    even though this kernel doesn't sample anything.
    let _sampler = dev
        .sampler_create(&SamplerDesc {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            mip_filter: Filter::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            max_anisotropy: 1,
            compare: None,
        })
        .map_err(|e| JsValue::from_str(&format!("sampler_create: {:?}", e)))?;

    // 3. Render pipeline.
    let pipeline = dev
        .pipeline_create(&PipelineDesc {
            source: Some(TRIANGLE_WGSL.as_bytes()),
            vertex_entry: "vertex_main",
            fragment_entry: "fragment_main",
            color_formats: alloc::vec![Format::RGBA8],
            ..PipelineDesc::default()
        })
        .map_err(|e| JsValue::from_str(&format!("pipeline_create: {:?}", e)))?;

    // 4. Render pass: clear to red, then draw the triangle (which paints blue).
    let mut pass: RenderPass = dev
        .render_begin(&target)
        .map_err(|e| JsValue::from_str(&format!("render_begin: {:?}", e)))?;
    pass.clear(Color::rgba(1.0, 0.0, 0.0, 1.0));
    pass.set_pipeline(&pipeline);
    pass.draw(3);
    let _pulse = dev
        .render_end(pass)
        .map_err(|e| JsValue::from_str(&format!("render_end: {:?}", e)))?;

    // 5. Read back: every pixel should be approximately the triangle color.
    dev.texture_read_async(&target)
        .await
        .map_err(|e| JsValue::from_str(&format!("texture_read_async: {:?}", e)))
}

extern crate alloc;

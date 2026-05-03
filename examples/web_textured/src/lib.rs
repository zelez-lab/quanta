//! Render-path smoke test for **step C** (2026-04-28) wiring.
//!
//! Exercises `RenderOp::SetTexture` + `RenderOp::SetSampler` end-to-
//! end on real WebGPU. The earlier `web_triangle` smoke test only
//! covers the clear+draw path with no texture binding; this one
//! puts the C-introduced wiring in the critical path:
//!
//! 1. Allocate a 1×1 source texture and write a known RGBA color
//!    `(255, 165, 64, 255)` (a saturated orange the eye can spot
//!    against the black clear).
//! 2. Allocate a 4×4 render target.
//! 3. Build a pipeline whose fragment shader samples the source
//!    texture at the fragment's normalized UV.
//! 4. Run a render pass that **calls `set_texture` + `set_sampler`**
//!    before drawing the full-screen triangle, then reads back the
//!    target.
//! 5. The browser harness validates every pixel matches the source
//!    color (within ±2 LSBs to absorb sRGB/IEEE rounding).
//!
//! ## Build
//!
//! ```sh
//! quanta build web web_textured
//! ```

#![cfg(target_arch = "wasm32")]

use quanta::webgpu::spawn_local;
use quanta::{
    AddressMode, Color, Filter, Format, GpuDevice as _, PipelineDesc, RenderPass, SamplerDesc,
    TextureDesc, TextureUsage,
};

#[link(wasm_import_module = "env")]
unsafe extern "C" {
    fn quanta_complete_bytes(task: u32, ptr: *const u8, len: usize);
    fn quanta_complete_err(task: u32, ptr: *const u8, len: usize);
}

const TEXTURED_WGSL: &str = r#"
struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Fullscreen-triangle trick: three vertices that cover the
    // viewport (and clip) without needing a vertex buffer.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var src_sampler: sampler;
@group(0) @binding(1) var src_texture: texture_2d<f32>;

@fragment
fn fragment_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    // 4×4 viewport: fc.xy ∈ [0.5, 3.5]. Normalize to UV ∈ [0, 1].
    // The source is 1×1 so every UV maps to the same texel — every
    // output pixel ends up the source color.
    let uv = fc.xy / 4.0;
    return textureSampleLevel(src_texture, src_sampler, uv, 0.0);
}
"#;

const SOURCE_PIXEL: [u8; 4] = [255, 165, 64, 255];

async fn run() -> Result<Vec<u8>, String> {
    let dev = quanta::webgpu::WebgpuDevice::new_async()
        .await
        .map_err(|e| format!("new_async: {:?}", e))?;

    // Source texture: 1×1, sampled in the fragment shader.
    let source = dev
        .texture_create(&TextureDesc {
            width: 1,
            height: 1,
            format: Format::RGBA8,
            usage: TextureUsage::SHADER_READ,
            ..TextureDesc::default()
        })
        .map_err(|e| format!("source texture_create: {:?}", e))?;
    dev.texture_write(&source, &SOURCE_PIXEL)
        .map_err(|e| format!("source texture_write: {:?}", e))?;

    // Render target: 4×4 RGBA8.
    let target = dev
        .texture_create(&TextureDesc {
            width: 4,
            height: 4,
            format: Format::RGBA8,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
        .map_err(|e| format!("target texture_create: {:?}", e))?;

    let pipeline = dev
        .pipeline_create(&PipelineDesc {
            source: Some(TEXTURED_WGSL.as_bytes()),
            vertex_entry: "vertex_main",
            fragment_entry: "fragment_main",
            color_formats: vec![Format::RGBA8],
            ..PipelineDesc::default()
        })
        .map_err(|e| format!("pipeline_create: {:?}", e))?;

    // Render pass that exercises SetSampler + SetTexture.
    let mut pass: RenderPass = dev
        .render_begin(&target)
        .map_err(|e| format!("render_begin: {:?}", e))?;
    pass.clear(Color::rgba(0.0, 0.0, 0.0, 1.0));
    pass.set_pipeline(&pipeline);
    pass.set_sampler(
        0,
        SamplerDesc {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            mip_filter: Filter::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            max_anisotropy: 1,
            compare: None,
        },
    );
    pass.set_texture(1, &source);
    pass.draw(3);
    let _pulse = dev
        .render_end(pass)
        .map_err(|e| format!("render_end: {:?}", e))?;

    dev.texture_read_async(&target)
        .await
        .map_err(|e| format!("texture_read_async: {:?}", e))
}

/// Smoke-test entry. JS-side harness calls
/// `wasm.web_textured_run(task)`; result delivered via
/// `quanta_complete_bytes` / `quanta_complete_err`.
#[unsafe(no_mangle)]
pub extern "C" fn web_textured_run(task: u32) {
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

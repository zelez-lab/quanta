//! Canvas-presentation smoke test for step 096.
//!
//! Runs the **public uniform frame loop** — the same shape as
//! `examples/native_window.rs` on macOS — against a browser canvas:
//! `init_async` → `create_surface(SurfaceTarget::Canvas { .. })` →
//! `acquire` → `gpu.render(frame.texture())` → `present`. Two full
//! loop iterations (a clear-only frame, then clear + centered
//! triangle) prove the acquire/present bookkeeping recycles. The page
//! validates what the compositor actually shows by drawing the WebGPU
//! canvas onto a 2d canvas and sampling pixels.
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

    // Frame 2 — clear + triangle. What the compositor shows.
    let frame = surface.acquire().map_err(|e| format!("acquire 2: {e:?}"))?;
    gpu.render(frame.texture())
        .map_err(|e| format!("render 2: {e:?}"))?
        .clear(Color::rgba(1.0, 0.0, 0.0, 1.0))
        .pipeline(&pipeline)
        .draw(3)
        .pulse()
        .map_err(|e| format!("pulse 2: {e:?}"))?;
    frame.present().map_err(|e| format!("present 2: {e:?}"))?;

    // Hand the negotiated format back so the page can report it.
    Ok(format!("{format:?}").into_bytes())
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

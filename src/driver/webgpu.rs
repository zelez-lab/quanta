//! WebGPU driver — browser-only, raw FFI scaffold.
//!
//! Step 050 of the roadmap. Lets Quanta kernels run inside a browser via
//! WebAssembly + the browser's WebGPU API. Native (non-wasm) targets continue
//! to use Metal / Vulkan / CPU backends.
//!
//! ## Status: scaffold only — no FFI yet
//!
//! This first cut delivers:
//!
//! - `webgpu` Cargo feature, gated to `target_arch = "wasm32"`.
//! - `src/driver/webgpu.rs` registered in `src/driver.rs`.
//! - `WebgpuDevice` implementing `GpuDevice` with `NotImplemented` stubs.
//! - Zero external wasm dependencies. `Cargo.toml` adds nothing for the
//!   `webgpu` feature beyond `std + jit`.
//!
//! ## Policy: no wrapper crates, ever
//!
//! Step 000 of the project rejects `ash`, `metal-rs`, `objc` for native
//! drivers. The same rule extends to wasm32: **no `web-sys`, no `wgpu`**.
//! Both are pre-generated typed binding libraries (`web-sys` is ~150K LOC
//! of generated WebIDL bindings; `wgpu` is a full GPU abstraction). Either
//! would put the entirety of a maintained external API surface inside
//! Quanta's TCB.
//!
//! When this scaffold gets filled in (after step 079 lands `emit_wgsl_jit`),
//! the FFI to WebGPU will be **hand-written `extern "C"` blocks via
//! `wasm-bindgen`** — the same pattern the project uses for Metal headers
//! and `vkSomething` loaders. Roughly 200-300 lines we own, covering only
//! the WebGPU surface Quanta actually uses (`requestAdapter`, `requestDevice`,
//! `createBuffer`, `createComputePipeline`, `dispatchWorkgroups`, `submit`,
//! `mapAsync`). `wasm-bindgen` itself is the wasm32 calling-convention
//! bridge (equivalent to libc), not a wrapper — it stays.
//!
//! ## What blocks the dispatch path
//!
//! The proc-macro JIT path serializes a `KernelDef` and the device's
//! `wave_jit` deserializes + emits the right shader format at runtime.
//! `quanta-ir` has `emit_spirv` (Vulkan) and `emit_msl` (Metal); a parallel
//! `emit_wgsl_jit` does not exist yet. Without it, `wave_jit` on this
//! driver cannot produce the WGSL string needed by
//! `GPUDevice.createShaderModule`. Tracked as step 079.
//!
//! ## Sync ↔ async impedance (resolved when bindings land)
//!
//! WebGPU's JS API is async for: `requestAdapter`, `requestDevice`,
//! `mapAsync` (buffer read-back), `onSubmittedWorkDone` (completion).
//! Synchronous for: buffer/encoder/pipeline create, `dispatchWorkgroups`,
//! `submit`, `writeBuffer`. The browser cannot block its event loop, so
//! `pulse_wait` and `field_read_bytes` (sync trait methods) cannot be
//! implemented as blocking sync calls. Public extension methods
//! (`pulse_wait_async`, `field_read_bytes_async`) will be added when the
//! real bindings land.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, Format, FormatCaps, Gpu, GpuDevice as QGpuDevice, Pipeline, Pulse,
    QuantaError, RenderPass, Texture, TextureDesc, Vendor, Wave,
};

/// WebGPU device — sits behind `target_arch = "wasm32"` and the `webgpu` feature.
pub struct WebgpuDevice {
    caps: Caps,
}

impl WebgpuDevice {
    fn err(msg: &'static str) -> QuantaError {
        QuantaError::invalid_param(msg)
    }
}

// ── Async init ──────────────────────────────────────────────────────────────

/// Initialize a WebGPU device. Async because the browser surfaces device
/// acquisition only via Promises. Call this once at app start.
///
/// ```ignore
/// let gpu = quanta::driver::webgpu::init_async().await?;
/// ```
///
/// Currently returns a stub device; real `requestAdapter` + `requestDevice`
/// integration ships once step 079 lands the JIT WGSL emitter and the
/// hand-rolled `wasm-bindgen` extern blocks are in place.
pub async fn init_async() -> Result<Gpu, QuantaError> {
    let caps = Caps {
        nuclei: 1,
        protons_per_nucleus: 1,
        quarks_per_proton: 1,
        memory_bytes: 0,
        max_quarks_per_dispatch: 65535,
        max_groups: [65535, 65535, 65535],
        vendor: Vendor::Software,
        name: String::from("WebGPU (browser, scaffold)"),
    };
    let dev: Box<dyn QGpuDevice> = Box::new(WebgpuDevice { caps });
    Ok(Gpu::new(Arc::from(dev)))
}

// ── GpuDevice impl (stubs) ──────────────────────────────────────────────────

impl QGpuDevice for WebgpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn field_alloc(&self, _size: usize, _usage: FieldUsage) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU buffer alloc pending step 079"))
    }
    fn field_free(&self, _handle: u64) {}
    fn field_write_bytes(&self, _handle: u64, _data: &[u8]) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU field_write pending step 079"))
    }
    fn field_read_bytes(&self, _handle: u64, _size: usize) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err(
            "field_read_bytes is async-only on WebGPU; pending field_read_async",
        ))
    }
    fn field_copy_bytes(&self, _dst: u64, _src: u64, _size: usize) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU field_copy pending step 079"))
    }

    fn texture_create(&self, _desc: &TextureDesc) -> Result<Texture, QuantaError> {
        Err(Self::err("WebGPU textures pending"))
    }
    fn texture_write(&self, _texture: &Texture, _data: &[u8]) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU textures pending"))
    }
    fn texture_read(&self, _texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err("WebGPU textures pending"))
    }
    fn sampler_create(
        &self,
        _desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        Err(Self::err("WebGPU samplers pending"))
    }
    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU mipmap gen pending"))
    }

    fn wave(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(Self::err(
            "WebGPU does not accept pre-compiled binaries; use the JIT path",
        ))
    }
    fn wave_jit(&self, _kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        Err(Self::err(
            "wave_jit pending: needs emit_wgsl_jit in quanta-ir (step 079)",
        ))
    }
    fn wave_dispatch(&self, _wave: &Wave, _groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        Err(Self::err("dispatch pending step 079"))
    }
    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(Self::err("WebGPU indirect dispatch pending"))
    }

    fn pipeline_create(&self, _desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        Err(Self::err("WebGPU render pending"))
    }
    fn render_begin(&self, _target: &Texture) -> Result<RenderPass, QuantaError> {
        Err(Self::err("WebGPU render pending"))
    }
    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        Err(Self::err("WebGPU render pending"))
    }

    fn pulse_wait(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        Err(Self::err(
            "pulse_wait is async-only on WebGPU; pending pulse_wait_async",
        ))
    }
    fn pulse_poll(&self, _pulse: &Pulse) -> bool {
        false
    }

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU mesh shaders not supported"))
    }
    fn build_acceleration_structure(&self, _geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU ray tracing not supported"))
    }
    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU ray tracing not supported"))
    }
    fn dispatch_rays(&self, _pipeline: u64, _w: u32, _h: u32) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU ray tracing not supported"))
    }
    fn destroy_acceleration_structure(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU ray tracing not supported"))
    }
    fn sparse_texture_create(&self, _desc: &TextureDesc) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU sparse textures not supported"))
    }
    fn sparse_map_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU sparse textures not supported"))
    }
    fn sparse_unmap_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
    ) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU sparse textures not supported"))
    }
    fn indirect_buffer_create(&self, _max_commands: u32) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU indirect command buffers pending"))
    }
    fn indirect_buffer_execute(&self, _handle: u64, _count: u32) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU indirect command buffers pending"))
    }
    fn indirect_buffer_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU indirect command buffers pending"))
    }
    fn bind_texture_array(&self, _textures: &[u64]) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU bindless pending"))
    }
    fn bind_buffer_array(&self, _buffers: &[u64]) -> Result<u64, QuantaError> {
        Err(Self::err("WebGPU bindless pending"))
    }

    fn format_caps(&self, _format: Format) -> FormatCaps {
        FormatCaps {
            filterable: true,
            renderable: true,
            storage: true,
            blendable: true,
            msaa: false,
            depth: false,
        }
    }
}

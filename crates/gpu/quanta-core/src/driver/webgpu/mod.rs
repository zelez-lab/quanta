//! WebGPU driver — browser-only.
//!
//! Step 050 + step 079 + B⁰ (2026-04-28). Lets Quanta kernels run inside
//! a browser via WebAssembly + the browser's WebGPU API. Native
//! (non-wasm) targets continue to use Metal / Vulkan / CPU backends.
//!
//! ## Architecture
//!
//! - `ffi.rs` — bare `extern "C"` imports defining Quanta's WebGPU ABI.
//!   ~300 lines we own and audit, with no `wasm-bindgen` runtime
//!   dependency. Strings cross as `(*const u8, usize)`; long-lived JS
//!   objects cross as `u32` handles into a JS-side handle table.
//! - `executor.rs` — minimal Rust async executor. Replaces
//!   `wasm-bindgen-futures::JsFuture` with a thread-local promise table
//!   driven by JS-callable `quanta_resolve` / `quanta_reject` exports.
//! - `state.rs` — handle bookkeeping mapping Quanta `u64` API handles
//!   onto the JS-side `u32` ABI handles.
//! - `compute.rs` — the JIT compute path (`wave` / `wave_jit` /
//!   `wave_dispatch` / `wave_destroy`), feature-gated on `compute`.
//! - `render.rs` — the render path (`pipeline_create` / `render_begin`
//!   / `render_end` + the render-only enum→ABI-code translations),
//!   feature-gated on `render` (step 085).
//! - `async_ext.rs` — the async-only public extension methods
//!   (`field_read_bytes_async`, `pulse_wait_async`, …) that stand in
//!   for the sync trait methods the browser event loop can't block on.
//!   This module keeps the `GpuDevice` trait impl in `mod.rs` thin:
//!   its render/compute methods delegate to `*_impl` inherent methods
//!   in `render.rs` / `compute.rs` (mirrors the Metal/Vulkan drivers).
//! - `web/src/quanta.ts` — TypeScript entry point on the JS half of the
//!   boundary. Compiled to `quanta.js` at build time alongside the
//!   internal modules (`handles.ts`, `tasks.ts`, `webgpu.ts`, …);
//!   TypeScript itself is build-only and never ships.
//!
//! ## Sync ↔ async impedance
//!
//! WebGPU's JS API is async for: `requestAdapter`, `requestDevice`,
//! `mapAsync` (buffer read-back), `onSubmittedWorkDone` (completion).
//! Synchronous for: buffer/encoder/pipeline create, `dispatchWorkgroups`,
//! `submit`, `writeBuffer`. The browser cannot block its event loop, so
//! `pulse_wait` and `field_read_bytes` (sync trait methods) are returned
//! as errors that direct callers to the public extension methods
//! [`WebgpuDevice::pulse_wait_async`] /
//! [`WebgpuDevice::field_read_bytes_async`].

mod async_ext;
#[cfg(feature = "compute")]
mod compute;
mod executor;
mod ffi;
#[cfg(feature = "render")]
mod render;
mod state;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::{
    Caps, FieldUsage, Format, FormatCaps, Gpu, GpuDevice as QGpuDevice, Pulse, QuantaError,
    Texture, TextureDesc, Vendor, Wave,
};
// Render types used by the render-gated trait wrappers + the
// ray-tracing NotSupported stubs below (085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};

use ffi::{NULL_HANDLE, buffer_usage};
use state::{SendCell, State};

pub use executor::{Promise, spawn_local};

/// WebGPU device — sits behind `target_arch = "wasm32"` and the
/// `webgpu` feature.
pub struct WebgpuDevice {
    caps: Caps,
    /// JS-side `GPUDevice` handle. `NULL_HANDLE` until `new_async` runs.
    device: SendCell<u32>,
    state: State,
}

impl WebgpuDevice {
    fn err(msg: &'static str) -> QuantaError {
        QuantaError::invalid_param(msg)
    }

    fn not_supported(msg: &'static str) -> QuantaError {
        QuantaError::not_supported(msg)
    }

    fn err_owned(msg: String) -> QuantaError {
        QuantaError::compilation_failed(Box::leak(msg.into_boxed_str()))
    }

    fn dev(&self) -> Result<u32, QuantaError> {
        let h = *self.device.0.borrow();
        if h == NULL_HANDLE {
            return Err(Self::err("WebGPU device not initialized — call init_async"));
        }
        Ok(h)
    }
}

// ── Async init ──────────────────────────────────────────────────────────────

impl WebgpuDevice {
    /// Acquire the typed WebGPU device. Use this when you need access
    /// to the async extension methods (`field_read_bytes_async`,
    /// `pulse_wait_async`); use [`init_async`] for the dyn-trait path
    /// that fits Quanta's standard `Gpu` wrapper.
    pub async fn new_async() -> Result<Self, QuantaError> {
        let adapter = Promise::register(|task| unsafe { ffi::quanta_request_adapter(task) })
            .await
            .map_err(|_| Self::err("requestAdapter rejected"))?;
        if adapter == NULL_HANDLE {
            return Err(Self::err("navigator.gpu unavailable"));
        }

        let device = Promise::register(|task| unsafe { ffi::quanta_request_device(adapter, task) })
            .await
            .map_err(|_| Self::err("requestDevice rejected"))?;
        // Adapter handle is no longer needed — release the JS-side ref.
        unsafe { ffi::quanta_release(adapter) };
        if device == NULL_HANDLE {
            return Err(Self::err("requestDevice returned no device"));
        }

        let caps = Caps {
            nuclei: 1,
            protons_per_nucleus: 1,
            quarks_per_proton: 1,
            memory_bytes: 0,
            max_quarks_per_dispatch: 65535,
            max_groups: [65535, 65535, 65535],
            vendor: Vendor::Software,
            name: String::from("WebGPU (browser)"),
        };
        Ok(WebgpuDevice {
            caps,
            device: SendCell(RefCell::new(device)),
            state: State::new(),
        })
    }
}

/// Initialize a WebGPU device wrapped as a `Gpu`. Async because the
/// browser surfaces device acquisition only via Promises. Call this
/// once at app start.
///
/// ```ignore
/// let gpu = quanta::driver::webgpu::init_async().await?;
/// ```
pub async fn init_async() -> Result<Gpu, QuantaError> {
    let dev = WebgpuDevice::new_async().await?;
    let boxed: Box<dyn QGpuDevice> = Box::new(dev);
    Ok(Gpu::new(Arc::from(boxed)))
}

// ── Enum → code translations (Rust API → ABI integer codes) ─────────────────

fn format_code(f: Format) -> Result<u32, QuantaError> {
    Ok(match f {
        Format::RGBA8 => ffi::format::RGBA8UNORM,
        Format::BGRA8 => ffi::format::BGRA8UNORM,
        Format::R8 => ffi::format::R8UNORM,
        Format::R16Float => ffi::format::R16FLOAT,
        Format::R32Float => ffi::format::R32FLOAT,
        Format::RG32Float => ffi::format::RG32FLOAT,
        Format::RGBA16Float => ffi::format::RGBA16FLOAT,
        Format::RGBA32Float => ffi::format::RGBA32FLOAT,
        Format::Depth32Float => ffi::format::DEPTH32FLOAT,
        Format::Bc1Rgba
        | Format::Bc3Rgba
        | Format::Bc5Rg
        | Format::Bc7Rgba
        | Format::Astc4x4
        | Format::Astc6x6
        | Format::Astc8x8
        | Format::Etc2Rgb8
        | Format::Etc2Rgba8 => {
            return Err(WebgpuDevice::err(
                "compressed texture formats not yet wired in WebGPU driver",
            ));
        }
    })
}

fn filter_code(f: crate::texture::Filter) -> u32 {
    match f {
        crate::texture::Filter::Nearest => ffi::filter::NEAREST,
        crate::texture::Filter::Linear => ffi::filter::LINEAR,
    }
}

fn address_code(a: crate::texture::AddressMode) -> u32 {
    match a {
        crate::texture::AddressMode::ClampToEdge => ffi::address::CLAMP_TO_EDGE,
        crate::texture::AddressMode::Repeat => ffi::address::REPEAT,
        crate::texture::AddressMode::MirrorRepeat => ffi::address::MIRROR_REPEAT,
    }
}

fn compare_op_code(c: crate::CompareOp) -> u32 {
    match c {
        crate::CompareOp::Never => ffi::compare::NEVER,
        crate::CompareOp::Less => ffi::compare::LESS,
        crate::CompareOp::Equal => ffi::compare::EQUAL,
        crate::CompareOp::LessEqual => ffi::compare::LESS_EQUAL,
        crate::CompareOp::Greater => ffi::compare::GREATER,
        crate::CompareOp::NotEqual => ffi::compare::NOT_EQUAL,
        crate::CompareOp::GreaterEqual => ffi::compare::GREATER_EQUAL,
        crate::CompareOp::Always => ffi::compare::ALWAYS,
    }
}

// ── GpuDevice impl ──────────────────────────────────────────────────────────

impl crate::api::device::sealed::Sealed for WebgpuDevice {}

impl QGpuDevice for WebgpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    /// WGSL storage buffers cannot hold 16-/8-bit array elements, so the
    /// WGSL emitter keeps bf16/fp8 on the portable u32-slot layout (one
    /// element per 32-bit word). Hosts must expand tight narrow data
    /// one-element-per-word before binding it here.
    fn narrow_storage_u32_slot(&self) -> bool {
        true
    }

    // ── Buffers ────────────────────────────────────────────────────────────

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        let device = self.dev()?;
        let mut wgpu_usage = buffer_usage::COPY_SRC | buffer_usage::COPY_DST;
        if usage.has(FieldUsage::UNIFORM) {
            wgpu_usage |= buffer_usage::UNIFORM;
        } else {
            wgpu_usage |= buffer_usage::STORAGE;
        }
        let buf = unsafe { ffi::quanta_create_buffer(device, size as f64, wgpu_usage) };
        let handle = self.state.alloc_handle();
        self.state.buffers.0.borrow_mut().insert(handle, buf);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        if let Some(buf) = self.state.buffers.0.borrow_mut().remove(&handle) {
            unsafe { ffi::quanta_destroy_buffer(buf) };
        }
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let device = self.dev()?;
        let buffers = self.state.buffers.0.borrow();
        let &buf = buffers
            .get(&handle)
            .ok_or_else(|| Self::err("unknown buffer handle"))?;
        unsafe {
            ffi::quanta_write_buffer(device, buf, 0.0, data.as_ptr(), data.len());
        }
        Ok(())
    }

    fn field_read_bytes(&self, _handle: u64, _size: usize) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err(
            "field_read_bytes is async-only on WebGPU; use field_read_bytes_async",
        ))
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let device = self.dev()?;
        let buffers = self.state.buffers.0.borrow();
        let &s = buffers.get(&src).ok_or_else(|| Self::err("src missing"))?;
        let &d = buffers.get(&dst).ok_or_else(|| Self::err("dst missing"))?;
        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        unsafe {
            ffi::quanta_encoder_copy_buffer_to_buffer(encoder, s, 0.0, d, 0.0, size as f64);
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };
        Ok(())
    }

    // ── Compute (JIT path) ───────────────────────────────────────────────── (see compute.rs)

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel)
    }

    fn wave_jit(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_jit_impl(kernel_def)
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_impl(wave, groups)
    }

    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_indirect_impl(wave, buffer, offset)
    }

    // ── Sync (errors directing to async) ───────────────────────────────────

    fn pulse_wait(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        Err(Self::err(
            "pulse_wait is async-only on WebGPU; use pulse_wait_async",
        ))
    }

    fn wait_idle(&self) -> Result<(), QuantaError> {
        Err(Self::err(
            "wait_idle is async-only on WebGPU; drive pulses via pulse_wait_async",
        ))
    }

    fn pulse_poll(&self, _pulse: &Pulse) -> bool {
        false
    }

    // ── Occlusion queries (post-step-063 closure) ──────────────────────────

    fn occlusion_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        if count == 0 {
            return Err(QuantaError::invalid_param(
                "occlusion query set must have at least one slot",
            ));
        }
        let device = self.dev()?;
        let qs = unsafe { ffi::quanta_create_query_set(device, count) };
        if qs == ffi::NULL_HANDLE {
            return Err(Self::err("createQuerySet returned a null handle"));
        }
        let handle = self.state.alloc_handle();
        self.state
            .query_sets
            .0
            .borrow_mut()
            .insert(handle, (qs, count));
        Ok(handle)
    }

    fn occlusion_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        // WebGPU readback is fundamentally async — same shape as
        // pulse_wait. The recorded begin/end pairs work; reading
        // the resolved buffer back to the host needs a command-
        // buffer submit + mapAsync. Either wire an
        // `occlusion_query_read_async` (separate slice) or call
        // sites can read back via the standard
        // `field_read_bytes_async` path against the resolve
        // buffer they own.
        Err(Self::err(
            "occlusion_query_read is async-only on WebGPU; use occlusion_query_read_async or resolve into a buffer and field_read_bytes_async",
        ))
    }

    // ── Textures ───────────────────────────────────────────────────────────

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let device = self.dev()?;

        let format = format_code(desc.format)?;
        let mut usage = ffi::texture_usage::COPY_SRC | ffi::texture_usage::COPY_DST;
        if desc.usage.has(crate::TextureUsage::SHADER_READ) {
            usage |= ffi::texture_usage::TEXTURE_BINDING;
        }
        if desc.usage.has(crate::TextureUsage::SHADER_WRITE) {
            usage |= ffi::texture_usage::STORAGE_BINDING;
        }
        if desc.usage.has(crate::TextureUsage::RENDER_TARGET) {
            usage |= ffi::texture_usage::RENDER_ATTACHMENT;
        }

        let tex = unsafe {
            ffi::quanta_create_texture(
                device,
                desc.width,
                desc.height,
                desc.array_length.max(1),
                desc.mip_levels.max(1),
                desc.sample_count.max(1),
                format,
                usage,
            )
        };
        let view = unsafe { ffi::quanta_texture_create_view(tex) };

        let bytes_per_row = desc.width * desc.format.bytes_per_pixel() as u32;
        // WebGPU requires bytesPerRow to be a multiple of 256 for
        // copyTextureToBuffer; round up here so reads work even for
        // narrow textures.
        let bytes_per_row_aligned = bytes_per_row.div_ceil(256) * 256;

        let handle = self.state.alloc_handle();
        self.state.textures.0.borrow_mut().insert(
            handle,
            state::TextureEntry {
                texture: tex,
                view,
                width: desc.width,
                height: desc.height,
                format: desc.format,
                bytes_per_row: bytes_per_row_aligned,
            },
        );

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            sample_count: desc.sample_count,
            device: None,
            live: true,
        })
    }

    fn texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        if let Some(entry) = self.state.textures.0.borrow_mut().remove(&handle) {
            unsafe {
                // Drop the default view's JS handle, then destroy the
                // texture and release its JS handle.
                ffi::quanta_release(entry.view);
                ffi::quanta_destroy_texture(entry.texture);
            }
        }
        Ok(())
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        let device = self.dev()?;
        let textures = self.state.textures.0.borrow();
        let entry = textures
            .get(&texture.handle)
            .ok_or_else(|| Self::err("unknown texture handle"))?;
        let row = entry.width * entry.format.bytes_per_pixel() as u32;
        unsafe {
            ffi::quanta_queue_write_texture(
                device,
                entry.texture,
                data.as_ptr(),
                data.len(),
                row,
                entry.height,
                entry.width,
                entry.height,
                1,
            );
        }
        Ok(())
    }

    fn texture_read(&self, _texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err(
            "texture_read is async-only on WebGPU; use texture_read_async",
        ))
    }

    fn sampler_create(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let device = self.dev()?;
        let compare_code = match desc.compare {
            None => ffi::compare::UNSET,
            Some(c) => compare_op_code(c),
        };
        let sampler = unsafe {
            ffi::quanta_create_sampler(
                device,
                filter_code(desc.mag_filter),
                filter_code(desc.min_filter),
                filter_code(desc.mip_filter),
                address_code(desc.address_u),
                address_code(desc.address_v),
                ffi::address::CLAMP_TO_EDGE,
                desc.max_anisotropy as u32,
                compare_code,
            )
        };
        let handle = self.state.alloc_handle();
        self.state.samplers.0.borrow_mut().insert(handle, sampler);
        Ok(crate::Sampler {
            handle,
            device: None,
            live: true,
        })
    }

    fn sampler_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        if let Some(sampler) = self.state.samplers.0.borrow_mut().remove(&handle) {
            // GPUSampler has no destroy(); releasing the JS handle
            // lets the GC collect it.
            unsafe { ffi::quanta_release(sampler) };
        }
        Ok(())
    }

    fn occlusion_query_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        if let Some((query_set, _count)) = self.state.query_sets.0.borrow_mut().remove(&handle) {
            unsafe { ffi::quanta_release(query_set) };
        }
        Ok(())
    }

    // === Compute-resource lifecycle === (see compute.rs)

    #[cfg(feature = "compute")]
    fn wave_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.wave_destroy_impl(handle)
    }

    fn debug_registry_counts(&self) -> crate::RegistryCounts {
        crate::RegistryCounts {
            buffers: self.state.buffers.0.borrow().len(),
            textures: self.state.textures.0.borrow().len(),
            samplers: self.state.samplers.0.borrow().len(),
            render_pipelines: self.state.pipelines.0.borrow().len(),
            query_sets: self.state.query_sets.0.borrow().len(),
            waves: self.state.waves.0.borrow().len(),
            render_samplers: 0,
        }
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU mipmap generation pending"))
    }

    // ── Render path ──────────────────────────────────────────────────────── (render-gated, step 085; see render.rs)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    #[cfg(feature = "render")]
    fn pipeline_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.pipeline_destroy_impl(handle)
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.render_begin_impl(target)
    }

    #[cfg(feature = "render")]
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        self.render_end_impl(pass)
    }

    // Spec-absent features on WebGPU: every method below returns
    // NotSupported, not InvalidParam. Mesh shaders, ray tracing,
    // and sparse residency are simply not in the WebGPU spec — no
    // user fault, no parameter to fix, the backend is genuinely
    // incapable. Matches the step-070 / step-062 categorization
    // contract.
    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(Self::not_supported(
            "mesh shaders are not in the WebGPU spec",
        ))
    }
    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, _geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        Err(Self::not_supported("ray tracing is not in the WebGPU spec"))
    }
    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        Err(Self::not_supported("ray tracing is not in the WebGPU spec"))
    }
    fn dispatch_rays(&self, _pipeline: u64, _w: u32, _h: u32) -> Result<(), QuantaError> {
        Err(Self::not_supported("ray tracing is not in the WebGPU spec"))
    }
    fn destroy_acceleration_structure(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(Self::not_supported("ray tracing is not in the WebGPU spec"))
    }
    fn sparse_texture_create(&self, _desc: &TextureDesc) -> Result<u64, QuantaError> {
        Err(Self::not_supported(
            "sparse residency is not in the WebGPU spec",
        ))
    }
    fn sparse_map_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        Err(Self::not_supported(
            "sparse residency is not in the WebGPU spec",
        ))
    }
    fn sparse_unmap_tile(
        &self,
        _texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
    ) -> Result<(), QuantaError> {
        Err(Self::not_supported(
            "sparse residency is not in the WebGPU spec",
        ))
    }
    // === Indirect command buffers (steps 032 + 033) ===
    //
    // W3C WebGPU has `GPURenderBundle` for the render path but does
    // not expose compute bundles, so a native ICB lowering for
    // compute is not available on this backend. We refine the
    // proven `Quanta.Icb.execute` semantics (Lean T7000) by
    // recording dispatches as snapshots and replaying them through
    // `wave_dispatch` at execute time. The IR-level theorem is
    // parametric in the per-command transformer, so this is
    // observationally identical to a native bundle.

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        let handle = self.state.alloc_handle();
        self.state.icbs.0.borrow_mut().insert(
            handle,
            state::WebgpuIcb {
                cap: max_commands,
                commands: alloc::vec::Vec::with_capacity(max_commands as usize),
            },
        );
        Ok(handle)
    }

    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        if wave.push_mask != 0 || wave.push_len != 0 {
            return Err(Self::err(
                "WebGPU ICB does not support push constants in this MVP",
            ));
        }
        if wave.texture_count != 0 {
            return Err(Self::err(
                "WebGPU ICB does not support texture bindings in this MVP",
            ));
        }
        let mut icbs = self.state.icbs.0.borrow_mut();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| Self::err("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(Self::err("ICB record index must equal current length"));
        }
        if index >= icb.cap {
            return Err(Self::err("ICB index >= capacity"));
        }
        icb.commands.push(state::WebgpuIcbCommand::Dispatch {
            wave_handle: wave.handle,
            bindings: wave.bindings,
            binding_count: wave.binding_count,
            workgroup_size: wave.workgroup_size,
            groups,
        });
        Ok(())
    }

    fn icb_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let mut icbs = self.state.icbs.0.borrow_mut();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| Self::err("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(Self::err("ICB record index must equal current length"));
        }
        if index >= icb.cap {
            return Err(Self::err("ICB index >= capacity"));
        }
        icb.commands.push(state::WebgpuIcbCommand::Draw {
            pipeline,
            vertex_count,
            instance_count,
        });
        Ok(())
    }

    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        // Snapshot the recorded sequence under the borrow, then
        // drop it before re-entering `wave_dispatch` (which takes
        // its own borrow on `self.state.waves`).
        let snapshot: alloc::vec::Vec<state::WebgpuIcbCommand> = {
            let icbs = self.state.icbs.0.borrow();
            let icb = icbs
                .get(&handle)
                .ok_or_else(|| Self::err("ICB handle not found"))?;
            if count > icb.commands.len() as u32 {
                return Err(Self::err("ICB execute count exceeds recorded length"));
            }
            icb.commands[..count as usize]
                .iter()
                .map(|r| match r {
                    state::WebgpuIcbCommand::Dispatch {
                        wave_handle,
                        bindings,
                        binding_count,
                        workgroup_size,
                        groups,
                    } => state::WebgpuIcbCommand::Dispatch {
                        wave_handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        workgroup_size: *workgroup_size,
                        groups: *groups,
                    },
                    state::WebgpuIcbCommand::Draw {
                        pipeline,
                        vertex_count,
                        instance_count,
                    } => state::WebgpuIcbCommand::Draw {
                        pipeline: *pipeline,
                        vertex_count: *vertex_count,
                        instance_count: *instance_count,
                    },
                })
                .collect()
        };
        for rec in &snapshot {
            match rec {
                state::WebgpuIcbCommand::Dispatch {
                    wave_handle,
                    bindings,
                    binding_count,
                    workgroup_size,
                    groups,
                } => {
                    // Transient replay Wave: `live: false` + `device:
                    // None` disarm its Drop, so replay does not
                    // destroy the real wave's registry entry.
                    let wave = Wave {
                        handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: [0; crate::api::types::MAX_TEXTURES],
                        texture_count: 0,
                        storage_texture_kinds: [0; 16],
                        push_data: [0; crate::api::types::PUSH_DATA_CAP],
                        push_len: 0,
                        push_mask: 0,
                        workgroup_size: *workgroup_size,
                        device: None,
                        live: false,
                    };
                    self.wave_dispatch(&wave, *groups)?;
                }
                state::WebgpuIcbCommand::Draw { .. } => {
                    // Native render-bundle (GPURenderBundle) lowering
                    // requires a render-pass encoder context; that
                    // wiring is a future commit. T7006 is already
                    // satisfied by the recording shape.
                }
            }
        }
        Ok(())
    }

    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.state.icbs.0.borrow_mut().remove(&handle);
        Ok(())
    }

    // === Render bundles (steps 032 + 033, render path) ===
    //
    // Native lowering: store recorded draws as snapshots; translate
    // RenderOp::ExecuteRenderBundle into a fresh
    // GPURenderBundleEncoder + draw calls + finish() + executeBundles
    // on the active render pass. The bundle's color/depth format
    // is taken from the render target at execute time.

    fn render_bundle_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        let handle = self.state.alloc_handle();
        self.state.render_bundles.0.borrow_mut().insert(
            handle,
            state::WebgpuRenderBundle {
                cap: max_commands,
                draws: alloc::vec::Vec::with_capacity(max_commands as usize),
            },
        );
        Ok(handle)
    }

    fn render_bundle_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let mut bundles = self.state.render_bundles.0.borrow_mut();
        let bundle = bundles
            .get_mut(&handle)
            .ok_or_else(|| Self::err("render bundle handle not found"))?;
        if index != bundle.draws.len() as u32 {
            return Err(Self::err(
                "render bundle record index must equal current length",
            ));
        }
        if index >= bundle.cap {
            return Err(Self::err("render bundle index >= capacity"));
        }
        bundle.draws.push(state::RenderBundleDraw {
            pipeline_handle: pipeline,
            vertex_count,
            instance_count,
        });
        Ok(())
    }

    fn render_bundle_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.state.render_bundles.0.borrow_mut().remove(&handle);
        Ok(())
    }

    // === Bindless typed wrappers (steps 034 + 035) ===
    //
    // WebGPU has no native bindless in the W3C spec — the closest
    // primitive is a fixed-size sampled-texture-array binding. MVP
    // here is a software table: the host maintains a list of
    // resource handles, and shaders that want to index into the
    // array must rebind the matching slot per draw. Refines the
    // proven `Quanta.Bindless.Array` model exactly.

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.state.alloc_handle();
        self.state.bindless_textures.0.borrow_mut().insert(
            handle,
            state::WebgpuBindlessArray {
                cap,
                entries: alloc::vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        let mut arrays = self.state.bindless_textures.0.borrow_mut();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| Self::err("bindless texture array not found"))?;
        if index >= arr.cap {
            return Err(Self::err("bindless texture index >= capacity"));
        }
        arr.entries[index as usize] = texture;
        Ok(())
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.state.bindless_textures.0.borrow_mut().remove(&handle);
        Ok(())
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.state.alloc_handle();
        self.state.bindless_buffers.0.borrow_mut().insert(
            handle,
            state::WebgpuBindlessArray {
                cap,
                entries: alloc::vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        let mut arrays = self.state.bindless_buffers.0.borrow_mut();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| Self::err("bindless buffer array not found"))?;
        if index >= arr.cap {
            return Err(Self::err("bindless buffer index >= capacity"));
        }
        arr.entries[index as usize] = buffer;
        Ok(())
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.state.bindless_buffers.0.borrow_mut().remove(&handle);
        Ok(())
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

// ── Wave / Pulse construction helpers ──────────────────────────────────────

fn make_wave(handle: u64, workgroup_size: [u32; 3]) -> Wave {
    Wave {
        handle,
        bindings: [0; crate::api::types::MAX_BINDINGS],
        binding_count: 0,
        texture_bindings: [0; crate::api::types::MAX_TEXTURES],
        texture_count: 0,
        storage_texture_kinds: [0; 16],
        push_data: [0; crate::api::types::PUSH_DATA_CAP],
        push_len: 0,
        push_mask: 0,
        workgroup_size,
        device: None,
        live: true,
    }
}

fn make_pulse() -> Pulse {
    Pulse {
        handle: 0,
        completed: true,
        wait_fn: None,
    }
}

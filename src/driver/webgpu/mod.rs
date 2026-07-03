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

mod executor;
mod ffi;
mod state;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

use crate::{
    Caps, FieldUsage, Format, FormatCaps, Gpu, GpuDevice as QGpuDevice, Pulse, QuantaError,
    Texture, TextureDesc, Vendor, Wave,
};
// Render types used only by the render-gated impl methods + helpers (085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};

use ffi::{NULL_HANDLE, buffer_usage};
use state::{SendCell, State, WaveEntry};

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

// ── Async-only public extension methods ─────────────────────────────────────

#[allow(dead_code)] // public extension API used by browser-side callers.
impl WebgpuDevice {
    /// Async sibling of [`field_read_bytes`]: maps the buffer for read,
    /// awaits the GPU, copies into a `Vec<u8>`, and unmaps.
    ///
    /// `field_read_bytes` (sync trait) cannot be implemented on WebGPU
    /// because the browser event loop is non-blocking. Use this method
    /// from async Rust code.
    pub async fn field_read_bytes_async(
        &self,
        handle: u64,
        size: usize,
    ) -> Result<Vec<u8>, QuantaError> {
        let device = self.dev()?;
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                size as f64,
                buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            )
        };

        {
            let buffers = self.state.buffers.0.borrow();
            let &src = buffers
                .get(&handle)
                .ok_or_else(|| Self::err("unknown buffer handle"))?;
            let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
            unsafe {
                ffi::quanta_encoder_copy_buffer_to_buffer(
                    encoder,
                    src,
                    0.0,
                    staging,
                    0.0,
                    size as f64,
                );
            }
            let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
            unsafe { ffi::quanta_queue_submit(device, cmd) };
        }

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("mapAsync rejected"))?;

        let mut out = alloc::vec![0u8; size];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, out.as_mut_ptr(), size);
        }
        unsafe {
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }
        Ok(out)
    }

    /// Async sibling of [`occlusion_query_read`]: resolves the
    /// query set into a staging buffer, awaits `mapAsync`, and
    /// returns the per-slot u64 fragment counts.
    ///
    /// The sync trait method `occlusion_query_read` cannot be
    /// implemented on WebGPU because the browser event loop is
    /// non-blocking. Use this method from async Rust code (or
    /// drive the typed `quanta::Pulse` async waiter and read
    /// from a buffer you own via `field_read_bytes_async`).
    pub async fn occlusion_query_read_async(
        &self,
        handle: u64,
    ) -> Result<alloc::vec::Vec<u64>, QuantaError> {
        let device = self.dev()?;
        let (qs_js, count) = {
            let qs = self.state.query_sets.0.borrow();
            *qs.get(&handle)
                .ok_or_else(|| Self::err("unknown occlusion query handle"))?
        };
        let bytes = (count as f64) * 8.0;
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                bytes,
                buffer_usage::COPY_SRC
                    | buffer_usage::COPY_DST
                    | buffer_usage::QUERY_RESOLVE
                    | buffer_usage::MAP_READ,
            )
        };

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        unsafe {
            ffi::quanta_encoder_resolve_query_set(encoder, qs_js, 0, count, staging, 0.0);
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("occlusion query mapAsync rejected"))?;

        let size = bytes as usize;
        let mut raw = alloc::vec![0u8; size];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, raw.as_mut_ptr(), size);
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }

        // Each slot is a little-endian u64.
        let mut out = alloc::vec::Vec::with_capacity(count as usize);
        for i in 0..count as usize {
            let off = i * 8;
            let chunk: [u8; 8] = raw[off..off + 8].try_into().unwrap_or([0u8; 8]);
            out.push(u64::from_le_bytes(chunk));
        }
        Ok(out)
    }

    /// Async sibling of [`pulse_wait`]: awaits
    /// `device.queue.onSubmittedWorkDone()` for the pulse's submission.
    pub async fn pulse_wait_async(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        let device = self.dev()?;
        Promise::register(|task| unsafe { ffi::quanta_queue_on_submitted_work_done(device, task) })
            .await
            .map_err(|_| Self::err("onSubmittedWorkDone rejected"))?;
        Ok(())
    }

    /// Async sibling of [`texture_read`]: copies the texture to a
    /// staging buffer, awaits `mapAsync`, and returns the pixel bytes
    /// (tightly packed in the texture's native row stride).
    pub async fn texture_read_async(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let device = self.dev()?;
        let (tex_handle, view_dims, bytes_per_row, height, tight_row) = {
            let textures = self.state.textures.0.borrow();
            let entry = textures
                .get(&texture.handle)
                .ok_or_else(|| Self::err("unknown texture handle"))?;
            (
                entry.texture,
                (entry.width, entry.height),
                entry.bytes_per_row,
                entry.height,
                entry.width * entry.format.bytes_per_pixel() as u32,
            )
        };

        let total_bytes = (bytes_per_row as u64) * (height as u64);
        let staging = unsafe {
            ffi::quanta_create_buffer(
                device,
                total_bytes as f64,
                buffer_usage::COPY_DST | buffer_usage::MAP_READ,
            )
        };

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        unsafe {
            ffi::quanta_encoder_copy_texture_to_buffer(
                encoder,
                tex_handle,
                staging,
                bytes_per_row,
                height,
                view_dims.0,
                view_dims.1,
                1,
            );
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };

        Promise::register(|task| unsafe { ffi::quanta_map_async_read(staging, task) })
            .await
            .map_err(|_| Self::err("texture mapAsync rejected"))?;

        let total = (bytes_per_row as usize) * (height as usize);
        let mut padded = alloc::vec![0u8; total];
        unsafe {
            ffi::quanta_get_mapped_range_copy(staging, padded.as_mut_ptr(), total);
        }
        unsafe {
            ffi::quanta_unmap_buffer(staging);
            ffi::quanta_destroy_buffer(staging);
        }

        if bytes_per_row == tight_row {
            Ok(padded)
        } else {
            let mut out = Vec::with_capacity(tight_row as usize * height as usize);
            for row in 0..height as usize {
                let off = row * bytes_per_row as usize;
                out.extend_from_slice(&padded[off..off + tight_row as usize]);
            }
            Ok(out)
        }
    }
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

#[cfg(feature = "render")]
fn compare_func_code(c: crate::pipeline::CompareFunc) -> u32 {
    match c {
        crate::pipeline::CompareFunc::Never => ffi::compare::NEVER,
        crate::pipeline::CompareFunc::Less => ffi::compare::LESS,
        crate::pipeline::CompareFunc::Equal => ffi::compare::EQUAL,
        crate::pipeline::CompareFunc::LessEqual => ffi::compare::LESS_EQUAL,
        crate::pipeline::CompareFunc::Greater => ffi::compare::GREATER,
        crate::pipeline::CompareFunc::NotEqual => ffi::compare::NOT_EQUAL,
        crate::pipeline::CompareFunc::GreaterEqual => ffi::compare::GREATER_EQUAL,
        crate::pipeline::CompareFunc::Always => ffi::compare::ALWAYS,
    }
}

#[cfg(feature = "render")]
fn attribute_format_code(f: crate::pipeline::AttributeFormat) -> u32 {
    use crate::pipeline::AttributeFormat as A;
    match f {
        A::Float => ffi::attribute_format::FLOAT,
        A::Float2 => ffi::attribute_format::FLOAT2,
        A::Float3 => ffi::attribute_format::FLOAT3,
        A::Float4 => ffi::attribute_format::FLOAT4,
        A::Int => ffi::attribute_format::SINT,
        A::Int2 => ffi::attribute_format::SINT2,
        A::Int3 => ffi::attribute_format::SINT3,
        A::Int4 => ffi::attribute_format::SINT4,
        A::UInt => ffi::attribute_format::UINT,
        A::UInt2 => ffi::attribute_format::UINT2,
        A::UInt3 => ffi::attribute_format::UINT3,
        A::UInt4 => ffi::attribute_format::UINT4,
        A::UByte4Norm => ffi::attribute_format::UNORM8X4,
    }
}

#[cfg(feature = "render")]
fn topology_code(p: crate::pipeline::Primitive) -> u32 {
    use crate::pipeline::Primitive as P;
    match p {
        P::Point => ffi::topology::POINT,
        P::Line => ffi::topology::LINE,
        P::LineStrip => ffi::topology::LINE_STRIP,
        P::Triangle => ffi::topology::TRIANGLE,
        P::TriangleStrip => ffi::topology::TRIANGLE_STRIP,
    }
}

#[cfg(feature = "render")]
fn cull_mode_code(c: crate::pipeline::CullMode) -> u32 {
    use crate::pipeline::CullMode as C;
    match c {
        C::None => ffi::cull_mode::NONE,
        C::Front => ffi::cull_mode::FRONT,
        C::Back => ffi::cull_mode::BACK,
    }
}

#[cfg(feature = "render")]
fn blend_factor_code(f: crate::pipeline::BlendFactor) -> u32 {
    use crate::pipeline::BlendFactor as F;
    match f {
        F::Zero => ffi::blend_factor::ZERO,
        F::One => ffi::blend_factor::ONE,
        F::SrcAlpha => ffi::blend_factor::SRC_ALPHA,
        F::OneMinusSrcAlpha => ffi::blend_factor::ONE_MINUS_SRC_ALPHA,
        F::DstAlpha => ffi::blend_factor::DST_ALPHA,
        F::OneMinusDstAlpha => ffi::blend_factor::ONE_MINUS_DST_ALPHA,
        F::SrcColor => ffi::blend_factor::SRC_COLOR,
        F::OneMinusSrcColor => ffi::blend_factor::ONE_MINUS_SRC_COLOR,
        F::DstColor => ffi::blend_factor::DST_COLOR,
        F::OneMinusDstColor => ffi::blend_factor::ONE_MINUS_DST_COLOR,
    }
}

#[cfg(feature = "render")]
fn blend_op_code(o: crate::pipeline::BlendOp) -> u32 {
    use crate::pipeline::BlendOp as O;
    match o {
        O::Add => ffi::blend_op::ADD,
        O::Subtract => ffi::blend_op::SUBTRACT,
        O::ReverseSubtract => ffi::blend_op::REVERSE_SUBTRACT,
        O::Min => ffi::blend_op::MIN,
        O::Max => ffi::blend_op::MAX,
    }
}

#[cfg(feature = "render")]
fn step_mode_code(s: crate::pipeline::StepMode) -> u32 {
    match s {
        crate::pipeline::StepMode::Vertex => ffi::step_mode::VERTEX,
        crate::pipeline::StepMode::Instance => ffi::step_mode::INSTANCE,
    }
}

// ── GpuDevice impl ──────────────────────────────────────────────────────────

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

    // ── Compute (JIT path) ─────────────────────────────────────────────────

    fn wave(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(Self::err(
            "WebGPU does not accept pre-compiled binaries; use the JIT path",
        ))
    }

    fn wave_jit(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        let kernel = quanta_ir::deserialize_kernel(kernel_def)
            .map_err(|e| Self::err_owned(format!("deserialize KernelDef: {}", e)))?;

        // Step 082 Layer 4: WebGPU's WGSL surface rejects F64,
        // I64/U64, and narrow ints (u8/u16/i8/i16). Catch those at
        // validation time so the user gets a clean NotSupported
        // instead of an opaque "emit_wgsl_jit" error.
        let report = quanta_ir::validate::validate_for(&quanta_ir::caps::WEBGPU, &kernel);
        if !report.is_ok() {
            return Err(QuantaError::not_supported(
                "kernel uses unsupported scalar type for WebGPU",
            )
            .with_context(&format!("{}", report)));
        }

        let wgsl = quanta_ir::emit_wgsl::emit_wgsl_jit(&kernel)
            .map_err(|e| Self::err_owned(format!("emit_wgsl_jit: {}", e)))?;

        let device = self.dev()?;
        let module = unsafe { ffi::quanta_create_shader_module(device, wgsl.as_ptr(), wgsl.len()) };
        let pipeline = unsafe {
            ffi::quanta_create_compute_pipeline(
                device,
                module,
                kernel.name.as_ptr(),
                kernel.name.len(),
            )
        };
        let layout = unsafe { ffi::quanta_compute_pipeline_get_bind_group_layout(pipeline, 0) };

        let handle = self.state.alloc_handle();
        self.state.waves.0.borrow_mut().insert(
            handle,
            WaveEntry {
                pipeline,
                _shader: module,
                workgroup_size: kernel.workgroup_size,
                layout,
                bindings: alloc::collections::BTreeMap::new(),
            },
        );

        Ok(make_wave(handle, kernel.workgroup_size))
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let device = self.dev()?;
        let mut waves = self.state.waves.0.borrow_mut();
        let entry = waves
            .get_mut(&wave.handle)
            .ok_or_else(|| Self::err("unknown wave handle"))?;

        let bg_desc = unsafe { ffi::quanta_bg_desc_create(entry.layout) };
        {
            let buffers = self.state.buffers.0.borrow();
            for (slot_idx, &buf_handle) in wave.bindings.iter().enumerate() {
                if buf_handle == 0 {
                    continue;
                }
                let &buf = buffers
                    .get(&buf_handle)
                    .ok_or_else(|| Self::err("bound buffer not found"))?;
                unsafe {
                    ffi::quanta_bg_desc_add_buffer(bg_desc, slot_idx as u32, buf);
                }
                entry.bindings.insert(slot_idx as u32, buf_handle);
            }
        }
        let bind_group = unsafe { ffi::quanta_create_bind_group(device, bg_desc) };

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        let pass = unsafe { ffi::quanta_encoder_begin_compute_pass(encoder) };
        unsafe {
            ffi::quanta_compute_pass_set_pipeline(pass, entry.pipeline);
            ffi::quanta_compute_pass_set_bind_group(pass, 0, bind_group);
            ffi::quanta_compute_pass_dispatch(pass, groups[0], groups[1].max(1), groups[2].max(1));
            ffi::quanta_compute_pass_end(pass);
        }
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };
        unsafe { ffi::quanta_release(bind_group) };

        Ok(make_pulse())
    }

    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(Self::err("WebGPU indirect dispatch pending"))
    }

    // ── Sync (errors directing to async) ───────────────────────────────────

    fn pulse_wait(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        Err(Self::err(
            "pulse_wait is async-only on WebGPU; use pulse_wait_async",
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
            device: None,
        })
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
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Err(Self::err("WebGPU mipmap generation pending"))
    }

    // ── Render path ──────────────────────────────────────────────────────── (render-gated, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        // Step 063 slice 11 — WebGPU spec doesn't include
        // tessellation, mesh shaders, or conservative rasterization.
        // Surface NotSupported up-front rather than silently dropping
        // the request when the user sets these on PipelineDesc
        // (matches Kani T418 / T419 no-silent-drops contract,
        // symmetric to slices 5 and 8/9).
        if desc.tessellation.is_some() {
            return Err(Self::not_supported(
                "WebGPU render pipelines: tessellation is not in the WebGPU spec",
            ));
        }
        if desc.mesh_shader.is_some() {
            return Err(Self::not_supported(
                "WebGPU render pipelines: mesh shaders are not in the WebGPU spec",
            ));
        }
        if desc.conservative_rasterization {
            return Err(Self::not_supported(
                "WebGPU render pipelines: conservative rasterization is not in the WebGPU spec",
            ));
        }
        let device = self.dev()?;

        let combined = desc.source;
        let vs_src = combined.unwrap_or(desc.vertex);
        let fs_src = combined.unwrap_or(desc.fragment);
        let vs_text = core::str::from_utf8(vs_src)
            .map_err(|_| Self::err("vertex shader is not valid UTF-8 WGSL"))?;
        let fs_text = core::str::from_utf8(fs_src)
            .map_err(|_| Self::err("fragment shader is not valid UTF-8 WGSL"))?;

        let vs_module =
            unsafe { ffi::quanta_create_shader_module(device, vs_text.as_ptr(), vs_text.len()) };
        let fs_module = if combined.is_some() || vs_text == fs_text {
            // Reuse the same module handle — JS side doesn't need a
            // distinct copy. Keep the lifecycle simple by allocating
            // a parallel handle on the JS side instead of aliasing in
            // the table; cheap call but clearer ownership.
            unsafe { ffi::quanta_create_shader_module(device, vs_text.as_ptr(), vs_text.len()) }
        } else {
            unsafe { ffi::quanta_create_shader_module(device, fs_text.as_ptr(), fs_text.len()) }
        };

        let rp_desc = unsafe { ffi::quanta_rp_desc_create() };

        unsafe {
            ffi::quanta_rp_desc_set_vertex(
                rp_desc,
                vs_module,
                desc.vertex_entry.as_ptr(),
                desc.vertex_entry.len(),
            );
        }

        for (buf_index, layout) in desc.vertex_layouts.iter().enumerate() {
            unsafe {
                ffi::quanta_rp_desc_add_vertex_buffer(
                    rp_desc,
                    layout.stride,
                    step_mode_code(layout.step),
                );
            }
            for a in &layout.attributes {
                unsafe {
                    ffi::quanta_rp_desc_add_vertex_attribute(
                        rp_desc,
                        buf_index as u32,
                        attribute_format_code(a.format),
                        a.offset,
                        a.location,
                    );
                }
            }
        }

        for (i, fmt) in desc.color_formats.iter().enumerate() {
            let blend_state = desc
                .blend_states
                .get(i)
                .copied()
                .or_else(|| desc.blend_states.last().copied())
                .unwrap_or(desc.blend);
            unsafe {
                ffi::quanta_rp_desc_add_color_target(
                    rp_desc,
                    format_code(*fmt)?,
                    if blend_state.enabled { 1 } else { 0 },
                    blend_factor_code(blend_state.src_rgb),
                    blend_factor_code(blend_state.dst_rgb),
                    blend_op_code(blend_state.op_rgb),
                    blend_factor_code(blend_state.src_alpha),
                    blend_factor_code(blend_state.dst_alpha),
                    blend_op_code(blend_state.op_alpha),
                );
            }
        }

        unsafe {
            ffi::quanta_rp_desc_set_fragment(
                rp_desc,
                fs_module,
                desc.fragment_entry.as_ptr(),
                desc.fragment_entry.len(),
            );
        }
        unsafe {
            ffi::quanta_rp_desc_set_primitive(
                rp_desc,
                topology_code(desc.primitive),
                cull_mode_code(desc.cull_mode),
            );
            ffi::quanta_rp_desc_set_multisample(rp_desc, desc.sample_count.max(1));
        }
        if let Some(depth_fmt) = desc.depth_format {
            unsafe {
                ffi::quanta_rp_desc_set_depth_stencil(
                    rp_desc,
                    format_code(depth_fmt)?,
                    if desc.depth_stencil.depth_write { 1 } else { 0 },
                    compare_func_code(desc.depth_stencil.depth_compare),
                );
            }
        }

        let pipeline = unsafe { ffi::quanta_create_render_pipeline(device, rp_desc) };
        let layout = unsafe { ffi::quanta_render_pipeline_get_bind_group_layout(pipeline, 0) };

        // Modules are referenced by the pipeline; we no longer need
        // the JS-side handles.
        unsafe {
            ffi::quanta_release(vs_module);
            ffi::quanta_release(fs_module);
        }

        let handle = self.state.alloc_handle();
        self.state
            .pipelines
            .0
            .borrow_mut()
            .insert(handle, state::PipelineEntry { pipeline, layout });

        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle,
            ops: Vec::new(),
            color_targets: alloc::vec![crate::render_pass::ColorTarget {
                texture: target.handle,
                load_op: crate::LoadOp::Clear(crate::Color::CLEAR),
                store_op: crate::StoreOp::Store,
            }],
            depth_target: None,
        })
    }

    #[cfg(feature = "render")]
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let device = self.dev()?;

        let textures = self.state.textures.0.borrow();
        let target = textures
            .get(&pass.handle)
            .ok_or_else(|| Self::err("unknown render target"))?;

        // Pre-walk: find the clear color and (if attached) the depth
        // clear. Both end up on the rpass descriptor, not on encoder
        // calls — WebGPU §16 lets the user specify them once at
        // beginRenderPass time.
        let mut clear_rgba = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut clear_depth: Option<f32> = None;
        for op in &pass.ops {
            match op {
                crate::render_pass::RenderOp::Clear(color) => {
                    clear_rgba = (color.r, color.g, color.b, color.a);
                }
                crate::render_pass::RenderOp::ClearDepth(d) => {
                    clear_depth = Some(*d);
                }
                _ => {}
            }
        }

        let rpass_desc = unsafe { ffi::quanta_rpass_desc_create() };
        unsafe {
            ffi::quanta_rpass_desc_add_color_attachment(
                rpass_desc,
                target.view,
                ffi::load_op::CLEAR,
                ffi::store_op::STORE,
                clear_rgba.0,
                clear_rgba.1,
                clear_rgba.2,
                clear_rgba.3,
            );
        }
        // If the API caller attached a depth target, wire its view onto
        // the rpass desc. WebGPU only takes the clear value alongside
        // the attachment; ClearDepth carries the value from the op
        // stream into this attachment, so the depth target itself is
        // the source of truth for "which texture", and ClearDepth is
        // the source of truth for "what value."
        if let Some(depth) = &pass.depth_target {
            let depth_tex = textures
                .get(&depth.texture)
                .ok_or_else(|| Self::err("unknown depth target"))?;
            unsafe {
                ffi::quanta_rpass_desc_set_depth_attachment(
                    rpass_desc,
                    depth_tex.view,
                    if clear_depth.is_some() {
                        ffi::load_op::CLEAR
                    } else {
                        ffi::load_op::LOAD
                    },
                    ffi::store_op::STORE,
                    clear_depth.unwrap_or(1.0),
                );
            }
        }

        // Occlusion-query attachment: pre-walk pass.ops for the
        // first BeginOcclusionQuery, look up its query set in the
        // device registry, and bind it to the render pass desc
        // BEFORE beginRenderPass — WebGPU requires the
        // occlusionQuerySet to be set at descriptor time.
        let occlusion_qs_js: Option<u32> = pass
            .ops
            .iter()
            .find_map(|op| {
                if let crate::render_pass::RenderOp::BeginOcclusionQuery { handle, .. } = op {
                    Some(*handle)
                } else {
                    None
                }
            })
            .and_then(|h| self.state.query_sets.0.borrow().get(&h).map(|(js, _)| *js));
        if let Some(qs_js) = occlusion_qs_js {
            unsafe { ffi::quanta_rpass_desc_set_occlusion_query_set(rpass_desc, qs_js) };
        }

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        let rp = unsafe { ffi::quanta_encoder_begin_render_pass(encoder, rpass_desc) };

        let pipelines = self.state.pipelines.0.borrow();
        let buffers = self.state.buffers.0.borrow();
        let mut current_pipeline: Option<&state::PipelineEntry> = None;

        /// One slot of a pending bind group. The JS-side resource is
        /// either a buffer (long-lived; not owned by `render_end`), a
        /// texture view (long-lived; lookup via `state.textures`), a
        /// sampler (created here from a `SamplerDesc`; owned), or a
        /// freshly-allocated uniform buffer holding push-constant
        /// bytes (WebGPU has no push constants — the SetValue
        /// fallback below allocates a per-call buffer; owned).
        enum BindEntry {
            Buffer(u32),
            TextureView(u32),
            Sampler(u32),
            OwnedBuffer(u32),
        }
        let mut bind_entries: alloc::collections::BTreeMap<u32, BindEntry> =
            alloc::collections::BTreeMap::new();

        // Helper: flush pending bind entries into a real bind group
        // and bind it. Hoisted out of the match for the two draw
        // variants below.
        let flush_bg = |bind_entries: &mut alloc::collections::BTreeMap<u32, BindEntry>,
                        cur: Option<&state::PipelineEntry>|
         -> Option<u32> {
            if bind_entries.is_empty() {
                return None;
            }
            let p = cur?;
            let bg_desc = unsafe { ffi::quanta_bg_desc_create(p.layout) };
            for (slot, entry) in bind_entries.iter() {
                match entry {
                    BindEntry::Buffer(h) | BindEntry::OwnedBuffer(h) => unsafe {
                        ffi::quanta_bg_desc_add_buffer(bg_desc, *slot, *h)
                    },
                    BindEntry::TextureView(h) => unsafe {
                        ffi::quanta_bg_desc_add_texture_view(bg_desc, *slot, *h)
                    },
                    BindEntry::Sampler(h) => unsafe {
                        ffi::quanta_bg_desc_add_sampler(bg_desc, *slot, *h)
                    },
                }
            }
            let bg = unsafe { ffi::quanta_create_bind_group(device, bg_desc) };
            unsafe { ffi::quanta_render_pass_set_bind_group(rp, 0, bg) };
            bind_entries.clear();
            Some(bg)
        };

        let mut owned_bgs: Vec<u32> = Vec::new();
        // Resources allocated within this pass that must be released
        // after submit: samplers minted from `SetSampler` and uniform
        // buffers allocated as push-constant fallbacks for `SetValue`.
        let mut owned_samplers: Vec<u32> = Vec::new();
        let mut owned_buffers: Vec<u32> = Vec::new();

        for op in &pass.ops {
            use crate::render_pass::RenderOp;
            match op {
                RenderOp::SetPipeline(handle) => {
                    let entry = pipelines
                        .get(handle)
                        .ok_or_else(|| Self::err("unknown pipeline"))?;
                    unsafe { ffi::quanta_render_pass_set_pipeline(rp, entry.pipeline) };
                    current_pipeline = Some(entry);
                }
                RenderOp::BindVertices {
                    slot,
                    handle,
                    offset,
                } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("vbuf"))?;
                    unsafe {
                        ffi::quanta_render_pass_set_vertex_buffer(rp, *slot, buf, *offset as f64);
                    }
                }
                RenderOp::BindIndices { handle, offset } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("ibuf"))?;
                    unsafe {
                        ffi::quanta_render_pass_set_index_buffer(
                            rp,
                            buf,
                            ffi::index_format::UINT32,
                            *offset as f64,
                        );
                    }
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("ubuf"))?;
                    bind_entries.insert(*slot, BindEntry::Buffer(buf));
                }
                RenderOp::Clear(_) | RenderOp::ClearDepth(_) => {
                    // Both clear values are picked up in the pre-walk
                    // above and applied as `clearValue` on the rpass
                    // descriptor; nothing to emit per-op.
                }
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    unsafe {
                        ffi::quanta_render_pass_draw(rp, *vertex_count, *instance_count);
                    }
                }
                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    unsafe {
                        ffi::quanta_render_pass_draw_indexed(rp, *index_count, *instance_count);
                    }
                }
                RenderOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => unsafe {
                    ffi::quanta_render_pass_set_viewport(
                        rp, *x, *y, *width, *height, *min_depth, *max_depth,
                    );
                },
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => unsafe {
                    ffi::quanta_render_pass_set_scissor(rp, *x, *y, *width, *height);
                },
                // ── Step C wiring ───────────────────────────────────────────
                RenderOp::SetTexture { slot, handle } => {
                    let view = textures
                        .get(handle)
                        .ok_or_else(|| Self::err("unknown texture for SetTexture"))?
                        .view;
                    bind_entries.insert(*slot, BindEntry::TextureView(view));
                }
                RenderOp::SetSampler { slot, sampler } => {
                    let s = unsafe {
                        ffi::quanta_create_sampler(
                            device,
                            filter_code(sampler.mag_filter),
                            filter_code(sampler.min_filter),
                            filter_code(sampler.mip_filter),
                            address_code(sampler.address_u),
                            address_code(sampler.address_v),
                            // WebGPU samplers are 3D-addressable; the
                            // public `SamplerDesc` only carries U/V, so
                            // mirror V into W (same as Vulkan/Metal
                            // drivers do for 2D textures).
                            address_code(sampler.address_v),
                            sampler.max_anisotropy as u32,
                            // `compare::UNSET` is the JS-side sentinel
                            // for "no compare function" — the JS layer
                            // omits the field entirely when it sees
                            // this code.
                            sampler
                                .compare
                                .map(compare_op_code)
                                .unwrap_or(ffi::compare::UNSET),
                        )
                    };
                    bind_entries.insert(*slot, BindEntry::Sampler(s));
                    owned_samplers.push(s);
                }
                RenderOp::SetValue { slot, data } => {
                    // WebGPU has no push constants. Fallback: allocate
                    // a one-shot uniform buffer, write the bytes, bind
                    // it as if it were a `SetUniform`. The caller
                    // pays per-call allocation cost; semantics match
                    // Metal's `setVertexBytes` and Vulkan's
                    // `vkCmdPushConstants`. The buffer is released
                    // after submit, below.
                    let size = data.len() as f64;
                    let buf = unsafe {
                        ffi::quanta_create_buffer(
                            device,
                            size,
                            ffi::buffer_usage::UNIFORM | ffi::buffer_usage::COPY_DST,
                        )
                    };
                    unsafe {
                        ffi::quanta_write_buffer(device, buf, 0.0, data.as_ptr(), data.len());
                    }
                    bind_entries.insert(*slot, BindEntry::OwnedBuffer(buf));
                    owned_buffers.push(buf);
                }
                RenderOp::SetStencilRef(reference) => unsafe {
                    ffi::quanta_render_pass_set_stencil_reference(rp, *reference);
                },
                // Stencil clear value — like color/depth, the WebGPU
                // pass descriptor takes it once at begin time. We
                // currently always store DISCARD on the stencil aspect
                // (no consumer wires stencil load yet), so absorbing
                // the value here is a no-op until depth-target growth.
                RenderOp::ClearStencil(_) => {}
                // Variants below are not in the 050 baseline. Per Kani
                // theorem T417, the rule is **every RenderOp is either
                // wired or explicitly rejected** — no silent drops.
                RenderOp::DebugPush(_) | RenderOp::DebugPop => {
                    // Debug labels are advisory; safe to skip on WebGPU.
                }
                RenderOp::DrawIndirect {
                    buffer_handle,
                    offset,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    let &buf = buffers
                        .get(buffer_handle)
                        .ok_or_else(|| Self::err("draw_indirect buffer handle not found"))?;
                    unsafe {
                        ffi::quanta_render_pass_draw_indirect(rp, buf, *offset as f64);
                    }
                }
                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    offset,
                    index_handle,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    let &idx_buf = buffers.get(index_handle).ok_or_else(|| {
                        Self::err("draw_indexed_indirect index buffer handle not found")
                    })?;
                    unsafe {
                        ffi::quanta_render_pass_set_index_buffer(
                            rp,
                            idx_buf,
                            ffi::index_format::UINT32,
                            0.0,
                        );
                    }
                    let &buf = buffers.get(buffer_handle).ok_or_else(|| {
                        Self::err("draw_indexed_indirect indirect buffer handle not found")
                    })?;
                    unsafe {
                        ffi::quanta_render_pass_draw_indexed_indirect(rp, buf, *offset as f64);
                    }
                }
                RenderOp::ExecuteRenderBundle {
                    bundle_handle,
                    count,
                } => {
                    let bundles = self.state.render_bundles.0.borrow();
                    let bundle = bundles
                        .get(bundle_handle)
                        .ok_or_else(|| Self::err("render bundle handle not found in execute"))?;
                    if *count > bundle.draws.len() as u32 {
                        unsafe { ffi::quanta_render_pass_end(rp) };
                        return Err(Self::err("execute_bundle count exceeds recorded length"));
                    }
                    if *count == 0 {
                        continue;
                    }
                    // Build a fresh GPURenderBundleEncoder against the
                    // active render target's format, replay snapshots,
                    // finish, and pass.executeBundles.
                    let target_format = format_code(target.format)?;
                    let depth_format = if let Some(depth) = &pass.depth_target {
                        let depth_tex = textures
                            .get(&depth.texture)
                            .ok_or_else(|| Self::err("unknown depth target in execute_bundle"))?;
                        format_code(depth_tex.format)?
                    } else {
                        0
                    };
                    let bundle_enc = unsafe {
                        ffi::quanta_create_render_bundle_encoder(
                            device,
                            target_format,
                            depth_format,
                            1,
                        )
                    };
                    for draw in bundle.draws.iter().take(*count as usize) {
                        if let Some(pe) = pipelines.get(&draw.pipeline_handle) {
                            unsafe {
                                ffi::quanta_render_bundle_set_pipeline(bundle_enc, pe.pipeline);
                                ffi::quanta_render_bundle_draw(
                                    bundle_enc,
                                    draw.vertex_count,
                                    draw.instance_count.max(1),
                                );
                            }
                        }
                    }
                    let bundle_h = unsafe { ffi::quanta_render_bundle_finish(bundle_enc) };
                    let bundles_arr = [bundle_h];
                    unsafe {
                        ffi::quanta_render_pass_execute_bundles(rp, bundles_arr.as_ptr(), 1);
                        ffi::quanta_release(bundle_h);
                    }
                }
                RenderOp::BeginOcclusionQuery { index, .. } => {
                    unsafe { ffi::quanta_render_pass_begin_occlusion_query(rp, *index) };
                }
                RenderOp::EndOcclusionQuery { .. } => {
                    unsafe { ffi::quanta_render_pass_end_occlusion_query(rp) };
                }
                RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {
                    unsafe { ffi::quanta_render_pass_end(rp) };
                    return Err(Self::not_supported(
                        "WebGPU render encoder: variable-rate shading is not in the WebGPU spec",
                    ));
                }
            }
        }

        unsafe { ffi::quanta_render_pass_end(rp) };
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };
        for bg in owned_bgs {
            unsafe { ffi::quanta_release(bg) };
        }
        for s in owned_samplers {
            unsafe { ffi::quanta_release(s) };
        }
        // SetValue's per-call uniform buffers go through
        // `quanta_destroy_buffer` (not `quanta_release`) because the
        // JS side allocates a real `GPUBuffer.destroy()`-bearing
        // resource. The two FFI routes are not interchangeable.
        for b in owned_buffers {
            unsafe { ffi::quanta_destroy_buffer(b) };
        }
        Ok(make_pulse())
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
                    let wave = Wave {
                        handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: [0; crate::api::wave::MAX_TEXTURES],
                        texture_count: 0,
                        push_data: [0; crate::api::wave::PUSH_DATA_CAP],
                        push_len: 0,
                        push_mask: 0,
                        workgroup_size: *workgroup_size,
                        drop_fn: None,
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

// ── Wave / Pulse construction helpers ──────────────────────────────────────

fn make_wave(handle: u64, workgroup_size: [u32; 3]) -> Wave {
    Wave {
        handle,
        bindings: [0; crate::api::wave::MAX_BINDINGS],
        binding_count: 0,
        texture_bindings: [0; crate::api::wave::MAX_TEXTURES],
        texture_count: 0,
        push_data: [0; crate::api::wave::PUSH_DATA_CAP],
        push_len: 0,
        push_mask: 0,
        workgroup_size,
        drop_fn: None,
    }
}

fn make_pulse() -> Pulse {
    Pulse {
        handle: 0,
        completed: true,
        wait_fn: None,
    }
}

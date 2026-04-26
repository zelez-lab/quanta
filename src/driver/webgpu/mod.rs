//! WebGPU driver — browser-only.
//!
//! Step 050 + step 079 of the roadmap. Lets Quanta kernels run inside a
//! browser via WebAssembly + the browser's WebGPU API. Native (non-wasm)
//! targets continue to use Metal / Vulkan / CPU backends.
//!
//! ## Architecture
//!
//! - `ffi.rs` — hand-written `#[wasm_bindgen]` extern blocks covering only
//!   the WebGPU surface Quanta uses (~200 lines we own and audit).
//! - `state.rs` — handle table mapping `u64` handles to `GPUBuffer` /
//!   `GPUComputePipeline` JsValues. Single-threaded (wasm32 only).
//! - `mod.rs` (this file) — `WebgpuDevice: GpuDevice` impl that translates
//!   trait calls into FFI calls and JIT-compiles WGSL via `quanta-ir`.
//!
//! ## Sync ↔ async impedance
//!
//! WebGPU's JS API is async for: `requestAdapter`, `requestDevice`,
//! `mapAsync` (buffer read-back), `onSubmittedWorkDone` (completion).
//! Synchronous for: buffer/encoder/pipeline create, `dispatchWorkgroups`,
//! `submit`, `writeBuffer`. The browser cannot block its event loop, so
//! `pulse_wait` and `field_read_bytes` (sync trait methods) are returned as
//! errors that direct callers to the public extension methods
//! [`WebgpuDevice::pulse_wait_async`] / [`WebgpuDevice::field_read_bytes_async`].

mod ffi;
mod state;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::cell::RefCell;

use js_sys::{Array, Object, Reflect, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, Format, FormatCaps, Gpu, GpuDevice as QGpuDevice, Pipeline, Pulse,
    QuantaError, RenderPass, Texture, TextureDesc, Vendor, Wave,
};

use ffi::{buffer_usage, map_mode};
use state::{SendCell, State, WaveEntry};

/// WebGPU device — sits behind `target_arch = "wasm32"` and the `webgpu`
/// feature.
pub struct WebgpuDevice {
    caps: Caps,
    device: SendCell<Option<ffi::GpuDevice>>,
    state: State,
}

impl WebgpuDevice {
    fn err(msg: &'static str) -> QuantaError {
        QuantaError::invalid_param(msg)
    }

    fn err_owned(msg: String) -> QuantaError {
        QuantaError::compilation_failed(Box::leak(msg.into_boxed_str()))
    }

    fn dev(&self) -> Result<core::cell::Ref<'_, Option<ffi::GpuDevice>>, QuantaError> {
        let r = self.device.0.borrow();
        if r.is_none() {
            return Err(Self::err("WebGPU device not initialized — call init_async"));
        }
        Ok(r)
    }

    /// Like [`dev`], but returns a cloned handle so the borrow can be
    /// released before crossing an `await` point. wasm_bindgen extern types
    /// can be re-anchored from a `JsValue` view of the same underlying JS
    /// object — that's the cheap "clone" we use here.
    fn dev_clone(&self) -> Result<ffi::GpuDevice, QuantaError> {
        let r = self.device.0.borrow();
        let d = r
            .as_ref()
            .ok_or_else(|| Self::err("WebGPU device not initialized — call init_async"))?;
        let js: JsValue = d.into();
        Ok(js.unchecked_into())
    }
}

// ── Async init ──────────────────────────────────────────────────────────────

impl WebgpuDevice {
    /// Acquire the typed WebGPU device. Use this when you need access to
    /// the async extension methods (`field_read_bytes_async`,
    /// `pulse_wait_async`); use [`init_async`] for the dyn-trait path that
    /// fits Quanta's standard `Gpu` wrapper.
    pub async fn new_async() -> Result<Self, QuantaError> {
        let gpu = ffi::gpu().map_err(|_| Self::err("navigator.gpu unavailable"))?;
        let adapter_js: JsValue = JsFuture::from(gpu.request_adapter())
            .await
            .map_err(|_| Self::err("requestAdapter failed"))?;
        let adapter: ffi::GpuAdapter = adapter_js
            .dyn_into()
            .map_err(|_| Self::err("adapter not GPUAdapter"))?;

        let device_js: JsValue = JsFuture::from(adapter.request_device())
            .await
            .map_err(|_| Self::err("requestDevice failed"))?;
        let device: ffi::GpuDevice = device_js
            .dyn_into()
            .map_err(|_| Self::err("device not GPUDevice"))?;

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
            device: SendCell(RefCell::new(Some(device))),
            state: State::new(),
        })
    }
}

/// Initialize a WebGPU device wrapped as a `Gpu`. Async because the browser
/// surfaces device acquisition only via Promises. Call this once at app
/// start.
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
        let device = self.dev_clone()?;

        let staging = create_buffer(
            &device,
            size as u64,
            buffer_usage::COPY_DST | buffer_usage::MAP_READ,
        );

        // Build the encoder under a short-lived borrow, then submit and
        // release the borrow before awaiting `mapAsync`.
        {
            let buffers = self.state.buffers.0.borrow();
            let src = buffers
                .get(&handle)
                .ok_or_else(|| Self::err("unknown buffer handle"))?;
            let encoder = device.create_command_encoder();
            encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size as u64);
            let cmd = encoder.finish();
            let arr = Array::new();
            arr.push(&cmd);
            device.queue().submit(&arr);
        }

        JsFuture::from(staging.map_async(map_mode::READ))
            .await
            .map_err(|_| Self::err("mapAsync failed"))?;
        let mapped = staging.get_mapped_range();
        let view = Uint8Array::new(&mapped);
        let mut out = alloc::vec![0u8; size];
        view.copy_to(&mut out[..]);
        staging.unmap();
        staging.destroy();
        Ok(out)
    }

    /// Async sibling of [`pulse_wait`]: awaits
    /// `device.queue.onSubmittedWorkDone()` for the pulse's submission.
    pub async fn pulse_wait_async(&self, _pulse: &mut Pulse) -> Result<(), QuantaError> {
        let device = self.dev_clone()?;
        JsFuture::from(device.queue().on_submitted_work_done())
            .await
            .map_err(|_| Self::err("onSubmittedWorkDone failed"))?;
        Ok(())
    }

    /// Async sibling of [`texture_read`]: copies the texture to a staging
    /// buffer, awaits `mapAsync`, and returns the pixel bytes (tightly
    /// packed in the texture's native row stride).
    pub async fn texture_read_async(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let device = self.dev_clone()?;
        let (tex_clone, view_dims, bytes_per_row, height, tight_row) = {
            let textures = self.state.textures.0.borrow();
            let entry = textures
                .get(&texture.handle)
                .ok_or_else(|| Self::err("unknown texture handle"))?;
            // wasm_bindgen extern types are JsValue-shaped; recreate a
            // typed handle from the JsValue view to drop the borrow.
            let js: JsValue = (&entry.texture).into();
            let cloned: ffi::GpuTexture = js.unchecked_into();
            (
                cloned,
                (entry.width, entry.height),
                entry.bytes_per_row,
                entry.height,
                entry.width * entry.format.bytes_per_pixel() as u32,
            )
        };

        let staging = create_buffer(
            &device,
            (bytes_per_row as u64) * (height as u64),
            buffer_usage::COPY_DST | buffer_usage::MAP_READ,
        );

        let src = Object::new();
        set(&src, "texture", &tex_clone);
        let dst = Object::new();
        set(&dst, "buffer", &staging);
        set(
            &dst,
            "bytesPerRow",
            &JsValue::from_f64(bytes_per_row as f64),
        );
        set(&dst, "rowsPerImage", &JsValue::from_f64(height as f64));
        let size = Object::new();
        set(&size, "width", &JsValue::from_f64(view_dims.0 as f64));
        set(&size, "height", &JsValue::from_f64(view_dims.1 as f64));
        set(&size, "depthOrArrayLayers", &JsValue::from_f64(1.0));

        let encoder = device.create_command_encoder();
        encoder.copy_texture_to_buffer(&src, &dst, &size);
        let cmd = encoder.finish();
        let arr = Array::new();
        arr.push(&cmd);
        device.queue().submit(&arr);

        JsFuture::from(staging.map_async(map_mode::READ))
            .await
            .map_err(|_| Self::err("texture mapAsync failed"))?;
        let mapped = staging.get_mapped_range();
        let view = Uint8Array::new(&mapped);
        let total = (bytes_per_row as usize) * (height as usize);
        let mut padded = alloc::vec![0u8; total];
        view.copy_to(&mut padded[..]);
        staging.unmap();
        staging.destroy();

        // Strip row padding back to the tight row width that callers expect.
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

// ── Helpers ────────────────────────────────────────────────────────────────

fn create_buffer(device: &ffi::GpuDevice, size: u64, usage: u32) -> ffi::GpuBuffer {
    let desc = Object::new();
    set(&desc, "size", &JsValue::from_f64(size as f64));
    set(&desc, "usage", &JsValue::from_f64(usage as f64));
    device.create_buffer(&desc)
}

fn set(obj: &Object, key: &str, value: &JsValue) {
    Reflect::set(obj, &JsValue::from_str(key), value).expect("Reflect::set on Object");
}

fn wgpu_format(f: Format) -> Result<&'static str, QuantaError> {
    Ok(match f {
        Format::RGBA8 => "rgba8unorm",
        Format::BGRA8 => "bgra8unorm",
        Format::R8 => "r8unorm",
        Format::R16Float => "r16float",
        Format::R32Float => "r32float",
        Format::RG32Float => "rg32float",
        Format::RGBA16Float => "rgba16float",
        Format::RGBA32Float => "rgba32float",
        Format::Depth32Float => "depth32float",
        // Compressed formats are not supported by the 050 baseline; surface
        // a clear error rather than returning an arbitrary mapping.
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

fn filter_name(f: crate::render_pass::Filter) -> &'static str {
    match f {
        crate::render_pass::Filter::Nearest => "nearest",
        crate::render_pass::Filter::Linear => "linear",
    }
}

fn address_name(a: crate::render_pass::AddressMode) -> &'static str {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => "clamp-to-edge",
        crate::render_pass::AddressMode::Repeat => "repeat",
        crate::render_pass::AddressMode::MirrorRepeat => "mirror-repeat",
    }
}

fn compare_name(c: crate::CompareOp) -> &'static str {
    match c {
        crate::CompareOp::Never => "never",
        crate::CompareOp::Less => "less",
        crate::CompareOp::Equal => "equal",
        crate::CompareOp::LessEqual => "less-equal",
        crate::CompareOp::Greater => "greater",
        crate::CompareOp::NotEqual => "not-equal",
        crate::CompareOp::GreaterEqual => "greater-equal",
        crate::CompareOp::Always => "always",
    }
}

fn compare_name_internal(c: crate::pipeline::CompareFunc) -> &'static str {
    match c {
        crate::pipeline::CompareFunc::Never => "never",
        crate::pipeline::CompareFunc::Less => "less",
        crate::pipeline::CompareFunc::Equal => "equal",
        crate::pipeline::CompareFunc::LessEqual => "less-equal",
        crate::pipeline::CompareFunc::Greater => "greater",
        crate::pipeline::CompareFunc::NotEqual => "not-equal",
        crate::pipeline::CompareFunc::GreaterEqual => "greater-equal",
        crate::pipeline::CompareFunc::Always => "always",
    }
}

fn attribute_format(f: crate::pipeline::AttributeFormat) -> &'static str {
    use crate::pipeline::AttributeFormat as A;
    match f {
        A::Float => "float32",
        A::Float2 => "float32x2",
        A::Float3 => "float32x3",
        A::Float4 => "float32x4",
        A::Int => "sint32",
        A::Int2 => "sint32x2",
        A::Int3 => "sint32x3",
        A::Int4 => "sint32x4",
        A::UInt => "uint32",
        A::UInt2 => "uint32x2",
        A::UInt3 => "uint32x3",
        A::UInt4 => "uint32x4",
        A::UByte4Norm => "unorm8x4",
    }
}

fn primitive_topology(p: crate::pipeline::Primitive) -> &'static str {
    use crate::pipeline::Primitive as P;
    match p {
        P::Point => "point-list",
        P::Line => "line-list",
        P::LineStrip => "line-strip",
        P::Triangle => "triangle-list",
        P::TriangleStrip => "triangle-strip",
    }
}

fn cull_mode(c: crate::pipeline::CullMode) -> &'static str {
    use crate::pipeline::CullMode as C;
    match c {
        C::None => "none",
        C::Front => "front",
        C::Back => "back",
    }
}

fn blend_factor(f: crate::pipeline::BlendFactor) -> &'static str {
    use crate::pipeline::BlendFactor as F;
    match f {
        F::Zero => "zero",
        F::One => "one",
        F::SrcAlpha => "src-alpha",
        F::OneMinusSrcAlpha => "one-minus-src-alpha",
        F::DstAlpha => "dst-alpha",
        F::OneMinusDstAlpha => "one-minus-dst-alpha",
        F::SrcColor => "src",
        F::OneMinusSrcColor => "one-minus-src",
        F::DstColor => "dst",
        F::OneMinusDstColor => "one-minus-dst",
    }
}

fn blend_op(o: crate::pipeline::BlendOp) -> &'static str {
    use crate::pipeline::BlendOp as O;
    match o {
        O::Add => "add",
        O::Subtract => "subtract",
        O::ReverseSubtract => "reverse-subtract",
        O::Min => "min",
        O::Max => "max",
    }
}

// ── GpuDevice impl ──────────────────────────────────────────────────────────

impl QGpuDevice for WebgpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // ── Buffers ────────────────────────────────────────────────────────────

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        // Map quanta's FieldUsage flag bits onto WebGPU buffer usage flags.
        // We always include COPY_SRC + COPY_DST so the buffer can participate
        // in staging-buffer round-trips for read-back.
        let mut wgpu_usage = buffer_usage::COPY_SRC | buffer_usage::COPY_DST;
        if usage.has(FieldUsage::UNIFORM) {
            wgpu_usage |= buffer_usage::UNIFORM;
        } else {
            // Compute / render fields use storage buffers under WebGPU.
            wgpu_usage |= buffer_usage::STORAGE;
        }

        let buf = create_buffer(device, size as u64, wgpu_usage);
        let handle = self.state.alloc_handle();
        self.state.buffers.0.borrow_mut().insert(handle, buf);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        if let Some(buf) = self.state.buffers.0.borrow_mut().remove(&handle) {
            buf.destroy();
        }
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();
        let buffers = self.state.buffers.0.borrow();
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| Self::err("unknown buffer handle"))?;
        device.queue().write_buffer(buf, 0, data);
        Ok(())
    }

    fn field_read_bytes(&self, _handle: u64, _size: usize) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err(
            "field_read_bytes is async-only on WebGPU; use field_read_bytes_async",
        ))
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();
        let buffers = self.state.buffers.0.borrow();
        let s = buffers.get(&src).ok_or_else(|| Self::err("src missing"))?;
        let d = buffers.get(&dst).ok_or_else(|| Self::err("dst missing"))?;
        let encoder = device.create_command_encoder();
        encoder.copy_buffer_to_buffer(s, 0, d, 0, size as u64);
        let cmd = encoder.finish();
        let arr = Array::new();
        arr.push(&cmd);
        device.queue().submit(&arr);
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

        let wgsl = quanta_ir::emit_wgsl::emit_wgsl_jit(&kernel)
            .map_err(|e| Self::err_owned(format!("emit_wgsl_jit: {}", e)))?;

        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        // createShaderModule({ code: wgsl })
        let mod_desc = Object::new();
        set(&mod_desc, "code", &JsValue::from_str(&wgsl));
        let module = device.create_shader_module(&mod_desc);

        // createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
        let compute = Object::new();
        set(&compute, "module", &module);
        set(&compute, "entryPoint", &JsValue::from_str(&kernel.name));
        let pipe_desc = Object::new();
        set(&pipe_desc, "layout", &JsValue::from_str("auto"));
        set(&pipe_desc, "compute", &compute);
        let pipeline = device.create_compute_pipeline(&pipe_desc);

        let layout = pipeline.get_bind_group_layout(0);

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

        // Re-construct a Wave matching the public API shape.
        Ok(make_wave(handle, kernel.workgroup_size))
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();
        let mut waves = self.state.waves.0.borrow_mut();
        let entry = waves
            .get_mut(&wave.handle)
            .ok_or_else(|| Self::err("unknown wave handle"))?;

        // Build a fresh bind group from the wave's current bindings. WebGPU
        // bind groups are immutable; we create one per dispatch — the cost
        // is small (allocator only) compared to the dispatch itself.
        let entries = Array::new();
        let buffers = self.state.buffers.0.borrow();
        for (slot_idx, &buf_handle) in wave.bindings.iter().enumerate() {
            if buf_handle == 0 {
                continue;
            }
            let buf = buffers
                .get(&buf_handle)
                .ok_or_else(|| Self::err("bound buffer not found"))?;

            let resource = Object::new();
            set(&resource, "buffer", buf);

            let entry_obj = Object::new();
            set(&entry_obj, "binding", &JsValue::from_f64(slot_idx as f64));
            set(&entry_obj, "resource", &resource);
            entries.push(&entry_obj);
            entry.bindings.insert(slot_idx as u32, buf_handle);
        }
        let bg_desc = Object::new();
        set(&bg_desc, "layout", &entry.layout);
        set(&bg_desc, "entries", &entries);
        let bind_group = device.create_bind_group(&bg_desc);

        let encoder = device.create_command_encoder();
        let pass = encoder.begin_compute_pass();
        pass.set_pipeline(&entry.pipeline);
        pass.set_bind_group(0, &bind_group);
        pass.dispatch_workgroups(groups[0], groups[1].max(1), groups[2].max(1));
        pass.end();
        let cmd = encoder.finish();
        let arr = Array::new();
        arr.push(&cmd);
        device.queue().submit(&arr);

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
        // No way to query without spinning the JS event loop; conservatively
        // say "not done" and let the caller use pulse_wait_async.
        false
    }

    // ── Textures ───────────────────────────────────────────────────────────

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        let format = wgpu_format(desc.format)?;
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

        let size_obj = Object::new();
        set(&size_obj, "width", &JsValue::from_f64(desc.width as f64));
        set(&size_obj, "height", &JsValue::from_f64(desc.height as f64));
        set(
            &size_obj,
            "depthOrArrayLayers",
            &JsValue::from_f64(desc.array_length.max(1) as f64),
        );

        let tex_desc = Object::new();
        set(&tex_desc, "size", &size_obj);
        set(&tex_desc, "format", &JsValue::from_str(format));
        set(&tex_desc, "usage", &JsValue::from_f64(usage as f64));
        set(
            &tex_desc,
            "sampleCount",
            &JsValue::from_f64(desc.sample_count.max(1) as f64),
        );
        set(
            &tex_desc,
            "mipLevelCount",
            &JsValue::from_f64(desc.mip_levels.max(1) as f64),
        );

        let tex = device.create_texture(&tex_desc);
        let view = tex.create_view();
        let bytes_per_row = desc.width * desc.format.bytes_per_pixel() as u32;
        // WebGPU requires bytesPerRow to be a multiple of 256 for
        // copyTextureToBuffer; round up here so reads work even for narrow
        // textures. writeTexture does not impose this alignment.
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
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();
        let textures = self.state.textures.0.borrow();
        let entry = textures
            .get(&texture.handle)
            .ok_or_else(|| Self::err("unknown texture handle"))?;

        let dst = Object::new();
        set(&dst, "texture", &entry.texture);

        let layout = Object::new();
        let row = entry.width * entry.format.bytes_per_pixel() as u32;
        set(&layout, "offset", &JsValue::from_f64(0.0));
        set(&layout, "bytesPerRow", &JsValue::from_f64(row as f64));
        set(
            &layout,
            "rowsPerImage",
            &JsValue::from_f64(entry.height as f64),
        );

        let size = Object::new();
        set(&size, "width", &JsValue::from_f64(entry.width as f64));
        set(&size, "height", &JsValue::from_f64(entry.height as f64));
        set(&size, "depthOrArrayLayers", &JsValue::from_f64(1.0));

        device.queue().write_texture(&dst, data, &layout, &size);
        Ok(())
    }

    fn texture_read(&self, _texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        Err(Self::err(
            "texture_read is async-only on WebGPU; use texture_read_async",
        ))
    }

    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        let s = Object::new();
        set(
            &s,
            "magFilter",
            &JsValue::from_str(filter_name(desc.mag_filter)),
        );
        set(
            &s,
            "minFilter",
            &JsValue::from_str(filter_name(desc.min_filter)),
        );
        set(
            &s,
            "mipmapFilter",
            &JsValue::from_str(filter_name(desc.mip_filter)),
        );
        set(
            &s,
            "addressModeU",
            &JsValue::from_str(address_name(desc.address_u)),
        );
        set(
            &s,
            "addressModeV",
            &JsValue::from_str(address_name(desc.address_v)),
        );
        if desc.max_anisotropy > 1 {
            set(
                &s,
                "maxAnisotropy",
                &JsValue::from_f64(desc.max_anisotropy as f64),
            );
        }
        if let Some(cmp) = desc.compare {
            set(&s, "compare", &JsValue::from_str(compare_name(cmp)));
        }

        let sampler = device.create_sampler(&s);
        let handle = self.state.alloc_handle();
        self.state.samplers.0.borrow_mut().insert(handle, sampler);

        Ok(crate::Sampler {
            handle,
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        // Mipmap generation requires a per-format reduction pass — typically
        // a small WGSL kernel that downsamples 4 texels into 1. Not part of
        // the 050 baseline; tracked separately.
        Err(Self::err("WebGPU mipmap generation pending"))
    }

    // ── Render path ────────────────────────────────────────────────────────

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        // The PipelineDesc carries `vertex` + `fragment` slots (or a combined
        // `source`) holding shader bytes. For WebGPU the bytes must be UTF-8
        // WGSL — the proc macro emits this already via
        // `quanta_ir::emit_wgsl::emit_vertex_shader/emit_fragment_shader`.
        let combined = desc.source;
        let vs_src = combined.unwrap_or(desc.vertex);
        let fs_src = combined.unwrap_or(desc.fragment);
        let vs_text = core::str::from_utf8(vs_src)
            .map_err(|_| Self::err("vertex shader is not valid UTF-8 WGSL"))?;
        let fs_text = core::str::from_utf8(fs_src)
            .map_err(|_| Self::err("fragment shader is not valid UTF-8 WGSL"))?;

        let vs_desc = Object::new();
        set(&vs_desc, "code", &JsValue::from_str(vs_text));
        let vs_module = device.create_shader_module(&vs_desc);

        let fs_module = if combined.is_some() || vs_text == fs_text {
            // Single combined source — reuse the same module for both stages.
            // wasm_bindgen extern types don't auto-derive Clone; recreate a
            // typed handle from the JsValue view of the same JS module.
            let js: JsValue = (&vs_module).into();
            js.unchecked_into::<ffi::GpuShaderModule>()
        } else {
            let fs_desc = Object::new();
            set(&fs_desc, "code", &JsValue::from_str(fs_text));
            device.create_shader_module(&fs_desc)
        };

        // Vertex stage descriptor.
        let vs_stage = Object::new();
        set(&vs_stage, "module", &vs_module);
        set(
            &vs_stage,
            "entryPoint",
            &JsValue::from_str(desc.vertex_entry),
        );
        if !desc.vertex_layouts.is_empty() {
            let buffers = Array::new();
            for layout in desc.vertex_layouts.iter() {
                let buf_desc = Object::new();
                set(
                    &buf_desc,
                    "arrayStride",
                    &JsValue::from_f64(layout.stride as f64),
                );
                set(
                    &buf_desc,
                    "stepMode",
                    &JsValue::from_str(match layout.step {
                        crate::pipeline::StepMode::Vertex => "vertex",
                        crate::pipeline::StepMode::Instance => "instance",
                    }),
                );
                let attrs = Array::new();
                for a in &layout.attributes {
                    let att = Object::new();
                    set(
                        &att,
                        "format",
                        &JsValue::from_str(attribute_format(a.format)),
                    );
                    set(&att, "offset", &JsValue::from_f64(a.offset as f64));
                    set(
                        &att,
                        "shaderLocation",
                        &JsValue::from_f64(a.location as f64),
                    );
                    attrs.push(&att);
                }
                set(&buf_desc, "attributes", &attrs);
                buffers.push(&buf_desc);
            }
            set(&vs_stage, "buffers", &buffers);
        }

        // Fragment stage descriptor with one or more color targets.
        let fs_stage = Object::new();
        set(&fs_stage, "module", &fs_module);
        set(
            &fs_stage,
            "entryPoint",
            &JsValue::from_str(desc.fragment_entry),
        );
        let targets = Array::new();
        for (i, fmt) in desc.color_formats.iter().enumerate() {
            let target = Object::new();
            set(&target, "format", &JsValue::from_str(wgpu_format(*fmt)?));
            let blend_state = desc
                .blend_states
                .get(i)
                .copied()
                .or_else(|| desc.blend_states.last().copied())
                .unwrap_or(desc.blend);
            if blend_state.enabled {
                let blend = Object::new();
                let color = Object::new();
                set(
                    &color,
                    "srcFactor",
                    &JsValue::from_str(blend_factor(blend_state.src_rgb)),
                );
                set(
                    &color,
                    "dstFactor",
                    &JsValue::from_str(blend_factor(blend_state.dst_rgb)),
                );
                set(
                    &color,
                    "operation",
                    &JsValue::from_str(blend_op(blend_state.op_rgb)),
                );
                let alpha = Object::new();
                set(
                    &alpha,
                    "srcFactor",
                    &JsValue::from_str(blend_factor(blend_state.src_alpha)),
                );
                set(
                    &alpha,
                    "dstFactor",
                    &JsValue::from_str(blend_factor(blend_state.dst_alpha)),
                );
                set(
                    &alpha,
                    "operation",
                    &JsValue::from_str(blend_op(blend_state.op_alpha)),
                );
                set(&blend, "color", &color);
                set(&blend, "alpha", &alpha);
                set(&target, "blend", &blend);
            }
            targets.push(&target);
        }
        set(&fs_stage, "targets", &targets);

        let primitive = Object::new();
        set(
            &primitive,
            "topology",
            &JsValue::from_str(primitive_topology(desc.primitive)),
        );
        set(
            &primitive,
            "cullMode",
            &JsValue::from_str(cull_mode(desc.cull_mode)),
        );

        let pipe_desc = Object::new();
        set(&pipe_desc, "layout", &JsValue::from_str("auto"));
        set(&pipe_desc, "vertex", &vs_stage);
        set(&pipe_desc, "fragment", &fs_stage);
        set(&pipe_desc, "primitive", &primitive);
        set(&pipe_desc, "multisample", &{
            let m = Object::new();
            set(
                &m,
                "count",
                &JsValue::from_f64(desc.sample_count.max(1) as f64),
            );
            m.into()
        });
        if let Some(depth_fmt) = desc.depth_format {
            let ds = Object::new();
            set(&ds, "format", &JsValue::from_str(wgpu_format(depth_fmt)?));
            set(
                &ds,
                "depthWriteEnabled",
                &JsValue::from_bool(desc.depth_stencil.depth_write),
            );
            set(
                &ds,
                "depthCompare",
                &JsValue::from_str(compare_name_internal(desc.depth_stencil.depth_compare)),
            );
            set(&pipe_desc, "depthStencil", &ds);
        }

        let pipeline = device.create_render_pipeline(&pipe_desc);
        let layout = pipeline.render_get_bind_group_layout(0);

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

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        // The WebGPU render-pass encoder is created lazily inside `render_end`
        // because the pass needs the recorded ops + the color target up front.
        // We just stash the target in a reserved op slot — here, in
        // `RenderPass::color_targets`.
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

    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let dev_ref = self.dev()?;
        let device = dev_ref.as_ref().unwrap();

        let textures = self.state.textures.0.borrow();
        let target = textures
            .get(&pass.handle)
            .ok_or_else(|| Self::err("unknown render target"))?;

        // Build the render-pass descriptor: one color attachment for now.
        let colors = Array::new();
        let attach = Object::new();
        set(&attach, "view", &target.view);
        // load_op default for Quanta is Clear(CLEAR); folding LoadOp/StoreOp
        // out of `ColorTarget` is left as a future refinement.
        set(&attach, "loadOp", &JsValue::from_str("clear"));
        set(&attach, "storeOp", &JsValue::from_str("store"));
        let clear = Object::new();
        set(&clear, "r", &JsValue::from_f64(0.0));
        set(&clear, "g", &JsValue::from_f64(0.0));
        set(&clear, "b", &JsValue::from_f64(0.0));
        set(&clear, "a", &JsValue::from_f64(0.0));
        set(&attach, "clearValue", &clear);
        colors.push(&attach);

        // Override clear color from the recorded ops if present.
        for op in &pass.ops {
            if let crate::render_pass::RenderOp::Clear(color) = op {
                let clear = Object::new();
                set(&clear, "r", &JsValue::from_f64(color.r as f64));
                set(&clear, "g", &JsValue::from_f64(color.g as f64));
                set(&clear, "b", &JsValue::from_f64(color.b as f64));
                set(&clear, "a", &JsValue::from_f64(color.a as f64));
                set(&attach, "clearValue", &clear);
            }
        }

        let pass_desc = Object::new();
        set(&pass_desc, "colorAttachments", &colors);

        let encoder = device.create_command_encoder();
        let rp = encoder.begin_render_pass(&pass_desc);

        // Replay the recorded ops.
        let pipelines = self.state.pipelines.0.borrow();
        let buffers = self.state.buffers.0.borrow();
        let mut current_pipeline: Option<&state::PipelineEntry> = None;
        let mut bind_entries: alloc::collections::BTreeMap<u32, JsValue> =
            alloc::collections::BTreeMap::new();

        for op in &pass.ops {
            use crate::render_pass::RenderOp;
            match op {
                RenderOp::SetPipeline(handle) => {
                    let entry = pipelines
                        .get(handle)
                        .ok_or_else(|| Self::err("unknown pipeline"))?;
                    rp.rp_set_pipeline(&entry.pipeline);
                    current_pipeline = Some(entry);
                }
                RenderOp::BindVertices {
                    slot,
                    handle,
                    offset,
                } => {
                    let buf = buffers.get(handle).ok_or_else(|| Self::err("vbuf"))?;
                    rp.rp_set_vertex_buffer(*slot, buf, *offset);
                }
                RenderOp::BindIndices { handle, offset } => {
                    let buf = buffers.get(handle).ok_or_else(|| Self::err("ibuf"))?;
                    rp.rp_set_index_buffer(buf, "uint32", *offset);
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    let buf = buffers.get(handle).ok_or_else(|| Self::err("ubuf"))?;
                    let resource = Object::new();
                    set(&resource, "buffer", buf);
                    let entry = Object::new();
                    set(&entry, "binding", &JsValue::from_f64(*slot as f64));
                    set(&entry, "resource", &resource);
                    bind_entries.insert(*slot, entry.into());
                }
                RenderOp::Clear(_) => { /* handled above as clearValue */ }
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    if !bind_entries.is_empty()
                        && let Some(p) = current_pipeline
                    {
                        let entries = Array::new();
                        for v in bind_entries.values() {
                            entries.push(v);
                        }
                        let bg_desc = Object::new();
                        set(&bg_desc, "layout", &p.layout);
                        set(&bg_desc, "entries", &entries);
                        let bg = device.create_bind_group(&bg_desc);
                        rp.rp_set_bind_group(0, &bg);
                        bind_entries.clear();
                    }
                    rp.draw(*vertex_count, *instance_count);
                }
                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    if !bind_entries.is_empty()
                        && let Some(p) = current_pipeline
                    {
                        let entries = Array::new();
                        for v in bind_entries.values() {
                            entries.push(v);
                        }
                        let bg_desc = Object::new();
                        set(&bg_desc, "layout", &p.layout);
                        set(&bg_desc, "entries", &entries);
                        let bg = device.create_bind_group(&bg_desc);
                        rp.rp_set_bind_group(0, &bg);
                        bind_entries.clear();
                    }
                    rp.draw_indexed(*index_count, *instance_count);
                }
                RenderOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    rp.rp_set_viewport(*x, *y, *width, *height, *min_depth, *max_depth);
                }
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => rp.rp_set_scissor(*x, *y, *width, *height),
                _ => {
                    // Other ops (debug labels, occlusion queries, MRT
                    // overrides, indirect draw, shading rate) are not in the
                    // 050 baseline. Silently drop — the surface returns no
                    // error so these calls remain compatible across drivers.
                }
            }
        }

        rp.rp_end();
        let cmd = encoder.finish();
        let arr = Array::new();
        arr.push(&cmd);
        device.queue().submit(&arr);

        Ok(make_pulse())
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

// ── Wave / Pulse construction helpers ──────────────────────────────────────

fn make_wave(handle: u64, workgroup_size: [u32; 3]) -> Wave {
    // Reuse the public Wave struct via mem::zeroed-equivalent — but Wave's
    // fields are pub(crate), so we can build it directly inside the crate.
    // The drop_fn is None: handle ownership is tracked via the state map.
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
        // Submission has already been issued; the pulse is "done" from the
        // caller's perspective unless they explicitly await via
        // `pulse_wait_async`. Sync `pulse_wait` still rejects.
        completed: true,
        wait_fn: None,
    }
}

//! Compute dispatch operations for the WebGPU driver.
//!
//! The JIT path: KernelDef IR → WGSL (via `quanta_ir::emit_wgsl`) →
//! `GPUShaderModule` → `GPUComputePipeline`, dispatched through a
//! compute pass with a per-dispatch bind group. WebGPU rejects
//! pre-compiled binaries, so `wave` errors and only `wave_jit` is
//! functional.

use alloc::format;

use crate::{Pulse, QuantaError, Wave};

use super::ffi;
use super::state::WaveEntry;
use super::{WebgpuDevice, make_pulse, make_wave};

impl WebgpuDevice {
    // ── Compute (JIT path) ─────────────────────────────────────────────────

    pub(super) fn wave_impl(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(Self::err(
            "WebGPU does not accept pre-compiled binaries; use the JIT path",
        ))
    }

    pub(super) fn wave_jit_impl(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
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

    pub(super) fn wave_dispatch_impl(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        // Compute texture bindings (storage/sampled images) are not wired on
        // the WebGPU backend; fail loudly rather than silently ignore them
        // (mirrors the ICB path). `supports_compute_textures()` returns false.
        if wave.texture_count != 0 {
            return Err(Self::err(
                "WebGPU does not support compute texture bindings",
            ));
        }
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

    pub(super) fn wave_dispatch_indirect_impl(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(Self::err("WebGPU indirect dispatch pending"))
    }

    // === Compute-resource lifecycle ===

    /// Destroy a wave: drop its registry entry and release the JS
    /// handles. GPUComputePipeline / GPUShaderModule /
    /// GPUBindGroupLayout have no destroy(); releasing the handles
    /// lets the GC collect them.
    #[cfg(feature = "compute")]
    pub(super) fn wave_destroy_impl(&self, handle: u64) -> Result<(), QuantaError> {
        if let Some(entry) = self.state.waves.0.borrow_mut().remove(&handle) {
            unsafe {
                ffi::quanta_release(entry.layout);
                ffi::quanta_release(entry._shader);
                ffi::quanta_release(entry.pipeline);
            }
        }
        Ok(())
    }
}

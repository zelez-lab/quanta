//! Compute dispatch methods on Gpu (wave, dispatch, batch, async compute).

use alloc::vec::Vec;

use crate::{Batch, Field, Pulse, QuantaError, QueueFamily, QueueType, Wave};

use super::Gpu;

impl Gpu {
    // === Compute ===

    /// Create a wave from a pre-compiled kernel binary.
    ///
    /// Repeat creations of the same bytes on one device return fresh
    /// `Wave` handles over ONE cached driver pipeline (see
    /// [`crate::api::wave_cache`]) — creation cost is paid once per
    /// distinct kernel, and bindings/push state stay per-`Wave`.
    pub fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        #[cfg(feature = "std")]
        {
            self.cached_wave(crate::api::wave_cache::WaveKind::Aot, kernel, |d| {
                d.wave(kernel)
            })
        }
        #[cfg(not(feature = "std"))]
        {
            let mut wave = self.ctx.device.wave(kernel)?;
            wave.device = Some(self.ctx.device.clone());
            Ok(wave)
        }
    }

    /// JIT-compile a kernel from its serialized KernelDef at runtime.
    ///
    /// Used by `#[quanta::kernel(jit)]` — the kernel IR is embedded in the
    /// binary and compiled to the appropriate GPU shader format at first use.
    /// Cached per device by the kernel bytes, like [`Gpu::wave`] — the
    /// sci/ml layers rebuild identical bytes once per op, and only the
    /// first build pays deserialize + emit + pipeline construction.
    pub fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        #[cfg(feature = "std")]
        {
            self.cached_wave(
                crate::api::wave_cache::WaveKind::Jit,
                kernel_def_bytes,
                |d| d.wave_jit(kernel_def_bytes),
            )
        }
        #[cfg(not(feature = "std"))]
        {
            let mut wave = self.ctx.device.wave_jit(kernel_def_bytes)?;
            wave.device = Some(self.ctx.device.clone());
            Ok(wave)
        }
    }

    /// The shared create path behind [`Gpu::wave`] / [`Gpu::wave_jit`]:
    /// hand out a fresh `Wave` over the cached pipeline, or compile
    /// through `create` and adopt the result into the cache. The
    /// compile runs OUTSIDE the cache lock (concurrent first-compiles
    /// of different kernels must not serialize); a lost same-kernel
    /// race keeps the incumbent entry and the loser's wave releases
    /// its own solo pipeline on drop.
    #[cfg(feature = "std")]
    fn cached_wave(
        &self,
        kind: crate::api::wave_cache::WaveKind,
        bytes: &[u8],
        create: impl FnOnce(&alloc::sync::Arc<dyn crate::GpuDevice>) -> Result<Wave, QuantaError>,
    ) -> Result<Wave, QuantaError> {
        use crate::api::types::{MAX_BINDINGS, MAX_TEXTURES, PUSH_DATA_CAP};
        use crate::api::wave_cache::SharedPipeline;

        if let Some((shared, workgroup_size, write_mask)) = self.ctx.wave_cache.get(kind, bytes) {
            return Ok(Wave {
                handle: shared.handle,
                bindings: [0u64; MAX_BINDINGS],
                binding_count: 0,
                texture_bindings: [0u64; MAX_TEXTURES],
                texture_count: 0,
                storage_texture_kinds: [0; 16],
                write_mask,
                push_data: [0u8; PUSH_DATA_CAP],
                push_len: 0,
                push_mask: 0,
                workgroup_size,
                device: Some(self.ctx.device.clone()),
                live: false,
                shared: Some(shared),
            });
        }
        let mut wave = create(&self.ctx.device)?;
        wave.device = Some(self.ctx.device.clone());
        let shared =
            alloc::sync::Arc::new(SharedPipeline::new(wave.handle, self.ctx.device.clone()));
        wave.live = false;
        wave.shared = Some(shared.clone());
        self.ctx
            .wave_cache
            .insert(kind, bytes, shared, wave.workgroup_size, wave.write_mask);
        Ok(wave)
    }

    pub fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        // A submission that bypasses the deferred lane: commit the
        // pending batch first so queue order stays program order (the
        // driver's hazard tracking orders the actual accesses).
        #[cfg(feature = "std")]
        self.ctx.pending.submit_pending()?;
        self.ctx.device.wave_dispatch(wave, groups)
    }

    /// Dispatch a 1D wave over exactly `quarks` threads.
    /// Metal uses dispatchThreads (clips to exact count).
    /// Vulkan uses dispatchGroups with ceil(quarks/workgroup_size[0]).
    ///
    /// Dispatch is **deferred**: the wave is encoded into the shared
    /// per-device batch and the returned pulse's `wait` flushes the
    /// lane (see [`crate::api::deferred`]). On backends without batch
    /// support — and for texture-binding waves, which the lane
    /// declines — the dispatch commits and completes inline (the
    /// returned pulse is already done). Only this 1-D entry point
    /// defers — `wave_dispatch` (explicit groups),
    /// `dispatch_indirect`, and `async_compute_dispatch` always
    /// commit, after ordering themselves against the lane.
    pub fn dispatch(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        #[cfg(feature = "std")]
        {
            if self
                .ctx
                .pending
                .encode(self.device_handle(), wave, quarks)?
            {
                return Ok(crate::api::deferred::lazy_pulse(
                    self.ctx.pending.clone(),
                    self.device_handle().clone(),
                ));
            }
            // Not encodable (batch-less backend, or a texture-binding
            // wave the lane declines): commit any pending work first
            // for ordering, then dispatch eagerly and wait — a dropped
            // pulse leaks nothing and reads stay correct without a
            // flush point.
            self.ctx.pending.submit_pending()?;
            let mut pulse = self.ctx.device.wave_dispatch_threads(wave, quarks)?;
            pulse.wait()?;
            Ok(pulse)
        }
        #[cfg(not(feature = "std"))]
        self.ctx.device.wave_dispatch_threads(wave, quarks)
    }

    /// Dispatch with group counts from a GPU buffer (GPU-driven).
    pub fn dispatch_indirect<T: Copy>(
        &self,
        wave: &Wave,
        buffer: &Field<T>,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        // Bypasses the lane — commit pending work first (see
        // `wave_dispatch`); the args buffer itself may also be a
        // deferred product.
        #[cfg(feature = "std")]
        self.ctx.pending.submit_pending()?;
        self.ctx
            .device
            .wave_dispatch_indirect(wave, buffer.handle(), offset)
    }

    // === Batch dispatch ===

    /// Begin a batch of dispatches. Multiple kernels are encoded into a single
    /// command buffer. Call `pulse()` on the batch to commit all at once.
    /// One commit + one fence instead of N — eliminates per-dispatch overhead.
    pub fn batch(&self) -> Result<Batch, QuantaError> {
        // A user batch is its own submission — commit pending deferred
        // work first so the two cannot interleave out of program order.
        #[cfg(feature = "std")]
        self.ctx.pending.submit_pending()?;
        Ok(Batch::new(
            self.ctx.device.batch_begin()?,
            self.ctx.device.clone(),
        ))
    }

    // === Async compute ===

    /// Whether this device supports a dedicated async compute queue.
    pub fn supports_async_compute(&self) -> bool {
        self.ctx.device.supports_async_compute()
    }

    /// Dispatch a compute wave on the async compute queue.
    pub fn async_compute_dispatch(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        // Cross-queue: a submit alone gives no ordering against the
        // async queue, so complete pending lane work outright before
        // handing anything to it.
        #[cfg(feature = "std")]
        self.ctx.pending.flush_and_wait()?;
        self.ctx.device.async_compute_dispatch(wave, groups)
    }

    // === M3.1: Multi-queue ===

    /// List available queue families on this device.
    pub fn queue_families(&self) -> Vec<QueueFamily> {
        self.ctx.device.queue_families()
    }

    /// Allocate a typed [`PrintfBuffer`](crate::PrintfBuffer)
    /// with the given message capacity. Step 049.
    ///
    /// Backends without printf return `NotSupported`. Use
    /// `cap >= 1`; the typed wrapper rejects 0.
    pub fn printf_buffer(&self, cap: u32) -> Result<crate::PrintfBuffer, QuantaError> {
        if cap == 0 {
            return Err(QuantaError::invalid_param(
                "printf buffer capacity must be >= 1",
            ));
        }
        let handle = self.ctx.device.printf_create(cap)?;
        Ok(crate::PrintfBuffer {
            handle,
            cap,
            device: self.ctx.device.clone(),
            live: true,
        })
    }

    /// Allocate a typed
    /// [`AsyncCopyQueue`](crate::AsyncCopyQueue) for off-graphics
    /// DMA copies. Step 044.
    ///
    /// Backends without a dedicated transfer engine return
    /// `NotSupported` here so user code can fall back to the main
    /// queue.
    pub fn async_copy_queue(&self) -> Result<crate::AsyncCopyQueue, QuantaError> {
        let handle = self.ctx.device.async_copy_create()?;
        Ok(crate::AsyncCopyQueue {
            handle,
            gpu: self.clone(),
            live: true,
        })
    }

    /// Allocate a typed [`Queue`](crate::Queue) for the given
    /// capability tier. Steps 018 + 019.
    ///
    /// Backends that don't expose multi-queue (single-queue
    /// software fallbacks, WebGPU global queue) return
    /// `NotSupported` here so user code can branch.
    pub fn queue(&self, queue_type: QueueType) -> Result<crate::Queue, QuantaError> {
        let handle = self.ctx.device.create_queue(queue_type)?;
        Ok(crate::Queue {
            handle,
            kind: queue_type,
            gpu: self.clone(),
            live: true,
        })
    }

    // === Hot reload ===

    /// Replace a wave's kernel while preserving its bindings and push constants.
    ///
    /// Compiles `kernel` into a new wave, transfers all bindings and push constants
    /// from `wave` to the new wave, then replaces `wave`'s handle.
    pub fn reload_wave(&self, wave: &mut Wave, kernel: &[u8]) -> Result<(), QuantaError> {
        // Deliberately bypasses the wave cache: hot reload feeds
        // freshly edited bytes, the one shape where dedup buys nothing.
        let mut new_wave = self.ctx.device.wave(kernel)?;
        new_wave.device = Some(self.ctx.device.clone());
        new_wave.bindings = wave.bindings;
        new_wave.binding_count = wave.binding_count;
        new_wave.write_mask = wave.write_mask;
        new_wave.texture_bindings = wave.texture_bindings;
        new_wave.texture_count = wave.texture_count;
        new_wave.storage_texture_kinds = wave.storage_texture_kinds;
        new_wave.push_data = wave.push_data;
        new_wave.push_len = wave.push_len;
        new_wave.push_mask = wave.push_mask;
        // Swap: the caller's wave takes the new handle (device attached
        // above); new_wave walks away with the OLD handle + device +
        // live flag, so its drop below releases the old registry entry.
        core::mem::swap(wave, &mut new_wave);
        Ok(())
    }

    // === M5.2: Indirect command buffers (steps 032 + 033) ===

    /// Create a typed [`IndirectCommandBuffer`] with the given
    /// capacity.
    ///
    /// The buffer can hold up to `max_commands` recorded dispatches.
    /// Records past capacity return an error; `Drop` releases the
    /// underlying handle. See
    /// [`IndirectCommandBuffer`](crate::IndirectCommandBuffer) for
    /// the full API.
    pub fn indirect_command_buffer(
        &self,
        max_commands: u32,
    ) -> Result<crate::IndirectCommandBuffer, QuantaError> {
        let handle = self.ctx.device.indirect_buffer_create(max_commands)?;
        Ok(crate::IndirectCommandBuffer {
            handle,
            cap: max_commands,
            recorded: 0,
            gpu: self.clone(),
            live: true,
        })
    }

    // === M5.3: Bindless resources ===

    /// Allocate a typed
    /// [`BindlessTextureArray`](crate::BindlessTextureArray) with
    /// the given capacity. Steps 034 + 035.
    pub fn bindless_textures(&self, cap: u32) -> Result<crate::BindlessTextureArray, QuantaError> {
        let handle = self.ctx.device.bindless_texture_create(cap)?;
        Ok(crate::BindlessTextureArray {
            handle,
            cap,
            device: self.ctx.device.clone(),
            live: true,
        })
    }

    /// Allocate a typed
    /// [`BindlessBufferArray`](crate::BindlessBufferArray) with the
    /// given capacity. Steps 034 + 035.
    pub fn bindless_buffers(&self, cap: u32) -> Result<crate::BindlessBufferArray, QuantaError> {
        let handle = self.ctx.device.bindless_buffer_create(cap)?;
        Ok(crate::BindlessBufferArray {
            handle,
            cap,
            device: self.ctx.device.clone(),
            live: true,
        })
    }
}

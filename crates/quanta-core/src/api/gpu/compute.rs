//! Compute dispatch methods on Gpu (wave, dispatch, batch, async compute).

use alloc::vec::Vec;

use crate::{Batch, Field, Pulse, QuantaError, QueueFamily, QueueType, Wave};

use super::Gpu;

impl Gpu {
    // === Compute ===

    pub fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        let mut wave = self.inner.wave(kernel)?;
        wave.device = Some(self.inner.clone());
        Ok(wave)
    }

    /// JIT-compile a kernel from its serialized KernelDef at runtime.
    ///
    /// Used by `#[quanta::kernel(jit)]` — the kernel IR is embedded in the
    /// binary and compiled to the appropriate GPU shader format at first use.
    pub fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        let mut wave = self.inner.wave_jit(kernel_def_bytes)?;
        wave.device = Some(self.inner.clone());
        Ok(wave)
    }

    pub fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch(wave, groups)
    }

    /// Dispatch a 1D wave over exactly `quarks` threads.
    /// Metal uses dispatchThreads (clips to exact count).
    /// Vulkan uses dispatchGroups with ceil(quarks/workgroup_size[0]).
    pub fn dispatch(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch_threads(wave, quarks)
    }

    /// Dispatch with group counts from a GPU buffer (GPU-driven).
    pub fn dispatch_indirect<T: Copy>(
        &self,
        wave: &Wave,
        buffer: &Field<T>,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.inner
            .wave_dispatch_indirect(wave, buffer.handle(), offset)
    }

    // === Batch dispatch ===

    /// Begin a batch of dispatches. Multiple kernels are encoded into a single
    /// command buffer. Call `pulse()` on the batch to commit all at once.
    /// One commit + one fence instead of N — eliminates per-dispatch overhead.
    pub fn batch(&self) -> Result<Batch, QuantaError> {
        self.inner.batch_begin()
    }

    // === Async compute ===

    /// Whether this device supports a dedicated async compute queue.
    pub fn supports_async_compute(&self) -> bool {
        self.inner.supports_async_compute()
    }

    /// Dispatch a compute wave on the async compute queue.
    pub fn async_compute_dispatch(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        self.inner.async_compute_dispatch(wave, groups)
    }

    // === M3.1: Multi-queue ===

    /// List available queue families on this device.
    pub fn queue_families(&self) -> Vec<QueueFamily> {
        self.inner.queue_families()
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
        let handle = self.inner.printf_create(cap)?;
        Ok(crate::PrintfBuffer {
            handle,
            cap,
            device: self.inner.clone(),
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
        let handle = self.inner.async_copy_create()?;
        Ok(crate::AsyncCopyQueue {
            handle,
            device: self.inner.clone(),
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
        let handle = self.inner.create_queue(queue_type)?;
        Ok(crate::Queue {
            handle,
            kind: queue_type,
            device: self.inner.clone(),
            live: true,
        })
    }

    // === Hot reload ===

    /// Replace a wave's kernel while preserving its bindings and push constants.
    ///
    /// Compiles `kernel` into a new wave, transfers all bindings and push constants
    /// from `wave` to the new wave, then replaces `wave`'s handle.
    pub fn reload_wave(&self, wave: &mut Wave, kernel: &[u8]) -> Result<(), QuantaError> {
        let mut new_wave = self.inner.wave(kernel)?;
        new_wave.device = Some(self.inner.clone());
        new_wave.bindings = wave.bindings;
        new_wave.binding_count = wave.binding_count;
        new_wave.texture_bindings = wave.texture_bindings;
        new_wave.texture_count = wave.texture_count;
        new_wave.f32_storage_texture_mask = wave.f32_storage_texture_mask;
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
        let handle = self.inner.indirect_buffer_create(max_commands)?;
        Ok(crate::IndirectCommandBuffer {
            handle,
            cap: max_commands,
            recorded: 0,
            device: self.inner.clone(),
            live: true,
        })
    }

    // === M5.3: Bindless resources ===

    /// Allocate a typed
    /// [`BindlessTextureArray`](crate::BindlessTextureArray) with
    /// the given capacity. Steps 034 + 035.
    pub fn bindless_textures(&self, cap: u32) -> Result<crate::BindlessTextureArray, QuantaError> {
        let handle = self.inner.bindless_texture_create(cap)?;
        Ok(crate::BindlessTextureArray {
            handle,
            cap,
            device: self.inner.clone(),
            live: true,
        })
    }

    /// Allocate a typed
    /// [`BindlessBufferArray`](crate::BindlessBufferArray) with the
    /// given capacity. Steps 034 + 035.
    pub fn bindless_buffers(&self, cap: u32) -> Result<crate::BindlessBufferArray, QuantaError> {
        let handle = self.inner.bindless_buffer_create(cap)?;
        Ok(crate::BindlessBufferArray {
            handle,
            cap,
            device: self.inner.clone(),
            live: true,
        })
    }
}

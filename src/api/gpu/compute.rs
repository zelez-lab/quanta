//! Compute dispatch methods on Gpu (wave, dispatch, batch, async compute).

use alloc::vec::Vec;

use crate::{Batch, Field, Pipeline, Pulse, QuantaError, QueueFamily, QueueType, Wave};

use super::Gpu;

impl Gpu {
    // === Compute ===

    pub fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave(kernel)
    }

    /// JIT-compile a kernel from its serialized KernelDef at runtime.
    ///
    /// Used by `#[quanta::kernel(jit)]` — the kernel IR is embedded in the
    /// binary and compiled to the appropriate GPU shader format at first use.
    pub fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave_jit(kernel_def_bytes)
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

    /// Create a queue of the given type.
    pub fn create_queue(&self, queue_type: QueueType) -> Result<u64, QuantaError> {
        self.inner.create_queue(queue_type)
    }

    /// Submit a compute dispatch to a specific queue.
    pub fn queue_dispatch(
        &self,
        queue: u64,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        self.inner.queue_dispatch(queue, wave, groups)
    }

    /// Signal a semaphore from a queue.
    pub fn queue_signal(&self, queue: u64, semaphore: u64) -> Result<(), QuantaError> {
        self.inner.queue_signal(queue, semaphore)
    }

    /// Wait on a semaphore before executing more work on a queue.
    pub fn queue_wait(&self, queue: u64, semaphore: u64) -> Result<(), QuantaError> {
        self.inner.queue_wait(queue, semaphore)
    }

    // === Hot reload ===

    /// Replace a wave's kernel while preserving its bindings and push constants.
    ///
    /// Compiles `kernel` into a new wave, transfers all bindings and push constants
    /// from `wave` to the new wave, then replaces `wave`'s handle.
    pub fn reload_wave(&self, wave: &mut Wave, kernel: &[u8]) -> Result<(), QuantaError> {
        let mut new_wave = self.inner.wave(kernel)?;
        new_wave.bindings = wave.bindings;
        new_wave.binding_count = wave.binding_count;
        new_wave.texture_bindings = wave.texture_bindings;
        new_wave.texture_count = wave.texture_count;
        new_wave.push_data = wave.push_data;
        new_wave.push_len = wave.push_len;
        new_wave.push_mask = wave.push_mask;
        // Swap: the old handle gets dropped via new_wave's eventual drop
        core::mem::swap(wave, &mut new_wave);
        Ok(())
    }

    // === M4.2: Mesh shaders ===

    /// Dispatch a mesh shader pipeline.
    pub fn dispatch_mesh(&self, pipeline: &Pipeline, groups: [u32; 3]) -> Result<(), QuantaError> {
        self.inner.dispatch_mesh(pipeline.handle(), groups)
    }

    // === M5.2: Indirect command buffers ===

    /// Create an indirect command buffer (GPU-driven draw/dispatch).
    pub fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        self.inner.indirect_buffer_create(max_commands)
    }

    /// Execute commands from an indirect command buffer.
    pub fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        self.inner.indirect_buffer_execute(handle, count)
    }

    /// Destroy an indirect command buffer.
    pub fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.indirect_buffer_destroy(handle)
    }

    // === M5.3: Bindless resources ===

    /// Create a bindless texture array (all textures accessible by index in shaders).
    pub fn bind_texture_array(&self, textures: &[u64]) -> Result<u64, QuantaError> {
        self.inner.bind_texture_array(textures)
    }

    /// Create a bindless buffer array (all buffers accessible by index in shaders).
    pub fn bind_buffer_array(&self, buffers: &[u64]) -> Result<u64, QuantaError> {
        self.inner.bind_buffer_array(buffers)
    }
}

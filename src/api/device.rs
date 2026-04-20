use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Pipeline, Pulse, QuantaError, RenderPass, ResourceState, Texture,
    TextureDesc, Timeline, Wave,
};

/// Core trait — every GPU driver implements this.
///
/// Methods use raw bytes and handles to keep the trait dyn-compatible.
/// Users interact with the `Gpu` wrapper which provides typed, ergonomic methods.
pub trait GpuDevice {
    // === Device info ===

    fn caps(&self) -> &Caps;

    // === Fields (GPU memory) ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError>;
    fn field_free(&self, handle: u64);
    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError>;
    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError>;
    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError>;

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError>;
    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError>;
    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError>;
    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError>;
    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError>;

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError>;
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError>;
    /// Dispatch with group counts from a GPU buffer (GPU decides grid size).
    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError>;

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError>;
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError>;
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError>;

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError>;
    fn pulse_poll(&self, pulse: &Pulse) -> bool;

    // === Queries ===

    /// Create a timestamp query set.
    fn query_set_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param("queries not supported"))
    }

    /// Read query results.
    fn query_set_read(
        &self,
        _handle: u64,
        _first: u32,
        _count: u32,
    ) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::invalid_param("queries not supported"))
    }

    // === Timestamps ===

    /// Create a timestamp query set with `count` slots.
    fn timestamp_query_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param("timestamps not supported"))
    }

    /// Read timestamp values from a query set.
    fn timestamp_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::invalid_param("timestamps not supported"))
    }

    // === Async compute ===

    /// Whether this device supports a dedicated async compute queue.
    fn supports_async_compute(&self) -> bool {
        false
    }

    /// Dispatch a compute wave on the async compute queue.
    /// Returns immediately; the returned Pulse signals completion.
    fn async_compute_dispatch(
        &self,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param("async compute not supported"))
    }

    // === Timeline semaphores ===

    /// Create a timeline semaphore (monotonic u64 counter for multi-frame sync).
    fn timeline_create(&self) -> Result<Timeline, QuantaError> {
        Err(QuantaError::invalid_param(
            "timeline semaphores not supported",
        ))
    }

    /// Signal a timeline to the given value.
    fn timeline_signal(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "timeline semaphores not supported",
        ))
    }

    /// Block until a timeline reaches at least the given value.
    fn timeline_wait(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "timeline semaphores not supported",
        ))
    }

    // === Barriers ===

    /// Full pipeline barrier — wait for all prior GPU work to complete.
    ///
    /// This is a heavyweight synchronization point. Prefer `barrier_buffer`
    /// or `barrier_texture` for fine-grained transitions when possible.
    fn barrier(&self) -> Result<(), QuantaError> {
        Ok(()) // default no-op for drivers that don't need explicit barriers
    }

    /// Transition a buffer between resource states.
    ///
    /// On Vulkan, this inserts a `VkBufferMemoryBarrier2` with the appropriate
    /// stage and access masks. On Metal, this is a no-op (hazard tracking).
    fn barrier_buffer(
        &self,
        _handle: u64,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Transition a texture (image) between resource states.
    ///
    /// On Vulkan, this inserts a `VkImageMemoryBarrier2` with the appropriate
    /// layout transition. On Metal, this is a no-op (hazard tracking).
    fn barrier_texture(
        &self,
        _texture: &Texture,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Debug ===

    /// Push a debug group label (shows in GPU profilers).
    fn debug_push(&self, _label: &str) {}

    /// Pop a debug group label.
    fn debug_pop(&self) {}
}

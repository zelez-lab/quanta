use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, Field, FieldUsage, Format, FormatCaps, GpuDevice, MappedField, OcclusionQuery, Pipeline,
    PipelineDesc, Pulse, QuantaError, QueueFamily, QueueType, RenderPass, ResourceState, Texture,
    TextureDesc, TextureUsage, TextureView, TextureViewDesc, Timeline, TimestampQuery, Wave,
};

/// A GPU device handle. The main entry point for Quanta.
///
/// All GPU operations go through this type. No trait imports needed.
///
/// ```ignore
/// let gpu = quanta::init()?;
/// let field = gpu.field::<f32>(1_000_000)?;
/// let wave = vector_add(&gpu)?;
/// gpu.dispatch(&wave, 1_000_000)?;
/// ```
pub struct Gpu {
    inner: Box<dyn GpuDevice>,
}

impl Gpu {
    #[allow(dead_code)]
    pub(crate) fn new(inner: Box<dyn GpuDevice>) -> Self {
        Self { inner }
    }

    // === Device info ===

    pub fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    pub fn nuclei(&self) -> u32 {
        self.caps().nuclei
    }

    pub fn protons_per_nucleus(&self) -> u32 {
        self.caps().protons_per_nucleus
    }

    pub fn quarks_per_proton(&self) -> u32 {
        self.caps().quarks_per_proton
    }

    pub fn total_quarks(&self) -> u32 {
        self.caps().total_quarks()
    }

    pub fn name(&self) -> &str {
        &self.caps().name
    }

    // === Fields (typed GPU memory) ===

    pub fn field<T: Copy>(&self, count: usize, usage: FieldUsage) -> Result<Field<T>, QuantaError> {
        let size = count * size_of::<T>();
        let handle = self.inner.field_alloc(size, usage)?;
        Ok(Field {
            handle,
            count,
            drop_fn: None,
            _marker: PhantomData,
        })
    }

    pub fn compute_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_compute())
    }

    pub fn render_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_render())
    }

    /// Allocate a uniform buffer field (read + uniform + transfer).
    pub fn uniform_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_uniform())
    }

    pub fn write_field<T: Copy>(&self, field: &Field<T>, data: &[T]) -> Result<(), QuantaError> {
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
        };
        self.inner.field_write_bytes(field.handle(), bytes)
    }

    pub fn read_field<T: Copy>(&self, field: &Field<T>) -> Result<Vec<T>, QuantaError> {
        let bytes = self
            .inner
            .field_read_bytes(field.handle(), field.byte_size())?;
        let mut result = vec![unsafe { core::mem::zeroed::<T>() }; field.len()];
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Ok(result)
    }

    /// Resize a field. Allocates a new field, copies existing data, returns new field.
    /// The old field remains valid until dropped.
    pub fn resize_field<T: Copy>(
        &self,
        old: &Field<T>,
        new_count: usize,
        usage: FieldUsage,
    ) -> Result<Field<T>, QuantaError> {
        let new = self.field::<T>(new_count, usage)?;
        let copy_size = old.byte_size().min(new.byte_size());
        self.inner
            .field_copy_bytes(new.handle(), old.handle(), copy_size)?;
        Ok(new)
    }

    pub fn copy_field<T: Copy>(&self, dst: &Field<T>, src: &Field<T>) -> Result<(), QuantaError> {
        let size = src.byte_size().min(dst.byte_size());
        self.inner
            .field_copy_bytes(dst.handle(), src.handle(), size)
    }

    /// Create a GPU buffer permanently mapped into CPU address space.
    ///
    /// Enables zero-copy writes: data written to the returned `MappedField`
    /// is immediately visible to the GPU (on unified memory architectures)
    /// or automatically synchronized on the next dispatch.
    pub fn field_mapped<T: Copy>(&self, count: usize) -> Result<MappedField<T>, QuantaError> {
        let size = count * size_of::<T>();
        let usage = FieldUsage::default_compute();
        let (handle, ptr) = self.inner.field_create_mapped(size, usage)?;
        Ok(MappedField {
            handle,
            ptr,
            count,
            drop_fn: None,
            _marker: PhantomData,
        })
    }

    // === Textures ===

    /// Create a texture from a descriptor (full control).
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        self.inner.texture_create(desc)
    }

    /// Create a simple RGBA8 texture (convenience).
    pub fn texture(&self, width: u32, height: u32) -> Result<Texture, QuantaError> {
        self.create_texture(&TextureDesc {
            width,
            height,
            format: Format::RGBA8,
            ..TextureDesc::default()
        })
    }

    /// Create a render target texture (can be drawn to and read from shaders).
    pub fn render_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<Texture, QuantaError> {
        self.create_texture(&TextureDesc {
            width,
            height,
            format,
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            ..TextureDesc::default()
        })
    }

    /// Create an MSAA render target.
    pub fn msaa_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<Texture, QuantaError> {
        self.create_texture(&TextureDesc {
            width,
            height,
            format,
            sample_count: samples,
            usage: TextureUsage::RENDER_TARGET,
            ..TextureDesc::default()
        })
    }

    pub fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.inner.texture_write(texture, data)
    }

    pub fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.inner.texture_read(texture)
    }

    /// Create a reusable sampler.
    pub fn sampler(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.inner.sampler_create(desc)
    }

    /// Generate mipmaps for a texture.
    pub fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.inner.generate_mipmaps(texture)
    }

    /// Resolve an MSAA texture to a single-sample texture.
    ///
    /// The source must be a multi-sampled render target, and the destination
    /// must be a single-sample texture of the same dimensions and format.
    pub fn resolve_texture(
        &self,
        msaa_src: &Texture,
        resolve_dst: &Texture,
    ) -> Result<(), QuantaError> {
        self.inner
            .resolve_texture(msaa_src.handle(), resolve_dst.handle())
    }

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
    /// Vulkan uses dispatchGroups with ceil(quarks/64).
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

    // === Render ===

    /// Create a render pipeline from a descriptor.
    pub fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.inner.pipeline_create(desc)
    }

    /// Begin a render pass targeting a texture.
    pub fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.inner.render_begin(target)
    }

    /// End a render pass and submit for execution.
    pub fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        self.inner.render_end(pass)
    }

    // === Sync ===

    pub fn wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        self.inner.pulse_wait(pulse)
    }

    /// Wait for a pulse, then reset it for reuse.
    pub fn wait_and_reset(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        self.inner.pulse_wait(pulse)?;
        pulse.reset();
        Ok(())
    }

    pub fn poll(&self, pulse: &Pulse) -> bool {
        self.inner.pulse_poll(pulse)
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

    // === Timeline semaphores ===

    /// Create a timeline semaphore for multi-frame synchronization.
    pub fn timeline_create(&self) -> Result<Timeline, QuantaError> {
        self.inner.timeline_create()
    }

    /// Signal a timeline to the given value.
    pub fn timeline_signal(&self, timeline: &Timeline, value: u64) -> Result<(), QuantaError> {
        self.inner.timeline_signal(timeline, value)
    }

    /// Block until a timeline reaches at least the given value.
    pub fn timeline_wait(&self, timeline: &Timeline, value: u64) -> Result<(), QuantaError> {
        self.inner.timeline_wait(timeline, value)
    }

    // === Barriers ===

    /// Full pipeline barrier — wait for all prior GPU work to complete.
    ///
    /// This is a heavyweight synchronization point. Prefer `barrier_buffer`
    /// or `barrier_texture` for fine-grained resource transitions.
    pub fn barrier(&self) -> Result<(), QuantaError> {
        self.inner.barrier()
    }

    /// Transition a buffer between resource states.
    ///
    /// On Vulkan, this inserts pipeline barriers with correct stage/access masks.
    /// On Metal, this is a no-op (automatic hazard tracking).
    pub fn barrier_buffer<T: Copy>(
        &self,
        field: &Field<T>,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.inner.barrier_buffer(field.handle(), from, to)
    }

    /// Transition a texture between resource states.
    ///
    /// On Vulkan, this inserts an image layout transition.
    /// On Metal, this is a no-op (automatic hazard tracking).
    pub fn barrier_texture(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.inner.barrier_texture(texture, from, to)
    }

    // === Queries ===

    /// Create a timestamp query set.
    pub fn query_set(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.query_set_create(count)
    }

    /// Read query results.
    pub fn read_queries(
        &self,
        handle: u64,
        first: u32,
        count: u32,
    ) -> Result<Vec<u64>, QuantaError> {
        self.inner.query_set_read(handle, first, count)
    }

    // === Timestamps ===

    /// Create a timestamp query set with `count` slots.
    pub fn timestamp_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.inner.timestamp_query_create(count)
    }

    /// Read timestamp values from a query set.
    pub fn timestamp_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        self.inner.timestamp_query_read(handle)
    }

    /// Create a `TimestampQuery` object wrapping a query set handle.
    pub fn timestamp_query(&self, count: u32) -> Result<TimestampQuery, QuantaError> {
        let handle = self.inner.timestamp_query_create(count)?;
        Ok(TimestampQuery { handle, count })
    }

    /// Write a timestamp at the given index in the query set.
    pub fn write_timestamp(&self, query: &TimestampQuery, index: u32) -> Result<(), QuantaError> {
        self.inner.timestamp_write(query.handle, index)
    }

    /// Read all timestamps from a query set.
    pub fn read_timestamps(&self, query: &TimestampQuery) -> Result<Vec<u64>, QuantaError> {
        self.inner.timestamp_query_read(query.handle)
    }

    /// Convert raw timestamp ticks to nanoseconds using the device frequency.
    pub fn timestamp_to_ns(&self, ticks: u64) -> u64 {
        let freq = self.inner.timestamp_frequency();
        if freq == 0 || freq == 1_000_000_000 {
            ticks
        } else {
            // ticks * 1_000_000_000 / freq, but avoid overflow with u128
            ((ticks as u128 * 1_000_000_000) / freq as u128) as u64
        }
    }

    // === M2.2: Format capability queries ===

    /// Query what a given format can do on this device.
    pub fn format_caps(&self, format: Format) -> FormatCaps {
        self.inner.format_caps(format)
    }

    // === M2.3: Texture views ===

    /// Create a view into an existing texture (sub-range of mips/layers, optional format reinterpret).
    pub fn texture_view_create(
        &self,
        texture: &Texture,
        desc: &TextureViewDesc,
    ) -> Result<TextureView, QuantaError> {
        let handle = self.inner.texture_view_create(texture.handle(), desc)?;
        Ok(TextureView {
            handle,
            drop_fn: None,
        })
    }

    /// Destroy a texture view.
    pub fn texture_view_destroy(&self, view: TextureView) -> Result<(), QuantaError> {
        self.inner.texture_view_destroy(view.handle())
    }

    // === M2.6: Stencil read-back ===

    /// Read stencil buffer contents from a depth/stencil texture.
    pub fn stencil_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.inner.stencil_read(texture.handle())
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

    // === M3.3: Occlusion queries ===

    /// Create an occlusion query set with `count` slots.
    pub fn occlusion_query_create(&self, count: u32) -> Result<OcclusionQuery, QuantaError> {
        let handle = self.inner.occlusion_query_create(count)?;
        Ok(OcclusionQuery { handle, count })
    }

    /// Read results from an occlusion query set (fragment counts per slot).
    pub fn occlusion_query_read(&self, query: &OcclusionQuery) -> Result<Vec<u64>, QuantaError> {
        self.inner.occlusion_query_read(query.handle)
    }

    // === M4.2: Mesh shaders ===

    /// Dispatch a mesh shader pipeline.
    pub fn dispatch_mesh(&self, pipeline: &Pipeline, groups: [u32; 3]) -> Result<(), QuantaError> {
        self.inner.dispatch_mesh(pipeline.handle(), groups)
    }

    // === M4.3: Ray tracing ===

    /// Build a bottom-level acceleration structure from geometry.
    pub fn build_acceleration_structure(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<u64, QuantaError> {
        self.inner.build_acceleration_structure(geometry)
    }

    /// Create a ray tracing pipeline from shader stages.
    pub fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        self.inner.create_ray_tracing_pipeline(desc)
    }

    /// Dispatch rays through a ray tracing pipeline.
    pub fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError> {
        self.inner.dispatch_rays(pipeline, width, height)
    }

    /// Destroy an acceleration structure.
    pub fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.destroy_acceleration_structure(handle)
    }

    // === M5.1: Sparse textures ===

    /// Create a sparse (virtual) texture.
    pub fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        self.inner.sparse_texture_create(desc)
    }

    /// Map a physical backing page to a sparse texture tile.
    pub fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        backing: u64,
    ) -> Result<(), QuantaError> {
        self.inner.sparse_map_tile(texture, mip, x, y, backing)
    }

    /// Unmap a sparse texture tile (release backing memory).
    pub fn sparse_unmap_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
    ) -> Result<(), QuantaError> {
        self.inner.sparse_unmap_tile(texture, mip, x, y)
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

    // === Debug ===

    /// Push a debug group label (visible in GPU profilers like Xcode GPU Capture).
    pub fn debug_push(&self, label: &str) {
        self.inner.debug_push(label);
    }

    /// Pop a debug group label.
    pub fn debug_pop(&self) {
        self.inner.debug_pop();
    }
}

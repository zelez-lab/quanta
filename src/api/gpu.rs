use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    Caps, Field, FieldUsage, Format, FormatCaps, GpuDevice, MappedField, QuantaError,
    ResourceState, Texture, TextureDesc, TextureUsage, TextureView, TextureViewDesc,
    TimestampQuery,
};

mod compute;
mod render;

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
    inner: Arc<dyn GpuDevice>,
}

impl Gpu {
    #[allow(dead_code)]
    pub(crate) fn new(inner: Arc<dyn GpuDevice>) -> Self {
        Self { inner }
    }

    /// Internal device-handle accessor for the `quanta-render` extension
    /// crate. The render methods live on `RenderGpu` (a sibling-crate
    /// extension trait) and reach the shared `GpuDevice` through this.
    /// Not part of the stable public surface — hidden from docs so the
    /// compute crate's documented surface stays render-free.
    #[doc(hidden)]
    pub fn device_handle(&self) -> &Arc<dyn GpuDevice> {
        &self.inner
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

    // === Feature support queries (step 063 slice 20) ===
    //
    // Lets callers check whether a feature has a native lowering
    // on the active backend before submitting work. The
    // alternative is trial-and-error: submit and inspect the
    // returned `QuantaError::NotSupported`. These queries return
    // exactly the bool the device's render path will gate on.

    /// Whether the active backend can lower
    /// `RenderOp::SetShadingRate` to a native VRS path. `false`
    /// when the extension / device-family is missing — the typed
    /// API still accepts VRS state, but the render encoder will
    /// surface NotSupported at submit time.
    pub fn supports_vrs(&self) -> bool {
        self.inner.supports_variable_rate_shading()
    }

    /// Whether the active backend can build acceleration structures
    /// and dispatch ray tracing.
    pub fn supports_ray_tracing(&self) -> bool {
        self.inner.supports_ray_tracing()
    }

    /// Whether the active backend can create mesh-shader pipelines.
    pub fn supports_mesh_shaders(&self) -> bool {
        self.inner.supports_mesh_shaders()
    }

    /// Whether the active backend can create tessellation pipelines
    /// (Vulkan tessellationShader feature / Metal Apple GPU
    /// family 4+).
    pub fn supports_tessellation(&self) -> bool {
        self.inner.supports_tessellation()
    }

    /// Whether the active backend can create sparse textures with
    /// residency control.
    pub fn supports_sparse_residency(&self) -> bool {
        self.inner.supports_sparse_residency()
    }

    /// Whether the active backend can run kernels that use 64-bit
    /// floats. The software lane and llvmpipe support f64; Metal and
    /// the Broadcom V3D GPU do not.
    pub fn supports_f64(&self) -> bool {
        self.inner.supports_f64()
    }

    /// Whether the active backend can run kernels that use 64-bit
    /// integers. The software lane and llvmpipe support i64/u64; the
    /// Broadcom V3D GPU does not.
    pub fn supports_i64(&self) -> bool {
        self.inner.supports_i64()
    }

    /// Hardware-supported VRS shading rates as `(width, height)`
    /// pairs. Empty when VRS isn't supported.
    pub fn supported_shading_rates(&self) -> Vec<(u32, u32)> {
        self.inner.supported_shading_rates()
    }

    pub fn name(&self) -> &str {
        &self.caps().name
    }

    // === Fields (typed GPU memory) ===

    /// Allocate a GPU field with default compute usage (storage + transfer).
    ///
    /// This is the most common allocation — use it for kernel inputs and outputs.
    /// For specific usage flags, see [`field_with_usage`](Gpu::field_with_usage).
    pub fn field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field_with_usage(count, FieldUsage::default_compute())
    }

    /// Allocate a GPU field with explicit usage flags.
    pub fn field_with_usage<T: Copy>(
        &self,
        count: usize,
        usage: FieldUsage,
    ) -> Result<Field<T>, QuantaError> {
        let size = count * size_of::<T>();
        let handle = self.inner.field_alloc(size, usage)?;
        Ok(Field {
            handle,
            count,
            device: self.inner.clone(),
            _marker: PhantomData,
        })
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
            device: self.inner.clone(),
            _marker: PhantomData,
        })
    }

    // === Textures ===

    /// Create a texture from a descriptor (full control).
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let mut tex = self.inner.texture_create(desc)?;
        tex.device = Some(self.inner.clone());
        Ok(tex)
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

    /// Create a reusable sampler.
    pub fn sampler(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.inner.sampler_create(desc)
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

    // === Sync ===

    // === Timeline semaphores ===

    /// Create a timeline semaphore for multi-frame synchronization.
    pub fn timeline_create(&self) -> Result<crate::Timeline, QuantaError> {
        self.inner.timeline_create()
    }

    /// Signal a timeline to the given value.
    pub fn timeline_signal(
        &self,
        timeline: &crate::Timeline,
        value: u64,
    ) -> Result<(), QuantaError> {
        self.inner.timeline_signal(timeline, value)
    }

    /// Block until a timeline reaches at least the given value.
    pub fn timeline_wait(&self, timeline: &crate::Timeline, value: u64) -> Result<(), QuantaError> {
        self.inner.timeline_wait(timeline, value)
    }

    // === Barriers ===

    /// Full pipeline barrier — wait for all prior GPU work to complete.
    ///
    /// This is a heavyweight synchronization point. Prefer `barrier_field`
    /// or `barrier_texture` for fine-grained resource transitions.
    pub fn barrier(&self) -> Result<(), QuantaError> {
        self.inner.barrier()
    }

    /// Transition a field between resource states.
    ///
    /// On Vulkan, this inserts pipeline barriers with correct stage/access masks.
    /// On Metal, this is a no-op (automatic hazard tracking).
    pub fn barrier_field<T: Copy>(
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

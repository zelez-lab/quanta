use core::marker::PhantomData;

use crate::{
    Caps, Field, FieldUsage, Format, GpuDevice, Pipeline, PipelineDesc, Pulse, QuantaError,
    RenderPass, Texture, TextureDesc, TextureUsage, Wave,
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
            core::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
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

    // === Compute ===

    pub fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave(kernel)
    }

    pub fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch(wave, groups)
    }

    /// Dispatch a 1D wave (convenience).
    pub fn dispatch(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch(wave, [quarks, 1, 1])
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

    pub fn wait(&self, pulse: Pulse) -> Result<(), QuantaError> {
        self.inner.pulse_wait(pulse)
    }

    pub fn poll(&self, pulse: &Pulse) -> bool {
        self.inner.pulse_poll(pulse)
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

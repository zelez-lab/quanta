use core::marker::PhantomData;

use crate::{
    Caps, Field, FieldUsage, Format, GpuDevice, Pipeline, PipelineDesc, Pulse, QuantaError,
    RenderPass, Texture, Wave,
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

    /// Device capabilities.
    pub fn caps(&self) -> &Caps {
        self.inner.caps()
    }

    /// Number of compute units (nuclei).
    pub fn nuclei(&self) -> u32 {
        self.caps().nuclei
    }

    /// Cores per compute unit (protons per nucleus).
    pub fn protons_per_nucleus(&self) -> u32 {
        self.caps().protons_per_nucleus
    }

    /// Threads per core (quarks per proton).
    pub fn quarks_per_proton(&self) -> u32 {
        self.caps().quarks_per_proton
    }

    /// Total parallel threads.
    pub fn total_quarks(&self) -> u32 {
        self.caps().total_quarks()
    }

    /// Device name.
    pub fn name(&self) -> &str {
        &self.caps().name
    }

    // === Fields (typed GPU memory) ===

    /// Allocate a typed GPU field.
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

    /// Allocate a compute field (read + write + compute + transfer).
    pub fn compute_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_compute())
    }

    /// Allocate a render field (read + render + transfer).
    pub fn render_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_render())
    }

    /// Write typed data to a field (CPU → GPU).
    pub fn write_field<T: Copy>(&self, field: &Field<T>, data: &[T]) -> Result<(), QuantaError> {
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.inner.field_write_bytes(field.handle(), bytes)
    }

    /// Read typed data from a field (GPU → CPU).
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

    /// Copy data between GPU fields (GPU → GPU, no CPU round-trip).
    pub fn copy_field<T: Copy>(&self, dst: &Field<T>, src: &Field<T>) -> Result<(), QuantaError> {
        let size = src.byte_size().min(dst.byte_size());
        self.inner
            .field_copy_bytes(dst.handle(), src.handle(), size)
    }

    // === Textures ===

    /// Allocate a 2D texture.
    pub fn texture(&self, width: u32, height: u32, format: Format) -> Result<Texture, QuantaError> {
        self.inner.texture(width, height, format)
    }

    /// Write pixel data to texture.
    pub fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.inner.texture_write(texture, data)
    }

    /// Read pixel data from texture.
    pub fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.inner.texture_read(texture)
    }

    // === Compute ===

    /// Create a compute wave from a compiled kernel binary.
    pub fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.inner.wave(kernel)
    }

    /// Dispatch a wave — launch quarks across groups [x, y, z].
    pub fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch(wave, groups)
    }

    /// Dispatch a 1D wave (convenience for [quarks, 1, 1]).
    pub fn dispatch(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.inner.wave_dispatch(wave, [quarks, 1, 1])
    }

    // === Render ===

    /// Create a render pipeline from shader binaries.
    pub fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.inner.pipeline(desc)
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

    /// Block until a pulse (GPU completion) fires.
    pub fn wait(&self, pulse: Pulse) -> Result<(), QuantaError> {
        self.inner.pulse_wait(pulse)
    }

    /// Check if a pulse has fired (non-blocking).
    pub fn poll(&self, pulse: &Pulse) -> bool {
        self.inner.pulse_poll(pulse)
    }
}

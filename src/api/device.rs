use crate::{
    Caps, Field, FieldUsage, Format, Pipeline, Pulse, QuantaError, RenderPass, Texture, Wave,
};
use core::marker::PhantomData;

/// Core trait — every GPU driver implements this.
///
/// On Zelez: implemented by compiled GPU drivers (AMD, NVIDIA, V3D).
/// On macOS: implemented by Metal wrapper.
/// On Linux: implemented by compiled Mesa driver or Vulkan wrapper.
/// On browser: implemented by WebGPU wrapper.
///
/// Methods use raw bytes (`&[u8]`) to keep the trait dyn-compatible.
/// Typed wrappers are provided via `GpuDeviceExt`.
pub trait GpuDevice {
    // === Device info ===

    /// Device capabilities (nuclei, protons, quarks, memory, vendor).
    fn caps(&self) -> &Caps;

    // === Fields (GPU memory) — raw byte interface ===

    /// Allocate a GPU buffer of `size` bytes.
    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError>;

    /// Free a GPU buffer.
    fn field_free(&self, handle: u64);

    /// Write bytes from CPU to GPU buffer.
    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError>;

    /// Read bytes from GPU buffer to CPU.
    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError>;

    /// Copy bytes between GPU buffers (GPU → GPU, no CPU round-trip).
    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError>;

    // === Textures (2D images) ===

    /// Allocate a 2D texture.
    fn texture(&self, width: u32, height: u32, format: Format) -> Result<Texture, QuantaError>;

    /// Write pixel data to texture.
    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError>;

    /// Read pixel data from texture.
    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError>;

    // === Compute ===

    /// Create a compute wave from a compiled kernel binary.
    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError>;

    /// Dispatch a wave — launch quarks across groups.
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError>;

    // === Render ===

    /// Create a render pipeline from shader binaries.
    fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError>;

    /// Begin a render pass targeting a texture.
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError>;

    /// End a render pass and submit for execution.
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError>;

    // === Sync ===

    /// Block until a pulse (GPU completion) fires.
    fn pulse_wait(&self, pulse: Pulse) -> Result<(), QuantaError>;

    /// Check if a pulse has fired (non-blocking).
    fn pulse_poll(&self, pulse: &Pulse) -> bool;
}

/// Render pipeline descriptor — vertex + fragment shader binaries.
pub struct PipelineDesc<'a> {
    pub vertex: &'a [u8],
    pub fragment: &'a [u8],
}

/// Typed convenience wrappers on top of the raw GpuDevice trait.
pub trait GpuDeviceExt: GpuDevice {
    fn nuclei(&self) -> u32 {
        self.caps().nuclei
    }

    fn protons_per_nucleus(&self) -> u32 {
        self.caps().protons_per_nucleus
    }

    fn quarks_per_proton(&self) -> u32 {
        self.caps().quarks_per_proton
    }

    fn total_quarks(&self) -> u32 {
        self.caps().total_quarks()
    }

    fn name(&self) -> &str {
        &self.caps().name
    }

    /// Allocate a typed GPU field.
    fn field<T: Copy>(&self, count: usize, usage: FieldUsage) -> Result<Field<T>, QuantaError> {
        let size = count * size_of::<T>();
        let handle = self.field_alloc(size, usage)?;
        Ok(Field {
            handle,
            count,
            drop_fn: None, // caller must set up drop via driver
            _marker: PhantomData,
        })
        // TODO: drop_fn needs a reference to the device to call field_free.
        // This will be solved when we have a concrete driver — the driver
        // wraps Field with its own drop logic.
    }

    /// Allocate a compute field (read + write + compute + transfer).
    fn compute_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_compute())
    }

    /// Allocate a render field (read + render + transfer).
    fn render_field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field(count, FieldUsage::default_render())
    }

    /// Write typed data to a field.
    fn write_field<T: Copy>(&self, field: &Field<T>, data: &[T]) -> Result<(), QuantaError> {
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        self.field_write_bytes(field.handle, bytes)
    }

    /// Copy data between GPU fields (GPU → GPU, no CPU round-trip).
    fn copy_field<T: Copy>(&self, dst: &Field<T>, src: &Field<T>) -> Result<(), QuantaError> {
        let size = src.byte_size().min(dst.byte_size());
        self.field_copy_bytes(dst.handle, src.handle, size)
    }

    /// Read typed data from a field.
    fn read_field<T: Copy>(&self, field: &Field<T>) -> Result<Vec<T>, QuantaError> {
        let bytes = self.field_read_bytes(field.handle, field.byte_size())?;
        let mut result = vec![unsafe { core::mem::zeroed::<T>() }; field.count];
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Ok(result)
    }

    /// Dispatch a 1D wave (convenience for [quarks, 1, 1]).
    fn wave_dispatch_1d(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.wave_dispatch(wave, [quarks, 1, 1])
    }
}

impl<T: GpuDevice + ?Sized> GpuDeviceExt for T {}

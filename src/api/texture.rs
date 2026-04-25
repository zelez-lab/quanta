use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::ops::Range;

use crate::{Format, GpuDevice, QuantaError};

/// GPU-resident 2D image.
///
/// Resources own their operations — write, read, and mipmap generation
/// are methods on Texture itself, not on Gpu.
pub struct Texture {
    pub(crate) handle: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: Format,
    pub(crate) device: Option<Arc<dyn GpuDevice>>,
}

impl Texture {
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn format(&self) -> Format {
        self.format
    }
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Write pixel data to this texture.
    pub fn write(&self, data: &[u8]) -> Result<(), QuantaError> {
        if let Some(ref dev) = self.device {
            dev.texture_write(self, data)
        } else {
            Err(QuantaError::invalid_param("texture has no device"))
        }
    }

    /// Read pixel data from this texture.
    pub fn read(&self) -> Result<Vec<u8>, QuantaError> {
        if let Some(ref dev) = self.device {
            dev.texture_read(self)
        } else {
            Err(QuantaError::invalid_param("texture has no device"))
        }
    }

    /// Generate mipmaps for this texture.
    pub fn generate_mipmaps(&self) -> Result<(), QuantaError> {
        if let Some(ref dev) = self.device {
            dev.generate_mipmaps(self)
        } else {
            Err(QuantaError::invalid_param("texture has no device"))
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        // Texture cleanup is handled by the driver when the device is dropped.
        // The device ref is held to keep the driver alive while textures exist
        // and to enable operations (write, read, mipmaps).
    }
}

/// Describes how to create a texture.
pub struct TextureDesc {
    pub width: u32,
    pub height: u32,
    /// Depth for 3D textures (1 for 2D).
    pub depth: u32,
    pub format: Format,
    /// Texture kind — 2D, 3D, cube, array.
    pub kind: TextureKind,
    /// MSAA sample count (1 = no MSAA).
    pub sample_count: u32,
    /// Number of mipmap levels (1 = no mipmaps, 0 = auto-calculate).
    pub mip_levels: u32,
    /// Array length (1 for non-array textures).
    pub array_length: u32,
    /// How this texture will be used.
    pub usage: TextureUsage,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            depth: 1,
            format: Format::RGBA8,
            kind: TextureKind::D2,
            sample_count: 1,
            mip_levels: 1,
            array_length: 1,
            usage: TextureUsage::SHADER_READ,
        }
    }
}

/// Texture dimensionality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    /// Standard 2D texture.
    D2,
    /// 3D volume texture.
    D3,
    /// Cube map (6 faces).
    Cube,
    /// Array of 2D textures.
    Array2D,
    /// Array of cube maps.
    ArrayCube,
}

/// How a texture will be used. Drivers optimize placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUsage(u8);

impl TextureUsage {
    /// Readable from shaders (sampling).
    pub const SHADER_READ: Self = Self(1 << 0);
    /// Writable from shaders (compute output).
    pub const SHADER_WRITE: Self = Self(1 << 1);
    /// Usable as a render target (color attachment).
    pub const RENDER_TARGET: Self = Self(1 << 2);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn has(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }
}

// === M2.3: Texture Views ===

/// Describes how to create a view into a texture (sub-range of mips/layers,
/// possibly reinterpreted format).
pub struct TextureViewDesc {
    /// Format override. `None` means use the parent texture's format.
    pub format: Option<Format>,
    /// Range of mip levels visible through this view.
    pub mip_range: Range<u32>,
    /// Range of array layers visible through this view.
    pub layer_range: Range<u32>,
}

/// A view into a texture — sub-range of mips/layers, possibly reinterpreted format.
///
/// Texture views allow shaders to access a portion of a texture array or mip chain
/// without creating a separate allocation.
pub struct TextureView {
    pub(crate) handle: u64,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

impl TextureView {
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for TextureView {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}

/// A reusable texture sampler.
pub struct Sampler {
    pub(crate) handle: u64,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

impl Sampler {
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}

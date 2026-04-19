use crate::Format;

/// GPU-resident 2D image.
pub struct Texture {
    pub(crate) handle: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) format: Format,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
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
}

impl Drop for Texture {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
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

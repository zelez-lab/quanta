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
    pub format: Format,
    /// MSAA sample count (1 = no MSAA).
    pub sample_count: u32,
    /// Generate mipmaps.
    pub mipmaps: bool,
    /// How this texture will be used.
    pub usage: TextureUsage,
}

impl Default for TextureDesc {
    fn default() -> Self {
        Self {
            width: 1,
            height: 1,
            format: Format::RGBA8,
            sample_count: 1,
            mipmaps: false,
            usage: TextureUsage::SHADER_READ,
        }
    }
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

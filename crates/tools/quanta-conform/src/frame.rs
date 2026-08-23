//! The unit of comparison: one rendered RGBA8 frame.

use std::io::{Read, Write};
use std::path::Path;

/// One rendered frame, tightly packed RGBA8, row 0 = top row (the
/// cross-backend orientation contract).
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes, RGBA, row-major from the top.
    pub bytes: Vec<u8>,
}

impl Frame {
    /// Wrap a readback, checking the byte count.
    pub fn rgba8(width: u32, height: u32, bytes: Vec<u8>) -> Result<Self, String> {
        let want = (width as usize) * (height as usize) * 4;
        if bytes.len() != want {
            return Err(format!(
                "frame is {} bytes, {width}x{height} RGBA8 needs {want}",
                bytes.len()
            ));
        }
        Ok(Self {
            width,
            height,
            bytes,
        })
    }

    /// Write as a minimal binary container (dims header + raw bytes).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.width.to_le_bytes())?;
        f.write_all(&self.height.to_le_bytes())?;
        f.write_all(&self.bytes)
    }

    /// Read a frame written by [`Frame::save`].
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut dims = [0u8; 8];
        f.read_exact(&mut dims)?;
        let width = u32::from_le_bytes(dims[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(dims[4..8].try_into().unwrap());
        let mut bytes = Vec::new();
        f.read_to_end(&mut bytes)?;
        Frame::rgba8(width, height, bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

//! Sparse (virtual) textures (steps 030 + 031).
//!
//! Sparse textures are virtual textures whose physical memory is
//! allocated lazily, one tile at a time. Backends:
//!
//! - Metal: `MTLHeap`-backed sparse residency on supporting GPUs.
//! - Vulkan: `VK_EXT_sparse_binding` + `vkQueueBindSparse`.
//! - WebGPU: not in W3C — `NotSupported`.
//! - CPU: software lifecycle only.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.SparseTexture.Texture` (Lean) and
//! `quanta-api/sparse_texture_safety.rs` (Verus):
//!
//! - `map_tile / unmap_tile` fail when the texture is destroyed.
//! - `Drop` calls `sparse_texture_destroy` exactly once.

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError, TextureDesc};

/// A typed sparse texture handle. Drop releases the backend handle.
///
/// Refines `Quanta.SparseTexture.Texture`.
pub struct SparseTexture {
    pub(crate) handle: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl SparseTexture {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Pixel width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Pixel height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Map the tile at `(mip, x, y)` to a backing buffer handle.
    ///
    /// Returns `Err(InvalidParam)` if the texture has been destroyed.
    /// Refines `Quanta.SparseTexture.Texture.mapTile` and the Verus
    /// theorem `t7651_map_then_contains`.
    pub fn map_tile(&self, mip: u32, x: u32, y: u32, backing: u64) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("sparse texture is not live"));
        }
        self.device.sparse_map_tile(self.handle, mip, x, y, backing)
    }

    /// Release the binding for tile `(mip, x, y)`.
    pub fn unmap_tile(&self, mip: u32, x: u32, y: u32) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("sparse texture is not live"));
        }
        self.device.sparse_unmap_tile(self.handle, mip, x, y)
    }
}

impl Drop for SparseTexture {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.sparse_texture_destroy(self.handle);
            self.live = false;
        }
    }
}

/// Re-export from the texture module so users only need the
/// sparse-texture import for typed sparse-texture work.
pub use crate::TextureDesc as SparseTextureDesc;

#[allow(dead_code)]
fn _desc_phantom() -> Option<TextureDesc> {
    None
}

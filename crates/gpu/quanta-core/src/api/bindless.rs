//! Typed bindless resource arrays (steps 034 + 035).
//!
//! Bindless lets a shader index into a "big array of resources" by
//! integer rather than fixed binding slots. The user creates a
//! `BindlessTextureArray` or `BindlessBufferArray`, populates it
//! with `update(index, &resource)`, and binds it to a wave or
//! render pass. Backends:
//!
//! - Metal: argument buffers (Tier 2 on M1+).
//! - Vulkan: VK_EXT_descriptor_indexing (core in 1.2+).
//! - WebGPU: software fallback (no native bindless in W3C spec).
//! - CPU: software array.
//!
//! The wrapper enforces the lifecycle proven in `Quanta.Bindless`
//! (Lean) and `quanta-api/bindless_safety.rs` (Verus):
//!
//! - `update` fails if `index >= capacity` or the array is dropped.
//! - `Drop` calls the matching destroy method exactly once.

use alloc::sync::Arc;

use crate::{Field, GpuDevice, QuantaError, Texture};

/// A bindless texture array — fixed capacity, indexable from the
/// shader by integer.
pub struct BindlessTextureArray {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl BindlessTextureArray {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum number of textures this array can hold.
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Update the slot at `index` to point at `texture`. Returns
    /// `Err(InvalidParam)` if the index is out of bounds or the
    /// array has been destroyed.
    ///
    /// Refines `Quanta.Bindless.Array.set`.
    pub fn update(&self, index: u32, texture: &Texture) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("bindless array is not live"));
        }
        if index >= self.cap {
            return Err(QuantaError::invalid_param(
                "bindless texture array index >= capacity",
            ));
        }
        self.device
            .bindless_texture_set(self.handle, index, texture.handle())
    }
}

impl Drop for BindlessTextureArray {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.bindless_texture_destroy(self.handle);
            self.live = false;
        }
    }
}

/// A bindless buffer array — fixed capacity, indexable from the
/// shader by integer.
pub struct BindlessBufferArray {
    pub(crate) handle: u64,
    pub(crate) cap: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl BindlessBufferArray {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum number of buffers this array can hold.
    pub fn capacity(&self) -> u32 {
        self.cap
    }

    /// Update the slot at `index` to point at `field`. Returns
    /// `Err(InvalidParam)` if the index is out of bounds or the
    /// array has been destroyed.
    ///
    /// Refines `Quanta.Bindless.Array.set`.
    pub fn update<T: Copy>(&self, index: u32, field: &Field<T>) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("bindless array is not live"));
        }
        if index >= self.cap {
            return Err(QuantaError::invalid_param(
                "bindless buffer array index >= capacity",
            ));
        }
        self.device
            .bindless_buffer_set(self.handle, index, field.handle())
    }
}

impl Drop for BindlessBufferArray {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.bindless_buffer_destroy(self.handle);
            self.live = false;
        }
    }
}

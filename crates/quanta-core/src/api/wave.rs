use alloc::sync::Arc;

use crate::api::types::{MAX_BINDINGS, MAX_TEXTURES, PUSH_DATA_CAP};
use crate::{Field, GpuDevice, Texture};

/// A compute dispatch binding — kernel + fields, ready to dispatch.
///
/// Reusable: rebind fields and dispatch again with different data.
/// All binding state is inline — no heap allocation on the hot path.
///
/// Dropping a `Wave` releases the underlying driver resource
/// (`GpuDevice::wave_destroy`), guarded by the `live` flag so the
/// handle is destroyed exactly once.
pub struct Wave {
    pub(crate) handle: u64,
    /// Field handles by slot index. 0 = unbound.
    pub(crate) bindings: [u64; MAX_BINDINGS],
    pub(crate) binding_count: u8,
    /// Texture handles by slot index. 0 = unbound.
    pub(crate) texture_bindings: [u64; MAX_TEXTURES],
    pub(crate) texture_count: u8,
    /// Inline push constant data — 16-byte aligned slots.
    pub(crate) push_data: [u8; PUSH_DATA_CAP],
    pub(crate) push_len: u16,
    /// Bitmask: bit N set = push slot N has been written.
    pub(crate) push_mask: u16,
    /// Workgroup (threadgroup) size [x, y, z]. Default: [64, 1, 1].
    /// Set by the proc macro from `#[quanta::kernel(workgroup = [...])]`.
    pub workgroup_size: [u32; 3],
    /// Drivers construct waves with `device: None`; the `Gpu`
    /// wrapper attaches the device Arc so Drop can release the handle.
    pub(crate) device: Option<Arc<dyn GpuDevice>>,
    /// True while this wrapper owns the driver-side resource. Cleared
    /// on destroy so Drop is idempotent-safe (no double-free).
    pub(crate) live: bool,
}

impl Wave {
    /// Bind a field at the given slot.
    pub fn bind<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        debug_assert!((slot as usize) < MAX_BINDINGS, "max 16 buffer bindings");
        self.bindings[slot as usize] = field.handle();
        if slot as u8 >= self.binding_count {
            self.binding_count = slot as u8 + 1;
        }
    }

    /// Bind a raw buffer handle at the given slot. Use this when working
    /// with handles obtained from low-level driver APIs that don't yield a
    /// typed `Field<T>` — e.g. the WebGPU driver's `field_alloc` returns a
    /// raw `u64`.
    pub fn bind_handle(&mut self, slot: u32, handle: u64) {
        debug_assert!((slot as usize) < MAX_BINDINGS, "max 16 buffer bindings");
        self.bindings[slot as usize] = handle;
        if slot as u8 >= self.binding_count {
            self.binding_count = slot as u8 + 1;
        }
    }

    /// Set a push constant value at the given slot (any size).
    ///
    /// Each slot occupies a 16-byte aligned region inside the inline buffer.
    pub fn set_value<V: Copy>(&mut self, slot: u32, value: V) {
        let size = size_of::<V>();
        let offset = slot as usize * 16; // 16-byte aligned slots
        debug_assert!(offset + size <= PUSH_DATA_CAP);
        debug_assert!((slot as usize) < 16);
        unsafe {
            core::ptr::copy_nonoverlapping(
                &value as *const V as *const u8,
                self.push_data[offset..].as_mut_ptr(),
                size,
            );
        }
        let end = (offset + size) as u16;
        if end > self.push_len {
            self.push_len = end;
        }
        self.push_mask |= 1 << slot;
    }

    /// Set raw bytes as push constant data at the given slot.
    pub fn set_bytes(&mut self, slot: u32, data: &[u8]) {
        let offset = slot as usize * 16;
        debug_assert!(offset + data.len() <= PUSH_DATA_CAP);
        debug_assert!((slot as usize) < 16);
        self.push_data[offset..offset + data.len()].copy_from_slice(data);
        let end = (offset + data.len()) as u16;
        if end > self.push_len {
            self.push_len = end;
        }
        self.push_mask |= 1 << slot;
    }

    /// Bind a read-only texture at the given slot for compute access.
    pub fn bind_texture(&mut self, slot: u32, texture: &Texture) {
        debug_assert!((slot as usize) < MAX_TEXTURES, "max 16 texture bindings");
        self.texture_bindings[slot as usize] = texture.handle();
        if slot as u8 >= self.texture_count {
            self.texture_count = slot as u8 + 1;
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Wave {
    fn drop(&mut self) {
        // Real release: remove the registry entry and free the native
        // object. The `live` flag guarantees at-most-once destruction;
        // driver-internal transients (device: None) drop silently.
        if self.live {
            self.live = false;
            if let Some(ref dev) = self.device {
                let _ = dev.wave_destroy(self.handle);
            }
        }
    }
}

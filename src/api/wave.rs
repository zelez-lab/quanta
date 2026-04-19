use crate::{Field, Texture};

/// A compute dispatch binding — kernel + fields, ready to dispatch.
///
/// Reusable: rebind fields and dispatch again with different data.
pub struct Wave {
    pub(crate) handle: u64,
    pub(crate) bindings: Vec<WaveBinding>,
    pub(crate) push_constants: Vec<PushConstant>,
    pub(crate) texture_bindings: Vec<TextureBinding>,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

pub(crate) struct WaveBinding {
    pub slot: u32,
    pub field_handle: u64,
}

pub(crate) struct PushConstant {
    pub slot: u32,
    pub data: Vec<u8>,
}

pub(crate) struct TextureBinding {
    pub slot: u32,
    pub texture_handle: u64,
}

impl Wave {
    /// Bind a field at the given slot.
    pub fn bind<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        self.bindings.retain(|b| b.slot != slot);
        self.bindings.push(WaveBinding {
            slot,
            field_handle: field.handle(),
        });
    }

    /// Set a push constant value at the given slot (any size).
    pub fn set_value<V: Copy>(&mut self, slot: u32, value: V) {
        let size = size_of::<V>();
        let mut data = vec![0u8; size];
        unsafe {
            core::ptr::copy_nonoverlapping(
                &value as *const V as *const u8,
                data.as_mut_ptr(),
                size,
            );
        }
        self.push_constants.retain(|p| p.slot != slot);
        self.push_constants.push(PushConstant { slot, data });
    }

    /// Set raw bytes as push constant data at the given slot.
    pub fn set_bytes(&mut self, slot: u32, data: &[u8]) {
        self.push_constants.retain(|p| p.slot != slot);
        self.push_constants.push(PushConstant {
            slot,
            data: data.to_vec(),
        });
    }

    /// Bind a read-only texture at the given slot for compute access.
    pub fn bind_texture(&mut self, slot: u32, texture: &Texture) {
        self.texture_bindings.retain(|b| b.slot != slot);
        self.texture_bindings.push(TextureBinding {
            slot,
            texture_handle: texture.handle(),
        });
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Wave {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}

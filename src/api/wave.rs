use crate::Field;

/// A compute dispatch binding — kernel + fields, ready to dispatch.
///
/// Created via [`GpuDevice::wave`]. Bind fields, then dispatch.
/// Reusable: rebind fields and dispatch again with different data.
///
/// Named after quantum wave — a wave carries energy through a field.
/// A Wave carries computation through Fields.
pub struct Wave {
    pub(crate) handle: u64,
    pub(crate) bindings: Vec<WaveBinding>,
    pub(crate) push_constants: Vec<PushConstant>,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

pub(crate) struct WaveBinding {
    pub slot: u32,
    pub field_handle: u64,
}

pub(crate) struct PushConstant {
    pub slot: u32,
    pub data: [u8; 16],
}

impl Wave {
    /// Bind a field at the given slot.
    pub fn bind<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        // Remove existing binding at this slot if any.
        self.bindings.retain(|b| b.slot != slot);
        self.bindings.push(WaveBinding {
            slot,
            field_handle: field.handle(),
        });
    }

    /// Set a push constant value at the given slot (up to 16 bytes).
    pub fn set_value<V: Copy>(&mut self, slot: u32, value: V) {
        assert!(size_of::<V>() <= 16, "push constant must be ≤ 16 bytes");
        let mut data = [0u8; 16];
        unsafe {
            core::ptr::copy_nonoverlapping(
                &value as *const V as *const u8,
                data.as_mut_ptr(),
                size_of::<V>(),
            );
        }
        self.push_constants.retain(|p| p.slot != slot);
        self.push_constants.push(PushConstant { slot, data });
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

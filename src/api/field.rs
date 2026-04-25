use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{GpuDevice, QuantaError};

/// GPU-resident typed buffer. Data that quarks operate on.
///
/// Created via [`Gpu::field`]. Freed when dropped.
/// Type parameter ensures type-safe reads and writes.
///
/// Resources own their operations — write, read, copy are methods
/// on Field itself, not on Gpu.
pub struct Field<T: Copy> {
    pub(crate) handle: u64,
    pub(crate) count: usize,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) _marker: PhantomData<T>,
}

/// A GPU buffer permanently mapped into CPU address space.
///
/// Enables zero-copy uploads: write directly to GPU-visible memory without
/// staging buffer copies. Ideal for uniform buffers, streaming vertex data,
/// and per-frame constants.
///
/// On Metal: backed by `MTLBuffer` with `StorageModeShared`.
/// On Vulkan: uses `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`.
pub struct MappedField<T: Copy> {
    pub(crate) handle: u64,
    pub(crate) ptr: *mut u8,
    pub(crate) count: usize,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) _marker: PhantomData<T>,
}

// Safety: MappedField's ptr points to GPU-visible memory that outlives the field.
// Access is either single-threaded or synchronized by the driver.
unsafe impl<T: Copy> Send for MappedField<T> {}
unsafe impl<T: Copy> Sync for MappedField<T> {}

impl<T: Copy> MappedField<T> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.count * size_of::<T>()
    }

    /// Raw GPU handle (for binding to waves/render passes).
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Write a single element at the given index.
    pub fn write(&mut self, index: usize, value: T) {
        assert!(index < self.count, "MappedField write out of bounds");
        unsafe {
            let dst = self.ptr.add(index * size_of::<T>()) as *mut T;
            core::ptr::write(dst, value);
        }
    }

    /// Read a single element at the given index.
    pub fn read(&self, index: usize) -> T {
        assert!(index < self.count, "MappedField read out of bounds");
        unsafe {
            let src = self.ptr.add(index * size_of::<T>()) as *const T;
            core::ptr::read(src)
        }
    }

    /// Get an immutable slice view of the entire mapped region.
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr as *const T, self.count) }
    }

    /// Get a mutable slice view of the entire mapped region.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.count) }
    }
}

impl<T: Copy> Drop for MappedField<T> {
    fn drop(&mut self) {
        self.device.field_free(self.handle);
    }
}

impl<T: Copy> Field<T> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.count * size_of::<T>()
    }

    /// Raw handle (for driver implementations).
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Write data from CPU to this GPU field.
    pub fn write(&self, data: &[T]) -> Result<(), QuantaError> {
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
        };
        self.device.field_write_bytes(self.handle, bytes)
    }

    /// Read data from this GPU field back to CPU.
    pub fn read(&self) -> Result<Vec<T>, QuantaError> {
        let bytes = self
            .device
            .field_read_bytes(self.handle, self.byte_size())?;
        let mut result = vec![unsafe { core::mem::zeroed::<T>() }; self.count];
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Ok(result)
    }

    /// Copy data from another field into this one.
    /// Copies min(self.byte_size(), src.byte_size()) bytes.
    pub fn copy_from(&self, src: &Field<T>) -> Result<(), QuantaError> {
        let size = self.byte_size().min(src.byte_size());
        self.device.field_copy_bytes(self.handle, src.handle, size)
    }
}

impl<T: Copy> Drop for Field<T> {
    fn drop(&mut self) {
        self.device.field_free(self.handle);
    }
}

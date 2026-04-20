use alloc::boxed::Box;
use core::marker::PhantomData;

/// GPU-resident typed buffer. Data that quarks operate on.
///
/// Created via [`GpuDevice::field`]. Freed when dropped.
/// Type parameter ensures type-safe reads and writes.
pub struct Field<T: Copy> {
    pub(crate) handle: u64,
    pub(crate) count: usize,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
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
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
    pub(crate) _marker: PhantomData<T>,
}

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
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
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
}

impl<T: Copy> Drop for Field<T> {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}

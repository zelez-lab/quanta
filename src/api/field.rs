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

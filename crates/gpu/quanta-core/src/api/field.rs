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

/// A field over caller-owned host memory — the inverse of
/// [`MappedField`]: there, the driver allocates and the host gets a
/// view; here, the host already owns the memory (typically an mmap'd
/// file region) and the device imports a view of it without copying.
///
/// Created via [`Gpu::field_from_host`](crate::Gpu::field_from_host)
/// (safe, lifetime-bound) or
/// [`Gpu::field_from_host_ptr`](crate::Gpu::field_from_host_ptr)
/// (raw, for owners the borrow checker can't see). Zero-copy where the
/// backend has an import path
/// ([`Gpu::supports_host_import`](crate::Gpu::supports_host_import));
/// elsewhere the same call succeeds through a staged copy and
/// [`is_imported`](Self::is_imported) reports `false` — the cost is
/// queryable, never silent.
///
/// **Read-only contract:** bind it only to `&[T]` kernel parameters
/// (via [`Wave::bind_host`](crate::Wave::bind_host)). The type exposes
/// no write path, and the shared `&'a [T]` borrow keeps the region
/// immutable for the field's whole life. Complete or wait in-flight
/// pulses that bind it before dropping it or ending the region's
/// lifetime (unmap).
pub struct HostField<'a, T: Copy> {
    pub(crate) handle: u64,
    pub(crate) count: usize,
    pub(crate) imported: bool,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) _borrow: PhantomData<&'a [T]>,
}

impl<T: Copy> HostField<'_, T> {
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

    /// Raw GPU handle (for binding to waves).
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// `true` when the device reads the caller's memory directly
    /// (zero-copy import); `false` when the backend had no import
    /// path and the data was staged into a device buffer with one
    /// copy at creation.
    pub fn is_imported(&self) -> bool {
        self.imported
    }
}

impl<T: Copy> Drop for HostField<'_, T> {
    fn drop(&mut self) {
        // Releases the driver-side view (buffer object / registry
        // entry) — never the caller's pages.
        self.device.field_free(self.handle);
    }
}

impl<T: Copy> core::fmt::Debug for HostField<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostField")
            .field("handle", &self.handle)
            .field("count", &self.count)
            .field("imported", &self.imported)
            .finish()
    }
}

/// One device buffer, many holders: clones share the same buffer,
/// every `&self` method (`write`, `read`, `handle`,
/// [`Field::native_handle`]) and `Wave::bind` work through `Deref`,
/// and the buffer is freed exactly once — when the last clone drops.
///
/// Enter the pattern with [`Field::into_shared`]. Sharing adds no
/// implicit synchronization: holders order their GPU work with
/// [`Pulse`](crate::Pulse)s exactly as a single owner does.
pub type SharedField<T> = Arc<Field<T>>;

/// A backend-native buffer handle exported from a [`Field`] for
/// zero-copy interop — the buffer sibling of
/// [`NativeTextureHandle`](crate::NativeTextureHandle), with the same
/// ownership/lifetime contract: a **borrow**, valid while the `Field`
/// (and its `Gpu`) live; no ownership transfer. An importer that
/// needs the native object longer must take its own reference through
/// the native API before the `Field` drops.
///
/// Marked `#[non_exhaustive]`: new backend variants (and additional
/// per-backend metadata) can be added without a breaking change —
/// always match with a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum NativeBufferHandle {
    /// Metal: the raw `id<MTLBuffer>` pointer. Non-null. The importer
    /// may message it directly (bind, blit, `retain` for extended
    /// lifetime).
    Metal {
        /// `id<MTLBuffer>` as a raw pointer.
        buffer: *mut core::ffi::c_void,
    },
    /// Vulkan: the `VkBuffer` plus its backing memory. Created by
    /// Quanta's `VkDevice`; cross-device / cross-process import
    /// additionally requires the external-memory extensions, which
    /// are not wired for export yet.
    Vulkan {
        /// The raw `VkBuffer`.
        buffer: *mut core::ffi::c_void,
        /// The `VkDeviceMemory` backing the buffer (offset 0).
        memory: *mut core::ffi::c_void,
        /// Buffer size in bytes.
        size: u64,
    },
    /// WebGPU: reserved — the export path is not implemented and the
    /// backend currently returns `NotSupported`.
    WebGpu {
        /// Registry id of the `GPUBuffer`.
        buffer: u64,
    },
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

    /// Write `data` into this field starting at element `offset`, leaving the
    /// rest of the buffer untouched. Used for partial updates (e.g. filling a
    /// padding tail) without re-uploading the whole buffer.
    pub fn write_at(&self, offset: usize, data: &[T]) -> Result<(), QuantaError> {
        let byte_offset = offset * core::mem::size_of::<T>();
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
        };
        self.device
            .field_write_bytes_at(self.handle, byte_offset, bytes)
    }

    /// Read data from this GPU field back to CPU.
    ///
    /// No implicit GPU sync: if a dispatch writing this field is still
    /// in flight, wait on its [`Pulse`](crate::Pulse) (or call
    /// `Gpu::wait_idle`) first — otherwise the read races the GPU.
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

    /// Move this field into shared ownership — one buffer, many
    /// holders (e.g. two actors reading and writing one table). See
    /// [`SharedField`] for the semantics; the buffer is freed exactly
    /// once, when the last clone drops.
    pub fn into_shared(self) -> SharedField<T> {
        Arc::new(self)
    }

    /// Export the backend-native handle behind this field's buffer,
    /// for zero-copy interop with an external consumer. The handle is
    /// a **borrow** — valid while this `Field` lives, no ownership
    /// transfer; see [`NativeBufferHandle`] and
    /// [`Texture::native_handle`](crate::Texture::native_handle),
    /// whose contract this mirrors. GPU work writing the buffer must
    /// be complete (`Pulse::wait`) before the importer reads it.
    /// Backends without an exportable object (the CPU software
    /// driver, WebGPU) return `NotSupported`; query
    /// `Gpu::supports_native_handle_export` to branch ahead of time.
    pub fn native_handle(&self) -> Result<NativeBufferHandle, QuantaError> {
        self.device.field_native_handle(self.handle)
    }
}

impl<T: Copy> Drop for Field<T> {
    fn drop(&mut self) {
        self.device.field_free(self.handle);
    }
}

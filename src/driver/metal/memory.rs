//! Buffer/field memory operations for Metal.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{FieldUsage, QuantaError};

use super::MetalDevice;
use super::ffi;

impl MetalDevice {
    pub(crate) fn field_alloc_impl(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<u64, QuantaError> {
        let options = if usage.has(FieldUsage::TRANSFER) {
            ffi::MTL_RESOURCE_STORAGE_MODE_SHARED
        } else {
            ffi::MTL_RESOURCE_STORAGE_MODE_PRIVATE
        };
        let buffer = unsafe { ffi::msg_new_buffer(self.device, size as u64, options) };
        let handle = self.alloc_handle();
        self.buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buffer);
        Ok(handle)
    }

    pub(crate) fn field_free_impl(&self, handle: u64) {
        if let Ok(mut buffers) = self.buffers.lock() {
            buffers.remove(&handle);
        }
    }

    pub(crate) fn field_write_bytes_impl(
        &self,
        handle: u64,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        unsafe {
            let contents = ffi::msg_ptr(*buffer, b"contents\0");
            std::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len());
        }
        Ok(())
    }

    pub(crate) fn field_read_bytes_impl(
        &self,
        handle: u64,
        size: usize,
    ) -> Result<Vec<u8>, QuantaError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        unsafe {
            let contents = ffi::msg_ptr(*buffer, b"contents\0");
            std::ptr::copy_nonoverlapping(contents, result.as_mut_ptr(), size);
        }
        Ok(result)
    }

    pub(crate) fn field_copy_bytes_impl(
        &self,
        dst: u64,
        src: u64,
        size: usize,
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let src_buf = buffers.get(&src).ok_or_else(|| {
            QuantaError::invalid_param("bad src handle")
                .with_context(&format!("field_copy_bytes: src handle {src}"))
        })?;
        let dst_buf = buffers.get(&dst).ok_or_else(|| {
            QuantaError::invalid_param("bad dst handle")
                .with_context(&format!("field_copy_bytes: dst handle {dst}"))
        })?;
        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let blit = ffi::msg_id(cmd, b"blitCommandEncoder\0");
            ffi::msg_copy_buffer(blit, *src_buf, 0, *dst_buf, 0, size as u64);
            ffi::msg_void(blit, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }
}

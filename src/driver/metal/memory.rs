//! Buffer/field memory operations for Metal.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{FieldUsage, QuantaError};
use metal as mtl;

use super::MetalDevice;

impl MetalDevice {
    pub(crate) fn field_alloc_impl(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<u64, QuantaError> {
        let options = if usage.has(FieldUsage::TRANSFER) {
            mtl::MTLResourceOptions::StorageModeShared
        } else {
            mtl::MTLResourceOptions::StorageModePrivate
        };
        let buffer = self.device.new_buffer(size as u64, options);
        let handle = self.alloc_handle();
        self.buffers.lock().unwrap().insert(handle, buffer);
        Ok(handle)
    }

    pub(crate) fn field_free_impl(&self, handle: u64) {
        self.buffers.lock().unwrap().remove(&handle);
    }

    pub(crate) fn field_write_bytes_impl(
        &self,
        handle: u64,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents() as *mut u8, data.len());
        }
        Ok(())
    }

    pub(crate) fn field_read_bytes_impl(
        &self,
        handle: u64,
        size: usize,
    ) -> Result<Vec<u8>, QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents() as *const u8,
                result.as_mut_ptr(),
                size,
            );
        }
        Ok(result)
    }

    pub(crate) fn field_copy_bytes_impl(
        &self,
        dst: u64,
        src: u64,
        size: usize,
    ) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let src_buf = buffers.get(&src).ok_or_else(|| {
            QuantaError::invalid_param("bad src handle")
                .with_context(&format!("field_copy_bytes: src handle {src}"))
        })?;
        let dst_buf = buffers.get(&dst).ok_or_else(|| {
            QuantaError::invalid_param("bad dst handle")
                .with_context(&format!("field_copy_bytes: dst handle {dst}"))
        })?;
        let cmd = self.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(src_buf, 0, dst_buf, 0, size as u64);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }
}

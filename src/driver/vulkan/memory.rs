//! Buffer/field memory operations for Vulkan.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::{FieldUsage, QuantaError};
use ash::vk;

use super::VulkanDevice;

impl VulkanDevice {
    pub(crate) fn find_memory_type(
        &self,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<u32, QuantaError> {
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };
        for i in 0..mem_props.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(properties)
            {
                return Ok(i);
            }
        }
        Err(QuantaError::out_of_memory())
    }

    pub(crate) fn field_alloc_impl(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<u64, QuantaError> {
        let mut vk_usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;

        if usage.has(FieldUsage::RENDER) {
            vk_usage |= vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER;
        }

        let buf_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            self.device
                .create_buffer(&buf_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };

        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let mem_props = if usage.has(FieldUsage::TRANSFER) {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };

        let mem_type = self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
        }

        let handle = self.alloc_handle();
        self.buffers.lock().unwrap().insert(
            handle,
            super::VkBuffer {
                buffer,
                memory,
                size: size as u64,
            },
        );
        Ok(handle)
    }

    pub(crate) fn field_free_impl(&self, handle: u64) {
        if let Some(buf) = self.buffers.lock().unwrap().remove(&handle) {
            unsafe {
                self.device.destroy_buffer(buf.buffer, None);
                self.device.free_memory(buf.memory, None);
            }
        }
    }

    pub(crate) fn field_write_bytes_impl(
        &self,
        handle: u64,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        unsafe {
            let ptr = self
                .device
                .map_memory(
                    buf.memory,
                    0,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context(&format!("field_write_bytes: handle {handle}"))
                })? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(buf.memory);
        }
        Ok(())
    }

    pub(crate) fn field_read_bytes_impl(
        &self,
        handle: u64,
        size: usize,
    ) -> Result<Vec<u8>, QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context(&format!("field_read_bytes: handle {handle}"))
                })? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
            self.device.unmap_memory(buf.memory);
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

        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;
            let region = vk::BufferCopy::default().size(size as u64);
            self.device
                .cmd_copy_buffer(cmd, src_buf.buffer, dst_buf.buffer, &[region]);
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(buffers);
        self.submit_and_wait(cmd)
    }
}

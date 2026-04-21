//! Buffer/field memory operations for Vulkan.

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{FieldUsage, QuantaError};

use super::VulkanDevice;
use super::ffi;

impl VulkanDevice {
    pub(crate) fn find_memory_type(
        &self,
        type_filter: u32,
        properties: u32,
    ) -> Result<u32, QuantaError> {
        let mut mem_props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceMemoryProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceMemoryProperties(self.physical_device, &mut mem_props) };
        ffi::find_memory_type(&mem_props, type_filter, properties)
            .ok_or_else(|| QuantaError::out_of_memory())
    }

    pub(crate) fn field_alloc_impl(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<u64, QuantaError> {
        let mut vk_usage = ffi::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | ffi::VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | ffi::VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        if usage.has(FieldUsage::RENDER) {
            vk_usage |= ffi::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                | ffi::VK_BUFFER_USAGE_INDEX_BUFFER_BIT
                | ffi::VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }

        let buf_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: size as u64,
            usage: vk_usage,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };

        let mut buffer = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateBuffer(self.device, &buf_info, core::ptr::null(), &mut buffer) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, buffer, &mut mem_reqs) };

        let mem_props = if usage.has(FieldUsage::TRANSFER) {
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        } else {
            ffi::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        };

        let mem_type = self.find_memory_type(mem_reqs.memory_type_bits, mem_props)?;

        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };

        let mut memory = ffi::null_handle();
        let result = unsafe {
            ffi::vkAllocateMemory(self.device, &alloc_info, core::ptr::null(), &mut memory)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let result = unsafe { ffi::vkBindBufferMemory(self.device, buffer, memory, 0) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        let handle = self.alloc_handle();
        self.buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
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
        if let Some(buf) = self.buffers.lock().ok().and_then(|mut b| b.remove(&handle)) {
            unsafe {
                ffi::vkDestroyBuffer(self.device, buf.buffer, core::ptr::null());
                ffi::vkFreeMemory(self.device, buf.memory, core::ptr::null());
            }
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
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        unsafe {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let result =
                ffi::vkMapMemory(self.device, buf.memory, 0, data.len() as u64, 0, &mut ptr);
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::invalid_param("map failed")
                    .with_context(&format!("field_write_bytes: handle {handle}")));
            }
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u8, data.len());
            ffi::vkUnmapMemory(self.device, buf.memory);
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
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        unsafe {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let r = ffi::vkMapMemory(self.device, buf.memory, 0, size as u64, 0, &mut ptr);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::invalid_param("map failed")
                    .with_context(&format!("field_read_bytes: handle {handle}")));
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, result.as_mut_ptr(), size);
            ffi::vkUnmapMemory(self.device, buf.memory);
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

        let cmd = self.alloc_command_buffer()?;
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            let region = ffi::VkBufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: size as u64,
            };
            ffi::vkCmdCopyBuffer(cmd, src_buf.buffer, dst_buf.buffer, 1, &region);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers);
        self.submit_and_wait(cmd)
    }
}

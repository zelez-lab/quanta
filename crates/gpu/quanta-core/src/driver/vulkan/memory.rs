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
            .ok_or_else(QuantaError::out_of_memory)
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

        // Slice 23 — add SHADER_DEVICE_ADDRESS + AS-build-input
        // usage when the device has bufferDeviceAddress enabled.
        // Free when feature is on; required so user-supplied
        // vertex Fields can be referenced by AS builds.
        if self.buffer_device_address_enabled {
            vk_usage |= ffi::VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | ffi::VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
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

        // Slice 23 — when SHADER_DEVICE_ADDRESS_BIT is set on the
        // buffer, the memory must also be allocated with the
        // matching VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT in a
        // VkMemoryAllocateFlagsInfo p_next chain. Validation layer
        // VUID-vkBindBufferMemory-bufferDeviceAddress-03339 fires
        // otherwise.
        let flags_info = ffi::VkMemoryAllocateFlagsInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            device_mask: 0,
        };
        let alloc_p_next: *const c_void = if self.buffer_device_address_enabled {
            &flags_info as *const _ as *const c_void
        } else {
            core::ptr::null()
        };

        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: alloc_p_next,
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

        // For HOST_VISIBLE buffers, map persistently to avoid map/unmap overhead.
        let mapped_ptr = if (mem_props & ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let r = unsafe { ffi::vkMapMemory(self.device, memory, 0, size as u64, 0, &mut ptr) };
            if r == ffi::VK_SUCCESS && !ptr.is_null() {
                Some(ptr as *mut u8)
            } else {
                None
            }
        } else {
            None
        };

        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::VkBuffer {
                    buffer,
                    memory,
                    size: size as u64,
                    mapped_ptr,
                },
            );
        Ok(handle)
    }

    pub(crate) fn field_import_host_impl(
        &self,
        ptr: *const u8,
        len: usize,
    ) -> Result<u64, QuantaError> {
        let (Some(min_align), Some(get_host_ptr_props)) =
            (self.host_import_min_align, self.get_mem_host_ptr_props_fn)
        else {
            return Err(QuantaError::not_supported(
                "VK_EXT_external_memory_host not available on this device",
            ));
        };
        let align = min_align as usize;
        if len == 0 || !(ptr as usize).is_multiple_of(align) || !len.is_multiple_of(align) {
            return Err(QuantaError::invalid_param(
                "host import requires pointer and length aligned to \
                 minImportedHostPointerAlignment",
            )
            .with_context(&format!("ptr {ptr:p}, len {len}, alignment {align}")));
        }

        // Plain storage buffer; TRANSFER_SRC so diagnostic readbacks
        // can staging-copy out of it. No TRANSFER_DST — the region is
        // read-only by contract.
        let buf_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: len as u64,
            usage: ffi::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | ffi::VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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
        let destroy_buffer = || unsafe {
            ffi::vkDestroyBuffer(self.device, buffer, core::ptr::null());
        };

        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, buffer, &mut mem_reqs) };
        if mem_reqs.size > len as u64 {
            destroy_buffer();
            return Err(QuantaError::invalid_param(
                "device needs more backing bytes than the imported region provides",
            )
            .with_context(&format!("requires {}, imported {len}", mem_reqs.size)));
        }

        // Which memory types can wrap this exact pointer — intersect
        // with the buffer's own requirements.
        let mut host_props = ffi::VkMemoryHostPointerPropertiesEXT {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT,
            p_next: core::ptr::null_mut(),
            memory_type_bits: 0,
        };
        let result = unsafe {
            get_host_ptr_props(
                self.device,
                ffi::VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                ptr as *const c_void,
                &mut host_props,
            )
        };
        let type_filter = mem_reqs.memory_type_bits & host_props.memory_type_bits;
        if result != ffi::VK_SUCCESS || type_filter == 0 {
            destroy_buffer();
            return Err(
                QuantaError::invalid_param("no memory type can import this host pointer")
                    .with_context(&format!("ptr {ptr:p}")),
            );
        }
        let mem_type =
            match self.find_memory_type(type_filter, ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                Ok(t) => t,
                Err(e) => {
                    destroy_buffer();
                    return Err(e);
                }
            };

        // allocation_size is the full imported range: `len` is a
        // min_align multiple (checked above) and covers mem_reqs.size.
        let import_info = ffi::VkImportMemoryHostPointerInfoEXT {
            s_type: ffi::VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
            p_next: core::ptr::null(),
            handle_type: ffi::VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
            p_host_pointer: ptr as *const c_void,
        };
        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: &import_info as *const _ as *const c_void,
            allocation_size: len as u64,
            memory_type_index: mem_type,
        };
        let mut memory = ffi::null_handle();
        let result = unsafe {
            ffi::vkAllocateMemory(self.device, &alloc_info, core::ptr::null(), &mut memory)
        };
        if result != ffi::VK_SUCCESS {
            destroy_buffer();
            return Err(QuantaError::invalid_param("host-pointer import failed")
                .with_context(&format!("vkAllocateMemory: {result}")));
        }
        let result = unsafe { ffi::vkBindBufferMemory(self.device, buffer, memory, 0) };
        if result != ffi::VK_SUCCESS {
            unsafe { ffi::vkFreeMemory(self.device, memory, core::ptr::null()) };
            destroy_buffer();
            return Err(QuantaError::out_of_memory());
        }

        // mapped_ptr stays None: imported memory was never
        // vkMapMemory'd, so the free path must not unmap it. Freeing
        // the entry releases the import (vkFreeMemory), never the
        // caller's pages.
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::VkBuffer {
                    buffer,
                    memory,
                    size: len as u64,
                    mapped_ptr: None,
                },
            );
        Ok(handle)
    }

    pub(crate) fn field_free_impl(&self, handle: u64) {
        if let Some(buf) = self
            .buffers
            .write()
            .ok()
            .and_then(|mut b| b.remove(&handle))
        {
            // Unmapping is a host-side operation, safe while the GPU may
            // still read the buffer; the destroy itself defers behind any
            // outstanding submission (a dropped vertex/storage buffer has
            // the same in-flight hazard as a dropped texture).
            if buf.mapped_ptr.is_some() {
                unsafe { ffi::vkUnmapMemory(self.device, buf.memory) };
            }
            self.retire_bin.retire(
                self.device,
                super::retire::Retired::Buffer {
                    buffer: buf.buffer,
                    memory: buf.memory,
                },
            );
        }
    }

    pub(crate) fn field_write_bytes_impl(
        &self,
        handle: u64,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        // If buffer is persistently mapped, just copy directly.
        if let Some(ptr) = buf.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        } else {
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
        }
        Ok(())
    }

    pub(crate) fn field_write_bytes_at_impl(
        &self,
        handle: u64,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes_at: handle {handle}"))
        })?;
        if let Some(ptr) = buf.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(byte_offset), data.len());
            }
        } else {
            // Map from offset 0 (avoids per-driver map-offset alignment
            // requirements) and write at the byte offset within the mapping.
            let map_len = byte_offset + data.len();
            unsafe {
                let mut ptr: *mut c_void = core::ptr::null_mut();
                let result =
                    ffi::vkMapMemory(self.device, buf.memory, 0, map_len as u64, 0, &mut ptr);
                if result != ffi::VK_SUCCESS {
                    return Err(QuantaError::invalid_param("map failed")
                        .with_context(&format!("field_write_bytes_at: handle {handle}")));
                }
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (ptr as *mut u8).add(byte_offset),
                    data.len(),
                );
                ffi::vkUnmapMemory(self.device, buf.memory);
            }
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
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        if let Some(ptr) = buf.mapped_ptr {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr as *const u8, result.as_mut_ptr(), size);
            }
        } else {
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
        }
        Ok(result)
    }

    pub(crate) fn field_create_mapped_impl(
        &self,
        size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        let vk_usage = ffi::VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | ffi::VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | ffi::VK_BUFFER_USAGE_TRANSFER_DST_BIT;

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

        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;

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

        // Map permanently
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let result = unsafe { ffi::vkMapMemory(self.device, memory, 0, size as u64, 0, &mut ptr) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("map failed")
                .with_context("field_create_mapped: persistent map"));
        }

        let mapped = ptr as *mut u8;
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::VkBuffer {
                    buffer,
                    memory,
                    size: size as u64,
                    mapped_ptr: Some(mapped),
                },
            );
        Ok((handle, mapped))
    }

    pub(crate) fn field_map_impl(&self, handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_map: handle {handle}"))
        })?;
        let mut ptr: *mut c_void = core::ptr::null_mut();
        let result = unsafe { ffi::vkMapMemory(self.device, buf.memory, 0, buf.size, 0, &mut ptr) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("map failed")
                .with_context(&format!("field_map: handle {handle}")));
        }
        Ok(ptr as *mut u8)
    }

    pub(crate) fn field_unmap_impl(&self, handle: u64) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_unmap: handle {handle}"))
        })?;
        unsafe { ffi::vkUnmapMemory(self.device, buf.memory) };
        Ok(())
    }

    pub(crate) fn field_copy_bytes_impl(
        &self,
        dst: u64,
        src: u64,
        size: usize,
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
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
        self.submit_and_wait(cmd).and_then(|mut p| p.wait())
    }
}

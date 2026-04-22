//! Compute dispatch operations for Vulkan.

use alloc::format;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{Pulse, QuantaError, Wave};
use std::ffi::CString;

use super::ffi;
use super::{VkComputePipeline, VulkanDevice};

impl VulkanDevice {
    pub(crate) fn wave_impl(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        // The compiler produces SPIR-V binary directly -- interpret bytes as u32 words.
        // Check for SPIR-V magic number (0x07230203). If absent, this is likely
        // WGSL text from the fallback emitter — reject with a clear error.
        if kernel.len() < 4 {
            return Err(QuantaError::compilation_failed(
                "kernel binary too short for SPIR-V",
            ));
        }
        let magic = u32::from_le_bytes([kernel[0], kernel[1], kernel[2], kernel[3]]);
        if magic != 0x07230203 {
            return Err(QuantaError::compilation_failed(
                "Vulkan requires SPIR-V binary (magic 0x07230203). Got text shader — \
                 install quanta-compiler or build with LLVM for SPIR-V output.",
            ));
        }
        // LLVM's SPIR-V backend may emit a trailing byte — truncate to word boundary.
        let kernel = &kernel[..kernel.len() & !3];
        let spirv_words: Vec<u32> = kernel
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // Create shader module
        let module_info = ffi::VkShaderModuleCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            code_size: kernel.len(),
            p_code: spirv_words.as_ptr(),
        };
        let mut shader_module = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateShaderModule(
                self.device,
                &module_info,
                core::ptr::null(),
                &mut shader_module,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "shader module: VkResult {}",
                result
            )));
        }

        // Descriptor set layout -- one storage buffer per binding.
        // Limit to 8 to stay within maxPerStageDescriptorStorageBuffers on mobile GPUs.
        let mut bindings = Vec::new();
        for i in 0..8u32 {
            bindings.push(ffi::VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: core::ptr::null(),
            });
        }
        let ds_layout_info = ffi::VkDescriptorSetLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut descriptor_set_layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorSetLayout(
                self.device,
                &ds_layout_info,
                core::ptr::null(),
                &mut descriptor_set_layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "ds layout: VkResult {}",
                result
            )));
        }

        // Declare a push constant range. Clamp to device limit (128 on mobile, 256 on desktop).
        let push_size = self.max_push_constants_size.min(256);
        let push_range = ffi::VkPushConstantRange {
            stage_flags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            offset: 0,
            size: push_size,
        };
        let pipeline_layout_info = ffi::VkPipelineLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
        };
        let mut pipeline_layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreatePipelineLayout(
                self.device,
                &pipeline_layout_info,
                core::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "pipeline layout: VkResult {}",
                result
            )));
        }

        let entry_name = CString::new("main").unwrap();
        let stage = ffi::VkPipelineShaderStageCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
            module: shader_module,
            p_name: entry_name.as_ptr(),
            p_specialization_info: core::ptr::null(),
        };

        let pipeline_info = ffi::VkComputePipelineCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage,
            layout: pipeline_layout,
            base_pipeline_handle: ffi::null_handle(),
            base_pipeline_index: -1,
        };

        let mut pipeline = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateComputePipelines(
                self.device,
                ffi::null_handle(),
                1,
                &pipeline_info,
                core::ptr::null(),
                &mut pipeline,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "compute pipeline: VkResult {}",
                result
            )));
        }

        // Clean up shader module (pipeline owns the code now)
        unsafe {
            ffi::vkDestroyShaderModule(self.device, shader_module, core::ptr::null());
        }

        let handle = self.alloc_handle();
        self.compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkComputePipeline {
                    pipeline,
                    layout: pipeline_layout,
                    descriptor_set_layout,
                },
            );

        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            drop_fn: None,
        })
    }

    pub(crate) fn wave_dispatch_impl(
        &self,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        let compute_pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;

        // Create descriptor pool + set for buffer bindings
        let pool_size = ffi::VkDescriptorPoolSize {
            ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 16,
        };
        let pool_info = ffi::VkDescriptorPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorPool(
                self.device,
                &pool_info,
                core::ptr::null(),
                &mut descriptor_pool,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &cp.descriptor_set_layout,
        };
        let mut ds = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        // Update descriptor set with buffer bindings (inline arrays)
        let buffers_guard = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 16] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut write_count = 0usize;

        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0 {
                if let Some(buf) = buffers_guard.get(&handle) {
                    buffer_infos[write_count] = ffi::VkDescriptorBufferInfo {
                        buffer: buf.buffer,
                        offset: 0,
                        range: ffi::VK_WHOLE_SIZE,
                    };
                    writes[write_count] = ffi::VkWriteDescriptorSet {
                        s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        p_next: core::ptr::null(),
                        dst_set: ds,
                        dst_binding: slot as u32,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        p_image_info: core::ptr::null(),
                        p_buffer_info: &buffer_infos[write_count],
                        p_texel_buffer_view: core::ptr::null(),
                    };
                    write_count += 1;
                }
            }
        }

        if write_count > 0 {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    write_count as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }

        // Record and submit command buffer
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
            ffi::vkCmdBindPipeline(cmd, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
            ffi::vkCmdBindDescriptorSets(
                cmd,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                cp.layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );

            // Push constants from inline buffer
            if wave.push_len > 0 {
                ffi::vkCmdPushConstants(
                    cmd,
                    cp.layout,
                    ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    wave.push_len as u32,
                    wave.push_data.as_ptr() as *const c_void,
                );
            }

            ffi::vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers_guard);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        // Clean up descriptor pool
        unsafe {
            ffi::vkDestroyDescriptorPool(self.device, descriptor_pool, core::ptr::null());
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: true,
        })
    }

    pub(crate) fn wave_dispatch_indirect_impl(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        let compute_pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;

        // Create descriptor pool + set (same as wave_dispatch)
        let pool_size = ffi::VkDescriptorPoolSize {
            ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 16,
        };
        let pool_info = ffi::VkDescriptorPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorPool(
                self.device,
                &pool_info,
                core::ptr::null(),
                &mut descriptor_pool,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool,
            descriptor_set_count: 1,
            p_set_layouts: &cp.descriptor_set_layout,
        };
        let mut ds = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        let buffers_guard = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 16] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut write_count = 0usize;

        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0 {
                if let Some(buf) = buffers_guard.get(&handle) {
                    buffer_infos[write_count] = ffi::VkDescriptorBufferInfo {
                        buffer: buf.buffer,
                        offset: 0,
                        range: ffi::VK_WHOLE_SIZE,
                    };
                    writes[write_count] = ffi::VkWriteDescriptorSet {
                        s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        p_next: core::ptr::null(),
                        dst_set: ds,
                        dst_binding: slot as u32,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        p_image_info: core::ptr::null(),
                        p_buffer_info: &buffer_infos[write_count],
                        p_texel_buffer_view: core::ptr::null(),
                    };
                    write_count += 1;
                }
            }
        }
        if write_count > 0 {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    write_count as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }

        let indirect_buf = buffers_guard.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
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
            ffi::vkCmdBindPipeline(cmd, ffi::VK_PIPELINE_BIND_POINT_COMPUTE, cp.pipeline);
            ffi::vkCmdBindDescriptorSets(
                cmd,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                cp.layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );
            ffi::vkCmdDispatchIndirect(cmd, indirect_buf.buffer, offset);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers_guard);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        unsafe {
            ffi::vkDestroyDescriptorPool(self.device, descriptor_pool, core::ptr::null());
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: true,
        })
    }
}

//! Compute dispatch operations for Vulkan.

use alloc::boxed::Box;
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
        if kernel.len() % 4 != 0 {
            return Err(QuantaError::compilation_failed(
                "SPIR-V binary length must be a multiple of 4",
            ));
        }
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

        // Descriptor set layout -- one storage buffer per binding (16 max)
        let mut bindings = Vec::new();
        for i in 0..16u32 {
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

        let pipeline_layout_info = ffi::VkPipelineLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges: core::ptr::null(),
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

        let handle = self.alloc_handle()?;
        self.compute_pipelines
            .lock()
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
            bindings: Vec::new(),
            push_constants: Vec::new(),
            texture_bindings: Vec::new(),
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
            .lock()
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

        // Update descriptor set with buffer bindings
        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: Vec<ffi::VkDescriptorBufferInfo> = Vec::new();
        let mut writes: Vec<ffi::VkWriteDescriptorSet> = Vec::new();

        for binding in &wave.bindings {
            if let Some(buf) = buffers.get(&binding.field_handle) {
                buffer_infos.push(ffi::VkDescriptorBufferInfo {
                    buffer: buf.buffer,
                    offset: 0,
                    range: ffi::VK_WHOLE_SIZE,
                });
            }
        }

        for (i, binding) in wave.bindings.iter().enumerate() {
            if i < buffer_infos.len() {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: binding.slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: &buffer_infos[i],
                    p_texel_buffer_view: core::ptr::null(),
                });
            }
        }

        if !writes.is_empty() {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    writes.len() as u32,
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

            // Push constants
            for pc in &wave.push_constants {
                let size = (std::mem::size_of::<u32>()).min(pc.data.len()) as u32;
                ffi::vkCmdPushConstants(
                    cmd,
                    cp.layout,
                    ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                    pc.slot * 4,
                    size,
                    pc.data.as_ptr() as *const c_void,
                );
            }

            ffi::vkCmdDispatch(cmd, groups[0], groups[1], groups[2]);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        drop(buffers);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        // Clean up descriptor pool
        unsafe {
            ffi::vkDestroyDescriptorPool(self.device, descriptor_pool, core::ptr::null());
        }

        Ok(Pulse {
            handle: self.alloc_handle()?,
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
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
            .lock()
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

        let buffers = self
            .buffers
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: Vec<ffi::VkDescriptorBufferInfo> = Vec::new();
        let mut writes: Vec<ffi::VkWriteDescriptorSet> = Vec::new();
        for binding in &wave.bindings {
            if let Some(buf) = buffers.get(&binding.field_handle) {
                buffer_infos.push(ffi::VkDescriptorBufferInfo {
                    buffer: buf.buffer,
                    offset: 0,
                    range: ffi::VK_WHOLE_SIZE,
                });
            }
        }
        for (i, binding) in wave.bindings.iter().enumerate() {
            if i < buffer_infos.len() {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: binding.slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: &buffer_infos[i],
                    p_texel_buffer_view: core::ptr::null(),
                });
            }
        }
        if !writes.is_empty() {
            unsafe {
                ffi::vkUpdateDescriptorSets(
                    self.device,
                    writes.len() as u32,
                    writes.as_ptr(),
                    0,
                    core::ptr::null(),
                );
            }
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
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
        drop(buffers);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        unsafe {
            ffi::vkDestroyDescriptorPool(self.device, descriptor_pool, core::ptr::null());
        }

        Ok(Pulse {
            handle: self.alloc_handle()?,
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
        })
    }
}

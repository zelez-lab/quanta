//! Compute dispatch operations for Vulkan.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use crate::{Pulse, QuantaError, Wave};
use ash::vk;
use std::ffi::CString;

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
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("shader module: {:?}", e)))?
        };

        // Descriptor set layout -- one storage buffer per binding
        // For now, create a layout with 16 bindings (max fields)
        let mut bindings = Vec::new();
        for i in 0..16u32 {
            bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            );
        }
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("ds layout: {:?}", e)))?
        };

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("pipeline layout: {:?}", e)))?
        };

        let entry_name = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_name);

        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| {
                    QuantaError::compilation_failed(format!("compute pipeline: {:?}", e.1))
                })?[0]
        };

        // Clean up shader module (pipeline owns the code now)
        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        let handle = self.alloc_handle();
        self.compute_pipelines.lock().unwrap().insert(
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
        let compute_pipelines = self.compute_pipelines.lock().unwrap();
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;

        // Create descriptor pool + set for buffer bindings
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|_| QuantaError::submit_failed())?
        };

        let layouts = [cp.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|_| QuantaError::submit_failed())?
        };
        let ds = descriptor_sets[0];

        // Update descriptor set with buffer bindings
        let buffers = self.buffers.lock().unwrap();
        let mut writes = Vec::new();
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();

        for binding in &wave.bindings {
            if let Some(buf) = buffers.get(&binding.field_handle) {
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
        }

        for (i, binding) in wave.bindings.iter().enumerate() {
            if i < buffer_infos.len() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(binding.slot)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_infos[i])),
                );
            }
        }

        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }

        // Record and submit command buffer
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cp.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cp.layout,
                0,
                &[ds],
                &[],
            );

            // Push constants
            for pc in &wave.push_constants {
                self.device.cmd_push_constants(
                    cmd,
                    cp.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    pc.slot * 4, // offset in bytes
                    &pc.data[..std::mem::size_of::<u32>().min(pc.data.len())],
                );
            }

            self.device
                .cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(buffers);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        // Clean up descriptor pool
        unsafe {
            self.device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
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
        let compute_pipelines = self.compute_pipelines.lock().unwrap();
        let cp = compute_pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;

        // Create descriptor pool + set (same as wave_dispatch)
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16);
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(std::slice::from_ref(&pool_size));
        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&pool_info, None)
                .map_err(|_| QuantaError::submit_failed())?
        };

        let layouts = [cp.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|_| QuantaError::submit_failed())?
        };
        let ds = descriptor_sets[0];

        let buffers = self.buffers.lock().unwrap();
        let mut writes = Vec::new();
        let mut buffer_infos: Vec<vk::DescriptorBufferInfo> = Vec::new();
        for binding in &wave.bindings {
            if let Some(buf) = buffers.get(&binding.field_handle) {
                buffer_infos.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE),
                );
            }
        }
        for (i, binding) in wave.bindings.iter().enumerate() {
            if i < buffer_infos.len() {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(binding.slot)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&buffer_infos[i])),
                );
            }
        }
        if !writes.is_empty() {
            unsafe {
                self.device.update_descriptor_sets(&writes, &[]);
            }
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
        })?;

        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, cp.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                cp.layout,
                0,
                &[ds],
                &[],
            );
            self.device
                .cmd_dispatch_indirect(cmd, indirect_buf.buffer, offset);
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(buffers);
        drop(compute_pipelines);
        self.submit_and_wait(cmd)?;

        unsafe {
            self.device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
        })
    }
}

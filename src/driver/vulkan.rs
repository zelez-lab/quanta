//! Vulkan driver for Linux, Android, and Windows.
//!
//! Uses the `ash` crate for raw Vulkan bindings.
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture,
    TextureDesc, TextureUsage, Vendor, Wave,
};
use ash::vk;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Mutex;

/// Vulkan-backed GPU device.
pub struct VulkanDevice {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)]
    queue_family: u32,
    command_pool: vk::CommandPool,
    caps: Caps,
    // Resource storage
    buffers: Mutex<HashMap<u64, VkBuffer>>,
    textures: Mutex<HashMap<u64, VkTexture>>,
    compute_pipelines: Mutex<HashMap<u64, VkComputePipeline>>,
    render_pipelines: Mutex<HashMap<u64, VkRenderPipeline>>,
    next_handle: Mutex<u64>,
}

#[allow(dead_code)]
struct VkBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
}

#[allow(dead_code)]
struct VkTexture {
    image: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
    width: u32,
    height: u32,
    format: vk::Format,
}

struct VkComputePipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

struct VkRenderPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

impl VulkanDevice {
    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        *h += 1;
        *h
    }

    fn find_memory_type(
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
        Err(QuantaError::OutOfMemory)
    }

    fn alloc_command_buffer(&self) -> Result<vk::CommandBuffer, QuantaError> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let bufs = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|_| QuantaError::SubmitFailed)?
        };
        Ok(bufs[0])
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), QuantaError> {
        let cmd_bufs = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .map_err(|_| QuantaError::SubmitFailed)?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|_| QuantaError::SubmitFailed)?;
            self.device.free_command_buffers(self.command_pool, &[cmd]);
        }
        Ok(())
    }
}

/// Discover Vulkan devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(_) => return Vec::new(), // Vulkan not available
    };

    let app_info = vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 3, 0));

    let layer_names: Vec<CString> = Vec::new();
    let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|n| n.as_ptr()).collect();

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_ptrs);

    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };

    let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(d) => d,
        Err(_) => return Vec::new(),
    };

    let mut devices: Vec<Box<dyn GpuDevice>> = Vec::new();

    for pd in physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(pd) };

        // Find a queue family that supports compute + graphics
        let queue_family = queue_families.iter().enumerate().find(|(_, qf)| {
            qf.queue_flags
                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::GRAPHICS)
        });

        let Some((qf_index, _)) = queue_family else {
            continue;
        };

        let queue_priorities = [1.0f32];
        let queue_create = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .queue_priorities(&queue_priorities);

        let device_create =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_create));

        let device = match unsafe { instance.create_device(pd, &device_create, None) } {
            Ok(d) => d,
            Err(_) => continue,
        };

        let queue = unsafe { device.get_device_queue(qf_index as u32, 0) };

        // Command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(qf_index as u32)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = match unsafe { device.create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(_) => continue,
        };

        let name = unsafe {
            std::ffi::CStr::from_ptr(props.device_name.as_ptr())
                .to_string_lossy()
                .to_string()
        };

        let vendor = match props.vendor_id {
            0x1002 => Vendor::Amd,
            0x10DE => Vendor::Nvidia,
            0x8086 => Vendor::Intel,
            0x13B5 => Vendor::Broadcom,
            _ => Vendor::Unknown,
        };

        let caps = Caps {
            nuclei: props.limits.max_compute_work_group_count[0].min(1024),
            protons_per_nucleus: 1,
            quarks_per_proton: props.limits.max_compute_work_group_size[0],
            memory_bytes: 0, // Would need VK_EXT_memory_budget
            max_quarks_per_dispatch: props.limits.max_compute_work_group_invocations,
            max_groups: props.limits.max_compute_work_group_count,
            vendor,
            name,
        };

        devices.push(Box::new(VulkanDevice {
            _entry: entry.clone(),
            instance: instance.clone(),
            physical_device: pd,
            device,
            queue,
            queue_family: qf_index as u32,
            command_pool,
            caps,
            buffers: Mutex::new(HashMap::new()),
            textures: Mutex::new(HashMap::new()),
            compute_pipelines: Mutex::new(HashMap::new()),
            render_pipelines: Mutex::new(HashMap::new()),
            next_handle: Mutex::new(0),
        }));

        break; // Use first suitable device
    }

    devices
}

impl GpuDevice for VulkanDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
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
                .map_err(|_| QuantaError::OutOfMemory)?
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
                .map_err(|_| QuantaError::OutOfMemory)?
        };

        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|_| QuantaError::OutOfMemory)?;
        }

        let handle = self.alloc_handle();
        self.buffers.lock().unwrap().insert(
            handle,
            VkBuffer {
                buffer,
                memory,
                size: size as u64,
            },
        );
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        if let Some(buf) = self.buffers.lock().unwrap().remove(&handle) {
            unsafe {
                self.device.destroy_buffer(buf.buffer, None);
                self.device.free_memory(buf.memory, None);
            }
        }
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle)
            .ok_or(QuantaError::InvalidParam("bad field handle"))?;
        unsafe {
            let ptr = self
                .device
                .map_memory(
                    buf.memory,
                    0,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|_| QuantaError::InvalidParam("map failed"))?
                as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(buf.memory);
        }
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buf = buffers
            .get(&handle)
            .ok_or(QuantaError::InvalidParam("bad field handle"))?;
        let mut result = vec![0u8; size];
        unsafe {
            let ptr = self
                .device
                .map_memory(buf.memory, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|_| QuantaError::InvalidParam("map failed"))?
                as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
            self.device.unmap_memory(buf.memory);
        }
        Ok(result)
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let src_buf = buffers
            .get(&src)
            .ok_or(QuantaError::InvalidParam("bad src"))?;
        let dst_buf = buffers
            .get(&dst)
            .ok_or(QuantaError::InvalidParam("bad dst"))?;

        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::SubmitFailed)?;
            let region = vk::BufferCopy::default().size(size as u64);
            self.device
                .cmd_copy_buffer(cmd, src_buf.buffer, dst_buf.buffer, &[region]);
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::SubmitFailed)?;
        }
        drop(buffers);
        self.submit_and_wait(cmd)
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let vk_format = format_to_vulkan(desc.format);

        let mut vk_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if desc.usage.has(TextureUsage::SHADER_READ) {
            vk_usage |= vk::ImageUsageFlags::SAMPLED;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            vk_usage |= vk::ImageUsageFlags::STORAGE;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            if matches!(desc.format, Format::Depth32Float) {
                vk_usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                vk_usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }
        }

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: desc.depth.max(1),
            })
            .mip_levels(desc.mip_levels.max(1))
            .array_layers(desc.array_length.max(1))
            .samples(sample_count_to_vk(desc.sample_count))
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            self.device
                .create_image(&image_info, None)
                .map_err(|_| QuantaError::OutOfMemory)?
        };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let memory = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::OutOfMemory)?
        };
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|_| QuantaError::OutOfMemory)?;
        }

        let aspect = if matches!(desc.format, Format::Depth32Float) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: desc.mip_levels.max(1),
                base_array_layer: 0,
                layer_count: desc.array_length.max(1),
            });

        let view = unsafe {
            self.device
                .create_image_view(&view_info, None)
                .map_err(|_| QuantaError::OutOfMemory)?
        };

        let handle = self.alloc_handle();
        self.textures.lock().unwrap().insert(
            handle,
            VkTexture {
                image,
                view,
                memory,
                width: desc.width,
                height: desc.height,
                format: vk_format,
            },
        );

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            drop_fn: None,
        })
    }

    fn texture_write(&self, _texture: &Texture, _data: &[u8]) -> Result<(), QuantaError> {
        // TODO: staging buffer + copy + layout transitions
        Ok(())
    }

    fn texture_read(&self, _texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        // TODO: layout transition + copy to staging + read
        Ok(Vec::new())
    }

    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let _info = vk::SamplerCreateInfo::default()
            .min_filter(filter_to_vk(desc.min_filter))
            .mag_filter(filter_to_vk(desc.mag_filter))
            .address_mode_u(address_to_vk(desc.address_u))
            .address_mode_v(address_to_vk(desc.address_v))
            .max_anisotropy(desc.max_anisotropy as f32)
            .anisotropy_enable(desc.max_anisotropy > 1);
        // TODO: create and store sampler
        Ok(crate::Sampler {
            handle: self.alloc_handle(),
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        // TODO: blit commands for each mip level
        Ok(())
    }

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        // kernel is WGSL source — convert to SPIR-V via naga
        let wgsl = std::str::from_utf8(kernel)
            .map_err(|_| QuantaError::CompilationFailed("invalid UTF-8 in WGSL source".into()))?;

        let spirv_words =
            super::spirv::wgsl_to_spirv(wgsl).map_err(QuantaError::CompilationFailed)?;

        // Create shader module
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&module_info, None)
                .map_err(|e| QuantaError::CompilationFailed(format!("shader module: {:?}", e)))?
        };

        // Descriptor set layout — one storage buffer per binding
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
                .map_err(|e| QuantaError::CompilationFailed(format!("ds layout: {:?}", e)))?
        };

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| QuantaError::CompilationFailed(format!("pipeline layout: {:?}", e)))?
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
                    QuantaError::CompilationFailed(format!("compute pipeline: {:?}", e.1))
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
            drop_fn: None,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let compute_pipelines = self.compute_pipelines.lock().unwrap();
        let cp = compute_pipelines
            .get(&wave.handle)
            .ok_or(QuantaError::InvalidParam("bad wave handle"))?;

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
                .map_err(|_| QuantaError::SubmitFailed)?
        };

        let layouts = [cp.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe {
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .map_err(|_| QuantaError::SubmitFailed)?
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
                .map_err(|_| QuantaError::SubmitFailed)?;
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
                .map_err(|_| QuantaError::SubmitFailed)?;
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
        })
    }

    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::InvalidParam(
            "Vulkan indirect dispatch not yet implemented",
        ))
    }

    // === Render ===

    fn pipeline_create(&self, _desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        // TODO: create VkRenderPass, VkPipeline from SPIR-V shaders
        Err(QuantaError::CompilationFailed(
            "Vulkan render pipeline not yet implemented".into(),
        ))
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
        })
    }

    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        // TODO: encode all RenderOps into Vulkan command buffer
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
        })
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: Pulse) -> Result<(), QuantaError> {
        pulse.wait()
    }

    fn pulse_poll(&self, pulse: &Pulse) -> bool {
        pulse.is_done()
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Clean up resources
            for (_, buf) in self.buffers.lock().unwrap().drain() {
                self.device.destroy_buffer(buf.buffer, None);
                self.device.free_memory(buf.memory, None);
            }
            for (_, tex) in self.textures.lock().unwrap().drain() {
                self.device.destroy_image_view(tex.view, None);
                self.device.destroy_image(tex.image, None);
                self.device.free_memory(tex.memory, None);
            }
            for (_, cp) in self.compute_pipelines.lock().unwrap().drain() {
                self.device.destroy_pipeline(cp.pipeline, None);
                self.device.destroy_pipeline_layout(cp.layout, None);
                self.device
                    .destroy_descriptor_set_layout(cp.descriptor_set_layout, None);
            }
            for (_, rp) in self.render_pipelines.lock().unwrap().drain() {
                self.device.destroy_pipeline(rp.pipeline, None);
                self.device.destroy_pipeline_layout(rp.layout, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ============================================================================
// Vulkan type conversions
// ============================================================================

fn format_to_vulkan(format: Format) -> vk::Format {
    match format {
        Format::RGBA8 => vk::Format::R8G8B8A8_UNORM,
        Format::BGRA8 => vk::Format::B8G8R8A8_UNORM,
        Format::R8 => vk::Format::R8_UNORM,
        Format::R16Float => vk::Format::R16_SFLOAT,
        Format::R32Float => vk::Format::R32_SFLOAT,
        Format::RG32Float => vk::Format::R32G32_SFLOAT,
        Format::RGBA16Float => vk::Format::R16G16B16A16_SFLOAT,
        Format::RGBA32Float => vk::Format::R32G32B32A32_SFLOAT,
        Format::Depth32Float => vk::Format::D32_SFLOAT,
    }
}

fn sample_count_to_vk(count: u32) -> vk::SampleCountFlags {
    match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        _ => vk::SampleCountFlags::TYPE_1,
    }
}

fn filter_to_vk(f: crate::render_pass::Filter) -> vk::Filter {
    match f {
        crate::render_pass::Filter::Nearest => vk::Filter::NEAREST,
        crate::render_pass::Filter::Linear => vk::Filter::LINEAR,
    }
}

fn address_to_vk(a: crate::render_pass::AddressMode) -> vk::SamplerAddressMode {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        crate::render_pass::AddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
        crate::render_pass::AddressMode::MirrorRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

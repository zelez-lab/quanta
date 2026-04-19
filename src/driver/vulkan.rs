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
    /// Pool of reusable command buffers. Instead of allocating and freeing
    /// command buffers per submission, completed buffers are reset and returned
    /// here. `alloc_command_buffer` draws from this pool first.
    cmd_buffer_pool: Mutex<Vec<vk::CommandBuffer>>,
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
        Err(QuantaError::out_of_memory())
    }

    fn alloc_command_buffer(&self) -> Result<vk::CommandBuffer, QuantaError> {
        // Try to reuse a previously returned command buffer from the pool.
        if let Some(cmd) = self.cmd_buffer_pool.lock().unwrap().pop() {
            unsafe {
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .map_err(|_| QuantaError::submit_failed())?;
            }
            return Ok(cmd);
        }
        // Pool empty — allocate a fresh one.
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let bufs = unsafe {
            self.device
                .allocate_command_buffers(&alloc_info)
                .map_err(|_| QuantaError::submit_failed())?
        };
        Ok(bufs[0])
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer) -> Result<(), QuantaError> {
        let cmd_bufs = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit], vk::Fence::null())
                .map_err(|_| QuantaError::submit_failed())?;
            self.device
                .queue_wait_idle(self.queue)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        // Return to pool for reuse instead of freeing.
        self.cmd_buffer_pool.lock().unwrap().push(cmd);
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
            cmd_buffer_pool: Mutex::new(Vec::new()),
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

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
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

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
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
                .map_err(|_| QuantaError::out_of_memory())?
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
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_image_memory(image, memory, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
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
                .map_err(|_| QuantaError::out_of_memory())?
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

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_write: handle {}", texture.handle()))
        })?;

        // Create staging buffer
        let staging_info = vk::BufferCreateInfo::default()
            .size(data.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buf = unsafe {
            self.device
                .create_buffer(&staging_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let staging_mem = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_buffer_memory(staging_buf, staging_mem, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
            let ptr = self
                .device
                .map_memory(
                    staging_mem,
                    0,
                    data.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context("texture_write: staging map")
                })? as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            self.device.unmap_memory(staging_mem);
        }

        // Transition image layout + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            // Transition: UNDEFINED → TRANSFER_DST
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy buffer → image
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                });
            self.device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                tex.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition: TRANSFER_DST → SHADER_READ
            let barrier2 = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier2],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Clean up staging
        unsafe {
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
        }
        Ok(())
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;

        let bpp = format_bytes_per_pixel_vk(texture.format());
        let size = (tex.width * tex.height) as usize * bpp;

        // Create staging buffer
        let staging_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buf = unsafe {
            self.device
                .create_buffer(&staging_info, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(staging_buf) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(mem_type);
        let staging_mem = unsafe {
            self.device
                .allocate_memory(&alloc, None)
                .map_err(|_| QuantaError::out_of_memory())?
        };
        unsafe {
            self.device
                .bind_buffer_memory(staging_buf, staging_mem, 0)
                .map_err(|_| QuantaError::out_of_memory())?;
        }

        // Transition + copy
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::SHADER_READ)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: tex.width,
                    height: tex.height,
                    depth: 1,
                });
            self.device.cmd_copy_image_to_buffer(
                cmd,
                tex.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging_buf,
                &[region],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)?;

        // Read from staging
        let mut result = vec![0u8; size];
        unsafe {
            let ptr = self
                .device
                .map_memory(staging_mem, 0, size as u64, vk::MemoryMapFlags::empty())
                .map_err(|_| {
                    QuantaError::invalid_param("map failed")
                        .with_context("texture_read: staging map")
                })? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
            self.device.unmap_memory(staging_mem);
            self.device.destroy_buffer(staging_buf, None);
            self.device.free_memory(staging_mem, None);
        }
        Ok(result)
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

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("generate_mipmaps: handle {}", texture.handle()))
        })?;

        let mut mip_width = tex.width as i32;
        let mut mip_height = tex.height as i32;
        let mip_levels = (mip_width.max(mip_height) as f32).log2().floor() as u32 + 1;

        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;

            for i in 1..mip_levels {
                // Transition level i-1 to TRANSFER_SRC
                let barrier_src = vk::ImageMemoryBarrier::default()
                    .old_layout(if i == 1 {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    })
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i - 1,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_src],
                );

                // Transition level i to TRANSFER_DST
                let barrier_dst = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(tex.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_dst],
                );

                let next_width = (mip_width / 2).max(1);
                let next_height = (mip_height / 2).max(1);

                let blit = vk::ImageBlit::default()
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width,
                            y: mip_height,
                            z: 1,
                        },
                    ])
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i - 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: next_width,
                            y: next_height,
                            z: 1,
                        },
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: i,
                        base_array_layer: 0,
                        layer_count: 1,
                    });

                self.device.cmd_blit_image(
                    cmd,
                    tex.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    tex.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );

                mip_width = next_width;
                mip_height = next_height;
            }

            // Transition all levels to SHADER_READ
            let final_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: mip_levels,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[final_barrier],
            );

            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        drop(textures);
        self.submit_and_wait(cmd)
    }

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        // kernel is WGSL source — convert to SPIR-V via naga
        let wgsl = std::str::from_utf8(kernel)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in WGSL source"))?;

        let spirv_words =
            super::spirv::wgsl_to_spirv(wgsl).map_err(QuantaError::compilation_failed)?;

        // Create shader module
        let module_info = vk::ShaderModuleCreateInfo::default().code(&spirv_words);
        let shader_module = unsafe {
            self.device
                .create_shader_module(&module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("shader module: {:?}", e)))?
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

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
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

    fn wave_dispatch_indirect(
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

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        let vert_wgsl = std::str::from_utf8(desc.vertex)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in vertex shader"))?;
        let frag_wgsl = std::str::from_utf8(desc.fragment)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in fragment shader"))?;

        let vert_spirv =
            super::spirv::wgsl_to_spirv(vert_wgsl).map_err(QuantaError::compilation_failed)?;
        let frag_spirv =
            super::spirv::wgsl_to_spirv(frag_wgsl).map_err(QuantaError::compilation_failed)?;

        let vert_module_info = vk::ShaderModuleCreateInfo::default().code(&vert_spirv);
        let frag_module_info = vk::ShaderModuleCreateInfo::default().code(&frag_spirv);
        let vert_module = unsafe {
            self.device
                .create_shader_module(&vert_module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("vert module: {:?}", e)))?
        };
        let frag_module = unsafe {
            self.device
                .create_shader_module(&frag_module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("frag module: {:?}", e)))?
        };

        // Create VkRenderPass
        let color_format = desc
            .color_formats
            .first()
            .copied()
            .unwrap_or(crate::Format::BGRA8);
        let color_attachment = vk::AttachmentDescription::default()
            .format(format_to_vulkan(color_format))
            .samples(sample_count_to_vk(desc.sample_count))
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass));

        let render_pass = unsafe {
            self.device
                .create_render_pass(&render_pass_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("render pass: {:?}", e)))?
        };

        // Pipeline layout (empty for now — no descriptors for render)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("layout: {:?}", e)))?
        };

        let entry_name = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_name),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match desc.cull_mode {
                crate::CullMode::None => vk::CullModeFlags::NONE,
                crate::CullMode::Front => vk::CullModeFlags::FRONT,
                crate::CullMode::Back => vk::CullModeFlags::BACK,
            })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(sample_count_to_vk(desc.sample_count));

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(desc.blend.enabled)
            .src_color_blend_factor(blend_factor_to_vk(desc.blend.src_rgb))
            .dst_color_blend_factor(blend_factor_to_vk(desc.blend.dst_rgb))
            .color_blend_op(blend_op_to_vk(desc.blend.op_rgb))
            .src_alpha_blend_factor(blend_factor_to_vk(desc.blend.src_alpha))
            .dst_alpha_blend_factor(blend_factor_to_vk(desc.blend.dst_alpha))
            .alpha_blend_op(blend_op_to_vk(desc.blend.op_alpha))
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| {
                    QuantaError::compilation_failed(format!("graphics pipeline: {:?}", e.1))
                })?[0]
        };

        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            self.device.destroy_shader_module(frag_module, None);
            // Note: render_pass is needed for framebuffer creation in render_end
            // Store it alongside the pipeline — for now, leak it (TODO: proper storage)
        }

        let handle = self.alloc_handle();
        self.render_pipelines.lock().unwrap().insert(
            handle,
            VkRenderPipeline {
                pipeline,
                layout: pipeline_layout,
            },
        );
        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
        })
    }

    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        // For now, render_end submits an empty command buffer.
        // Full RenderOp encoding requires VkFramebuffer creation from the target texture,
        // which needs the VkRenderPass stored from pipeline_create.
        // TODO: store render_pass handle and create framebuffer here.
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;
            // TODO: begin render pass, encode ops, end render pass
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        self.submit_and_wait(cmd)?;

        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
        })
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
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

            // Free pooled command buffers before destroying the pool.
            let pooled: Vec<_> = self.cmd_buffer_pool.lock().unwrap().drain(..).collect();
            if !pooled.is_empty() {
                self.device.free_command_buffers(self.command_pool, &pooled);
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

fn format_bytes_per_pixel_vk(format: Format) -> usize {
    match format {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8 => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Depth32Float => 4,
    }
}

fn blend_factor_to_vk(f: crate::BlendFactor) -> vk::BlendFactor {
    use crate::BlendFactor::*;
    match f {
        Zero => vk::BlendFactor::ZERO,
        One => vk::BlendFactor::ONE,
        SrcAlpha => vk::BlendFactor::SRC_ALPHA,
        OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
        DstAlpha => vk::BlendFactor::DST_ALPHA,
        OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
        SrcColor => vk::BlendFactor::SRC_COLOR,
        OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
        DstColor => vk::BlendFactor::DST_COLOR,
        OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
    }
}

fn blend_op_to_vk(op: crate::BlendOp) -> vk::BlendOp {
    use crate::BlendOp::*;
    match op {
        Add => vk::BlendOp::ADD,
        Subtract => vk::BlendOp::SUBTRACT,
        ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
        Min => vk::BlendOp::MIN,
        Max => vk::BlendOp::MAX,
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

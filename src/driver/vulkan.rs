//! Vulkan driver for Linux, Android, Windows, and macOS (via MoltenVK).
//!
//! Uses raw FFI bindings (no `ash` dependency).
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
pub(crate) mod ffi;
mod memory;
mod render;
mod sync;
mod texture;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, QueueFamily, QueueType,
    RenderPass, ResourceState, Texture, TextureDesc, TextureViewDesc, Vendor, Wave,
};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

/// Vulkan-backed GPU device.
pub struct VulkanDevice {
    instance: ffi::VkInstance,
    physical_device: ffi::VkPhysicalDevice,
    device: ffi::VkDevice,
    queue: ffi::VkQueue,
    #[allow(dead_code)]
    queue_family: u32,
    command_pool: ffi::VkCommandPool,
    caps: Caps,
    max_push_constants_size: u32,
    // Resource storage — RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    buffers: RwLock<HashMap<u64, VkBuffer>>,
    textures: RwLock<HashMap<u64, VkTexture>>,
    compute_pipelines: RwLock<HashMap<u64, VkComputePipeline>>,
    render_pipelines: RwLock<HashMap<u64, VkRenderPipeline>>,
    samplers: RwLock<HashMap<u64, ffi::VkSampler>>,
    /// Standalone image views created via texture_view_create (not tied to a full VkTexture).
    image_views: RwLock<HashMap<u64, ffi::VkImageView>>,
    query_pools: RwLock<HashMap<u64, VkQueryPool>>,
    queues: RwLock<HashMap<u64, ffi::VkQueue>>,
    next_handle: AtomicU64,
    /// Pool of reusable command buffers — Mutex since push/pop are always writes.
    cmd_buffer_pool: Mutex<Vec<ffi::VkCommandBuffer>>,
}

struct VkQueryPool {
    pool: ffi::VkQueryPool,
    count: u32,
}

#[allow(dead_code)]
struct VkBuffer {
    buffer: ffi::VkBuffer,
    memory: ffi::VkDeviceMemory,
    size: u64,
}

#[allow(dead_code)]
struct VkTexture {
    image: ffi::VkImage,
    view: ffi::VkImageView,
    memory: ffi::VkDeviceMemory,
    width: u32,
    height: u32,
    format: u32,
    mip_levels: u32,
}

struct VkComputePipeline {
    pipeline: ffi::VkPipeline,
    layout: ffi::VkPipelineLayout,
    descriptor_set_layout: ffi::VkDescriptorSetLayout,
}

struct VkRenderPipeline {
    pipeline: ffi::VkPipeline,
    layout: ffi::VkPipelineLayout,
    render_pass: ffi::VkRenderPass,
    descriptor_set_layout: ffi::VkDescriptorSetLayout,
}

impl VulkanDevice {
    fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Check if a device extension is available on the physical device.
    fn has_device_extension(&self, ext_name: &[u8]) -> bool {
        let mut count = 0u32;
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                core::ptr::null_mut(),
            )
        };
        if result != ffi::VK_SUCCESS || count == 0 {
            return false;
        }
        let mut props = vec![ffi::VkExtensionProperties::default(); count as usize];
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                props.as_mut_ptr(),
            )
        };
        if result != ffi::VK_SUCCESS {
            return false;
        }
        // ext_name is null-terminated; compare up to the null byte.
        let target = &ext_name[..ext_name.len() - 1]; // strip trailing \0
        props.iter().any(|p| {
            let name_bytes = &p.extension_name;
            let len = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            &name_bytes[..len] == target
        })
    }

    fn alloc_command_buffer(&self) -> Result<ffi::VkCommandBuffer, QuantaError> {
        // Try to reuse a previously returned command buffer from the pool.
        if let Some(cmd) = self
            .cmd_buffer_pool
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .pop()
        {
            let result = unsafe { ffi::vkResetCommandBuffer(cmd, 0) };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            return Ok(cmd);
        }
        // Pool empty -- allocate a fresh one.
        let alloc_info = ffi::VkCommandBufferAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.command_pool,
            level: ffi::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateCommandBuffers(self.device, &alloc_info, &mut cmd) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        Ok(cmd)
    }

    fn submit_and_wait(&self, cmd: ffi::VkCommandBuffer) -> Result<(), QuantaError> {
        let submit = ffi::VkSubmitInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            p_wait_dst_stage_mask: core::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkQueueSubmit(self.queue, 1, &submit, ffi::null_handle());
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            let r = ffi::vkQueueWaitIdle(self.queue);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        // Return to pool for reuse instead of freeing.
        self.cmd_buffer_pool
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .push(cmd);
        Ok(())
    }
}

/// Discover Vulkan devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let app_info = ffi::VkApplicationInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: core::ptr::null(),
        p_application_name: core::ptr::null(),
        application_version: 0,
        p_engine_name: core::ptr::null(),
        engine_version: 0,
        api_version: ffi::make_api_version(0, 1, 3, 0),
    };

    let create_info = ffi::VkInstanceCreateInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: core::ptr::null(),
    };

    let mut instance = ffi::null_handle();
    let result = unsafe { ffi::vkCreateInstance(&create_info, core::ptr::null(), &mut instance) };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    let mut count = 0u32;
    let result =
        unsafe { ffi::vkEnumeratePhysicalDevices(instance, &mut count, core::ptr::null_mut()) };
    if result != ffi::VK_SUCCESS || count == 0 {
        return Vec::new();
    }

    let mut physical_devices = vec![ffi::null_handle(); count as usize];
    let result = unsafe {
        ffi::vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr())
    };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    let mut devices: Vec<Box<dyn GpuDevice>> = Vec::new();

    for pd in physical_devices {
        let mut props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceProperties(pd, &mut props) };

        let mut qf_count = 0u32;
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut qf_count, core::ptr::null_mut())
        };
        let mut queue_families = vec![ffi::VkQueueFamilyProperties::default(); qf_count as usize];
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(
                pd,
                &mut qf_count,
                queue_families.as_mut_ptr(),
            )
        };

        // Find a queue family that supports compute + graphics
        let queue_family = queue_families.iter().enumerate().find(|(_, qf)| {
            (qf.queue_flags & ffi::VK_QUEUE_GRAPHICS_BIT) != 0
                && (qf.queue_flags & ffi::VK_QUEUE_COMPUTE_BIT) != 0
        });

        let Some((qf_index, _)) = queue_family else {
            continue;
        };

        let queue_priorities = [1.0f32];
        let queue_create = ffi::VkDeviceQueueCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            queue_family_index: qf_index as u32,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };

        // Enable synchronization2 (Vulkan 1.3 core) for vkCmdPipelineBarrier2
        #[repr(C)]
        struct VkPhysicalDeviceSynchronization2Features {
            s_type: u32,
            p_next: *const core::ffi::c_void,
            synchronization2: u32,
        }
        let sync2_features = VkPhysicalDeviceSynchronization2Features {
            s_type: 1000314007, // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
            p_next: core::ptr::null(),
            synchronization2: 1, // VK_TRUE
        };

        let device_create = ffi::VkDeviceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: &sync2_features as *const _ as *const core::ffi::c_void,
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create,
            enabled_layer_count: 0,
            pp_enabled_layer_names: core::ptr::null(),
            enabled_extension_count: 0,
            pp_enabled_extension_names: core::ptr::null(),
            p_enabled_features: core::ptr::null(),
        };

        let mut device = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateDevice(pd, &device_create, core::ptr::null(), &mut device) };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        let mut queue = ffi::null_handle();
        unsafe { ffi::vkGetDeviceQueue(device, qf_index as u32, 0, &mut queue) };

        // Command pool
        let pool_info = ffi::VkCommandPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: qf_index as u32,
        };
        let mut command_pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateCommandPool(device, &pool_info, core::ptr::null(), &mut command_pool)
        };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        let name = unsafe {
            let cstr =
                std::ffi::CStr::from_ptr(props.device_name.as_ptr() as *const core::ffi::c_char);
            cstr.to_string_lossy().to_string()
        };

        let vendor = match props.vendor_id {
            0x1002 => Vendor::Amd,
            0x10DE => Vendor::Nvidia,
            0x8086 => Vendor::Intel,
            0x13B5 | 0x14E4 => Vendor::Broadcom,
            _ => Vendor::Unknown,
        };

        // Query total device memory from the largest heap
        let mut mem_props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceMemoryProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceMemoryProperties(pd, &mut mem_props) };
        let total_memory = (0..mem_props.memory_heap_count as usize)
            .map(|i| mem_props.memory_heaps[i].size)
            .max()
            .unwrap_or(0);

        let caps = Caps {
            nuclei: props.limits.max_compute_work_group_count[0].min(1024),
            protons_per_nucleus: 1,
            quarks_per_proton: props.limits.max_compute_work_group_size[0],
            memory_bytes: total_memory,
            max_quarks_per_dispatch: props.limits.max_compute_work_group_invocations,
            max_groups: props.limits.max_compute_work_group_count,
            vendor,
            name,
        };

        devices.push(Box::new(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            queue,
            queue_family: qf_index as u32,
            command_pool,
            caps,
            max_push_constants_size: props.limits.max_push_constants_size,
            buffers: RwLock::new(HashMap::new()),
            textures: RwLock::new(HashMap::new()),
            compute_pipelines: RwLock::new(HashMap::new()),
            render_pipelines: RwLock::new(HashMap::new()),
            samplers: RwLock::new(HashMap::new()),
            image_views: RwLock::new(HashMap::new()),
            query_pools: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(0),
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
        self.field_alloc_impl(size, usage)
    }

    fn field_free(&self, handle: u64) {
        self.field_free_impl(handle)
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        self.field_write_bytes_impl(handle, data)
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        self.field_read_bytes_impl(handle, size)
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        self.field_copy_bytes_impl(dst, src, size)
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        self.texture_create_impl(desc)
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.texture_write_impl(texture, data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.texture_read_impl(texture)
    }

    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.sampler_create_impl(desc)
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.generate_mipmaps_impl(texture)
    }

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel)
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_impl(wave, groups)
    }

    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_indirect_impl(wave, buffer, offset)
    }

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.render_begin_impl(target)
    }

    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        self.render_end_impl(pass)
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        pulse.wait()
    }

    fn pulse_poll(&self, pulse: &Pulse) -> bool {
        pulse.is_done()
    }

    // === Mapped buffers ===

    fn field_map(&self, handle: u64, size: usize) -> Result<*mut u8, QuantaError> {
        self.field_map_impl(handle, size)
    }

    fn field_unmap(&self, handle: u64) -> Result<(), QuantaError> {
        self.field_unmap_impl(handle)
    }

    fn field_create_mapped(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        self.field_create_mapped_impl(size, usage)
    }

    // === Timestamps ===

    fn timestamp_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        self.timestamp_query_create_impl(count)
    }

    fn timestamp_write(&self, query_handle: u64, index: u32) -> Result<(), QuantaError> {
        self.timestamp_write_impl(query_handle, index)
    }

    fn timestamp_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        self.timestamp_query_read_impl(handle)
    }

    // === MSAA Resolve ===

    fn resolve_texture(&self, src_handle: u64, dst_handle: u64) -> Result<(), QuantaError> {
        self.resolve_texture_impl(src_handle, dst_handle)
    }

    // === Texture views ===

    fn texture_view_create(
        &self,
        texture_handle: u64,
        desc: &TextureViewDesc,
    ) -> Result<u64, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures
            .get(&texture_handle)
            .ok_or_else(|| QuantaError::invalid_param("bad texture handle"))?;

        let format = match desc.format {
            Some(f) => format_to_vulkan(f),
            None => tex.format,
        };

        let aspect_mask = if format == ffi::VK_FORMAT_D32_SFLOAT {
            ffi::VK_IMAGE_ASPECT_DEPTH_BIT
        } else {
            ffi::VK_IMAGE_ASPECT_COLOR_BIT
        };

        let view_info = ffi::VkImageViewCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            image: tex.image,
            view_type: ffi::VK_IMAGE_VIEW_TYPE_2D,
            format,
            components: ffi::VkComponentMapping::default(),
            subresource_range: ffi::VkImageSubresourceRange {
                aspect_mask,
                base_mip_level: desc.mip_range.start,
                level_count: desc.mip_range.end.saturating_sub(desc.mip_range.start),
                base_array_layer: desc.layer_range.start,
                layer_count: desc.layer_range.end.saturating_sub(desc.layer_range.start),
            },
        };

        let mut view = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateImageView(self.device, &view_info, core::ptr::null(), &mut view)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::internal(
                "vkCreateImageView failed for texture view",
            ));
        }

        let handle = self.alloc_handle();
        drop(textures);
        self.image_views
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, view);
        Ok(handle)
    }

    fn texture_view_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let view = self
            .image_views
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(v) = view {
            unsafe {
                ffi::vkDestroyImageView(self.device, v, core::ptr::null());
            }
        }
        Ok(())
    }

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
        self.barrier_impl()
    }

    fn barrier_buffer(
        &self,
        handle: u64,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.barrier_buffer_impl(handle, from, to)
    }

    fn barrier_texture(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.barrier_texture_impl(texture, from, to)
    }

    // === Multi-queue (M3.1) ===

    fn queue_families(&self) -> Vec<QueueFamily> {
        let mut qf_count = 0u32;
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(
                self.physical_device,
                &mut qf_count,
                core::ptr::null_mut(),
            )
        };
        let mut props = vec![ffi::VkQueueFamilyProperties::default(); qf_count as usize];
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(
                self.physical_device,
                &mut qf_count,
                props.as_mut_ptr(),
            )
        };

        props
            .iter()
            .map(|qf| {
                let queue_type = if (qf.queue_flags & ffi::VK_QUEUE_GRAPHICS_BIT) != 0 {
                    QueueType::Graphics
                } else if (qf.queue_flags & ffi::VK_QUEUE_COMPUTE_BIT) != 0 {
                    QueueType::Compute
                } else {
                    QueueType::Transfer
                };
                QueueFamily {
                    queue_type,
                    count: qf.queue_count,
                }
            })
            .collect()
    }

    fn create_queue(&self, _queue_type: QueueType) -> Result<u64, QuantaError> {
        // Get queue index 0 from the device's queue family.
        // A full implementation would track per-family queue indices.
        let mut queue = ffi::null_handle();
        unsafe { ffi::vkGetDeviceQueue(self.device, self.queue_family, 0, &mut queue) };
        if queue.is_null() {
            return Err(QuantaError::internal("failed to get Vulkan queue"));
        }
        let handle = self.alloc_handle();
        self.queues
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, queue);
        Ok(handle)
    }

    fn queue_signal(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        // Full implementation would use VkSemaphore for cross-queue sync.
        // Single-queue signal is implicit in Vulkan submit ordering.
        Ok(())
    }

    fn queue_wait(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        // Single-queue wait is implicit in Vulkan submit ordering.
        Ok(())
    }

    // === Occlusion queries (M3.3) ===

    fn occlusion_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        let pool_info = ffi::VkQueryPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            query_type: ffi::VK_QUERY_TYPE_OCCLUSION,
            query_count: count,
            pipeline_statistics: 0,
        };
        let mut pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateQueryPool(self.device, &pool_info, core::ptr::null(), &mut pool)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param(
                "occlusion query pool creation failed",
            ));
        }
        let handle = self.alloc_handle();
        self.query_pools
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, VkQueryPool { pool, count });
        Ok(handle)
    }

    fn occlusion_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        let pools = self
            .query_pools
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let qp = pools
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("occlusion query pool not found"))?;

        let count = qp.count as usize;
        let mut results = vec![0u64; count];
        let result = unsafe {
            ffi::vkGetQueryPoolResults(
                self.device,
                qp.pool,
                0,
                qp.count,
                count * core::mem::size_of::<u64>(),
                results.as_mut_ptr() as *mut core::ffi::c_void,
                core::mem::size_of::<u64>() as u64,
                ffi::VK_QUERY_RESULT_64_BIT | ffi::VK_QUERY_RESULT_WAIT_BIT,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::invalid_param("occlusion query read failed"));
        }
        Ok(results)
    }

    // === Mesh shaders (M4.2) ===

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        // Mesh shaders require VK_EXT_mesh_shader. Check if the extension
        // is available on this physical device.
        let has_ext = self.has_device_extension(b"VK_EXT_mesh_shader\0");
        if !has_ext {
            return Err(QuantaError::invalid_param(
                "mesh shaders require VK_EXT_mesh_shader — not available on this device",
            ));
        }
        // Full implementation would use vkCmdDrawMeshTasksEXT.
        Ok(())
    }

    // === Ray tracing (M4.3) ===

    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        let has_accel = self.has_device_extension(b"VK_KHR_acceleration_structure\0");
        if !has_accel {
            return Err(QuantaError::invalid_param(
                "ray tracing requires VK_KHR_acceleration_structure — not available on this device",
            ));
        }
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }

        // Allocate a device-local buffer as backing storage for the BLAS.
        let accel_size = geometry
            .iter()
            .map(|g| g.vertex_count as u64 * 48)
            .sum::<u64>()
            .max(256);
        let handle = self.field_alloc_impl(
            accel_size as usize,
            FieldUsage::READ.union(FieldUsage::WRITE),
        )?;
        Ok(handle)
    }

    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        let has_rt = self.has_device_extension(b"VK_KHR_ray_tracing_pipeline\0");
        if !has_rt {
            return Err(QuantaError::invalid_param(
                "ray tracing pipelines require VK_KHR_ray_tracing_pipeline — not available on this device",
            ));
        }
        // Pipeline creation would compile shader stages via VkRayTracingPipelineCreateInfoKHR.
        let handle = self.alloc_handle();
        Ok(handle)
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        let has_rt = self.has_device_extension(b"VK_KHR_ray_tracing_pipeline\0");
        if !has_rt {
            return Err(QuantaError::invalid_param(
                "ray dispatch requires VK_KHR_ray_tracing_pipeline — not available on this device",
            ));
        }
        // Full implementation would use vkCmdTraceRaysKHR.
        Ok(())
    }

    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        // Release the backing buffer.
        self.field_free_impl(handle);
        Ok(())
    }

    // === Sparse textures (M5.1) ===

    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        // Check for sparse binding support via physical device features.
        let mut features = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceFeatures>() };
        unsafe { ffi::vkGetPhysicalDeviceFeatures(self.physical_device, &mut features) };
        if features.sparse_binding == 0 {
            return Err(QuantaError::invalid_param(
                "sparse textures require VK_EXT_sparse_binding — not available on this device",
            ));
        }
        // Create a regular texture as the sparse resource. Full implementation would
        // use VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT.
        let tex = self.texture_create_impl(desc)?;
        let handle = tex.handle();
        Ok(handle)
    }

    fn sparse_map_tile(
        &self,
        texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        if !textures.contains_key(&texture) {
            return Err(QuantaError::invalid_param(
                "sparse texture handle not found",
            ));
        }
        // Full implementation would use vkQueueBindSparse.
        Ok(())
    }

    fn sparse_unmap_tile(
        &self,
        texture: u64,
        _mip: u32,
        _x: u32,
        _y: u32,
    ) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        if !textures.contains_key(&texture) {
            return Err(QuantaError::invalid_param(
                "sparse texture handle not found",
            ));
        }
        Ok(())
    }

    // === Indirect command buffers (M5.2) ===

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        // Vulkan indirect draw/dispatch is core (no extension needed).
        // Each indirect draw command is 16 bytes (VkDrawIndirectCommand).
        let size = max_commands as usize * 16;
        let handle = self.field_alloc_impl(
            size,
            FieldUsage::READ
                .union(FieldUsage::WRITE)
                .union(FieldUsage::TRANSFER),
        )?;
        Ok(handle)
    }

    fn indirect_buffer_execute(&self, handle: u64, _count: u32) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        if !buffers.contains_key(&handle) {
            return Err(QuantaError::invalid_param(
                "indirect command buffer handle not found",
            ));
        }
        // Full execution would use vkCmdDrawIndirectCount during a render pass.
        Ok(())
    }

    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.field_free_impl(handle);
        Ok(())
    }

    // === Bindless resources (M5.3) ===

    fn bind_texture_array(&self, textures: &[u64]) -> Result<u64, QuantaError> {
        // Vulkan descriptor indexing (VK_EXT_descriptor_indexing) is core in Vulkan 1.2+.
        // Validate texture handles.
        let tex_map = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for &tex_handle in textures {
            if !tex_map.contains_key(&tex_handle) {
                return Err(QuantaError::invalid_param("bad texture handle in array"));
            }
        }
        drop(tex_map);

        // Allocate a buffer to track the array binding. Full implementation would create
        // a descriptor set with variable descriptor count.
        let size = (textures.len().max(1) * 8) as usize;
        let handle = self.field_alloc_impl(size, FieldUsage::READ.union(FieldUsage::TRANSFER))?;
        Ok(handle)
    }

    fn bind_buffer_array(&self, buffers: &[u64]) -> Result<u64, QuantaError> {
        let buf_map = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for &buf_handle in buffers {
            if !buf_map.contains_key(&buf_handle) {
                return Err(QuantaError::invalid_param("bad buffer handle in array"));
            }
        }
        drop(buf_map);

        let size = (buffers.len().max(1) * 8) as usize;
        let handle = self.field_alloc_impl(size, FieldUsage::READ.union(FieldUsage::TRANSFER))?;
        Ok(handle)
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            ffi::vkDeviceWaitIdle(self.device);

            // Clean up resources — write locks since we're draining.
            if let Ok(mut buffers) = self.buffers.write() {
                for (_, buf) in buffers.drain() {
                    ffi::vkDestroyBuffer(self.device, buf.buffer, core::ptr::null());
                    ffi::vkFreeMemory(self.device, buf.memory, core::ptr::null());
                }
            }
            if let Ok(mut textures) = self.textures.write() {
                for (_, tex) in textures.drain() {
                    ffi::vkDestroyImageView(self.device, tex.view, core::ptr::null());
                    ffi::vkDestroyImage(self.device, tex.image, core::ptr::null());
                    ffi::vkFreeMemory(self.device, tex.memory, core::ptr::null());
                }
            }
            if let Ok(mut pipelines) = self.compute_pipelines.write() {
                for (_, cp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, cp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, cp.layout, core::ptr::null());
                    ffi::vkDestroyDescriptorSetLayout(
                        self.device,
                        cp.descriptor_set_layout,
                        core::ptr::null(),
                    );
                }
            }
            if let Ok(mut pipelines) = self.render_pipelines.write() {
                for (_, rp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, rp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, rp.layout, core::ptr::null());
                    ffi::vkDestroyRenderPass(self.device, rp.render_pass, core::ptr::null());
                    ffi::vkDestroyDescriptorSetLayout(
                        self.device,
                        rp.descriptor_set_layout,
                        core::ptr::null(),
                    );
                }
            }
            if let Ok(mut samplers) = self.samplers.write() {
                for (_, sampler) in samplers.drain() {
                    ffi::vkDestroySampler(self.device, sampler, core::ptr::null());
                }
            }
            if let Ok(mut views) = self.image_views.write() {
                for (_, view) in views.drain() {
                    ffi::vkDestroyImageView(self.device, view, core::ptr::null());
                }
            }
            if let Ok(mut pools) = self.query_pools.write() {
                for (_, qp) in pools.drain() {
                    ffi::vkDestroyQueryPool(self.device, qp.pool, core::ptr::null());
                }
            }

            // Free pooled command buffers before destroying the pool.
            let pooled: Vec<_> = self
                .cmd_buffer_pool
                .lock()
                .map(|mut pool| pool.drain(..).collect())
                .unwrap_or_default();
            if !pooled.is_empty() {
                ffi::vkFreeCommandBuffers(
                    self.device,
                    self.command_pool,
                    pooled.len() as u32,
                    pooled.as_ptr(),
                );
            }

            ffi::vkDestroyCommandPool(self.device, self.command_pool, core::ptr::null());
            ffi::vkDestroyDevice(self.device, core::ptr::null());
            ffi::vkDestroyInstance(self.instance, core::ptr::null());
        }
    }
}

// ============================================================================
// Vulkan type conversions
// ============================================================================

fn format_to_vulkan(format: Format) -> u32 {
    match format {
        Format::RGBA8 => ffi::VK_FORMAT_R8G8B8A8_UNORM,
        Format::BGRA8 => ffi::VK_FORMAT_B8G8R8A8_UNORM,
        Format::R8 => ffi::VK_FORMAT_R8_UNORM,
        Format::R16Float => ffi::VK_FORMAT_R16_SFLOAT,
        Format::R32Float => ffi::VK_FORMAT_R32_SFLOAT,
        Format::RG32Float => ffi::VK_FORMAT_R32G32_SFLOAT,
        Format::RGBA16Float => ffi::VK_FORMAT_R16G16B16A16_SFLOAT,
        Format::RGBA32Float => ffi::VK_FORMAT_R32G32B32A32_SFLOAT,
        Format::Depth32Float => ffi::VK_FORMAT_D32_SFLOAT,
        // Compressed formats
        Format::Bc1Rgba => ffi::VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
        Format::Bc3Rgba => ffi::VK_FORMAT_BC3_UNORM_BLOCK,
        Format::Bc5Rg => ffi::VK_FORMAT_BC5_SNORM_BLOCK,
        Format::Bc7Rgba => ffi::VK_FORMAT_BC7_UNORM_BLOCK,
        Format::Astc4x4 => ffi::VK_FORMAT_ASTC_4X4_UNORM_BLOCK,
        Format::Astc6x6 => ffi::VK_FORMAT_ASTC_6X6_UNORM_BLOCK,
        Format::Astc8x8 => ffi::VK_FORMAT_ASTC_8X8_UNORM_BLOCK,
        Format::Etc2Rgb8 => ffi::VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
        Format::Etc2Rgba8 => ffi::VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
    }
}

fn sample_count_to_vk(count: u32) -> u32 {
    match count {
        1 => ffi::VK_SAMPLE_COUNT_1_BIT,
        2 => ffi::VK_SAMPLE_COUNT_2_BIT,
        4 => ffi::VK_SAMPLE_COUNT_4_BIT,
        8 => ffi::VK_SAMPLE_COUNT_8_BIT,
        16 => ffi::VK_SAMPLE_COUNT_16_BIT,
        _ => ffi::VK_SAMPLE_COUNT_1_BIT,
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
        // Compressed: block size in bytes
        Format::Bc1Rgba | Format::Etc2Rgb8 => 8,
        Format::Bc3Rgba | Format::Bc5Rg | Format::Bc7Rgba | Format::Etc2Rgba8 => 16,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
    }
}

fn blend_factor_to_vk(f: crate::BlendFactor) -> u32 {
    use crate::BlendFactor::*;
    match f {
        Zero => ffi::VK_BLEND_FACTOR_ZERO,
        One => ffi::VK_BLEND_FACTOR_ONE,
        SrcAlpha => ffi::VK_BLEND_FACTOR_SRC_ALPHA,
        OneMinusSrcAlpha => ffi::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        DstAlpha => ffi::VK_BLEND_FACTOR_DST_ALPHA,
        OneMinusDstAlpha => ffi::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        SrcColor => ffi::VK_BLEND_FACTOR_SRC_COLOR,
        OneMinusSrcColor => ffi::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        DstColor => ffi::VK_BLEND_FACTOR_DST_COLOR,
        OneMinusDstColor => ffi::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    }
}

fn blend_op_to_vk(op: crate::BlendOp) -> u32 {
    use crate::BlendOp::*;
    match op {
        Add => ffi::VK_BLEND_OP_ADD,
        Subtract => ffi::VK_BLEND_OP_SUBTRACT,
        ReverseSubtract => ffi::VK_BLEND_OP_REVERSE_SUBTRACT,
        Min => ffi::VK_BLEND_OP_MIN,
        Max => ffi::VK_BLEND_OP_MAX,
    }
}

fn compare_op_to_vk(op: crate::CompareOp) -> u32 {
    use crate::CompareOp::*;
    match op {
        Never => 0,
        Less => 1,
        Equal => 2,
        LessEqual => 3,
        Greater => 4,
        NotEqual => 5,
        GreaterEqual => 6,
        Always => 7,
    }
}

fn filter_to_vk(f: crate::render_pass::Filter) -> u32 {
    match f {
        crate::render_pass::Filter::Nearest => ffi::VK_FILTER_NEAREST,
        crate::render_pass::Filter::Linear => ffi::VK_FILTER_LINEAR,
    }
}

fn address_to_vk(a: crate::render_pass::AddressMode) -> u32 {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        crate::render_pass::AddressMode::Repeat => ffi::VK_SAMPLER_ADDRESS_MODE_REPEAT,
        crate::render_pass::AddressMode::MirrorRepeat => {
            ffi::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
        }
    }
}

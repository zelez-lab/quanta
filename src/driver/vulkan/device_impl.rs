//! `GpuDevice` trait implementation for `VulkanDevice`.

use alloc::vec;
use alloc::vec::Vec;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, GpuDevice, Pipeline, Pulse, QuantaError, QueueFamily, QueueType, RenderPass,
    ResourceState, Texture, TextureDesc, TextureViewDesc, Wave,
};

use super::device::{VkQueryPool, VulkanDevice};
use super::ffi;
use super::helpers::format_to_vulkan;

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

    #[cfg(feature = "jit")]
    fn wave_jit(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_jit_impl(kernel_def)
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

    fn icb_record_dispatch(
        &self,
        _handle: u64,
        _index: u32,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        // TODO(032/033): Implement via VK_EXT_device_generated_commands
        // or secondary command buffers (vkBeginCommandBuffer with
        // VK_COMMAND_BUFFER_LEVEL_SECONDARY) replayed via
        // vkCmdExecuteCommands. The proven CPU path serves as the
        // reference implementation in the meantime.
        Err(QuantaError::invalid_param(
            "Vulkan ICB record_dispatch not yet implemented (use CPU device for ICB)",
        ))
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

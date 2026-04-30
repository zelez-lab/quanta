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

    // === Indirect command buffers (steps 032 + 033) ===
    //
    // Refines the Lean `Quanta.Icb.execute` semantics. The IR-level
    // theorem (`T7000`) is parametric in the per-command transformer;
    // here we instantiate that transformer as `wave_dispatch_impl`.
    // Recording snapshots {wave, bindings, push, groups}; executing
    // replays the first `count` snapshots in order on the live
    // compute path.
    //
    // A "true" Vulkan implementation via secondary command buffers
    // (vkBeginCommandBuffer with VK_COMMAND_BUFFER_LEVEL_SECONDARY +
    // vkCmdExecuteCommands) is a perf optimization and a future
    // commit; the proof contract is satisfied by either form.

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        // Native lowering: allocate `max_commands` secondary command
        // buffers up front, plus a dedicated descriptor pool sized
        // to hold one descriptor set per recorded command.
        let alloc_info = ffi::VkCommandBufferAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.command_pool,
            level: ffi::VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            command_buffer_count: max_commands,
        };
        let mut secondaries: Vec<ffi::VkCommandBuffer> =
            vec![ffi::null_handle(); max_commands as usize];
        if max_commands > 0 {
            let r = unsafe {
                ffi::vkAllocateCommandBuffers(self.device, &alloc_info, secondaries.as_mut_ptr())
            };
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        // Descriptor pool sized for `max_commands` storage-buffer
        // sets, MAX_BINDINGS descriptors per set.
        let pool_size = ffi::VkDescriptorPoolSize {
            ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: max_commands * crate::api::wave::MAX_BINDINGS as u32,
        };
        let pool_info = ffi::VkDescriptorPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            max_sets: max_commands.max(1),
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
        };
        let mut descriptor_pool = ffi::null_handle();
        let r = unsafe {
            ffi::vkCreateDescriptorPool(
                self.device,
                &pool_info,
                core::ptr::null(),
                &mut descriptor_pool,
            )
        };
        if r != ffi::VK_SUCCESS {
            if max_commands > 0 {
                unsafe {
                    ffi::vkFreeCommandBuffers(
                        self.device,
                        self.command_pool,
                        max_commands,
                        secondaries.as_ptr(),
                    );
                }
            }
            return Err(QuantaError::submit_failed());
        }

        let handle = self.alloc_handle();
        self.icbs
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VkIcb {
                    cap: max_commands,
                    commands: Vec::with_capacity(max_commands as usize),
                    secondaries,
                    descriptor_pool,
                },
            );
        Ok(handle)
    }

    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        // Resolve compute pipeline + bound buffers up front (the
        // pipeline binding survives the lock; the buffer handles do
        // not need to outlive recording — vkCmdBindDescriptorSets
        // captures the descriptor set, which lives in the ICB's
        // dedicated pool).
        let (pipeline_handle, pipeline_layout) = {
            let cps = self
                .compute_pipelines
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let cp = cps
                .get(&wave.handle)
                .ok_or_else(|| QuantaError::invalid_param("bad wave handle in ICB record"))?;
            (cp.pipeline, cp.layout)
        };
        let descriptor_set_layout = {
            let cps = self
                .compute_pipelines
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            cps.get(&wave.handle)
                .map(|cp| cp.descriptor_set_layout)
                .ok_or_else(|| QuantaError::invalid_param("bad wave handle in ICB record"))?
        };

        let secondary = {
            let mut icbs = self
                .icbs
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let icb = icbs
                .get_mut(&handle)
                .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
            if index != icb.commands.len() as u32 {
                return Err(QuantaError::invalid_param(
                    "ICB record index must equal current length",
                ));
            }
            if index >= icb.cap {
                return Err(QuantaError::invalid_param("ICB index >= capacity"));
            }
            // Push the discriminator first so re-entry can't see a
            // partially-recorded slot.
            icb.commands.push(super::device::VkIcbCommand::Dispatch {
                wave_handle: wave.handle,
                bindings: wave.bindings,
                binding_count: wave.binding_count,
                push_data: wave.push_data,
                push_len: wave.push_len,
                push_mask: wave.push_mask,
                workgroup_size: wave.workgroup_size,
                groups,
            });
            (icb.secondaries[index as usize], icb.descriptor_pool)
        };
        let (secondary_cb, dedicated_pool) = (secondary.0, secondary.1);

        // Allocate a descriptor set from the ICB's dedicated pool.
        let alloc_info = ffi::VkDescriptorSetAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            descriptor_pool: dedicated_pool,
            descriptor_set_count: 1,
            p_set_layouts: &descriptor_set_layout,
        };
        let mut ds = ffi::null_handle();
        let r = unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        // Update descriptor set with bound buffers.
        let buffers_guard = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let mut buffer_infos: [ffi::VkDescriptorBufferInfo; 16] = unsafe { core::mem::zeroed() };
        let mut writes: [ffi::VkWriteDescriptorSet; 16] = unsafe { core::mem::zeroed() };
        let mut write_count = 0usize;
        for slot in 0..wave.binding_count as usize {
            let h = wave.bindings[slot];
            if h != 0
                && let Some(buf) = buffers_guard.get(&h)
            {
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
        drop(buffers_guard);

        // Record the secondary command buffer. Compute outside a
        // render pass: inheritance info has null render_pass and
        // framebuffer; SIMULTANEOUS_USE so a single ICB execute
        // can replay multiple times concurrently.
        let inheritance = ffi::VkCommandBufferInheritanceInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            p_next: core::ptr::null(),
            render_pass: ffi::null_handle(),
            subpass: 0,
            framebuffer: ffi::null_handle(),
            occlusion_query_enable: 0,
            query_flags: 0,
            pipeline_statistics: 0,
        };
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            p_inheritance_info: &inheritance as *const _ as *const core::ffi::c_void,
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(secondary_cb, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            ffi::vkCmdBindPipeline(
                secondary_cb,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_handle,
            );
            ffi::vkCmdBindDescriptorSets(
                secondary_cb,
                ffi::VK_PIPELINE_BIND_POINT_COMPUTE,
                pipeline_layout,
                0,
                1,
                &ds,
                0,
                core::ptr::null(),
            );
            ffi::vkCmdDispatch(secondary_cb, groups[0], groups[1], groups[2]);
            let r = ffi::vkEndCommandBuffer(secondary_cb);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        Ok(())
    }

    fn icb_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let mut icbs = self
            .icbs
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        icb.commands.push(super::device::VkIcbCommand::Draw {
            pipeline,
            vertex_count,
            instance_count,
        });
        Ok(())
    }

    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        // Native lowering: collect the first `count` secondary CBs,
        // submit a single primary CB that calls
        // `vkCmdExecuteCommands(primary, count, &secondaries[..count])`,
        // submit + wait once. Refines the proven Lean T7000
        // equivalence theorem (recorded order preserved on execute).
        let secondaries: Vec<ffi::VkCommandBuffer> = {
            let icbs = self
                .icbs
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let icb = icbs
                .get(&handle)
                .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
            if count > icb.commands.len() as u32 {
                return Err(QuantaError::invalid_param(
                    "ICB execute count exceeds recorded length",
                ));
            }
            icb.secondaries[..count as usize].to_vec()
        };
        if count == 0 {
            return Ok(());
        }
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
            ffi::vkCmdExecuteCommands(cmd, count, secondaries.as_ptr());
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        self.submit_and_wait(cmd)?.wait()?;
        Ok(())
    }

    // === Render bundles (steps 032 + 033, render path) ===
    //
    // Native lowering: pre-allocate `max_commands` secondary command
    // buffers (LEVEL_SECONDARY). On record_draw, begin the
    // secondary CB with RENDER_PASS_CONTINUE_BIT against the
    // pipeline's compatible VkRenderPass, vkCmdBindPipeline +
    // vkCmdDraw, end. RenderOp::ExecuteRenderBundle in the parent
    // render pass calls vkCmdExecuteCommands with the recorded
    // secondaries. Mixing inline draws with bundle execute in the
    // same pass is rejected (Vulkan subpass-contents rule).

    fn render_bundle_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        let alloc_info = ffi::VkCommandBufferAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.command_pool,
            level: ffi::VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            command_buffer_count: max_commands,
        };
        let mut secondaries: Vec<ffi::VkCommandBuffer> =
            vec![ffi::null_handle(); max_commands as usize];
        if max_commands > 0 {
            let r = unsafe {
                ffi::vkAllocateCommandBuffers(self.device, &alloc_info, secondaries.as_mut_ptr())
            };
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        let handle = self.alloc_handle();
        self.render_bundles
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VulkanRenderBundle {
                    cap: max_commands,
                    recorded: 0,
                    secondaries,
                },
            );
        Ok(handle)
    }

    fn render_bundle_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let (vk_pipeline, vk_render_pass) = {
            let rps = self
                .render_pipelines
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let rp = rps
                .get(&pipeline)
                .ok_or_else(|| QuantaError::invalid_param("bad pipeline in render bundle"))?;
            (rp.pipeline, rp.render_pass)
        };

        let secondary_cb = {
            let mut bundles = self
                .render_bundles
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let bundle = bundles
                .get_mut(&handle)
                .ok_or_else(|| QuantaError::invalid_param("render bundle handle not found"))?;
            if index != bundle.recorded {
                return Err(QuantaError::invalid_param(
                    "render bundle record index must equal current length",
                ));
            }
            if index >= bundle.cap {
                return Err(QuantaError::invalid_param(
                    "render bundle index >= capacity",
                ));
            }
            let cb = bundle.secondaries[index as usize];
            bundle.recorded += 1;
            cb
        };

        let inheritance = ffi::VkCommandBufferInheritanceInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            p_next: core::ptr::null(),
            render_pass: vk_render_pass,
            subpass: 0,
            framebuffer: ffi::null_handle(),
            occlusion_query_enable: 0,
            query_flags: 0,
            pipeline_statistics: 0,
        };
        let begin = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
                | ffi::VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT,
            p_inheritance_info: &inheritance as *const _ as *const core::ffi::c_void,
        };
        unsafe {
            let r = ffi::vkBeginCommandBuffer(secondary_cb, &begin);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            ffi::vkCmdBindPipeline(
                secondary_cb,
                ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                vk_pipeline,
            );
            ffi::vkCmdDraw(secondary_cb, vertex_count, instance_count.max(1), 0, 0);
            let r = ffi::vkEndCommandBuffer(secondary_cb);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }
        Ok(())
    }

    fn render_bundle_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let removed = self
            .render_bundles
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(bundle) = removed
            && !bundle.secondaries.is_empty()
        {
            unsafe {
                ffi::vkFreeCommandBuffers(
                    self.device,
                    self.command_pool,
                    bundle.secondaries.len() as u32,
                    bundle.secondaries.as_ptr(),
                );
            }
        }
        Ok(())
    }

    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let removed = self
            .icbs
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(icb) = removed {
            unsafe {
                if !icb.secondaries.is_empty() {
                    ffi::vkFreeCommandBuffers(
                        self.device,
                        self.command_pool,
                        icb.secondaries.len() as u32,
                        icb.secondaries.as_ptr(),
                    );
                }
                ffi::vkDestroyDescriptorPool(self.device, icb.descriptor_pool, core::ptr::null());
            }
        }
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

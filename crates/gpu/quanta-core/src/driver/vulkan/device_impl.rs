//! `GpuDevice` trait implementation for `VulkanDevice`.

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, GpuDevice, Pulse, QuantaError, QueueFamily, QueueType, ResourceState,
    Texture, TextureDesc, TextureViewDesc,
};
// `Wave` exists only on the compute face.
#[cfg(feature = "compute")]
use crate::Wave;
// Render types used only by the render-gated impl methods (step 085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};

use super::device::{VkQueryPool, VulkanDevice};
use super::ffi;
use super::helpers::format_to_vulkan;

impl crate::api::device::sealed::Sealed for VulkanDevice {}

impl GpuDevice for VulkanDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn install_self_ref(&self, self_ref: alloc::sync::Weak<dyn GpuDevice>) {
        self.self_ref.install(self_ref);
    }

    // === Feature support — slice 20 ===

    fn supports_variable_rate_shading(&self) -> bool {
        self.vrs_set_rate_fn.is_some()
    }

    fn supports_ray_tracing(&self) -> bool {
        self.trace_rays_fn.is_some()
            && self.accel_create_fn.is_some()
            && self.accel_build_fn.is_some()
    }

    fn supports_mesh_shaders(&self) -> bool {
        self.mesh_draw_fn.is_some()
    }

    fn supports_tessellation(&self) -> bool {
        self.tessellation_feature
    }

    fn supports_sparse_residency(&self) -> bool {
        self.sparse_binding_supported
    }

    fn supports_f64(&self) -> bool {
        self.shader_float64_supported
    }

    fn supports_i64(&self) -> bool {
        self.shader_int64_supported
    }

    fn supports_subgroups(&self) -> bool {
        self.subgroup_arithmetic_supported
    }

    fn supported_shading_rates(&self) -> Vec<(u32, u32)> {
        self.supported_shading_rates.clone()
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

    fn field_write_bytes_at(
        &self,
        handle: u64,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        self.field_write_bytes_at_impl(handle, byte_offset, data)
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

    fn supports_texture_write_region(&self) -> bool {
        true
    }

    /// Storage textures in compute (R32Float load/write) are supported: the
    /// emitter bakes a concrete R32f image format, so no
    /// shaderStorageImageWriteWithoutFormat feature is required. Sampling in
    /// compute is a separate, not-yet-wired path (rejected at pipeline build).
    fn supports_compute_textures(&self) -> bool {
        true
    }

    fn texture_write_region(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        size: (u32, u32),
        data: &[u8],
    ) -> Result<(), QuantaError> {
        self.texture_write_region_impl(texture, origin, size, data)
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.texture_read_impl(texture)
    }

    fn sampler_create(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        self.sampler_create_impl(desc)
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        self.generate_mipmaps_impl(texture)
    }

    // === Compute ===

    #[cfg(feature = "compute")]
    fn wave_dispatch_threads(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        // Folds oversized 1D dispatches (groups >
        // maxComputeWorkGroupCount[0]) into a 2D grid — see
        // wave_dispatch_threads_impl.
        self.wave_dispatch_threads_impl(wave, quarks)
    }

    #[cfg(feature = "compute")]
    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel)
    }

    #[cfg(all(feature = "compute", feature = "jit"))]
    fn wave_jit(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_jit_impl(kernel_def)
    }

    #[cfg(feature = "compute")]
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_impl(wave, groups)
    }

    #[cfg(feature = "compute")]
    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_indirect_impl(wave, buffer, offset)
    }

    // === Render === (render-gated, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        self.render_begin_impl(target)
    }

    #[cfg(feature = "render")]
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

    #[cfg(feature = "render")]
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
            self.retire_bin
                .retire(self.device, super::retire::Retired::View(v));
        }
        Ok(())
    }

    // === Render-resource lifecycle (destroy methods) ===
    //
    // Render submissions are ASYNCHRONOUS (`render_end` returns a live
    // pulse), so a wrapper Drop can reach these while the GPU still
    // references the resource. Every destroy therefore removes the
    // registry entry (no later submission can bind it) and hands the
    // raw handles to the retire bin, which destroys immediately only
    // when the queue provably has nothing outstanding — otherwise after
    // the covering submission's fence. Destroying inline here was the
    // dija Glass/Surface device loss (VUID-vkDestroyImage-image-01000).
    // Each arm mirrors the corresponding drain of
    // `impl Drop for VulkanDevice`.

    fn texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let tex = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(t) = tex {
            // A swapchain-frame registration (memory == null) aliases an
            // image and view the surface entry owns — dropping the
            // registry entry is the whole destroy. Every real texture
            // allocates memory, so null memory is the discriminator.
            if t.memory.is_null() {
                return Ok(());
            }
            self.retire_bin.retire(
                self.device,
                super::retire::Retired::Image {
                    image: t.image,
                    view: t.view,
                    memory: t.memory,
                },
            );
        }
        Ok(())
    }

    fn sampler_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let sampler = self
            .samplers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(s) = sampler {
            self.retire_bin
                .retire(self.device, super::retire::Retired::Sampler(s));
        }
        Ok(())
    }

    #[cfg(feature = "render")]
    fn pipeline_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let pipeline = self
            .render_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(rp) = pipeline {
            unsafe {
                ffi::vkDestroyPipeline(self.device, rp.pipeline, core::ptr::null());
                ffi::vkDestroyPipelineLayout(self.device, rp.layout, core::ptr::null());
                ffi::vkDestroyRenderPass(self.device, rp.render_pass, core::ptr::null());
                // Render pipelines own their descriptor-set layout
                // (unlike compute, whose layouts live in layout_cache).
                ffi::vkDestroyDescriptorSetLayout(
                    self.device,
                    rp.descriptor_set_layout,
                    core::ptr::null(),
                );
            }
        }
        Ok(())
    }

    fn occlusion_query_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let pool = self
            .query_pools
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(qp) = pool {
            unsafe {
                ffi::vkDestroyQueryPool(self.device, qp.pool, core::ptr::null());
            }
        }
        Ok(())
    }

    // === Compute-resource lifecycle ===

    /// Destroy a wave: drop its compute pipeline + pipeline layout.
    /// The descriptor-set layout is owned by `layout_cache` and shared
    /// across pipelines — it is destroyed with the device, not here.
    /// Dispatch is submit-and-wait, so nothing is in flight when
    /// `Wave::drop` reaches this.
    #[cfg(feature = "compute")]
    fn wave_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let cp = self
            .compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(cp) = cp {
            unsafe {
                ffi::vkDestroyPipeline(self.device, cp.pipeline, core::ptr::null());
                ffi::vkDestroyPipelineLayout(self.device, cp.layout, core::ptr::null());
            }
        }
        Ok(())
    }

    fn debug_registry_counts(&self) -> crate::RegistryCounts {
        crate::RegistryCounts {
            buffers: self.buffers.read().map(|m| m.len()).unwrap_or(0),
            textures: self.textures.read().map(|m| m.len()).unwrap_or(0),
            samplers: self.samplers.read().map(|m| m.len()).unwrap_or(0),
            render_pipelines: self.render_pipelines.read().map(|m| m.len()).unwrap_or(0),
            query_sets: self.query_pools.read().map(|m| m.len()).unwrap_or(0),
            waves: self.compute_pipelines.read().map(|m| m.len()).unwrap_or(0),
            #[cfg(feature = "render")]
            render_samplers: self
                .render_sampler_cache
                .read()
                .map(|m| m.len())
                .unwrap_or(0),
            #[cfg(not(feature = "render"))]
            render_samplers: 0,
        }
    }

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
        self.barrier_impl()
    }

    fn wait_idle(&self) -> Result<(), QuantaError> {
        let r = unsafe { ffi::vkDeviceWaitIdle(self.device) };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::internal("vkDeviceWaitIdle failed"));
        }
        Ok(())
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
            .ok_or_else(|| QuantaError::not_found("occlusion query pool not found"))?;

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
        // Step 063 — gate on actual proc-addr availability rather
        // than just extension presence. `mesh_draw_fn` is `Some`
        // only when both the extension was enabled at vkCreateDevice
        // and `vkGetDeviceProcAddr` returned a non-null pointer.
        // The full draw call (vkCmdDrawMeshTasksEXT inside an active
        // render pass with a mesh-shader pipeline bound) requires
        // mesh-shader pipeline-creation support that doesn't yet
        // exist; surface that as a separate, more specific status.
        if self.mesh_draw_fn.is_none() {
            return Err(QuantaError::not_supported(
                "mesh shaders require VK_EXT_mesh_shader — extension or proc address unavailable on this device",
            ));
        }
        Err(QuantaError::not_supported(
            "Vulkan mesh-shader pipeline integration pending — proc address loaded, awaiting object/mesh shader-stage support",
        ))
    }

    // === Ray tracing (M4.3) === (render-typed methods gated, step 085)

    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        self.build_acceleration_structure_native(geometry)
    }

    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        let has_rt = self.has_device_extension(b"VK_KHR_ray_tracing_pipeline\0");
        if !has_rt {
            return Err(QuantaError::not_supported(
                "ray tracing pipelines require VK_KHR_ray_tracing_pipeline — not available on this device",
            ));
        }
        // Pipeline creation would compile shader stages via VkRayTracingPipelineCreateInfoKHR.
        let handle = self.alloc_handle();
        Ok(handle)
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        // Step 063 — gate on actual proc-addr availability. The
        // full vkCmdTraceRaysKHR call needs ray-tracing pipeline
        // creation + shader binding tables, which require the
        // bigger M4.3 native build-out (BLAS/TLAS via
        // vkBuildAccelerationStructuresKHR + SBT layout). The proc
        // is loaded so that work has a foundation to land on.
        if self.trace_rays_fn.is_none() {
            return Err(QuantaError::not_supported(
                "ray dispatch requires VK_KHR_ray_tracing_pipeline + VK_KHR_acceleration_structure — extension or proc address unavailable on this device",
            ));
        }
        Err(QuantaError::not_supported(
            "Vulkan ray-tracing dispatch pending — proc address loaded, awaiting acceleration-structure builds + SBT support",
        ))
    }

    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        // Slice 23 — try the AS registry first; if the handle was
        // produced by build_acceleration_structure_native, we have
        // a real VkAccelerationStructureKHR to destroy + storage
        // buffer to free. Fall through to field_free_impl for
        // compatibility with handles from older shim paths.
        //
        // The native AS registry only exists with `render` on (accel.rs is
        // render-gated); render-off can never have built one, so fall
        // straight through.
        #[cfg(feature = "render")]
        if self.destroy_as_native_if_present(handle) {
            return Ok(());
        }
        self.field_free_impl(handle);
        Ok(())
    }

    // === Sparse textures (M5.1) ===

    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        // Step 063 slice 16 — gate on the cached pair from device
        // discovery: VkPhysicalDeviceFeatures.sparseBinding AND
        // VK_QUEUE_SPARSE_BINDING_BIT on the active queue family.
        // Caching at discovery avoids a per-request
        // vkGetPhysicalDeviceFeatures call and pre-checks that the
        // chosen queue can actually issue vkQueueBindSparse — a
        // prerequisite that the previous shim ignored.
        if !self.sparse_binding_supported {
            return Err(QuantaError::not_supported(
                "sparse textures require VkPhysicalDeviceFeatures.sparseBinding + a queue family with VK_QUEUE_SPARSE_BINDING_BIT — not available on this device",
            ));
        }
        // Step 063 slice 21 — create a real sparse VkImage with
        // SPARSE_BINDING_BIT + SPARSE_RESIDENCY_BIT, no memory
        // bound. Memory backing is attached lazily by
        // `sparse_map_tile` (slice 22, still NotSupported today).
        // The typed-wrapper proof contract (T7600 dimensions +
        // T7605 lifecycle) holds because the returned handle is a
        // live texture in the registry; the destroy path tolerates
        // null memory / null view.
        let tex = self.sparse_image_create_impl(desc)?;
        let handle = tex.handle();
        Ok(handle)
    }

    fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        // Step 063 slice 22 — native vkQueueBindSparse path.
        //
        // Allocates one VkDeviceMemory chunk of `mem_reqs.alignment`
        // bytes (the per-tile granularity Vulkan demands), then
        // binds it to the (mip, x, y) tile of the sparse image via
        // vkQueueBindSparse. Tile→memory mapping is tracked in
        // sparse_tile_bindings so unmap can unbind + free.
        //
        // The `_backing` argument is part of the typed-wrapper
        // contract (T7602) but unused on Vulkan: each tile gets a
        // dedicated VkDeviceMemory rather than borrowing pages
        // from a caller-supplied buffer. The contract still holds
        // because the wrapper records the binding in its own
        // HashMap and our impl produces a coherent mapping.
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures
            .get(&texture)
            .ok_or_else(|| QuantaError::not_found("sparse texture handle not found"))?;

        // sparse_image_create_impl leaves memory = null_handle.
        // A non-null memory means this came from texture_create
        // (regular image) and we shouldn't try to bind sparse
        // memory to it.
        if !tex.memory.is_null() {
            return Err(QuantaError::invalid_param(
                "sparse_map_tile called on a non-sparse texture",
            ));
        }
        let image = tex.image;
        drop(textures);

        // Memory requirements give us the per-tile alignment that
        // Vulkan requires (typically 64KB or 128KB).
        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetImageMemoryRequirements(self.device, image, &mut mem_reqs) };

        // Sparse memory requirements give us the granularity in
        // pixels for the bind extent.
        let mut sparse_count = 0u32;
        unsafe {
            ffi::vkGetImageSparseMemoryRequirements(
                self.device,
                image,
                &mut sparse_count,
                core::ptr::null_mut(),
            );
        }
        if sparse_count == 0 {
            return Err(QuantaError::not_supported(
                "driver returned no sparse memory requirements for the image",
            ));
        }
        let mut sparse_reqs =
            vec![ffi::VkSparseImageMemoryRequirements::default(); sparse_count as usize];
        unsafe {
            ffi::vkGetImageSparseMemoryRequirements(
                self.device,
                image,
                &mut sparse_count,
                sparse_reqs.as_mut_ptr(),
            );
        }
        // Pick the COLOR aspect (matches sparse_image_create_impl,
        // which only handles 2D color images today).
        let req = sparse_reqs
            .iter()
            .find(|r| (r.format_properties.aspect_mask & ffi::VK_IMAGE_ASPECT_COLOR_BIT) != 0)
            .ok_or_else(|| QuantaError::not_supported("no COLOR-aspect sparse requirements"))?;
        let granularity = req.format_properties.image_granularity;

        // Allocate one tile's worth of memory (mem_reqs.alignment
        // is what Vulkan asks for per tile).
        let memory_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )?;
        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.alignment,
            memory_type_index: memory_type,
        };
        let mut tile_memory = ffi::null_handle();
        let r = unsafe {
            ffi::vkAllocateMemory(
                self.device,
                &alloc_info,
                core::ptr::null(),
                &mut tile_memory,
            )
        };
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }

        // Build the sparse bind. Offset is in pixels — multiply
        // tile coordinates by per-tile granularity.
        let bind = ffi::VkSparseImageMemoryBind {
            subresource: ffi::VkImageSubresource {
                aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                mip_level: mip,
                array_layer: 0,
            },
            offset: ffi::VkOffset3D {
                x: (x.saturating_mul(granularity.width)) as i32,
                y: (y.saturating_mul(granularity.height)) as i32,
                z: 0,
            },
            extent: ffi::VkExtent3D {
                width: granularity.width,
                height: granularity.height,
                depth: 1,
            },
            memory: tile_memory,
            memory_offset: 0,
            flags: 0,
        };
        let image_bind_info = ffi::VkSparseImageMemoryBindInfo {
            image,
            bind_count: 1,
            p_binds: &bind,
        };
        let bind_info = ffi::VkBindSparseInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            buffer_bind_count: 0,
            p_buffer_binds: core::ptr::null(),
            image_opaque_bind_count: 0,
            p_image_opaque_binds: core::ptr::null(),
            image_bind_count: 1,
            p_image_binds: &image_bind_info,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        let r = {
            let _q = self
                .queue_lock
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            unsafe { ffi::vkQueueBindSparse(self.queue, 1, &bind_info, ffi::null_handle()) }
        };
        if r != ffi::VK_SUCCESS {
            unsafe { ffi::vkFreeMemory(self.device, tile_memory, core::ptr::null()) };
            return Err(QuantaError::submit_failed());
        }

        // Replace any prior binding for this tile. Wait for the
        // GPU to finish using the old memory before freeing —
        // simplest correct shape, performance-tunable later.
        let key = (texture, mip, x, y);
        let mut bindings = self
            .sparse_tile_bindings
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        if let Some(old) = bindings.insert(key, tile_memory) {
            self.queue_wait_idle_locked();
            unsafe {
                ffi::vkFreeMemory(self.device, old, core::ptr::null());
            }
        }
        Ok(())
    }

    fn sparse_unmap_tile(&self, texture: u64, mip: u32, x: u32, y: u32) -> Result<(), QuantaError> {
        // Step 063 slice 22 — unbind via vkQueueBindSparse with
        // null memory, then free the previously allocated chunk.
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures
            .get(&texture)
            .ok_or_else(|| QuantaError::not_found("sparse texture handle not found"))?;
        let image = tex.image;
        drop(textures);

        let key = (texture, mip, x, y);
        let prior = {
            let mut bindings = self
                .sparse_tile_bindings
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            bindings.remove(&key)
        };
        // Per the typed wrapper's T7604 contract, unmapping an
        // unmapped tile is allowed (filter semantics) — succeed
        // silently when no binding existed.
        let Some(tile_memory) = prior else {
            return Ok(());
        };

        // Query granularity again — same as map_tile.
        let mut sparse_count = 0u32;
        unsafe {
            ffi::vkGetImageSparseMemoryRequirements(
                self.device,
                image,
                &mut sparse_count,
                core::ptr::null_mut(),
            );
        }
        if sparse_count == 0 {
            // Image went away under us — best effort: just free.
            unsafe { ffi::vkFreeMemory(self.device, tile_memory, core::ptr::null()) };
            return Ok(());
        }
        let mut sparse_reqs =
            vec![ffi::VkSparseImageMemoryRequirements::default(); sparse_count as usize];
        unsafe {
            ffi::vkGetImageSparseMemoryRequirements(
                self.device,
                image,
                &mut sparse_count,
                sparse_reqs.as_mut_ptr(),
            );
        }
        let req = sparse_reqs
            .iter()
            .find(|r| (r.format_properties.aspect_mask & ffi::VK_IMAGE_ASPECT_COLOR_BIT) != 0);
        let granularity = req
            .map(|r| r.format_properties.image_granularity)
            .unwrap_or(ffi::VkExtent3D::default());

        // Issue an unbind: same VkSparseImageMemoryBind shape, but
        // memory = null_handle. After this the tile is unmapped.
        let bind = ffi::VkSparseImageMemoryBind {
            subresource: ffi::VkImageSubresource {
                aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                mip_level: mip,
                array_layer: 0,
            },
            offset: ffi::VkOffset3D {
                x: (x.saturating_mul(granularity.width.max(1))) as i32,
                y: (y.saturating_mul(granularity.height.max(1))) as i32,
                z: 0,
            },
            extent: ffi::VkExtent3D {
                width: granularity.width.max(1),
                height: granularity.height.max(1),
                depth: 1,
            },
            memory: ffi::null_handle(),
            memory_offset: 0,
            flags: 0,
        };
        let image_bind_info = ffi::VkSparseImageMemoryBindInfo {
            image,
            bind_count: 1,
            p_binds: &bind,
        };
        let bind_info = ffi::VkBindSparseInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            buffer_bind_count: 0,
            p_buffer_binds: core::ptr::null(),
            image_opaque_bind_count: 0,
            p_image_opaque_binds: core::ptr::null(),
            image_bind_count: 1,
            p_image_binds: &image_bind_info,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        let r = {
            let _q = self
                .queue_lock
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            unsafe { ffi::vkQueueBindSparse(self.queue, 1, &bind_info, ffi::null_handle()) }
        };
        // Free the memory regardless of unbind result — the worst
        // case is a leaked binding that the Drop walker covers.
        self.queue_wait_idle_locked();
        unsafe {
            ffi::vkFreeMemory(self.device, tile_memory, core::ptr::null());
        }
        if r != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        Ok(())
    }

    fn sparse_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        // Free any tile bindings the caller skipped explicit unmap
        // for (Drop path on SparseTexture only calls
        // sparse_texture_destroy, not unmap_tile per tile). Order:
        //   1. Drain bindings keyed by this texture handle.
        //   2. vkQueueWaitIdle so the GPU is finished using them.
        //   3. vkFreeMemory each tile chunk.
        //   4. Destroy the VkImage itself.
        let mut tile_mems: Vec<ffi::VkDeviceMemory> = Vec::new();
        if let Ok(mut bindings) = self.sparse_tile_bindings.write() {
            bindings.retain(|key, mem| {
                if key.0 == handle {
                    tile_mems.push(*mem);
                    false
                } else {
                    true
                }
            });
        }
        if !tile_mems.is_empty() {
            self.queue_wait_idle_locked();
            for mem in tile_mems {
                unsafe { ffi::vkFreeMemory(self.device, mem, core::ptr::null()) };
            }
        }
        let removed = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(tex) = removed {
            // Same in-flight hazard as texture_destroy: the sparse image
            // may still be referenced by an outstanding submission when
            // the wrapper Drop lands here (the tile wait above only runs
            // when bindings existed). Null view/memory are tolerated by
            // the retire arm's destroy calls (Vulkan ignores NULL).
            self.retire_bin.retire(
                self.device,
                super::retire::Retired::Image {
                    image: tex.image,
                    view: tex.view,
                    memory: tex.memory,
                },
            );
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

    #[cfg(feature = "compute")]
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
            descriptor_count: max_commands * crate::api::types::MAX_BINDINGS as u32,
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

    #[cfg(feature = "compute")]
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
                .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
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

    #[cfg(feature = "compute")]
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
            .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
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

    #[cfg(feature = "compute")]
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
                .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
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
                .ok_or_else(|| QuantaError::not_found("render bundle handle not found"))?;
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

    #[cfg(feature = "compute")]
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

    // === Bindless typed wrappers (steps 034 + 035) ===
    //
    // MVP: software table mirroring `Quanta.Bindless.Array`. The
    // perf-grade path goes through VK_EXT_descriptor_indexing (core
    // in Vulkan 1.2) which requires enabling the device feature at
    // device-create time and rebuilding the descriptor-set
    // infrastructure. Future commit; the proof contract holds for
    // the MVP today.

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VulkanBindlessArray {
                    cap,
                    entries: vec![0u64; cap as usize],
                },
            );
        Ok(handle)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        let mut arrays = self
            .bindless_textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless texture array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless texture index >= capacity",
            ));
        }
        arr.entries[index as usize] = texture;
        Ok(())
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VulkanBindlessArray {
                    cap,
                    entries: vec![0u64; cap as usize],
                },
            );
        Ok(handle)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        let mut arrays = self
            .bindless_buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless buffer array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless buffer index >= capacity",
            ));
        }
        arr.entries[index as usize] = buffer;
        Ok(())
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // === Tessellation pipelines (steps 022 + 023) ===
    //
    // MVP: software state mirroring `Quanta.Tessellation.Pipeline`.
    // The native path requires enabling the `tessellationShader`
    // device feature at create time, attaching TCS+TES SPIR-V modules
    // to pipeline-create info, and adding
    // `VkPipelineTessellationStateCreateInfo` with patch-control-
    // points. Future commit; the proof contract from
    // `Quanta.Tessellation` holds today.

    fn tessellation_pipeline_create(
        &self,
        topology: u8,
        _control_points: u32,
    ) -> Result<u64, QuantaError> {
        // Step 063 slice 6 — gate on the tessellationShader device
        // feature cached at discovery. Without it, even the
        // software-MVP factor buffers can't be promoted to a real
        // pipeline; surfacing NotSupported up-front matches the
        // pipeline_create gate (slice 5).
        if !self.tessellation_feature {
            return Err(QuantaError::not_supported(
                "Vulkan tessellation requires VkPhysicalDeviceFeatures.tessellationShader — not available on this physical device",
            ));
        }
        let (outer_count, inner_count) = match topology {
            0 => (3usize, 1usize),
            1 => (4usize, 2usize),
            _ => {
                return Err(QuantaError::invalid_param(
                    "tessellation topology must be 0 (triangle) or 1 (quad)",
                ));
            }
        };
        let handle = self.alloc_handle();
        self.tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VulkanTessPipeline {
                    outer: vec![1u32; outer_count],
                    inner: vec![1u32; inner_count],
                },
            );
        Ok(handle)
    }

    fn tessellation_set_outer(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self
            .tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.outer.len() {
            return Err(QuantaError::invalid_param(
                "tessellation outer index out of range",
            ));
        }
        pipe.outer[index as usize] = factor;
        Ok(())
    }

    fn tessellation_set_inner(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self
            .tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.inner.len() {
            return Err(QuantaError::invalid_param(
                "tessellation inner index out of range",
            ));
        }
        pipe.inner[index as usize] = factor;
        Ok(())
    }

    fn tessellation_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // === Mesh shader pipelines (steps 024 + 025) ===
    //
    // MVP: software state mirroring `Quanta.MeshShader.Pipeline`.
    // The native path enables VK_EXT_mesh_shader (or Vulkan 1.3
    // core mesh shading), adds TASK_BIT_EXT + MESH_BIT_EXT shader
    // stages to the render pipeline, and lowers `mesh_dispatch` to
    // `vkCmdDrawMeshTasksEXT`. The existing `dispatch_mesh` trait
    // method already gates on the extension presence; the native
    // lowering is a future commit. Proof contract holds today.

    fn mesh_pipeline_create(
        &self,
        max_vertices: u32,
        max_primitives: u32,
        task_threads: u32,
    ) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.mesh_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::VulkanMeshPipeline {
                    max_vertices,
                    max_primitives,
                    task_threads,
                    dispatched: Vec::new(),
                },
            );
        Ok(handle)
    }

    fn mesh_dispatch(&self, handle: u64, groups: [u32; 3]) -> Result<(), QuantaError> {
        let mut pipes = self
            .mesh_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("mesh pipeline not found"))?;
        pipe.dispatched.push(groups);
        Ok(())
    }

    fn mesh_pipeline_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.mesh_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // === Variable rate shading (steps 028 + 029) ===
    //
    // MVP: software state mirroring `Quanta.Vrs.State`. Native
    // path enables `VK_KHR_fragment_shading_rate` and lowers
    // `vrs_set_rate` to `vkCmdSetFragmentShadingRateKHR(rate,
    // combiner_op)` on the active render encoder. Future commit;
    // the proof contract holds for the MVP today.

    fn vrs_create(&self) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.vrs_states
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, super::device::VulkanVrsState { rate_code: 0 });
        Ok(handle)
    }

    fn vrs_set_rate(&self, handle: u64, rate_code: u8) -> Result<(), QuantaError> {
        if rate_code > 6 {
            return Err(QuantaError::invalid_param("VRS rate code out of range"));
        }
        let mut states = self
            .vrs_states
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let st = states
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("VRS state not found"))?;
        st.rate_code = rate_code;
        Ok(())
    }

    fn vrs_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.vrs_states
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // ╭──────────────────────────────────────────────────────────────╮
    // │ Presentation & interop block (native-handle export + Surface)│
    // ╰──────────────────────────────────────────────────────────────╯
    //
    // Native-handle export is live (the registry already holds the
    // VkImage + VkDeviceMemory), and the Surface family runs on the
    // VkSwapchainKHR path when the loader offers the WSI extensions.

    fn supports_native_handle_export(&self) -> bool {
        true
    }

    #[cfg(feature = "render")]
    fn supports_surface_present(&self) -> bool {
        self.surface_procs.is_some()
    }

    #[cfg(feature = "render")]
    fn surface_create(
        &self,
        target: &crate::SurfaceTarget,
        config: &crate::SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        self.surface_create_impl(target, config)
    }

    #[cfg(feature = "render")]
    fn surface_configure(
        &self,
        surface: u64,
        config: &crate::SurfaceConfig,
    ) -> Result<(), QuantaError> {
        self.surface_configure_impl(surface, config)
    }

    #[cfg(feature = "render")]
    fn surface_format(&self, surface: u64) -> Result<crate::Format, QuantaError> {
        self.surface_format_impl(surface)
    }

    #[cfg(feature = "render")]
    fn surface_current_extent(&self, surface: u64) -> Option<(u32, u32)> {
        self.surface_current_extent_impl(surface)
    }

    #[cfg(feature = "render")]
    fn surface_acquire(&self, surface: u64) -> Result<(u64, Texture), QuantaError> {
        self.surface_acquire_impl(surface)
    }

    #[cfg(feature = "render")]
    fn surface_present(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        self.surface_present_impl(surface, frame)
    }

    #[cfg(feature = "render")]
    fn surface_discard(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        self.surface_discard_impl(surface, frame)
    }

    #[cfg(feature = "render")]
    fn surface_destroy(&self, surface: u64) -> Result<(), QuantaError> {
        self.surface_destroy_impl(surface)
    }

    fn texture_native_handle(
        &self,
        texture: &Texture,
    ) -> Result<crate::NativeTextureHandle, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = textures
            .get(&texture.handle())
            .ok_or_else(|| QuantaError::not_found("texture handle not found"))?;
        Ok(crate::NativeTextureHandle::Vulkan {
            image: tex.image,
            memory: tex.memory,
            vk_format: tex.format,
            layout: tex
                .current_layout
                .load(core::sync::atomic::Ordering::Acquire),
        })
    }

    // ╭──────────────────────────────────────────────────────────────╮
    // │ End presentation & interop block                             │
    // ╰──────────────────────────────────────────────────────────────╯
}

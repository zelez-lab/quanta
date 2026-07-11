//! GpuDevice trait implementation for MetalDevice, type conversions, and batch dispatch.

#[cfg(feature = "compute")]
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pulse, QuantaError, QueueFamily, QueueType, ResourceState,
    Texture, TextureDesc, TextureViewDesc,
};
// `Wave` and the batch plumbing exist only on the compute face.
#[cfg(feature = "compute")]
use crate::Wave;
// Render types used only by the render-gated impl methods (step 085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};

use super::device::MetalDevice;
use super::ffi;

impl crate::api::device::sealed::Sealed for MetalDevice {}

impl GpuDevice for MetalDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Feature support — slice 20 ===
    //
    // VRS via MTLRasterizationRateMap is gated on Apple7 family
    // alongside sparse, but the encoder also needs the descriptor
    // attachment path which MTLRasterizationRateMap supports
    // universally on supported families. Use sparse_supported as
    // the proxy since both gate on family Apple7.

    fn supports_variable_rate_shading(&self) -> bool {
        self.sparse_supported
    }

    fn supports_ray_tracing(&self) -> bool {
        self.ray_tracing_supported
    }

    fn supports_mesh_shaders(&self) -> bool {
        self.mesh_shader_supported
    }

    fn supports_tessellation(&self) -> bool {
        self.tessellation_supported
    }

    fn supports_sparse_residency(&self) -> bool {
        self.sparse_supported
    }

    fn supports_cooperative_matrix(&self) -> bool {
        // `simdgroup_matrix` is available on Apple GPU family 7+ (and Mac2);
        // reuse the family-7 proxy that gates sparse/VRS here.
        self.sparse_supported
    }

    fn supports_subgroups(&self) -> bool {
        // SIMD-group reductions (`simd_sum` / `simd_min` / `simd_max` /
        // prefix sums) are available on every Metal GPU Quanta targets;
        // the subgroup prims path has always run on this backend.
        true
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

    fn field_create_mapped(
        &self,
        size: usize,
        usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        self.field_create_mapped_impl(size, usage)
    }

    fn field_map(&self, handle: u64, size: usize) -> Result<*mut u8, QuantaError> {
        self.field_map_impl(handle, size)
    }

    fn field_unmap(&self, handle: u64) -> Result<(), QuantaError> {
        self.field_unmap_impl(handle)
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
    fn wave(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel_source)
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
    fn wave_dispatch_threads(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_threads_impl(wave, quarks)
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

    // === Batch ===

    #[cfg(feature = "compute")]
    fn batch_begin(&self) -> Result<crate::Batch, QuantaError> {
        let cmd = unsafe { ffi::msg_id(self.queue, b"commandBuffer\0") };
        let encoder = unsafe { ffi::msg_id(cmd, b"computeCommandEncoder\0") };
        Ok(crate::Batch {
            inner: Box::new(MetalBatch {
                device: self as *const MetalDevice,
                cmd,
                encoder,
            }),
        })
    }

    // === Render === (render-gated, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
            color_targets: Vec::new(),
            depth_target: None,
        })
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

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }

    fn wait_idle(&self) -> Result<(), QuantaError> {
        // Same-queue command buffers complete in commit order, so an
        // empty buffer committed now finishes only after everything
        // already submitted — a host-blocking queue drain.
        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }

    fn barrier_buffer(
        &self,
        _handle: u64,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    fn barrier_texture(
        &self,
        _texture: &Texture,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Timestamps ===

    fn timestamp_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        // Allocate a shared buffer to store u64 timestamp values.
        let size = count as usize * 8;
        let buf = unsafe {
            ffi::msg_new_buffer(
                self.device,
                size as u64,
                ffi::MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buf);
        Ok(handle)
    }

    fn timestamp_write(&self, query_handle: u64, index: u32) -> Result<(), QuantaError> {
        // Metal does not support inline GPU timestamp writes like Vulkan.
        // Use sampleTimestamps:gpuTimestamp: for a CPU-side approximation.
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers
            .get(&query_handle)
            .ok_or_else(|| QuantaError::invalid_param("bad query handle"))?;
        unsafe {
            let ptr = ffi::msg_ptr(*buf, b"contents\0") as *mut u64;
            let mut cpu_ts: u64 = 0;
            let mut gpu_ts: u64 = 0;
            ffi::msg_sample_timestamps(self.device, &mut cpu_ts, &mut gpu_ts);
            *ptr.add(index as usize) = gpu_ts;
        }
        Ok(())
    }

    fn timestamp_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("bad query handle"))?;
        unsafe {
            let ptr = ffi::msg_ptr(*buf, b"contents\0") as *const u64;
            let size = ffi::msg_u64(*buf, b"length\0") as usize / 8;
            let mut result = Vec::with_capacity(size);
            for i in 0..size {
                result.push(*ptr.add(i));
            }
            Ok(result)
        }
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
        let base = textures
            .get(&texture_handle)
            .ok_or_else(|| QuantaError::invalid_param("bad texture handle"))?;

        let format = match desc.format {
            Some(f) => format_to_metal(f),
            None => unsafe { ffi::msg_u64(*base, b"pixelFormat\0") },
        };

        let tex_type = unsafe { ffi::msg_u64(*base, b"textureType\0") };

        let mip_count = desc.mip_range.end.saturating_sub(desc.mip_range.start);
        let layer_count = desc.layer_range.end.saturating_sub(desc.layer_range.start);

        let view = unsafe {
            ffi::msg_new_texture_view(
                *base,
                format,
                tex_type,
                ffi::NSRange {
                    location: desc.mip_range.start as u64,
                    length: mip_count as u64,
                },
                ffi::NSRange {
                    location: desc.layer_range.start as u64,
                    length: layer_count as u64,
                },
            )
        };

        if view.is_null() {
            return Err(QuantaError::internal("Metal newTextureView returned nil"));
        }

        let handle = self.alloc_handle();
        drop(textures); // release read lock before taking write lock
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, view);
        Ok(handle)
    }

    fn texture_view_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        // Views live in the texture registry (they are MTLTextures).
        let view = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(v) = view {
            // newTextureViewWithPixelFormat:… returns +1 retained.
            unsafe { ffi::msg_void(v, b"release\0") };
        }
        Ok(())
    }

    // === Render-resource lifecycle (destroy methods) ===
    //
    // Dispatch/render submission is synchronous (submit-and-wait), so
    // nothing is in flight when a wrapper Drop reaches these; releasing
    // the +1-retained ObjC objects here is safe.

    fn texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let tex = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(t) = tex {
            // newTextureWithDescriptor: returns +1 retained.
            unsafe { ffi::msg_void(t, b"release\0") };
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
            // newSamplerStateWithDescriptor: returns +1 retained.
            unsafe { ffi::msg_void(s, b"release\0") };
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
        if let Some(p) = pipeline {
            unsafe { ffi::msg_void(p, b"release\0") };
        }
        // The paired depth/stencil state shares the pipeline handle.
        let ds = self
            .depth_stencil_states
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(d) = ds {
            unsafe { ffi::msg_void(d, b"release\0") };
        }
        Ok(())
    }

    fn occlusion_query_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        // Occlusion query sets are shared MTLBuffers in the buffer
        // registry (see occlusion_query_create).
        let buf = self
            .buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(b) = buf {
            unsafe { ffi::msg_void(b, b"release\0") };
        }
        Ok(())
    }

    // === Compute-resource lifecycle ===

    /// Destroy a wave: drop its compute pipeline state. Dispatch
    /// submission is synchronous (submit-and-wait), so nothing is in
    /// flight when `Wave::drop` reaches this.
    #[cfg(feature = "compute")]
    fn wave_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let pipeline = self
            .compute_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(p) = pipeline {
            // newComputePipelineStateWithFunction: returns +1 retained.
            unsafe { ffi::msg_void(p, b"release\0") };
        }
        Ok(())
    }

    fn debug_registry_counts(&self) -> crate::RegistryCounts {
        crate::RegistryCounts {
            buffers: self.buffers.read().map(|m| m.len()).unwrap_or(0),
            textures: self.textures.read().map(|m| m.len()).unwrap_or(0),
            samplers: self.samplers.read().map(|m| m.len()).unwrap_or(0),
            render_pipelines: self.render_pipelines.read().map(|m| m.len()).unwrap_or(0),
            query_sets: 0,
            waves: self.compute_pipelines.read().map(|m| m.len()).unwrap_or(0),
        }
    }

    // === MSAA Resolve ===

    #[cfg(feature = "render")]
    fn resolve_texture(&self, src_handle: u64, dst_handle: u64) -> Result<(), QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let src = textures
            .get(&src_handle)
            .ok_or_else(|| QuantaError::invalid_param("bad src texture handle"))?;
        let dst = textures
            .get(&dst_handle)
            .ok_or_else(|| QuantaError::invalid_param("bad dst texture handle"))?;

        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let rpd = ffi::msg_id(
                ffi::cls(b"MTLRenderPassDescriptor\0") as ffi::Id,
                b"renderPassDescriptor\0",
            );
            let color_attachments = ffi::msg_id(rpd, b"colorAttachments\0");
            let color0 = ffi::msg_id_u64(color_attachments, b"objectAtIndexedSubscript:\0", 0);
            ffi::msg_void_id(color0, b"setTexture:\0", *src);
            ffi::msg_void_id(color0, b"setResolveTexture:\0", *dst);
            ffi::msg_void_u64(color0, b"setLoadAction:\0", ffi::MTL_LOAD_ACTION_LOAD);
            ffi::msg_void_u64(
                color0,
                b"setStoreAction:\0",
                ffi::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
            );

            let encoder = ffi::msg_new_render_encoder(cmd, rpd);
            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }

    // === Multi-queue (M3.1) ===

    fn queue_families(&self) -> Vec<QueueFamily> {
        // Metal has one universal queue family that supports everything.
        vec![QueueFamily {
            queue_type: QueueType::Graphics,
            count: 4,
        }]
    }

    fn create_queue(&self, _queue_type: QueueType) -> Result<u64, QuantaError> {
        let queue = unsafe { ffi::msg_id(self.device, b"newCommandQueue\0") };
        if queue.is_null() {
            return Err(QuantaError::internal(
                "failed to create Metal command queue",
            ));
        }
        let handle = self.alloc_handle();
        self.queues
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, queue);
        Ok(handle)
    }

    fn queue_signal(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        // Metal command buffers within the same queue are ordered.
        // Cross-queue sync would use MTLSharedEvent; for now this is a no-op.
        Ok(())
    }

    fn queue_wait(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        // Same-queue ordering is implicit in Metal.
        Ok(())
    }

    // === Occlusion queries (M3.3) ===

    fn occlusion_query_create(&self, count: u32) -> Result<u64, QuantaError> {
        // Allocate a shared buffer to hold u64 visibility results.
        let size = count as u64 * 8;
        let buf = unsafe {
            ffi::msg_new_buffer(self.device, size, ffi::MTL_RESOURCE_STORAGE_MODE_SHARED)
        };
        if buf.is_null() {
            return Err(QuantaError::internal(
                "failed to create occlusion query buffer",
            ));
        }
        // Zero-initialize the buffer.
        unsafe {
            let ptr = ffi::msg_ptr(buf, b"contents\0");
            core::ptr::write_bytes(ptr, 0, size as usize);
        }
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buf);
        Ok(handle)
    }

    fn occlusion_query_read(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buf = buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::invalid_param("bad occlusion query handle"))?;
        unsafe {
            let ptr = ffi::msg_ptr(*buf, b"contents\0") as *const u64;
            let size = ffi::msg_u64(*buf, b"length\0") as usize / 8;
            let mut results = Vec::with_capacity(size);
            for i in 0..size {
                results.push(*ptr.add(i));
            }
            Ok(results)
        }
    }

    // === Mesh shaders (M4.2) ===

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        // Metal mesh shaders require MTLMeshRenderPipelineDescriptor (Metal 3, Apple M3+).
        // Check GPU family: mesh shaders need Apple GPU family 9 (M3).
        Err(QuantaError::not_supported(
            "mesh shaders require Metal 3 (Apple M3+) — not available on this device",
        ))
    }

    // === Ray tracing (M4.3) ===

    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        // Metal ray tracing requires Apple GPU family 6+ (A14/M1 and later).
        // Check via supportsFamily: with MTLGPUFamilyApple6 (= 1006).
        if !self.ray_tracing_supported {
            return Err(QuantaError::not_supported(
                "ray tracing requires Apple GPU family 6+ (A14/M1)",
            ));
        }
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }

        // Allocate a private buffer as backing storage for the acceleration structure.
        // Real implementation would use MTLAccelerationStructure APIs; for now we
        // allocate a placeholder and return its handle.
        let accel_size = geometry
            .iter()
            .map(|g| g.vertex_count as u64 * 48)
            .sum::<u64>()
            .max(256);
        let buf = unsafe {
            ffi::msg_new_buffer(
                self.device,
                accel_size,
                ffi::MTL_RESOURCE_STORAGE_MODE_PRIVATE,
            )
        };
        if buf.is_null() {
            return Err(QuantaError::internal(
                "failed to allocate acceleration structure backing",
            ));
        }
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buf);
        Ok(handle)
    }

    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        // Metal ray tracing uses compute pipelines with intersection functions.
        // Check hardware support first.
        if !self.ray_tracing_supported {
            return Err(QuantaError::not_supported(
                "ray tracing pipelines require Apple GPU family 6+ (A14/M1)",
            ));
        }
        // Pipeline creation would compile ray generation/hit/miss shaders as compute
        // functions with visible function tables. Return a handle to track the pipeline.
        let handle = self.alloc_handle();
        Ok(handle)
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        // Step 063 slice 10 — close the silent-drop. The previous
        // shim returned Ok(()) on supported hardware without
        // dispatching the intersection compute encoder, which
        // violated the no-silent-drops contract. The full path
        // (encode an intersection compute pipeline with visible
        // function tables, dispatch (width × height × 1) threads)
        // is a separate native track.
        if !self.ray_tracing_supported {
            return Err(QuantaError::not_supported(
                "ray dispatch requires Apple GPU family 6+ (A14/M1)",
            ));
        }
        Err(QuantaError::not_supported(
            "Metal ray-tracing dispatch pending — hardware supports it, but the intersection compute pipeline integration is not yet wired",
        ))
    }

    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        // Release the backing buffer.
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // === Sparse textures (M5.1) ===

    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        if !self.sparse_supported {
            return Err(QuantaError::not_supported(
                "sparse textures require Apple GPU family 7+ (A15/M2)",
            ));
        }
        self.sparse_texture_create_native(desc)
    }

    fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        _backing: u64,
    ) -> Result<(), QuantaError> {
        // _backing is part of the typed-wrapper contract (T7602)
        // but unused on Metal — placement-heap pages are committed
        // by the resource state encoder, not borrowed from a
        // caller-supplied buffer. Same shape as Vulkan slice 22.
        self.sparse_update_tile(texture, mip, x, y, /*map=*/ true)
    }

    fn sparse_unmap_tile(&self, texture: u64, mip: u32, x: u32, y: u32) -> Result<(), QuantaError> {
        self.sparse_update_tile(texture, mip, x, y, /*map=*/ false)
    }

    fn sparse_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        // Release the texture and its placement heap. The texture
        // borrows pages from the heap, so release the texture
        // first; ObjC `release` decrements the retain count, the
        // actual free happens when it hits zero. Without this
        // override the trait default no-op leaks the heap until
        // device Drop.
        let removed_tex = self
            .textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        let removed_sparse = self
            .sparse_textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        unsafe {
            if let Some(tex) = removed_tex {
                ffi::msg_void(tex, b"release\0");
            }
            if let Some(s) = removed_sparse {
                ffi::msg_void(s.heap, b"release\0");
            }
        }
        Ok(())
    }

    // === Indirect command buffers (steps 032 + 033) ===
    //
    // Refines the Lean `Quanta.Icb.execute` semantics + the Verus
    // `quanta-api/icb_safety.rs` invariants. Recording lowers each
    // dispatch into an MTLIndirectComputeCommand slot;
    // execute(count) wraps an executeCommandsInBuffer:withRange:
    // call inside a fresh compute encoder, so the GPU replays the
    // first `count` recorded commands without host re-issue.
    //
    // Limitations of the current Metal MVP:
    //   - Push constants (setBytes:length:atIndex:) are not
    //     supported on indirect command commands; recording rejects
    //     waves with non-zero push state.
    //   - Texture bindings are not yet recorded into ICB commands;
    //     recording rejects waves with bound textures.

    #[cfg(feature = "compute")]
    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        // MTLIndirectCommandBuffer is available on all Apple GPUs.
        // Set commandTypes to ConcurrentDispatch so the slots accept
        // concurrentDispatchThreadgroups: writes.
        let icb = unsafe {
            let desc = ffi::msg_new_icb_descriptor(
                ffi::MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH,
                crate::api::types::MAX_BINDINGS as ffi::NSUInteger,
            );
            ffi::msg_new_icb(
                self.device,
                desc,
                max_commands as ffi::NSUInteger,
                ffi::MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        if icb.is_null() {
            return Err(QuantaError::internal(
                "failed to create MTLIndirectCommandBuffer",
            ));
        }
        let handle = self.alloc_handle();
        self.icbs
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::MetalIcb {
                    icb,
                    cap: max_commands,
                    used_buffers: Vec::new(),
                    recorded: 0,
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
        if wave.push_mask != 0 || wave.push_len != 0 {
            return Err(QuantaError::invalid_param(
                "Metal ICB does not support push constants (setBytes); \
                 record dispatches with explicit field bindings instead",
            ));
        }
        if wave.texture_count != 0 {
            return Err(QuantaError::invalid_param(
                "Metal ICB texture bindings not yet supported",
            ));
        }
        let pipelines = self
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline = pipelines
            .get(&wave.handle)
            .copied()
            .ok_or_else(|| QuantaError::invalid_param("bad wave handle in ICB record"))?;
        drop(pipelines);

        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        // Translate slot bindings into (slot, MTLBuffer) pairs.
        let mut bound: Vec<(usize, ffi::Id, u64)> = Vec::new();
        for slot in 0..wave.binding_count as usize {
            let h = wave.bindings[slot];
            if h != 0 {
                let buf = *buffers
                    .get(&h)
                    .ok_or_else(|| QuantaError::invalid_param("bad buffer handle in ICB record"))?;
                bound.push((slot, buf, h));
            }
        }
        drop(buffers);

        let mut icbs = self
            .icbs
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let icb_state = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
        if index != icb_state.recorded {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb_state.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        unsafe {
            let cmd =
                ffi::msg_icb_compute_command_at_index(icb_state.icb, index as ffi::NSUInteger);
            if cmd.is_null() {
                return Err(QuantaError::internal(
                    "MTLIndirectCommandBuffer.indirectComputeCommandAtIndex returned null",
                ));
            }
            ffi::msg_icc_set_compute_pipeline(cmd, pipeline);
            for (slot, buf, _) in &bound {
                ffi::msg_icc_set_kernel_buffer(cmd, *buf, 0, *slot as u64);
            }
            let group_size = ffi::MTLSize::new(
                wave.workgroup_size[0] as u64,
                wave.workgroup_size[1] as u64,
                wave.workgroup_size[2] as u64,
            );
            let groups_3d = ffi::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
            ffi::msg_icc_concurrent_dispatch_threadgroups(cmd, groups_3d, group_size);
            // Sequence subsequent commands after this one. Without
            // this, two ICB dispatches that touch the same buffer
            // race — both read the initial value, both write the
            // same +1 result, second clobbers first. With the
            // barrier, command N sees command N-1's writes.
            ffi::msg_icc_set_barrier(cmd);
        }
        for (_, _, h) in bound {
            if !icb_state.used_buffers.contains(&h) {
                icb_state.used_buffers.push(h);
            }
        }
        icb_state.recorded += 1;
        Ok(())
    }

    #[cfg(feature = "compute")]
    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        // Snapshot the ICB Id + used_buffers under the lock, then
        // drop it before issuing the command buffer (which may
        // re-enter device methods).
        let (icb_id, used_buffer_handles) = {
            let icbs = self
                .icbs
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            let icb_state = icbs
                .get(&handle)
                .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
            if count > icb_state.recorded {
                return Err(QuantaError::invalid_param(
                    "ICB execute count exceeds recorded length",
                ));
            }
            (icb_state.icb, icb_state.used_buffers.clone())
        };
        if count == 0 {
            return Ok(());
        }
        // Resolve buffer Ids while holding the buffers lock briefly.
        let used_buffer_ids: Vec<ffi::Id> = {
            let buffers = self
                .buffers
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            used_buffer_handles
                .iter()
                .filter_map(|h| buffers.get(h).copied())
                .collect()
        };
        unsafe {
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let encoder = ffi::msg_id(cmd, b"computeCommandEncoder\0");
            const MTL_RESOURCE_USAGE_READ: ffi::NSUInteger = 1;
            const MTL_RESOURCE_USAGE_WRITE: ffi::NSUInteger = 2;
            for buf in &used_buffer_ids {
                ffi::msg_use_resource(
                    encoder,
                    *buf,
                    MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                );
            }
            let range = ffi::NSRange {
                location: 0,
                length: count as u64,
            };
            ffi::msg_execute_commands_in_buffer(encoder, icb_id, range);
            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");
        }
        Ok(())
    }

    #[cfg(feature = "compute")]
    fn icb_record_draw(
        &self,
        _handle: u64,
        _index: u32,
        _pipeline: u64,
        _vertex_count: u32,
        _instance_count: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "use IndirectRenderBundle (gpu.render_bundle()) for Metal \
             render-path ICB; the dispatch ICB cannot mix DRAW commands",
        ))
    }

    // === Indirect render bundles (Metal native MTLIndirectRenderCommand) ===

    fn render_bundle_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        // Allocate an ICB whose command type is DRAW. Distinct from
        // the compute ICB's ConcurrentDispatch type.
        let icb = unsafe {
            let desc = ffi::msg_new_icb_descriptor(
                ffi::MTL_INDIRECT_COMMAND_TYPE_DRAW,
                crate::api::types::MAX_BINDINGS as ffi::NSUInteger,
            );
            ffi::msg_new_icb(
                self.device,
                desc,
                max_commands as ffi::NSUInteger,
                ffi::MTL_RESOURCE_STORAGE_MODE_SHARED,
            )
        };
        if icb.is_null() {
            return Err(QuantaError::internal(
                "failed to create MTLIndirectCommandBuffer (DRAW)",
            ));
        }
        let handle = self.alloc_handle();
        self.render_bundles
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::MetalRenderBundle {
                    icb,
                    cap: max_commands,
                    recorded: 0,
                    used_buffers: Vec::new(),
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
        let render_pipelines = self
            .render_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline_id = *render_pipelines
            .get(&pipeline)
            .ok_or_else(|| QuantaError::invalid_param("bad pipeline handle in render bundle"))?;
        drop(render_pipelines);

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
        unsafe {
            let cmd = ffi::msg_icb_render_command_at_index(bundle.icb, index as ffi::NSUInteger);
            ffi::msg_irc_set_render_pipeline(cmd, pipeline_id);
            ffi::msg_irc_draw_primitives(
                cmd,
                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                0,
                vertex_count as u64,
                instance_count.max(1) as u64,
                0,
            );
        }
        bundle.recorded += 1;
        Ok(())
    }

    fn render_bundle_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let removed = self
            .render_bundles
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(bundle) = removed {
            unsafe {
                ffi::msg_void(bundle.icb, b"release\0");
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
        if let Some(state) = removed {
            unsafe {
                ffi::msg_void(state.icb, b"release\0");
            }
        }
        Ok(())
    }

    // === Bindless typed wrappers (steps 034 + 035) ===
    //
    // Metal argument buffers (Tier 2 on M1+) hold an array of
    // resource IDs. We allocate a shared MTLBuffer sized for `cap`
    // 8-byte slots; `update(index, handle)` writes the resource's
    // gpuResourceID into slot `index`. Shaders index this argument
    // buffer to access resources bindlessly.

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let size = ((cap.max(1) as usize) * 8) as u64;
        let buf = unsafe {
            ffi::msg_new_buffer(self.device, size, ffi::MTL_RESOURCE_STORAGE_MODE_SHARED)
        };
        if buf.is_null() {
            return Err(QuantaError::internal(
                "failed to create Metal argument buffer for bindless textures",
            ));
        }
        unsafe {
            let ptr = ffi::msg_ptr(buf, b"contents\0");
            core::ptr::write_bytes(ptr, 0, size as usize);
        }
        let handle = self.alloc_handle();
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buf);
        Ok(handle)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let arg_buf = *buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless texture array handle not found"))?;
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let tex = *textures
            .get(&texture)
            .ok_or_else(|| QuantaError::invalid_param("bindless: bad texture handle"))?;
        let cap = unsafe { ffi::msg_u64(arg_buf, b"length\0") } / 8;
        if (index as u64) >= cap {
            return Err(QuantaError::invalid_param(
                "bindless texture index >= capacity",
            ));
        }
        unsafe {
            // Read MTLResourceID for the texture.
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel) -> u64 =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            let res_id = f(tex, ffi::sel(b"gpuResourceID\0"));
            let ptr = ffi::msg_ptr(arg_buf, b"contents\0") as *mut u64;
            *ptr.add(index as usize) = res_id;
        }
        Ok(())
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let removed = self
            .buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(buf) = removed {
            unsafe {
                ffi::msg_void(buf, b"release\0");
            }
        }
        Ok(())
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        // Same shape as bindless_texture_create — argument buffer
        // holds resource IDs (8 bytes each).
        self.bindless_texture_create(cap)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let arg_buf = *buffers
            .get(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless buffer array handle not found"))?;
        let target = *buffers
            .get(&buffer)
            .ok_or_else(|| QuantaError::invalid_param("bindless: bad buffer handle"))?;
        let cap = unsafe { ffi::msg_u64(arg_buf, b"length\0") } / 8;
        if (index as u64) >= cap {
            return Err(QuantaError::invalid_param(
                "bindless buffer index >= capacity",
            ));
        }
        unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel) -> u64 =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            let res_id = f(target, ffi::sel(b"gpuAddress\0"));
            let ptr = ffi::msg_ptr(arg_buf, b"contents\0") as *mut u64;
            *ptr.add(index as usize) = res_id;
        }
        Ok(())
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_texture_destroy(handle)
    }

    // === Tessellation pipelines (steps 022 + 023) ===
    //
    // Metal has no fixed-function tessellator. The factor buffer
    // we allocate here is the real `MTLBuffer` that a future
    // render-pipeline integration will bind via
    // `setTessellationFactorBuffer:offset:instanceStride:`. Each
    // factor write goes straight into the buffer's host-visible
    // contents pointer.
    //
    // MVP: factors are stored as u32 (matching the typed wrapper
    // input). Hardware consumes `MTL{Triangle,Quad}TessellationFactorsHalf`
    // (u16); the conversion + drawIndexedPatches wiring lands when
    // the render path is rebuilt to include tessellation. The proof
    // contract from `Quanta.Tessellation` holds today.

    fn tessellation_pipeline_create(
        &self,
        topology: u8,
        _control_points: u32,
    ) -> Result<u64, QuantaError> {
        // Slice 8 + 17 — cached supportsFamily:Apple4 check.
        if !self.tessellation_supported {
            return Err(QuantaError::not_supported(
                "Metal tessellation requires Apple GPU family 4+ (A11+) — not available on this device",
            ));
        }
        let (outer_count, inner_count) = match topology {
            0 => (3u32, 1u32),
            1 => (4u32, 2u32),
            _ => {
                return Err(QuantaError::invalid_param(
                    "tessellation topology must be 0 (triangle) or 1 (quad)",
                ));
            }
        };
        // 4 bytes per factor (u32). Layout: outer[..outer_count], then
        // inner[..inner_count]. Initialized to 1 (no subdivision).
        let total_factors = (outer_count + inner_count) as usize;
        let size = (total_factors * 4) as u64;
        let buf = unsafe {
            ffi::msg_new_buffer(self.device, size, ffi::MTL_RESOURCE_STORAGE_MODE_SHARED)
        };
        if buf.is_null() {
            return Err(QuantaError::internal(
                "failed to create Metal tessellation factor buffer",
            ));
        }
        unsafe {
            let ptr = ffi::msg_ptr(buf, b"contents\0") as *mut u32;
            for i in 0..total_factors {
                *ptr.add(i) = 1;
            }
        }
        let handle = self.alloc_handle();
        self.tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::MetalTessPipeline {
                    factor_buf: buf,
                    outer_count,
                    inner_count,
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
        let pipes = self
            .tess_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipe = pipes
            .get(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if index >= pipe.outer_count {
            return Err(QuantaError::invalid_param(
                "tessellation outer index out of range",
            ));
        }
        unsafe {
            let ptr = ffi::msg_ptr(pipe.factor_buf, b"contents\0") as *mut u32;
            *ptr.add(index as usize) = factor;
        }
        Ok(())
    }

    fn tessellation_set_inner(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let pipes = self
            .tess_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipe = pipes
            .get(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if index >= pipe.inner_count {
            return Err(QuantaError::invalid_param(
                "tessellation inner index out of range",
            ));
        }
        unsafe {
            let ptr = ffi::msg_ptr(pipe.factor_buf, b"contents\0") as *mut u32;
            *ptr.add((pipe.outer_count + index) as usize) = factor;
        }
        Ok(())
    }

    fn tessellation_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        let removed = self
            .tess_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        if let Some(pipe) = removed {
            unsafe {
                ffi::msg_void(pipe.factor_buf, b"release\0");
            }
        }
        Ok(())
    }

    // === Mesh shader pipelines (steps 024 + 025) ===
    //
    // MVP: software state mirroring `Quanta.MeshShader.Pipeline`.
    // The native path requires building an
    // `MTLMeshRenderPipelineDescriptor` with object + mesh + fragment
    // functions and replacing the classical vertex stage with the
    // mesh path; future commit. The proof contract holds today.

    fn mesh_pipeline_create(
        &self,
        max_vertices: u32,
        max_primitives: u32,
        task_threads: u32,
    ) -> Result<u64, QuantaError> {
        // Slice 9 + 17 — cached supportsFamily:Metal3 check.
        if !self.mesh_shader_supported {
            return Err(QuantaError::not_supported(
                "Metal mesh shaders require Metal 3 (MTLGPUFamilyMetal3) — not available on this device",
            ));
        }
        let handle = self.alloc_handle();
        self.mesh_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                super::device::MetalMeshPipeline {
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
    // MVP: software state mirroring `Quanta.Vrs.State`. Native path
    // builds an `MTLRasterizationRateMap` per render pass on Apple
    // Silicon; that lowering lands when the render encoder is
    // rebuilt to consume rate maps. Proof contract holds today.

    fn vrs_create(&self) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.vrs_states
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, super::device::MetalVrsState { rate_code: 0 });
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

    fn supports_native_handle_export(&self) -> bool {
        true
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
        Ok(crate::NativeTextureHandle::Metal { texture: *tex })
    }

    fn supports_surface_present(&self) -> bool {
        true
    }

    #[cfg(feature = "render")]
    fn surface_create(
        &self,
        target: &crate::surface::SurfaceTarget,
        config: &crate::surface::SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        self.surface_create_impl(target, config)
    }

    #[cfg(feature = "render")]
    fn surface_configure(
        &self,
        surface: u64,
        config: &crate::surface::SurfaceConfig,
    ) -> Result<(), QuantaError> {
        self.surface_configure_impl(surface, config)
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

    // ╭──────────────────────────────────────────────────────────────╮
    // │ End presentation & interop block                             │
    // ╰──────────────────────────────────────────────────────────────╯
}

// ============================================================================
// Metal type conversions
// ============================================================================

pub(crate) fn format_to_metal(format: Format) -> ffi::NSUInteger {
    match format {
        Format::RGBA8 => ffi::MTL_PIXEL_FORMAT_RGBA8_UNORM,
        Format::BGRA8 => ffi::MTL_PIXEL_FORMAT_BGRA8_UNORM,
        Format::R8 => ffi::MTL_PIXEL_FORMAT_R8_UNORM,
        Format::R16Float => ffi::MTL_PIXEL_FORMAT_R16_FLOAT,
        Format::R32Float => ffi::MTL_PIXEL_FORMAT_R32_FLOAT,
        Format::RG32Float => ffi::MTL_PIXEL_FORMAT_RG32_FLOAT,
        Format::RGBA16Float => ffi::MTL_PIXEL_FORMAT_RGBA16_FLOAT,
        Format::RGBA32Float => ffi::MTL_PIXEL_FORMAT_RGBA32_FLOAT,
        Format::Depth32Float => ffi::MTL_PIXEL_FORMAT_DEPTH32_FLOAT,
        Format::Bc1Rgba => ffi::MTL_PIXEL_FORMAT_BC1_RGBA,
        Format::Bc3Rgba => ffi::MTL_PIXEL_FORMAT_BC3_RGBA,
        Format::Bc5Rg => ffi::MTL_PIXEL_FORMAT_BC5_RG_SNORM,
        Format::Bc7Rgba => ffi::MTL_PIXEL_FORMAT_BC7_RGBA_UNORM,
        Format::Astc4x4 => ffi::MTL_PIXEL_FORMAT_ASTC_4X4_LDR,
        Format::Astc6x6 => ffi::MTL_PIXEL_FORMAT_ASTC_6X6_LDR,
        Format::Astc8x8 => ffi::MTL_PIXEL_FORMAT_ASTC_8X8_LDR,
        Format::Etc2Rgb8 => ffi::MTL_PIXEL_FORMAT_ETC2_RGB8,
        Format::Etc2Rgba8 => ffi::MTL_PIXEL_FORMAT_EAC_RGBA8,
    }
}

pub(crate) fn format_bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8 => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Depth32Float => 4,
        Format::Bc1Rgba | Format::Etc2Rgb8 => 8,
        Format::Bc3Rgba | Format::Bc5Rg | Format::Bc7Rgba | Format::Etc2Rgba8 => 16,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
    }
}

pub(crate) fn filter_to_metal(f: crate::texture::Filter) -> ffi::NSUInteger {
    match f {
        crate::texture::Filter::Nearest => ffi::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        crate::texture::Filter::Linear => ffi::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
    }
}

pub(crate) fn address_to_metal(a: crate::texture::AddressMode) -> ffi::NSUInteger {
    match a {
        crate::texture::AddressMode::ClampToEdge => ffi::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        crate::texture::AddressMode::Repeat => ffi::MTL_SAMPLER_ADDRESS_MODE_REPEAT,
        crate::texture::AddressMode::MirrorRepeat => ffi::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT,
    }
}

pub(crate) fn compare_op_to_metal(op: crate::CompareOp) -> ffi::NSUInteger {
    use crate::CompareOp::*;
    match op {
        Never => ffi::MTL_COMPARE_NEVER,
        Less => ffi::MTL_COMPARE_LESS,
        Equal => ffi::MTL_COMPARE_EQUAL,
        LessEqual => ffi::MTL_COMPARE_LESS_EQUAL,
        Greater => ffi::MTL_COMPARE_GREATER,
        NotEqual => ffi::MTL_COMPARE_NOT_EQUAL,
        GreaterEqual => ffi::MTL_COMPARE_GREATER_EQUAL,
        Always => ffi::MTL_COMPARE_ALWAYS,
    }
}

#[cfg(feature = "render")]
pub(crate) fn stencil_op_to_metal(op: crate::StencilOp) -> ffi::NSUInteger {
    use crate::StencilOp::*;
    match op {
        Keep => ffi::MTL_STENCIL_OP_KEEP,
        Zero => ffi::MTL_STENCIL_OP_ZERO,
        Replace => ffi::MTL_STENCIL_OP_REPLACE,
        IncrementClamp => ffi::MTL_STENCIL_OP_INCREMENT_CLAMP,
        DecrementClamp => ffi::MTL_STENCIL_OP_DECREMENT_CLAMP,
        Invert => ffi::MTL_STENCIL_OP_INVERT,
        IncrementWrap => ffi::MTL_STENCIL_OP_INCREMENT_WRAP,
        DecrementWrap => ffi::MTL_STENCIL_OP_DECREMENT_WRAP,
    }
}

#[cfg(feature = "render")]
pub(crate) fn blend_factor_to_metal(f: crate::BlendFactor) -> ffi::NSUInteger {
    use crate::BlendFactor::*;
    match f {
        Zero => ffi::MTL_BLEND_FACTOR_ZERO,
        One => ffi::MTL_BLEND_FACTOR_ONE,
        SrcAlpha => ffi::MTL_BLEND_FACTOR_SRC_ALPHA,
        OneMinusSrcAlpha => ffi::MTL_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        DstAlpha => ffi::MTL_BLEND_FACTOR_DST_ALPHA,
        OneMinusDstAlpha => ffi::MTL_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
        SrcColor => ffi::MTL_BLEND_FACTOR_SRC_COLOR,
        OneMinusSrcColor => ffi::MTL_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
        DstColor => ffi::MTL_BLEND_FACTOR_DST_COLOR,
        OneMinusDstColor => ffi::MTL_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
    }
}

#[cfg(feature = "render")]
pub(crate) fn blend_op_to_metal(op: crate::BlendOp) -> ffi::NSUInteger {
    use crate::BlendOp::*;
    match op {
        Add => ffi::MTL_BLEND_OP_ADD,
        Subtract => ffi::MTL_BLEND_OP_SUBTRACT,
        ReverseSubtract => ffi::MTL_BLEND_OP_REVERSE_SUBTRACT,
        Min => ffi::MTL_BLEND_OP_MIN,
        Max => ffi::MTL_BLEND_OP_MAX,
    }
}

// ── Batched dispatch ────────────────────────────────────────────────────────

#[cfg(feature = "compute")]
struct MetalBatch {
    device: *const MetalDevice,
    cmd: ffi::Id,
    encoder: ffi::Id,
}

#[cfg(feature = "compute")]
impl crate::batch::BatchInner for MetalBatch {
    fn encode_dispatch(&mut self, wave: &Wave, quarks: u32) -> Result<(), QuantaError> {
        let device = unsafe { &*self.device };
        let pipelines = device
            .compute_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let pipeline = pipelines
            .get(&wave.handle)
            .ok_or_else(|| QuantaError::invalid_param("bad wave handle"))?;

        unsafe {
            ffi::msg_void_id(self.encoder, b"setComputePipelineState:\0", *pipeline);
        }

        let buffers = device
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        for slot in 0..wave.binding_count as usize {
            let handle = wave.bindings[slot];
            if handle != 0
                && let Some(buf) = buffers.get(&handle)
            {
                unsafe {
                    ffi::msg_set_buffer(
                        self.encoder,
                        b"setBuffer:offset:atIndex:\0",
                        *buf,
                        0,
                        slot as u64,
                    );
                }
            }
        }

        // Push constants
        let mut mask = wave.push_mask;
        while mask != 0 {
            let slot = mask.trailing_zeros() as usize;
            let offset = slot * 16;
            let remaining = wave.push_len as usize - offset;
            let len = remaining.min(16);
            unsafe {
                ffi::msg_set_bytes(
                    self.encoder,
                    b"setBytes:length:atIndex:\0",
                    wave.push_data[offset..].as_ptr() as *const _,
                    len as u64,
                    slot as u64,
                );
            }
            mask &= mask - 1;
        }

        let groups_x = quarks.div_ceil(wave.workgroup_size[0]);
        let grid = ffi::MTLSize::new(groups_x as u64, 1, 1);
        let group_size = ffi::MTLSize::new(
            wave.workgroup_size[0] as u64,
            wave.workgroup_size[1] as u64,
            wave.workgroup_size[2] as u64,
        );
        unsafe {
            ffi::msg_dispatch_threadgroups(self.encoder, grid, group_size);
        }
        Ok(())
    }

    fn submit(self: Box<Self>) -> Result<Pulse, QuantaError> {
        unsafe {
            ffi::msg_void(self.encoder, b"endEncoding\0");
        }
        let device = unsafe { &*self.device };
        Ok(super::device::make_async_pulse(device, self.cmd))
    }
}

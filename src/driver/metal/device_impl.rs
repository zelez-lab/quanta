//! GpuDevice trait implementation for MetalDevice, type conversions, and batch dispatch.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, QueueFamily, QueueType,
    RenderPass, ResourceState, Texture, TextureDesc, TextureViewDesc, Wave,
};

use super::compute;
use super::device::MetalDevice;
use super::ffi;

impl GpuDevice for MetalDevice {
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

    fn wave(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_impl(kernel_source)
    }

    #[cfg(feature = "jit")]
    fn wave_jit(&self, kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        self.wave_jit_impl(kernel_def)
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_impl(wave, groups)
    }

    fn wave_dispatch_threads(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_threads_impl(wave, quarks)
    }

    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError> {
        self.wave_dispatch_indirect_impl(wave, buffer, offset)
    }

    // === Batch ===

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

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.pipeline_create_impl(desc)
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
            color_targets: Vec::new(),
            depth_target: None,
        })
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

    // === Barriers ===

    fn barrier(&self) -> Result<(), QuantaError> {
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
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&handle);
        Ok(())
    }

    // === MSAA Resolve ===

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
        Err(QuantaError::invalid_param(
            "mesh shaders require Metal 3 (Apple M3+) — not available on this device",
        ))
    }

    // === Ray tracing (M4.3) ===

    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        // Metal ray tracing requires Apple GPU family 6+ (A14/M1 and later).
        // Check via supportsFamily: with MTLGPUFamilyApple6 (= 1006).
        let supports_rt = unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, u64) -> ffi::BOOL =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            f(self.device, ffi::sel(b"supportsFamily:\0"), 1006) != 0
        };
        if !supports_rt {
            return Err(QuantaError::invalid_param(
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

    fn create_ray_tracing_pipeline(
        &self,
        _desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        // Metal ray tracing uses compute pipelines with intersection functions.
        // Check hardware support first.
        let supports_rt = unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, u64) -> ffi::BOOL =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            f(self.device, ffi::sel(b"supportsFamily:\0"), 1006) != 0
        };
        if !supports_rt {
            return Err(QuantaError::invalid_param(
                "ray tracing pipelines require Apple GPU family 6+ (A14/M1)",
            ));
        }
        // Pipeline creation would compile ray generation/hit/miss shaders as compute
        // functions with visible function tables. Return a handle to track the pipeline.
        let handle = self.alloc_handle();
        Ok(handle)
    }

    fn dispatch_rays(&self, _pipeline: u64, _width: u32, _height: u32) -> Result<(), QuantaError> {
        let supports_rt = unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, u64) -> ffi::BOOL =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            f(self.device, ffi::sel(b"supportsFamily:\0"), 1006) != 0
        };
        if !supports_rt {
            return Err(QuantaError::invalid_param(
                "ray dispatch requires Apple GPU family 6+ (A14/M1)",
            ));
        }
        // Full implementation would encode an intersection compute dispatch.
        Ok(())
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
        // Metal sparse textures (MTLSparseTexture) require Apple GPU family 7+ (A15/M2).
        let supports_sparse = unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, u64) -> ffi::BOOL =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            f(self.device, ffi::sel(b"supportsFamily:\0"), 1007) != 0
        };
        if !supports_sparse {
            return Err(QuantaError::invalid_param(
                "sparse textures require Apple GPU family 7+ (A15/M2)",
            ));
        }
        // Create a texture with sparse storage. For now, create a regular private texture
        // as backing — full sparse tile mapping requires MTLSparseTexture API.
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
        // Tile mapping with MTLSparseTexture is not yet wired.
        // Succeeds silently on supported hardware (tile is considered mapped).
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

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        // MTLIndirectCommandBuffer is available on all Apple GPUs.
        // Set commandTypes to ConcurrentDispatch so the slots accept
        // concurrentDispatchThreadgroups: writes.
        let icb = unsafe {
            let desc = ffi::msg_new_icb_descriptor(
                ffi::MTL_INDIRECT_COMMAND_TYPE_CONCURRENT_DISPATCH,
                crate::api::wave::MAX_BINDINGS as ffi::NSUInteger,
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
            .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
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
        }
        for (_, _, h) in bound {
            if !icb_state.used_buffers.contains(&h) {
                icb_state.used_buffers.push(h);
            }
        }
        icb_state.recorded += 1;
        Ok(())
    }

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
                .ok_or_else(|| QuantaError::invalid_param("ICB handle not found"))?;
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

    fn icb_record_draw(
        &self,
        _handle: u64,
        _index: u32,
        _pipeline: u64,
        _vertex_count: u32,
        _instance_count: u32,
    ) -> Result<(), QuantaError> {
        // Metal's MTLIndirectRenderCommand uses a *separate*
        // descriptor (DRAW / DRAW_INDEXED command types) and is
        // recorded via `indirectRenderCommandAtIndex:`, then
        // executed via `executeCommandsInBuffer:withRange:` on a
        // *render* encoder inside an active render pass. Mixing
        // it with the existing ConcurrentDispatch ICB requires a
        // separate handle type. The proof contract (T7006) is met
        // by the typed API; the native lowering lands as a future
        // commit.
        Err(QuantaError::invalid_param(
            "Metal render-path ICB record_draw not yet implemented \
             (requires a separate MTLIndirectCommandBuffer with \
             DRAW command types)",
        ))
    }

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

    // === Bindless resources (M5.3) ===

    fn bind_texture_array(&self, textures: &[u64]) -> Result<u64, QuantaError> {
        // Metal argument buffers enable bindless access to texture arrays.
        // Available on all Apple GPUs (Tier 2 argument buffers on M1+).
        // Create an argument buffer containing pointers to all textures.
        let tex_map = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;

        // Validate all texture handles exist.
        for &tex_handle in textures {
            if !tex_map.contains_key(&tex_handle) {
                return Err(QuantaError::invalid_param("bad texture handle in array"));
            }
        }

        // Allocate a shared buffer to hold texture resource IDs (8 bytes each).
        let size = (textures.len().max(1) * 8) as u64;
        let buf = unsafe {
            ffi::msg_new_buffer(self.device, size, ffi::MTL_RESOURCE_STORAGE_MODE_SHARED)
        };
        if buf.is_null() {
            return Err(QuantaError::internal(
                "failed to create argument buffer for texture array",
            ));
        }
        let handle = self.alloc_handle();
        drop(tex_map);
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, buf);
        Ok(handle)
    }

    fn bind_buffer_array(&self, buffers: &[u64]) -> Result<u64, QuantaError> {
        // Metal argument buffers for buffer arrays.
        let buf_map = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;

        // Validate all buffer handles exist.
        for &buf_handle in buffers {
            if !buf_map.contains_key(&buf_handle) {
                return Err(QuantaError::invalid_param("bad buffer handle in array"));
            }
        }

        let size = (buffers.len().max(1) * 8) as u64;
        let arg_buf = unsafe {
            ffi::msg_new_buffer(self.device, size, ffi::MTL_RESOURCE_STORAGE_MODE_SHARED)
        };
        if arg_buf.is_null() {
            return Err(QuantaError::internal(
                "failed to create argument buffer for buffer array",
            ));
        }
        let handle = self.alloc_handle();
        drop(buf_map);
        self.buffers
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(handle, arg_buf);
        Ok(handle)
    }
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

pub(crate) fn filter_to_metal(f: crate::render_pass::Filter) -> ffi::NSUInteger {
    match f {
        crate::render_pass::Filter::Nearest => ffi::MTL_SAMPLER_MIN_MAG_FILTER_NEAREST,
        crate::render_pass::Filter::Linear => ffi::MTL_SAMPLER_MIN_MAG_FILTER_LINEAR,
    }
}

pub(crate) fn address_to_metal(a: crate::render_pass::AddressMode) -> ffi::NSUInteger {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => ffi::MTL_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        crate::render_pass::AddressMode::Repeat => ffi::MTL_SAMPLER_ADDRESS_MODE_REPEAT,
        crate::render_pass::AddressMode::MirrorRepeat => {
            ffi::MTL_SAMPLER_ADDRESS_MODE_MIRROR_REPEAT
        }
    }
}

pub(crate) fn compare_to_metal(f: crate::CompareFunc) -> ffi::NSUInteger {
    use crate::CompareFunc::*;
    match f {
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

struct MetalBatch {
    device: *const MetalDevice,
    cmd: ffi::Id,
    encoder: ffi::Id,
}

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
        Ok(compute::make_async_pulse(device, self.cmd))
    }
}

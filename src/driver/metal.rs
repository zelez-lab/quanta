//! Metal driver for macOS/iOS.

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture,
    TextureDesc, TextureUsage, Vendor, Wave, render_pass::RenderOp,
};
use metal as mtl;
use std::collections::HashMap;
use std::sync::Mutex;

/// Metal-backed GPU device.
pub struct MetalDevice {
    device: mtl::Device,
    queue: mtl::CommandQueue,
    caps: Caps,
    // Resource storage — keyed by handle
    buffers: Mutex<HashMap<u64, mtl::Buffer>>,
    textures: Mutex<HashMap<u64, mtl::Texture>>,
    compute_pipelines: Mutex<HashMap<u64, mtl::ComputePipelineState>>,
    render_pipelines: Mutex<HashMap<u64, mtl::RenderPipelineState>>,
    depth_stencil_states: Mutex<HashMap<u64, mtl::DepthStencilState>>,
    next_handle: Mutex<u64>,
}

impl MetalDevice {
    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        *h += 1;
        *h
    }
}

/// Discover Metal devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let Some(device) = mtl::Device::system_default() else {
        return Vec::new();
    };

    let name = device.name().to_string();
    let max_threads = device.max_threads_per_threadgroup();
    let caps = Caps {
        nuclei: (max_threads.width / 32).max(1) as u32,
        protons_per_nucleus: 32,
        quarks_per_proton: 32,
        memory_bytes: device.recommended_max_working_set_size(),
        max_quarks_per_dispatch: u32::MAX,
        max_groups: [u32::MAX, u32::MAX, u32::MAX],
        vendor: Vendor::Apple,
        name,
    };

    let queue = device.new_command_queue();

    vec![Box::new(MetalDevice {
        device,
        queue,
        caps,
        buffers: Mutex::new(HashMap::new()),
        textures: Mutex::new(HashMap::new()),
        compute_pipelines: Mutex::new(HashMap::new()),
        render_pipelines: Mutex::new(HashMap::new()),
        depth_stencil_states: Mutex::new(HashMap::new()),
        next_handle: Mutex::new(0),
    })]
}

impl GpuDevice for MetalDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        let options = if usage.has(FieldUsage::TRANSFER) {
            mtl::MTLResourceOptions::StorageModeShared
        } else {
            mtl::MTLResourceOptions::StorageModePrivate
        };
        let buffer = self.device.new_buffer(size as u64, options);
        let handle = self.alloc_handle();
        self.buffers.lock().unwrap().insert(handle, buffer);
        Ok(handle)
    }

    fn field_free(&self, handle: u64) {
        self.buffers.lock().unwrap().remove(&handle);
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_write_bytes: handle {handle}"))
        })?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.contents() as *mut u8, data.len());
        }
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buffer = buffers.get(&handle).ok_or_else(|| {
            QuantaError::invalid_param("bad field handle")
                .with_context(&format!("field_read_bytes: handle {handle}"))
        })?;
        let mut result = vec![0u8; size];
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.contents() as *const u8,
                result.as_mut_ptr(),
                size,
            );
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
        let cmd = self.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(src_buf, 0, dst_buf, 0, size as u64);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let mtl_desc = mtl::TextureDescriptor::new();
        mtl_desc.set_width(desc.width as u64);
        mtl_desc.set_height(desc.height as u64);
        mtl_desc.set_pixel_format(format_to_metal(desc.format));
        mtl_desc.set_sample_count(desc.sample_count as u64);

        let mut usage = mtl::MTLTextureUsage::empty();
        if desc.usage.has(TextureUsage::SHADER_READ) {
            usage |= mtl::MTLTextureUsage::ShaderRead;
        }
        if desc.usage.has(TextureUsage::SHADER_WRITE) {
            usage |= mtl::MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.has(TextureUsage::RENDER_TARGET) {
            usage |= mtl::MTLTextureUsage::RenderTarget;
        }
        if usage.is_empty() {
            usage = mtl::MTLTextureUsage::ShaderRead;
        }
        mtl_desc.set_usage(usage);

        // Storage mode: Private for render-only, Shared if CPU needs access
        if desc.usage.has(TextureUsage::RENDER_TARGET) && !desc.usage.has(TextureUsage::SHADER_READ)
        {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Private);
        } else {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        }

        // Texture type
        use crate::TextureKind;
        match desc.kind {
            TextureKind::D2 if desc.sample_count > 1 => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D2Multisample);
            }
            TextureKind::D2 => {} // default
            TextureKind::D3 => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D3);
                mtl_desc.set_depth(desc.depth as u64);
            }
            TextureKind::Cube => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::Cube);
            }
            TextureKind::Array2D => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::D2Array);
                mtl_desc.set_array_length(desc.array_length as u64);
            }
            TextureKind::ArrayCube => {
                mtl_desc.set_texture_type(mtl::MTLTextureType::CubeArray);
                mtl_desc.set_array_length(desc.array_length as u64);
            }
        }

        if desc.mip_levels > 1 {
            mtl_desc.set_mipmap_level_count(desc.mip_levels as u64);
        } else if desc.mip_levels == 0 {
            // Auto-calculate: floor(log2(max(w,h))) + 1
            let max_dim = desc.width.max(desc.height);
            let levels = (max_dim as f32).log2().floor() as u64 + 1;
            mtl_desc.set_mipmap_level_count(levels);
        }

        let tex = self.device.new_texture(&mtl_desc);
        let handle = self.alloc_handle();
        self.textures.lock().unwrap().insert(handle, tex);

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
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let region = mtl::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        tex.replace_region(region, 0, data.as_ptr() as *const _, bytes_per_row);
        Ok(())
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("texture_read: handle {}", texture.handle()))
        })?;
        let bytes_per_pixel = format_bytes_per_pixel(texture.format());
        let size = (texture.width() * texture.height()) as usize * bytes_per_pixel;
        let mut result = vec![0u8; size];
        let region = mtl::MTLRegion::new_2d(0, 0, texture.width() as u64, texture.height() as u64);
        let bytes_per_row = texture.width() as u64 * bytes_per_pixel as u64;
        tex.get_bytes(result.as_mut_ptr() as *mut _, bytes_per_row, region, 0);
        Ok(result)
    }

    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let sd = mtl::SamplerDescriptor::new();
        sd.set_min_filter(filter_to_metal(desc.min_filter));
        sd.set_mag_filter(filter_to_metal(desc.mag_filter));
        sd.set_mip_filter(match desc.mip_filter {
            crate::render_pass::Filter::Nearest => mtl::MTLSamplerMipFilter::Nearest,
            crate::render_pass::Filter::Linear => mtl::MTLSamplerMipFilter::Linear,
        });
        sd.set_address_mode_s(address_to_metal(desc.address_u));
        sd.set_address_mode_t(address_to_metal(desc.address_v));
        sd.set_max_anisotropy(desc.max_anisotropy as u64);
        let _sampler = self.device.new_sampler(&sd);
        let handle = self.alloc_handle();
        // TODO: store sampler for binding
        Ok(crate::Sampler {
            handle,
            drop_fn: None,
        })
    }

    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError> {
        let textures = self.textures.lock().unwrap();
        let tex = textures.get(&texture.handle()).ok_or_else(|| {
            QuantaError::invalid_param("bad texture handle")
                .with_context(&format!("generate_mipmaps: handle {}", texture.handle()))
        })?;
        let cmd = self.queue.new_command_buffer();
        let blit = cmd.new_blit_command_encoder();
        blit.generate_mipmaps(tex);
        blit.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        Ok(())
    }

    // === Compute ===

    fn wave(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        let source_str = std::str::from_utf8(kernel_source)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in shader source"))?;

        // Accept either MSL or WGSL — convert WGSL to MSL via naga if needed
        let msl_source = if source_str.contains("#include <metal_stdlib>")
            || source_str.contains("kernel void")
        {
            source_str.to_string() // Already MSL
        } else {
            // Assume WGSL — convert via naga
            #[cfg(feature = "naga-shaders")]
            {
                super::shader_convert::wgsl_to_msl(source_str)
                    .map_err(|e| QuantaError::compilation_failed(format!("WGSL→MSL: {}", e)))?
            }
            #[cfg(not(feature = "naga-shaders"))]
            {
                return Err(QuantaError::compilation_failed(
                    "WGSL source requires naga-shaders feature",
                ));
            }
        };

        let opts = mtl::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(&msl_source, &opts)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        let func_names = library.function_names();
        let func_name = func_names
            .first()
            .ok_or_else(|| QuantaError::compilation_failed("no functions in kernel"))?;
        let func = library
            .get_function(func_name, None)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;

        let handle = self.alloc_handle();
        self.compute_pipelines
            .lock()
            .unwrap()
            .insert(handle, pipeline);
        Ok(Wave {
            handle,
            bindings: Vec::new(),
            push_constants: Vec::new(),
            texture_bindings: Vec::new(),
            drop_fn: None,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        // Note: Metal's CommandQueue.new_command_buffer() internally pools command
        // buffers. There is no need for an explicit pool — Metal manages reuse
        // automatically when command buffers complete.
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let pipelines = self.compute_pipelines.lock().unwrap();
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch: handle {}", wave.handle))
        })?;
        encoder.set_compute_pipeline_state(pipeline);

        let buffers = self.buffers.lock().unwrap();
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                encoder.set_buffer(b.slot as u64, Some(buf), 0);
            }
        }
        for pc in &wave.push_constants {
            encoder.set_bytes(
                pc.slot as u64,
                pc.data.len() as u64,
                pc.data.as_ptr() as *const _,
            );
        }

        // Bind textures for compute access
        let textures = self.textures.lock().unwrap();
        for tb in &wave.texture_bindings {
            if let Some(tex) = textures.get(&tb.texture_handle) {
                encoder.set_texture(tb.slot as u64, Some(tex));
            }
        }
        drop(textures);

        let grid = mtl::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
        let group_size = mtl::MTLSize::new(64, 1, 1);
        encoder.dispatch_threads(grid, group_size);
        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
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
        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();

        let pipelines = self.compute_pipelines.lock().unwrap();
        let pipeline = pipelines.get(&wave.handle).ok_or_else(|| {
            QuantaError::invalid_param("bad wave handle")
                .with_context(&format!("wave_dispatch_indirect: handle {}", wave.handle))
        })?;
        encoder.set_compute_pipeline_state(pipeline);

        let buffers = self.buffers.lock().unwrap();
        for b in &wave.bindings {
            if let Some(buf) = buffers.get(&b.field_handle) {
                encoder.set_buffer(b.slot as u64, Some(buf), 0);
            }
        }
        for pc in &wave.push_constants {
            encoder.set_bytes(
                pc.slot as u64,
                pc.data.len() as u64,
                pc.data.as_ptr() as *const _,
            );
        }

        let indirect_buf = buffers.get(&buffer).ok_or_else(|| {
            QuantaError::invalid_param("bad indirect buffer")
                .with_context(&format!("wave_dispatch_indirect: buffer handle {buffer}"))
        })?;
        let group_size = mtl::MTLSize::new(64, 1, 1);
        encoder.dispatch_thread_groups_indirect(indirect_buf, offset, group_size);
        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
            poll_fn: None,
            completed: false,
        })
    }

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        let opts = mtl::CompileOptions::new();

        // Support combined source or separate vertex/fragment sources
        let (vert_fn, frag_fn) = if let Some(combined) = desc.source {
            let src = std::str::from_utf8(combined)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in shader source"))?;
            let lib = self
                .device
                .new_library_with_source(src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("shader: {}", e)))?;
            let vf = lib.get_function(desc.vertex_entry, None).map_err(|e| {
                QuantaError::compilation_failed(format!("vertex fn '{}': {}", desc.vertex_entry, e))
            })?;
            let ff = lib.get_function(desc.fragment_entry, None).map_err(|e| {
                QuantaError::compilation_failed(format!(
                    "fragment fn '{}': {}",
                    desc.fragment_entry, e
                ))
            })?;
            (vf, ff)
        } else {
            let vert_src = std::str::from_utf8(desc.vertex)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in vertex shader"))?;
            let frag_src = std::str::from_utf8(desc.fragment)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in fragment shader"))?;
            let vert_lib = self
                .device
                .new_library_with_source(vert_src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("vertex: {}", e)))?;
            let frag_lib = self
                .device
                .new_library_with_source(frag_src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("fragment: {}", e)))?;
            let vf = vert_lib
                .get_function(desc.vertex_entry, None)
                .map_err(|e| QuantaError::compilation_failed(format!("vertex fn: {}", e)))?;
            let ff = frag_lib
                .get_function(desc.fragment_entry, None)
                .map_err(|e| QuantaError::compilation_failed(format!("fragment fn: {}", e)))?;
            (vf, ff)
        };

        let pipe_desc = mtl::RenderPipelineDescriptor::new();
        pipe_desc.set_vertex_function(Some(&vert_fn));
        pipe_desc.set_fragment_function(Some(&frag_fn));

        for (i, fmt) in desc.color_formats.iter().enumerate() {
            let ca = pipe_desc.color_attachments().object_at(i as u64).unwrap();
            ca.set_pixel_format(format_to_metal(*fmt));
            if desc.blend.enabled {
                ca.set_blending_enabled(true);
                ca.set_source_rgb_blend_factor(blend_factor_to_metal(desc.blend.src_rgb));
                ca.set_destination_rgb_blend_factor(blend_factor_to_metal(desc.blend.dst_rgb));
                ca.set_source_alpha_blend_factor(blend_factor_to_metal(desc.blend.src_alpha));
                ca.set_destination_alpha_blend_factor(blend_factor_to_metal(desc.blend.dst_alpha));
                ca.set_rgb_blend_operation(blend_op_to_metal(desc.blend.op_rgb));
                ca.set_alpha_blend_operation(blend_op_to_metal(desc.blend.op_alpha));
            }
        }

        if let Some(depth_fmt) = desc.depth_format {
            pipe_desc.set_depth_attachment_pixel_format(format_to_metal(depth_fmt));
        }

        pipe_desc.set_sample_count(desc.sample_count as u64);

        let pipeline_state = self
            .device
            .new_render_pipeline_state(&pipe_desc)
            .map_err(|e| QuantaError::compilation_failed(format!("render pipeline: {}", e)))?;

        // Create depth/stencil state
        let ds_desc = mtl::DepthStencilDescriptor::new();
        if desc.depth_stencil.depth_test {
            ds_desc.set_depth_compare_function(compare_to_metal(desc.depth_stencil.depth_compare));
            ds_desc.set_depth_write_enabled(desc.depth_stencil.depth_write);
        }
        if let Some(ref front) = desc.depth_stencil.stencil_front {
            let s = mtl::StencilDescriptor::new();
            s.set_stencil_failure_operation(stencil_op_to_metal(front.fail));
            s.set_depth_failure_operation(stencil_op_to_metal(front.depth_fail));
            s.set_depth_stencil_pass_operation(stencil_op_to_metal(front.pass));
            s.set_stencil_compare_function(compare_to_metal(front.compare));
            s.set_read_mask(front.read_mask);
            s.set_write_mask(front.write_mask);
            ds_desc.set_front_face_stencil(Some(&s));
        }
        if let Some(ref back) = desc.depth_stencil.stencil_back {
            let s = mtl::StencilDescriptor::new();
            s.set_stencil_failure_operation(stencil_op_to_metal(back.fail));
            s.set_depth_failure_operation(stencil_op_to_metal(back.depth_fail));
            s.set_depth_stencil_pass_operation(stencil_op_to_metal(back.pass));
            s.set_stencil_compare_function(compare_to_metal(back.compare));
            s.set_read_mask(back.read_mask);
            s.set_write_mask(back.write_mask);
            ds_desc.set_back_face_stencil(Some(&s));
        }
        let ds_state = self.device.new_depth_stencil_state(&ds_desc);

        let handle = self.alloc_handle();
        self.render_pipelines
            .lock()
            .unwrap()
            .insert(handle, pipeline_state);
        self.depth_stencil_states
            .lock()
            .unwrap()
            .insert(handle, ds_state);
        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }

    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        // Store target handle — render_end will use it
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
        })
    }

    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let target = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::invalid_param("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        // Create render pass descriptor
        let rpd = mtl::RenderPassDescriptor::new();
        let color_attach = rpd.color_attachments().object_at(0).unwrap();
        color_attach.set_texture(Some(target));
        color_attach.set_load_action(mtl::MTLLoadAction::Clear);
        color_attach.set_store_action(mtl::MTLStoreAction::Store);
        color_attach.set_clear_color(mtl::MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_render_command_encoder(rpd);

        let buffers = self.buffers.lock().unwrap();
        let render_pipelines = self.render_pipelines.lock().unwrap();

        for op in &pass.ops {
            match op {
                RenderOp::SetPipeline(handle) => {
                    if let Some(ps) = render_pipelines.get(handle) {
                        encoder.set_render_pipeline_state(ps);
                    }
                    let ds_states = self.depth_stencil_states.lock().unwrap();
                    if let Some(ds) = ds_states.get(handle) {
                        encoder.set_depth_stencil_state(ds);
                    }
                }
                RenderOp::BindVertices {
                    slot,
                    handle,
                    offset,
                } => {
                    if let Some(buf) = buffers.get(handle) {
                        encoder.set_vertex_buffer(*slot as u64, Some(buf), *offset);
                    }
                }
                RenderOp::BindIndices { .. } => {
                    // Index buffer is bound at draw_indexed time in Metal
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    if let Some(buf) = buffers.get(handle) {
                        encoder.set_vertex_buffer(*slot as u64, Some(buf), 0);
                    }
                }
                RenderOp::SetTexture { slot, handle } => {
                    if let Some(tex) = textures.get(handle) {
                        encoder.set_fragment_texture(*slot as u64, Some(tex));
                    }
                }
                RenderOp::SetValue { slot, data } => {
                    encoder.set_vertex_bytes(
                        *slot as u64,
                        data.len() as u64,
                        data.as_ptr() as *const _,
                    );
                }
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    if *instance_count <= 1 {
                        encoder.draw_primitives(
                            mtl::MTLPrimitiveType::Triangle,
                            0,
                            *vertex_count as u64,
                        );
                    } else {
                        encoder.draw_primitives_instanced(
                            mtl::MTLPrimitiveType::Triangle,
                            0,
                            *vertex_count as u64,
                            *instance_count as u64,
                        );
                    }
                }
                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    // Find the last BindIndices op to get the index buffer
                    let idx_handle = pass.ops.iter().rev().find_map(|op| {
                        if let RenderOp::BindIndices { handle, .. } = op {
                            Some(*handle)
                        } else {
                            None
                        }
                    });
                    if let Some(ih) = idx_handle
                        && let Some(idx_buf) = buffers.get(&ih)
                    {
                        if *instance_count <= 1 {
                            encoder.draw_indexed_primitives(
                                mtl::MTLPrimitiveType::Triangle,
                                *index_count as u64,
                                mtl::MTLIndexType::UInt32,
                                idx_buf,
                                0,
                            );
                        } else {
                            encoder.draw_indexed_primitives_instanced(
                                mtl::MTLPrimitiveType::Triangle,
                                *index_count as u64,
                                mtl::MTLIndexType::UInt32,
                                idx_buf,
                                0,
                                *instance_count as u64,
                            );
                        }
                    }
                }
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => {
                    encoder.set_scissor_rect(mtl::MTLScissorRect {
                        x: *x as u64,
                        y: *y as u64,
                        width: *width as u64,
                        height: *height as u64,
                    });
                }
                RenderOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    encoder.set_viewport(mtl::MTLViewport {
                        originX: *x as f64,
                        originY: *y as f64,
                        width: *width as f64,
                        height: *height as f64,
                        znear: *min_depth as f64,
                        zfar: *max_depth as f64,
                    });
                }
                RenderOp::Clear(_color) => {
                    // Clear is handled by load action on the render pass descriptor.
                    // Dynamic clear within a pass would need a new encoder — skip for now.
                }
                RenderOp::ClearDepth(_depth) => {
                    // Same — handled by render pass descriptor load action.
                }
                RenderOp::SetStencilRef(value) => {
                    encoder.set_stencil_reference_value(*value);
                }
                RenderOp::ClearStencil(_) => {
                    // Handled by render pass descriptor load action
                }
                RenderOp::DrawIndirect {
                    buffer_handle,
                    offset,
                } => {
                    if let Some(buf) = buffers.get(buffer_handle) {
                        encoder.draw_primitives_indirect(
                            mtl::MTLPrimitiveType::Triangle,
                            buf,
                            *offset,
                        );
                    }
                }
                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    offset,
                    index_handle,
                } => {
                    if let Some(buf) = buffers.get(buffer_handle)
                        && let Some(idx_buf) = buffers.get(index_handle)
                    {
                        encoder.draw_indexed_primitives_indirect(
                            mtl::MTLPrimitiveType::Triangle,
                            mtl::MTLIndexType::UInt32,
                            idx_buf,
                            0,
                            buf,
                            *offset,
                        );
                    }
                }
                RenderOp::DebugPush(label) => {
                    encoder.push_debug_group(label);
                }
                RenderOp::DebugPop => {
                    encoder.pop_debug_group();
                }
                RenderOp::SetSampler { .. } => {
                    // TODO: create MTLSamplerState and bind to fragment
                }
            }
        }

        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
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

// ============================================================================
// Metal type conversions
// ============================================================================

fn format_to_metal(format: Format) -> mtl::MTLPixelFormat {
    match format {
        Format::RGBA8 => mtl::MTLPixelFormat::RGBA8Unorm,
        Format::BGRA8 => mtl::MTLPixelFormat::BGRA8Unorm,
        Format::R8 => mtl::MTLPixelFormat::R8Unorm,
        Format::R16Float => mtl::MTLPixelFormat::R16Float,
        Format::R32Float => mtl::MTLPixelFormat::R32Float,
        Format::RG32Float => mtl::MTLPixelFormat::RG32Float,
        Format::RGBA16Float => mtl::MTLPixelFormat::RGBA16Float,
        Format::RGBA32Float => mtl::MTLPixelFormat::RGBA32Float,
        Format::Depth32Float => mtl::MTLPixelFormat::Depth32Float,
    }
}

fn format_bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8 => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Depth32Float => 4,
    }
}

fn filter_to_metal(f: crate::render_pass::Filter) -> mtl::MTLSamplerMinMagFilter {
    match f {
        crate::render_pass::Filter::Nearest => mtl::MTLSamplerMinMagFilter::Nearest,
        crate::render_pass::Filter::Linear => mtl::MTLSamplerMinMagFilter::Linear,
    }
}

fn address_to_metal(a: crate::render_pass::AddressMode) -> mtl::MTLSamplerAddressMode {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => mtl::MTLSamplerAddressMode::ClampToEdge,
        crate::render_pass::AddressMode::Repeat => mtl::MTLSamplerAddressMode::Repeat,
        crate::render_pass::AddressMode::MirrorRepeat => mtl::MTLSamplerAddressMode::MirrorRepeat,
    }
}

fn compare_to_metal(f: crate::CompareFunc) -> mtl::MTLCompareFunction {
    use crate::CompareFunc::*;
    match f {
        Never => mtl::MTLCompareFunction::Never,
        Less => mtl::MTLCompareFunction::Less,
        Equal => mtl::MTLCompareFunction::Equal,
        LessEqual => mtl::MTLCompareFunction::LessEqual,
        Greater => mtl::MTLCompareFunction::Greater,
        NotEqual => mtl::MTLCompareFunction::NotEqual,
        GreaterEqual => mtl::MTLCompareFunction::GreaterEqual,
        Always => mtl::MTLCompareFunction::Always,
    }
}

fn stencil_op_to_metal(op: crate::StencilOp) -> mtl::MTLStencilOperation {
    use crate::StencilOp::*;
    match op {
        Keep => mtl::MTLStencilOperation::Keep,
        Zero => mtl::MTLStencilOperation::Zero,
        Replace => mtl::MTLStencilOperation::Replace,
        IncrementClamp => mtl::MTLStencilOperation::IncrementClamp,
        DecrementClamp => mtl::MTLStencilOperation::DecrementClamp,
        Invert => mtl::MTLStencilOperation::Invert,
        IncrementWrap => mtl::MTLStencilOperation::IncrementWrap,
        DecrementWrap => mtl::MTLStencilOperation::DecrementWrap,
    }
}

fn blend_factor_to_metal(f: crate::BlendFactor) -> mtl::MTLBlendFactor {
    use crate::BlendFactor::*;
    match f {
        Zero => mtl::MTLBlendFactor::Zero,
        One => mtl::MTLBlendFactor::One,
        SrcAlpha => mtl::MTLBlendFactor::SourceAlpha,
        OneMinusSrcAlpha => mtl::MTLBlendFactor::OneMinusSourceAlpha,
        DstAlpha => mtl::MTLBlendFactor::DestinationAlpha,
        OneMinusDstAlpha => mtl::MTLBlendFactor::OneMinusDestinationAlpha,
        SrcColor => mtl::MTLBlendFactor::SourceColor,
        OneMinusSrcColor => mtl::MTLBlendFactor::OneMinusSourceColor,
        DstColor => mtl::MTLBlendFactor::DestinationColor,
        OneMinusDstColor => mtl::MTLBlendFactor::OneMinusDestinationColor,
    }
}

fn blend_op_to_metal(op: crate::BlendOp) -> mtl::MTLBlendOperation {
    use crate::BlendOp::*;
    match op {
        Add => mtl::MTLBlendOperation::Add,
        Subtract => mtl::MTLBlendOperation::Subtract,
        ReverseSubtract => mtl::MTLBlendOperation::ReverseSubtract,
        Min => mtl::MTLBlendOperation::Min,
        Max => mtl::MTLBlendOperation::Max,
    }
}

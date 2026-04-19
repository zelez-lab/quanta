//! Metal driver for macOS/iOS.

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture, Vendor,
    Wave,
};
use metal as mtl;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Metal-backed GPU device.
pub struct MetalDevice {
    device: mtl::Device,
    queue: mtl::CommandQueue,
    caps: Caps,
    buffers: Arc<Mutex<HashMap<u64, mtl::Buffer>>>,
    pipelines: Arc<Mutex<HashMap<u64, mtl::ComputePipelineState>>>,
    next_handle: Arc<Mutex<u64>>,
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

    // Apple doesn't expose exact CU/core counts via Metal API.
    // We estimate from the device family and max threads.
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
        buffers: Arc::new(Mutex::new(HashMap::new())),
        pipelines: Arc::new(Mutex::new(HashMap::new())),
        next_handle: Arc::new(Mutex::new(0)),
    })]
}

impl GpuDevice for MetalDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError> {
        // Pick storage mode based on usage:
        // - TRANSFER set → CPU needs access → StorageModeShared
        // - No TRANSFER  → GPU only → StorageModePrivate (faster VRAM access)
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
        let buffer = buffers
            .get(&handle)
            .ok_or(QuantaError::InvalidParam("bad field handle"))?;
        unsafe {
            let ptr = buffer.contents() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let buffer = buffers
            .get(&handle)
            .ok_or(QuantaError::InvalidParam("bad field handle"))?;
        let mut result = vec![0u8; size];
        unsafe {
            let ptr = buffer.contents() as *const u8;
            std::ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size);
        }
        Ok(result)
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let buffers = self.buffers.lock().unwrap();
        let src_buf = buffers
            .get(&src)
            .ok_or(QuantaError::InvalidParam("bad src handle"))?;
        let dst_buf = buffers
            .get(&dst)
            .ok_or(QuantaError::InvalidParam("bad dst handle"))?;

        let cmd_buffer = self.queue.new_command_buffer();
        let blit = cmd_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(src_buf, 0, dst_buf, 0, size as u64);
        blit.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
        Ok(())
    }

    fn texture_create(&self, desc: &crate::TextureDesc) -> Result<Texture, QuantaError> {
        let mtl_desc = mtl::TextureDescriptor::new();
        mtl_desc.set_width(desc.width as u64);
        mtl_desc.set_height(desc.height as u64);
        mtl_desc.set_pixel_format(format_to_metal(desc.format));
        mtl_desc.set_sample_count(desc.sample_count as u64);

        let mut usage = mtl::MTLTextureUsage::empty();
        if desc.usage.has(crate::TextureUsage::SHADER_READ) {
            usage |= mtl::MTLTextureUsage::ShaderRead;
        }
        if desc.usage.has(crate::TextureUsage::SHADER_WRITE) {
            usage |= mtl::MTLTextureUsage::ShaderWrite;
        }
        if desc.usage.has(crate::TextureUsage::RENDER_TARGET) {
            usage |= mtl::MTLTextureUsage::RenderTarget;
        }
        if usage.is_empty() {
            usage = mtl::MTLTextureUsage::ShaderRead;
        }
        mtl_desc.set_usage(usage);

        if desc.usage.has(crate::TextureUsage::RENDER_TARGET)
            && !desc.usage.has(crate::TextureUsage::SHADER_READ)
        {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Private);
        } else {
            mtl_desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        }

        if desc.sample_count > 1 {
            mtl_desc.set_texture_type(mtl::MTLTextureType::D2Multisample);
        }

        let _tex = self.device.new_texture(&mtl_desc);
        let handle = self.alloc_handle();

        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            drop_fn: None,
        })
    }

    fn texture_write(&self, _texture: &Texture, _data: &[u8]) -> Result<(), QuantaError> {
        // TODO: blit encoder to upload pixel data
        Ok(())
    }

    fn texture_read(&self, _texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        // TODO: blit encoder to read back pixel data
        Ok(Vec::new())
    }

    fn wave(&self, kernel_source: &[u8]) -> Result<Wave, QuantaError> {
        // kernel_source is MSL source string for Metal
        let source = std::str::from_utf8(kernel_source)
            .map_err(|_| QuantaError::CompilationFailed("invalid UTF-8 in MSL source".into()))?;

        let opts = mtl::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &opts)
            .map_err(|e| QuantaError::CompilationFailed(e.to_string()))?;

        // Get the first function (convention: kernel function named "main0" or the only one)
        let func_names = library.function_names();
        let func_name = func_names
            .first()
            .ok_or_else(|| QuantaError::CompilationFailed("no functions in kernel".into()))?;

        let func = library
            .get_function(func_name, None)
            .map_err(|e| QuantaError::CompilationFailed(e.to_string()))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| QuantaError::CompilationFailed(e.to_string()))?;

        let handle = self.alloc_handle();
        self.pipelines.lock().unwrap().insert(handle, pipeline);

        Ok(Wave {
            handle,
            bindings: Vec::new(),
            push_constants: Vec::new(),
            drop_fn: None,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let cmd_buffer = self.queue.new_command_buffer();
        let encoder = cmd_buffer.new_compute_command_encoder();

        // Bind pipeline
        let pipelines = self.pipelines.lock().unwrap();
        let pipeline = pipelines
            .get(&wave.handle)
            .ok_or(QuantaError::InvalidParam("bad wave handle"))?;
        encoder.set_compute_pipeline_state(pipeline);

        // Bind fields
        let buffers = self.buffers.lock().unwrap();
        for binding in &wave.bindings {
            if let Some(buf) = buffers.get(&binding.field_handle) {
                encoder.set_buffer(binding.slot as u64, Some(buf), 0);
            }
        }

        // Bind push constants
        for pc in &wave.push_constants {
            encoder.set_bytes(
                pc.slot as u64,
                pc.data.len() as u64,
                pc.data.as_ptr() as *const _,
            );
        }

        // Dispatch
        let threads_per_group = mtl::MTLSize::new(64, 1, 1);
        let grid_size = mtl::MTLSize::new(groups[0] as u64, groups[1] as u64, groups[2] as u64);
        encoder.dispatch_threads(grid_size, threads_per_group);
        encoder.end_encoding();

        cmd_buffer.commit();

        let handle = self.alloc_handle();
        // Clone the command buffer reference for the wait closure
        let cmd_buf_clone = cmd_buffer.to_owned();

        Ok(Pulse {
            handle,
            wait_fn: Some(Box::new(move |_| {
                cmd_buf_clone.wait_until_completed();
                Ok(())
            })),
            poll_fn: None,
        })
    }

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        // Compile vertex + fragment shaders from MSL source
        let opts = mtl::CompileOptions::new();

        let vert_src = std::str::from_utf8(desc.vertex)
            .map_err(|_| QuantaError::CompilationFailed("invalid UTF-8 in vertex shader".into()))?;
        let frag_src = std::str::from_utf8(desc.fragment).map_err(|_| {
            QuantaError::CompilationFailed("invalid UTF-8 in fragment shader".into())
        })?;

        let vert_lib = self
            .device
            .new_library_with_source(vert_src, &opts)
            .map_err(|e| QuantaError::CompilationFailed(format!("vertex: {}", e)))?;
        let frag_lib = self
            .device
            .new_library_with_source(frag_src, &opts)
            .map_err(|e| QuantaError::CompilationFailed(format!("fragment: {}", e)))?;

        let vert_fn = vert_lib
            .get_function(desc.vertex_entry, None)
            .map_err(|e| {
                QuantaError::CompilationFailed(format!(
                    "vertex entry '{}': {}",
                    desc.vertex_entry, e
                ))
            })?;
        let frag_fn = frag_lib
            .get_function(desc.fragment_entry, None)
            .map_err(|e| {
                QuantaError::CompilationFailed(format!(
                    "fragment entry '{}': {}",
                    desc.fragment_entry, e
                ))
            })?;

        let pipe_desc = mtl::RenderPipelineDescriptor::new();
        pipe_desc.set_vertex_function(Some(&vert_fn));
        pipe_desc.set_fragment_function(Some(&frag_fn));

        // Color attachment
        let color_attach = pipe_desc.color_attachments().object_at(0).unwrap();
        color_attach.set_pixel_format(format_to_metal(desc.color_format));

        if desc.blend.enabled {
            color_attach.set_blending_enabled(true);
            color_attach.set_source_rgb_blend_factor(blend_factor_to_metal(desc.blend.src_rgb));
            color_attach
                .set_destination_rgb_blend_factor(blend_factor_to_metal(desc.blend.dst_rgb));
            color_attach.set_source_alpha_blend_factor(blend_factor_to_metal(desc.blend.src_alpha));
            color_attach
                .set_destination_alpha_blend_factor(blend_factor_to_metal(desc.blend.dst_alpha));
            color_attach.set_rgb_blend_operation(blend_op_to_metal(desc.blend.op_rgb));
            color_attach.set_alpha_blend_operation(blend_op_to_metal(desc.blend.op_alpha));
        }

        // Depth
        if let Some(depth_fmt) = desc.depth_format {
            pipe_desc.set_depth_attachment_pixel_format(format_to_metal(depth_fmt));
        }

        // MSAA
        pipe_desc.set_sample_count(desc.sample_count as u64);

        let _pipeline_state = self
            .device
            .new_render_pipeline_state(&pipe_desc)
            .map_err(|e| QuantaError::CompilationFailed(format!("render pipeline: {}", e)))?;

        let handle = self.alloc_handle();
        // TODO: store _pipeline_state for use in render_begin/render_end
        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }

    fn render_begin(&self, _target: &Texture) -> Result<RenderPass, QuantaError> {
        let handle = self.alloc_handle();
        Ok(RenderPass {
            handle,
            ops: Vec::new(),
        })
    }

    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        // TODO: encode render commands and submit
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
        })
    }

    fn pulse_wait(&self, pulse: Pulse) -> Result<(), QuantaError> {
        pulse.wait()
    }

    fn pulse_poll(&self, pulse: &Pulse) -> bool {
        pulse.is_done()
    }
}

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

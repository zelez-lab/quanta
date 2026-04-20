//! Metal driver for macOS/iOS.
//!
//! Uses the `metal` crate for Metal API bindings.
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
mod memory;
mod render;
mod texture;

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, Texture,
    TextureDesc, Vendor, Wave,
};
use metal as mtl;
use std::collections::HashMap;
use std::sync::Mutex;

/// Metal-backed GPU device.
pub struct MetalDevice {
    pub(crate) device: mtl::Device,
    pub(crate) queue: mtl::CommandQueue,
    caps: Caps,
    // Resource storage — keyed by handle
    pub(crate) buffers: Mutex<HashMap<u64, mtl::Buffer>>,
    pub(crate) textures: Mutex<HashMap<u64, mtl::Texture>>,
    pub(crate) compute_pipelines: Mutex<HashMap<u64, mtl::ComputePipelineState>>,
    pub(crate) render_pipelines: Mutex<HashMap<u64, mtl::RenderPipelineState>>,
    pub(crate) depth_stencil_states: Mutex<HashMap<u64, mtl::DepthStencilState>>,
    pub(crate) samplers: Mutex<HashMap<u64, mtl::SamplerState>>,
    pub(crate) next_handle: Mutex<u64>,
}

impl MetalDevice {
    pub(crate) fn alloc_handle(&self) -> u64 {
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
        samplers: Mutex::new(HashMap::new()),
        next_handle: Mutex::new(0),
    })]
}

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
        // Store target handle — render_end will use it
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
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
}

// ============================================================================
// Metal type conversions
// ============================================================================

pub(crate) fn format_to_metal(format: Format) -> mtl::MTLPixelFormat {
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

pub(crate) fn format_bytes_per_pixel(format: Format) -> usize {
    match format {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8 => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Depth32Float => 4,
    }
}

pub(crate) fn filter_to_metal(f: crate::render_pass::Filter) -> mtl::MTLSamplerMinMagFilter {
    match f {
        crate::render_pass::Filter::Nearest => mtl::MTLSamplerMinMagFilter::Nearest,
        crate::render_pass::Filter::Linear => mtl::MTLSamplerMinMagFilter::Linear,
    }
}

pub(crate) fn address_to_metal(a: crate::render_pass::AddressMode) -> mtl::MTLSamplerAddressMode {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => mtl::MTLSamplerAddressMode::ClampToEdge,
        crate::render_pass::AddressMode::Repeat => mtl::MTLSamplerAddressMode::Repeat,
        crate::render_pass::AddressMode::MirrorRepeat => mtl::MTLSamplerAddressMode::MirrorRepeat,
    }
}

pub(crate) fn compare_to_metal(f: crate::CompareFunc) -> mtl::MTLCompareFunction {
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

pub(crate) fn stencil_op_to_metal(op: crate::StencilOp) -> mtl::MTLStencilOperation {
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

pub(crate) fn blend_factor_to_metal(f: crate::BlendFactor) -> mtl::MTLBlendFactor {
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

pub(crate) fn blend_op_to_metal(op: crate::BlendOp) -> mtl::MTLBlendOperation {
    use crate::BlendOp::*;
    match op {
        Add => mtl::MTLBlendOperation::Add,
        Subtract => mtl::MTLBlendOperation::Subtract,
        ReverseSubtract => mtl::MTLBlendOperation::ReverseSubtract,
        Min => mtl::MTLBlendOperation::Min,
        Max => mtl::MTLBlendOperation::Max,
    }
}

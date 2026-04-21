//! Metal driver for macOS/iOS.
//!
//! Uses raw ObjC/Metal FFI bindings — no external Metal crate dependency.
//! Covers compute dispatch, render pass execution, texture management,
//! depth/stencil, instanced/indexed/indirect draw, MRT, and debug labels.

mod compute;
pub(crate) mod ffi;
mod memory;
mod render;
mod texture;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{
    Caps, FieldUsage, Format, GpuDevice, Pipeline, Pulse, QuantaError, RenderPass, ResourceState,
    Texture, TextureDesc, Vendor, Wave,
};
use std::collections::HashMap;
use std::sync::Mutex;

/// Metal-backed GPU device.
pub struct MetalDevice {
    pub(crate) device: ffi::Id,
    pub(crate) queue: ffi::Id,
    caps: Caps,
    // Resource storage — keyed by handle
    pub(crate) buffers: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) textures: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) compute_pipelines: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) render_pipelines: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) depth_stencil_states: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) samplers: Mutex<HashMap<u64, ffi::Id>>,
    pub(crate) next_handle: AtomicU64,
}

impl MetalDevice {
    pub(crate) fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Discover Metal devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let device = unsafe { ffi::MTLCreateSystemDefaultDevice() };
    if device.is_null() {
        return Vec::new();
    }

    let name = unsafe {
        let ns_name = ffi::msg_id(device, b"name\0");
        let cstr = ffi::msg_utf8_string(ns_name);
        std::ffi::CStr::from_ptr(cstr as *const _)
            .to_string_lossy()
            .into_owned()
    };

    let max_threads = unsafe { ffi::msg_mtlsize(device, b"maxThreadsPerThreadgroup\0") };
    let memory_bytes = unsafe { ffi::msg_u64(device, b"recommendedMaxWorkingSetSize\0") };

    let caps = Caps {
        nuclei: (max_threads.width as u32 / 32).max(1),
        protons_per_nucleus: 32,
        quarks_per_proton: 32,
        memory_bytes,
        max_quarks_per_dispatch: u32::MAX,
        max_groups: [u32::MAX, u32::MAX, u32::MAX],
        vendor: Vendor::Apple,
        name,
    };

    let queue = unsafe { ffi::msg_id(device, b"newCommandQueue\0") };

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
        next_handle: AtomicU64::new(0),
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

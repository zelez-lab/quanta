//! Standalone Vulkan format and blend conversion helpers.

use crate::Format;

use super::ffi;

pub(super) fn format_to_vulkan(format: Format) -> u32 {
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

pub(super) fn sample_count_to_vk(count: u32) -> u32 {
    match count {
        1 => ffi::VK_SAMPLE_COUNT_1_BIT,
        2 => ffi::VK_SAMPLE_COUNT_2_BIT,
        4 => ffi::VK_SAMPLE_COUNT_4_BIT,
        8 => ffi::VK_SAMPLE_COUNT_8_BIT,
        16 => ffi::VK_SAMPLE_COUNT_16_BIT,
        _ => ffi::VK_SAMPLE_COUNT_1_BIT,
    }
}

pub(super) fn format_bytes_per_pixel_vk(format: Format) -> usize {
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

pub(super) fn blend_factor_to_vk(f: crate::BlendFactor) -> u32 {
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

pub(super) fn blend_op_to_vk(op: crate::BlendOp) -> u32 {
    use crate::BlendOp::*;
    match op {
        Add => ffi::VK_BLEND_OP_ADD,
        Subtract => ffi::VK_BLEND_OP_SUBTRACT,
        ReverseSubtract => ffi::VK_BLEND_OP_REVERSE_SUBTRACT,
        Min => ffi::VK_BLEND_OP_MIN,
        Max => ffi::VK_BLEND_OP_MAX,
    }
}

pub(super) fn compare_op_to_vk(op: crate::CompareOp) -> u32 {
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

pub(super) fn filter_to_vk(f: crate::render_pass::Filter) -> u32 {
    match f {
        crate::render_pass::Filter::Nearest => ffi::VK_FILTER_NEAREST,
        crate::render_pass::Filter::Linear => ffi::VK_FILTER_LINEAR,
    }
}

pub(super) fn address_to_vk(a: crate::render_pass::AddressMode) -> u32 {
    match a {
        crate::render_pass::AddressMode::ClampToEdge => ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        crate::render_pass::AddressMode::Repeat => ffi::VK_SAMPLER_ADDRESS_MODE_REPEAT,
        crate::render_pass::AddressMode::MirrorRepeat => {
            ffi::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT
        }
    }
}

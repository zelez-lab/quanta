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

/// Reverse of [`format_to_vulkan`]: map a `VkFormat` back to the quanta
/// [`Format`] it stands for, or `None` when quanta has no equivalent.
///
/// Used by swapchain format negotiation to accept a surface-offered
/// format that isn't one of the preferred chain members — a surface that
/// offers only formats this returns `None` for is genuinely
/// inexpressible and the negotiation rejects it. Kept in lockstep with
/// `format_to_vulkan` (every arm there has its inverse here); the
/// `_UNORM`/`_SFLOAT` families round-trip, so the two are exact inverses
/// on the formats quanta names.
///
/// Only the render face negotiates swapchain formats, so this is
/// `render`-gated — without it the function would be dead code (the
/// surface module that calls it is itself `render`-gated).
#[cfg(feature = "render")]
pub(super) fn vulkan_to_format(vk_format: u32) -> Option<Format> {
    Some(match vk_format {
        ffi::VK_FORMAT_R8G8B8A8_UNORM => Format::RGBA8,
        ffi::VK_FORMAT_B8G8R8A8_UNORM => Format::BGRA8,
        ffi::VK_FORMAT_R8_UNORM => Format::R8,
        ffi::VK_FORMAT_R16_SFLOAT => Format::R16Float,
        ffi::VK_FORMAT_R32_SFLOAT => Format::R32Float,
        ffi::VK_FORMAT_R32G32_SFLOAT => Format::RG32Float,
        ffi::VK_FORMAT_R16G16B16A16_SFLOAT => Format::RGBA16Float,
        ffi::VK_FORMAT_R32G32B32A32_SFLOAT => Format::RGBA32Float,
        ffi::VK_FORMAT_D32_SFLOAT => Format::Depth32Float,
        ffi::VK_FORMAT_BC1_RGBA_UNORM_BLOCK => Format::Bc1Rgba,
        ffi::VK_FORMAT_BC3_UNORM_BLOCK => Format::Bc3Rgba,
        ffi::VK_FORMAT_BC5_SNORM_BLOCK => Format::Bc5Rg,
        ffi::VK_FORMAT_BC7_UNORM_BLOCK => Format::Bc7Rgba,
        ffi::VK_FORMAT_ASTC_4X4_UNORM_BLOCK => Format::Astc4x4,
        ffi::VK_FORMAT_ASTC_6X6_UNORM_BLOCK => Format::Astc6x6,
        ffi::VK_FORMAT_ASTC_8X8_UNORM_BLOCK => Format::Astc8x8,
        ffi::VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK => Format::Etc2Rgb8,
        ffi::VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK => Format::Etc2Rgba8,
        _ => return None,
    })
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

#[cfg(feature = "render")]
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

#[cfg(feature = "render")]
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

pub(super) fn filter_to_vk(f: crate::texture::Filter) -> u32 {
    match f {
        crate::texture::Filter::Nearest => ffi::VK_FILTER_NEAREST,
        crate::texture::Filter::Linear => ffi::VK_FILTER_LINEAR,
    }
}

pub(super) fn address_to_vk(a: crate::texture::AddressMode) -> u32 {
    match a {
        crate::texture::AddressMode::ClampToEdge => ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        crate::texture::AddressMode::Repeat => ffi::VK_SAMPLER_ADDRESS_MODE_REPEAT,
        crate::texture::AddressMode::MirrorRepeat => ffi::VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
    }
}

/// Build a `VkSamplerCreateInfo` from a `SamplerDesc`.
///
/// The single source of truth for translating a descriptor to sampler
/// create-info, shared by the named-sampler path (`sampler_create_impl`)
/// and the render-path sampler cache — so a given `SamplerDesc` always
/// maps to the same VkSampler state regardless of which path created it.
/// `address_mode_w` is fixed to CLAMP_TO_EDGE (2D sampling only) and mip
/// LOD is left unclamped, matching the descriptor's 2D-texture contract.
#[cfg(feature = "render")]
pub(super) fn sampler_create_info(desc: &crate::texture::SamplerDesc) -> ffi::VkSamplerCreateInfo {
    let (compare_enable, compare_op) = match desc.compare {
        Some(cmp) => (1u32, compare_op_to_vk(cmp)),
        None => (0u32, 0u32),
    };
    ffi::VkSamplerCreateInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        mag_filter: filter_to_vk(desc.mag_filter),
        min_filter: filter_to_vk(desc.min_filter),
        mipmap_mode: match desc.mip_filter {
            crate::texture::Filter::Nearest => ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST,
            crate::texture::Filter::Linear => ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR,
        },
        address_mode_u: address_to_vk(desc.address_u),
        address_mode_v: address_to_vk(desc.address_v),
        address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        mip_lod_bias: 0.0,
        anisotropy_enable: if desc.max_anisotropy > 1 { 1 } else { 0 },
        max_anisotropy: desc.max_anisotropy as f32,
        compare_enable,
        compare_op,
        min_lod: 0.0,
        max_lod: ffi::VK_LOD_CLAMP_NONE,
        border_color: 0,
        unnormalized_coordinates: 0,
    }
}

/// The layout an image settles in between explicit GPU operations,
/// with the access mask and stage a barrier INTO that layout should
/// use. Derived from the image's creation usage so the transition is
/// always licensed: sampled images rest shader-readable, attachment-
/// only images (MSAA intermediates, swapchain frames) rest as
/// attachments, anything else rests in GENERAL — always valid.
pub(super) fn image_rest_state(usage: u32) -> (u32, u32, u32) {
    if usage & ffi::VK_IMAGE_USAGE_SAMPLED_BIT != 0 {
        (
            ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            ffi::VK_ACCESS_SHADER_READ_BIT,
            ffi::VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        )
    } else if usage & ffi::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT != 0 {
        (
            ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            ffi::VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        )
    } else {
        (
            ffi::VK_IMAGE_LAYOUT_GENERAL,
            ffi::VK_ACCESS_MEMORY_READ_BIT | ffi::VK_ACCESS_MEMORY_WRITE_BIT,
            ffi::VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        )
    }
}

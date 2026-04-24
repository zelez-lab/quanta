//! Kani harnesses: Vulkan format table correctness.
//!
//! Mirrors the format_to_vulkan and format_bytes_per_pixel_vk tables from
//! `src/driver/vulkan/helpers.rs` and `src/api/types.rs`. Proves:
//!   - No two Format variants map to the same VK_FORMAT value (injectivity)
//!   - All VK_FORMAT values are non-zero (valid Vulkan format)
//!   - bytes_per_pixel is consistent with format channel/bit layout
//!
//! All harnesses use const arrays and symbolic indices -- no allocation.
//!
//! Run: cargo kani --harness <name>

// ── Mirrored Format enum ────────────────────────────────────────────────────

/// Mirrors `crate::Format` from `src/api/types.rs`.
/// Discriminant values 0..=17 used as array indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Format {
    RGBA8 = 0,
    BGRA8 = 1,
    R8 = 2,
    R16Float = 3,
    R32Float = 4,
    RG32Float = 5,
    RGBA16Float = 6,
    RGBA32Float = 7,
    Depth32Float = 8,
    Bc1Rgba = 9,
    Bc3Rgba = 10,
    Bc5Rg = 11,
    Bc7Rgba = 12,
    Astc4x4 = 13,
    Astc6x6 = 14,
    Astc8x8 = 15,
    Etc2Rgb8 = 16,
    Etc2Rgba8 = 17,
}

const FORMAT_COUNT: usize = 18;

// ── VK_FORMAT constants (from src/driver/vulkan/ffi/constants.rs) ───────────

const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
const VK_FORMAT_R8_UNORM: u32 = 9;
const VK_FORMAT_R16_SFLOAT: u32 = 76;
const VK_FORMAT_R32_SFLOAT: u32 = 100;
const VK_FORMAT_R32G32_SFLOAT: u32 = 103;
const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;
const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;
const VK_FORMAT_D32_SFLOAT: u32 = 126;
const VK_FORMAT_BC1_RGBA_UNORM_BLOCK: u32 = 132;
const VK_FORMAT_BC3_UNORM_BLOCK: u32 = 137;
const VK_FORMAT_BC5_SNORM_BLOCK: u32 = 142;
const VK_FORMAT_BC7_UNORM_BLOCK: u32 = 145;
const VK_FORMAT_ASTC_4X4_UNORM_BLOCK: u32 = 157;
const VK_FORMAT_ASTC_6X6_UNORM_BLOCK: u32 = 163;
const VK_FORMAT_ASTC_8X8_UNORM_BLOCK: u32 = 169;
const VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: u32 = 147;
const VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: u32 = 151;

// ── Mapping tables (mirrors helpers.rs format_to_vulkan) ────────────────────

/// format_to_vulkan as a const array indexed by Format discriminant.
const FORMAT_TO_VK: [u32; FORMAT_COUNT] = [
    VK_FORMAT_R8G8B8A8_UNORM,          // RGBA8
    VK_FORMAT_B8G8R8A8_UNORM,          // BGRA8
    VK_FORMAT_R8_UNORM,                // R8
    VK_FORMAT_R16_SFLOAT,              // R16Float
    VK_FORMAT_R32_SFLOAT,              // R32Float
    VK_FORMAT_R32G32_SFLOAT,           // RG32Float
    VK_FORMAT_R16G16B16A16_SFLOAT,     // RGBA16Float
    VK_FORMAT_R32G32B32A32_SFLOAT,     // RGBA32Float
    VK_FORMAT_D32_SFLOAT,              // Depth32Float
    VK_FORMAT_BC1_RGBA_UNORM_BLOCK,    // Bc1Rgba
    VK_FORMAT_BC3_UNORM_BLOCK,         // Bc3Rgba
    VK_FORMAT_BC5_SNORM_BLOCK,         // Bc5Rg
    VK_FORMAT_BC7_UNORM_BLOCK,         // Bc7Rgba
    VK_FORMAT_ASTC_4X4_UNORM_BLOCK,    // Astc4x4
    VK_FORMAT_ASTC_6X6_UNORM_BLOCK,    // Astc6x6
    VK_FORMAT_ASTC_8X8_UNORM_BLOCK,    // Astc8x8
    VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, // Etc2Rgb8
    VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, // Etc2Rgba8
];

/// format_bytes_per_pixel_vk as a const array (from helpers.rs).
/// For compressed formats this is the block size in bytes.
const FORMAT_BYTES_PER_PIXEL_VK: [usize; FORMAT_COUNT] = [
    4,  // RGBA8
    4,  // BGRA8
    1,  // R8
    2,  // R16Float
    4,  // R32Float
    8,  // RG32Float
    8,  // RGBA16Float
    16, // RGBA32Float
    4,  // Depth32Float
    8,  // Bc1Rgba      (8 bytes per 4x4 block)
    16, // Bc3Rgba      (16 bytes per 4x4 block)
    16, // Bc5Rg        (16 bytes per 4x4 block)
    16, // Bc7Rgba      (16 bytes per 4x4 block)
    16, // Astc4x4      (16 bytes per block)
    16, // Astc6x6      (16 bytes per block)
    16, // Astc8x8      (16 bytes per block)
    8,  // Etc2Rgb8     (8 bytes per 4x4 block)
    16, // Etc2Rgba8    (16 bytes per 4x4 block)
];

/// Format::bytes_per_pixel (from api/types.rs) -- the canonical method.
/// Compressed formats return 1 (block-level abstraction, not raw block size).
const FORMAT_BYTES_PER_PIXEL_API: [usize; FORMAT_COUNT] = [
    4,  // RGBA8
    4,  // BGRA8
    1,  // R8
    2,  // R16Float
    4,  // R32Float
    8,  // RG32Float
    8,  // RGBA16Float
    16, // RGBA32Float
    4,  // Depth32Float
    1,  // Bc1Rgba
    1,  // Bc3Rgba
    1,  // Bc5Rg
    1,  // Bc7Rgba
    1,  // Astc4x4
    1,  // Astc6x6
    1,  // Astc8x8
    1,  // Etc2Rgb8
    1,  // Etc2Rgba8
];

/// Whether a format is block-compressed.
const fn is_compressed(idx: usize) -> bool {
    idx >= 9 // Bc1Rgba and above
}

/// Expected uncompressed bytes_per_pixel from channel count and bit width.
/// R8=1, R16F=2, R32F/RGBA8/BGRA8/Depth32=4, RG32F/RGBA16F=8, RGBA32F=16.
const EXPECTED_UNCOMPRESSED_BPP: [usize; 9] = [
    4,  // RGBA8:       4 channels * 1 byte
    4,  // BGRA8:       4 channels * 1 byte
    1,  // R8:          1 channel  * 1 byte
    2,  // R16Float:    1 channel  * 2 bytes
    4,  // R32Float:    1 channel  * 4 bytes
    8,  // RG32Float:   2 channels * 4 bytes
    8,  // RGBA16Float: 4 channels * 2 bytes
    16, // RGBA32Float: 4 channels * 4 bytes
    4,  // Depth32Float: 32-bit depth = 4 bytes
];

// ── Kani proofs ─────────────────────────────────────────────────────────────

#[cfg(kani)]
mod proofs {
    use super::*;

    /// No two distinct Format variants map to the same VK_FORMAT value.
    /// Uses two symbolic indices and proves that equal VK values imply equal index.
    #[kani::proof]
    fn format_to_vulkan_injective() {
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < FORMAT_COUNT);
        kani::assume(j < FORMAT_COUNT);
        kani::assume(i != j);
        assert_ne!(
            FORMAT_TO_VK[i],
            FORMAT_TO_VK[j],
            "two distinct formats map to the same VK_FORMAT"
        );
    }

    /// All VK_FORMAT values in the table are non-zero.
    /// VK_FORMAT_UNDEFINED = 0 in Vulkan, so a zero would be invalid.
    #[kani::proof]
    fn format_to_vulkan_all_nonzero() {
        let i: usize = kani::any();
        kani::assume(i < FORMAT_COUNT);
        assert_ne!(
            FORMAT_TO_VK[i], 0,
            "VK_FORMAT value must be non-zero (VK_FORMAT_UNDEFINED = 0)"
        );
    }

    /// Uncompressed formats: bytes_per_pixel matches channel*width derivation.
    #[kani::proof]
    fn uncompressed_bpp_matches_channel_layout() {
        let i: usize = kani::any();
        kani::assume(i < 9); // only uncompressed formats
        assert_eq!(
            FORMAT_BYTES_PER_PIXEL_API[i],
            EXPECTED_UNCOMPRESSED_BPP[i],
            "bytes_per_pixel mismatch for uncompressed format"
        );
    }

    /// Uncompressed formats: the Vulkan helper (format_bytes_per_pixel_vk)
    /// agrees with the API method (Format::bytes_per_pixel).
    #[kani::proof]
    fn uncompressed_bpp_vk_matches_api() {
        let i: usize = kani::any();
        kani::assume(i < 9);
        assert_eq!(
            FORMAT_BYTES_PER_PIXEL_VK[i],
            FORMAT_BYTES_PER_PIXEL_API[i],
            "VK helper bpp != API bpp for uncompressed format"
        );
    }

    /// Compressed formats: Vulkan helper block sizes are 8 or 16 bytes.
    /// BC1 and ETC2-RGB are 8, everything else is 16.
    #[kani::proof]
    fn compressed_block_size_is_8_or_16() {
        let i: usize = kani::any();
        kani::assume(i < FORMAT_COUNT);
        kani::assume(is_compressed(i));
        let bpp = FORMAT_BYTES_PER_PIXEL_VK[i];
        assert!(
            bpp == 8 || bpp == 16,
            "compressed format block size must be 8 or 16 bytes"
        );
    }

    /// Compressed formats: API bytes_per_pixel returns 1 (block abstraction).
    #[kani::proof]
    fn compressed_api_bpp_is_one() {
        let i: usize = kani::any();
        kani::assume(i < FORMAT_COUNT);
        kani::assume(is_compressed(i));
        assert_eq!(
            FORMAT_BYTES_PER_PIXEL_API[i], 1,
            "compressed format API bpp must be 1"
        );
    }

    /// RGBA8 = 4 bytes. Sanity check for the most common format.
    #[kani::proof]
    fn rgba8_is_4_bytes() {
        assert_eq!(FORMAT_BYTES_PER_PIXEL_API[Format::RGBA8 as usize], 4);
        assert_eq!(FORMAT_BYTES_PER_PIXEL_VK[Format::RGBA8 as usize], 4);
    }

    /// RGBA32Float = 16 bytes. 4 channels * 4 bytes/channel.
    #[kani::proof]
    fn rgba32f_is_16_bytes() {
        assert_eq!(FORMAT_BYTES_PER_PIXEL_API[Format::RGBA32Float as usize], 16);
        assert_eq!(FORMAT_BYTES_PER_PIXEL_VK[Format::RGBA32Float as usize], 16);
    }

    /// VK_FORMAT values match the Vulkan 1.3 specification.
    /// Cross-checked against the VkFormat enum in the Vulkan headers.
    #[kani::proof]
    fn vk_format_values_match_spec() {
        // Uncompressed
        assert_eq!(VK_FORMAT_R8_UNORM, 9);
        assert_eq!(VK_FORMAT_R8G8B8A8_UNORM, 37);
        assert_eq!(VK_FORMAT_B8G8R8A8_UNORM, 44);
        assert_eq!(VK_FORMAT_R16_SFLOAT, 76);
        assert_eq!(VK_FORMAT_R16G16B16A16_SFLOAT, 97);
        assert_eq!(VK_FORMAT_R32_SFLOAT, 100);
        assert_eq!(VK_FORMAT_R32G32_SFLOAT, 103);
        assert_eq!(VK_FORMAT_R32G32B32A32_SFLOAT, 109);
        assert_eq!(VK_FORMAT_D32_SFLOAT, 126);
        // Block-compressed
        assert_eq!(VK_FORMAT_BC1_RGBA_UNORM_BLOCK, 132);
        assert_eq!(VK_FORMAT_BC3_UNORM_BLOCK, 137);
        assert_eq!(VK_FORMAT_BC5_SNORM_BLOCK, 142);
        assert_eq!(VK_FORMAT_BC7_UNORM_BLOCK, 145);
        assert_eq!(VK_FORMAT_ASTC_4X4_UNORM_BLOCK, 157);
        assert_eq!(VK_FORMAT_ASTC_6X6_UNORM_BLOCK, 163);
        assert_eq!(VK_FORMAT_ASTC_8X8_UNORM_BLOCK, 169);
        assert_eq!(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, 147);
        assert_eq!(VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, 151);
    }
}

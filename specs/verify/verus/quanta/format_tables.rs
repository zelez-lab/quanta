//! Verus mirror of Quanta format conversion tables.
//!
//! Mirrors the production conversion functions:
//!   - `src/driver/vulkan/helpers.rs::format_to_vulkan`
//!   - `src/driver/metal/device_impl.rs::format_to_metal`
//!   - `src/driver/vulkan/helpers.rs::format_bytes_per_pixel_vk`
//!   - `src/driver/metal/device_impl.rs::format_bytes_per_pixel`
//!
//! Verified properties:
//!
//! | Theorem                  | What it proves                                        |
//! |--------------------------|-------------------------------------------------------|
//! | T2.2 exhaustive          | Every Format variant maps to exactly one VK_FORMAT.   |
//! | T2.2 injective           | No two distinct Formats share a VK_FORMAT constant.   |
//! | T2.3 exhaustive          | Every Format variant maps to exactly one MTL constant. |
//! | T2.3 injective           | No two distinct Formats share a MTL constant.          |
//! | byte_size_cross_backend  | Vulkan and Metal agree on bytes-per-pixel (uncompressed). |
//! | vk_no_collision          | All 18 VK_FORMAT values are distinct.                 |
//! | mtl_no_collision         | All 18 MTL_PIXEL_FORMAT values are distinct.           |

use vstd::prelude::*;

verus! {

// ── Format enum (mirror of src/api/types.rs::Format) ───────────────

pub enum Format {
    RGBA8,
    BGRA8,
    R8,
    R16Float,
    R32Float,
    RG32Float,
    RGBA16Float,
    RGBA32Float,
    Depth32Float,
    Bc1Rgba,
    Bc3Rgba,
    Bc5Rg,
    Bc7Rgba,
    Astc4x4,
    Astc6x6,
    Astc8x8,
    Etc2Rgb8,
    Etc2Rgba8,
}

// ── Vulkan format constants (mirror of src/driver/vulkan/ffi/constants.rs) ──

pub open spec fn vk_format_r8_unorm()              -> u32 { 9u32 }
pub open spec fn vk_format_r16_sfloat()             -> u32 { 76u32 }
pub open spec fn vk_format_r32_sfloat()             -> u32 { 100u32 }
pub open spec fn vk_format_r32g32_sfloat()          -> u32 { 103u32 }
pub open spec fn vk_format_r8g8b8a8_unorm()         -> u32 { 37u32 }
pub open spec fn vk_format_b8g8r8a8_unorm()         -> u32 { 44u32 }
pub open spec fn vk_format_r16g16b16a16_sfloat()    -> u32 { 97u32 }
pub open spec fn vk_format_r32g32b32a32_sfloat()    -> u32 { 109u32 }
pub open spec fn vk_format_d32_sfloat()             -> u32 { 126u32 }
pub open spec fn vk_format_bc1_rgba_unorm_block()   -> u32 { 132u32 }
pub open spec fn vk_format_bc3_unorm_block()        -> u32 { 137u32 }
pub open spec fn vk_format_bc5_snorm_block()        -> u32 { 142u32 }
pub open spec fn vk_format_bc7_unorm_block()        -> u32 { 145u32 }
pub open spec fn vk_format_astc_4x4_unorm_block()   -> u32 { 157u32 }
pub open spec fn vk_format_astc_6x6_unorm_block()   -> u32 { 163u32 }
pub open spec fn vk_format_astc_8x8_unorm_block()   -> u32 { 169u32 }
pub open spec fn vk_format_etc2_r8g8b8_unorm_block()    -> u32 { 147u32 }
pub open spec fn vk_format_etc2_r8g8b8a8_unorm_block()  -> u32 { 151u32 }

// ── Metal format constants (mirror of src/driver/metal/ffi/constants.rs) ──

pub open spec fn mtl_pixel_format_r8_unorm()        -> u64 { 10u64 }
pub open spec fn mtl_pixel_format_r16_float()       -> u64 { 25u64 }
pub open spec fn mtl_pixel_format_r32_float()       -> u64 { 55u64 }
pub open spec fn mtl_pixel_format_rg32_float()      -> u64 { 63u64 }
pub open spec fn mtl_pixel_format_rgba8_unorm()     -> u64 { 70u64 }
pub open spec fn mtl_pixel_format_bgra8_unorm()     -> u64 { 80u64 }
pub open spec fn mtl_pixel_format_rgba16_float()    -> u64 { 115u64 }
pub open spec fn mtl_pixel_format_rgba32_float()    -> u64 { 125u64 }
pub open spec fn mtl_pixel_format_depth32_float()   -> u64 { 252u64 }
pub open spec fn mtl_pixel_format_bc1_rgba()        -> u64 { 130u64 }
pub open spec fn mtl_pixel_format_bc3_rgba()        -> u64 { 132u64 }
pub open spec fn mtl_pixel_format_bc5_rg_snorm()    -> u64 { 135u64 }
pub open spec fn mtl_pixel_format_bc7_rgba_unorm()  -> u64 { 140u64 }
pub open spec fn mtl_pixel_format_astc_4x4_ldr()    -> u64 { 204u64 }
pub open spec fn mtl_pixel_format_astc_6x6_ldr()    -> u64 { 208u64 }
pub open spec fn mtl_pixel_format_astc_8x8_ldr()    -> u64 { 212u64 }
pub open spec fn mtl_pixel_format_etc2_rgb8()       -> u64 { 180u64 }
pub open spec fn mtl_pixel_format_eac_rgba8()       -> u64 { 178u64 }

// ── Conversion spec functions ──────────────────────────────────────

/// Mirror of format_to_vulkan (src/driver/vulkan/helpers.rs).
pub open spec fn format_to_vk(f: Format) -> u32 {
    match f {
        Format::RGBA8        => vk_format_r8g8b8a8_unorm(),
        Format::BGRA8        => vk_format_b8g8r8a8_unorm(),
        Format::R8           => vk_format_r8_unorm(),
        Format::R16Float     => vk_format_r16_sfloat(),
        Format::R32Float     => vk_format_r32_sfloat(),
        Format::RG32Float    => vk_format_r32g32_sfloat(),
        Format::RGBA16Float  => vk_format_r16g16b16a16_sfloat(),
        Format::RGBA32Float  => vk_format_r32g32b32a32_sfloat(),
        Format::Depth32Float => vk_format_d32_sfloat(),
        Format::Bc1Rgba      => vk_format_bc1_rgba_unorm_block(),
        Format::Bc3Rgba      => vk_format_bc3_unorm_block(),
        Format::Bc5Rg        => vk_format_bc5_snorm_block(),
        Format::Bc7Rgba      => vk_format_bc7_unorm_block(),
        Format::Astc4x4      => vk_format_astc_4x4_unorm_block(),
        Format::Astc6x6      => vk_format_astc_6x6_unorm_block(),
        Format::Astc8x8      => vk_format_astc_8x8_unorm_block(),
        Format::Etc2Rgb8     => vk_format_etc2_r8g8b8_unorm_block(),
        Format::Etc2Rgba8    => vk_format_etc2_r8g8b8a8_unorm_block(),
    }
}

/// Mirror of format_to_metal (src/driver/metal/device_impl.rs).
pub open spec fn format_to_mtl(f: Format) -> u64 {
    match f {
        Format::RGBA8        => mtl_pixel_format_rgba8_unorm(),
        Format::BGRA8        => mtl_pixel_format_bgra8_unorm(),
        Format::R8           => mtl_pixel_format_r8_unorm(),
        Format::R16Float     => mtl_pixel_format_r16_float(),
        Format::R32Float     => mtl_pixel_format_r32_float(),
        Format::RG32Float    => mtl_pixel_format_rg32_float(),
        Format::RGBA16Float  => mtl_pixel_format_rgba16_float(),
        Format::RGBA32Float  => mtl_pixel_format_rgba32_float(),
        Format::Depth32Float => mtl_pixel_format_depth32_float(),
        Format::Bc1Rgba      => mtl_pixel_format_bc1_rgba(),
        Format::Bc3Rgba      => mtl_pixel_format_bc3_rgba(),
        Format::Bc5Rg        => mtl_pixel_format_bc5_rg_snorm(),
        Format::Bc7Rgba      => mtl_pixel_format_bc7_rgba_unorm(),
        Format::Astc4x4      => mtl_pixel_format_astc_4x4_ldr(),
        Format::Astc6x6      => mtl_pixel_format_astc_6x6_ldr(),
        Format::Astc8x8      => mtl_pixel_format_astc_8x8_ldr(),
        Format::Etc2Rgb8     => mtl_pixel_format_etc2_rgb8(),
        Format::Etc2Rgba8    => mtl_pixel_format_eac_rgba8(),
    }
}

/// Bytes per pixel for Vulkan (mirror of format_bytes_per_pixel_vk).
pub open spec fn bytes_per_pixel_vk(f: Format) -> nat {
    match f {
        Format::R8                                         => 1,
        Format::R16Float                                   => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8   => 4,
        Format::RG32Float | Format::RGBA16Float            => 8,
        Format::RGBA32Float                                => 16,
        Format::Depth32Float                               => 4,
        Format::Bc1Rgba | Format::Etc2Rgb8                 => 8,
        Format::Bc3Rgba | Format::Bc5Rg
            | Format::Bc7Rgba | Format::Etc2Rgba8          => 16,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
    }
}

/// Bytes per pixel for Metal (mirror of format_bytes_per_pixel).
pub open spec fn bytes_per_pixel_mtl(f: Format) -> nat {
    match f {
        Format::R8                                         => 1,
        Format::R16Float                                   => 2,
        Format::R32Float | Format::RGBA8 | Format::BGRA8   => 4,
        Format::RG32Float | Format::RGBA16Float            => 8,
        Format::RGBA32Float                                => 16,
        Format::Depth32Float                               => 4,
        Format::Bc1Rgba | Format::Etc2Rgb8                 => 8,
        Format::Bc3Rgba | Format::Bc5Rg
            | Format::Bc7Rgba | Format::Etc2Rgba8          => 16,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 16,
    }
}

// ── Theorems ────────────────────────────────────────────────────────

// ── T2.2: Vulkan mapping is exhaustive + injective ──────────────────

/// T2.2 exhaustive: every Format variant produces a non-zero VK_FORMAT.
/// (Exhaustive by construction -- match is total; the non-zero postcondition
/// confirms no variant accidentally maps to VK_FORMAT_UNDEFINED = 0.)
proof fn t2_2_vk_exhaustive(f: Format)
    ensures format_to_vk(f) > 0u32,
{
    match f {
        Format::RGBA8        => {},
        Format::BGRA8        => {},
        Format::R8           => {},
        Format::R16Float     => {},
        Format::R32Float     => {},
        Format::RG32Float    => {},
        Format::RGBA16Float  => {},
        Format::RGBA32Float  => {},
        Format::Depth32Float => {},
        Format::Bc1Rgba      => {},
        Format::Bc3Rgba      => {},
        Format::Bc5Rg        => {},
        Format::Bc7Rgba      => {},
        Format::Astc4x4      => {},
        Format::Astc6x6      => {},
        Format::Astc8x8      => {},
        Format::Etc2Rgb8     => {},
        Format::Etc2Rgba8    => {},
    }
}

/// T2.2 injective: distinct Format variants produce distinct VK_FORMAT values.
proof fn t2_2_vk_injective(a: Format, b: Format)
    requires format_to_vk(a) == format_to_vk(b),
    ensures a == b,
{
    match a {
        Format::RGBA8        => { match b { Format::RGBA8        => {} _ => {} } },
        Format::BGRA8        => { match b { Format::BGRA8        => {} _ => {} } },
        Format::R8           => { match b { Format::R8           => {} _ => {} } },
        Format::R16Float     => { match b { Format::R16Float     => {} _ => {} } },
        Format::R32Float     => { match b { Format::R32Float     => {} _ => {} } },
        Format::RG32Float    => { match b { Format::RG32Float    => {} _ => {} } },
        Format::RGBA16Float  => { match b { Format::RGBA16Float  => {} _ => {} } },
        Format::RGBA32Float  => { match b { Format::RGBA32Float  => {} _ => {} } },
        Format::Depth32Float => { match b { Format::Depth32Float => {} _ => {} } },
        Format::Bc1Rgba      => { match b { Format::Bc1Rgba      => {} _ => {} } },
        Format::Bc3Rgba      => { match b { Format::Bc3Rgba      => {} _ => {} } },
        Format::Bc5Rg        => { match b { Format::Bc5Rg        => {} _ => {} } },
        Format::Bc7Rgba      => { match b { Format::Bc7Rgba      => {} _ => {} } },
        Format::Astc4x4      => { match b { Format::Astc4x4      => {} _ => {} } },
        Format::Astc6x6      => { match b { Format::Astc6x6      => {} _ => {} } },
        Format::Astc8x8      => { match b { Format::Astc8x8      => {} _ => {} } },
        Format::Etc2Rgb8     => { match b { Format::Etc2Rgb8     => {} _ => {} } },
        Format::Etc2Rgba8    => { match b { Format::Etc2Rgba8    => {} _ => {} } },
    }
}

// ── T2.3: Metal mapping is exhaustive + injective ───────────────────

/// T2.3 exhaustive: every Format variant produces a non-zero MTL_PIXEL_FORMAT.
proof fn t2_3_mtl_exhaustive(f: Format)
    ensures format_to_mtl(f) > 0u64,
{
    match f {
        Format::RGBA8        => {},
        Format::BGRA8        => {},
        Format::R8           => {},
        Format::R16Float     => {},
        Format::R32Float     => {},
        Format::RG32Float    => {},
        Format::RGBA16Float  => {},
        Format::RGBA32Float  => {},
        Format::Depth32Float => {},
        Format::Bc1Rgba      => {},
        Format::Bc3Rgba      => {},
        Format::Bc5Rg        => {},
        Format::Bc7Rgba      => {},
        Format::Astc4x4      => {},
        Format::Astc6x6      => {},
        Format::Astc8x8      => {},
        Format::Etc2Rgb8     => {},
        Format::Etc2Rgba8    => {},
    }
}

/// T2.3 injective: distinct Format variants produce distinct MTL values.
proof fn t2_3_mtl_injective(a: Format, b: Format)
    requires format_to_mtl(a) == format_to_mtl(b),
    ensures a == b,
{
    match a {
        Format::RGBA8        => { match b { Format::RGBA8        => {} _ => {} } },
        Format::BGRA8        => { match b { Format::BGRA8        => {} _ => {} } },
        Format::R8           => { match b { Format::R8           => {} _ => {} } },
        Format::R16Float     => { match b { Format::R16Float     => {} _ => {} } },
        Format::R32Float     => { match b { Format::R32Float     => {} _ => {} } },
        Format::RG32Float    => { match b { Format::RG32Float    => {} _ => {} } },
        Format::RGBA16Float  => { match b { Format::RGBA16Float  => {} _ => {} } },
        Format::RGBA32Float  => { match b { Format::RGBA32Float  => {} _ => {} } },
        Format::Depth32Float => { match b { Format::Depth32Float => {} _ => {} } },
        Format::Bc1Rgba      => { match b { Format::Bc1Rgba      => {} _ => {} } },
        Format::Bc3Rgba      => { match b { Format::Bc3Rgba      => {} _ => {} } },
        Format::Bc5Rg        => { match b { Format::Bc5Rg        => {} _ => {} } },
        Format::Bc7Rgba      => { match b { Format::Bc7Rgba      => {} _ => {} } },
        Format::Astc4x4      => { match b { Format::Astc4x4      => {} _ => {} } },
        Format::Astc6x6      => { match b { Format::Astc6x6      => {} _ => {} } },
        Format::Astc8x8      => { match b { Format::Astc8x8      => {} _ => {} } },
        Format::Etc2Rgb8     => { match b { Format::Etc2Rgb8     => {} _ => {} } },
        Format::Etc2Rgba8    => { match b { Format::Etc2Rgba8    => {} _ => {} } },
    }
}

// ── Cross-backend byte size consistency ─────────────────────────────

/// Vulkan and Metal drivers agree on bytes-per-pixel for every format.
proof fn byte_size_cross_backend(f: Format)
    ensures bytes_per_pixel_vk(f) == bytes_per_pixel_mtl(f),
{
    match f {
        Format::RGBA8        => {},
        Format::BGRA8        => {},
        Format::R8           => {},
        Format::R16Float     => {},
        Format::R32Float     => {},
        Format::RG32Float    => {},
        Format::RGBA16Float  => {},
        Format::RGBA32Float  => {},
        Format::Depth32Float => {},
        Format::Bc1Rgba      => {},
        Format::Bc3Rgba      => {},
        Format::Bc5Rg        => {},
        Format::Bc7Rgba      => {},
        Format::Astc4x4      => {},
        Format::Astc6x6      => {},
        Format::Astc8x8      => {},
        Format::Etc2Rgb8     => {},
        Format::Etc2Rgba8    => {},
    }
}

/// Byte sizes are always positive.
proof fn byte_size_positive(f: Format)
    ensures bytes_per_pixel_vk(f) > 0,
{
    match f {
        Format::RGBA8        => {},
        Format::BGRA8        => {},
        Format::R8           => {},
        Format::R16Float     => {},
        Format::R32Float     => {},
        Format::RG32Float    => {},
        Format::RGBA16Float  => {},
        Format::RGBA32Float  => {},
        Format::Depth32Float => {},
        Format::Bc1Rgba      => {},
        Format::Bc3Rgba      => {},
        Format::Bc5Rg        => {},
        Format::Bc7Rgba      => {},
        Format::Astc4x4      => {},
        Format::Astc6x6      => {},
        Format::Astc8x8      => {},
        Format::Etc2Rgb8     => {},
        Format::Etc2Rgba8    => {},
    }
}

// ── No-collision: all 18 VK_FORMAT constants are pairwise distinct ──

/// All 18 Vulkan format constants used in the table are pairwise distinct.
/// Proved via injectivity: if format_to_vk(a) == format_to_vk(b) then a == b,
/// so the contrapositive gives a != b implies format_to_vk(a) != format_to_vk(b).
proof fn vk_no_collision(a: Format, b: Format)
    requires a != b,
    ensures format_to_vk(a) != format_to_vk(b),
{
    // Contrapositive of t2_2_vk_injective.
    if format_to_vk(a) == format_to_vk(b) {
        t2_2_vk_injective(a, b);
        // Now a == b, contradicting requires.
    }
}

/// All 18 Metal format constants used in the table are pairwise distinct.
proof fn mtl_no_collision(a: Format, b: Format)
    requires a != b,
    ensures format_to_mtl(a) != format_to_mtl(b),
{
    if format_to_mtl(a) == format_to_mtl(b) {
        t2_3_mtl_injective(a, b);
    }
}

} // verus!

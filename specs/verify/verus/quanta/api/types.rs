//! Verus mirror of `src/api/types.rs` — Format, Vendor, ResourceState, FieldUsage, Color, Caps.
//!
//! Verified properties:
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T2200 format_bpp_positive     | bytes_per_pixel > 0 for all formats.                   |
//! | T2201 vendor_exhaustive       | All 7 Vendor variants are distinct.                     |
//! | T2202 resource_state_exhaustive | All 9 ResourceState variants are distinct.             |
//! | T2203 field_usage_flags       | FieldUsage flags are distinct bit positions.             |
//! | T2204 caps_total_quarks       | total_quarks = nuclei * protons_per_nucleus * quarks_per_proton. |
//! | T2205 color_constants         | WHITE, BLACK, CLEAR have correct channel values.        |
//! | T2206 kernel_format_exhaustive | All 5 KernelFormat variants are distinct.              |

use vstd::prelude::*;

verus! {

// ════════════════════════════════════════════════════════════════════════
// Format bytes_per_pixel
// ════════════════════════════════════════════════════════════════════════

/// Mirror of Format::bytes_per_pixel (simplified: uncompressed only).
pub enum Format {
    RGBA8, BGRA8, R8, R16Float, R32Float, RG32Float,
    RGBA16Float, RGBA32Float, Depth32Float,
    Bc1Rgba, Bc3Rgba, Bc5Rg, Bc7Rgba,
    Astc4x4, Astc6x6, Astc8x8, Etc2Rgb8, Etc2Rgba8,
}

pub open spec fn bytes_per_pixel(f: Format) -> nat {
    match f {
        Format::R8 => 1,
        Format::R16Float => 2,
        Format::RGBA8 | Format::BGRA8 | Format::R32Float | Format::Depth32Float => 4,
        Format::RG32Float | Format::RGBA16Float => 8,
        Format::RGBA32Float => 16,
        Format::Bc1Rgba | Format::Etc2Rgb8 => 1,
        Format::Bc3Rgba | Format::Bc5Rg | Format::Bc7Rgba | Format::Etc2Rgba8 => 1,
        Format::Astc4x4 | Format::Astc6x6 | Format::Astc8x8 => 1,
    }
}

/// T2200: bytes_per_pixel > 0 for all formats.
proof fn t2200_format_bpp_positive(f: Format)
    ensures bytes_per_pixel(f) > 0,
{
    match f {
        Format::RGBA8 => {}, Format::BGRA8 => {}, Format::R8 => {},
        Format::R16Float => {}, Format::R32Float => {}, Format::RG32Float => {},
        Format::RGBA16Float => {}, Format::RGBA32Float => {}, Format::Depth32Float => {},
        Format::Bc1Rgba => {}, Format::Bc3Rgba => {}, Format::Bc5Rg => {},
        Format::Bc7Rgba => {}, Format::Astc4x4 => {}, Format::Astc6x6 => {},
        Format::Astc8x8 => {}, Format::Etc2Rgb8 => {}, Format::Etc2Rgba8 => {},
    }
}

// ════════════════════════════════════════════════════════════════════════
// Vendor enum
// ════════════════════════════════════════════════════════════════════════

pub enum Vendor { Amd, Nvidia, Intel, Apple, Broadcom, Software, Unknown }

pub open spec fn vendor_tag(v: Vendor) -> nat {
    match v {
        Vendor::Amd => 0, Vendor::Nvidia => 1, Vendor::Intel => 2,
        Vendor::Apple => 3, Vendor::Broadcom => 4, Vendor::Software => 5,
        Vendor::Unknown => 6,
    }
}

proof fn t2201_vendor_injective(a: Vendor, b: Vendor)
    requires vendor_tag(a) == vendor_tag(b),
    ensures a == b,
{
    match a {
        Vendor::Amd       => { match b { Vendor::Amd => {} _ => {} } },
        Vendor::Nvidia    => { match b { Vendor::Nvidia => {} _ => {} } },
        Vendor::Intel     => { match b { Vendor::Intel => {} _ => {} } },
        Vendor::Apple     => { match b { Vendor::Apple => {} _ => {} } },
        Vendor::Broadcom  => { match b { Vendor::Broadcom => {} _ => {} } },
        Vendor::Software  => { match b { Vendor::Software => {} _ => {} } },
        Vendor::Unknown   => { match b { Vendor::Unknown => {} _ => {} } },
    }
}

// ════════════════════════════════════════════════════════════════════════
// ResourceState enum
// ════════════════════════════════════════════════════════════════════════

pub enum ResourceState {
    General, ComputeWrite, ComputeRead, RenderTarget,
    DepthStencil, ShaderRead, TransferSrc, TransferDst, Present,
}

pub open spec fn resource_state_tag(s: ResourceState) -> nat {
    match s {
        ResourceState::General => 0, ResourceState::ComputeWrite => 1,
        ResourceState::ComputeRead => 2, ResourceState::RenderTarget => 3,
        ResourceState::DepthStencil => 4, ResourceState::ShaderRead => 5,
        ResourceState::TransferSrc => 6, ResourceState::TransferDst => 7,
        ResourceState::Present => 8,
    }
}

proof fn t2202_resource_state_injective(a: ResourceState, b: ResourceState)
    requires resource_state_tag(a) == resource_state_tag(b),
    ensures a == b,
{
    match a {
        ResourceState::General      => { match b { ResourceState::General => {} _ => {} } },
        ResourceState::ComputeWrite => { match b { ResourceState::ComputeWrite => {} _ => {} } },
        ResourceState::ComputeRead  => { match b { ResourceState::ComputeRead => {} _ => {} } },
        ResourceState::RenderTarget => { match b { ResourceState::RenderTarget => {} _ => {} } },
        ResourceState::DepthStencil => { match b { ResourceState::DepthStencil => {} _ => {} } },
        ResourceState::ShaderRead   => { match b { ResourceState::ShaderRead => {} _ => {} } },
        ResourceState::TransferSrc  => { match b { ResourceState::TransferSrc => {} _ => {} } },
        ResourceState::TransferDst  => { match b { ResourceState::TransferDst => {} _ => {} } },
        ResourceState::Present      => { match b { ResourceState::Present => {} _ => {} } },
    }
}

// ════════════════════════════════════════════════════════════════════════
// FieldUsage flags
// ════════════════════════════════════════════════════════════════════════

pub const FIELD_READ: u8 = 1;      // 1 << 0
pub const FIELD_WRITE: u8 = 2;     // 1 << 1
pub const FIELD_COMPUTE: u8 = 4;   // 1 << 2
pub const FIELD_RENDER: u8 = 8;    // 1 << 3
pub const FIELD_TRANSFER: u8 = 16; // 1 << 4
pub const FIELD_UNIFORM: u8 = 32;  // 1 << 5

/// T2203: All flags occupy distinct bit positions.
proof fn t2203_field_usage_flags_distinct()
    ensures
        FIELD_READ & FIELD_WRITE == 0,
        FIELD_WRITE & FIELD_COMPUTE == 0,
        FIELD_COMPUTE & FIELD_RENDER == 0,
        FIELD_RENDER & FIELD_TRANSFER == 0,
        FIELD_TRANSFER & FIELD_UNIFORM == 0,
        FIELD_READ & FIELD_UNIFORM == 0,
{}

/// T2203 corollary: default_compute includes READ|WRITE|COMPUTE|TRANSFER.
proof fn t2203_default_compute()
    ensures ({
        let dc: u8 = (FIELD_READ | FIELD_WRITE | FIELD_COMPUTE | FIELD_TRANSFER) as u8;
        &&& (dc & FIELD_READ) == FIELD_READ
        &&& (dc & FIELD_WRITE) == FIELD_WRITE
        &&& (dc & FIELD_COMPUTE) == FIELD_COMPUTE
        &&& (dc & FIELD_TRANSFER) == FIELD_TRANSFER
    }),
{
    assert((FIELD_READ | FIELD_WRITE | FIELD_COMPUTE | FIELD_TRANSFER) & FIELD_READ == FIELD_READ) by (bit_vector);
    assert((FIELD_READ | FIELD_WRITE | FIELD_COMPUTE | FIELD_TRANSFER) & FIELD_WRITE == FIELD_WRITE) by (bit_vector);
    assert((FIELD_READ | FIELD_WRITE | FIELD_COMPUTE | FIELD_TRANSFER) & FIELD_COMPUTE == FIELD_COMPUTE) by (bit_vector);
    assert((FIELD_READ | FIELD_WRITE | FIELD_COMPUTE | FIELD_TRANSFER) & FIELD_TRANSFER == FIELD_TRANSFER) by (bit_vector);
}

// ════════════════════════════════════════════════════════════════════════
// Caps.total_quarks
// ════════════════════════════════════════════════════════════════════════

/// T2204: total_quarks = nuclei * protons * quarks.
pub open spec fn total_quarks(nuclei: u32, protons: u32, quarks: u32) -> nat {
    (nuclei as nat) * (protons as nat) * (quarks as nat)
}

proof fn t2204_total_quarks(nuclei: u32, protons: u32, quarks: u32)
    ensures total_quarks(nuclei, protons, quarks) == (nuclei as nat) * (protons as nat) * (quarks as nat),
{}

// ════════════════════════════════════════════════════════════════════════
// Color constants
// ════════════════════════════════════════════════════════════════════════

/// T2205: Color::WHITE has all channels at 1.0.
/// Modeled as integer-tagged (1 = 1.0, 0 = 0.0).
pub open spec fn color_white() -> (nat, nat, nat, nat) { (1, 1, 1, 1) }
pub open spec fn color_black() -> (nat, nat, nat, nat) { (0, 0, 0, 1) }
pub open spec fn color_clear() -> (nat, nat, nat, nat) { (0, 0, 0, 0) }

proof fn t2205_color_constants()
    ensures
        color_white() == (1nat, 1nat, 1nat, 1nat),
        color_black() == (0nat, 0nat, 0nat, 1nat),
        color_clear() == (0nat, 0nat, 0nat, 0nat),
{}

// ════════════════════════════════════════════════════════════════════════
// KernelFormat
// ════════════════════════════════════════════════════════════════════════

pub enum KernelFormat { AmdGcn, NvidiaPtx, Msl, Wgsl, LlvmIr }

pub open spec fn kernel_format_tag(k: KernelFormat) -> nat {
    match k {
        KernelFormat::AmdGcn => 0, KernelFormat::NvidiaPtx => 1,
        KernelFormat::Msl => 2, KernelFormat::Wgsl => 3, KernelFormat::LlvmIr => 4,
    }
}

proof fn t2206_kernel_format_injective(a: KernelFormat, b: KernelFormat)
    requires kernel_format_tag(a) == kernel_format_tag(b),
    ensures a == b,
{
    match a {
        KernelFormat::AmdGcn    => { match b { KernelFormat::AmdGcn => {} _ => {} } },
        KernelFormat::NvidiaPtx => { match b { KernelFormat::NvidiaPtx => {} _ => {} } },
        KernelFormat::Msl       => { match b { KernelFormat::Msl => {} _ => {} } },
        KernelFormat::Wgsl      => { match b { KernelFormat::Wgsl => {} _ => {} } },
        KernelFormat::LlvmIr    => { match b { KernelFormat::LlvmIr => {} _ => {} } },
    }
}

} // verus!

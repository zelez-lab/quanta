//! Tier 1 (host, no GPU) conformance tests — API type invariants.
//!
//! Tests Format, ResourceState, LoadOp, StoreOp, CompareOp, BlendState
//! presets, SamplerDesc defaults, PipelineDesc defaults, and GpuType impls.
//!
//! Run: cargo test --test host_types

use quanta::*;

// ===========================================================================
// Format: bytes_per_pixel for all 18 formats
// ===========================================================================

#[test]
fn format_bytes_per_pixel_uncompressed() {
    assert_eq!(Format::R8.bytes_per_pixel(), 1);
    assert_eq!(Format::R16Float.bytes_per_pixel(), 2);
    assert_eq!(Format::RGBA8.bytes_per_pixel(), 4);
    assert_eq!(Format::BGRA8.bytes_per_pixel(), 4);
    assert_eq!(Format::R32Float.bytes_per_pixel(), 4);
    assert_eq!(Format::Depth32Float.bytes_per_pixel(), 4);
    assert_eq!(Format::RG32Float.bytes_per_pixel(), 8);
    assert_eq!(Format::RGBA16Float.bytes_per_pixel(), 8);
    assert_eq!(Format::RGBA32Float.bytes_per_pixel(), 16);
}

#[test]
fn format_bytes_per_pixel_compressed() {
    // Block-compressed formats report per-pixel average (rounded)
    assert_eq!(Format::Bc1Rgba.bytes_per_pixel(), 1);
    assert_eq!(Format::Bc3Rgba.bytes_per_pixel(), 1);
    assert_eq!(Format::Bc5Rg.bytes_per_pixel(), 1);
    assert_eq!(Format::Bc7Rgba.bytes_per_pixel(), 1);
    assert_eq!(Format::Astc4x4.bytes_per_pixel(), 1);
    assert_eq!(Format::Astc6x6.bytes_per_pixel(), 1);
    assert_eq!(Format::Astc8x8.bytes_per_pixel(), 1);
    assert_eq!(Format::Etc2Rgb8.bytes_per_pixel(), 1);
    assert_eq!(Format::Etc2Rgba8.bytes_per_pixel(), 1);
}

#[test]
fn format_all_18_variants_exist() {
    // Verify we can construct all 18 variants
    let formats = [
        Format::RGBA8,
        Format::BGRA8,
        Format::R8,
        Format::R16Float,
        Format::R32Float,
        Format::RG32Float,
        Format::RGBA16Float,
        Format::RGBA32Float,
        Format::Depth32Float,
        Format::Bc1Rgba,
        Format::Bc3Rgba,
        Format::Bc5Rg,
        Format::Bc7Rgba,
        Format::Astc4x4,
        Format::Astc6x6,
        Format::Astc8x8,
        Format::Etc2Rgb8,
        Format::Etc2Rgba8,
    ];
    assert_eq!(formats.len(), 18);
    // Each should have a non-panicking bytes_per_pixel
    for f in &formats {
        let _ = f.bytes_per_pixel();
    }
}

// ===========================================================================
// ResourceState: all 9 variants exist and are distinct
// ===========================================================================

#[test]
fn resource_state_all_9_variants_distinct() {
    let states = [
        ResourceState::General,
        ResourceState::ComputeWrite,
        ResourceState::ComputeRead,
        ResourceState::RenderTarget,
        ResourceState::DepthStencil,
        ResourceState::ShaderRead,
        ResourceState::TransferSrc,
        ResourceState::TransferDst,
        ResourceState::Present,
    ];
    assert_eq!(states.len(), 9);

    // Each pair must be distinct
    for i in 0..states.len() {
        for j in (i + 1)..states.len() {
            assert_ne!(states[i], states[j], "states[{}] == states[{}]", i, j);
        }
    }
}

// ===========================================================================
// LoadOp/StoreOp: variant existence
// ===========================================================================

#[test]
fn load_op_variants() {
    let _clear = LoadOp::Clear(Color::BLACK);
    let _load = LoadOp::Load;
    let _dont_care = LoadOp::DontCare;
}

#[test]
fn store_op_variants() {
    let _store = StoreOp::Store;
    let _dont_care = StoreOp::DontCare;
    let _resolve = StoreOp::Resolve(0xDEAD);
}

// ===========================================================================
// CompareOp: all 8 variants
// ===========================================================================

#[test]
fn compare_op_all_8_variants() {
    let ops = [
        CompareOp::Never,
        CompareOp::Less,
        CompareOp::Equal,
        CompareOp::LessEqual,
        CompareOp::Greater,
        CompareOp::NotEqual,
        CompareOp::GreaterEqual,
        CompareOp::Always,
    ];
    assert_eq!(ops.len(), 8);

    // Each pair must be distinct
    for i in 0..ops.len() {
        for j in (i + 1)..ops.len() {
            assert_ne!(ops[i], ops[j], "ops[{}] == ops[{}]", i, j);
        }
    }
}

// ===========================================================================
// BlendState presets
// ===========================================================================

#[test]
fn blend_state_none_preset() {
    let bs = BlendState::NONE;
    assert!(!bs.enabled);
    assert_eq!(bs.src_rgb, BlendFactor::One);
    assert_eq!(bs.dst_rgb, BlendFactor::Zero);
    assert_eq!(bs.src_alpha, BlendFactor::One);
    assert_eq!(bs.dst_alpha, BlendFactor::Zero);
    assert_eq!(bs.op_rgb, BlendOp::Add);
    assert_eq!(bs.op_alpha, BlendOp::Add);
}

#[test]
fn blend_state_premultiplied_alpha_preset() {
    let bs = BlendState::PREMULTIPLIED_ALPHA;
    assert!(bs.enabled);
    assert_eq!(bs.src_rgb, BlendFactor::One);
    assert_eq!(bs.dst_rgb, BlendFactor::OneMinusSrcAlpha);
    assert_eq!(bs.src_alpha, BlendFactor::One);
    assert_eq!(bs.dst_alpha, BlendFactor::OneMinusSrcAlpha);
}

#[test]
fn blend_state_alpha_preset() {
    let bs = BlendState::ALPHA;
    assert!(bs.enabled);
    assert_eq!(bs.src_rgb, BlendFactor::SrcAlpha);
    assert_eq!(bs.dst_rgb, BlendFactor::OneMinusSrcAlpha);
    assert_eq!(bs.src_alpha, BlendFactor::One);
    assert_eq!(bs.dst_alpha, BlendFactor::OneMinusSrcAlpha);
}

// ===========================================================================
// SamplerDesc default values
// ===========================================================================

#[test]
fn sampler_desc_defaults() {
    let s = SamplerDesc::default();
    assert_eq!(s.min_filter, Filter::Linear);
    assert_eq!(s.mag_filter, Filter::Linear);
    assert_eq!(s.mip_filter, Filter::Nearest);
    assert_eq!(s.address_u, AddressMode::ClampToEdge);
    assert_eq!(s.address_v, AddressMode::ClampToEdge);
    assert_eq!(s.max_anisotropy, 1);
    assert_eq!(s.compare, None);
}

// ===========================================================================
// PipelineDesc default values
// ===========================================================================

#[test]
fn pipeline_desc_defaults() {
    let pd = PipelineDesc::default();
    assert_eq!(pd.vertex, &[] as &[u8]);
    assert_eq!(pd.fragment, &[] as &[u8]);
    assert!(pd.source.is_none());
    assert_eq!(pd.vertex_entry, "vertex_main");
    assert_eq!(pd.fragment_entry, "fragment_main");
    assert!(pd.vertex_layouts.is_empty());
    assert_eq!(pd.color_formats, vec![Format::BGRA8]);
    assert_eq!(pd.depth_format, None);
    assert_eq!(pd.sample_count, 1);
    assert_eq!(pd.cull_mode, CullMode::None);
    assert_eq!(pd.primitive, Primitive::Triangle);
    assert!(pd.specialization.is_empty());
    assert!(pd.tessellation.is_none());
    assert!(pd.mesh_shader.is_none());
    assert!(!pd.conservative_rasterization);
}

#[test]
fn pipeline_desc_default_blend_is_premultiplied() {
    let pd = PipelineDesc::default();
    assert!(pd.blend.enabled);
    assert_eq!(pd.blend.src_rgb, BlendFactor::One);
    assert_eq!(pd.blend.dst_rgb, BlendFactor::OneMinusSrcAlpha);
}

// ===========================================================================
// GpuType impls: verify gpu_size() and scalar_type()
// ===========================================================================

#[test]
fn gpu_type_f32() {
    assert_eq!(<f32 as GpuType>::gpu_size(), 4);
    assert_eq!(<f32 as GpuType>::scalar_type(), ScalarType::F32);
}

#[test]
fn gpu_type_u32() {
    assert_eq!(<u32 as GpuType>::gpu_size(), 4);
    assert_eq!(<u32 as GpuType>::scalar_type(), ScalarType::U32);
}

#[test]
fn gpu_type_i32() {
    assert_eq!(<i32 as GpuType>::gpu_size(), 4);
    assert_eq!(<i32 as GpuType>::scalar_type(), ScalarType::I32);
}

#[test]
fn gpu_type_f64() {
    assert_eq!(<f64 as GpuType>::gpu_size(), 8);
    assert_eq!(<f64 as GpuType>::scalar_type(), ScalarType::F64);
}

#[test]
fn gpu_type_u64() {
    assert_eq!(<u64 as GpuType>::gpu_size(), 8);
    assert_eq!(<u64 as GpuType>::scalar_type(), ScalarType::U64);
}

#[test]
fn gpu_type_i64() {
    assert_eq!(<i64 as GpuType>::gpu_size(), 8);
    assert_eq!(<i64 as GpuType>::scalar_type(), ScalarType::I64);
}

#[test]
fn gpu_type_u8() {
    assert_eq!(<u8 as GpuType>::gpu_size(), 1);
    assert_eq!(<u8 as GpuType>::scalar_type(), ScalarType::U8);
}

#[test]
fn gpu_type_u16() {
    assert_eq!(<u16 as GpuType>::gpu_size(), 2);
    assert_eq!(<u16 as GpuType>::scalar_type(), ScalarType::U16);
}

#[test]
fn gpu_type_i16() {
    assert_eq!(<i16 as GpuType>::gpu_size(), 2);
    assert_eq!(<i16 as GpuType>::scalar_type(), ScalarType::I16);
}

#[test]
fn gpu_type_i8() {
    assert_eq!(<i8 as GpuType>::gpu_size(), 1);
    assert_eq!(<i8 as GpuType>::scalar_type(), ScalarType::I8);
}

// ===========================================================================
// ScalarType method coverage
// ===========================================================================

#[test]
fn scalar_type_msl_names() {
    assert_eq!(ScalarType::F16.msl_name(), "half");
    assert_eq!(ScalarType::F32.msl_name(), "float");
    assert_eq!(ScalarType::F64.msl_name(), "double");
    assert_eq!(ScalarType::U8.msl_name(), "uint8_t");
    assert_eq!(ScalarType::U16.msl_name(), "ushort");
    assert_eq!(ScalarType::U32.msl_name(), "uint");
    assert_eq!(ScalarType::U64.msl_name(), "ulong");
    assert_eq!(ScalarType::I8.msl_name(), "int8_t");
    assert_eq!(ScalarType::I16.msl_name(), "short");
    assert_eq!(ScalarType::I32.msl_name(), "int");
    assert_eq!(ScalarType::I64.msl_name(), "long");
    assert_eq!(ScalarType::Bool.msl_name(), "bool");
}

#[test]
fn scalar_type_wgsl_names() {
    assert_eq!(ScalarType::F16.wgsl_name(), "f16");
    assert_eq!(ScalarType::F32.wgsl_name(), "f32");
    assert_eq!(ScalarType::F64.wgsl_name(), "f64");
    assert_eq!(ScalarType::U8.wgsl_name(), "u32");
    assert_eq!(ScalarType::U16.wgsl_name(), "u32");
    assert_eq!(ScalarType::U32.wgsl_name(), "u32");
    assert_eq!(ScalarType::U64.wgsl_name(), "u64");
    assert_eq!(ScalarType::I8.wgsl_name(), "i32");
    assert_eq!(ScalarType::I16.wgsl_name(), "i32");
    assert_eq!(ScalarType::I32.wgsl_name(), "i32");
    assert_eq!(ScalarType::I64.wgsl_name(), "i64");
    assert_eq!(ScalarType::Bool.wgsl_name(), "bool");
}

// ===========================================================================
// FieldUsage flags
// ===========================================================================

#[test]
fn field_usage_flags() {
    let compute = FieldUsage::default_compute();
    assert!(compute.has(FieldUsage::READ));
    assert!(compute.has(FieldUsage::WRITE));
    assert!(compute.has(FieldUsage::COMPUTE));
    assert!(compute.has(FieldUsage::TRANSFER));
    assert!(!compute.has(FieldUsage::RENDER));
    assert!(!compute.has(FieldUsage::UNIFORM));

    let render = FieldUsage::default_render();
    assert!(render.has(FieldUsage::READ));
    assert!(render.has(FieldUsage::RENDER));
    assert!(render.has(FieldUsage::TRANSFER));
    assert!(!render.has(FieldUsage::WRITE));

    let uniform = FieldUsage::default_uniform();
    assert!(uniform.has(FieldUsage::READ));
    assert!(uniform.has(FieldUsage::UNIFORM));
    assert!(uniform.has(FieldUsage::TRANSFER));

    let combined = FieldUsage::READ.union(FieldUsage::WRITE);
    assert!(combined.has(FieldUsage::READ));
    assert!(combined.has(FieldUsage::WRITE));
}

// ===========================================================================
// Color constants
// ===========================================================================

#[test]
fn color_constants() {
    let w = Color::WHITE;
    assert_eq!(w.r, 1.0);
    assert_eq!(w.g, 1.0);
    assert_eq!(w.b, 1.0);
    assert_eq!(w.a, 1.0);

    let b = Color::BLACK;
    assert_eq!(b.r, 0.0);
    assert_eq!(b.g, 0.0);
    assert_eq!(b.b, 0.0);
    assert_eq!(b.a, 1.0);

    let c = Color::CLEAR;
    assert_eq!(c.r, 0.0);
    assert_eq!(c.a, 0.0);

    let rgb = Color::rgb(0.5, 0.6, 0.7);
    assert_eq!(rgb.a, 1.0);

    let rgba = Color::rgba(0.1, 0.2, 0.3, 0.4);
    assert_eq!(rgba.a, 0.4);
}

// ===========================================================================
// Vendor enum
// ===========================================================================

#[test]
fn vendor_variants_distinct() {
    let vendors = [
        Vendor::Amd,
        Vendor::Nvidia,
        Vendor::Intel,
        Vendor::Apple,
        Vendor::Broadcom,
        Vendor::Software,
        Vendor::Unknown,
    ];
    assert_eq!(vendors.len(), 7);
    for i in 0..vendors.len() {
        for j in (i + 1)..vendors.len() {
            assert_ne!(vendors[i], vendors[j]);
        }
    }
}

// ===========================================================================
// ShaderStage enum
// ===========================================================================

#[test]
fn shader_stage_all_variants() {
    let stages = [
        ShaderStage::Vertex,
        ShaderStage::Fragment,
        ShaderStage::TessControl,
        ShaderStage::TessEval,
        ShaderStage::Task,
        ShaderStage::Mesh,
        ShaderStage::RayGen,
        ShaderStage::ClosestHit,
        ShaderStage::Miss,
    ];
    assert_eq!(stages.len(), 9);
    for i in 0..stages.len() {
        for j in (i + 1)..stages.len() {
            assert_ne!(stages[i], stages[j]);
        }
    }
}

// ===========================================================================
// DepthStencilState presets
// ===========================================================================

#[test]
fn depth_stencil_state_presets() {
    let none = DepthStencilState::NONE;
    assert!(!none.depth_test);
    assert!(!none.depth_write);
    assert_eq!(none.depth_compare, CompareFunc::Always);

    let less = DepthStencilState::DEPTH_LESS;
    assert!(less.depth_test);
    assert!(less.depth_write);
    assert_eq!(less.depth_compare, CompareFunc::Less);

    let read_only = DepthStencilState::DEPTH_READ_ONLY;
    assert!(read_only.depth_test);
    assert!(!read_only.depth_write);
    assert_eq!(read_only.depth_compare, CompareFunc::Less);
}

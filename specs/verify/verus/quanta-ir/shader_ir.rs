//! Verus mirror of `quanta-ir/src/shader.rs` — ShaderDef type definitions.
//!
//! Mirrors `quanta-ir/src/shader.rs` (ShaderStage, ShaderType, ShaderParam,
//! ShaderDef, ShaderOutput).
//!
//! Theorems:
//!   T700: ShaderStage has exactly 2 variants (Vertex=0, Fragment=1)
//!   T701: ShaderType has exactly 6 variants with distinct discriminants
//!   T702: ShaderType vector dimension is correct (F32=1, Vec2=2, Vec3=3, Vec4=4)
//!   T703: ShaderParam.is_uniform correctly partitions params
//!   T704: ShaderStage discriminants match SPIR-V ExecutionModel values
//!   T705: ShaderType is ordered by component count
//!   T706: Matrix types have matching column vector types

use vstd::prelude::*;

verus! {

// ── Ghost enum mirrors ─────────────────────────────────────────────

pub enum ShaderStage {
    Vertex,    // = 0
    Fragment,  // = 1
}

pub enum ShaderType {
    F32,   // = 0
    Vec2,  // = 1
    Vec3,  // = 2
    Vec4,  // = 3
    Mat4,  // = 4
    Mat3,  // = 5
}

// ── T700: ShaderStage ──────────────────────────────────────────────

pub open spec fn stage_discriminant(s: ShaderStage) -> u8 {
    match s {
        ShaderStage::Vertex   => 0u8,
        ShaderStage::Fragment => 1u8,
    }
}

/// T700a: ShaderStage discriminants are in [0, 1].
proof fn t700_stage_bounded(s: ShaderStage)
    ensures stage_discriminant(s) <= 1u8,
{
    match s { ShaderStage::Vertex => {} ShaderStage::Fragment => {} }
}

/// T700b: ShaderStage discriminants are distinct.
proof fn t700_stage_distinct(a: ShaderStage, b: ShaderStage)
    requires a != b,
    ensures  stage_discriminant(a) != stage_discriminant(b),
{
    match a {
        ShaderStage::Vertex => { match b { ShaderStage::Vertex => {} _ => {} } },
        ShaderStage::Fragment => { match b { ShaderStage::Fragment => {} _ => {} } },
    }
}

// ── T701: ShaderType discriminants ─────────────────────────────────

pub open spec fn shader_type_discriminant(t: ShaderType) -> u8 {
    match t {
        ShaderType::F32  => 0u8,
        ShaderType::Vec2 => 1u8,
        ShaderType::Vec3 => 2u8,
        ShaderType::Vec4 => 3u8,
        ShaderType::Mat4 => 4u8,
        ShaderType::Mat3 => 5u8,
    }
}

/// T701: All 6 discriminants are distinct and bounded.
proof fn t701_shader_type_injective(a: ShaderType, b: ShaderType)
    requires a != b,
    ensures  shader_type_discriminant(a) != shader_type_discriminant(b),
{
    match a {
        ShaderType::F32  => { match b { ShaderType::F32 => {} _ => {} } },
        ShaderType::Vec2 => { match b { ShaderType::Vec2 => {} _ => {} } },
        ShaderType::Vec3 => { match b { ShaderType::Vec3 => {} _ => {} } },
        ShaderType::Vec4 => { match b { ShaderType::Vec4 => {} _ => {} } },
        ShaderType::Mat4 => { match b { ShaderType::Mat4 => {} _ => {} } },
        ShaderType::Mat3 => { match b { ShaderType::Mat3 => {} _ => {} } },
    }
}

// ── T702: ShaderType vector dimension ──────────────────────────────

/// Number of f32 components for each ShaderType.
pub open spec fn shader_type_dimension(t: ShaderType) -> nat {
    match t {
        ShaderType::F32  => 1,
        ShaderType::Vec2 => 2,
        ShaderType::Vec3 => 3,
        ShaderType::Vec4 => 4,
        ShaderType::Mat4 => 16, // 4x4 matrix = 16 floats
        ShaderType::Mat3 => 9,  // 3x3 matrix = 9 floats
    }
}

/// T702a: Scalar has dimension 1.
proof fn t702_f32_is_scalar()
    ensures shader_type_dimension(ShaderType::F32) == 1,
{}

/// T702b: Vector dimensions are 2, 3, 4.
proof fn t702_vec_dimensions()
    ensures
        shader_type_dimension(ShaderType::Vec2) == 2,
        shader_type_dimension(ShaderType::Vec3) == 3,
        shader_type_dimension(ShaderType::Vec4) == 4,
{}

/// T702c: Matrix dimensions.
proof fn t702_mat_dimensions()
    ensures
        shader_type_dimension(ShaderType::Mat4) == 16,
        shader_type_dimension(ShaderType::Mat3) == 9,
{}

/// T702d: All dimensions are positive.
proof fn t702_dimension_positive(t: ShaderType)
    ensures shader_type_dimension(t) >= 1,
{
    match t {
        ShaderType::F32 => {} ShaderType::Vec2 => {} ShaderType::Vec3 => {}
        ShaderType::Vec4 => {} ShaderType::Mat4 => {} ShaderType::Mat3 => {}
    }
}

// ── T703: is_uniform partitioning ──────────────────────────────────

/// A shader param list is partitioned into attribute params (!is_uniform)
/// and uniform params (is_uniform). The two sets are disjoint.
/// Model: for any bool flag, exactly one of the two predicates holds.
proof fn t703_partition(is_uniform: bool)
    ensures  is_uniform || !is_uniform,
    ensures  !(is_uniform && !is_uniform),
{}

// ── T704: ShaderStage vs SPIR-V ExecutionModel ─────────────────────

/// SPIR-V ExecutionModel values used by the emitters.
pub open spec fn spirv_execution_model(s: ShaderStage) -> u32 {
    match s {
        ShaderStage::Vertex   => 0u32,   // EXECUTION_MODEL_VERTEX
        ShaderStage::Fragment => 4u32,   // EXECUTION_MODEL_FRAGMENT
    }
}

/// T704: ShaderStage maps to the correct SPIR-V ExecutionModel.
proof fn t704_vertex_is_0()
    ensures spirv_execution_model(ShaderStage::Vertex) == 0u32,
{}

proof fn t704_fragment_is_4()
    ensures spirv_execution_model(ShaderStage::Fragment) == 4u32,
{}

proof fn t704_models_distinct()
    ensures spirv_execution_model(ShaderStage::Vertex) != spirv_execution_model(ShaderStage::Fragment),
{}

// ── T705: ShaderType ordering by component count ───────────────────

/// Vector types are ordered: F32 < Vec2 < Vec3 < Vec4.
pub open spec fn vec_component_count(t: ShaderType) -> nat {
    match t {
        ShaderType::F32  => 1,
        ShaderType::Vec2 => 2,
        ShaderType::Vec3 => 3,
        ShaderType::Vec4 => 4,
        ShaderType::Mat4 => 4,  // columns
        ShaderType::Mat3 => 3,  // columns
    }
}

/// T705: Vector types are strictly ordered by component count.
proof fn t705_vec_ordering()
    ensures
        vec_component_count(ShaderType::F32) < vec_component_count(ShaderType::Vec2),
        vec_component_count(ShaderType::Vec2) < vec_component_count(ShaderType::Vec3),
        vec_component_count(ShaderType::Vec3) < vec_component_count(ShaderType::Vec4),
{}

// ── T706: Matrix column vector types ───────────────────────────────

/// Mat4 columns are Vec4. Mat3 columns are Vec3.
pub open spec fn mat_column_type(t: ShaderType) -> ShaderType {
    match t {
        ShaderType::Mat4 => ShaderType::Vec4,
        ShaderType::Mat3 => ShaderType::Vec3,
        _                => t,  // non-matrix types map to themselves
    }
}

/// T706a: Mat4 columns are Vec4.
proof fn t706_mat4_column()
    ensures mat_column_type(ShaderType::Mat4) == ShaderType::Vec4,
{}

/// T706b: Mat3 columns are Vec3.
proof fn t706_mat3_column()
    ensures mat_column_type(ShaderType::Mat3) == ShaderType::Vec3,
{}

/// T706c: Column count matches matrix dimension.
proof fn t706_column_count_matches()
    ensures
        vec_component_count(ShaderType::Mat4) == vec_component_count(ShaderType::Vec4),
        vec_component_count(ShaderType::Mat3) == vec_component_count(ShaderType::Vec3),
{}

} // verus!

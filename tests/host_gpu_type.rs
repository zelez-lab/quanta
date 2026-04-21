//! Tier 1 (host, no GPU) conformance tests — gpu_type derive correctness.
//!
//! Extends the existing gpu_type_test.rs with additional coverage:
//! - Nested structs (struct containing another gpu_type struct)
//! - All scalar types as fields
//! - Array fields [f32; 3], [u32; 4], [f32; 16]
//! - Alignment verification (offsets match repr(C))
//!
//! Run: cargo test --test host_gpu_type

use quanta::GpuType;

// ===========================================================================
// Nested struct (struct containing another gpu_type struct)
// ===========================================================================

#[quanta::gpu_type]
struct Inner {
    x: f32,
    y: f32,
    z: f32,
}

#[quanta::gpu_type]
struct Outer {
    position: [f32; 3],
    scale: f32,
}

#[test]
fn inner_struct_size() {
    assert_eq!(Inner::GPU_SIZE, 12); // 3 * f32
    assert_eq!(Inner::GPU_SIZE, core::mem::size_of::<Inner>());
}

#[test]
fn inner_struct_fields() {
    assert_eq!(Inner::GPU_FIELDS.len(), 3);
    assert_eq!(Inner::GPU_FIELDS[0], ("x", "f32", 0));
    assert_eq!(Inner::GPU_FIELDS[1], ("y", "f32", 4));
    assert_eq!(Inner::GPU_FIELDS[2], ("z", "f32", 8));
}

#[test]
fn outer_struct_size() {
    assert_eq!(Outer::GPU_SIZE, 16); // [f32;3](12) + f32(4)
    assert_eq!(Outer::GPU_SIZE, core::mem::size_of::<Outer>());
}

#[test]
fn outer_struct_fields() {
    assert_eq!(Outer::GPU_FIELDS.len(), 2);
    assert_eq!(Outer::GPU_FIELDS[0], ("position", "[f32; 3]", 0));
    assert_eq!(Outer::GPU_FIELDS[1], ("scale", "f32", 12));
}

// ===========================================================================
// All scalar types as fields
// ===========================================================================

#[quanta::gpu_type]
struct AllScalars {
    f: f32,
    u: u32,
    i: i32,
}

#[test]
fn all_scalars_size() {
    assert_eq!(AllScalars::GPU_SIZE, 12); // 3 * 4 bytes
    assert_eq!(AllScalars::GPU_SIZE, core::mem::size_of::<AllScalars>());
}

#[test]
fn all_scalars_fields() {
    assert_eq!(AllScalars::GPU_FIELDS.len(), 3);
    assert_eq!(AllScalars::GPU_FIELDS[0], ("f", "f32", 0));
    assert_eq!(AllScalars::GPU_FIELDS[1], ("u", "u32", 4));
    assert_eq!(AllScalars::GPU_FIELDS[2], ("i", "i32", 8));
}

#[test]
fn all_scalars_msl() {
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS.contains("float f"));
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS.contains("uint u"));
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS.contains("int i"));
}

#[test]
fn all_scalars_wgsl() {
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS_WGSL.contains("f: f32"));
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS_WGSL.contains("u: u32"));
    assert!(__QUANTA_GPU_TYPE_ALLSCALARS_WGSL.contains("i: i32"));
}

// ===========================================================================
// Array fields: [f32; 3], [u32; 4], [f32; 16]
// ===========================================================================

#[quanta::gpu_type]
struct ArrayFields {
    vec3: [f32; 3],
    ivec4: [u32; 4],
    mat4: [f32; 16],
}

#[test]
fn array_fields_size() {
    // [f32;3](12) + [u32;4](16) + [f32;16](64) = 92
    assert_eq!(ArrayFields::GPU_SIZE, 92);
    assert_eq!(ArrayFields::GPU_SIZE, core::mem::size_of::<ArrayFields>());
}

#[test]
fn array_fields_offsets() {
    assert_eq!(ArrayFields::GPU_FIELDS.len(), 3);
    assert_eq!(ArrayFields::GPU_FIELDS[0], ("vec3", "[f32; 3]", 0));
    assert_eq!(ArrayFields::GPU_FIELDS[1], ("ivec4", "[u32; 4]", 12));
    assert_eq!(ArrayFields::GPU_FIELDS[2], ("mat4", "[f32; 16]", 28));
}

#[test]
fn array_fields_msl() {
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS.contains("float3 vec3"));
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS.contains("uint4 ivec4"));
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS.contains("float4x4 mat4"));
}

#[test]
fn array_fields_wgsl() {
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS_WGSL.contains("vec3: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS_WGSL.contains("ivec4: vec4<u32>"));
    assert!(__QUANTA_GPU_TYPE_ARRAYFIELDS_WGSL.contains("mat4: mat4x4<f32>"));
}

// ===========================================================================
// Alignment verification (offsets match repr(C))
// ===========================================================================

#[quanta::gpu_type]
struct AlignedStruct {
    a: f32,
    b: [f32; 2],
    c: u32,
    d: [f32; 4],
}

#[test]
fn aligned_struct_repr_c_layout() {
    // repr(C) layout: f32(4) + [f32;2](8) + u32(4) + [f32;4](16) = 32
    assert_eq!(
        AlignedStruct::GPU_SIZE,
        core::mem::size_of::<AlignedStruct>()
    );

    // Verify offsets match what repr(C) produces
    assert_eq!(AlignedStruct::GPU_FIELDS[0], ("a", "f32", 0));
    assert_eq!(AlignedStruct::GPU_FIELDS[1], ("b", "[f32; 2]", 4));
    assert_eq!(AlignedStruct::GPU_FIELDS[2], ("c", "u32", 12));
    assert_eq!(AlignedStruct::GPU_FIELDS[3], ("d", "[f32; 4]", 16));

    // Cross-check with Rust's actual layout
    let base = &AlignedStruct {
        a: 0.0,
        b: [0.0; 2],
        c: 0,
        d: [0.0; 4],
    } as *const AlignedStruct as usize;
    let s = AlignedStruct {
        a: 0.0,
        b: [0.0; 2],
        c: 0,
        d: [0.0; 4],
    };
    let a_offset = &s.a as *const f32 as usize - &s as *const AlignedStruct as usize;
    let b_offset = &s.b as *const [f32; 2] as usize - &s as *const AlignedStruct as usize;
    let c_offset = &s.c as *const u32 as usize - &s as *const AlignedStruct as usize;
    let d_offset = &s.d as *const [f32; 4] as usize - &s as *const AlignedStruct as usize;

    assert_eq!(a_offset, 0);
    assert_eq!(b_offset, 4);
    assert_eq!(c_offset, 12);
    assert_eq!(d_offset, 16);
    let _ = base; // suppress warning
}

// ===========================================================================
// GpuType trait implementation correctness
// ===========================================================================

#[test]
fn gpu_type_impl_inner() {
    assert_eq!(<Inner as GpuType>::gpu_size(), 12);
}

#[test]
fn gpu_type_impl_outer() {
    assert_eq!(<Outer as GpuType>::gpu_size(), 16);
}

#[test]
fn gpu_type_impl_array_fields() {
    assert_eq!(<ArrayFields as GpuType>::gpu_size(), 92);
}

// ===========================================================================
// Copy semantics
// ===========================================================================

#[test]
fn gpu_type_structs_are_copy() {
    let inner = Inner {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };
    let _a = inner;
    let _b = inner; // Only compiles if Copy

    let outer = Outer {
        position: [1.0, 2.0, 3.0],
        scale: 1.0,
    };
    let _c = outer;
    let _d = outer;

    let arr = ArrayFields {
        vec3: [1.0, 2.0, 3.0],
        ivec4: [0, 1, 2, 3],
        mat4: [0.0; 16],
    };
    let _e = arr;
    let _f = arr;
}

// ===========================================================================
// Multiple array sizes
// ===========================================================================

#[quanta::gpu_type]
struct Mat3x3 {
    data: [f32; 9],
}

#[quanta::gpu_type]
struct Vec4Wrap {
    data: [f32; 4],
}

#[test]
fn mat3x3_size_and_msl() {
    assert_eq!(Mat3x3::GPU_SIZE, 36); // 9 * 4
    assert!(__QUANTA_GPU_TYPE_MAT3X3.contains("float3x3 data"));
}

#[test]
fn vec4_wrap_size_and_msl() {
    assert_eq!(Vec4Wrap::GPU_SIZE, 16); // 4 * 4
    // [f32; 4] maps to float4 (vector), not float2x2 (matrix)
    assert!(
        __QUANTA_GPU_TYPE_VEC4WRAP.contains("float4 data"),
        "MSL type for [f32; 4]: {}",
        __QUANTA_GPU_TYPE_VEC4WRAP
    );
}

// ===========================================================================
// Single-field struct
// ===========================================================================

#[quanta::gpu_type]
struct SingleField {
    value: u32,
}

#[test]
fn single_field_metadata() {
    assert_eq!(SingleField::GPU_SIZE, 4);
    assert_eq!(SingleField::GPU_FIELDS.len(), 1);
    assert_eq!(SingleField::GPU_FIELDS[0], ("value", "u32", 0));
}

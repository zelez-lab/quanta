//! Compile-time and runtime verification that `#[quanta::gpu_type]` generates
//! correct metadata, trait impls, and shader declarations.
//!
//! Run: cargo test --test gpu_type_test

// === Basic struct ===

#[quanta::gpu_type]
struct Particle {
    pos: [f32; 3],
    vel: [f32; 3],
    mass: f32,
}

// === All scalar types ===

#[quanta::gpu_type]
struct ScalarMix {
    a: f32,
    b: u32,
    c: i32,
}

// === Vector types ===

#[quanta::gpu_type]
struct Vectors {
    pos2: [f32; 2],
    pos3: [f32; 3],
    pos4: [f32; 4],
    idx: [u32; 4],
}

// === Matrix types ===

#[quanta::gpu_type]
struct Matrices {
    model: [f32; 16],
    normal: [f32; 9],
}

// === Single field ===

#[quanta::gpu_type]
struct Scalar {
    value: f32,
}

// === Already has derive(Copy, Clone) ===

#[quanta::gpu_type]
#[derive(Copy, Clone, Debug)]
struct WithExistingDerives {
    x: f32,
    y: f32,
}

#[test]
fn particle_gpu_size() {
    // repr(C): [f32;3](12) + [f32;3](12) + f32(4) = 28, align 4 => 28
    // But GPU_SIZE uses core::mem::size_of which respects repr(C)
    assert_eq!(Particle::GPU_SIZE, core::mem::size_of::<Particle>());
    assert_eq!(Particle::GPU_SIZE, 28);
}

#[test]
fn particle_gpu_fields() {
    assert_eq!(Particle::GPU_FIELDS.len(), 3);
    assert_eq!(Particle::GPU_FIELDS[0], ("pos", "[f32; 3]", 0));
    assert_eq!(Particle::GPU_FIELDS[1], ("vel", "[f32; 3]", 12));
    assert_eq!(Particle::GPU_FIELDS[2], ("mass", "f32", 24));
}

#[test]
fn particle_gpu_type_trait() {
    use quanta::GpuType;
    assert_eq!(<Particle as GpuType>::gpu_size(), 28);
}

#[test]
fn particle_msl_declaration() {
    assert!(__QUANTA_GPU_TYPE_PARTICLE.contains("struct Particle"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE.contains("float3 pos"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE.contains("float3 vel"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE.contains("float mass"));
}

#[test]
fn particle_wgsl_declaration() {
    assert!(__QUANTA_GPU_TYPE_PARTICLE_WGSL.contains("struct Particle"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE_WGSL.contains("pos: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE_WGSL.contains("vel: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_PARTICLE_WGSL.contains("mass: f32"));
}

#[test]
fn scalar_mix_offsets() {
    assert_eq!(ScalarMix::GPU_FIELDS.len(), 3);
    assert_eq!(ScalarMix::GPU_FIELDS[0], ("a", "f32", 0));
    assert_eq!(ScalarMix::GPU_FIELDS[1], ("b", "u32", 4));
    assert_eq!(ScalarMix::GPU_FIELDS[2], ("c", "i32", 8));
}

#[test]
fn vectors_msl() {
    assert!(__QUANTA_GPU_TYPE_VECTORS.contains("float2 pos2"));
    assert!(__QUANTA_GPU_TYPE_VECTORS.contains("float3 pos3"));
    assert!(__QUANTA_GPU_TYPE_VECTORS.contains("float4 pos4"));
    assert!(__QUANTA_GPU_TYPE_VECTORS.contains("uint4 idx"));
}

#[test]
fn vectors_wgsl() {
    assert!(__QUANTA_GPU_TYPE_VECTORS_WGSL.contains("pos2: vec2<f32>"));
    assert!(__QUANTA_GPU_TYPE_VECTORS_WGSL.contains("pos3: vec3<f32>"));
    assert!(__QUANTA_GPU_TYPE_VECTORS_WGSL.contains("pos4: vec4<f32>"));
    assert!(__QUANTA_GPU_TYPE_VECTORS_WGSL.contains("idx: vec4<u32>"));
}

#[test]
fn matrices_msl() {
    assert!(__QUANTA_GPU_TYPE_MATRICES.contains("float4x4 model"));
    assert!(__QUANTA_GPU_TYPE_MATRICES.contains("float3x3 normal"));
}

#[test]
fn matrices_wgsl() {
    assert!(__QUANTA_GPU_TYPE_MATRICES_WGSL.contains("model: mat4x4<f32>"));
    assert!(__QUANTA_GPU_TYPE_MATRICES_WGSL.contains("normal: mat3x3<f32>"));
}

#[test]
fn scalar_struct_size() {
    assert_eq!(Scalar::GPU_SIZE, 4);
    assert_eq!(Scalar::GPU_FIELDS.len(), 1);
    assert_eq!(Scalar::GPU_FIELDS[0], ("value", "f32", 0));
}

#[test]
fn struct_is_copy_clone() {
    // This compiles only if Copy + Clone are derived
    let p = Particle {
        pos: [1.0, 2.0, 3.0],
        vel: [0.0; 3],
        mass: 1.0,
    };
    let _copy = p;
    let _also = p; // only works if Copy
}

#[test]
fn existing_derives_preserved() {
    // Debug should still work since we had #[derive(Copy, Clone, Debug)]
    let w = WithExistingDerives { x: 1.0, y: 2.0 };
    let _s = format!("{:?}", w);
}

# Ray tracing

Hardware-accelerated ray tracing through `AccelerationStructure` (BVH) and
`RayTracingPipeline` (raygen/closest-hit/miss).

## Capability gate

```rust
let gpu = quanta::init()?;
if !gpu.supports_ray_tracing() {
    return Err("RT requires Vulkan with VK_KHR_ray_tracing_pipeline".into());
}
```

## Build a BLAS

A BLAS (bottom-level acceleration structure) is the BVH built over your
geometry.

```rust
use quanta::*;

// 12 vertices = 4 triangles, 3 floats per vertex.
let vertices = gpu.field::<f32>(36)?;
vertices.write(&[
    // triangle 1
    0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.5, 1.0, 0.0,
    // triangle 2
    1.0, 0.0, 0.0,   2.0, 0.0, 0.0,   1.5, 1.0, 0.0,
    // triangle 3
    0.0, 0.0, 1.0,   1.0, 0.0, 1.0,   0.5, 1.0, 1.0,
    // triangle 4
    1.0, 0.0, 1.0,   2.0, 0.0, 1.0,   1.5, 1.0, 1.0,
])?;

let blas = gpu.acceleration_structure_blas(&[GeometryDesc {
    vertices: vertices.handle(),
    indices: None,
    vertex_count: 12,
    index_count: 0,
    vertex_stride: 12, // bytes per vertex (3 × f32)
}])?;
```

The `AccelerationStructure` is `Drop`-safe — backend memory and scratch
buffers are released when the wrapper falls out of scope. Pass multiple
`GeometryDesc` entries for a single BLAS containing several meshes.

## Build a ray-tracing pipeline

```rust
let pipe = gpu.ray_tracing_pipeline(&RayTracingPipelineDesc {
    ray_gen:     &raygen_binary,
    closest_hit: &chit_binary,
    miss:        &miss_binary,
    max_recursion: 2,
})?;
```

`max_recursion` clamps to `MAX_RECURSION_DEPTH` (31). Use the lowest depth
that produces correct results — recursion costs scratch memory.

## Companion shaders

```rust
#[quanta::ray_gen]
fn raygen() {
    // trace a ray for this (x, y) pixel and write color to the output image
}

#[quanta::closest_hit]
fn chit() {
    // shade the surface at the closest hit
}

#[quanta::miss]
fn miss() {
    // background colour for rays that hit nothing
}
```

## Dispatch rays

```rust
// One ray-gen invocation per (x, y) pair.
pipe.dispatch_rays(1920, 1080)?;
```

Width and height each clamp to `MAX_DISPATCH_DIM` (65535).

## Backend notes

| Backend | Status |
|---------|--------|
| Vulkan  | Pipeline create + dispatch live; AS build path gated `NotSupported` pending hardware validation |
| Metal   | Pending intersector tables (Apple family 6+) |
| WebGPU  | `NotSupported` (not in spec) |

## See also

- [Mesh shaders](mesh-shaders.md) — pair with GPU-driven instance culling
- [Expert: Ray tracing](../expert/ray-tracing.md) — per-backend lowering
- [Guide: Ray tracing](../guide/12-ray-tracing.md) — full reference

# Ray tracing

Hardware-accelerated ray tracing on Quanta uses two typed wrappers:

- `AccelerationStructure` — a BVH built over your geometry (BLAS) or over BLAS
  instances (TLAS).
- `RayTracingPipeline` — the ray-gen, closest-hit, and miss shaders combined,
  with a fixed maximum recursion depth.

This chapter is the user-facing introduction. For per-backend lowering details
see [Expert: Ray tracing](../expert/ray-tracing.md).

## Capability gate

Ray tracing is available on Vulkan (with `VK_KHR_acceleration_structure` +
`VK_KHR_ray_tracing_pipeline`) and pending on Metal (Apple family 6+).
WebGPU returns `NotSupported` — the spec doesn't include RT.

```rust
if !gpu.supports_ray_tracing() {
    // fall back to rasterization or compute-based tracing
}
```

## Building a BLAS

```rust
use quanta::*;

let vertices = gpu.field::<f32>(36)?;
vertices.write(&[/* 12 positions × 3 floats */])?;

let blas = gpu.acceleration_structure_blas(&[GeometryDesc {
    vertices: vertices.handle(),
    indices: None,
    vertex_count: 12,
    index_count: 0,
    vertex_stride: 12, // bytes per vertex (3 × f32)
}])?;
```

You may pass multiple `GeometryDesc` entries for a single BLAS containing
several meshes; passing an empty slice returns `InvalidParam`.

The `AccelerationStructure` is `Drop`-safe — its memory and any backend
scratch buffer are freed when it falls out of scope.

## Building a ray tracing pipeline

```rust
let pipe = gpu.ray_tracing_pipeline(&RayTracingPipelineDesc {
    ray_gen:     &raygen_binary,
    closest_hit: &chit_binary,
    miss:        &miss_binary,
    max_recursion: 2,
})?;
```

`max_recursion` clamps to `MAX_RECURSION_DEPTH` (31). The shader binaries come
from the matching attribute macros:

```rust
#[quanta::ray_gen]    fn raygen() { /* trace_ray(...) */ }
#[quanta::closest_hit] fn chit()  { /* shade hit */ }
#[quanta::miss]        fn miss()  { /* shade miss */ }
```

## Dispatching rays

```rust
pipe.dispatch_rays(1920, 1080)?;
```

Width and height are clamped to `MAX_DISPATCH_DIM` (65535) per axis. One
ray-gen invocation runs per `(x, y)` pair.

## Backend status (v0.1)

| Backend | Status                                                         |
|---------|----------------------------------------------------------------|
| Vulkan  | Build path (`vkCmdBuildAccelerationStructuresKHR`) gated `NotSupported` pending hardware validation; pipeline create + dispatch live |
| Metal   | Pending intersector tables (Apple family 6+)                   |
| WebGPU  | `NotSupported` (not in the spec)                               |
| CPU     | Software lifecycle only                                        |

## Constants

| Constant                | Value | Meaning                              |
|-------------------------|-------|--------------------------------------|
| `MAX_RECURSION_DEPTH`   | 31    | Largest `max_recursion` accepted     |
| `MAX_DISPATCH_DIM`      | 65535 | Per-axis limit on `dispatch_rays`    |

## Next

- [Expert: Ray tracing](../expert/ray-tracing.md) -- per-backend lowering
- [Mesh shaders](11-mesh-shaders.md) -- pair with GPU-driven instance culling

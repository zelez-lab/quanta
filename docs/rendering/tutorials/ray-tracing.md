# Ray tracing

Hardware-accelerated ray tracing on Quanta uses two typed wrappers:

- `AccelerationStructure` — a BVH built over your geometry (BLAS) or over BLAS
  instances (TLAS).
- `RayTracingPipeline` — the ray-gen, closest-hit, and miss shaders combined,
  with a fixed maximum recursion depth.

This chapter is the user-facing introduction. For per-backend lowering details
see [Expert: Ray tracing](../../expert/ray-tracing.md).

## Capability gate

Ray tracing is **live on Metal** (Apple GPU family 6+ — every M-series
chip; compute-based intersectors, so no RT silicon is required, and M3+
hardware accelerates them transparently). On Vulkan the
acceleration-structure foundation is in place behind
`VK_KHR_acceleration_structure` + `VK_KHR_ray_tracing_pipeline`, with
the build execution gated `NotSupported` pending real RT hardware.
WebGPU returns `NotSupported` — the spec doesn't include RT.

```rust
if !gpu.supports_ray_tracing() {
    // fall back to rasterization or compute-based tracing
}
```

## Building a BLAS

```rust
use quanta::*; // brings the RenderGpu extension trait into scope

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
    ray_gen:     RAY_GEN.as_bytes(),  // native MSL on Metal (below)
    closest_hit: &[],
    miss:        &[],
    max_recursion: 2,
})?;
```

`max_recursion` clamps to `MAX_RECURSION_DEPTH` (31). On Metal —
the backend where the full path runs today — `ray_gen` is **native MSL
source**: an intersector compute kernel following the MVP ABI (the
acceleration structure at `buffer(0)`, a `Field<f32>` output at
`buffer(1)`, one thread per ray):

```rust
const RAY_GEN: &str = r#"
#include <metal_stdlib>
using namespace metal;
using namespace metal::raytracing;
kernel void trace(primitive_acceleration_structure accel [[buffer(0)]],
                  device float* out [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
  ray r;
  r.origin = float3(0.25, 0.25, 0.0);
  r.direction = float3(0.0, 0.0, 1.0);
  r.min_distance = 0.0;
  r.max_distance = 100.0;
  intersector<triangle_data> isect;
  intersection_result<triangle_data> res = isect.intersect(r, accel);
  out[tid] = (res.type == intersection_type::triangle) ? res.distance : -1.0;
}
"#;
```

(The `#[quanta::ray_gen]` / `#[quanta::closest_hit]` / `#[quanta::miss]`
proc-macros exist as stage-tagged stubs; portable RT shader authoring
through them is the planned route once the IR grows the RT stages.)

## Dispatching rays

```rust
let out = gpu.field::<f32>(width * height)?;
pipe.dispatch_rays(&blas, &out, width, height)?;
```

The dispatch binds the acceleration structure and the output field,
then runs one ray-gen invocation per `(x, y)` pair. Width and height
are clamped to `MAX_DISPATCH_DIM` (65535) per axis. A triangle in the
`z = 0.5` plane hit by a `+z` ray from `z = 0` reads back exactly
`t = 0.5` — the shape `tests/gpu_ray_tracing.rs` pins end to end.

## Backend status (v0.1)

| Backend | Status                                                         |
|---------|----------------------------------------------------------------|
| Metal   | ✅ Real: `MTLAccelerationStructure` build + intersector compute pipeline + `dispatch_rays` (Apple family 6+, compute-based — no RT hardware needed) |
| Vulkan  | AS create/storage/destroy native; build execution gated `NotSupported` pending real RT hardware; dispatch pending shader-binding-table work |
| WebGPU  | `NotSupported` (not in the spec)                               |
| CPU     | Lifecycle tier — the dispatch is recorded, the output untouched |

## Constants

| Constant                | Value | Meaning                              |
|-------------------------|-------|--------------------------------------|
| `MAX_RECURSION_DEPTH`   | 31    | Largest `max_recursion` accepted     |
| `MAX_DISPATCH_DIM`      | 65535 | Per-axis limit on `dispatch_rays`    |

## Next

- [Expert: Ray tracing](../../expert/ray-tracing.md) -- per-backend lowering
- [Mesh shaders](mesh-shaders.md) -- pair with GPU-driven instance culling

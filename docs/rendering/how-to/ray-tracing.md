# Ray tracing

Hardware-accelerated ray tracing through `AccelerationStructure` (BVH) and
`RayTracingPipeline` (raygen/closest-hit/miss).

## Capability gate

```rust
let gpu = quanta::init()?;
if !gpu.supports_ray_tracing() {
    return Err("no RT on this device".into());
}
```

True on Metal for Apple GPU family 6+ (every M-series chip — the path
is compute-based, no RT silicon required) and on Vulkan devices
exposing `VK_KHR_ray_tracing_pipeline`.

## Build a BLAS

A BLAS (bottom-level acceleration structure) is the BVH built over your
geometry.

```rust
use quanta::*; // brings the RenderGpu extension trait into scope

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
    ray_gen:     RAY_GEN.as_bytes(),  // native MSL on Metal (below)
    closest_hit: &[],
    miss:        &[],
    max_recursion: 2,
})?;
```

`max_recursion` clamps to `MAX_RECURSION_DEPTH` (31). Use the lowest depth
that produces correct results — recursion costs scratch memory.

## Companion shaders

On Metal, `ray_gen` is native MSL: an intersector compute kernel with
the acceleration structure at `buffer(0)`, a `device float*` output at
`buffer(1)`, one thread per ray — see the
[ray-tracing tutorial](../tutorials/ray-tracing.md) for a complete
kernel. The `#[quanta::ray_gen]` / `#[quanta::closest_hit]` /
`#[quanta::miss]` proc-macros are stage-tagged stubs today; portable
RT shader authoring lands with the IR's RT stages.

## Dispatch rays

```rust
let out = gpu.field::<f32>(w * h)?;
// One ray-gen invocation per (x, y) pair; hit distances land in `out`.
pipe.dispatch_rays(&blas, &out, w, h)?;
```

Width and height each clamp to `MAX_DISPATCH_DIM` (65535).

## Backend notes

| Backend | Status |
|---------|--------|
| Metal   | ✅ Real: `MTLAccelerationStructure` build + intersector compute pipeline + `dispatch_rays` (Apple family 6+, compute-based) |
| Vulkan  | AS create/storage/destroy native; build execution + dispatch return `NotSupported` pending shader-binding-table work + real RT hardware |
| WebGPU  | `NotSupported` (not in spec) |
| CPU     | Lifecycle tier — dispatch recorded, output untouched |

## See also

- [Mesh shaders](mesh-shaders.md) — pair with GPU-driven instance culling
- [Expert: Ray tracing](../../expert/ray-tracing.md) — per-backend lowering
- [Guide: Ray tracing](../../rendering/tutorials/ray-tracing.md) — full reference

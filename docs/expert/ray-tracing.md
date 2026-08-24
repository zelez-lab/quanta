# Expert: Ray Tracing

Hardware-accelerated ray tracing via Quanta. Requires GPU support
(NVIDIA RTX, AMD RDNA 2+, Apple GPU family 6+ with Metal). The current
v0.1 surface ships **the full Metal path** — real
`MTLAccelerationStructure` builds plus an intersector compute pipeline
behind `dispatch_rays` (compute-based, so every M-series chip
qualifies; M3+ accelerates the intersector in hardware) — and the
**acceleration-structure foundation** on Vulkan; the Vulkan
ray-tracing pipeline + shader binding tables + `vkCmdTraceRaysKHR`
dispatch require IR-side work and are deferred.

## What ships in v0.1

| Surface | State |
|---------|-------|
| `gpu.acceleration_structure_blas(&[GeometryDesc])` typed wrapper (a `RenderGpu` extension method) | ✅ exists |
| Vulkan: extension detect + enable for `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure` + `VK_KHR_buffer_device_address` + `VK_KHR_deferred_host_operations` | ✅ at device discovery |
| Vulkan: proc-addr resolution for `vkCreateAccelerationStructureKHR`, `vkDestroyAccelerationStructureKHR`, `vkGetAccelerationStructureBuildSizesKHR`, `vkCmdBuildAccelerationStructuresKHR`, `vkCmdTraceRaysKHR`, `vkGetBufferDeviceAddress` | ✅ cached on `VulkanDevice` |
| Vulkan: `bufferDeviceAddress` + `accelerationStructure` features chained at `vkCreateDevice` | ✅ slice 18 + 23 |
| Vulkan: `field_alloc_impl` adds `SHADER_DEVICE_ADDRESS_BIT` + `AS_BUILD_INPUT_READ_ONLY_BIT_KHR` when buffer device address is enabled | ✅ |
| Vulkan: AS-storage + scratch buffer allocation + `vkCreateAccelerationStructureKHR` | ✅ validator-clean |
| Vulkan: actual `vkCmdBuildAccelerationStructuresKHR` execution | ⚠️ Returns `NotSupported` — validator-clean inputs but lavapipe segfaults inside `vkQueueWaitIdle`. Pending real RT hardware (AMDGPU runner) to confirm whether this is a Mesa lavapipe bug or a deeper driver issue. |
| Metal: hardware-feature gate via `[device supportsFamily:Apple6]` | ✅ |
| Metal: acceleration-structure builds via `MTLAccelerationStructure` | ✅ Real — sizes via `accelerationStructureSizesWithDescriptor:`, triangle geometry from the vertex `Field` (indexed or not), built on a command buffer and waited. |
| Metal: `dispatch_rays` | ✅ Real — the `ray_gen` MSL source compiles to an intersector compute pipeline; the dispatch binds the AS at `buffer(0)` and the `Field<f32>` output at `buffer(1)`, one thread per ray. `tests/gpu_ray_tracing.rs` pins the exact `t = 0.5` hit end to end. |
| Portable ray-tracing pipelines + shader binding tables | ❌ Needs new IR shader stages (raygen / closest-hit / miss / any-hit / intersection). Metal runs on native MSL ray-gen source today; Vulkan waits on this work. |
| WebGPU | `NotSupported("ray tracing is not in the WebGPU spec")`. |

## Available API today

```rust
use quanta::{GeometryDesc, Gpu, RenderGpu};

let gpu: Gpu = quanta::init()?;

// Capability check.
if !gpu.supports_ray_tracing() {
    // No RT extensions / proc addresses on this device.
    return Ok(());
}

// Build a BLAS from one geometry. Vertex Field must be created
// via gpu.field<f32>(...) — Quanta's Vulkan driver augments the
// allocation with SHADER_DEVICE_ADDRESS_BIT when bufferDeviceAddress
// is enabled, so the AS build can reference it by GPU device address.
let verts = gpu.field::<f32>(9)?;          // 3 verts x 3 floats
verts.write(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])?;

let blas = gpu.acceleration_structure_blas(&[GeometryDesc {
    vertices: verts.handle(),
    indices: None,
    vertex_count: 3,
    index_count: 0,
    vertex_stride: 12,
}])?;

// Metal: compile the MSL ray-gen intersector and fire the rays.
let pipe = gpu.ray_tracing_pipeline(&RayTracingPipelineDesc {
    ray_gen: RAY_GEN_MSL.as_bytes(),
    closest_hit: &[],
    miss: &[],
    max_recursion: 1,
})?;
let out = gpu.field::<f32>(1)?;
pipe.dispatch_rays(&blas, &out, 1, 1)?;   // out[0] = hit distance
```

The returned `AccelerationStructure` is `Drop`-safe — dropping it calls
the backend destroy exactly once (guarded by a `live` flag), tearing down
the AS handle + storage buffer + storage memory.

## Capability matrix at v0.1

| Backend | Builds | Dispatch | Notes |
|---------|--------|----------|-------|
| Metal | ✅ native `MTLAccelerationStructure` build | ✅ intersector compute pipeline | Compute-based on Apple family 6+ (no RT silicon required; M3+ accelerates in hardware). `ray_gen` is native MSL source under the MVP ABI (AS at `buffer(0)`, output at `buffer(1)`). |
| Vulkan + RADV / NVIDIA | ⚠️ AS create + storage + destroy native; build call gated as NotSupported | ❌ pending IR work | The proc-addr foundation is loaded; the build sequence follows once we have hardware to validate against. |
| Vulkan + lavapipe | ⚠️ Same as above | ❌ | Lavapipe technically supports `VK_KHR_ray_tracing_pipeline` but its build execution path is the source of the segfault that gates slice 23. |
| CPU | lifecycle tier | lifecycle tier | Wrappers construct and destroy; the dispatch is recorded, the output untouched. |
| WebGPU | n/a | n/a | Not in spec. |

## What comes next

- Vulkan: real `vkCmdBuildAccelerationStructuresKHR` execution
  (validator-clean inputs already in place; needs hardware
  validation to ship).
- Vulkan: full ray-tracing pipeline + shader binding tables +
  `vkCmdTraceRaysKHR`.
- Portable RT shader authoring: raygen / closest-hit / miss / any-hit /
  intersection shader stages in `quanta-ir` and matching MSL / SPIR-V
  emitters. The `#[quanta::ray_gen]`, `#[quanta::closest_hit]`,
  `#[quanta::miss]` proc-macros exist (in `quanta-render-dsl`,
  re-exported by `quanta-render`) and the `ShaderStage` enum carries
  the variants, but no backend emitter consumes them yet — Metal runs
  on native MSL ray-gen source until then.

## How to validate against real hardware

The blocker for the build dispatch is hardware availability, not code:

1. Register the AMDGPU self-hosted runner per
   [`docs/internals/testing.md`](../internals/testing.md). One Linux
   box with an RX 6000+ tagged `[self-hosted, linux, gpu-amd]`.
2. Trigger the `diff-amdgpu` workflow with the `run-amd-diff` PR
   label. RADV's RT build path is widely production-tested;
   if it works there, the slice 23 build call's `NotSupported` gate
   can be flipped to `Ok(handle)` with high confidence.
3. As an alternative, paid NVIDIA cloud GPU runners would also
   validate the build path.

Until either hardware path is registered, the build call's
`NotSupported` is the honest behavior — the foundation work
(handle creation, scratch + storage, command submission, destroy)
is validated; only the GPU-side build execution is unverified.

# Mesh shaders

Replace the fixed vertex/index pipeline with two compute-style stages that
emit *meshlets* — small clusters of geometry — on the fly. Use mesh shaders
for GPU-driven culling, procedural geometry, and pipelines that don't fit
the vertex-attribute model.

## Capability gate

```rust
let gpu = quanta::init()?;
if !gpu.supports_mesh_shaders() {
    return Err("mesh shaders require Metal 3+ or VK_EXT_mesh_shader".into());
}
```

## Pipeline

```rust
use quanta::*;

// 64-vertex / 124-primitive meshlets, 32 task threads per group.
let pipe = gpu.mesh_pipeline(MeshPipelineDesc {
    max_vertices_per_meshlet: 64,
    max_primitives_per_meshlet: 124,
    task_threads_per_group: 32,
})?;
```

`MeshPipelineDesc::default()` gives `64 / 124 / 1` (mesh-only, no task
stage). Each field range-checks at construction:

| Field | Range |
|---|---|
| `max_vertices_per_meshlet` | `1..=256` |
| `max_primitives_per_meshlet` | `1..=256` |
| `task_threads_per_group` | `1..=128` |

## Dispatch

```rust
// Launch [groups_x, groups_y, groups_z] task workgroups.
pipe.dispatch([1024, 1, 1])?;
```

Each task workgroup runs your `#[quanta::task]` shader, which amplifies
into mesh-shader invocations that emit the meshlet vertices and primitives.
Each axis clamps to `MAX_GROUP_COUNT` (65535).

## Companion shaders

```rust
#[quanta::task]
fn task_shader() {
    // optional amplification stage; emits per-meshlet visibility
}

#[quanta::mesh]
fn mesh_shader() {
    // emit up to max_vertices / max_primitives for this meshlet
}
```

Use `#[quanta::task]` only if you set `task_threads_per_group > 1`. With
the default (`1`), the pipeline runs mesh-only.

## GPU-driven culling sketch

The task stage is the natural place for per-meshlet culling: frustum,
occlusion, backface. Each task thread inspects one meshlet's bounding
sphere and emits 0 or 1 mesh invocations.

```rust
#[quanta::task]
fn cull_meshlets() {
    // Read meshlet bounds from a storage buffer indexed by group_id.
    // If visible, emit mesh shader work; otherwise return.
}
```

This is the GPU-side equivalent of the indirect-draw culling pattern,
without the indirection.

## Backend notes

| Backend | Path |
|---------|------|
| Metal   | `MTLMeshRenderPipelineDescriptor` + `drawMeshThreadgroups:` (Metal 3+) |
| Vulkan  | `VK_EXT_mesh_shader` + `vkCmdDrawMeshTasksEXT` |
| WebGPU  | `NotSupported` (not in spec) |

## See also

- [Indirect commands](indirect-commands.md) — pair with GPU-driven culling
- [Guide: Mesh shaders](../../rendering/tutorials/mesh-shaders.md) — full reference

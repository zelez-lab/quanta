# Mesh shaders

Mesh shaders replace the fixed vertex/index pipeline with two compute-style
stages that emit *meshlets* — small (≤ 256-vertex) clusters of geometry — on
the fly. Use them for GPU-driven culling, procedural geometry, and pipelines
that don't fit the vertex-attribute model.

## Capability gate

Available on Metal 3+ (Apple) and Vulkan with `VK_EXT_mesh_shader`. WebGPU
returns `NotSupported`.

```rust
if !gpu.supports_mesh_shaders() {
    // fall back to vertex/fragment + indirect draws
}
```

## Creating a mesh pipeline

```rust
use quanta::*; // brings the RenderGpu extension trait into scope

let pipe = gpu.mesh_pipeline(MeshPipelineDesc {
    max_vertices_per_meshlet: 64,
    max_primitives_per_meshlet: 124,
    task_threads_per_group: 32,
})?;
```

`MeshPipelineDesc::default()` gives you `64 / 124 / 1` (mesh-only, no task
stage). Each field is range-checked at construction time:

| Field                        | Range                    | Returns on OOB    |
|------------------------------|--------------------------|-------------------|
| `max_vertices_per_meshlet`   | `1..=MAX_MESH_VERTICES` (256)   | `InvalidParam` |
| `max_primitives_per_meshlet` | `1..=MAX_MESH_PRIMITIVES` (256) | `InvalidParam` |
| `task_threads_per_group`     | `1..=MAX_TASK_THREADS` (128)    | `InvalidParam` |

## Dispatching

```rust
// Launch [groups_x, groups_y, groups_z] task workgroups.
pipe.dispatch([1024, 1, 1])?;
```

Each component clamps to `MAX_GROUP_COUNT` (65535). The semantics match
`vkCmdDrawMeshTasksEXT` / `dispatchThreadgroups:` — each group runs your task
shader, which then amplifies into mesh-shader invocations that emit the
meshlet vertices and primitives.

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

Use `#[quanta::task]` only if you set `task_threads_per_group > 1`. With the
default (`1`), the pipeline runs mesh-only.

## Backend matrix

| Backend | Status                                                |
|---------|-------------------------------------------------------|
| Metal   | `MTLMeshRenderPipelineDescriptor` + `drawMeshThreadgroups:` (Metal 3+) |
| Vulkan  | `VK_EXT_mesh_shader` + `vkCmdDrawMeshTasksEXT`        |
| WebGPU  | `NotSupported`                                        |
| CPU     | Software lifecycle for testing only                   |

## Constants

| Constant                | Value  | Meaning                              |
|-------------------------|--------|--------------------------------------|
| `MAX_MESH_VERTICES`     | 256    | Cap on `max_vertices_per_meshlet`    |
| `MAX_MESH_PRIMITIVES`   | 256    | Cap on `max_primitives_per_meshlet`  |
| `MAX_TASK_THREADS`      | 128    | Cap on `task_threads_per_group`      |
| `MAX_GROUP_COUNT`       | 65535  | Per-axis limit on `dispatch`         |

## Next

- [Tessellation](tessellation.md) -- the older subdivision pipeline
- [Indirect commands](indirect-commands.md) -- pair with GPU-driven culling

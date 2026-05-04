# Tessellation

Subdivide patches on the GPU for smooth curved surfaces, displacement-mapped
terrain, or adaptive LOD without paying the upload cost of a fully-tessellated
mesh.

## Capability gate

```rust
let gpu = quanta::init()?;
if !gpu.supports_tessellation() {
    return Err("tessellation requires Metal family 4+ or Vulkan".into());
}
```

## Pipeline + factors

```rust
use quanta::*;

// Triangle patches with 3 control points each.
let pipe = gpu.tessellation_pipeline(TessTopology::Triangle, 3)?;

// Subdivide every edge into 8 segments and the interior into 8 layers.
for edge in 0..3 {
    pipe.set_outer(edge, 8)?;
}
pipe.set_inner(0, 8)?;
```

`set_outer(i, factor)` and `set_inner(i, factor)` clamp to
`[1, MAX_TESS_LEVEL]` (64) and return `InvalidParam` for out-of-range
indices. Triangle patches use 3 outer + 1 inner; quad patches use 4 outer
+ 2 inner.

## Companion shaders

```rust
#[quanta::tess_control]
fn tcs() {
    // emit per-edge / per-patch tessellation factors
}

#[quanta::tess_eval]
fn tes() -> Vec4 {
    // sample the surface at the generated point and return clip-space position
}
```

The `TessellationPipeline` slots between your vertex stage (which emits
patches) and your fragment stage.

## Adaptive LOD

Drive `set_outer` from camera distance instead of a constant:

```rust
let factor = (32.0 / camera_distance).clamp(1.0, 64.0) as u32;
for edge in 0..3 {
    pipe.set_outer(edge, factor)?;
}
pipe.set_inner(0, factor)?;
```

Far patches collapse to flat triangles; near patches subdivide finely.

## Backend notes

| Backend | Path |
|---------|------|
| Metal   | Native TCS + TES on Apple family 4+ |
| Vulkan  | `tessellationShader` core feature |
| WebGPU  | `NotSupported` (not in spec) |

## See also

- [Mesh shaders](../guide/11-mesh-shaders.md) — the modern alternative
- [Guide: Tessellation](../guide/10-tessellation.md) — full reference

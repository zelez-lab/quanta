# Tessellation

Tessellation subdivides patches (triangles or quads) into a denser mesh on the
GPU. Use it for smooth curved surfaces, displacement-mapped terrain, and
adaptive level-of-detail without paying the upload cost of a fully-tessellated
mesh.

## Capability gate

Tessellation is implemented on Metal (family 4+) and Vulkan; WebGPU returns
`NotSupported`. Always query first:

```rust
let gpu = quanta::init()?;
if !gpu.supports_tessellation() {
    // fall back to a pre-tessellated mesh
}
```

## Creating a tessellation pipeline

```rust
use quanta::*;

let pipe = gpu.tessellation_pipeline(TessTopology::Triangle, 3)?;
```

The two parameters are:

| Parameter        | Range            | Meaning                                       |
|------------------|------------------|-----------------------------------------------|
| `topology`       | `Triangle`/`Quad`| What each patch becomes after subdivision     |
| `control_points` | `1..=MAX_PATCH_SIZE` (32) | Vertices per input patch             |

`TessTopology::Triangle` patches use 3 outer + 1 inner factor;
`TessTopology::Quad` patches use 4 outer + 2 inner factors.

## Setting tessellation factors

Outer factors control edge subdivision; inner factors control interior
subdivision. Both clamp to `[1, MAX_TESS_LEVEL]` (64).

```rust
// Subdivide every triangle edge into 8 segments, interior into 8 layers.
for edge in 0..3 {
    pipe.set_outer(edge, 8)?;
}
pipe.set_inner(0, 8)?;
```

Each call returns `QuantaErrorKind::InvalidParam` if the index is out of range
for the topology, or if the factor exceeds `MAX_TESS_LEVEL`. Index counts:

| Topology | Outer count | Inner count |
|----------|-------------|-------------|
| Triangle | 3           | 1           |
| Quad     | 4           | 2           |

## Companion shaders

Two new shader stages produce and consume tessellated patches:

```rust
#[quanta::tess_control]
fn tcs(/* per-patch inputs */) {
    // emit per-edge / per-patch tessellation factors
}

#[quanta::tess_eval]
fn tes(/* barycentric or u/v coords */) -> Vec4 {
    // sample the surface at the generated point and return clip-space position
}
```

The pipeline you build with `gpu.tessellation_pipeline(...)` slots in between
your vertex stage (which emits patches) and your fragment stage. See
[Vertex and fragment shaders: Other stages](vertex-fragment.md#other-shader-stages).

## Backend matrix

| Backend | Status                                                  |
|---------|---------------------------------------------------------|
| Metal   | Native TCS + TES on family 4+; software emulation older |
| Vulkan  | Native (`tessellationShader` is core)                   |
| WebGPU  | `NotSupported`                                          |
| CPU     | Software lifecycle for testing only                     |

## Constants

| Constant            | Value | Meaning                       |
|---------------------|-------|-------------------------------|
| `MAX_TESS_LEVEL`    | 64    | Largest outer/inner factor    |
| `MAX_PATCH_SIZE`    | 32    | Max control points per patch  |

## Next

- [Mesh shaders](mesh-shaders.md) -- the modern alternative
- [Rendering](rendering.md) -- the surrounding pipeline

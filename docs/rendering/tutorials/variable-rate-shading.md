# Variable Rate Shading

Variable Rate Shading (VRS) lets the rasterizer shade more than one pixel per
fragment-shader invocation in regions of the frame where you don't need
pixel-perfect detail — peripheral vision in VR, far-field pixels behind motion
blur, regions that will be downsampled anyway. The classic case is to shade
the center of the screen at `1×1` and the edges at `2×2` or `4×4`.

## Capability gate

VRS is available on Vulkan with `VK_KHR_fragment_shading_rate` and on Apple
Silicon via `MTLRasterizationRateMap`. WebGPU returns `NotSupported`.

```rust
if !gpu.supports_vrs() {
    // skip VRS configuration; render at full rate
}
let rates: Vec<(u32, u32)> = gpu.supported_shading_rates();
// e.g. [(1, 1), (1, 2), (2, 1), (2, 2)] on a typical desktop GPU
```

`supported_shading_rates()` returns an empty `Vec` when VRS is unavailable,
so you can use it directly as the source of truth.

## Shading rates

The seven rates Quanta exposes:

| Variant            | Pixels per fragment | Code |
|--------------------|---------------------|------|
| `ShadingRate::R1x1` | 1 (full rate)      | 0    |
| `ShadingRate::R1x2` | 2 (vertical)       | 1    |
| `ShadingRate::R2x1` | 2 (horizontal)     | 2    |
| `ShadingRate::R2x2` | 4                  | 3    |
| `ShadingRate::R2x4` | 8                  | 4    |
| `ShadingRate::R4x2` | 8                  | 5    |
| `ShadingRate::R4x4` | 16                 | 6    |

Not every backend supports the full set — check `supported_shading_rates()`.

## Pipeline-level rate

Create a `VrsState` once and update it as you traverse the scene:

```rust
use quanta::*; // brings the RenderGpu extension trait into scope

let mut vrs = gpu.vrs_state()?;  // starts at R1x1
vrs.set_rate(ShadingRate::R2x2)?;
println!("now shading at {:?}", vrs.current());
```

`set_rate` returns `NotSupported` if the rate isn't in the list above.

## Per-draw rate inside a render pass

The render builder exposes two ways to set the shading rate during a pass:

```rust
gpu.render(&target)?
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .shading_rate(ShadingRate::R2x2) // pipeline-rate before this draw
    .draw(3)
    .shading_rate(ShadingRate::R1x1)
    .draw(3)
    .pulse()?
    .wait()?;
```

For per-tile rates (e.g. lower rate at the edges of a foveated render), upload
a one-byte-per-tile texture and bind it:

```rust
// in the builder chain:
.shading_rate_image(&rate_texture)
```

The texture format and tile size are backend-specific — see the per-backend
VRS notes in [Internals: Drivers](../../internals/drivers.md).

## Backend matrix

| Backend | Status                                                  |
|---------|---------------------------------------------------------|
| Vulkan  | `VK_KHR_fragment_shading_rate` + `vkCmdSetFragmentShadingRateKHR` |
| Metal   | `MTLRasterizationRateMap` (Apple Silicon)               |
| WebGPU  | `NotSupported`                                          |
| CPU     | Software lifecycle for testing only                     |

## Next

- [Indirect commands](indirect-commands.md) -- combine with GPU culling
- [Rendering](rendering.md) -- the surrounding pipeline

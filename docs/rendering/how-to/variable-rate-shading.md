# Variable rate shading

VRS lets the rasterizer shade more than one pixel per fragment-shader
invocation in regions where you don't need pixel-perfect detail —
peripheral vision in VR, far-field pixels behind motion blur, regions
that will be downsampled anyway.

## Capability gate

```rust
let gpu = quanta::init()?;
if !gpu.supports_vrs() {
    // not available — render at full rate
}
let rates: Vec<(u32, u32)> = gpu.supported_shading_rates();
// e.g. [(1, 1), (1, 2), (2, 1), (2, 2)] on a typical desktop GPU
```

`supported_shading_rates()` returns an empty `Vec` when VRS is
unavailable, so it's safe to use as the source of truth.

## Pipeline-level rate

```rust
use quanta::*;

let mut vrs = gpu.vrs_state()?;       // starts at R1x1
vrs.set_rate(ShadingRate::R2x2)?;     // 4 pixels per shaded fragment
println!("now shading at {:?}", vrs.current());
```

`set_rate` returns `NotSupported` if the rate isn't in the device's
supported list.

The seven rates Quanta exposes:

| Variant | Pixels per fragment |
|---|---|
| `R1x1` | 1 (full rate) |
| `R1x2`, `R2x1` | 2 |
| `R2x2` | 4 |
| `R2x4`, `R4x2` | 8 |
| `R4x4` | 16 |

## Per-draw rate inside a render pass

```rust
gpu.render(&target)?
    .clear(Color::BLACK)
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .set_shading_rate(ShadingRate::R1x1) // full rate for the foreground
    .draw(3)
    .set_shading_rate(ShadingRate::R2x2) // half rate for the background
    .draw(3)
    .pulse()?
    .wait()?;
```

## Foveated rendering with a rate image

For per-tile rates (e.g. lower rate at the edges of a foveated render),
upload a one-byte-per-tile texture and bind it:

```rust
pass.set_shading_rate_image(&rate_texture);
```

The texture format and tile size are backend-specific — see
[Internals: Drivers](../../internals/drivers.md).

## Backend notes

| Backend | Path |
|---------|------|
| Vulkan  | `VK_KHR_fragment_shading_rate` + `vkCmdSetFragmentShadingRateKHR` |
| Metal   | `MTLRasterizationRateMap` (Apple Silicon) |
| WebGPU  | `NotSupported` (not in spec) |

## See also

- [Indirect commands](indirect-commands.md) — combine with GPU culling
- [Guide: Variable Rate Shading](../../rendering/tutorials/variable-rate-shading.md) — full reference

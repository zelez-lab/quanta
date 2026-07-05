# quanta-core

The **shared GPU substrate of Quanta** — the device line every Quanta face is
built on: the sealed `GpuDevice` trait, the in-tree drivers (Metal / Vulkan /
CPU software / WebGPU, all raw FFI, zero external dependencies), and the
resource surface they speak — the `Gpu` handle, fields, textures, samplers,
pulses, timelines, queries, errors, and capability queries.

**Consumers do not depend on this crate directly.** Reach it through one of
the two faces:

- the [`quanta`](../..) facade re-exports the whole surface and adds the
  compute face (`#[quanta::kernel]`, `Wave` dispatch, the scan library)
  behind its `compute` feature;
- [`quanta-render`](../quanta-render) builds on this crate's `render` feature
  and adds the render face (the `RenderGpu` extension trait, `RenderBuilder`,
  typed render wrappers, `Surface` presentation).

## What lives here (and why)

The compute/render boundary cuts *through* the driver line: the `GpuDevice`
trait itself speaks both data models, and each driver implements both halves.
So the trait, the drivers, and both data models live here, each half behind a
feature:

- **always on** — device discovery (`init` / `devices` / `init_cpu`), `Gpu`,
  `Field<T>`, `Texture` / `TextureView` / `Sampler` (with
  `Texture::native_handle()` zero-copy export), `Pulse`, timelines,
  timestamp/format/capability queries, `QuantaError`.
- **`render`** — the render data model the trait and drivers speak:
  `PipelineDesc` / `ShaderSource`, `RenderPass`, `ColorTarget` /
  `DepthTarget`, shader binaries, surface configuration (`SurfaceConfig`,
  `SurfaceTarget`, `PresentMode`), ray-tracing / VRS / mesh descriptors.
  `quanta-render` builds the typed user surface on top.
- **`compute`** — the compute data model: `Wave`, `Batch`, queues, compute
  indirect command buffers.

Resource wrappers own their driver handle: dropping a `Texture`, `Pipeline`,
`Sampler`, or `TextureView` releases the underlying driver resource exactly
once (`live`-flag guarded).

## Features

Mirrors the facade's feature axes; the facade and `quanta-render` both depend
on this crate with `default-features = false` and forward explicitly. The
defaults here (`metal`, `render`, `compute`) only serve a bare
`cargo build -p quanta-core`.

| Feature | What it gates |
|---|---|
| `metal` / `vulkan` / `webgpu` | Backend drivers |
| `software` | CPU executor (implies `compute` + `jit`) |
| `render` | Render half of the device trait + drivers + data model |
| `compute` | Compute half of the device trait + drivers + data model |
| `jit` | Runtime shader emission from embedded KernelDef IR |

## Verification

The API modules and drivers in this crate are covered by the Lean/Verus
verification chain (API invariants, resource lifecycles, memory model) — see
[`docs/verification`](../../docs/verification/index.md) and
[`specs/THEOREMS.md`](../../specs/THEOREMS.md).

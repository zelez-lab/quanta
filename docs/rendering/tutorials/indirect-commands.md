# Indirect commands

Indirect command buffers (ICBs) record dispatches and draws once on the host,
then replay them — possibly with arguments the GPU itself wrote. They are the
substrate for GPU-driven culling, multi-draw instancing, and frame-graph
replay.

Quanta exposes two typed wrappers:

- `IndirectCommandBuffer` — a list of dispatches and/or non-indexed draws,
  recorded host-side, executed via a single submit.
- `IndirectRenderBundle` — a list of draws recorded once, replayable inside
  any compatible render pass (the WebGPU/D3D12 "render bundle" pattern).

Indirect *draws* — draws whose arguments live in a GPU buffer — are exposed
on the render builder directly (`.draw_indirect`, `.draw_indexed_indirect`).

## IndirectCommandBuffer (compute)

```rust
use quanta::*;

let mut icb = gpu.indirect_command_buffer(64)?; // up to 64 commands
let mut wave = my_kernel(&gpu)?;
wave.bind(0, &input);

icb.record_dispatch(&wave, [256, 1, 1])?;
icb.record_dispatch(&wave, [128, 1, 1])?;

icb.execute_all()?; // or icb.execute(1) to run only the first command
```

| Method                          | Effect                                       |
|---------------------------------|----------------------------------------------|
| `record_dispatch(&wave, groups)`| Append a compute dispatch                    |
| `record_draw(&pipe, vc, ic)`    | Append a non-indexed draw (render-path ICB)  |
| `execute(count)`                | Replay the first `count` recorded commands   |
| `execute_all()`                 | Replay every recorded command                |
| `len()` / `is_empty()`          | How many commands have been recorded         |
| `capacity()`                    | The cap passed to `indirect_command_buffer`  |

Recording past `capacity()` returns `InvalidParam`.

## IndirectRenderBundle

A render bundle records draws once and replays them inside any render pass
that uses the same color/depth format. WebGPU calls these "render bundles";
Vulkan calls them secondary command buffers.

```rust
// gpu.render_bundle comes from the RenderGpu extension trait,
// in scope via `use quanta::*;` (or `use quanta::RenderGpu;`).
let mut bundle = gpu.render_bundle(32)?;
bundle.record_draw(&pipeline, /*vertex_count=*/3, /*instance_count=*/1)?;
bundle.record_draw(&pipeline, 6, 1)?;
```

Replay goes through the manual render-pass API
(`RenderPass::execute_bundle` — the chainable builder does not expose it):

```rust
let mut pass = gpu.device_handle().render_begin(&target)?;
pass.clear(Color::BLACK);
pass.execute_bundle(&bundle, /*count=*/2);
let mut pulse = gpu.device_handle().render_end(pass)?;
pulse.wait()?;
```

## Indirect draws (`draw_indirect`)

For draws whose vertex/instance counts come from a GPU buffer:

```rust
// 5 × u32 = one DrawIndexedIndirect command
let args = gpu.field::<u32>(5)?;
args.write(&[
    /* index_count   */ 6,
    /* instance_count*/ 1,
    /* first_index   */ 0,
    /* base_vertex   */ 0,
    /* first_instance*/ 0,
])?;

let indices = gpu.field::<u32>(6)?;
indices.write(&[0, 1, 2, 1, 2, 3])?;

gpu.render(&target)?
    .pipeline(&pipeline)
    .vertices(0, &vb)
    .draw_indexed_indirect(&args, /*offset=*/0, &indices)
    .pulse()?
    .wait()?;
```

The argument layout is fixed by the spec:

| Variant                   | Size  | Fields                                                       |
|---------------------------|-------|--------------------------------------------------------------|
| `draw_indirect`           | 16 B  | `vertex_count`, `instance_count`, `first_vertex`, `first_instance` |
| `draw_indexed_indirect`   | 20 B  | `index_count`, `instance_count`, `first_index`, `base_vertex` (i32), `first_instance` |

## Backend matrix

| Backend | ICB                              | Render bundles                              |
|---------|----------------------------------|---------------------------------------------|
| Metal   | `MTLIndirectCommandBuffer`       | `executeCommandsInBuffer:withRange:`        |
| Vulkan  | Recorded primary command buffer  | Secondary CB inside `RENDER_PASS_CONTINUE`  |
| WebGPU  | Software shim                    | `GPURenderBundle` + `pass.executeBundles`   |
| CPU     | Software command queue           | Software command queue                      |

## Next

- [Multi-queue](multi-queue.md) -- record on one queue, replay on another
- [Rendering](rendering.md) -- the surrounding pipeline

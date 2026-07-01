# Indirect commands

Record dispatches and draws once, then replay them — possibly with
arguments the GPU itself wrote. The substrate for GPU-driven culling,
multi-draw instancing, and frame-graph replay.

## ICB for compute

`IndirectCommandBuffer` records dispatches host-side and replays them via
a single submit:

```rust
use quanta::*;

let mut icb = gpu.indirect_command_buffer(64)?; // capacity = 64 commands

let mut wave = my_kernel(&gpu)?;
wave.bind(0, &input);

icb.record_dispatch(&wave, [256, 1, 1])?;
icb.record_dispatch(&wave, [128, 1, 1])?;

icb.execute_all()?;       // run both
// or icb.execute(1)      // run only the first
```

| Method | Effect |
|---|---|
| `record_dispatch(&wave, groups)` | Append a compute dispatch |
| `record_draw(&pipe, vc, ic)` | Append a non-indexed draw |
| `execute(count)` | Replay the first `count` recorded commands |
| `execute_all()` | Replay every recorded command |
| `len()` / `is_empty()` / `capacity()` | Bookkeeping |

Recording past `capacity()` returns `InvalidParam`.

## Render bundles

A render bundle records draws once and replays them inside any render pass
that uses the same color/depth format.

```rust
let mut bundle = gpu.render_bundle(32)?;
bundle.record_draw(&pipeline, /*vertex_count=*/3, /*instance_count=*/1)?;
bundle.record_draw(&pipeline, 6, 1)?;

gpu.render(&target)?
    .clear(Color::BLACK)
    .execute_bundle(&bundle, /*count=*/2)
    .pulse()?
    .wait()?;
```

## Indirect draws (GPU-written args)

For draws whose vertex/instance counts come from a GPU buffer:

```rust
// 5 × u32 = one DrawIndexedIndirect command.
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

| Variant | Size | Fields |
|---|---|---|
| `draw_indirect` | 16 B | `vertex_count`, `instance_count`, `first_vertex`, `first_instance` |
| `draw_indexed_indirect` | 20 B | `index_count`, `instance_count`, `first_index`, `base_vertex` (i32), `first_instance` |

A culling compute kernel can populate `args` per frame, then the render
pass replays whatever the GPU computed — no host roundtrip.

## Backend notes

| Backend | ICB | Render bundles |
|---|---|---|
| Metal  | `MTLIndirectCommandBuffer` | `executeCommandsInBuffer:withRange:` |
| Vulkan | Recorded primary CB | Secondary CB inside `RENDER_PASS_CONTINUE` |
| WebGPU | Software shim | `GPURenderBundle` + `pass.executeBundles` |

## See also

- [Multi-queue](multi-queue.md) — record on one queue, replay on another
- [Guide: Indirect commands](../../rendering/tutorials/indirect-commands.md) — full reference

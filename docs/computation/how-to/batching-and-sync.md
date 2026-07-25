# Batch GPU work and control when it executes

Dispatches don't submit one by one — they **encode** into a per-device
batch and the whole batch submits at the next *sync point*. Chains of
small kernels cost one submission instead of one each (~15× on the
per-op overhead for a define-by-run chain; the mechanism is
[Execution model → Deferred dispatch](../../concepts/execution-model.md#deferred-dispatch)).
This page is the practical side: what counts as a sync point, when you
need an explicit one, and how to structure loops to keep the batching.

## The sync points

Work encoded so far is submitted and completed by any of:

| Sync point | Use it when |
|---|---|
| `pulse.wait()` on any pulse a dispatch returned | the ordinary case — before reading results |
| `field.read()` / `write()` / `copy_from()` / `native_handle()` | automatic: completes pending work that touches *that buffer* |
| `gpu.flush()` | you have no pulse in hand (fire-and-forget kernels, external readers) |
| `gpu.wait_idle()` | drain absolutely everything |

A wait completes the whole pending batch, not just "its" dispatch —
over-waiting is the conservative direction and always correct.

## Keep chains un-synced until the end

```rust,ignore
// GOOD: 3 kernels, ONE submission — sync only at the read.
step_a(&gpu, &fields)?;               // encodes
step_b(&gpu, &fields)?;               // encodes
step_c(&gpu, &fields)?.wait()?;       // submits all three, waits once
let out = result_field.read()?;

// WASTEFUL: per-step waits fragment the batch back into
// one-submission-per-kernel (the old cost model).
step_a(&gpu, &fields)?.wait()?;
step_b(&gpu, &fields)?.wait()?;
step_c(&gpu, &fields)?.wait()?;
```

Encode order is execution order — a kernel reading another's output
needs no barrier or wait between them, on every backend.

## Training-loop shape

Read host values (a loss, a metric) as rarely as you can afford; each
host read is the frame boundary of a batch:

```rust,ignore
for step in 0..steps {
    let loss = train_step(&gpu, &mut params, batch)?;   // all encoded
    if step % 50 == 0 {
        println!("loss {}", loss.to_vec()?[0]);          // sync point
    }
}
gpu.flush()?;                                            // final drain
```

Long read-free stretches are fine: the lane auto-submits every 512
encodes (without waiting), so the GPU starts executing while you keep
encoding.

## When you MUST sync explicitly

- **Fire-and-forget work with no later read**: un-synced encoded work
  may never execute. If a kernel's only purpose is a side effect you
  observe elsewhere, `gpu.flush()` after it.
- **External readers**: anything consuming a
  [`native_handle`](../../reference/api.md) export or a `MappedField`'s
  `as_slice()` — the lane cannot intercept raw-memory reads. Flush
  first (`native_handle()` itself flushes; mapped views do not).
- **Cross-API handoff** (a windowing system, another Vulkan/Metal
  consumer): flush before handing over.

Render passes, ICB executes, async copies, and typed-queue submits
order themselves against pending compute automatically — no manual
flush needed there.

## The explicit `Batch` API

`gpu.batch()` still exists for when you want to name the boundary
yourself — N dispatches, one commit, one pulse:

```rust,ignore
let mut batch = gpu.batch()?;
for pass in 0..passes {
    batch.dispatch(&wave, n)?;
}
batch.pulse()?.wait()?;
```

It behaves like the implicit lane (same ordering guarantees); use it
when a subsystem wants its submission decoupled from the shared lane's
cadence.

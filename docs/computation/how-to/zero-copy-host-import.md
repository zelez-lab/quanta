# Zero-Copy Input from mmap

Scan file-resident data on the GPU without copying it into a device
buffer first. The recipe for any storage engine whose on-disk format is
its in-memory format: mmap the file, import the region, dispatch.

## When to use this

- The data already lives in host memory in exactly the layout the
  kernel wants (an mmap'd column, a tensor file, a page-aligned arena).
- The kernel only **reads** it. Host import is read-only; results go to
  a normal output `Field`.
- You would otherwise call `gpu.field(n)` + `field.write(&data)` —
  paying a copy for data that is already resident.

## The recipe

```rust
#[quanta::kernel]
fn knn_distances(vectors: &[f32], query: &[f32], distances: &mut [f32]) {
    let j = quark_id();
    let mut acc = 0.0f32;
    for k in 0..8 {
        let d = vectors[j * 8 + k] - query[k];
        acc = acc + d * d;
    }
    distances[j] = acc;
}

// 1. mmap the file (page-aligned by construction). Pad the file to a
//    multiple of gpu.host_import_alignment() so the whole mapping
//    satisfies the length contract.
let region: &[f32] = mapped_file_as_f32s();

// 2. Import — zero-copy where the backend supports it.
let vectors = gpu.field_from_host(region)?;
assert!(vectors.is_imported() || !gpu.supports_host_import());

// 3. Bind read-only and dispatch like any field.
let query = gpu.field::<f32>(8)?;
query.write(&my_query)?;
let distances = gpu.field::<f32>(n)?;

let mut wave = knn_distances(&gpu)?;
wave.bind_host(0, &vectors);   // the imported region — &[T] params only
wave.bind(1, &query);
wave.bind(2, &distances);
gpu.dispatch(&wave, n as u32)?.wait()?;

// 4. Small results come back normally; top-k over n distances is a
//    host-side sort.
let d = distances.read()?;
```

The runnable version — including the hand-rolled `mmap` with no extra
dependency — is `examples/headless_compute` (the knn phase), and
`examples/bench_host_import.rs` measures the imported-vs-owned scan
ratio on your device.

## The contract, in full

| Rule | Why |
|------|-----|
| Base pointer and byte length are multiples of `gpu.host_import_alignment()` | Metal and Vulkan import whole pages; the check is at import time and violations are `InvalidParam`, never a silent copy |
| Read-only: bind to `&[T]` kernel parameters only | The memory is the caller's; the type exposes no write path, and the software lane fails a dispatch whose kernel wrote an imported field |
| Keep the region alive and unmutated while dispatches that bind it are in flight | The safe `&'a [T]` borrow enforces immutability; completion ordering is yours — `wait()` the pulse before dropping the field or unmapping |
| Check `is_imported()` when the copy matters | On backends without an import path the call succeeds through one staged copy — cost is queryable, never silent |

## Per-backend truth

| Backend | Path | Granularity |
|---------|------|-------------|
| Metal | `newBufferWithBytesNoCopy`, shared storage | VM page size (16 KiB on Apple silicon) |
| Vulkan | `VK_EXT_external_memory_host` | `minImportedHostPointerAlignment` (typically 4 KiB) |
| Software | pointer passthrough | 1 |
| WebGPU | staged copy (`supports_host_import()` = false) | — |

Pair the decision with [`gpu.memory_topology()`](../../concepts/memory-model.md#transfers-and-memory-topology):
on `Unified` topology an imported scan runs at device speed (within a
few percent of a device-owned field); on `Discrete` the driver reads
host memory across the interconnect — profile whether import or an
explicit upload wins for your access pattern.

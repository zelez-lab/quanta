# GPU Architecture

A CPU has a few powerful cores optimized for latency.
A GPU has thousands of simple cores optimized for throughput.

If a CPU is a few fast horses, a GPU is a thousand slow ants — massively parallel.

## Hardware hierarchy

```
GPU (device)
 +-- Nucleus 0 (compute unit)
 |    +-- Proton 0 (warp/wave)
 |    |    +-- Quark 0..31 (threads/lanes)
 |    +-- Proton 1
 |    |    +-- Quark 32..63
 |    +-- ...
 |    +-- [shared memory: ~48KB]
 +-- Nucleus 1
 |    +-- ...
 +-- ...
 +-- [global memory: 8-80GB]
```

## Quanta naming

| Quanta term | Hardware term | What it is |
|-------------|---------------|------------|
| **Quark** | Thread / lane | One execution unit. Runs your kernel function once. |
| **Proton** | Warp (NVIDIA) / Wave (AMD) | 32 or 64 quarks executing in lockstep (SIMT). |
| **Nucleus** | Compute unit / SM / CU | Multiple protons sharing fast local memory. |
| **GPU** | Device | The whole chip. Many nuclei firing together. |

## How dispatch maps to hardware

```rust
#[quanta::kernel]
fn double(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] * 2.0;
}

// Launch 1024 quarks
gpu.dispatch(&wave, 1024)?;
```

What happens:

1. The GPU divides 1024 quarks into protons (32 quarks each = 32 protons).
2. Protons are scheduled across available nuclei.
3. Each quark calls `quark_id()` to get its unique index (0..1023).
4. Each quark reads its element, doubles it, writes it back.
5. All 1024 operations happen in parallel.

## Scale

| GPU | Nuclei | Protons/Nucleus | Quarks/Proton | Total quarks |
|-----|--------|-----------------|---------------|--------------|
| Apple M4 | 10 | 4 | 32 | 1,280 |
| RTX 4090 | 128 | 4 | 32 | 16,384 |
| AMD 7900 XTX | 96 | 2 | 64 | 12,288 |
| RPi 5 (V3D) | 4 | 4 | 16 | 256 |

You don't need to know these numbers. Quanta queries them at runtime:

```rust
let gpu = quanta::init()?;
println!("{}: {} quarks", gpu.name(), gpu.total_quarks());
```

## Key insight

One quark is slower than one CPU core. But you can run tens of thousands simultaneously.
GPU wins when the same operation applies to many data elements independently.

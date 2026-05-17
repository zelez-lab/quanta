# Getting started with quanta-prims

A 10-minute walkthrough: from `cargo new` to a working program
that uses a block-cooperative primitive on the GPU.

## What this crate is *for*

`quanta-prims` ships **block-cooperative GPU primitives** —
reduce, scan, sort. The unit of cooperation is the workgroup:
256 threads collaborating to produce one block-wide result.
Each primitive comes in two flavours:

- A **device function** (`block_reduce_add_u32_kernel`,
  `block_scan_add_u32_kernel`, …) you call from inside your own
  `#[quanta::kernel]`. The valuable shape — composable inside
  larger pipelines.
- A **top-level kernel** (`block_reduce_add_u32_buffer`,
  `block_radix_sort_u32_buffer`, …) for the common "do this op
  on a whole buffer" case.

Every primitive ships:

1. The GPU kernel above.
2. A reference single-thread CPU implementation in
   `quanta_prims::reference` — the correctness oracle.
3. (For Tier 1) a Lean theorem about the reference's behaviour
   and a Verus invariant about its operational shape.

## What you need

- **Rust 1.85+** (`rustup show` to check)
- **The wasm32 target**: `rustup target add wasm32-unknown-unknown`
- **git** on `PATH`.

For real-GPU dispatch you also need Metal Toolchain (macOS),
Vulkan SDK (Linux/Windows), or a Chromium-based browser (WebGPU).

**Block-cooperative primitives need a real GPU.** Unlike Quanta's
trivially-parallel kernels (RNG fills, element-wise compute),
the primitives in this crate require multiple threads per
workgroup. The software CPU JIT backend (`quanta::init_cpu()`)
is single-threaded — subgroup intrinsics degrade to identity
functions there, so cooperative reduce/scan/sort produce
degenerate results. Use `quanta::init()?` and skip if no GPU is
present (the examples in this crate show the pattern).

## Step 1 — Create a new project

```sh
cargo new --bin my_prims_app
cd my_prims_app
```

## Step 2 — Add quanta and quanta-prims

```sh
cargo add quanta       --git https://github.com/zelez-lab/quanta
cargo add quanta-prims --git https://github.com/zelez-lab/quanta --features gpu
```

That produces:

```toml
[dependencies]
quanta       = { git = "https://github.com/zelez-lab/quanta" }
quanta-prims = { git = "https://github.com/zelez-lab/quanta", features = ["gpu"] }
```

The `gpu` feature is what makes the `*_buffer` top-level
kernels visible. Without it you only get
`quanta_prims::reference` (pure CPU, no GPU dispatch) — useful
when you want the reference impl as an oracle.

## Step 3 — Your first reduce

Replace `src/main.rs` with:

```rust,ignore
use quanta_prims::block_reduce_add_u32_buffer;

fn main() -> Result<(), quanta::QuantaError> {
    // Block-cooperative primitives need a real GPU. Single-thread
    // init_cpu() returns degenerate results — see the
    // troubleshooting section.
    let gpu = quanta::init()?;

    // 256 inputs, ramp 1..=256.
    let n = 256usize;
    let data: Vec<u32> = (1..=n as u32).collect();

    let input = gpu.field::<u32>(n)?;
    let output = gpu.field::<u32>(1)?;  // one sum per block
    input.write(&data)?;
    output.write(&[0u32])?;

    let mut wave = block_reduce_add_u32_buffer(&gpu)?;
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, n as u32)?;
    pulse.wait()?;

    let sum = output.read()?[0];
    let expected = (1..=n as u32).sum::<u32>();
    assert_eq!(sum, expected);
    println!("Block reduce computed {sum}, expected {expected}");

    Ok(())
}
```

```sh
cargo run --release
```

Expected output: `Block reduce computed 32896, expected 32896`.

## Step 4 — Pick another op

The shape is identical for every primitive — only the kernel
name changes. To compute the minimum:

```rust,ignore
use quanta_prims::block_reduce_min_u32_buffer;

let mut wave = block_reduce_min_u32_buffer(&gpu)?;
```

To compute an inclusive prefix-sum (output has the same shape
as input, one prefix per lane):

```rust,ignore
use quanta_prims::block_scan_add_u32_buffer;

let output = gpu.field::<u32>(n)?;  // same size as input
output.write(&vec![0u32; n])?;

let mut wave = block_scan_add_u32_buffer(&gpu)?;
wave.bind(0, &input);
wave.bind(1, &output);
gpu.dispatch(&wave, n as u32)?.wait()?;

let scan = output.read()?;
// scan[k] == data[0] + data[1] + ... + data[k]
```

To sort a buffer of u32 keys (block-local — each 256-element
block sorted independently):

```rust,ignore
use quanta_prims::block_radix_sort_u32_buffer;

let mut wave = block_radix_sort_u32_buffer(&gpu)?;
wave.bind(0, &input);
wave.bind(1, &output);
gpu.dispatch(&wave, n as u32)?.wait()?;
```

## Step 5 — Use a primitive inside your own kernel

The real value is composability — calling these inside your own
`#[quanta::kernel]`. The pattern: declare a `[u32; 32]` shared
scratch array at slot 0, identity-init it, then call the device
fn.

```rust,ignore
use quanta::*;
use quanta_prims::block_reduce_add_u32_kernel;

#[quanta::kernel(workgroup_size = [256, 1, 1])]
fn dot_product_step(a: &[u32], b: &[u32], out: &mut [u32]) {
    #[quanta::shared] let scratch: [u32; 32];

    let i = quark_id();
    let block = nucleus_id();
    let lane = proton_id();

    if lane < 32u32 { scratch[lane] = 0u32; }
    barrier();

    // Each lane contributes a[i] * b[i] to the block-wide sum.
    let product = a[i as usize] * b[i as usize];
    let block_sum = block_reduce_add_u32_kernel(product);

    if lane == 0u32 {
        out[block as usize] = block_sum;
    }
}
```

That's the device-fn pattern downstream BLAS / FFT / sort
crates will use to compose primitives into bigger kernels.

## Step 6 — Use the reference impl on the CPU

Useful in tests or when you need an oracle:

```rust
use quanta_prims::reference;

let xs = vec![1u32, 2, 3, 4];
assert_eq!(reference::reduce_add_u32(&xs), 10);
assert_eq!(reference::scan_add_u32(&xs), vec![1, 3, 6, 10]);
assert_eq!(reference::radix_sort_u32(&[3, 1, 4, 1, 5]), vec![1, 1, 3, 4, 5]);
```

The reference module is always available — no GPU feature
needed. It's pure-Rust single-threaded code, deliberately
simple so its correctness is obvious by inspection.

## Step 7 — Real GPU

Swap `quanta::init_cpu()` for `quanta::init()?`:

```rust,ignore
let gpu = quanta::init()?;
```

Note the `?`: `init()` returns `Result<Gpu, QuantaError>` and
fails with `NoDevice` when no supported GPU is present.
`init_cpu()` is infallible — the software backend always works.

This picks the best available backend: Metal on macOS, Vulkan
on Linux/Windows, WebGPU in a browser.

## Where to go next

- **[README.md](README.md)** — full crate overview.
- **[COOKBOOK.md](COOKBOOK.md)** — recipe catalogue.
- **[CHANGELOG.md](CHANGELOG.md)** — release history.
- **`tests/block_reduce_family.rs`** — 9 differential cases
  across the reduce family. Good template for writing your own
  tests.

## Troubleshooting

**`error[E0432]: unresolved import quanta_prims::block_reduce_add_u32_buffer`**
You forgot `features = ["gpu"]` in your `Cargo.toml`. Without
it the `*_buffer` top-level kernels aren't compiled in.

**`QuantaError { kind: NoDevice }` from `quanta::init()`**
No supported GPU found on this machine. The block-cooperative
primitives in this crate can't fall back to `init_cpu()` —
they need multiple threads per workgroup. The pure-Rust
`quanta_prims::reference` module always works (single-thread,
no GPU dependency) and is the right tool for unit-test oracles
or when you only need the CPU result.

**My output looks like the input was just copied through.**
You're probably running on `quanta::init_cpu()`. The software
backend is single-threaded — every subgroup intrinsic returns
its own value, so cooperative reduce/scan/sort produce
trivial results. Switch to `quanta::init()?` on a machine
with a real GPU.

**My results don't match across runs (only the reduce sums)**
Floating-point reductions on the GPU use a tree pattern with a
different evaluation order than the sequential CPU reference.
This produces tiny (≪ ULP-per-element) differences. Compare
with a relative tolerance, not bit-exact, when working with
`f32` — see `tests/block_reduce_family.rs::add_f32_matches_reference_within_tolerance`
for the recommended check.

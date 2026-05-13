# Getting started with quanta-rand

A 10-minute walkthrough: from `cargo new` to a working program that
draws random numbers on the GPU.

## What you need

- **Rust 1.85+** (`rustup show` to check)
- **The wasm32 target**: `rustup target add wasm32-unknown-unknown`
- This `quanta` workspace cloned somewhere on your machine. The path
  matters for the local dependency in step 2.

That's it for the software backend (works on any laptop). To run on
a real GPU you also need Metal Toolchain (macOS), Vulkan SDK (Linux/
Windows), or a Chromium-based browser (WebGPU). The software backend
runs the exact same kernels via a CPU JIT — perfect for getting
started.

## Step 1 — Create a new project

```sh
cargo new --bin my_rng_app
cd my_rng_app
```

## Step 2 — Add quanta-rand to `Cargo.toml`

Adjust the path to point at the `quanta-rand` crate inside this
workspace. If your project sits next to the `quanta_project`
checkout, that's:

```toml
[package]
name = "my_rng_app"
version = "0.1.0"
edition = "2024"

[dependencies]
quanta = { path = "../quanta_project/quanta" }
quanta-rand = { path = "../quanta_project/quanta/crates/quanta-rand", features = ["gpu"] }
```

The `gpu` feature is what makes the `fill_*_gpu` helpers visible.
Without it you only get the host-side `Rng` (pure CPU, no GPU
dispatch).

## Step 3 — Write the program

Replace `src/main.rs` with:

```rust
use quanta_rand::fill_uniform_f32_gpu;

fn main() -> Result<(), quanta::QuantaError> {
    // For real-GPU dispatch use quanta::init()?; here we use the
    // software backend so this works on any machine.
    let gpu = quanta::init_cpu();

    let n = 1_000;
    let seed = 0xCAFE_BABE_DEAD_BEEFu64;

    let samples: Vec<f32> = fill_uniform_f32_gpu(&gpu, n, seed)?;

    let mean: f32 = samples.iter().sum::<f32>() / n as f32;
    println!("Drew {n} samples from uniform [0, 1)");
    println!("First 5: {:.4?}", &samples[..5]);
    println!("Mean   : {mean:.4} (expected ≈ 0.5)");

    Ok(())
}
```

## Step 4 — Build and run

```sh
cargo run --release
```

The first build will take a minute (it compiles `quanta` and a few
backend crates). Subsequent runs are fast.

Expected output:

```
Drew 1000 samples from uniform [0, 1)
First 5: [0.7869, 0.7375, 0.7636, 0.6787, 0.8780]
Mean   : 0.4983 (expected ≈ 0.5)
```

The exact numbers depend on the seed — change `seed` and the values
change, but the same seed always gives the same bytes.

## Step 5 — Try other distributions

Each distribution has the same shape. Swap the import and the call:

```rust
use quanta_rand::fill_normal_f32_gpu;
// ...
let samples = fill_normal_f32_gpu(&gpu, n, seed)?;   // N(0, 1)
```

```rust
use quanta_rand::fill_bernoulli_u32_gpu;
// ...
let p = 0.3;
let mask = fill_bernoulli_u32_gpu(&gpu, n, seed, p)?;
```

```rust
use quanta_rand::fill_exponential_f32_gpu;
// ...
let lambda = 2.0;
let samples = fill_exponential_f32_gpu(&gpu, n, seed, lambda)?;
```

See the [README](README.md#what-you-get) for the full table.

## Step 6 — Real GPU (optional)

Swap `quanta::init_cpu()` for `quanta::init()?`:

```rust
let gpu = quanta::init()?;
println!("backend: {}", gpu.name());
```

This picks the best available backend: Metal on macOS, Vulkan on
Linux/Windows, WebGPU in a browser. Output is bit-identical with
the software backend — that's the whole point of counter-based
RNGs.

If `quanta::init()` returns `NoDevice`, the runtime couldn't find
a supported GPU — fall back to `init_cpu()` for the same algorithm
at slower throughput.

## Step 7 — CPU-side `Rng`

For single-thread CPU code (not a kernel), the `Rng` type wraps
xoshiro128++ with a familiar streaming API:

```rust
use quanta_rand::Rng;

let mut rng = Rng::from_seed(0xC0FFEE);
let x: u32 = rng.next_u32();
let f: f32 = rng.next_f32();
let z: f32 = rng.next_normal_f32();

// Spawn an independent stream by advancing 2^64 steps:
let mut other = rng.clone();
other.jump();
```

`Rng` is useful when you need one or two draws (the GPU dispatch
overhead would dominate), or as a deterministic CPU oracle in
tests for GPU-side code.

## Where to go next

- **[README.md](README.md)** — full API reference with the
  distribution table and v0.1 limits.
- **`examples/fill_buffer.rs`** — one block per distribution,
  copy-pasteable.
- **`examples/nn_weight_init.rs`** — Glorot / He initialization for
  a neural network.
- **`examples/monte_carlo_pi.rs`** — classic Monte Carlo estimate.
- **`examples/dropout_mask.rs`** — inverted dropout for NN training.

Run any of them from the workspace root:

```sh
cargo run --example fill_buffer -p quanta-rand --features gpu --release
```

## Troubleshooting

**`error[E0432]: unresolved import quanta_rand::fill_uniform_f32_gpu`**
You forgot `features = ["gpu"]` in your `Cargo.toml`. Without it
the fill helpers aren't compiled in.

**`error: linker `rust-lld` failed` / `wasm32-unknown-unknown` not found**
Run `rustup target add wasm32-unknown-unknown`. The kernel macro
shells out to rustc with that target.

**`QuantaError { kind: NoDevice }` from `quanta::init()`**
No supported GPU found on this machine. Use `quanta::init_cpu()`
instead — same kernels, same output, slower throughput.

**Build takes forever**
The first build compiles the whole Quanta stack (LLVM-based
codegen). After that, incremental builds are seconds. If you only
need the host-side `Rng` (no kernels), drop the `gpu` feature
and the build is much smaller.

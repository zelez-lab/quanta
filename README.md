# Quanta

**One Rust function → native GPU binary for every platform, embedded at compile time.**

Quanta is a GPU compute and rendering API for Rust. Write your kernel or shader in plain Rust and annotate it with `#[quanta::kernel]`, `#[quanta::vertex]`, or `#[quanta::fragment]`. At build time, Quanta compiles it to native binaries for NVIDIA (PTX), AMD (GCN), Vulkan (SPIR-V), and Apple (metallib) — and embeds them in your binary. At runtime, the right one runs on whatever GPU is present.

No separate shader files. No runtime shader compilation. No `build.rs` choreography to ship a parallel shader crate.

```rust
use quanta::prelude::*;

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    let n = 1_000_000;

    let a = gpu.compute_field::<f32>(n)?;
    let b = gpu.compute_field::<f32>(n)?;
    let result = gpu.compute_field::<f32>(n)?;

    gpu.write_field(&a, &vec![1.0; n])?;
    gpu.write_field(&b, &vec![2.0; n])?;

    let mut wave = vector_add(&gpu)?;
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    gpu.dispatch(&wave, n as u32)?.wait()?;

    let out = gpu.read_field(&result)?;
    assert_eq!(out[0], 3.0);
    Ok(())
}
```

That single Rust function compiles to PTX (NVIDIA), GCN (AMD), MSL (Apple), and WGSL (WebGPU) — all embedded in your binary at build time.

---

## Why Quanta

- **Write GPU code in Rust, not in a shader DSL.** Real generics, real control flow, real types, the borrow checker. Refactor with `cargo`. Test with `cargo test`. No string-templated shader code, no separate compilation pipeline.
- **One source, all major GPUs.** A single `#[quanta::kernel]` ships to NVIDIA, AMD, Apple, and the web. The right ISA loads at runtime based on the device — you write it once.
- **Build-time compilation.** Kernel errors are compile errors. No surprise shader-compile failures the first time a user runs your program on Vulkan.
- **Compute and rendering in one library.** Same `Field`, same `Wave`, same kernel macro. `RenderPass` for graphics, `dispatch` for compute. No need to glue two stacks together.

---

## How Quanta compares

| | **Quanta** | **CUDA** | **wgpu** | **rust-gpu** | **CubeCL** |
|---|---|---|---|---|---|
| **Kernel language** | Rust (`#[quanta::kernel]`) | CUDA C++ | WGSL | Rust subset (custom rustc backend) | Rust (`#[cube]` proc macro) |
| **NVIDIA** | ✓ direct PTX | ✓ native | ✓ via Vulkan/DX12 | ✓ via SPIR-V/Vulkan | ✓ native CUDA runtime |
| **AMD** | ✓ direct GCN | ✗ | ✓ via Vulkan/DX12 | ✓ via SPIR-V/Vulkan | ✓ native ROCm/HIP |
| **Apple/Metal** | ✓ native MSL | ✗ | ✓ native Metal | ✗ (SPIR-V only) | ✓ via wgpu |
| **Intel** | ✓ via Vulkan | ✗ | ✓ | ✓ via Vulkan | ✓ via wgpu |
| **WebGPU/browser** | WGSL source emitted¹ | ✗ | ✓ first-class | ✗ | ✓ via wgpu |
| **Compute** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Render pipeline** | ✓ | ✗² | ✓ | ✓ | ✗ |
| **Compile target** | **Build-time, all 4 ISAs embedded** | Offline (PTX) + JIT to SASS | Runtime (WGSL → native) | Build-time (SPIR-V) | JIT |
| **System deps** | LLVM 22.1 | CUDA Toolkit + driver | OS-shipped Vulkan/Metal/DX | Nightly rustc + Vulkan SDK | rustc; CUDA/HIP optional |
| **`no_std` kernels** | Planned | n/a | partial | ✓ (by design) | unverified |
| **Stable since** | v0.1 (beta) | 2007 | 2018 | 2024 (alpha) | 2024 (pre-1.0) |
| **License** | Apache-2.0 OR MIT | NVIDIA EULA (proprietary) | Apache-2.0 OR MIT | Apache-2.0 OR MIT | Apache-2.0 OR MIT |

<sub>¹ Quanta emits WGSL source at build time. To run in a browser, pair with [`wgpu`](https://github.com/gfx-rs/wgpu) on the consumer side; Quanta does not ship a browser runtime in v1.</sub>
<sub>² CUDA itself is compute-only; graphics interop with OpenGL/Vulkan/DX is supported but not a first-class render pipeline.</sub>

### When to choose another tool

**CUDA** — if you need absolute peak performance on NVIDIA hardware, or any piece of the CUDA ecosystem (cuBLAS, cuDNN, CUTLASS, TensorRT, Triton, NCCL). 18 years of vendor-tuned libraries and tooling. Trade-off: NVIDIA lock-in, proprietary EULA.

**wgpu** — if your workload is primarily graphics, you need browser deployment (WebGPU + WebGL2 fallback), or you want a battle-tested API powering Firefox, Servo, Deno, and Bevy. The most mature Rust GPU library by a wide margin.

**rust-gpu** — if you need to write the full graphics pipeline (vertex/fragment/mesh/ray-tracing) in real Rust source and share `no_std` library code between CPU and GPU crates. Targets SPIR-V; Vulkan-centric.

**CubeCL** — if you need a Rust-native compute DSL with first-class native CUDA *and* ROCm backends and a JIT-compiled IR, on the same stack as the Burn deep-learning framework. Compute-only.

---

## Backends

| Backend | Cargo feature | Status |
|---|---|---|
| Apple Metal | `metal` (default) | ✓ Compute + Render |
| Vulkan (Linux/Windows/Android) | `vulkan` | ✓ Compute + Render |
| Software (CPU reference) | — | Planned |

NVIDIA and AMD targets do not require a separate Quanta backend — they consume the PTX / GCN ISA emitted by the compiler at build time, dispatched through the platform's native GPU runtime (CUDA / ROCm).

---

## Examples

```bash
cargo run --release --example hello_quanta      # vector_add
cargo run --release --example bench_compute     # heavy math, CPU vs GPU
cargo run --release --example bench_mandelbrot  # 4K Mandelbrot, CPU vs GPU
cargo run --release --example bench_nbody       # 16K-particle N-body, CPU vs GPU
```

Each benchmark prints CPU vs GPU timing and a speedup ratio for that workload on your hardware.

---

## Installation

```toml
# Cargo.toml
[dependencies]
quanta = "0.1"
```

### System requirements

- **Rust 1.85+** (edition 2024).
- **LLVM 22.1**, dynamically linked. The kernel compiler uses LLVM to emit PTX, GCN, and SPIR-V. Install via your platform's package manager:
  - macOS: `brew install llvm@22`
  - Debian/Ubuntu: see [apt.llvm.org](https://apt.llvm.org/)
  - Other: build from source or use a binary distribution
- **Quanta compiler binary**, installed once:

  ```bash
  cargo install quanta-compiler
  ```

  Without it, only the MSL and WGSL targets compile (Apple + WebGPU). With it, all four ISAs (PTX, GCN, MSL, WGSL) are emitted.

- **Platform GPU runtime** for whichever target you actually run on (Metal ships with macOS; Vulkan via your distro or [LunarG SDK](https://vulkan.lunarg.com/); CUDA driver from NVIDIA; ROCm from AMD).

---

## How it works

```
Your Rust function with #[quanta::kernel]
        │
        ▼
   syn AST  ──►  Quanta IR (KernelOps)
                       │
                       ▼
              quanta-compiler (LLVM)
                       │
        ┌──────┬───────┼────────┬──────┐
        ▼      ▼       ▼        ▼      ▼
       PTX   GCN    SPIR-V     MSL    WGSL
        │      │       │        │      │
        └──────┴───────┴────────┴──────┘
                       │
                       ▼
        embedded into your binary at build time
                       │
                       ▼
              runtime: dispatch on detected GPU
```

The `#[quanta::kernel]` proc macro parses your Rust function, lowers it to Quanta's typed IR, and invokes `quanta-compiler` to emit ISA for every target. The resulting binaries are embedded as `static` arrays. At runtime, Quanta detects the GPU vendor and selects the matching ISA.

---

## Status

Quanta is **beta** — compute and rendering are fully functional on Metal (macOS)
and Vulkan (Linux/Android/Windows). Both platforms produce identical pixel output
for all rendering tests.

**Verified on hardware:**
- macOS Metal: Apple Silicon (M-series)
- Vulkan: Raspberry Pi 5 (Broadcom V3D)

**What works (306 tests, 0 failures):**
- Compute: reductions, atomics, shared memory, warp primitives, math intrinsics
- Rendering: vertex/index buffers, depth testing, instanced draw, texture sampling
- Shader body evaluation: arithmetic, math functions (30), matrix-vector multiply,
  if/else conditionals, vertex→fragment varyings, texture sampling via `sample()`
- Binary-only output: SPIR-V + metallib (no runtime shader compilation)
- Push constants for uniform parameters (MVP matrices, etc.)
- Zero validation errors on both Metal and Vulkan

**What's tracked for follow-up releases:**
- WebGPU runtime backend (WGSL emitter exists, browser host pending)
- Tensor-core / matrix-engine intrinsics
- Software (CPU) backend for headless testing
- Multi-GPU dispatch primitives

---

## License

Dual-licensed under either of:

- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

Contributions are accepted under the same dual-license terms.

---

## Contributing

Issues and pull requests welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development setup, the IR design, and how to add a new backend.

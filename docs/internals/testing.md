# Testing

## Test tiers

```
Tier 1: Host tests       No GPU needed. Pure logic, IR roundtrips, serialization.
Tier 2: GPU tests        Real GPU dispatch. Requires Metal or Vulkan device.
Tier 3: Visual tests     Render output compared pixel-by-pixel to reference.
Tier 4: Stress tests     Long-running: memory pressure, dispatch storms, error recovery.
```

### Tier 1: Host tests

Run with `cargo test` (no GPU required):

```
quanta-ir/tests/roundtrip.rs     Serialize/deserialize every KernelOp variant
quanta-ir/src/wire/tests.rs      Wire format edge cases
quanta-macros/                    Parse + validate (syn-based, no GPU)
quanta-compiler/                  IR -> LLVM IR text (no code emission)
```

These run in CI on any platform. Target: <5 seconds total.

```bash
cargo test -p quanta-ir
cargo test -p quanta-macros
cargo test -p quanta-compiler
```

### Tier 2: GPU tests

Integration tests that dispatch real kernels and verify results:

```
tests/conformance_test.rs        Entry point for conformance suite
tests/conformance/
    compute.rs                   Compute dispatch: vector_add, reduce, atomics
    memory.rs                    Field alloc/write/read/copy, mapped fields
    texture.rs                   Texture create/write/read, format support
tests/shader_compile_test.rs     Vertex/fragment shader compilation
```

Run with `cargo test --features metal` (or `vulkan`):

```bash
# macOS
cargo test --features metal -- --test-threads=1

# Linux with Vulkan
cargo test --features vulkan -- --test-threads=1
```

Single-threaded because GPU tests share device state.

### Tier 3: Visual tests

Render-to-texture tests that compare output against reference images:

- Render a known scene (triangle, textured quad, etc.)
- Read back framebuffer pixels.
- Compare against a golden reference (PNG).
- Fail if RMSE exceeds threshold (allows for driver rounding differences).

### Tier 4: Stress tests

Long-running tests for reliability:

- **Memory pressure**: allocate until OOM, verify graceful error.
- **Rapid dispatch**: 10K dispatches in sequence, verify no handle leaks.
- **Large fields**: 1GB+ allocations, verify correct addressing.
- **Error recovery**: dispatch with invalid bindings, verify error without crash.

## Generated host oracles (differential testing)

Every `#[quanta::kernel]` in the single-quark-pure subset (no shared
memory, atomics, barriers, subgroup/collective ops, textures, or f16)
also emits a hidden `<name>_host_oracle` twin: the same rewritten
kernel body compiled natively by rustc and looped over quark ids.
Struct-ref kernels take the same `&mut Data` struct as the dispatch
wrapper; flat kernels mirror the kernel's own signature with `&[T]` /
`&mut [T]` slices plus scalars in declaration order.

Running the kernel on the CPU backend must reproduce the oracle
bit-exactly — both sides are IEEE f32 through the same libm, so any
divergence is a lowering or IR-execution bug, not float noise. This
is the systematized form of the hand-written replicas that caught the
2026-06 redirect-chain miscompiles; its first in-tree use
(`tests/host_oracle_parity.rs`) immediately caught a fourth one (an
unguarded intra-block tail after an outer-targeting `br_if`).

A parity test is three lines:

```rust
let got = my_kernel_gpu(&gpu, n, ...)?;            // CPU backend
unsafe { my_kernel_host_oracle(n as u32, &mut want) };
assert_eq!(got, want.out);
```

Kernels outside the pure subset simply get no oracle — referencing
the missing fn is a compile error, never a wrong comparison. Existing
parity suites: `tests/host_oracle_parity.rs` (lowering-bug shapes)
and `crates/quanta-rand/tests/ptrd_host_oracle.rs` (PTRD + uniform
fills, plus one hand-written replica kept as an independent
cross-check of the twin generator itself).

## External validation suites

### dEQP (drawElements Quality Program)

Khronos conformance tests for Vulkan. Run against the Quanta Vulkan driver
to verify correctness of rendering and compute operations.

```bash
# Build dEQP
cd external/deqp && cmake . && make

# Run compute subset
./deqp-vk --deqp-case=dEQP-VK.compute.*
```

### Metal validation layer

Xcode's GPU validation catches:
- Uninitialized buffer reads
- Out-of-bounds texture access
- Invalid pipeline state
- Resource hazards

Enable with `QUANTA_VALIDATE=1` or via Xcode scheme settings.

### spirv-val

Validates SPIR-V binaries produced by the compiler:

```bash
# Validate compiler output
quanta-compiler --test-spirv | spirv-val
```

Beyond the manual pipe, validity is asserted in the test suites: the
`quanta-ir` `emit_spirv_*` tests (bool-mask, narrow-storage, bool-cast,
signedness, …), `tests/validate_spirv.rs` / `tests/validate_compiler_output.rs`,
and the subgroup / texture-compute GPU tests all run the emitted module
through `spirv-val` and **fail** on an invalid module. (The build-time gate
inside the compiler only logs; the tests are the hard check.) They skip the
validation silently when `spirv-val` isn't installed, so SPIRV-Tools is a
soft dependency of the dev loop.

### ptxas

NVIDIA's PTX assembler validates PTX text:

```bash
# Check PTX syntax
quanta-compiler --test-ptx > /tmp/test.ptx
ptxas --gpu-name=sm_86 /tmp/test.ptx
```

## Local Vulkan testing on alternative hardware

Beyond CI, you can run `cargo test --features vulkan` against any
device that exposes a Vulkan ICD. Useful when the AMDGPU
self-hosted runner isn't registered yet:

- **Raspberry Pi 5** (BCM2712, V3D 7.1, Vulkan 1.2 via Mesa V3DV).
  Validates basic compute + sparseBinding (slice 16 cache + slice
  18 enable). Does **not** expose VK_KHR_fragment_shading_rate /
  VK_EXT_mesh_shader / ray-tracing extensions, so those slices'
  hardware paths stay NotSupported there.
- **Linux + AMD GPU** (gfx1030+ / RDNA2). RADV exposes all the
  extensions we ship gates for (VRS, mesh shaders, ray tracing).
  Same setup as the AMDGPU CI runner without the registration.
- **Linux + lavapipe** (any x86_64 box with Mesa). Software
  Vulkan, same coverage as the per-PR CI lane.

Run with `cargo test --features vulkan --target
aarch64-unknown-linux-gnu` (RPi 5) or just
`cargo test --features vulkan` (x86_64).

## CI strategy

```
+-- GitHub Actions (per-PR) ---------+
|                                    |
|  [Linux x86_64]                    |
|    Tier 1: host tests              |
|    Differential CI (software lane) |
|    Compiler: IR -> LLVM IR text    |
|    spirv-val on SPIR-V output      |
|                                    |
|  [chromium-webgpu via Playwright]  |
|    web smoke tests (4 examples)    |
|    + golden-image SHA assertion    |
|                                    |
|  [macOS-14 / Apple GPU]            |
|    Differential CI (Metal lane)    |
|    [promoted from nightly @ 063]   |
|                                    |
|  [Linux + lavapipe]                |
|    Differential CI (Vulkan lane)   |
|    [promoted from nightly @ 063]   |
|                                    |
+------------------------------------+

+-- GitHub Actions (nightly cron) ---+
|                                    |
|  Same lanes as per-PR — the cron   |
|  schedule is the safety net for    |
|  branches without recent activity. |
|                                    |
+------------------------------------+

+-- Self-hosted (label-gated) -------+
|                                    |
|  [Linux + AMD GPU + RADV]          |
|    Differential CI (Vulkan lane)   |
|    Triggered by `run-amd-diff`     |
|    label on a PR. Covers VRS /     |
|    mesh-shader / ray-tracing       |
|    extensions lavapipe lacks.      |
|                                    |
+------------------------------------+
```

### Registering the AMDGPU self-hosted runner

The `diff-amdgpu` job in
[`.github/workflows/diff-full.yml`](../../.github/workflows/diff-full.yml)
expects a runner with the labels `[self-hosted, linux, gpu-amd]`. Until
the runner is registered the job stays inert (skipped on every event)
because it gates on `workflow_dispatch` or the `run-amd-diff` PR label,
neither of which fires by default.

One-time setup:

1. Provision a Linux box with an AMD GPU (gfx9+ recommended).
2. Install drivers + Vulkan tools:
   ```sh
   sudo apt-get install \
     mesa-vulkan-drivers libvulkan-dev vulkan-tools \
     vulkan-validationlayers
   ```
   Confirm `vulkaninfo --summary` shows `AMD RADV ...`.
3. **Settings → Actions → Runners → New self-hosted runner** in the
   GitHub repo.
4. Tag the runner with all three labels: `self-hosted`, `linux`,
   `gpu-amd`.
5. Verify with a manual `workflow_dispatch` of `Differential CI (full
   lanes)` — the `diff-amdgpu` job should pick up and pass.

The job uses `run-amd-diff` (not `run-full-diff`) as its PR-label gate
so that ordinary "run the full diff matrix" PR labels don't queue
against a runner that may be offline.

## Writing a new conformance test

```rust
// tests/conformance/compute.rs

#[test]
fn test_atomic_add() {
    let gpu = quanta::init().unwrap();

    // Create field initialized to zero
    let counter = gpu.compute_field::<u32>(1).unwrap();
    gpu.write_field(&counter, &[0u32]).unwrap();

    // Kernel: 1024 quarks each add 1
    let mut wave = atomic_increment(&gpu).unwrap();
    wave.bind(0, &counter);
    let mut pulse = gpu.dispatch(&wave, 1024).unwrap();
    gpu.wait(&mut pulse).unwrap();

    // Verify: counter should be exactly 1024
    let result = gpu.read_field(&counter).unwrap();
    assert_eq!(result[0], 1024);
}
```

Rules:
- Each test allocates its own resources (no shared state between tests).
- Tests must pass on both Metal and Vulkan.
- Floating-point comparisons use epsilon tolerance (1e-5 for f32).
- Tests that need specific hardware features should check `gpu.caps()` and skip.

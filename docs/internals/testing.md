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

### ptxas

NVIDIA's PTX assembler validates PTX text:

```bash
# Check PTX syntax
quanta-compiler --test-ptx > /tmp/test.ptx
ptxas --gpu-name=sm_86 /tmp/test.ptx
```

## RPi 5 Vulkan testing

The Raspberry Pi 5 has a V3D 7.1 GPU with Vulkan 1.2 support.
Used for testing on real (low-end) Vulkan hardware:

```bash
# On RPi 5
cargo test --features vulkan --target aarch64-unknown-linux-gnu
```

Tests that run on RPi 5 verify:
- Basic compute dispatch works on mobile-class Vulkan.
- Memory-constrained scenarios (8GB shared with CPU).
- Vulkan 1.2 features (no 1.3 extensions available).

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
+------------------------------------+

+-- GitHub Actions (nightly cron) ---+
|                                    |
|  [macOS-14 / Apple GPU]            |
|    Differential CI (Metal lane)    |
|                                    |
|  [Linux + lavapipe]                |
|    Differential CI (Vulkan lane)   |
|                                    |
+------------------------------------+

+-- Self-hosted (label-gated) -------+
|                                    |
|  [Linux + AMD GPU + RADV]          |
|    Differential CI (Vulkan lane)   |
|    Triggered by `run-amd-diff`     |
|    label on a PR.                  |
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

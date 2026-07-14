# Error Reference

All `QuantaError` types, their causes, and how to handle them.

## Error structure

```rust
#[non_exhaustive]
pub struct QuantaError {
    pub kind: QuantaErrorKind,
    pub context: Option<String>,
}
```

Every error has a `kind` (the category) and an optional `context` string
describing which operation produced it.

`QuantaError` implements `core::error::Error`, so it composes with `?`,
`Box<dyn Error>`, and error-reporting frameworks. Both the struct and
`QuantaErrorKind` are `#[non_exhaustive]` — match kinds with a wildcard
arm.

```rust
match err.kind {
    QuantaErrorKind::OutOfMemory => free_unused_resources(),
    QuantaErrorKind::CompilationFailed(ref msg) => eprintln!("shader: {msg}"),
    _ => panic!("{err}"),
}
```

---

## Error kinds

### `NoDevice`

No GPU device found.

**Causes:**
- No GPU hardware present
- GPU driver not installed or not loaded
- Running in an environment without GPU access (SSH, headless server without GPU)

**Resolution:**
- Check `quanta::devices()` to list available GPUs
- Verify driver installation
- On macOS, Metal is always available on Apple Silicon

```rust
let gpu = quanta::init().map_err(|e| {
    if e.kind == QuantaErrorKind::NoDevice {
        eprintln!("No GPU found. Is the driver installed?");
    }
    e
})?;
```

### `OutOfMemory`

GPU memory allocation failed.

**Causes:**
- Allocating a field or texture larger than available GPU memory
- Too many live allocations (GPU memory fragmented)
- Other processes consuming GPU memory

**Resolution:**
- Drop unused `Field` and `Texture` objects (memory is freed on drop)
- Reduce allocation sizes
- Check `gpu.caps().memory_bytes` for total available memory

```rust
let field = gpu.field::<f32>(count).map_err(|e| {
    if e.kind == QuantaErrorKind::OutOfMemory {
        let mb = count * 4 / 1_000_000;
        eprintln!("Cannot allocate {mb} MB on GPU");
    }
    e
})?;
```

### `CompilationFailed(String)`

Shader or kernel compilation failed.

**Causes:**
- Invalid kernel source (syntax error in generated MSL/WGSL)
- Unsupported operation for the target GPU
- No compiled binary available for the current GPU vendor
- Driver compiler bug (rare)

**Resolution:**
- Check the error message for the specific compilation error
- Verify the kernel uses only supported types and operations
- Set `QUANTA_VALIDATE=1` for additional diagnostics

```rust
let wave = my_kernel(&gpu).map_err(|e| {
    if let QuantaErrorKind::CompilationFailed(ref msg) = e.kind {
        eprintln!("Kernel compilation failed:\n{msg}");
    }
    e
})?;
```

### `SubmitFailed`

Command submission to the GPU failed.

**Causes:**
- GPU device was lost during operation
- Invalid resource handle (field/texture was dropped before dispatch)
- Driver internal error
- Command buffer overflow (too many operations without submitting)

**Resolution:**
- Ensure all bound resources are still alive during dispatch
- Check for `DeviceLost` errors on subsequent operations
- Reduce batch size if submitting very large command buffers

### `Timeout`

GPU operation did not complete within the expected time.

**Causes:**
- Kernel running too long (infinite loop in shader code)
- GPU hang (hardware lockup)
- System-level GPU timeout (Windows TDR, macOS watchdog)

**Resolution:**
- Check kernel for infinite loops or excessive iteration counts
- Reduce dispatch size for debugging
- On Windows, TDR timeout is ~2 seconds by default
- Add bounds checks in kernels: `if quark_id() >= count { return; }`

### `DeviceLost`

The GPU device was lost and can no longer be used.

**Causes:**
- GPU hardware physically removed (eGPU)
- Driver crash
- GPU reset due to hang/timeout
- Power management put the GPU to sleep

**Resolution:**
- Re-initialize: `quanta::init()` to get a new device
- All existing `Field`, `Texture`, `Wave`, and `Pipeline` objects are invalid
- Application must recreate all GPU resources

### `InvalidParam(String)`

An API function was called with invalid arguments **the caller could
have caught**. Distinct from `NotSupported` (feature is genuinely
unavailable on this backend) and `NotFound` (handle does not exist).

**Causes:**
- Zero-size allocation
- Mismatched field/texture dimensions
- Invalid slot number
- Negative or out-of-range values

**Resolution:**
- Read the error message for which parameter is wrong
- Check that array sizes and counts are positive and within limits

```rust
let field = gpu.field::<f32>(0); // Error: zero count
```

### `NotSupported(String)`

The requested feature is not implemented on this backend or hardware.
This is an *expected* return value, not a bug — callers should branch on
it and fall back to a non-accelerated path or skip the feature.

**Causes:**
- Backend gap: e.g. ray tracing on WebGPU, sparse residency on the
  software CPU device, mesh shaders on a backend that doesn't expose
  `VK_EXT_mesh_shader`.
- Hardware gap: e.g. ray tracing on Apple GPUs older than family 6,
  sparse textures on Apple GPUs older than family 7.
- v0.1 limit: e.g. sparse textures restricted to 2D + single-mip,
  ray-tracing build dispatch gated until AMDGPU validation lands.

**Resolution:**
- Capability-check first: `gpu.supports_ray_tracing()`,
  `gpu.supports_sparse_residency()`, `gpu.supports_mesh_shaders()`,
  `gpu.supported_shading_rates()`, etc.
- On `NotSupported`, take the fallback path explicitly. Do not retry
  blindly — the answer will not change for this device.

```rust
let blas = if gpu.supports_ray_tracing() {
    Some(gpu.acceleration_structure_blas(&[geom])?)
} else {
    None  // fall back to rasterizer
};
```

The error message identifies *which* feature and *why* (backend gap vs.
hardware gap vs. v0.1 limit), so it's safe to log and surface to the
end user.

### `NotFound(String)`

The given handle does not refer to a live resource on this device.

**Causes:**
- The wrapping typed handle (`Field`, `Texture`, `Wave`, `Pulse`,
  `AccelerationStructure`, `SparseTexture`, ...) was already dropped.
  Its backend handle was freed, so subsequent calls referring to it by
  raw `u64` no longer resolve.
- A handle from a different `Gpu` instance was passed in. Each driver
  maintains its own handle table; cross-device handles never resolve.
- Internal bookkeeping bug — the typed wrapper recorded the handle but
  the driver's table doesn't know about it. Reproduce + file an issue.

**Resolution:**
- Keep typed wrappers alive for the full duration you need the GPU
  resource. `Drop` is what frees the handle — nothing else.
- If you're using raw `u64` handles via `.handle()`, do not let the
  owning typed wrapper drop while the handle is still in flight.
- Don't pass handles from one `Gpu` to another.

### `SurfaceOutdated(String)`

The presentation surface no longer matches its target — the window /
layer was resized or its properties changed since the last
`configure`. Returned by `Surface::acquire` (and configure-time
validation). **Recoverable**: this is the surface's resize signal, not
a failure.

**Resolution:**
- Call `Surface::configure` with the new extent, then acquire again:

```rust
let frame = match surface.acquire() {
    Ok(frame) => frame,
    Err(e) if matches!(e.kind, QuantaErrorKind::SurfaceOutdated(_)) => {
        surface.configure(SurfaceConfig::new(new_w, new_h))?;
        continue; // retry next loop iteration
    }
    Err(e) => return Err(e),
};
```

- Frames acquired before the reconfigure must be presented or dropped
  first.

### `Internal(String)`

Something inside Quanta went wrong that the caller cannot fix. Lock
poisoning, broken invariants, "this code path was supposed to be
unreachable." File a bug.

---

## Specific error stories

A few errors have precise, diagnosable texts worth recognising. The
first two are **build-time** errors (they abort `cargo build` through the
proc macro); the last two are returned at pipeline-build / dispatch time.

### Stale compiler (fatal, build time)

When the resolved `quanta-compiler` binary reports a build revision that
provably differs from the quanta crate being compiled, the build stops —
a stale compiler can emit invalid SPIR-V that crashes some drivers:

> quanta-compiler at `<path>` was built from rev `<bin_rev>` but this
> quanta build is rev `<own_rev>`. A mismatched compiler can emit invalid
> SPIR-V that crashes some drivers, so this is a hard error. Reinstall the
> matching compiler: `cargo install --path crates/lang/quanta-compiler --locked
> --force`. To proceed anyway (e.g. a rig pinning a known-compatible
> compiler), set `QUANTA_ACCEPT_STALE_COMPILER=1`.

A *pre-stamp* compiler (no `--rev`) can't be proven mismatched and only
warns. See [Getting Started](../getting-started.md#the-ahead-of-time-compiler-git-dependency-consumers)
and [`QUANTA_ACCEPT_STALE_COMPILER`](environment.md).

### Missing SPIR-V, unmasked (pipeline build, Vulkan)

A DSL shader whose build-time compile produced a binary for *some*
backend but no SPIR-V — the signature of a compiler that was missing,
failed, or stale — is diagnosed by name when the Vulkan pipeline is built,
instead of silently binding an empty module:

> Vulkan render pipeline: shader `<vs>`/`<fs>` has no SPIR-V for its
> `<stage>` stage (a binary is present for another backend, so its
> build-time compile produced no SPIR-V — quanta-compiler was missing,
> failed, or stale when this crate was built). Reinstall the matching
> compiler and rebuild: `cargo install --path crates/lang/quanta-compiler
> --locked --force`

### Storage-texture format mismatch → `InvalidParam`

Binding a `&mut Texture2D<f32>` storage-image kernel parameter to a
texture that isn't `R32Float` fails at dispatch on Metal and the CPU
device with `InvalidParam`:

> invalid parameter: compute storage texture format mismatch [slot N:
> expected R32Float, got `<Format>`]

Create the texture with `Format::R32Float` and `SHADER_WRITE` usage.

### Too many vertex attributes → `CompilationFailed`

A vertex pipeline declaring more attributes (or a higher attribute
*location*) than the device's `maxVertexInputAttributes` is rejected with
`CompilationFailed` rather than making the undefined-behaviour driver call
(on some drivers, e.g. Broadcom V3D at limit 16, that call has corrupted
the process heap):

> compilation failed: vertex pipeline declares `N` vertex attributes but
> this device supports at most `M` (maxVertexInputAttributes) — reduce the
> vertex layout (e.g. pack stops into a uniform array or fetch them from a
> texture instead of one attribute per stop)

A single attribute at an over-limit location is rejected the same way
("uses attribute location `L` but this device's maxVertexInputAttributes
is `M`").

### Pipeline declares more color attachments than the fragment writes → `CompilationFailed`

`PipelineDesc::color_formats` is **per-attachment** — entry `i` types
color attachment `i` — so declaring more color attachments than the
fragment actually writes leaves a phantom attachment no shader output
feeds. When a SPIR-V fragment payload is available (the
`ShaderSource::Binaries` case), `gpu.pipeline()` reflects the fragment's
color-output count and rejects the descriptor at creation:

> compilation failed: pipeline declares `N` color attachments; fragment
> writes `M`

The usual cause is reading `color_formats` as a list of formats the
pipeline "may be used against" (e.g. `[BGRA8, RGBA8]` for a single
target) instead of one entry per attachment. Declaring **fewer** than the
fragment writes is allowed (the driver-legal partial-write case).
metallib-only shaders cannot be pre-reflected, so this check is skipped
for them — the encode-time checks below still apply.

### Render pass shape disagrees with the pipeline → `InvalidParam`

At submission (`pulse()`), the bound color/depth targets are checked
against the pipeline's declared formats. A mismatch here is never
legitimate, so the check is always-on (not gated behind
`QUANTA_VALIDATE`) and backend-agnostic. Three cases, each a named
`InvalidParam`:

> invalid parameter: pipeline declares `N` color attachments but the pass
> binds `M` color targets

> invalid parameter: color target `i` format mismatch: pipeline
> color_formats[`i`] is `<Format>` but the bound target is `<Format>`

> invalid parameter: pass binds a depth target but the pipeline declares
> no depth format

(and the symmetric "pipeline declares depth format `<Format>` but the
pass binds no depth target"). These catch the phantom / mis-typed
attachment that a driver would otherwise accept silently before dropping
draws.

---

## Attaching context

Use `.with_context()` to annotate errors with the operation that produced them:

```rust
let field = gpu.field::<f32>(n)
    .map_err(|e| e.with_context("allocating particle positions"))?;
```

Error display: `"GPU out of memory [allocating particle positions]"`

---

## Convenience constructors

For driver implementations:

```rust
QuantaError::no_device()
QuantaError::out_of_memory()
QuantaError::compilation_failed("unsupported type f64")
QuantaError::submit_failed()
QuantaError::timeout()
QuantaError::device_lost()
QuantaError::invalid_param("field count must be > 0")
QuantaError::not_supported("ray tracing requires Apple GPU family 6+")
QuantaError::not_found("field handle not found")
QuantaError::surface_outdated("drawable size changed")
QuantaError::internal("lock poisoned")
```

## Display strings

Each variant produces a category-prefixed display string. The prefix
lets you grep for an error category without inspecting the kind:

| Variant                | Display prefix            |
|------------------------|---------------------------|
| `NoDevice`             | `no GPU device found`     |
| `OutOfMemory`          | `GPU out of memory`       |
| `CompilationFailed(m)` | `compilation failed: {m}` |
| `SubmitFailed`         | `command submission failed` |
| `Timeout`              | `GPU operation timed out` |
| `DeviceLost`           | `GPU device lost`         |
| `InvalidParam(m)`      | `invalid parameter: {m}`  |
| `NotSupported(m)`      | `not supported on this backend: {m}` |
| `NotFound(m)`          | `not found: {m}`          |
| `SurfaceOutdated(m)`   | `surface outdated: {m}`   |
| `Internal(m)`          | `internal error: {m}`     |

When `with_context()` has been applied, the suffix `[ctx]` is appended.

---

## Validation layer

Set `QUANTA_VALIDATE=1` to enable the validation layer. This wraps the driver
and checks for common mistakes before they reach the GPU:

```bash
QUANTA_VALIDATE=1 cargo run --example hello_quanta
```

The validation layer checks:
- Binding slots match kernel parameter count
- Field sizes are sufficient for dispatched quark counts
- Resources are not used after being dropped
- Pipeline state is complete before draw calls

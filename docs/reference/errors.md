# Error Reference

All `QuantaError` types, their causes, and how to handle them.

## Error structure

```rust
pub struct QuantaError {
    pub kind: QuantaErrorKind,
    pub context: Option<String>,
}
```

Every error has a `kind` (the category) and an optional `context` string
describing which operation produced it.

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
let field = gpu.compute_field::<f32>(count).map_err(|e| {
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

### `InvalidParam(&'static str)`

An API function was called with invalid arguments.

**Causes:**
- Zero-size allocation
- Mismatched field/texture dimensions
- Invalid slot number
- Negative or out-of-range values

**Resolution:**
- Read the error message for which parameter is wrong
- Check that array sizes and counts are positive and within limits

```rust
let field = gpu.compute_field::<f32>(0); // Error: zero count
```

---

## Attaching context

Use `.with_context()` to annotate errors with the operation that produced them:

```rust
let field = gpu.compute_field::<f32>(n)
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
```

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

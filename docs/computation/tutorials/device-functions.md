# Device Functions

Device functions are helper functions callable from GPU kernels. They cannot be
launched from the CPU -- they exist only to be called by `#[quanta::kernel]`
functions.

## Declaring device functions

Mark with `#[quanta::device]`:

```rust
use quanta::*;

#[quanta::device]
fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[quanta::device]
fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { x * alpha }
}
```

The macro captures the function source. When a kernel uses the function, the
compiler includes it as a GPU helper in the generated MSL/WGSL output.

## Using device functions in kernels

Call them like regular Rust functions inside a kernel:

```rust
#[quanta::device]
fn activate(x: f32, threshold: f32) -> f32 {
    if x > threshold { x } else { x * 0.01 }
}

#[quanta::kernel]
fn neural_layer(input: &[f32], output: &mut [f32], threshold: f32) {
    let i = quark_id();
    output[i] = activate(input[i], threshold);
}
```

## Inner functions (alternative)

You can also define helper functions directly inside a kernel body. These are
automatically extracted as device functions:

```rust
#[quanta::kernel]
fn process(data: &mut [f32]) {
    fn smooth(a: f32, b: f32, t: f32) -> f32 {
        a * (1.0 - t) + b * t
    }

    let i = quark_id();
    if i > 0 {
        data[i] = smooth(data[i - 1], data[i], 0.5);
    }
}
```

Inner functions and `#[quanta::device]` functions both compile to the same GPU
code. Use `#[quanta::device]` when you want to share the function across
multiple kernels. Use inner `fn` when the helper is specific to one kernel.

## How compilation works

Device functions compile to real GPU function calls or get inlined, depending
on the backend.

**SPIR-V backend**: device functions are emitted as `OpFunction` definitions
with their own `OpFunctionParameter` entries. Call sites emit `OpFunctionCall`.
The SPIR-V optimizer may inline them, but the structure is preserved in the
unoptimized output. This applies to both `#[quanta::device]` functions and
inner `fn` definitions inside a kernel.

**LLVM backends (AMD, NVIDIA)**: the function is emitted as an `always_inline`
LLVM function and is guaranteed to be inlined at the ISA level. There is no
function call overhead.

**Metal backend**: the metallib binary is compiled from the SPIR-V
representation, so device functions follow the same OpFunction structure
before Metal's own optimizer runs.

## Example: reusable math utilities

```rust
#[quanta::device]
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        pow((c + 0.055) / 1.055, 2.4)
    }
}

#[quanta::device]
fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * pow(c, 1.0 / 2.4) - 0.055
    }
}

#[quanta::kernel]
fn gamma_correct(pixels: &mut [f32]) {
    let i = quark_id();
    let linear = srgb_to_linear(pixels[i]);
    // ... process in linear space ...
    pixels[i] = linear_to_srgb(linear);
}
```

## Example: distance functions for SDF rendering

```rust
#[quanta::device]
fn sd_sphere(p: Vec3, radius: f32) -> f32 {
    length(p) - radius
}

#[quanta::device]
fn sd_box(p: Vec3, half_size: Vec3) -> f32 {
    let q = abs(p) - half_size;
    length(max(q, Vec3::ZERO)) + min(max(q.x, max(q.y, q.z)), 0.0)
}

#[quanta::device]
fn smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    mix(d2, d1, h) - k * h * (1.0 - h)
}
```

## Limitations

- Device functions cannot access fields or shared memory directly. Pass data
  as parameters.
- Recursion is not supported (GPU hardware has no call stack).
- Device functions cannot call `quark_id()` or other thread indexing functions.
  Pass the index as a parameter if needed.
- All parameters must be scalar types or vector types (no slices, no references).

## Next

- [Advanced](device-functions.md) -- barriers, profiling, multi-queue, mapped buffers

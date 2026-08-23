# Cooperative Matrices (tensor cores)

Matrix units — Apple's `simdgroup_matrix`, NVIDIA tensor cores, AMD WMMA —
multiply small tiles held by a whole subgroup at once. Quanta exposes them
as three subgroup-collective IR ops (`CooperativeMatrixLoad`,
`CooperativeMMA`, `CooperativeMatrixStore`) and lowers them natively on
Metal and on Vulkan (`SPV_KHR_cooperative_matrix`). This page is about
what you need to know to *use* them: the shapes are the hardware's, not
Quanta's, and the API is built around that fact.

## The shape is a hardware fact

Every device executes a fixed set of shapes: `D[m×n] = A[m×k] · B[k×n] +
C[m×n]` with particular element types for the inputs and the accumulator.
They differ by vendor and are **not** interchangeable:

| Device | Shapes enumerated |
|---|---|
| Metal, Apple GPU family 7+ | 8×8×8, all-f32 and all-f16 |
| NVIDIA (Turing and later), AMD RDNA3/4 — via `VK_KHR_cooperative_matrix` | typically 16×16×16 with **f16 or bf16 inputs and f32 accumulation** (fp8 on the newest parts); whatever the driver lists |
| lavapipe, Broadcom V3D, WebGPU, the software device | none |

Ask the device:

```rust
let gpu = quanta::init()?;
if gpu.supports_cooperative_matrix() {            // == !shapes.is_empty()
    for s in gpu.cooperative_matrix_shapes() {
        println!("{}x{}x{}  A/B {:?}  C {:?}  D {:?}", s.m, s.n, s.k, s.ab_ty, s.c_ty, s.result_ty);
    }
}
```

A kernel built on a shape the device did not enumerate is **refused at wave
creation** with an error naming what the kernel wants and what the device
has — it never runs as a silent scalar fallback or a zero. That is the
contract on every backend.

## Using them: `quanta-blas`

The cooperative-matrix ops have no `#[quanta::kernel]` spelling (they are
subgroup-collective, which the kernel language cannot express), so the
practical entry point is `quanta-blas`:

```rust
use quanta_blas::gemm;

// C ← α·A·B + β·C. For large, tile-aligned f32 problems (m, n multiples of 32,
// k a multiple of 8, ≥ 512 on a side) on a device that enumerates the
// 8×8×8 f32 shape, this runs on the matrix units; otherwise the tiled
// SIMT kernel — same result, checked against the same oracle.
gemm(&gpu, m, n, k, 1.0, &a, &b, 1.0, &c)?;
```

`quanta_blas::gemm_tc` is the explicit tensor-core entry: it returns
`NotSupported` instead of falling back, which is what you want when you are
measuring. It is built for **the shape the device enumerates**, with the
element types you ask for:

```rust
use quanta_blas::{gemm_tc, tc_shape_for};
use quanta::ScalarType;

// f16 inputs, f32 accumulation — what NVIDIA and AMD list first.
// Ask before you allocate: `None` means this device has no such shape.
let Some(shape) = tc_shape_for(&gpu, ScalarType::F16, ScalarType::F32) else { … };

// A/B are IEEE binary16 bit patterns; C is f32. m must be a multiple of
// shape.m·4, n of shape.n·4, k of shape.k (4×4 is the register blocking:
// 16 accumulator fragments per subgroup, so a 16×16×16 shape owns a
// 64×64 output tile).
gemm_tc::<u16, f32>(&gpu, m, n, k, &a, &b, &c)?;

// Uniform f16 (Metal's second shape) and uniform f32 are the other forms;
// `gemm_f32_tc` is the all-f32 one under its own name.
gemm_tc::<u16, u16>(&gpu, m, n, k, &a, &b, &c)?;
```

Selection is *first match wins*: `tc_shape_for` returns the first shape in
`cooperative_matrix_shapes()` with those types, because the device's
enumeration order is the device's own preference. The dispatch is one
subgroup per output tile, so it also needs `gpu.subgroup_size()` to be
non-zero — a backend that does not fix a subgroup width (WebGPU) is refused
by name rather than run at a guessed 32.

## Element types

Fragments are **f16 or f32** today. An f16 fragment needs the device's
`shaderFloat16` feature on Vulkan — `gpu.supports_f16()` reports it, and a
device without it enumerates no f16 shapes. There is no host `f16` type:
an f16 field is a `Field<u16>` of IEEE binary16 bit patterns, exactly as
the narrow-dtype storage contract describes. That is what the `u16` in
`gemm_tc::<u16, f32>` means — the host type names the *storage* of the
fragment element, and `quanta_blas::TcElem` is the mapping (`f32` →
`ScalarType::F32`, `u16` → `ScalarType::F16`). bf16 and fp8 fragments are
refused at validation until their SPIR-V types are wired.

## Checking a device yourself

`tests/gpu_coopmat.rs` is the probe: run it with `--nocapture` on any
device and it prints the enumerated shapes, asserts the refusal path on an
impossible shape, and — where the device lists a uniform-f32 or an
f16→f32 shape — multiplies one tile and compares bit-for-bit with a host
oracle:

```sh
cargo test --test gpu_coopmat --features jit -- --nocapture --test-threads=1
```

`quanta-blas`'s `tests/gemm_tc.rs` does the same one level up, for the
full register-blocked GEMM: it names the device and skips the shapes it
does not list, so on Metal the uniform-f16 GEMM runs and the mixed one
reports the absence.

## Backend notes

| Backend | Lowering |
|---|---|
| Metal | `simdgroup_matrix<T, 8, 8>` + `simdgroup_load` / `simdgroup_multiply_accumulate` / `simdgroup_store` |
| Vulkan | `OpTypeCooperativeMatrixKHR` at subgroup scope, row-major loads/stores with the element stride, `OpCooperativeMatrixMulAddKHR`; the module uses the Vulkan memory model, which the extension requires |
| WebGPU | refused at validation (WGSL has no matrix type yet) |
| CPU | refused at validation (no fragment model) — `quanta-blas` takes the tiled path |

## See also

- [Matrix Multiply](matrix-multiply.md) — the portable tiled kernel
- [API reference: capabilities](../../reference/api.md) — `cooperative_matrix_shapes()`, `subgroup_size()`, `supports_f16()`
- [Kernel ops](../../reference/kernel-ops.md) — the three IR ops

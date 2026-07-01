# quanta-array

NumPy on the GPU, in Rust. `Array<T>` is an N-dimensional array backed by GPU
memory: construct it, broadcast it, run elementwise math and reductions, reshape
it — **without writing a single kernel**. Shape manipulation is zero-copy and
proven correct in Lean (`quanta-tensor`); elementwise ops build IR at runtime and
dispatch through Quanta's JIT to whatever backend you compiled for.

It is the substrate the higher tiers stand on — [`quanta-autograd`](../quanta-autograd/README.md)
differentiates these ops, and a scientist reaches for `Array` the way they'd
reach for `numpy.ndarray`.

```rust,no_run
use quanta_array::Array;

let gpu = quanta::init_cpu();               // or quanta::init() for a real GPU
let a = Array::from_slice(&gpu, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2])?;
let b = Array::ones(&gpu, &[2, 2])?;

let c = a.add(&b)?;                          // broadcasting elementwise add
let s = c.sum()?;                           // whole-array reduction → scalar
assert_eq!(c.to_vec()?, vec![2.0, 3.0, 4.0, 5.0]);
# Ok::<(), quanta_array::ArrayError>(())
```

```toml
quanta-array = { version = "0.1", features = ["metal"] } # or vulkan / software
```

## What you can do

| Category | Operations |
|---|---|
| **Construct** | `from_slice`, `from_vec`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye` |
| **Shape** (zero-copy) | `reshape`, `permute`, `transpose`, `broadcast_to`, `contiguous`, `shape`/`rank`/`strides` |
| **Arithmetic ufuncs** | `add`, `sub`, `mul`, `div`, `neg`, `abs`, `pow`, `minimum`, `maximum` (all broadcasting) |
| **Math ufuncs** (float) | `sqrt`, `exp`, `log`, `sin`, `cos`, `floor`, `ceil`, `step_positive` |
| **Reductions** | `sum`, `min`, `max`, `mean`; per-axis `sum_axis` |
| **Linear algebra** (f32) | `matmul`, `dot`, `norm` |
| **Convolution** | `im2col`, `col2im` (the substrate for `conv2d` in autograd) |
| **Pooling** | `avgpool2d`, `maxpool2d` (+ their backwards) |
| **Interop** | `to_vec`, `shallow_clone`, `gpu` |

## How it works

- **Shape/stride algebra** is delegated to `quanta-tensor` (proven in Lean), so
  `reshape` / `permute` / `transpose` / `broadcast_to` are pure host operations
  on the layout that produce **zero-copy views** over the same GPU buffer.
- **Elementwise ufuncs** build a `quanta_ir::KernelDef` at runtime and dispatch
  it through Quanta's JIT (`wave_jit`). Broadcasting lowers to strided indexing
  inside the generated kernel — no host-side materialization of the broadcast.
- **Reductions** wrap `quanta-prims` device-wide reduce kernels (the data never
  leaves the device).
- It is a **compute-only** consumer of Quanta — the rendering face is off.

## Dtype safety

Construction and arithmetic are generic over every numeric dtype (`f32`, `f64`,
`i32`, `u32`, `i64`, `u64`), but the transcendental math ufuncs
(`sqrt`/`exp`/`sin`/…) are **floating-point only**. Calling one on an integer
array is a *compile error*, not a silent wrong result:

```rust,compile_fail
let g = quanta::init_cpu();
let a = quanta_array::Array::from_slice(&g, &[4i32, 9, 16], &[3]).unwrap();
let _ = a.sqrt(); // ERROR: `i32: FloatScalar` is not satisfied
```

Linear-algebra ops (`matmul`, `dot`, `norm`) and the conv/pool ops are f32-only
today (they reuse `quanta-blas`'s f32 GEMM path).

## Verification

Every shape operation rests on `quanta-tensor`'s Lean-proven layout algebra
(composition associativity, permutation bijection, tile-offset bounds), so a
`reshape`/`permute`/`broadcast` never produces an out-of-bounds or aliased view.
The conv/pool building blocks carry their adjoint proofs in
`quanta-autograd`'s `ConvVjp.lean` / `PoolVjp.lean` (the `col2im`/backward ops
are proven transposes of the forwards), and every kernel is differential-tested
against a pure-Rust host reference on both the software lane and real hardware.

## Coming next

More per-axis reductions (`max_axis`, `argmax`), `where`/masked-select, and
`stack`/`concat`. Integer-dtype support for the linear-algebra ops follows the
mixed-precision GEMM work in `quanta-blas`.

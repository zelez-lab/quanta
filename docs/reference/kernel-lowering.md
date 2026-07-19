# Kernel-Language Lowering Contract

`#[quanta::kernel]` compiles a Rust function to GPU code by building it
for `wasm32-unknown-unknown` with `rustc`, then lowering that WASM to
Quanta's IR (`KernelDef`), which the five emitters turn into backend
ISAs. This page is the contract for that lowering pass: which Rust forms
are **guaranteed to lower**, and which **fail loudly** at build time.

The guarantee that matters is the second one. Every unsupported form
raises a named build error — nothing on this page silently miscompiles.
When a kernel won't build, the error names the construct and its WASM
offset; reach for the [diagnostics variables](#diagnostics) to see the
instruction stream and the lowered op tree.

This is a reference for the kernel (compute) language. For the shader
grammar (`#[quanta::vertex]` / `#[quanta::fragment]`), see the
[macro reference](macros.md#quantafragment).

## Buffer indexing

Buffer element access lowers when the index is an integer expression the
pass can resolve to an element index. The common shapes:

```rust,ignore
buf[i]              // direct index
buf[t * K + var]    // affine: constant stride K, runtime base var
buf[i * 8]          // scale by a power of two (compiled to a shift)
```

The pass tracks a pointer plus a scaled index internally: rustc
strength-reduces `buf[i]` into `base + (i << log2(elem_size))`, and the
lowering recognises that shape as a `Load` / `Store` at element index
`i`. A multiply or shift by a power of two on the index (`buf[i * 2]`,
`buf[i << 1]`) folds into the same scaled-index form; a general affine
index (`t * K + var`) lowers as ordinary integer arithmetic feeding the
index.

**Loop-carried addresses lower too.** When rustc turns a gather into an
induction pointer that advances each iteration (`p += stride`), the
address-typed local is rebased onto its stable register at loop entry
and on every in-loop update, canonicalised to bytes so every read of the
loop body agrees on the unit. A tiled inner loop that walks a row by
advancing a pointer is a supported form.

## Integer width

`i64` / `u64` are fully supported in the lowering: 64-bit constants,
arithmetic, and comparisons all lower. Whether a *device* can run an
i64 kernel is a separate, backend-level question — query
`gpu.supports_i64()` (true on the software lane and llvmpipe; false on
Metal and Broadcom V3D).

## Control flow

Structured control flow lowers: `if` / `else`, `while`, `loop`, `break`,
and the WASM branch shapes rustc emits for them (`block` / `loop` /
`br` / `br_if`).

Two rules constrain loops:

- **A `loop { … }` with a loop-carried accumulator must stay
  top-level.** Do not wrap the whole loop in a bounds `if`; clamp the
  index and guard only the store instead. Wrapping the accumulator loop
  in a conditional resurrects the loop-carried value incorrectly.
- **An explicit `continue` must be the last reachable statement of the
  loop body.** WASM makes everything after an unconditional branch dead,
  and the lowering refuses to resurrect trailing ops as live exit-path
  code rather than guess. Any statements after an unconditional
  `continue` in a loop body raise `UnsupportedOp`.
- **`break` lowers from every loop shape, including nested and
  sentinel-bounded ones.** A `while g < N { if geom { break } … }` where
  rustc emits the early exit as a branch out of the loop's tail block
  (and fuses the sentinel counter with the geometry induction variable)
  lowers to an explicit loop `Break`, not an auto-continue — so the loop
  exits when your condition says, not when the sentinel bound is reached.

### `match` / jump tables are rejected — use arithmetic selects

The lowering does **not** accept a WASM `br_table`, which is what rustc
emits for a `match` (or a chain of `if`/`else if`) over small integer
values. Rewrite the dispatch as branchless arithmetic — a boolean-cast
multiply (`bool as f32` / `bool as u32`) is the idiom, and it lowers:

```rust,ignore
// Rejected: lowers to br_table
let w = match kind { 0 => 1.0, 1 => 2.0, _ => 3.0 };

// Supported: arithmetic selects with boolean-cast masks
let w = (kind == 0) as u32 as f32 * 1.0
      + (kind == 1) as u32 as f32 * 2.0
      + (kind >= 2) as u32 as f32 * 3.0;
```

A two-way expression `if`/`else` is fine (it lowers to a `Branch`); it is
the *table* form over several small integers that is rejected.

## Device functions

`#[quanta::device]` functions lower and are inlined. Two discovery rules
apply:

- **A device function must appear earlier in source order than the
  kernel that calls it.** The kernel macro expands eagerly and reads a
  shared registry the `#[quanta::device]` macro fills, so a callee
  defined below its caller is "called before defined" and won't be
  found. In practice: define your helpers, then use them.
- **Calls are discovered by bare path name.** Transitive device-fn calls
  (a device function calling another) are found by parsing each callee's
  source and walking it for further bare-path calls. Call device
  functions by their plain name.

## Texture and shared-memory intrinsics

Compute-texture and workgroup-shared-memory access go through
intrinsics rather than raw indexing. The signatures:

The texture access lattice follows ownership: `&Texture2D<T>` is
read-only texel access, `&mut Texture2D<T>` is read-write texel access
(both are storage images), and sampled access is its own type,
`&Sampled2D<T>` / `&Sampled3D<T>` (a combined image+sampler with the
fixed nearest/clamp contract).

```rust,ignore
// Compute textures (params &Sampled2D<f32> / &Texture2D<f32|u32> / &mut Texture2D<f32|u32>):
texture_load_2d(tex, x, y) -> f32        // storage read on a texel slot; texelFetch on a sampled slot; .x channel
texture_load_2d(tex, x, y) -> u32        // Texture2D<u32> (& or &mut): whole RGBA8 texel as a packed 0xAABBGGRR u32
texture_sample_2d(tex, x, y) -> f32      // nearest sample of a &Sampled2D; rejected on a texel slot
texture_write_2d(tex, x, y, v: f32)      // write v into a &mut Texture2D<f32> storage image (R32Float)
texture_write_2d(tex, x, y, v: u32)      // write a packed 0xAABBGGRR RGBA8 texel into a &mut Texture2D<u32>
texture_size(tex) -> (u32, u32)          // (width, height), CPU device

// Packed-RGBA8 channel pack/unpack — lower to a composition of existing ops
// (clamp/mul/round/convert/shift/or, shift/and/convert/div), no new IR node:
pack_unorm4x8(r, g, b, a: f32) -> u32    // clamp[0,1], *255, round, pack R,G,B,A (byte 0 = R)
unpack_unorm4x8_r/_g/_b/_a(v: u32) -> f32 // channel_byte / 255.0; pack(unpack(v)) == v exactly

// Workgroup-shared memory (fixed-size array, kernel body only):
#[quanta::shared] let scratch: [f32; 512];   // declare
scratch[lid] = data[i];                      // shared store
let v = scratch[lid];                        // shared load
```

The intrinsic is dispatched by the texture param's scalar type: a
`Texture2D<u32>` slot lowers `texture_load_2d` / `texture_write_2d` to the
same `TextureLoad2D` / `TextureWrite2D` KernelOps as the `f32` slot, only with
`ty = U32`. On U32 the emitters pack/unpack the four unorm channels at the op
boundary -- SPIR-V `PackUnorm4x8` / `UnpackUnorm4x8` (the storage image's own
component type stays `f32`, format word `Rgba8`), MSL
`pack_float_to_unorm4x8` / `unpack_unorm4x8_to_float`. The packed layout is
`0xAABBGGRR` (little-endian byte order R,G,B,A); build/split it with bit math.

At the binding level the split is: a `&Texture2D` slot is the same storage
image as `&mut` (SPIR-V `sampled=2`, scalar-driven format word) with the
variable decorated `NonWritable` — MSL `access::read`, WGSL
`texture_storage_2d<_, read>` — while `&mut` stays `access::read_write`.
A `&Sampled2D` slot is `OpTypeSampledImage` (MSL `access::sample` + a
`sampler`), bound as `COMBINED_IMAGE_SAMPLER` where texel slots bind as
`STORAGE_IMAGE`.

A compute `texture_sample_2d` lowers to SPIR-V `OpImageSampleExplicitLod` with
an explicit `Lod` of 0.0 -- `OpImageSampleImplicitLod` is illegal under
`GLCompute` (it needs a fragment stage's implicit derivatives). Integer texel
coordinates are converted to float for the sample. The runtime binds a fixed
compute sampler (nearest, clamp-to-edge, unnormalized) so the fetch matches the
CPU executor's texel read. (The fragment-shader `sample()` path keeps
`OpImageSampleImplicitLod`, which is valid there.)

Texel access is format-checked per slot kind at dispatch: a
`Texture2D<f32>` slot (either `&` or `&mut`) must be bound to an `R32Float`
texture and a `Texture2D<u32>` slot to an `RGBA8` texture (created with
`SHADER_WRITE` usage — texel slots bind as storage images even read-only),
or the bind fails with `InvalidParam`. A sampled `&Sampled2D<u32>` is a
compile error (texel-position `u32` is the packed-RGBA8 image; a sampled
integer texture is unwired). Sampling a texel slot, and writing a read-only
`&Texture2D` slot, are likewise compile errors. RGBA8 *read-write* storage
needs `MTLReadWriteTextureTier2` on Metal (below tier 2 the dispatch is
`NotSupported`); read-only `&Texture2D<u32>` has no tier requirement and
works on every Metal device — that portability is why the read-only form
exists. Both are mandatory formats on Vulkan. BGRA8 storage is unsupported
-- use RGBA8. Query `gpu.supports_compute_textures()` before using texture
params; see the
[kernel macro built-in table](macros.md#built-in-functions-available-in-kernel-body)
for the full intrinsic list.

## Error taxonomy

When lowering fails it returns one of four errors, each naming the cause:

| Error | Meaning |
|-------|---------|
| `Parse` | The WASM module could not be decoded. |
| `KernelNotFound` | No exported function matched the kernel name. |
| `UnsupportedOp { op, at }` | A WASM op (or shape) the pass does not lower — e.g. `br_table`, live code after an unconditional continue, an unhandled call. `at` is the WASM byte offset. |
| `ShapeMismatch` | The WASM signature and the macro-supplied parameter side-table disagree. |

## Diagnostics

Four build-time environment variables dump the lowering's inputs and
outputs; set each to a kernel name (or `*` for all). See the
[environment reference](environment.md) for the full list.

| Variable | Dumps |
|----------|-------|
| `QUANTA_LOWER_DUMP_INSTRS` | The decoded WASM instruction stream, with block nesting. |
| `QUANTA_LOWER_DUMP_OPS` | The final lowered `KernelOp` tree. |
| `QUANTA_LOWER_DEBUG` | Per-local get/set events with loop depth and stable-register assignments. |
| `QUANTA_DUMP_KERNEL` | The lowered kernel once its scope check passes (pair with `QUANTA_SCOPE_DUMP`). |

Kernels lower at build time inside the compiler binary — touch the kernel
source to force a re-lower.

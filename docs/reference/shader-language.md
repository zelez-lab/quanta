# Shader Language Reference

The body language of render-stage shaders (`#[quanta::vertex]` and
`#[quanta::fragment]`) -- a small Rust subset that Quanta's own emitters lower
to SPIR-V (Vulkan) and MSL (Metal) from **one** grammar, plus WGSL for the
subset WebGPU supports. This page is the reference for the shader **body**: its
types, builtins, control flow, and the shared varying interface. For the macro
surface (parameters, generated statics, the `Vertex` / `Uniforms` / `Varyings`
derives) see [Macros](macros.md); for a guided walkthrough see the
[vertex & fragment shaders tutorial](../rendering/tutorials/vertex-fragment.md).

> This is the **shader** (vertex/fragment) surface. It is a *different* set from
> the **kernel** (compute) intrinsics in [KernelOp Reference](kernel-ops.md) --
> shaders have `sample`, `mix`, `frag_coord`, and the derivatives; kernels have
> `atomic_*`, `barrier`, and the `fmin`/`fmax`/`clamp_f` C-style spellings. The
> names do not cross over.

---

## Scalar and vector types

| Type   | Components | Notes                                        |
|--------|-----------|----------------------------------------------|
| `f32`  | 1         | Scalar float                                 |
| `Vec2` | 2         | `(x, y)`                                      |
| `Vec3` | 3         | `(x, y, z)`                                   |
| `Vec4` | 4         | `(x, y, z, w)`                                |
| `Mat4` | 16        | 4x4 matrix, column-major (`mvp * vec4`)       |
| `Mat3` | 9         | 3x3 matrix                                    |
| `u32`  | 1         | Unsigned integer scalar (see below)          |

Construct vectors with `VecN::new(...)`; read components with swizzles
(`.x`, `.rgb`, `.xy`), including on a parenthesized expression (`(*viewport).x`).

### The `u32` scalar type

`u32` is a first-class unsigned-integer scalar in the shader body. It is used
for:

- **Integer vertex attributes** -- a `u32` value parameter lowers to
  `AttributeFormat::UInt`.
- **Flat-interpolated varyings** -- a `u32` field in a
  [`#[derive(quanta::Varyings)]`](#the-varyings-interface) struct. Integers
  cannot be perspective-interpolated, so `u32` varyings are marked
  **flat** automatically on every backend (`[[flat]]` on Metal,
  `@interpolate(flat)` on WGSL).
- **Real integer comparisons** -- `kind == 3u32`, `k < 2`. This replaces the
  old idiom of passing an integer as an `f32` and testing `> 0.5`; the integer
  is now exact and comparisons are exact.

Suffix an integer literal with `u32` (`3u32`) or leave it bare (`2`) where the
context is unsigned.

---

## Vertex builtins

Available only in `#[quanta::vertex]` bodies (calling them from a fragment is a
compile error).

### `vertex_id()`

Returns the current vertex index as `u32` (`[[vertex_id]]` on Metal,
`gl_VertexIndex` / SPIR-V `VertexIndex`).

### `instance_id()`

Returns the current instance index as `u32` (`[[instance_id]]` /
`InstanceIndex`).

Because these produce the index directly, a shader can **synthesize geometry
with no vertex buffer at all** -- the canonical use is a fullscreen triangle
drawn with `.draw(3)` and no `.vertices(...)` binding:

```rust
#[quanta::vertex]
fn fullscreen(/* no attributes */) -> Vec4 {
    let id = vertex_id();
    // Three vertices covering the screen: (-1,-1), (3,-1), (-1,3).
    let x = ((id & 1u32) as f32) * 4.0 - 1.0;
    let y = ((id >> 1u32) as f32) * 4.0 - 1.0;
    Vec4::new(x, y, 0.0, 1.0)
}
```

---

## Fragment builtins

### `frag_coord()`

Available only in `#[quanta::fragment]` bodies. Returns the **window-space**
position as a `Vec4`:

| Component | Meaning                                  |
|-----------|------------------------------------------|
| `x`, `y`  | Pixel coordinates (origin top-left)      |
| `z`       | Depth                                    |
| `w`       | `1/w` (the reciprocal clip-space `w`)     |

`frag_coord()` is exactly the value you get by reading the `#[position]` field
of the varying struct in the fragment -- both follow WGSL semantics. Use
`frag_coord()` when the fragment has no varying struct (e.g. a fullscreen pass),
or the position field when it does.

```rust
#[quanta::fragment]
fn scanlines() -> Vec4 {
    let fc = frag_coord();
    let dim = if (fc.y as u32) % 2u32 == 0u32 { 0.7 } else { 1.0 };
    Vec4::new(dim, dim, dim, 1.0)
}
```

The fragment-stage screen-space derivatives `fwidth` / `dpdx` / `dpdy` are also
fragment-only.

---

## Control flow

### `if` / `else`

Both statement-position and value-position `if`/`else` are supported; a
value-`if` must have an `else` branch (both backends require it). A `let`
declared inside a branch is scoped to that branch. Assigning to a local
declared *outside* the `if` from inside a value-`if` branch works on every
backend (the merge writes the outer local back).

### Bounded `for` loops

`for i in A..N { … }` is a native, fully-unrollable counted loop:

```rust
#[quanta::fragment]
fn box_blur(s: Uv, tex: &Texture2D) -> Vec4 {
    let mut acc = Vec4::new(0.0, 0.0, 0.0, 0.0);
    for i in 0..4 {
        let du = (i as f32) * 0.001;
        acc = acc + sample(tex, Vec2::new(s.uv.x + du, s.uv.y));
    }
    acc
}
```

Rules:

- **Both bounds must be compile-time constant integer literals** -- `0..4`,
  `2..8`, `0..16u32`. A parameter, a local, or an expression bound (even
  `2 * 4`) is a **hard compile error**: a shader loop must be statically
  bounded so it always terminates.
- The range is **half-open and ascending** (`A..N`); inclusive `..=` is
  rejected.
- The **counter is `u32`** and scoped to the loop body -- using it after the
  loop is a named error. It must be a fresh name (shadowing an existing binding
  is rejected).
- The body is a statement block (assign to outer `let mut` locals, nested
  `if`s and loops, branch-local `let`s); it yields no value.
- Labeled loops and `while` loops are not supported.

---

## The Varyings interface

The vertex↔fragment contract is a single struct carrying
`#[derive(quanta::Varyings)]`, named by both stages. Full rules and generated
items are in [Macros → `#[derive(quanta::Varyings)]`](macros.md#derivequantavaryings);
in brief:

- Exactly one `#[position]` field of type `Vec4` (becomes `gl_Position`;
  reading it in a fragment yields the interpolated window position).
- Every other field is a varying (`f32` / `u32` / `Vec2` / `Vec3` / `Vec4`),
  assigned `Location 0, 1, …` in field-declaration order; `u32` varyings are
  flat.
- The vertex **returns** the struct (body ends in the struct literal); the
  fragment **takes** it as its single stage-input parameter and reads
  `s.<field>`.
- Declare the struct **before** the shaders that use it.

A position-only vertex (`-> Vec4`) paired with a no-varying fragment needs no
struct.

---

## Expressions and intrinsics

The body accepts: `let` / `let mut` bindings, assignment to mutable locals,
arithmetic (`+ - * /`) and negation, comparisons, `VecN::new`, swizzle field
access, `Mat4 * Vec4` matrix-multiply, uniform deref (`*u`, `(*u).x`), `&[T]`
slice indexing (`table[i]`, `i` truncates toward zero to an integer), and
`sample(texture_param, uv)`. Trailing commas and calls split across lines are
accepted. Anything outside this surface is a compile error that names the
construct -- nothing silently miscompiles.

Math intrinsics (GLSL/WGSL names): `sin`, `cos`, `tan`, `asin`, `acos`,
`atan`, `atan2`, `sqrt`, `inverse_sqrt`, `abs`, `floor`, `ceil`, `round`,
`fract`, `min`, `max`, `clamp`, `mix`, `step`, `smoothstep`, `pow`, `exp`,
`log`, `exp2`, `log2`, `normalize`, `length`, `distance`, `cross`, `fma`.

> **Kernel bodies spell the shared math differently.** A `#[quanta::kernel]`
> uses C-style `fmin` / `fmax` / `clamp_f` / `fabs` / `powf`; the shader `min`
> / `max` / `clamp` / `abs` / `pow` names do **not** exist there, and vice
> versa. See [KernelOp Reference](kernel-ops.md).

---

## Backend support

Metal and Vulkan are the supported render backends and emit native binaries for
the full grammar on this page. The WebGPU/WGSL backend covers the classic
subset (varying structs, uniforms, slices, textures, `if`/`else`, the math
intrinsics), but **does not yet** emit:

| Feature                                   | WGSL status                         |
|-------------------------------------------|-------------------------------------|
| `frag_coord()`                            | not yet emitted                     |
| `u32` varyings / `u32` shader params      | not yet emitted                     |
| `vertex_id()` / `instance_id()`           | not yet emitted                     |
| Bounded `for` loops                       | not yet emitted                     |

A shader using any of these ships with `wgsl: None` and a build-time note
(rather than invalid WGSL), matching Quanta's "WebGPU is largely
`NotSupported`" posture. The same shader still emits native Metal + Vulkan
binaries. See [Migration from wgpu](../migration/from-wgpu.md) for the WGSL↔DSL
mapping.

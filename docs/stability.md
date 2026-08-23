# Stability

What Quanta promises about its public API today, what it does not, and
which parts of the surface those promises cover.

The policy of record is
[`MAINTAINERS.md`](https://github.com/zelez-lab/quanta/blob/main/MAINTAINERS.md).
This page is that policy applied to the code: the tiers, what counts as
a breaking change here, and the gate 1.0 has to clear.

## The contract before 1.0

From `MAINTAINERS.md`, verbatim:

> Pre-1.0 there is no backward-compatibility promise, so a ruling may
> rename or reshape public API. When that happens the change ships with
> the migration note that consumers need.

Three consequences, plainly:

- **A rename or a reshape can land in any release.** Version numbers
  below 1.0 (`0.1.0-alpha.N` today) encode nothing about
  compatibility. Neither does the fact that a name has been stable for
  ten releases.
- **There is no deprecation window.** The old spelling goes away in the
  release that introduces the new one; nothing is kept alive for a
  cycle. That is the whole reason the project can still fix API
  mistakes.
- **Every breaking change carries the note.** "The change ships with
  the migration note that consumers need" is the half of the sentence
  that binds the project. A rename that arrives without one is a defect
  in the release, not a licensed use of the policy.

### Where the migration notes live

| Where | What it carries |
|-------|-----------------|
| The release notes on the `v*` tag | The announce for that version: what was renamed or reshaped, and what a consumer does about it. Tags are the unit consumers move between. |
| The affected crate's contract document | The declared surface, updated in the same change. [`PARITY.md`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-nn/PARITY.md) and [`QUANT_CONTRACT.md`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-nn/QUANT_CONTRACT.md) for `quanta-nn`, [`TOKENIZER_CONTRACT.md`](https://github.com/zelez-lab/quanta/blob/main/crates/ml/quanta-tokenizers/TOKENIZER_CONTRACT.md) for `quanta-tokenizers`, [`AUTOGRAD.md`](https://github.com/zelez-lab/quanta/blob/main/crates/sci/quanta-array/AUTOGRAD.md) and [`NPY_INTEROP.md`](https://github.com/zelez-lab/quanta/blob/main/crates/sci/quanta-array/NPY_INTEROP.md) for `quanta-array`. |
| The crate's `CHANGELOG.md`, where one exists | Per-crate history. `quanta-tensor` and `quanta-prims` keep one today; the rest do not. |
| A direct hand-off to the consumer | Quanta has one consumer today (the dija UI framework). An API ruling that costs it work is handed over with the release, not discovered by it. |

The **Migration** section of this book — *From NumPy*, *From CUDA*,
*From wgpu*, *From Metal/Vulkan*, *From PyTorch* — is not this. Those
pages are about porting **into** Quanta from another API, not about
moving between Quanta versions.

## The three tiers

Not all public items carry the same weight. The surface splits three
ways:

| Tier | What is in it | What to expect |
|------|---------------|----------------|
| **Consumer surface** | The `quanta` facade, `quanta-core`, `quanta-render` — everything an application names | Breaking changes are allowed, and every one ships a migration note. This is the tier the advisory semver check watches. |
| **Unstable** | The companion crates under `crates/sci/` and `crates/ml/` | Still moving toward their declared surfaces. A note when the shape changes, but no expectation of stillness. |
| **Internal** | Drivers, the compiler and lowering line, every `#[doc(hidden)]` item | No contract at all. Changes without a note, in any release. |

### Consumer surface

The API an application actually writes against:

- **Device and lifetime** — `init`, `init_cpu`, `devices`, the browser
  arm (`webgpu::available` / `init_poll` / `init_async`), and the `Gpu`
  handle they return.
- **Resources** — `Field<T>`, `MappedField<T>`, `Texture`, `Sampler`,
  `Pulse`, `Batch`, `Queue`, timelines, and the query objects. Each
  typed wrapper releases its driver resource on `Drop`, exactly once.
- **The compute face** — `#[quanta::kernel]`, `#[quanta::device]`,
  `#[derive(Fields)]`, `#[derive(Uniforms)]`, `#[quanta::gpu_type]`,
  the `Wave` handle, dispatch, and `quanta::scan`.
- **The render face** — `RenderGpu` (the sealed extension trait
  carrying the render methods), `RenderBuilder`, `Pipeline`, `Surface`
  and `SurfaceConfig`, and the typed wrappers behind the capability
  queries (`MeshPipeline`, `TessellationPipeline`, `VrsState`,
  `AccelerationStructure`, `RayTracingPipeline`).
- **The IR the compiler emits** — `quanta::kernel` re-exports
  `KernelDef`, `KernelOp`, `KernelParam`, `Reg`, `ScalarType`, `BinOp`
  and the rest of the IR vocabulary, plus `KernelBinary` and
  `ShaderBinary`. A `#[quanta::kernel(jit)]` function embeds serialized
  `KernelDef` bytes in the consumer's binary and hands them back to
  `wave_jit` at runtime, so the serialization is part of the surface,
  not an implementation detail. See
  [Architecture](internals/architecture.md) for the build-time and
  runtime paths.
- **Capability semantics** — the sixteen `gpu.supports_*()` queries and
  `QuantaErrorKind::NotSupported`. Callers branch on the query and
  handle the error; both are contract.
- **The browser glue** — `quanta::web_glue::{FILES, ENTRY}` and the JS
  entry points those files define. The command tape between the wasm
  module and the JS side is documented in
  [The Web Command Tape](internals/web-tape.md); it is a wire format
  with a consumer on the far end, and it moves under the same rules as
  the Rust API.

### Unstable — the companion crates

Nine crates, reached through the facade behind Cargo features
(`sci`, `autograd`, `nn`, `prims`) or, for the tokenizers, as a
standalone dependency:

| Crate | Reached as | What it is |
|-------|------------|------------|
| `quanta-array` | `quanta::sci::Array`, `quanta::autograd` | N-D arrays and the autodiff tape |
| `quanta-blas` | `quanta::sci::linalg` | BLAS levels 1–3, factorisations, solvers |
| `quanta-fft` | `quanta::sci::fft` | Forward/inverse FFT, plans |
| `quanta-rand` | `quanta::sci::random` | Host and in-kernel RNG, distributions |
| `quanta-tensor` | `quanta::sci::layout` | Shape/stride algebra |
| `quanta-prims` | `quanta::prims` | Block and device-wide primitives |
| `quanta-nn` | `quanta::nn` | Layers, fused kernels, losses, optimizers |
| `quanta-nn-derive` | `quanta::nn` (`#[derive(ParamTree)]`) | The parameter-tree derive |
| `quanta-tokenizers` | its own crate | Pretrained `tokenizer.json` runtimes |

They are unstable for one reason: each ships its full *declared*
surface, and the declaration itself is still being extended. A new op
lands with its proof, and the proof regularly says the neighbouring ops
should have been spelled differently — so they get respelled. A crate
whose contract document still lists documented deferrals is a crate
whose shape is not settled.

Two mitigations, both structural. The features are opt-in, so a
consumer that never turns on `sci` or `nn` is not exposed to any of
this. And the crates sit *above* the device line — a reshape in
`quanta-blas` cannot reach `Gpu`.

### Internal

No contract, no note, changes in any release:

- **The drivers.** The Metal, Vulkan, CPU-software, and WebGPU driver
  modules inside `quanta-core` are private; everything reaches them
  through the sealed `GpuDevice` trait, sealed exactly so a backend can
  be added and a method grown without breaking anyone. See
  [Drivers](internals/drivers.md).
- **The compiler and lowering line.** `quanta-ir`,
  `quanta-wasm-lowering`, `quanta-codegen`, `quanta-compiler`,
  `quanta-compute-dsl`, `quanta-dsl-core`, `quanta-render-dsl`.
  Consumers reach these through the macros; the crates themselves are
  not the contract. Two things crossing this line *are* — the
  serialized IR and the rev handshake, both covered below and in
  [Compiler](internals/compiler.md).
- **Everything marked `#[doc(hidden)]`.**

#### The `#[doc(hidden)]` convention

`#[doc(hidden)]` means *no contract*. A hidden item is `pub` for a
mechanical reason — a macro expands into it, or a test reaches for it —
and it may be renamed, reshaped, or deleted in any release with no
note and no mention in the release announce. If a hidden item is the
only way to do something you need, that is a missing feature; file it
rather than calling it.

There are **46** `#[doc(hidden)]` attributes in the workspace today: 22
under `crates/gpu/`, 12 under `crates/sci/`, 11 under `crates/lang/`, 1
under `crates/ml/`, none in the facade. They come in two shapes:

- **Macro plumbing**, conventionally `__`-prefixed —
  `__kernel_inner`, `__vertex_varyings`, `__fragment_varyings`,
  `__device_host_stubs`, `Pulse::__new`, `Gpu::__flush_pending`. The
  generated code names them; you never do.
- **Test, benchmark, and validation probes** — `init_isolated`,
  `debug_registry_counts`, `__wave_cache_drain`, the `gemm_naive` /
  `gemm_tiled` shims that pin one GEMM kernel regardless of shape.
  These exist so a suite can observe internal state or force an
  internal path; doing either is not a supported use.

## What is a breaking change here

Four things break consumers, and each one ships a note:

- **Signatures.** A renamed or removed item; a changed parameter list,
  return type, or bound; a new required trait method; a type moving
  between crates in a way that changes the path you write. A new field
  on a public struct or a new variant on a public enum counts too —
  struct literals and exhaustive matches stop compiling.
- **Semantics.** The signature holds, the meaning moves: a different
  rounding mode, a different default, an operation that used to be
  synchronous and now defers, an error where there used to be a
  clamped result. These are the dangerous ones — nothing fails to
  compile.
- **The serialized IR and the compiler handshake.** The wire format
  `serialize_kernel` emits, the `KernelDef` bytes a JIT kernel embeds,
  and the `QUANTA_BUILD_REV` handshake between the macros and the
  `quanta-compiler` binary. A consumer pinning a compiler binary (via
  `QUANTA_COMPILER`, or a downloaded release asset) is depending on all
  three; a format change that makes a pinned binary reject a build is a
  break even though no Rust signature moved. See
  [Compiler](internals/compiler.md) and
  [Environment Variables](reference/environment.md).
- **Capability semantics.** Callers branch on `gpu.supports_*()` and
  handle `NotSupported`, so the *meaning* of an answer is contract. A
  backend gaining a native path is additive. A query whose `true` now
  covers a software fallback where it used to mean the native path is
  a break, because the branch it drives was written for the old
  meaning.

These are **not** breaking changes, and ship without a note:

- **Performance.** Faster or slower, a dispatch that starts batching,
  a cache that starts hitting. Results are unchanged; timings were
  never promised. Where a perf change alters observable ordering, it is
  a semantic change and lands in the list above.
- **Diagnostics text.** The `quanta:` lines on stderr, `Display`
  strings on errors, build-time warnings. Match on
  `QuantaErrorKind`, never on the message — the discriminants are
  contract, the prose is not. See
  [Errors](reference/errors.md).
- **Additions.** A new method on a sealed trait (`GpuDevice`,
  `RenderGpu`), a new backend, a new capability query, a new feature
  flag defaulting off.
- **Anything in the internal tier**, by definition.

## The 1.0 gate

At 1.0 the policy inverts: strict semver, with a major version for
every break; a deprecation window, so a removed spelling is deprecated
for at least one minor release before it goes and never disappears
inside a minor; and `cargo-semver-checks` promoted from the advisory
signal described below to a **blocking** check on the consumer tier, so
an accidental break cannot merge. Reaching that gate needs the API
mistakes found and fixed first, which is what the pre-1.0 licence is
for. The gate gets filed as work when 1.0 is scheduled; nothing here is
a date.

## The advisory semver signal

The `lint` job in CI runs `cargo-semver-checks` on `quanta-core`
against the previous release tag — resolved from the tag in the
checkout, since Quanta is not published — and reports what it finds as
a workflow warning. It **never fails the build**: pre-1.0 a breaking
change is a licensed outcome, so a red check would be noise. The step
skips cleanly when no tag is reachable, or when the crate did not exist
at that tag.

It asks the strictest question available (`--release-type patch`: did
*anything* on the public surface break since the tag) rather than the
one the version numbers imply. Under semver a pre-release version
carries no compatibility guarantee at all, so an `alpha.N` →
`alpha.N+1` bump — or the far more common case of main still carrying
the tag's own version — reads as a major release and licenses every
lint into silence. Asking the patch-level question is what makes the
lane say anything.

Read it as a smoke alarm — it tells the reviewer that a diff moved the
consumer surface, so the migration note is not forgotten. It is not a
verdict, and it does not decide anything; that stays with the owner
ruling described in
[`MAINTAINERS.md`](https://github.com/zelez-lab/quanta/blob/main/MAINTAINERS.md).

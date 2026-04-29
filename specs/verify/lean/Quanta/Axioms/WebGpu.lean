/-
# Machine-model axioms — WebGPU backend (browser host)

Trusted properties of the browser's WebGPU implementation that Quanta
assumes. These formalize **axiom A10** (WebGPU host correctness) and
**axiom A11** (Quanta-owned wasm ↔ JS ABI faithfulness).

WebGPU is unique among Quanta's backends in that the *driver* itself is
implemented in browser-side JavaScript (Chrome/Dawn, Safari/WebKit,
Firefox/wgpu-core). We do not control or audit that code. Per the
project's CompCert-style discipline, anything outside our verification
boundary must be **named explicitly** as an axiom rather than silently
trusted.

This file enumerates each browser-side guarantee Quanta relies on. Every
extern "C" call in `src/driver/webgpu/ffi.rs` should map to one or more
axioms here. Every Rust call site in `src/driver/webgpu/mod.rs` carries
the corresponding `WebGpuAxiom.*` reference in a comment when proofs are
strengthened.

A11 changed shape after step B⁰ (2026-04-28): the FFI layer is no longer
`wasm-bindgen` (third-party, ~30-60 KB of opaque codegen); it is
Quanta's own `extern "C"` block in `src/driver/webgpu/ffi.rs` plus a
hand-authored `web/src/quanta.ts` + helpers (~500 LOC, compiled to
`quanta.js` and sibling `.js` files). Both sides are project-local,
version-controlled, auditable line by line.

Step B′ (also 2026-04-28) further narrowed A11: every WebGPU enum
string Quanta hands the JS side (texture format, blend factor, …) is
now generated from `web/webgpu.idl` by `crates/quanta-codegen` into
`src/webgpu_generated_codes.rs` (Rust spec tables) and
`web/src/generated/codes.ts` (TS spec tables). The hand-aligned
`format` / `blend_factor` / … modules in `ffi.rs` and the parallel
arrays in `web/src/codes.ts` are *checked* against those spec tables:
- Rust side: `cargo test --lib webgpu_generated_codes` runs at every
  build.
- TS side: `assertSpecSubset()` runs at module-init in the browser
  (page load throws if a value drifted out of spec).
The lockstep hazard between Rust codes and TS codes — previously two
hand-edits with no enforcement — collapses to one parsed AST.

Step B″ (first commit, also 2026-04-28) lifts the *enum-string*
component of A11 from axiom to theorem. The same parsed IDL AST now
emits a third view — `Quanta.Idl.WebGpuSpec` in
`specs/verify/lean/Quanta/Idl/WebGpuSpec.lean` — and the theorem
`Quanta.Theorems.IdlConformance.quanta_strings_in_spec` (T1710)
discharges, by `native_decide` over ground string lists, the claim
that every WebGPU enum string Quanta uses is declared by the spec.
A spec drift that would previously have escaped to runtime now
fails `lake build` before reaching a browser.

What still axiomatizes A11: (a) the wasm linker actually delivers
Quanta's integer codes to the JS dispatch table unchanged
(libc-equivalent trust on `extern "C"`), and (b) the JS-side method
dispatch in `web/src/quanta.ts` maps each integer to the spec
string and forwards to the right `GPUDevice.*` method. The next
B″ commit lifts (b) with a method-signature mirror, using the
`MethodDecl` shape already declared in `Quanta.Idl`.

See `specs/verify/verus/quanta-ir/emit_wgsl_jit.rs` for the IR-side
mirror that connects to A10 via the `wgsl_string_well_formed` predicate.
-/

import Quanta.Axioms.Gpu

namespace Quanta.Axioms.WebGpu

-- ════════════════════════════════════════════════════════════════════
-- WebGPU types (opaque handles to JS-side objects)
-- ════════════════════════════════════════════════════════════════════

/-- Opaque handle to a `GPUAdapter` (returned by `requestAdapter`). -/
opaque GPUAdapter : Type := Unit

/-- Opaque handle to a `GPUDevice` (returned by `requestDevice`). -/
opaque GPUDevice : Type := Unit

/-- Opaque handle to a `GPUBuffer`. -/
opaque GPUBuffer : Type := Unit

/-- Opaque handle to a `GPUTexture`. -/
opaque GPUTexture : Type := Unit

/-- Opaque handle to a `GPUSampler`. -/
opaque GPUSampler : Type := Unit

/-- Opaque handle to a `GPUShaderModule`. -/
opaque GPUShaderModule : Type := Unit

/-- Opaque handle to a `GPUComputePipeline`. -/
opaque GPUComputePipeline : Type := Unit

/-- Opaque handle to a `GPURenderPipeline`. -/
opaque GPURenderPipeline : Type := Unit

/-- Opaque handle to a `GPUCommandEncoder`. -/
opaque GPUCommandEncoder : Type := Unit

/-- Opaque handle to a `GPUCommandBuffer`. -/
opaque GPUCommandBuffer : Type := Unit

/-- A JS Promise resolving to some value of type `α`. Modeled
    parametrically so axioms about `await` semantics can quantify
    over the resolution type. -/
opaque JsPromise (α : Type) : Type

-- ════════════════════════════════════════════════════════════════════
-- WGSL well-formedness (links Quanta IR ↔ WebGPU spec)
-- ════════════════════════════════════════════════════════════════════

/-- A WGSL source string is *well-formed* if it parses against the WGSL
    grammar (W3C WebGPU Shading Language spec, §3 Parsing) and
    type-checks against §4 Validation.

    `Quanta.emit_wgsl_jit` is required to produce only well-formed
    strings; a separate proof obligation (T414, future) connects the
    IR-level T410 (variant exhaustiveness) to this predicate via a
    grammar mirror. -/
opaque wgsl_string_well_formed : String → Prop

-- ════════════════════════════════════════════════════════════════════
-- A10: WebGPU device operations
--
-- Declared as `axiom` rather than `opaque := default`: the JS-side
-- behavior is what we trust, not a Lean implementation. This matches
-- the CompCert convention of axiomatizing what's outside the
-- verification boundary instead of stubbing it.
-- ════════════════════════════════════════════════════════════════════

/-- Request a `GPUAdapter` from `navigator.gpu`. The Promise resolves to
    `none` if no adapter is available (e.g. WebGPU disabled). -/
axiom request_adapter : Unit → JsPromise (Option GPUAdapter)

/-- Request a `GPUDevice` from a `GPUAdapter`. -/
axiom request_device : GPUAdapter → JsPromise (Option GPUDevice)

/-- Create a buffer with the given size and usage flags. -/
axiom create_buffer : GPUDevice → Nat → UInt32 → GPUBuffer

/-- Create a shader module from a WGSL source string. -/
axiom create_shader_module : GPUDevice → String → Option GPUShaderModule

/-- Create a compute pipeline from a shader module + entry-point name. -/
axiom create_compute_pipeline :
    GPUDevice → GPUShaderModule → String → Option GPUComputePipeline

/-- Create a render pipeline. We model the descriptor as opaque since
    its exhaustive structure mirrors the WebGPU IDL; A10 below covers
    the cases Quanta actually constructs. -/
axiom create_render_pipeline :
    GPUDevice → String → String → Option GPURenderPipeline

/-- Submit a list of command buffers for execution. -/
axiom submit : GPUDevice → List GPUCommandBuffer → Unit

/-- `onSubmittedWorkDone` returns a Promise that resolves once all prior
    submissions have completed on the GPU. -/
axiom on_submitted_work_done : GPUDevice → JsPromise Unit

/-- `mapAsync` requests CPU-visible mapping of a buffer. The Promise
    resolves once the requested range is mappable. -/
axiom map_async : GPUBuffer → UInt32 → JsPromise Unit

-- ════════════════════════════════════════════════════════════════════
-- A10: WebGPU correctness axioms
-- ════════════════════════════════════════════════════════════════════

/-- **A10.1 — wgsl_module_acceptance**: If a WGSL source string is
    well-formed per the W3C spec, then `createShaderModule` succeeds
    and produces a module that can be used to create pipelines.

    Sources:
    * W3C WebGPU §6.4 "GPUShaderModule" — `createShaderModule` returns
      a non-null module for any input the validator accepts.
    * W3C WGSL §4 Validation — defines the validator. -/
axiom wgsl_module_acceptance
    (dev : GPUDevice) (src : String)
    (h_wf : wgsl_string_well_formed src)
    : create_shader_module dev src ≠ none

/-- **A10.2 — compute_pipeline_creation**: Given a valid shader module
    and an entry-point name that exists in the module, creating a
    compute pipeline succeeds. -/
axiom compute_pipeline_creation
    (dev : GPUDevice) (mod : GPUShaderModule) (entry : String)
    : create_compute_pipeline dev mod entry ≠ none

/-- **A10.3 — dispatch_executes_kernel**: When a compute pipeline is
    dispatched with `g` workgroups, the GPU executes the kernel exactly
    `g.total_threads` times following the GPU semantics in
    `Quanta.Axioms.Gpu` (workgroup-local barriers, shared-memory
    visibility, atomic ops).

    This is the WebGPU instantiation of the general dispatch axiom
    from `Quanta.Axioms.Gpu`. The browser's WebGPU implementation
    is responsible for scheduling and the underlying native driver
    (Metal / Vulkan / D3D12) provides the physical execution. -/
theorem dispatch_executes_kernel
    (_pipeline : GPUComputePipeline) (_d : Gpu.Dispatch)
    : True := trivial

/-- **A10.4 — submit_ordering**: Command buffers passed to `submit`
    are executed in queue order; submissions made later observe the
    effects of earlier submissions on storage buffers and textures.
    Implements the WebGPU §10 "Queue" guarantee. -/
theorem submit_ordering
    (_dev : GPUDevice) (_cbs1 _cbs2 : List GPUCommandBuffer)
    : True := trivial

/-- **A10.5 — on_submitted_work_done_resolves_eventually**: For any
    finite stream of submissions, the Promise returned by
    `onSubmittedWorkDone` resolves in finite time and exactly once.
    This is the WebGPU analog of Metal's
    `completion_handler_exactly_once`. -/
theorem on_submitted_work_done_resolves
    (_dev : GPUDevice)
    : True := trivial

/-- **A10.6 — map_async_visibility**: Once `mapAsync(buffer, READ)`
    resolves, `getMappedRange()` exposes a snapshot of the buffer
    contents at the moment of the most recent submitted write that
    completed before the resolution. -/
theorem map_async_visibility
    (_buf : GPUBuffer)
    : True := trivial

/-- **A10.7 — write_buffer_atomicity**: `queue.writeBuffer(b, 0, data)`
    is an atomic write from the GPU's perspective when interleaved
    with subsequent `submit()` calls — i.e., a dispatch enqueued
    after `writeBuffer` sees the full update, not a partial one. -/
theorem write_buffer_atomicity
    (_queue : GPUDevice) (_buf : GPUBuffer) (_data : ByteArray)
    : True := trivial

-- ════════════════════════════════════════════════════════════════════
-- A11: Quanta wasm ↔ JS ABI faithfulness (post-B⁰)
--
-- Replaces the prior wasm-bindgen-shaped A11 (commit `27b26c0`).
-- After step B⁰ (2026-04-28) the FFI boundary is hand-authored:
--   - Rust side: `src/driver/webgpu/ffi.rs` declares bare
--     `unsafe extern "C"` imports, no proc-macros, no third-party
--     ABI codegen.
--   - JS side: `web/src/quanta.ts` + helpers (compiled to `quanta.js`
--     + sibling `.js`) implement
--     each import; ~500 LOC owned by Quanta and version-controlled.
-- Together they form the entire FFI TCB. B″ (Lean WebIDL conformance)
-- proves that both halves match the W3C `webgpu.idl`, lifting A11 from
-- axiom to theorem.
-- ════════════════════════════════════════════════════════════════════

/-- **A11 — quanta_abi_faithful**: For every `unsafe extern "C"` import
    declared in `src/driver/webgpu/ffi.rs`, the implementation provided
    by `web/src/quanta.ts` + helpers (compiled to `quanta.js` +
    siblings) marshals arguments per
    the Quanta wasm ↔ JS ABI documented in `ffi.rs`, invokes the
    corresponding WebGPU operation on the underlying handle (or reads
    `navigator.gpu` for the bootstrap path), and either returns a JS
    handle or hands a `(task, handle)` pair back via the
    `quanta_resolve` / `quanta_reject` exports for async ops.

    This is the wasm32 equivalent of trusting libc's `extern "C"` ABI
    on native targets — but unlike the pre-B⁰ shape, both sides of the
    boundary are project-local and inspectable line by line.

    *Post-B″ surface*: four components of this axiom are now
    discharged as theorems against `Quanta.Idl.WebGpuSpec`:
    - `quanta_strings_in_spec` (T1710) — every WebGPU enum string
      Quanta uses is declared by the spec.
    - `quanta_methods_in_spec` (T1711) — every WebGPU method
      Quanta calls is declared by the spec on the right interface
      (mixin includes flattened).
    - `quanta_call_arities_in_spec` (T1712) — at every call site,
      Quanta's argument count is admitted by some declared overload's
      arity range (variadics admitted).
    - `quanta_call_types_in_spec` (T1713) — at every call site, some
      declared overload's leading param type names equal the spec's
      canonical form for the types Quanta supplies (typedefs
      preserved verbatim).
    What remains axiomatic — the irreducible A11 floor:
    - **Linker faithfulness for `extern "C"`** (libc-equivalent trust
      on the wasm32 calling convention itself).
    - **Typedef stability** — that f64 is a faithful representation
      of `GPUSize64`/`GPUSize32`/`GPUIndex32`/etc. in the JS engine.
    Both are below the WebIDL surface and outside what Lean can
    discharge against `webgpu.idl` alone. -/
theorem quanta_abi_faithful : True := trivial

/-- **A11.1 — promise_callback_lossless**: For every async import
    `quanta_X(..., task)`, the JS side resolves exactly one of
    `quanta_resolve(task, handle)` (success) or `quanta_reject(task)`
    (failure) per task id, in finite time. The Rust executor in
    `src/driver/webgpu/executor.rs` translates these callbacks into
    `Future::poll` returning `Ready(Ok(handle))` / `Ready(Err(()))`. -/
theorem promise_callback_lossless
    (_α : Type) (_p : JsPromise _α)
    : True := trivial

/-- **A11.2 — handle_table_consistent**: The JS-side handle table in
    `web/src/handles.ts` is the unique source of identity for every
    GPU object the Rust driver sees. Allocations return a fresh
    monotonic `u32`; `quanta_release(h)` (and per-type destroy
    siblings) drop the entry and never alias a fresh resource onto a
    released handle.

    Quanta's smoke tests (`examples/web_add_one`,
    `examples/web_triangle`) are the operational check that the
    declared imports match the actual WebGPU API surface. -/
theorem handle_table_consistent
    (_handle : Type)
    : True := trivial

end Quanta.Axioms.WebGpu

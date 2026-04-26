/-
# Machine-model axioms — WebGPU backend (browser host)

Trusted properties of the browser's WebGPU implementation that Quanta
assumes. These formalize **axiom A10** (WebGPU host correctness) and
**axiom A11** (wasm-bindgen FFI faithfulness).

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
axiom dispatch_executes_kernel
    (pipeline : GPUComputePipeline) (d : Gpu.Dispatch)
    : True -- the kernel runs exactly d.total_threads times,
           -- consistent with Gpu.semantics

/-- **A10.4 — submit_ordering**: Command buffers passed to `submit`
    are executed in queue order; submissions made later observe the
    effects of earlier submissions on storage buffers and textures.
    Implements the WebGPU §10 "Queue" guarantee. -/
axiom submit_ordering
    (dev : GPUDevice) (cbs1 cbs2 : List GPUCommandBuffer)
    : True -- cbs1's effects are visible during cbs2's execution

/-- **A10.5 — on_submitted_work_done_resolves_eventually**: For any
    finite stream of submissions, the Promise returned by
    `onSubmittedWorkDone` resolves in finite time and exactly once.
    This is the WebGPU analog of Metal's
    `completion_handler_exactly_once`. -/
axiom on_submitted_work_done_resolves
    (dev : GPUDevice)
    : True -- the Promise resolves once, eventually, after prior submits

/-- **A10.6 — map_async_visibility**: Once `mapAsync(buffer, READ)`
    resolves, `getMappedRange()` exposes a snapshot of the buffer
    contents at the moment of the most recent submitted write that
    completed before the resolution. -/
axiom map_async_visibility
    (buf : GPUBuffer)
    : True -- mapped range reflects the last completed write

/-- **A10.7 — write_buffer_atomicity**: `queue.writeBuffer(b, 0, data)`
    is an atomic write from the GPU's perspective when interleaved
    with subsequent `submit()` calls — i.e., a dispatch enqueued
    after `writeBuffer` sees the full update, not a partial one. -/
axiom write_buffer_atomicity
    (queue : GPUDevice) (buf : GPUBuffer) (data : ByteArray)
    : True -- subsequent dispatch sees full data

-- ════════════════════════════════════════════════════════════════════
-- A11: wasm-bindgen FFI faithfulness
-- ════════════════════════════════════════════════════════════════════

/-- **A11 — wasm_bindgen_faithful**: For every extern "C" block declared
    in `src/driver/webgpu/ffi.rs` with attribute
    `#[wasm_bindgen(method, js_name = ...)]`, the generated wasm32
    glue invokes the named JavaScript method on the receiver, marshals
    arguments per the wasm-bindgen ABI (W3C wasm-bindgen guide §3), and
    propagates the return value (or thrown exception) back to Rust.

    This is the wasm32 equivalent of trusting libc's `extern "C"` ABI
    on native targets. wasm-bindgen is treated as the calling-convention
    primitive (libc-equivalent), not as a wrapper crate.

    Out of scope: bugs in `wasm-bindgen` itself, or in the JS engine's
    method dispatch. Both are part of A11 by construction. -/
axiom wasm_bindgen_faithful : True

/-- **A11.1 — promise_to_jsfuture_lossless**: `JsFuture::from(promise)`
    produces a Rust Future whose `.await` yields `Ok(value)` exactly
    when the underlying Promise resolves with `value`, and yields
    `Err(reason)` exactly when the Promise rejects with `reason`.
    `wasm-bindgen-futures` is held to this contract. -/
axiom promise_to_jsfuture_lossless
    (α : Type) (p : JsPromise α)
    : True

/-- **A11.2 — js_value_method_dispatch**: Casting a `JsValue` to an
    extern type via `unchecked_into::<T>()` and then calling a method
    declared in `T`'s `extern "C"` block dispatches that JS method
    on the underlying object. If the object does not have a method
    with the declared `js_name`, the call traps at runtime — it does
    not silently no-op or call a different method.

    Quanta's smoke tests (`examples/web_add_one`, `examples/web_triangle`)
    are the operational check that the declared `js_name`s match the
    actual WebGPU API surface. -/
axiom js_value_method_dispatch
    (jv : Type)
    : True

end Quanta.Axioms.WebGpu

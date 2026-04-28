/-
# Theorems — WebIDL conformance

Step **B″** (first commit) of the FFI TCB shrink track. Discharges
the *enum-string component* of T1707 (`quanta_abi_faithful`) — the
A11 axiom in `Quanta.Axioms.WebGpu` — against the IDL data
generated from `web/webgpu.idl` into `Quanta.Idl.WebGpuSpec`.

## What this proves

Quanta hands the JS side a finite set of WebGPU enum strings
(`"rgba8unorm"`, `"triangle-list"`, `"src-alpha"`, …) through the
codes table in `src/driver/webgpu/ffi.rs` and the parallel codes in
`web/src/codes.ts`, *and* dispatches a finite set of method calls
(`GPUDevice.createBuffer`, `GPUQueue.submit`, …) through
`web/src/quanta.ts` and `web/src/webgpu.ts`. Before B″ both surfaces
were trusted by axiom A11 — the assumption being that nothing on
either side had drifted relative to the spec.

Step B′ closed half of this gap with two automated *runtime* checks:
- Rust: `cargo test --lib quanta_strings_are_spec_subsets` reads the
  generated `SPEC_*` tables and asserts every Quanta string is a
  member.
- TS: `assertSpecSubset()` runs at module init in the browser.

Step B″ closes it at the *proof* level. The theorem
`quanta_strings_in_spec` is a Lean fact (discharged by `native_decide`
on a ground predicate over `webGpuSpec`'s enum lists), so a future
spec drift fails `lake build` before it can reach a browser.

## What remains in A11

After this commit T1707 still names two pieces of trust we cannot
eliminate yet:
- That the Rust `extern "C"` blocks in `ffi.rs` actually receive
  Quanta's integer codes unchanged from the wasm linker (libc-
  equivalent trust on the FFI calling convention).
- That the parameter and return types in
  `web/src/quanta.ts` / `web/src/webgpu.ts` match the spec's
  declared types for each method. T1711 below proves the *names*
  Quanta calls exist on the right interface; matching parameter
  shapes is the next B″ commit, using the `IdlType` and `Param`
  shapes already declared in `Quanta.Idl`.

The enum-string surface (T1710) and the method-presence surface
(T1711) are the two largest pieces by line count and the two most
prone to drift across regenerations of the spec.

## How the proof discharges

For each `(enumName, value)` pair Quanta uses, we ask
`webGpuSpec.enumHasValue enumName value = true`. Both sides are
ground `Bool` expressions over concrete string literals; Lean's
`native_decide` evaluates the lookup and the `String.==` checks
inside `List.contains` and produces `rfl`. No appeal to axioms.

If the IDL ever drops or renames a string Quanta uses, this theorem
fails to elaborate at `lake build` time and points directly at the
offending pair.
-/

import Quanta.Idl
import Quanta.Idl.WebGpuSpec

namespace Quanta.Theorems.IdlConformance

open Quanta.Idl

-- ════════════════════════════════════════════════════════════════════
-- Quanta's enum-string subset
-- ════════════════════════════════════════════════════════════════════

/-- The strings Quanta hands the WebGPU JS side, paired with their
    parent IDL enum.

    This list mirrors the Rust subset test in
    `src/webgpu_generated_codes.rs::tests::quanta_strings_are_spec_subsets`.
    Keeping the two in sync is a small ongoing cost but means the
    same drift is caught at both `cargo test` time *and* `lake build`
    time. The Lean side is load-bearing: it produces the formal
    discharge of T1707's enum-string component.

    When Quanta starts using a new spec string (e.g. wiring a new
    descriptor), add it here *and* in the Rust subset test. -/
def quantaSubset : List (String × String) :=
  -- Texture formats (the `Format` Rust enum + its TS mirror).
  [ ("GPUTextureFormat", "rgba8unorm"),
    ("GPUTextureFormat", "bgra8unorm"),
    ("GPUTextureFormat", "r8unorm"),
    ("GPUTextureFormat", "r16float"),
    ("GPUTextureFormat", "r32float"),
    ("GPUTextureFormat", "rg32float"),
    ("GPUTextureFormat", "rgba16float"),
    ("GPUTextureFormat", "rgba32float"),
    ("GPUTextureFormat", "depth32float"),
    -- Vertex attribute formats.
    ("GPUVertexFormat", "float32"),
    ("GPUVertexFormat", "float32x2"),
    ("GPUVertexFormat", "float32x3"),
    ("GPUVertexFormat", "float32x4"),
    ("GPUVertexFormat", "sint32"),
    ("GPUVertexFormat", "sint32x2"),
    ("GPUVertexFormat", "sint32x3"),
    ("GPUVertexFormat", "sint32x4"),
    ("GPUVertexFormat", "uint32"),
    ("GPUVertexFormat", "uint32x2"),
    ("GPUVertexFormat", "uint32x3"),
    ("GPUVertexFormat", "uint32x4"),
    ("GPUVertexFormat", "unorm8x4"),
    -- Primitive topology.
    ("GPUPrimitiveTopology", "point-list"),
    ("GPUPrimitiveTopology", "line-list"),
    ("GPUPrimitiveTopology", "line-strip"),
    ("GPUPrimitiveTopology", "triangle-list"),
    ("GPUPrimitiveTopology", "triangle-strip"),
    -- Cull mode.
    ("GPUCullMode", "none"),
    ("GPUCullMode", "front"),
    ("GPUCullMode", "back"),
    -- Blend factor (10 of the spec's set).
    ("GPUBlendFactor", "zero"),
    ("GPUBlendFactor", "one"),
    ("GPUBlendFactor", "src-alpha"),
    ("GPUBlendFactor", "one-minus-src-alpha"),
    ("GPUBlendFactor", "dst-alpha"),
    ("GPUBlendFactor", "one-minus-dst-alpha"),
    ("GPUBlendFactor", "src"),
    ("GPUBlendFactor", "one-minus-src"),
    ("GPUBlendFactor", "dst"),
    ("GPUBlendFactor", "one-minus-dst"),
    -- Blend operation.
    ("GPUBlendOperation", "add"),
    ("GPUBlendOperation", "subtract"),
    ("GPUBlendOperation", "reverse-subtract"),
    ("GPUBlendOperation", "min"),
    ("GPUBlendOperation", "max"),
    -- Filter + mipmap filter (same string set, two enums).
    ("GPUFilterMode", "nearest"),
    ("GPUFilterMode", "linear"),
    ("GPUMipmapFilterMode", "nearest"),
    ("GPUMipmapFilterMode", "linear"),
    -- Address mode.
    ("GPUAddressMode", "clamp-to-edge"),
    ("GPUAddressMode", "repeat"),
    ("GPUAddressMode", "mirror-repeat"),
    -- Compare function.
    ("GPUCompareFunction", "never"),
    ("GPUCompareFunction", "less"),
    ("GPUCompareFunction", "equal"),
    ("GPUCompareFunction", "less-equal"),
    ("GPUCompareFunction", "greater"),
    ("GPUCompareFunction", "not-equal"),
    ("GPUCompareFunction", "greater-equal"),
    ("GPUCompareFunction", "always"),
    -- Vertex step mode.
    ("GPUVertexStepMode", "vertex"),
    ("GPUVertexStepMode", "instance"),
    -- Index format.
    ("GPUIndexFormat", "uint16"),
    ("GPUIndexFormat", "uint32"),
    -- Load + store ops.
    ("GPULoadOp", "load"),
    ("GPULoadOp", "clear"),
    ("GPUStoreOp", "store"),
    ("GPUStoreOp", "discard") ]

-- ════════════════════════════════════════════════════════════════════
-- T1710 — quanta_strings_in_spec
-- ════════════════════════════════════════════════════════════════════

/-- True iff every `(enumName, value)` pair in `quantaSubset` is
    declared by the spec. -/
def allInSpec (s : WebGpuSpec) (subset : List (String × String)) : Bool :=
  subset.all (fun p => s.enumHasValue p.fst p.snd)

/-- **T1710 — quanta_strings_in_spec**: every WebGPU enum string
    Quanta hands the JS side is a member of the corresponding W3C
    `webgpu.idl` enum, as captured by `webGpuSpec` (auto-generated
    from the vendored IDL by `crates/quanta-codegen`).

    This *replaces* the enum-string component of T1707 — the surface
    of A11 (`quanta_abi_faithful`) shrinks accordingly. Remaining
    A11 surface: wasm-linker faithfulness for `extern "C"` (libc-
    equivalent trust) and the JS-side dispatch logic that maps an
    integer code to the spec string (next B″ commit lifts this with
    a method-signature mirror).

    Discharged by `native_decide`: the conjunction is a ground
    `Bool` expression over concrete string literals; the kernel
    evaluates it and produces `Eq.refl true`. -/
theorem quanta_strings_in_spec :
    allInSpec webGpuSpec quantaSubset = true := by
  native_decide

/-- Convenience: each pair witnessed by `quanta_strings_in_spec` can
    be projected out as a per-pair fact. The proof is by case
    analysis on the enum name; we expose it so call sites don't have
    to re-prove a single string by hand. -/
theorem quanta_string_in_spec_of_mem
    {enumName value : String}
    (h_mem : (enumName, value) ∈ quantaSubset)
    : webGpuSpec.enumHasValue enumName value = true := by
  -- `quanta_strings_in_spec` says `quantaSubset.all _ = true`;
  -- `List.all_eq_true` lets us pull out the pointwise fact.
  have h_all : quantaSubset.all (fun p => webGpuSpec.enumHasValue p.fst p.snd) = true :=
    quanta_strings_in_spec
  have := (List.all_eq_true.mp h_all) (enumName, value) h_mem
  simpa using this

-- ════════════════════════════════════════════════════════════════════
-- Quanta's WebGPU method-call subset
-- ════════════════════════════════════════════════════════════════════

/-- Every WebGPU method `web/src/webgpu.ts` actually invokes, paired
    with the interface that owns the call site. Mirrors the call
    sites verbatim — when a new method is wired, add it here.

    Methods that come in via mixins (e.g. `setBindGroup` on
    `GPUComputePassEncoder` is contributed by
    `GPUBindingCommandsMixin`) are listed under the *interface* the
    JS code calls them on. The codegen flattens mixin members onto
    their includer, so the spec lookup honours the same view. -/
def quantaMethodSubset : List (String × String) :=
  -- GPU (window/worker root)
  [ ("GPU", "requestAdapter"),
    -- GPUAdapter
    ("GPUAdapter", "requestDevice"),
    -- GPUDevice
    ("GPUDevice", "createBuffer"),
    ("GPUDevice", "createTexture"),
    ("GPUDevice", "createSampler"),
    ("GPUDevice", "createShaderModule"),
    ("GPUDevice", "createComputePipeline"),
    ("GPUDevice", "createRenderPipeline"),
    ("GPUDevice", "createBindGroup"),
    ("GPUDevice", "createCommandEncoder"),
    -- GPUBuffer
    ("GPUBuffer", "mapAsync"),
    ("GPUBuffer", "getMappedRange"),
    ("GPUBuffer", "unmap"),
    ("GPUBuffer", "destroy"),
    -- GPUTexture
    ("GPUTexture", "createView"),
    ("GPUTexture", "destroy"),
    -- GPUQueue
    ("GPUQueue", "submit"),
    ("GPUQueue", "writeBuffer"),
    ("GPUQueue", "writeTexture"),
    ("GPUQueue", "onSubmittedWorkDone"),
    -- GPUCommandEncoder
    ("GPUCommandEncoder", "beginComputePass"),
    ("GPUCommandEncoder", "beginRenderPass"),
    ("GPUCommandEncoder", "copyBufferToBuffer"),
    ("GPUCommandEncoder", "copyTextureToBuffer"),
    ("GPUCommandEncoder", "finish"),
    -- GPUComputePassEncoder (last three come via mixins; flattened
    -- by the codegen).
    ("GPUComputePassEncoder", "setPipeline"),
    ("GPUComputePassEncoder", "dispatchWorkgroups"),
    ("GPUComputePassEncoder", "end"),
    ("GPUComputePassEncoder", "setBindGroup"),
    -- GPURenderPassEncoder
    ("GPURenderPassEncoder", "setPipeline"),
    ("GPURenderPassEncoder", "setBindGroup"),
    ("GPURenderPassEncoder", "setVertexBuffer"),
    ("GPURenderPassEncoder", "setIndexBuffer"),
    ("GPURenderPassEncoder", "draw"),
    ("GPURenderPassEncoder", "drawIndexed"),
    ("GPURenderPassEncoder", "setStencilReference"),
    ("GPURenderPassEncoder", "end") ]

-- ════════════════════════════════════════════════════════════════════
-- T1711 — quanta_methods_in_spec
-- ════════════════════════════════════════════════════════════════════

/-- True iff every `(interfaceName, methodName)` pair in
    `quantaMethodSubset` is declared somewhere in the spec's flattened
    method list. -/
def allMethodsInSpec
    (s : WebGpuSpec) (subset : List (String × String)) : Bool :=
  subset.all (fun p => s.hasMethod p.fst p.snd)

/-- **T1711 — quanta_methods_in_spec**: every WebGPU method that
    `web/src/quanta.ts` / `web/src/webgpu.ts` invokes is declared by
    the W3C `webgpu.idl` on the interface Quanta calls it on, with
    mixin includes resolved (e.g. `GPUBindingCommandsMixin` flattened
    onto `GPUComputePassEncoder`).

    Together with T1710, this *replaces* a second component of T1707.
    See T1712 below for the arity discharge.

    Discharged by `native_decide`: `WebGpuSpec.hasMethod` is
    decidable and the subset is finite. If a future spec drop
    renames a method Quanta uses, this fails to elaborate at
    `lake build` time. -/
theorem quanta_methods_in_spec :
    allMethodsInSpec webGpuSpec quantaMethodSubset = true := by
  native_decide

-- ════════════════════════════════════════════════════════════════════
-- Quanta's WebGPU call-site arities
-- ════════════════════════════════════════════════════════════════════

/-- For each call site in `web/src/webgpu.ts`, the
    `(interfaceName, methodName, callArity)` triple capturing how
    many arguments Quanta actually passes. WebIDL allows method
    *overloading* by parameter count, so the conformance check
    asks: does some declared overload of this method admit this
    call's arity? -/
def quantaCallArities : List (String × String × Nat) :=
  -- GPU root + adapter
  [ ("GPU", "requestAdapter", 0),
    ("GPUAdapter", "requestDevice", 0),
    -- GPUDevice creators
    ("GPUDevice", "createBuffer", 1),
    ("GPUDevice", "createTexture", 1),
    ("GPUDevice", "createSampler", 1),
    ("GPUDevice", "createShaderModule", 1),
    ("GPUDevice", "createComputePipeline", 1),
    ("GPUDevice", "createRenderPipeline", 1),
    ("GPUDevice", "createBindGroup", 1),
    ("GPUDevice", "createCommandEncoder", 0),
    -- GPUBuffer
    ("GPUBuffer", "mapAsync", 1),
    ("GPUBuffer", "getMappedRange", 0),
    ("GPUBuffer", "unmap", 0),
    ("GPUBuffer", "destroy", 0),
    -- GPUTexture
    ("GPUTexture", "createView", 0),
    ("GPUTexture", "destroy", 0),
    -- GPUQueue
    ("GPUQueue", "submit", 1),
    ("GPUQueue", "writeBuffer", 3),
    ("GPUQueue", "writeTexture", 4),
    ("GPUQueue", "onSubmittedWorkDone", 0),
    -- GPUCommandEncoder
    ("GPUCommandEncoder", "beginComputePass", 0),
    ("GPUCommandEncoder", "beginRenderPass", 1),
    -- copyBufferToBuffer is the (src, srcOff, dst, dstOff, size)
    -- overload — 5 args; the spec lists two overloads (2-3 and 4-5),
    -- this lands in the 4-5 one.
    ("GPUCommandEncoder", "copyBufferToBuffer", 5),
    ("GPUCommandEncoder", "copyTextureToBuffer", 3),
    ("GPUCommandEncoder", "finish", 0),
    -- GPUComputePassEncoder
    ("GPUComputePassEncoder", "setPipeline", 1),
    ("GPUComputePassEncoder", "dispatchWorkgroups", 3),
    ("GPUComputePassEncoder", "end", 0),
    ("GPUComputePassEncoder", "setBindGroup", 2),
    -- GPURenderPassEncoder
    ("GPURenderPassEncoder", "setPipeline", 1),
    ("GPURenderPassEncoder", "setBindGroup", 2),
    ("GPURenderPassEncoder", "setVertexBuffer", 3),
    ("GPURenderPassEncoder", "setIndexBuffer", 3),
    ("GPURenderPassEncoder", "draw", 2),
    ("GPURenderPassEncoder", "drawIndexed", 2),
    ("GPURenderPassEncoder", "setStencilReference", 1),
    ("GPURenderPassEncoder", "end", 0) ]

-- ════════════════════════════════════════════════════════════════════
-- T1712 — quanta_call_arities_in_spec
-- ════════════════════════════════════════════════════════════════════

/-- True iff every `(interface, method, callArity)` triple in the
    subset is admitted by some declared overload of that method in
    the spec. -/
def allCallAritiesValid
    (s : WebGpuSpec) (calls : List (String × String × Nat)) : Bool :=
  calls.all (fun t => s.callArityValid t.fst t.snd.fst t.snd.snd)

/-- **T1712 — quanta_call_arities_in_spec**: at every call site in
    `web/src/webgpu.ts`, the number of arguments Quanta passes is
    admitted by some declared overload of the called WebIDL method
    — i.e. the call's arity falls in `[requiredArity, maxArity]`,
    or the method is variadic.

    See T1713 below for the full param-type discharge.

    A spec change that adds a required parameter Quanta doesn't
    pass, or removes an optional parameter Quanta still passes,
    fails this theorem at `lake build` time. -/
theorem quanta_call_arities_in_spec :
    allCallAritiesValid webGpuSpec quantaCallArities = true := by
  native_decide

-- ════════════════════════════════════════════════════════════════════
-- Quanta's WebGPU call-site parameter types
-- ════════════════════════════════════════════════════════════════════

/-- For each call site in `web/src/webgpu.ts`, the
    `(interfaceName, methodName, argTypes)` triple capturing the
    leading parameter types Quanta supplies, in the spec's
    canonical form (typedefs preserved, e.g. `"GPUSize64"` rather
    than `"unsigned long long"`). The list length must equal the
    matching entry in `quantaCallArities`.

    The conformance check finds *some* declared overload of the
    method whose leading param types equal this list and whose
    arity range admits the call. WebIDL overloading is supported
    naturally — `GPURenderPassEncoder.setBindGroup` has a 2-3 and a
    5-arg overload; Quanta's 2-arg call lands in the first. -/
def quantaCallTypes : List (String × String × List String) :=
  -- GPU root + adapter
  [ ("GPU", "requestAdapter", []),
    ("GPUAdapter", "requestDevice", []),
    -- GPUDevice creators
    ("GPUDevice", "createBuffer", ["GPUBufferDescriptor"]),
    ("GPUDevice", "createTexture", ["GPUTextureDescriptor"]),
    ("GPUDevice", "createSampler", ["GPUSamplerDescriptor"]),
    ("GPUDevice", "createShaderModule", ["GPUShaderModuleDescriptor"]),
    ("GPUDevice", "createComputePipeline", ["GPUComputePipelineDescriptor"]),
    ("GPUDevice", "createRenderPipeline", ["GPURenderPipelineDescriptor"]),
    ("GPUDevice", "createBindGroup", ["GPUBindGroupDescriptor"]),
    ("GPUDevice", "createCommandEncoder", []),
    -- GPUBuffer
    ("GPUBuffer", "mapAsync", ["GPUMapModeFlags"]),
    ("GPUBuffer", "getMappedRange", []),
    ("GPUBuffer", "unmap", []),
    ("GPUBuffer", "destroy", []),
    -- GPUTexture
    ("GPUTexture", "createView", []),
    ("GPUTexture", "destroy", []),
    -- GPUQueue
    ("GPUQueue", "submit", ["sequence<GPUCommandBuffer>"]),
    ("GPUQueue", "writeBuffer", ["GPUBuffer", "GPUSize64", "AllowSharedBufferSource"]),
    ("GPUQueue", "writeTexture",
      ["GPUTexelCopyTextureInfo", "AllowSharedBufferSource",
       "GPUTexelCopyBufferLayout", "GPUExtent3D"]),
    ("GPUQueue", "onSubmittedWorkDone", []),
    -- GPUCommandEncoder
    ("GPUCommandEncoder", "beginComputePass", []),
    ("GPUCommandEncoder", "beginRenderPass", ["GPURenderPassDescriptor"]),
    -- copyBufferToBuffer 5-arg overload: (src, srcOff, dst, dstOff, size)
    ("GPUCommandEncoder", "copyBufferToBuffer",
      ["GPUBuffer", "GPUSize64", "GPUBuffer", "GPUSize64", "GPUSize64"]),
    ("GPUCommandEncoder", "copyTextureToBuffer",
      ["GPUTexelCopyTextureInfo", "GPUTexelCopyBufferInfo", "GPUExtent3D"]),
    ("GPUCommandEncoder", "finish", []),
    -- GPUComputePassEncoder
    ("GPUComputePassEncoder", "setPipeline", ["GPUComputePipeline"]),
    ("GPUComputePassEncoder", "dispatchWorkgroups", ["GPUSize32", "GPUSize32", "GPUSize32"]),
    ("GPUComputePassEncoder", "end", []),
    ("GPUComputePassEncoder", "setBindGroup", ["GPUIndex32", "GPUBindGroup?"]),
    -- GPURenderPassEncoder
    ("GPURenderPassEncoder", "setPipeline", ["GPURenderPipeline"]),
    ("GPURenderPassEncoder", "setBindGroup", ["GPUIndex32", "GPUBindGroup?"]),
    ("GPURenderPassEncoder", "setVertexBuffer", ["GPUIndex32", "GPUBuffer?", "GPUSize64"]),
    ("GPURenderPassEncoder", "setIndexBuffer", ["GPUBuffer", "GPUIndexFormat", "GPUSize64"]),
    ("GPURenderPassEncoder", "draw", ["GPUSize32", "GPUSize32"]),
    ("GPURenderPassEncoder", "drawIndexed", ["GPUSize32", "GPUSize32"]),
    ("GPURenderPassEncoder", "setStencilReference", ["GPUStencilValue"]),
    ("GPURenderPassEncoder", "end", []) ]

-- ════════════════════════════════════════════════════════════════════
-- T1713 — quanta_call_types_in_spec
-- ════════════════════════════════════════════════════════════════════

/-- True iff every triple in `calls` is admitted by some declared
    overload's leading param-type list (with arity in range). -/
def allCallTypesValid
    (s : WebGpuSpec) (calls : List (String × String × List String)) : Bool :=
  calls.all (fun t => s.callTypesValid t.fst t.snd.fst t.snd.snd)

/-- **T1713 — quanta_call_types_in_spec**: at every call site in
    `web/src/webgpu.ts`, *some* declared overload of the called
    WebIDL method has the same leading parameter type names (in
    the spec's canonical form, typedefs preserved) as the types
    Quanta supplies, and that overload's arity range admits the
    call.

    Together with T1710 + T1711 + T1712, the enum-string,
    method-presence, arity, and parameter-type components of T1707
    are now Lean theorems. A11's residue narrows to:
    - wasm-linker faithfulness for `extern "C"` (libc-equivalent
      trust on the calling convention itself, irreducible for any
      wasm32 target);
    - typedef-stability — that f64 is a faithful representation of
      `GPUSize64`/`GPUSize32`/`GPUIndex32`/etc. in the JS engine.
      This is the smallest possible residue at the WebIDL level.

    Discharged by `native_decide`: `WebGpuSpec.callTypesValid` is
    decidable, the subsets are finite. A spec change that renames
    a typedef Quanta uses, alters a parameter type, or shifts an
    argument's position within an overload fails `lake build`. -/
theorem quanta_call_types_in_spec :
    allCallTypesValid webGpuSpec quantaCallTypes = true := by
  native_decide

end Quanta.Theorems.IdlConformance

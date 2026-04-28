# Quanta GPU Machine Model

> This document defines the hardware and driver axioms for formal verification.
> Everything proven about Quanta is proven **relative to this model**.
> If a GPU driver or hardware violates these axioms, the fault is in the
> driver/hardware, not in Quanta.

## Axioms

### A1: Metal driver correctness

The Metal driver correctly implements:
- `MTLComputeCommandEncoder` encodes and dispatches compute work
- `MTLRenderCommandEncoder` encodes and submits render passes
- `MTLBuffer` contents are visible to the GPU after CPU write (shared storage mode)
- `newLibraryWithData:` accepts valid metallib and produces working pipelines
- `addCompletedHandler:` calls the block exactly once after GPU completion
- `dispatchThreadgroups:` launches the specified number of threadgroups

### A2: Vulkan driver correctness

The Vulkan driver correctly implements:
- `vkCreateShaderModule` accepts valid SPIR-V and creates a working module
- `vkCmdDispatch` launches the specified number of workgroups
- `vkQueueSubmit` with a VkFence signals the fence after GPU completion
- Host-visible memory is coherent after `vkMapMemory` / `vkUnmapMemory`
- `VK_KHR_synchronization2` barriers enforce correct memory ordering

### A3: GPU hardware execution

The GPU hardware correctly executes:
- SPIR-V instructions as defined in the SPIR-V 1.6 specification
- Metal shader bytecode (metallib) as defined by Apple
- Workgroup barriers synchronize all invocations within a workgroup
- Atomic operations are sequentially consistent within a workgroup
- `gl_GlobalInvocationID` uniquely identifies each invocation

### A4: LLVM codegen

LLVM 22 correctly:
- Emits PTX from valid LLVM IR targeting `nvptx64-nvidia-cuda`
- Emits GCN ELF from valid LLVM IR targeting `amdgcn-amd-amdhsa`
- Respects `mem2reg`, optimization passes, and data layout

### A5: Rust compiler

`rustc` correctly:
- Parses proc macro output as valid Rust
- Evaluates `const` expressions at compile time
- Enforces type safety across the macro boundary

### A6: Vulkan memory model

The `VK_KHR_vulkan_memory_model` extension provides, with the SPIR-V `Vulkan`
memory model:
- A scoped modification order: for each atomic object and each memory scope
  (`Invocation` ⊂ `Subgroup` ⊂ `Workgroup` ⊂ `QueueFamily` ⊂ `CrossDevice`),
  there is a total order of all modifications performed at that scope.
- Sequentially consistent atomics at workgroup scope: within a workgroup,
  `MemorySemanticsSequentiallyConsistentKHR` atomics admit a total order.
- Cross-workgroup acquire-release: `OpAtomicStore(Release, Device)` synchronizes
  with `OpAtomicLoad(Acquire, Device)` that reads the stored value.
- Barriers flush relaxed ops: `OpControlBarrier Workgroup Workgroup
  AcquireRelease|WorkgroupMemory` makes prior workgroup-memory writes visible.
- Storage-buffer fences propagate writes across dispatches on the same queue.

Quanta requires this extension; behavior without it is unspecified.

### A7: PTX release-acquire scoping (NVIDIA CUDA)

PTX (ISA 8.5) provides scoped release-acquire ordering:
- `st.release.cta` / `ld.acquire.cta` synchronize threads within a CTA
  (= workgroup).
- `st.release.gpu` / `ld.acquire.gpu` synchronize threads across CTAs on the
  same GPU.
- `st.release.sys` / `ld.acquire.sys` synchronize GPU and CPU threads on
  PCIe-coherent memory.
- `bar.sync 0` is a full acquire-release fence at CTA scope.
- A fence at a wider scope subsumes fences at narrower scopes (`fence.gpu`
  implies `fence.cta`; `fence.sys` implies `fence.gpu`).

### A8: Metal memory model (Apple GPU)

The Metal Shading Language memory model (MSL spec, §6.13) provides:
- Default-relaxed device memory: non-atomic device-memory accesses have no
  cross-thread ordering. Ordering requires explicit `atomic_*` functions.
- C++14-style atomics: `atomic_store_explicit(release)` synchronizes with
  `atomic_load_explicit(acquire)` at the same address.
- `threadgroup_barrier(mem_threadgroup)` flushes threadgroup-memory writes
  within a workgroup; `mem_device` flushes device-memory writes.
- `simdgroup_barrier` synchronizes within a SIMD-group (32 threads).
- Threadgroup memory is coherent within a workgroup after a barrier; before
  the barrier, cross-thread visibility is not guaranteed.
- Threadgroup memory is workgroup-local — it is not visible across workgroups,
  ever. Cross-workgroup communication requires device memory with atomics.

### A9: AMD RDNA memory model

AMD RDNA 1/2/3 (and Sea Islands+) ISA provides:
- Scoped acquire-release at workgroup, agent (= device), and system scope via
  `flat_atomic` / `s_buffer_atomic` instructions with the `glc` (globally
  coherent) flag and scope annotation.
- `glc` on cross-workgroup loads bypasses the L1 vector cache, reading from
  L2 directly. Without `glc`, agent-scope ordering may still see stale L1.
- `s_barrier` (preceded by `s_waitcnt lgkmcnt(0)`) is a full workgroup-scope
  acquire-release fence — both LDS and (appropriately scoped) global memory.
- LDS (Local Data Share, threadgroup memory) is coherent within a workgroup
  after `s_barrier`; before the barrier, cross-wave visibility is not
  guaranteed.
- Lanes within a wave (32 or 64 threads) execute in lockstep — no explicit
  synchronization is needed for wave-level communication via `ds_permute` or
  `v_readlane`.
- The AMD Vulkan driver (AMDVLK / RADV) lowers SPIR-V `gl_ScopeWorkgroup` to
  workgroup scope, `gl_ScopeDevice` to agent scope, and `gl_ScopeSubgroup` to
  wave scope.

### Cross-model agreement

All four backends agree on two minimal properties that Quanta relies on:
- **Workgroup barrier semantics:** `workgroup_barrier()` is a full acquire-release
  fence at workgroup scope for shared memory. Quanta emits, respectively:
  `OpControlBarrier Workgroup Workgroup AcqRel|WorkgroupMemory` (Vulkan),
  `bar.sync 0` (PTX), `threadgroup_barrier(mem_threadgroup)` (Metal),
  `s_waitcnt lgkmcnt(0)` + `s_barrier` (RDNA).
- **Cross-workgroup requires device scope:** communication between threads in
  different workgroups requires device-scope acquire-release atomics. Shared /
  threadgroup / LDS memory is workgroup-local on every backend.

Lean formalization: `specs/verify/lean/Quanta/Axioms/MemoryModels.lean`
(namespaces `VulkanMM`, `PtxMM`, `MetalMM`, `RdnaMM`).

Empirical validation: `specs/verify/herd7/` — three litmus tests
(`message_passing.litmus`, `store_buffer.litmus`,
`atomic_add_visibility.litmus`) check the message-passing and store-buffer
patterns under release-acquire on a Cat-language model compatible with A6.

### A10: WebGPU host correctness (browser)

The browser's WebGPU implementation (Chrome/Dawn, Safari/WebKit,
Firefox/wgpu-core) correctly implements:
- `navigator.gpu.requestAdapter` and `adapter.requestDevice` resolve to
  a working `GPUDevice` when WebGPU is available.
- `device.createShaderModule({ code })` accepts any WGSL source string
  that is well-formed per the W3C WGSL spec (§3 Parsing + §4 Validation),
  and produces a non-null module.
- `device.createComputePipeline` and `createRenderPipeline` succeed for
  any shader module + entry-point name that exists in the module.
- `dispatchWorkgroups(x, y, z)` runs the kernel exactly `x*y*z`
  workgroup invocations, consistent with A3 (GPU hardware execution).
- `queue.submit([...])` orders command buffers in queue order; later
  submissions observe earlier submissions' effects on storage resources.
- `queue.onSubmittedWorkDone()` returns a Promise that resolves exactly
  once after prior submissions complete.
- `buffer.mapAsync(READ)` resolves with a snapshot consistent with the
  most recent submitted write that completed before resolution.
- `queue.writeBuffer(b, 0, data)` is atomic with respect to subsequent
  `submit()` calls — a later dispatch sees the full `data`, never a
  partial update.

This is what makes WebGPU's JavaScript implementation a **reasonable
backend** — without it, the soundness of the WGSL emitter (T410) buys
no end-to-end correctness for the browser. A10 is the WebGPU analog of
A1 (Metal) / A2 (Vulkan).

### A11: Quanta wasm ↔ JS ABI faithfulness (post-B⁰)

After step B⁰ (2026-04-28) the FFI boundary is hand-authored on both
sides; A11 is the trust statement covering it. For every `unsafe extern
"C"` import declared in `src/driver/webgpu/ffi.rs`:
- The corresponding implementation in `web/src/glue.ts` (compiled to
  `glue.js` at build time) marshals arguments per the Quanta ABI:
  long-lived JS objects cross as `u32` handles into a JS-side handle
  table; strings cross as `(ptr, len)` into wasm linear memory;
  `u64` sizes cross as `f64`.
- Async imports take a `task: u32` argument and resolve exactly one of
  `quanta_resolve(task, handle)` (success) or `quanta_reject(task)`
  (failure) per task id, which the executor in
  `src/driver/webgpu/executor.rs` turns into `Future::poll` returning
  `Ready(Ok(handle))` / `Ready(Err(()))`.
- The JS-side handle table is the unique source of GPU-object identity;
  released handles never alias fresh resources.

Pre-B⁰ this axiom covered the `wasm-bindgen` runtime crate (~30-60 KB
of opaque codegen). Post-B⁰ it covers only project-local code:
`src/driver/webgpu/ffi.rs` (~300 lines) plus `web/src/*.ts` (~500
lines). Both sides are version-controlled, auditable line by line, and
shrink-able further to a Lean theorem under B″.

The smoke tests (`examples/web_add_one`, `examples/web_triangle`) are
the operational check that the ABI matches the actual WebGPU surface.

Lean formalization: `specs/verify/lean/Quanta/Axioms/WebGpu.lean`
(axioms `wgsl_module_acceptance`, `compute_pipeline_creation`,
`dispatch_executes_kernel`, `submit_ordering`, `quanta_abi_faithful`,
and siblings).

### API-layer invariants (step 075)

The trait surface (`GpuDevice`, `Pulse`, `Field`, `Wave`) sits above
the per-driver axioms — properties stated here hold across all
backends and are proven directly in Verus.

Verus formalization: `specs/verify/verus/quanta-api/`:
- `pulse_lifetime.rs` — Pulse state machine. T720 (monotonic-with-reset),
  T721 (`is_done` agrees with state), T722 (closure consumed exactly
  once: no double-fire of the deferred-wait `FnOnce`).
- `field_lifetime.rs` — Field handle invariants. T730 (handle is
  immutable post-construction), T731 (drop frees exactly once),
  T732 (no use-after-free at the mirror level), T733 (`byte_size =
  count * size_of::<T>()`).
- `wave_lifetime.rs` — Wave binding invariants. T740 (slot-indexed
  writes leave other slots unchanged), T741 (**capability
  monotonicity**: `binding_count` / `texture_count` never shrink),
  T742 (handle 0 is the unbound sentinel), T743 (push-data slot
  alignment).
- `submission_safety.rs` — submission-layer state machine.
  T750 (submit-after-close is forbidden), T751 (fence ordering: a
  fence at submission N is visible only after N+1 submits, and stays
  visible monotonically), T752 (no double-submit: `CommandBuffer` is
  consumed by `submit`), T753 (pulse ↔ submission position
  correspondence), T754 (hazard-free declaration the per-backend
  driver discharges).

Verified by Verus: **27 theorems across 4 files, 0 errors**.

---

## Theorems (proven by Quanta)

### T1: IR type correctness

```
∀ kernel_fn parsed by #[quanta::kernel]:
  ∀ param in kernel_fn.params:
    typeof(param) ∈ {&[T], &mut [T], T, &Texture2D<T>}
    → KernelParam variant matches the Rust type
```

### T2: SPIR-V opcode correctness

```
∀ op in KernelOps:
  emit_spirv(op) produces the SPIR-V opcode defined by the SPIR-V 1.6 spec
  
  Specifically:
    BinOp::Add    → OpIAdd (128) or OpFAdd (129)
    BinOp::BitAnd → OpBitwiseAnd (199)    // NOT 197
    BinOp::BitOr  → OpBitwiseOr (197)     // NOT 198
    BinOp::BitXor → OpBitwiseXor (198)    // NOT 199
    ...
```

### T3: Wire format roundtrip

```
∀ kernel_def: KernelDef:
  deserialize(serialize(kernel_def)) = kernel_def

∀ output: CompilerOutput:
  deserialize(serialize(output)) = output

∀ shader: ShaderDef:
  deserialize(serialize(shader)) = shader
```

### T4: MSL text validity

```
∀ kernel_def: KernelDef:
  emit_msl(kernel_def) produces text that xcrun metal accepts without error
```

### T5: WGSL text validity

```
∀ kernel_def: KernelDef:
  emit_wgsl(kernel_def) produces text that is valid WGSL per the WebGPU spec
```

### T6: Push constant layout

```
∀ param at slot S with type T:
  CPU: wave.set_value(S, value) writes sizeof(T) bytes at offset S * 16
  GPU: push constant struct member S is at offset S * 16
  → CPU write and GPU read access the same bytes
```

### T7: Vertex descriptor consistency

```
∀ VertexLayout in PipelineDesc:
  Metal: MTLVertexDescriptor stride/format/offset matches VertexLayout
  Vulkan: VkVertexInputBindingDescription/AttributeDescription matches VertexLayout
  → vertex shader reads the correct data from vertex buffers
```

### T8: Shader varying coordination

```
∀ vertex shader with N attribute params (param[0] = position):
  vertex output Location[i] = param[i+1] for i in 0..N-1
  fragment input Location[i] matches vertex output Location[i]
  → fragment shader receives the correct interpolated values
```

---

## Undecidable (outside our proof boundary)

| Property | Why |
|----------|-----|
| GPU timing determinism | Hardware scheduling is non-deterministic |
| Driver optimization correctness | Vendor shader compilers may reorder/optimize |
| Cross-device memory coherence | PCIe ordering depends on hardware topology |
| Texture sampling precision | Implementation-defined rounding in hardware |
| Power management effects | Thermal throttling changes execution time |

---

## Proof strategy

| Theorem | Tool | Status |
|---------|------|--------|
| T2 (SPIR-V opcodes) | Kani | pending |
| T3 (wire roundtrip) | Kani | pending |
| T6 (push constant layout) | Kani | pending |
| T1, T4, T5 | Property tests | partial (existing test suite) |
| T7, T8 | Integration tests | done (render_triangle_test) |

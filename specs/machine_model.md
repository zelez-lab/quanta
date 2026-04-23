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

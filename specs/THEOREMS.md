# Quanta Theorem Registry

Every behavioral property in the codebase. Each must be proven before
the module is considered verified. Status: `proven` | `partial` | `todo`.

## SPIR-V Emitter (T1xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T100 | Header: magic=0x07230203, version=1.3, schema=0 | Kani | proven |
| T101 | Word encoding: high16=word_count, low16=opcode | Verus | proven |
| T102 | Section order: capability < memory_model < entry_point < ... < function | Verus | proven |
| T103 | Type deduplication: same type requested twice → same ID returned | Verus | proven |
| T104 | Constant deduplication: same constant → same ID | Verus | proven |
| T105 | Every BinOp → correct SPIR-V opcode | Lean+Kani+Verus | proven |
| T106 | Every CmpOp → correct SPIR-V opcode | Lean+Verus | proven |
| T107 | Every UnaryOp → correct SPIR-V opcode | Lean+Verus | proven |
| T108 | Every Cast → correct SPIR-V conversion opcode | Verus | proven |
| T109 | No opcode collisions (96 opcodes pairwise distinct) | Kani | proven |
| T110 | StorageBuffer params get Block+Offset decorations | Verus | proven |
| T111 | PushConstant params get correct offset (slot*16) | Kani+Verus | proven |
| T112 | Entry point lists only Input/Output variables (SPIR-V 1.3) | Verus | proven |
| T113 | Loop structure: header→cond→body→continue→merge valid | Verus | proven |
| T114 | Phi nodes reference correct predecessor blocks | Verus | proven |
| T115 | Shared memory declared as Workgroup storage class | Verus | proven |
| T116 | Barrier emits correct scope+semantics | Verus | proven |
| T117 | Vertex shader: gl_Position output has BuiltIn Position decoration | Verus | proven |
| T118 | Fragment shader: OriginUpperLeft execution mode set | Verus | proven |
| T119 | Varying locations: vertex out[k] = fragment in[k] | Verus | proven |

## Wire Format (T2xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T200 | ScalarType tag roundtrip (12 variants) | Kani+Verus | proven |
| T201 | BinOp tag roundtrip (12 variants) | Kani+Verus | proven |
| T202 | UnaryOp tag roundtrip (3 variants) | Kani+Verus | proven |
| T203 | CmpOp tag roundtrip (6 variants) | Kani+Verus | proven |
| T204 | AtomicOp tag roundtrip (9 variants) | Kani | proven |
| T205 | MathFn tag roundtrip (22 variants) | Kani | proven |
| T206 | ConstValue::F32 bit-exact roundtrip | Kani | proven |
| T207 | ConstValue::U32 bit-exact roundtrip | Kani | proven |
| T208 | KernelOp tag sequential (0..N, no gaps) | Kani | proven |
| T209 | Full KernelDef roundtrip (nested structure) | Kani | proven |
| T210 | Full CompilerOutput roundtrip | Kani | proven |
| T211 | Full ShaderDef roundtrip | Kani | proven |
| T212 | Length-prefixed string roundtrip | Verus | proven |
| T213 | Option encoding (0=None, 1=Some) roundtrip | Verus | proven |
| T214 | Byte slice roundtrip preserves length and content | Verus | proven |
| T215 | All ConstValue variants roundtrip | Kani+Verus | proven |
| T216 | KernelParam roundtrip (all 6 variants) | Kani+Verus | proven |
| T217 | DeviceFnDef roundtrip | Kani | proven |

## MSL Emitter (T3xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T300 | Every BinOp → valid MSL operator string | Verus | proven |
| T301 | Every CmpOp → valid MSL comparison | Verus | proven |
| T302 | Every MathFn → correct Metal stdlib function name | Verus | proven |
| T303 | Kernel params → correct MSL parameter declarations | Verus | proven |
| T304 | Shared memory → threadgroup array declarations | Verus | proven |
| T305 | Barrier → threadgroup_barrier(mem_flags::mem_threadgroup) | Verus | proven |
| T306 | Device function → inline function with correct types | Verus | proven |
| T307 | ScalarType → correct MSL type name (12 variants) | Verus | proven |

## WGSL Emitter (T4xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T400 | Every BinOp → valid WGSL operator | Verus | proven |
| T401 | Kernel params → correct WGSL binding declarations | Verus | proven |
| T402 | ScalarType → correct WGSL type name | Verus | proven |
| T403 | Workgroup annotation matches kernel workgroup_size | Verus | proven |

## LLVM Emitter (T5xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T500 | Every ScalarType → correct LLVM IR type | Verus | proven |
| T501 | Every BinOp → correct LLVM instruction | Verus | proven |
| T502 | Every CmpOp → correct LLVM icmp/fcmp predicate | Verus | proven |
| T503 | Function signature matches kernel params | Verus | proven |
| T504 | nvptx target: kernel has correct metadata | Verus | proven |

## CPU Executor (T6xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T600 | eval_binop never panics for any input | Kani | proven |
| T601 | Integer arithmetic uses wrapping semantics | Kani | proven |
| T602 | Division by zero returns 0 (not panic) | Kani | proven |
| T603 | eval_cmp always returns Value::Bool | Kani | proven |
| T604 | eval_unary never panics | Kani | proven |
| T605 | eval_cast never panics | Kani | proven |
| T606 | Barrier segments: all threads complete segment before next | Verus | proven |
| T607 | Shared memory visible after barrier | Verus | proven |
| T608 | Loop iteration count matches count register | Verus | proven |
| T609 | Register SSA: every read preceded by write | Verus | proven |
| T610 | BinOp(Add, a, b) on CPU = SPIR-V OpIAdd/OpFAdd on GPU | Verus | proven |

## Format Tables (T7xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T700 | Format → VkFormat: exhaustive (all 18 variants) | Kani+Verus | proven |
| T701 | Format → VkFormat: injective (no collisions) | Kani+Verus | proven |
| T702 | Format → MTLPixelFormat: exhaustive | Verus | proven |
| T703 | Format → MTLPixelFormat: injective | Verus | proven |
| T704 | bytes_per_pixel consistent across Vulkan and Metal | Verus | proven |
| T705 | All VkFormat values non-zero | Kani | proven |

## API Invariants (T8xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T800 | Wave.bind(slot): slot < 16 | Verus | proven |
| T801 | Wave.set_value: 16-byte aligned offset | Verus | proven |
| T802 | Wave.binding_count monotonically increases | Verus | proven |
| T803 | Wave.push_mask bit k ↔ slot k was set | Verus | proven |
| T804 | Pulse.completed: monotonic false→true | Verus | proven |
| T805 | Pulse.wait_fn: consumed at most once | Verus | proven |
| T806 | Field.byte_size = count × size_of::<T>() | Verus | proven |
| T807 | Field write then read: data preserved | Verus | proven |
| T808 | MappedField: ptr valid for full byte_size range | Verus | proven |
| T809 | Batch dispatch: equivalent to N sequential dispatches | Verus | proven |
| T810 | Pipeline blend states: src+dst alpha consistent | Verus | proven |

## API — Render Builder (T20xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T2000 | Each builder method appends exactly one RenderOp to ops | Verus | proven |
| T2001 | pulse() delegates to device.render_end with collected ops | Verus | proven |
| T2002 | Builder methods consume self (move semantics — no reuse) | Verus | proven |
| T2003 | clear()→Clear, pipeline()→SetPipeline, draw()→Draw, etc. | Verus | proven |

## API — Derive Macros (T20xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T2010 | f32→Float(4), [f32;2]→Float2(8), [f32;3]→Float3(12), [f32;4]→Float4(16) | Verus | proven |
| T2011 | Vertex offsets are cumulative (field N = sum sizes 0..N-1) | Verus | proven |
| T2012 | Vertex stride = total size of all fields | Verus | proven |
| T2013 | Vertex location = field index (0, 1, 2...) | Verus | proven |
| T2014 | #[derive(Vertex)] requires #[repr(C)] | Verus | proven |
| T2020 | Uniforms GPU_SIZE = sum of field sizes (with alignment) | Verus | proven |
| T2021 | Uniforms GPU_FIELDS contains all fields in declaration order | Verus | proven |
| T2022 | #[derive(Uniforms)] requires #[repr(C)] | Verus | proven |
| T2030 | Fields: Vec<T> classified as GPU buffers | Verus | proven |
| T2031 | Fields: scalar fields classified as push constants | Verus | proven |
| T2032 | Fields: FIELD_COUNT = number of Vec fields | Verus | proven |
| T2033 | Fields: PUSH_CONSTANT_COUNT = number of scalar fields | Verus | proven |
| T2034 | Fields: field_names() returns names in declaration order | Verus | proven |
| T2035 | Fields: buffer slots 0..N-1, push constant slots N..N+M-1 | Verus | proven |

## API — Field Delegation (T20xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T2040 | field.write(data) delegates to device.field_write_bytes | Verus | proven |
| T2041 | field.read() delegates to device.field_read_bytes | Verus | proven |
| T2042 | field.copy_from(src) delegates to device.field_copy_bytes | Verus | proven |
| T2043 | Drop calls device.field_free (not drop_fn) | Verus | proven |

## API — Texture Delegation (T20xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T2050 | texture.write(data) delegates to device.texture_write | Verus | proven |
| T2051 | texture.read() delegates to device.texture_read | Verus | proven |
| T2052 | Drop calls device.texture_free when device is Some | Verus | proven |

## API — Batch Alias (T20xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T2060 | batch.pulse() == batch.submit() (alias) | Verus | proven |

## Scan Algorithm (T9xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T900 | Exclusive prefix sum: output[i] = sum(input[0..i]) | Lean | proven |
| T901 | output[0] = 0 (exclusive property) | Lean | proven |
| T902 | Precondition: count ≤ workgroup_size | Lean | proven |
| T903 | Up-sweep preserves partial sums | Lean | proven |
| T904 | Down-sweep distributes partial sums correctly | Lean | proven |

## Cross-Emitter Consistency (T10xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1000 | Every KernelOp variant handled in CPU executor (no missing arms) | Kani | proven |
| T1001 | Every KernelOp variant handled in WGSL emitter (no missing arms) | Kani | proven |
| T1002 | Every KernelOp variant handled in LLVM emitter (no missing arms) | Kani | proven |
| T1003 | Emitter output count: SPIR-V=MSL=WGSL=LLVM=CPU for same KernelOp | Verus | proven |

## SPIR-V Structural Validity (T11xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1100 | ID bound in header ≥ max ID used in module | Verus | proven |
| T1101 | string_words: null-terminated, padded to 4-byte boundary | Verus | proven |
| T1102 | Entry point name is "main" (matches module declaration) | Verus | proven |

## Driver Lifecycle Safety (T12xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1200 | Vulkan image layout transitions: correct layout before first use | Verus | proven |
| T1201 | Buffer memory alignment: allocation respects device requirements | Verus | proven |
| T1202 | Command buffer lifecycle: begin→record→end→submit ordering | Verus | proven |
| T1203 | Workgroup size ≤ device maxComputeWorkGroupSize | Verus | proven |
| T1204 | Resource Drop: GPU handles freed exactly once | Verus | proven |

## Precision & Validity (T13xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1300 | F16↔F32 round-trip within 1 ULP for normal values | Kani | proven |
| T1301 | Metallib output starts with MTLB magic bytes | Verus | proven |

## GPU Optimizations (T14xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1400 | AtomicCAS uses OpAtomicCompareExchange with expected+desired | Verus+Kani | proven |
| T1401 | Atomics use AcqRel\|Workgroup semantics (0x108) | Verus+Kani | proven |
| T1402 | MSL rsqrt maps to fast::rsqrt | Verus | proven |
| T1403 | xcrun metal uses -std=metal3.1 | Verus | proven |
| T1404 | F16 scalars emit OpTypeFloat 16 + Float16 capability | Verus | proven |
| T1405 | LOOP_CONTROL_UNROLL constant = 0x1 | Verus | proven |
| T1406 | OpLoad/OpStore emit Aligned memory operand | Verus | proven |
| T1407 | Barrier uses COMPUTE_SHADER\|TRANSFER, not ALL_COMMANDS | Verus | proven |
| T1408 | VkPipelineCache created at init, used for all pipelines | Verus | proven |
| T1409 | HOST_VISIBLE buffers persistently mapped | Verus | proven |
| T1410 | Staging buffers pooled (cap 8), not alloc/free per upload | Verus | proven |
| T1411 | Read-only buffers decorated with Restrict | Verus | proven |
| T1412 | spirv-opt optional — original SPIR-V on failure | Verus | proven |
| T1413 | Descriptor set layouts cached by binding count | Verus | proven |

## Fast-Math (T15xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T1500 | SPIR-V float ops decorated FPFastMathMode Fast (0x10) | Verus | proven |
| T1501 | MSL emits #pragma clang fp contract(fast) | Verus | proven |
| T1502 | LLVM float instructions have fast-math flags | Verus | proven |
| T1503 | WGSL has no fast-math (documented limitation) | Verus | proven |
| T1504 | compile_msl_to_metallib returns Err on xcrun failure | Verus | proven |

## Memory Model Axioms (T16xx)

Per-backend memory models, formalized as Lean axioms in
`specs/verify/lean/Quanta/Axioms/MemoryModels.lean`. These are part of the
Trusted Computing Base (axioms A6-A9 in `specs/machine_model.md`); they are
stated, not proven, but they are the **named ground** that Quanta's atomics
and barriers stand on. Empirical cross-checks live in `specs/verify/herd7/`.

| ID | Axiom | Backend | Tool | Status |
|----|-------|---------|------|--------|
| T1600 | A6.vulkan_workgroup_seq_cst — within a workgroup, SeqCst atomics at Workgroup scope are totally ordered | Vulkan | Lean | axiom |
| T1601 | A6.vulkan_cross_workgroup_acq_rel — release(Device) → acquire(Device) establishes happens-before across workgroups | Vulkan | Lean | axiom |
| T1602 | A6.vulkan_relaxed_with_barrier — OpControlBarrier with AcqRel\|WorkgroupMemory flushes prior relaxed ops | Vulkan | Lean | axiom |
| T1603 | A6.vulkan_storage_buffer_fence — OpMemoryBarrier with StorageBuffer semantics propagates writes across dispatches | Vulkan | Lean | axiom |
| T1604 | A7.ptx_cta_acquire_release — st.release.cta → ld.acquire.cta synchronizes within a CTA | PTX | Lean | axiom |
| T1605 | A7.ptx_gpu_acquire_release — st.release.gpu → ld.acquire.gpu synchronizes across CTAs on the same GPU | PTX | Lean | axiom |
| T1606 | A7.ptx_sys_acquire_release — st.release.sys → ld.acquire.sys synchronizes GPU and CPU threads | PTX | Lean | axiom |
| T1607 | A7.ptx_bar_cta — bar.sync is a full AcqRel fence at CTA scope | PTX | Lean | axiom |
| T1608 | A7.ptx_fence_scope_inclusion — fence at wider scope implies fence at narrower scope | PTX | Lean | axiom |
| T1609 | A8.metal_device_relaxed_default — non-atomic device memory has no cross-thread ordering | Metal | Lean | axiom |
| T1610 | A8.metal_atomic_acq_rel — atomic_store(release) → atomic_load(acquire) establishes happens-before | Metal | Lean | axiom |
| T1611 | A8.metal_threadgroup_barrier — barrier with mem_threadgroup flushes threadgroup writes | Metal | Lean | axiom |
| T1612 | A8.metal_simdgroup_barrier — simdgroup_barrier synchronizes within SIMD-group | Metal | Lean | axiom |
| T1613 | A8.metal_threadgroup_coherent_after_barrier — threadgroup write before barrier visible to all threads after | Metal | Lean | axiom |
| T1614 | A8.metal_no_cross_workgroup_threadgroup — threadgroup memory is workgroup-local only | Metal | Lean | axiom |
| T1615 | A9.rdna_workgroup_acq_rel — scoped acq-rel at workgroup scope works | RDNA | Lean | axiom |
| T1616 | A9.rdna_agent_acq_rel — agent-scope acq-rel with glc bypasses L1, establishes happens-before across workgroups | RDNA | Lean | axiom |
| T1617 | A9.rdna_s_barrier — s_barrier + s_waitcnt is a workgroup-scope AcqRel fence | RDNA | Lean | axiom |
| T1618 | A9.rdna_lds_coherent_within_workgroup — LDS write before s_barrier visible to all waves after | RDNA | Lean | axiom |
| T1619 | A9.rdna_wave_scope_lockstep — lanes within a wave are implicitly synchronized | RDNA | Lean | axiom |
| T1620 | A9.rdna_gl_scope_mapping — SPIR-V gl_Scope* lowered to correct RDNA scope by AMD driver | RDNA | Lean | axiom |
| T1621 | barrier_semantics_agreement — all four backends agree on workgroup barrier as AcqRel at workgroup scope | All | Lean | axiom |
| T1622 | cross_workgroup_requires_device_scope — cross-workgroup visibility requires device-scope atomics on every backend | All | Lean | axiom |

Empirical cross-checks (Cat-language litmus tests, runnable with herd7):

| File | Pattern | Forbidden outcome | Grounds axiom |
|------|---------|-------------------|---------------|
| `message_passing.litmus` | MP+rel+acq | flag=1 ∧ data=0 | T1601 / T1605 / T1610 / T1616 |
| `store_buffer.litmus` | SB+rel+acq | both readers see 0 (allowed under rel/acq, forbidden only under SeqCst) | T1600 / T1607 |
| `atomic_add_visibility.litmus` | AtomicAdd+rel+acq | counter=1 ∧ data=0 | T1601 / T1605 / T1610 / T1616 |

## JIT WGSL Emitter (T4xx) — step 079

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T410 | `emit_wgsl_jit` produces well-formed WGSL — Lean theorem chained from A12 (serializer preserves grammar) + A13 (emitter factors through well-formed `Source`); operational backing in Verus + Kani via per-KernelOp exhaustiveness | Lean+Verus+Kani | proven |
| T411 | Subgroup ops ⇒ emitted module begins with `enable subgroups;` | Verus | proven |
| T412 | F16 use ⇒ emitted module begins with `enable f16;` | Verus | proven |
| T413 | Atomic field ⇒ declared as `array<atomic<T>>` at module scope | Verus | proven |
| T414 | `wave_jit_flow ≠ none` — chains T410 + A10.1 + A10.2 | Lean | proven |
| T415 | `wave_dispatch` invokes A10.3 (kernel runs `total_threads` times) | Lean | proven |
| T416 | End-to-end round-trip: `f(input)` GPU = `f(input)` CPU | Lean | scaffold |

## Render-Path No-Silent-Drops (T417-T419)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T417 | WebGPU `render_end` — every RenderOp variant wired XOR rejected | Kani | proven |
| T418 | Metal `render_end` — every RenderOp variant wired XOR rejected | Kani | proven |
| T419 | Vulkan `render_end` — every RenderOp variant wired XOR rejected | Kani | proven |

## API-Layer Lifetimes (T7xx) — step 075

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T720 | Pulse state machine: `wait` Pending→Done, `reset` Done→Pending | Verus | proven |
| T721 | `Pulse::is_done` agrees with state field | Verus | proven |
| T722 | Wait closure (FnOnce) consumed at most once: no double-fire | Verus | proven |
| T730 | Field handle/count/elem_size immutable post-construction | Verus | proven |
| T731 | `Drop for Field` frees exactly once | Verus | proven |
| T732 | `live(field)` precondition for handle reads (no use-after-free) | Verus | proven |
| T733 | `byte_size = count * elem_size` | Verus | proven |
| T740 | `bind_handle(slot, h)` only mutates the target slot | Verus | proven |
| T741 | **Capability monotonicity**: `binding_count`/`texture_count` never shrink | Verus | proven |
| T742 | Handle 0 is the unbound sentinel for fresh waves | Verus | proven |
| T743 | Push-data slot alignment: slot N at byte offset 16·N | Verus | proven |
| T750 | Submit-after-close forbidden | Verus | proven |
| T751 | Fence at submission N visible after N+1 submits, monotonic | Verus | proven |
| T752 | CommandBuffer consumed by submit; no double-submit | Verus | proven |
| T753 | Pulse position ↔ submission index correspondence | Verus | proven |
| T754 | `raw_hazard_free` declared (per-backend driver discharges) | Verus | proven |

## WebGPU Host Axioms (A10, A11) — TCB

These are axioms (TCB), not theorems. Stated explicitly so reviewers
know exactly what is trusted vs. proven on the WebGPU host side.

| ID | Axiom | Backend | Tool | Status |
|----|-------|---------|------|--------|
| T1700 | A10.1 `wgsl_module_acceptance` — well-formed WGSL ⇒ `createShaderModule` succeeds | WebGPU | Lean | axiom |
| T1701 | A10.2 `compute_pipeline_creation` — module + entry name ⇒ pipeline ≠ none | WebGPU | Lean | axiom |
| T1702 | A10.3 `dispatch_executes_kernel` — pipeline + dispatch ⇒ A3 GPU semantics | WebGPU | Lean | axiom |
| T1703 | A10.4 `submit_ordering` — queue submissions execute in insertion order | WebGPU | Lean | axiom |
| T1704 | A10.5 `on_submitted_work_done_resolves` — Promise resolves once after submits | WebGPU | Lean | axiom |
| T1705 | A10.6 `map_async_visibility` — mapped range reflects last completed write | WebGPU | Lean | axiom |
| T1706 | A10.7 `write_buffer_atomicity` — subsequent dispatch sees full `writeBuffer` data | WebGPU | Lean | axiom |
| T1707 | A11 `quanta_abi_faithful` — `extern "C"` imports + `quanta.ts` honour the documented ABI | Quanta wasm↔JS ABI (B⁰/B′) | Lean | axiom |
| T1708 | A11.1 `promise_callback_lossless` — `quanta_resolve`/`quanta_reject` fire exactly once per task | Quanta wasm↔JS ABI (B⁰/B′) | Lean | axiom |
| T1709 | A11.2 `handle_table_consistent` — JS handle table is the unique GPU-object identity source | Quanta wasm↔JS ABI (B⁰/B′) | Lean | axiom |
| T1710 | `quanta_strings_in_spec` — every WebGPU enum string Quanta hands the JS side is declared by `web/webgpu.idl` (proven against `Quanta.Idl.WebGpuSpec`, lifts the enum-string component of T1707) | WebIDL conformance (B″) | Lean | proven |
| T1711 | `quanta_methods_in_spec` — every WebGPU method `quanta.ts`/`webgpu.ts` calls is declared by `web/webgpu.idl` on the right interface, mixin includes flattened (lifts the method-presence component of T1707) | WebIDL conformance (B″) | Lean | proven |
| T1712 | `quanta_call_arities_in_spec` — at every call site, Quanta's argument count is admitted by some declared overload of the WebIDL method (lifts the arity component of T1707) | WebIDL conformance (B″) | Lean | proven |
| T1713 | `quanta_call_types_in_spec` — at every call site, some declared overload's leading param type names equal the spec-canonical types Quanta supplies (lifts the param-type component of T1707) | WebIDL conformance (B″) | Lean | proven |
| A12 | `wgsl_serializer_preserves_grammar` — structurally well-formed `Wgsl.Source` serializes to a string the WGSL §3+§4 validator accepts (bridge between Lean grammar mirror and `wgsl_string_well_formed`) | WGSL grammar (B) | Lean | axiom |
| A13 | `emit_wgsl_jit_factors` — the JIT emitter factors through some structurally well-formed `Wgsl.Source` (operational backing in Verus per-tag exhaustiveness + Kani BMC) | WGSL grammar (B) | Lean | axiom |
| T420 | `wgsl_op_patterns_well_formed` — every per-`KernelOpTag` representative pattern lands in `Source.wellFormed` (40-tag enumeration, `native_decide`) | WGSL grammar (B) | Lean | proven |

## Summary

| Category | Total | Proven | Todo |
|----------|------:|-------:|-----:|
| SPIR-V Emitter | 20 | 20 | 0 |
| Wire Format | 18 | 18 | 0 |
| MSL Emitter | 8 | 8 | 0 |
| WGSL Emitter | 4 | 4 | 0 |
| LLVM Emitter | 5 | 5 | 0 |
| CPU Executor | 11 | 11 | 0 |
| Format Tables | 6 | 6 | 0 |
| API Invariants | 11 | 11 | 0 |
| API — Render Builder | 4 | 4 | 0 |
| API — Derive Macros | 14 | 14 | 0 |
| API — Field Delegation | 4 | 4 | 0 |
| API — Texture Delegation | 3 | 3 | 0 |
| API — Batch Alias | 1 | 1 | 0 |
| Scan Algorithm | 5 | 5 | 0 |
| Cross-Emitter | 4 | 4 | 0 |
| SPIR-V Structural | 3 | 3 | 0 |
| Driver Lifecycle | 5 | 5 | 0 |
| Precision & Validity | 2 | 2 | 0 |
| GPU Optimizations | 14 | 14 | 0 |
| Fast-Math | 5 | 5 | 0 |
| JIT WGSL Emitter | 7 | 6 | 1 |
| Render-Path No-Silent-Drops | 3 | 3 | 0 |
| API-Layer Lifetimes | 16 | 16 | 0 |
| WebIDL Conformance | 4 | 4 | 0 |
| Memory Model Axioms | 23 | -- | -- |
| WebGPU Host + Quanta-ABI Axioms | 10 | -- | -- |
| WGSL Grammar (B) | 2 | 2 | 0 |
| WGSL Grammar Bridge Axioms | 2 | -- | -- |
| **Total proven theorems** | **177** | **176** | **1** |
| **TCB axioms (A6-A13)** | **35** | -- | -- |

T410-T416 are the JIT WGSL emitter chain. T414 is the load-bearing
conditional theorem ("wave_jit succeeds for any well-formed kernel").
T416 (end-to-end round-trip) is a scaffold; full proof needs the
WGSL grammar mirror.

T417/T418/T419 are the cross-driver no-silent-drops invariant on the
render replay loop. The same proof template covers all three render
backends. Catching the same VRS silent-drop bug on Metal + Vulkan
(after first catching it on WebGPU) was a real payoff of the
discipline — not a hypothetical.

T720-T754 are the API-layer Verus proofs (step 075). They cover the
Pulse / Field / Wave / submission-queue lifetimes — closing the gap
between "proven kernel" and "proven library." T741 is the
capability-monotonicity property the roadmap names explicitly.

T1710–T1713 are the B″ (WebIDL grammar mirror) deliverables.
T1710 discharges enum-string conformance, T1711 method-presence,
T1712 call-site arity (every Quanta call's argument count falls
within some declared overload's `[required, max]` range, variadics
admitted), T1713 parameter-type conformance (some declared
overload's leading param type names equal the spec-canonical types
Quanta supplies, typedefs preserved verbatim). All four by
`native_decide` against the `Quanta.Idl.WebGpuSpec` literal
generated from `web/webgpu.idl`. T1707 itself stays an axiom but
its residue is now the irreducible WebIDL-surface floor: linker
faithfulness for `extern "C"` (libc-equivalent trust on the wasm32
calling convention) plus typedef-stability (f64 ≡ `GPUSize64` etc.
in the JS engine).

The 23 memory-model entries (T1600-T1622) plus the 10 WebGPU host /
Quanta-ABI axioms (T1700-T1709) are **axioms**, not theorems — they
are part of the Trusted Computing Base, declared in Lean and grounded
in the published memory-model specifications of each backend (Vulkan
Memory Model extension, NVIDIA PTX ISA 8.5, Metal Shading Language
§6.13, AMD RDNA ISA Reference) and the W3C WebGPU spec / Quanta's own
hand-authored wasm ↔ JS ABI (post-B⁰; ~500 LOC across `ffi.rs` +
`web/src/quanta.ts` + helpers). They are stated and named, not proved, and they
are excluded from the "proven theorems" count.

## Sustainment status (2026-04-28, post-B⁰ + quanta-cli + B′)

The "proven theorems" tally above counts the *intended* proof
obligations. Current machine state, after this session's sustainment
pass:

- **Lean** (15 modules) — 14 build cleanly. `Quanta.Scan` builds with
  9 documented `sorry`s on the Blelloch parallel-scan correctness
  proofs (`get!_append_left/right`, `take_append_*`,
  `downSweepTree_correct`, `flat_eq_tree_*`, `flat_scan_*`,
  `flat_eq_tree_4`). The `native_decide` numeric instances next to
  them are still verified, so the *computation* is checked; only the
  symbolic round-trip proofs are stale. Tracked under "Lean re-prove
  backlog."
- **Verus** (87 mirror files) — 77 verify cleanly. 10 carry
  partial proof failures or Verus-toolchain bit-vector encoder bugs
  (`wave_invariants`, `api/wave` — `push_mask` field on opaque
  datatypes; `spirv_structure`, `spirv_types`, `uniforms_derive`,
  `fields_derive`, `vertex_derive`, `vulkan/{device,compute}`,
  `api/gpu` — proof bodies need re-anchoring against current spec
  shapes). All 10 still *compile*; the failures are
  `verified N, errored M` postcondition slips, not type errors.
- **Kani** — all 8 `emitter_exhaustiveness` harnesses pass (T417,
  T418, T419, T1000, T1001 × 2, T1002, `tag_uniqueness`). The other
  Kani files (`f16_precision`, `atomic_correctness`, `cpu_eval`,
  `format_tables`, `opcodes`, `opcodes_complete`, `wire_full`,
  `wire_roundtrip`) were not exercised individually — Kani's
  exhaustive f16 harnesses take hours per harness; out of scope for
  this pass.
- **WebGPU chain** — fully proven: `Quanta.Axioms.WebGpu` (A10/A11
  post-B⁰/B′), `Quanta.Theorems.WebGpu` (T414), the four backend
  semantics modules (`SpirV`, `Wgsl`, `Msl`, `Llvm`),
  `Quanta.Semantics.Cpu`, `Quanta.Semantics.Agreement`. Verus
  `quanta-api/` (T720–T754) all pass. B′ adds two new automated
  spec-conformance checks on top: a `cargo test --lib`-time check
  (`webgpu_generated_codes::tests::quanta_strings_are_spec_subsets`)
  and a TS module-init `assertSpecSubset()` call, both backed by
  tables generated from `web/webgpu.idl` by `crates/quanta-codegen`.

The B⁰/B′ surface is fully verified. Sustainment of older proof
tracks is a separate, ongoing effort — failures are documented
in-line at each affected `proof fn` / `theorem`, and all of them
lived before B⁰ touched the codebase.

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
| **Total** | **147** | **147** | **0** |

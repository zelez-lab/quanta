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

## Scan Algorithm (T9xx)

| ID | Property | Tool | Status |
|----|----------|------|--------|
| T900 | Exclusive prefix sum: output[i] = sum(input[0..i]) | Lean | proven |
| T901 | output[0] = 0 (exclusive property) | Lean | proven |
| T902 | Precondition: count ≤ workgroup_size | Lean | proven |
| T903 | Up-sweep preserves partial sums | Lean | proven |
| T904 | Down-sweep distributes partial sums correctly | Lean | proven |

## Summary

| Category | Total | Proven |
|----------|------:|-------:|
| SPIR-V Emitter | 20 | 20 |
| Wire Format | 18 | 18 |
| MSL Emitter | 8 | 8 |
| WGSL Emitter | 4 | 4 |
| LLVM Emitter | 5 | 5 |
| CPU Executor | 11 | 11 |
| Format Tables | 6 | 6 |
| API Invariants | 11 | 11 |
| Scan Algorithm | 5 | 5 |
| **Total** | **88** | **88** |

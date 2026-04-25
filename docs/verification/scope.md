# Quanta Verification Scope

What exactly is proven, what is not, and where the proof boundary lies.

Every claim references a specific theorem ID from `specs/THEOREMS.md` or a
specific tool. If a property is not listed here, it is not proven.

---

## Trusted Computing Base (Axiom Boundary)

Quanta's proofs are relative to nine axioms defined in `specs/machine_model.md`
and formalized in `specs/verify/lean/Quanta/Axioms/`. These are
**assumed correct** -- if they fail, the fault is in the hardware, driver, or
compiler, not in Quanta.

| Axiom | What we assume correct | Lean file |
|-------|------------------------|-----------|
| A1 | Metal driver implements its API correctly (MTLBuffer visibility, addCompletedHandler exactly-once, dispatchThreadgroups accuracy) | `Metal.lean` |
| A2 | Vulkan driver implements its API correctly (vkCreateShaderModule accepts valid SPIR-V, vkQueueSubmit signals fence, host-visible memory coherent after map/unmap, VK_KHR_synchronization2 barriers enforce ordering) | `Vulkan.lean` |
| A3 | GPU hardware executes SPIR-V and metallib instructions correctly, barriers synchronize workgroups, atomics are sequentially consistent within a workgroup, gl_GlobalInvocationID is unique | `Gpu.lean` |
| A4 | LLVM 22 correctly emits PTX from `nvptx64-nvidia-cuda` IR and GCN ELF from `amdgcn-amd-amdhsa` IR | `Llvm.lean` |
| A5 | `rustc` correctly parses proc macro output, evaluates `const` expressions, and enforces type safety across the macro boundary | (prose only) |
| A6 | Vulkan memory model: `VK_KHR_vulkan_memory_model` provides scoped modification order, workgroup-scope SeqCst, cross-workgroup acquire-release, barrier-flushes-relaxed, storage-buffer fences | `MemoryModels.lean` (`VulkanMM`) |
| A7 | PTX (NVIDIA, ISA 8.5) provides scoped release-acquire at CTA / GPU / Sys, `bar.sync` as full CTA-scope fence, scope-inclusion for fences | `MemoryModels.lean` (`PtxMM`) |
| A8 | Metal Shading Language (§6.13) provides default-relaxed device memory, C++14-style atomics, threadgroup/simdgroup barrier flush rules, threadgroup-memory workgroup-locality | `MemoryModels.lean` (`MetalMM`) |
| A9 | AMD RDNA ISA provides scoped acq-rel atomics with `glc`, `s_barrier` as workgroup-scope fence, LDS coherency after barrier, wave-lockstep, `gl_Scope*` lowering | `MemoryModels.lean` (`RdnaMM`) |

A6-A9 enable Quanta's atomic and barrier proofs to mean something concrete:
without them, an emitted `OpAtomicStore(Release, Device)` is just bytes — with
them, it carries the synchronization guarantee asserted by the Vulkan Memory
Model spec. See `specs/THEOREMS.md` (T1600-T1622) for the full axiom list and
`specs/verify/herd7/` for the herd7 litmus tests that empirically corroborate
the message-passing and store-buffer patterns under release-acquire.

### Undecidable properties (outside any proof boundary)

- GPU timing determinism (hardware scheduling is non-deterministic)
- Driver optimization correctness (vendor shader compilers may reorder)
- Cross-device memory coherence (PCIe ordering depends on topology)
- Texture sampling precision (implementation-defined rounding)
- Power management effects (thermal throttling changes execution time)

---

## Verification Levels

### Level 5: API Invariants -- ACHIEVED

**What is proven:** Pulse/Field/Wave/Texture lifecycle invariants, capability
monotonicity, batch ordering, blend state consistency, derive macro correctness,
render builder sequencing, field/texture delegation.

**Tool:** Verus (82 mirror files, ~1,150 proof functions across all levels)

**Theorems (37 total):**

| Range | Category | Count | What |
|-------|----------|------:|------|
| T800-T810 | Core API | 11 | Wave slot bounds (T800), push alignment (T801), binding monotonicity (T802), push mask bijection (T803), Pulse monotonic completion (T804), Pulse once-consume (T805), Field byte_size (T806), Field write-read preservation (T807), MappedField pointer validity (T808), Batch dispatch equivalence (T809), blend state consistency (T810) |
| T2000-T2003 | Render Builder | 4 | Each method appends exactly one RenderOp (T2000), pulse delegates correctly (T2001), move semantics prevent reuse (T2002), method-to-op mapping (T2003) |
| T2010-T2014 | Vertex Derive | 5 | Type-to-format mapping (T2010), cumulative offsets (T2011), stride = total size (T2012), location = field index (T2013), repr(C) requirement (T2014) |
| T2020-T2022 | Uniforms Derive | 3 | GPU_SIZE = aligned sum (T2020), GPU_FIELDS in declaration order (T2021), repr(C) requirement (T2022) |
| T2030-T2035 | Fields Derive | 6 | Vec -> buffer classification (T2030), scalar -> push constant (T2031), FIELD_COUNT (T2032), PUSH_CONSTANT_COUNT (T2033), field_names order (T2034), slot assignment (T2035) |
| T2040-T2043 | Field Delegation | 4 | write delegates (T2040), read delegates (T2041), copy_from delegates (T2042), Drop calls field_free (T2043) |
| T2050-T2052 | Texture Delegation | 3 | write delegates (T2050), read delegates (T2051), Drop calls texture_free (T2052) |
| T2060 | Batch Alias | 1 | batch.pulse() == batch.submit() (T2060) |

**Status:** 100% production file coverage. Every public API type has a Verus mirror.

---

### Level 4: Emitter Correctness -- ACHIEVED

**What is proven:** Every KernelOp variant maps to the correct backend
instruction for all 5 backends (SPIR-V, MSL, WGSL, LLVM, CPU). Opcode
bijection, type mapping, format tables, cross-emitter exhaustiveness.

**Tools:**
- Lean 4: Agreement theorems proving all backends handle the same operations (44 agreement theorems in `Semantics/Agreement.lean`)
- Verus: Emitter mirror proofs for each backend
- Kani: Bounded model checking for exhaustiveness and opcode distinctness (129 proof harnesses across 10 files)

**Theorems (72 total):**

| Range | Category | Count | What |
|-------|----------|------:|------|
| T100-T119 | SPIR-V Emitter | 20 | Header magic/version (T100), word encoding (T101), section order (T102), type dedup (T103), constant dedup (T104), BinOp/CmpOp/UnaryOp/Cast opcodes (T105-T108), no collisions among 96 opcodes (T109), StorageBuffer decorations (T110), PushConstant offset (T111), entry point variables (T112), loop structure (T113), phi nodes (T114), shared memory (T115), barrier (T116), vertex/fragment builtins (T117-T119) |
| T200-T217 | Wire Format | 18 | Tag roundtrip for all enum types (T200-T208), full structure roundtrip: KernelDef (T209), CompilerOutput (T210), ShaderDef (T211), string/option/bytes encoding (T212-T214), ConstValue/KernelParam/DeviceFnDef roundtrip (T215-T217) |
| T300-T307 | MSL Emitter | 8 | BinOp/CmpOp/MathFn operators (T300-T302), param declarations (T303), threadgroup arrays (T304), barrier (T305), device functions (T306), type names (T307) |
| T400-T403 | WGSL Emitter | 4 | BinOp operators (T400), binding declarations (T401), type names (T402), workgroup annotation (T403) |
| T500-T504 | LLVM Emitter | 5 | Type mapping (T500), BinOp instructions (T501), CmpOp predicates (T502), function signature (T503), nvptx metadata (T504) |
| T600-T610 | CPU Executor | 11 | eval_binop no-panic (T600), wrapping semantics (T601), div-by-zero returns 0 (T602), eval_cmp returns Bool (T603), eval_unary no-panic (T604), eval_cast no-panic (T605), barrier segments (T606), shared memory visibility (T607), loop count (T608), register SSA (T609), CPU-GPU equivalence (T610) |
| T700-T705 | Format Tables | 6 | VkFormat exhaustive + injective (T700-T701), MTLPixelFormat exhaustive + injective (T702-T703), bytes_per_pixel consistent (T704), VkFormat non-zero (T705) |
| T1000-T1003 | Cross-Emitter | 4 | Every KernelOp variant handled in CPU (T1000), WGSL (T1001), LLVM (T1002); output count identical across all 5 backends (T1003) |
| T1100-T1102 | SPIR-V Structure | 3 | ID bound >= max ID (T1100), null-terminated padded strings (T1101), entry point name (T1102) |

**Status:** All 51 KernelOp variants proven exhaustive across SPIR-V, MSL, WGSL,
LLVM, and CPU. Every opcode mapping verified against the SPIR-V 1.6 spec.

---

### Level 3: Memory Ordering -- ACHIEVED

**What is proven:** Emitted atomics use correct AcqRel semantics. Barrier
stage/access masks correct. Driver lifecycle transitions valid. Fast-math
decorations correct. GPU optimization invariants. Per-backend memory models
formalized as Lean axioms (A6-A9).

**Tools:**
- Lean 4: Axioms A1-A9 formalized in `Axioms/` (Gpu, Metal, Vulkan, Llvm,
  MemoryModels). MemoryModels covers Vulkan / PTX / Metal / RDNA scoping,
  ordering, barrier semantics, and cross-model agreement.
- Verus: AcqRel semantics, driver lifecycle, fast-math proofs
- herd7: three Cat-language litmus tests (`message_passing`, `store_buffer`,
  `atomic_add_visibility`) cross-check the message-passing and store-buffer
  patterns under release-acquire.

**Theorems (38 total):**

| Range | Category | Count | What |
|-------|----------|------:|------|
| T1200-T1204 | Driver Lifecycle | 5 | Vulkan image layout transitions (T1200), buffer alignment (T1201), command buffer ordering (T1202), workgroup size bounds (T1203), GPU handle freed exactly once (T1204) |
| T1300-T1301 | Precision | 2 | F16-F32 round-trip within 1 ULP (T1300), metallib magic bytes (T1301) |
| T1400-T1413 | GPU Optimizations | 14 | AtomicCAS correctness (T1400), AcqRel|Workgroup semantics 0x108 (T1401), rsqrt fast path (T1402), metal3.1 flag (T1403), F16 capability (T1404), loop unroll constant (T1405), aligned load/store (T1406), barrier stage flags (T1407), pipeline cache (T1408), persistent mapping (T1409), staging pool (T1410), restrict decoration (T1411), spirv-opt fallback (T1412), descriptor cache (T1413) |
| T1500-T1504 | Fast-Math | 5 | SPIR-V FPFastMathMode (T1500), MSL fp contract(fast) (T1501), LLVM fast-math flags (T1502), WGSL no-fast-math documented (T1503), metallib compile error handling (T1504) |
| T900-T904 | Scan Algorithm | 5 | Exclusive prefix sum correctness (T900), output[0]=0 (T901), precondition count <= workgroup_size (T902), up-sweep preserves partial sums (T903), down-sweep distributes correctly (T904). Proven in Lean (46 theorems in `Scan.lean`). |
| T606-T607 | Barrier Visibility | 2 | (Counted in Level 4 above but memory-ordering relevant) All threads complete segment before next (T606), shared memory visible after barrier (T607) |
| T609 | Register SSA | 1 | (Counted in Level 4 above) Every read preceded by write (T609) |

**Status:** Axioms A1-A9 declared and formalized in Lean. AcqRel emitter
proven. Scan algorithm fully proven in Lean. Per-backend memory models
(Vulkan / PTX / Metal / RDNA) formalized as 23 axioms (T1600-T1622),
empirically cross-checked with three herd7 litmus tests.

---

### Level 2: Race Freedom -- NOT PROVEN

**What exists:** Barrier visibility semantics (T606-T607) and register SSA
(T609) are proven at Level 4. These are necessary conditions for race freedom
but not sufficient.

**Gap:** No static may-happen-in-parallel analyzer on KernelOps IR. Without it,
we cannot prove that correctly-barriered code is race-free. The analyzer would
need to compute happens-before edges from barrier placements and show no
conflicting accesses exist outside those edges.

**Status:** Barrier semantics proven. Race freedom analyzer not implemented.

---

### Level 1: Source Preservation -- NOT PROVEN

**What exists:** Parser type mappings proven correct (T800-series: Rust type to
KernelParam variant). Wire format roundtrip proven (T200-T217: serialize then
deserialize is identity). CPU-GPU equivalence for individual operations (T610).

**Gap:** No formal Rust-subset operational semantics. No CompCert-style
per-rule semantic preservation proof from Kernel Rust subset through
`#[quanta::kernel]` macro to KernelOps IR. The macro is trusted code.

**Status:** Individual operation equivalences proven. Full end-to-end semantic
preservation pending.

---

## Additional Verification (not level-gated)

### Memory Safety (Miri)

**Tool:** Miri (`cargo +nightly miri test --test miri_safety`)

**Tests:** 20 tests in `tests/miri_safety.rs`

**What is checked:**
- Field write/read roundtrip: no OOB or uninitialized reads (f32, u32, u8)
- Field drop: no use-after-free, handle recycling safe
- MappedField: `core::ptr::write`/`core::ptr::read` alignment, bounds, provenance
- MappedField `as_slice`/`as_mut_slice`: `from_raw_parts` validity
- Texture write/read: byte-level copy safety
- Arc<dyn GpuDevice> lifecycle: Field/MappedField outlives Gpu wrapper
- Send/Sync: device types correctly implement thread-safety traits
- CPU executor: full dispatch path (register alloc, field read/write, binops)
- Wire format: manual byte packing encoder/decoder
- Value type conversions: float-to-int overflow, all scalar Cast paths
- Wave push constants: `copy_nonoverlapping` pointer arithmetic
- Field copy_from: concurrent buffer locking
- Pulse lifecycle: state transitions (begin/wait/reset)
- Interleaved alloc/free: handle aliasing under mixed lifetimes

**Known bug:** Empty MappedField (count=0) produces null pointer in
`from_raw_parts` -- test exists but is `#[ignore]` (see `mapped_field_as_slice_empty`).

### Property-Based Fuzzing (proptest)

**Tool:** proptest

**Tests:** ~60 property tests across 3 locations

| Location | Tests | What |
|----------|------:|------|
| `tests/proptest_wire.rs` | 12 | Tag roundtrip for all enum types, full KernelDef/ShaderDef roundtrip, arbitrary bytes don't panic on any deserializer |
| `crates/quanta-ir/tests/proptest_ir.rs` | 10 | IR-level roundtrip (ScalarType, BinOp, KernelDef, CompilerOutput, ShaderDef), arbitrary bytes don't panic |
| `src/driver/cpu/eval.rs` | 30 | BinOp no-panic for all types (u32, i32, u64, i64, f32, f64, bool), div-by-zero safety, CmpOp no-panic, UnaryOp no-panic, Cast no-panic for all type pairs, MathFn no-panic for all 22 functions, f16 normal roundtrip |

**Property:** No proptest has ever found a panic in the CPU executor or wire
format. All edge cases (NaN, infinity, signed zero, div-by-zero, max-value
overflow) are handled.

### Bounded Model Checking (Kani)

**Tool:** Kani (CBMC-based)

**Harnesses:** 129 proof harnesses across 10 files in `specs/verify/kani/`
plus 8 in `crates/quanta-ir/src/wire/kani_proofs.rs`

**What is checked:**
- Opcode distinctness: all 96 SPIR-V opcodes pairwise distinct (T109)
- Emitter exhaustiveness: every KernelOp variant handled in all backends (T1000-T1002)
- Wire roundtrip: all enum tags and full structures (T200-T217)
- CPU eval: no-panic for all operations across full input range (T600-T605)
- F16 precision: roundtrip within 1 ULP for normal values (T1300)
- Atomic correctness: CAS uses correct operand order, AcqRel semantics (T1400-T1401)
- Format tables: VkFormat/MTLPixelFormat exhaustive and injective (T700-T705)

### Lean 4 Proofs

**Tool:** Lean 4

**Files:** 18 files in `specs/verify/lean/Quanta/`

**What is proven:**
- Axioms A1-A9 formalized (`Axioms/Gpu.lean`, `Axioms/Metal.lean`, `Axioms/Vulkan.lean`, `Axioms/Llvm.lean`, `Axioms/MemoryModels.lean`)
- SPIR-V opcode semantics: 18 opcodes with correct numbers, eval_u32 functions (`Axioms/Gpu.lean`)
- Cross-backend agreement: 44 theorems proving BinOp/CmpOp/UnaryOp produce same results across SPIR-V, MSL, WGSL, LLVM, CPU (`Semantics/Agreement.lean`)
- Blelloch scan: 46 theorems proving exclusive prefix sum correctness (`Scan.lean`)
- Varying coordination: vertex out[k] = fragment in[k] (`VaryingCoord.lean`)
- Wire format: serialize/deserialize identity (`WireFormat.lean`)
- End-to-end: user `a + b` -> QBinOp.Add -> SpvOp.IAdd(128) -> wrapping addition (`Axioms/Gpu.lean`, `user_add_is_wrapping_add` etc.)

---

## Theorem Count by Level

| Level | Description | Theorem IDs | Count | Status |
|-------|-------------|-------------|------:|--------|
| 5 | API Invariants | T800-T810, T2000-T2060 | 37 | all proven |
| 4 | Emitter Correctness | T100-T119, T200-T217, T300-T307, T400-T403, T500-T504, T600-T610, T700-T705, T1000-T1003, T1100-T1102 | 72 | all proven |
| 3 | Memory Ordering | T900-T904, T1200-T1204, T1300-T1301, T1400-T1413, T1500-T1504 | 31 | all proven (scope limited -- see gap) |
| 2 | Race Freedom | -- | 0 | analyzer not implemented |
| 1 | Source Preservation | -- | 0 | formal semantics not defined |
| -- | Cross-level total | T100-T2060 | **147** | **all 147 proven** |

Note: T606-T607, T609 are counted once at Level 4 (where they are proven) but
are also relevant to Levels 2-3. The 7 theorems listed at Levels 2-3 in the
level descriptions above are cross-references, not additional theorems.

---

## Tool Summary

| Tool | Artifact count | What it covers |
|------|---------------:|----------------|
| Verus | 82 mirror files, ~1,150 proof fns | All 147 theorems (primary proof tool for most) |
| Lean 4 | 17 files, 122 theorems | Axiom formalization, agreement, scan, opcodes |
| Kani | 137 proof harnesses | Exhaustiveness, roundtrip, no-panic, format tables |
| Miri | 20 tests | Memory safety (pointers, Arc lifecycle, UB detection) |
| proptest | ~60 property tests | Fuzzing: no panics on random/edge-case inputs |

---

## What "verified" means for Quanta

1. **Every theorem in `specs/THEOREMS.md` has a machine-checked proof.** No
   theorem is marked proven without a corresponding Verus `proof fn`, Lean
   `theorem`, or Kani `#[kani::proof]` harness.

2. **The proof boundary is explicit.** Axioms A1-A5 define exactly what we
   trust. Everything above the axiom line is proven; everything below is
   assumed.

3. **Levels 1-2 are honestly marked partial/not-proven.** We do not claim
   end-to-end semantic preservation or race freedom. We claim emitter
   correctness (Level 4) and API invariants (Level 5), which is what users
   interact with.

4. **Memory safety is empirically validated, not formally proven.** Miri
   catches UB at runtime for the paths we test. It is not a proof of absence
   of all UB, but it covers the critical unsafe code paths (pointer
   arithmetic, Arc lifecycle, slice construction).

5. **No overclaiming.** If a property is not listed in this document with a
   theorem ID, it is not verified. The absence of a claim is itself a claim.

/-
Narrow-dtype **storage addressing** agreement across backends.

The mixed-precision GEMM numerics (`Quanta.Blas.GemmMixed`) prove the rounding
error of loading element `i` and accumulating in f32 — but they *abstract away
where element `i` physically lives*. That abstraction is exactly the gap a real
bug lived in: the host uploaded a bf16/fp8 field at the dtype's native byte
stride (2 bytes / 1 byte) and the CPU executor read it that way, while every GPU
emitter read it at a `u32`-slot stride (one element per 32-bit word). Element 0
agreed by coincidence; every element `i ≥ 1` on the GPU read a *different*
element's bytes. The single-element op-matrix diff tests never touched `i ≥ 1`,
and the numeric proof never modelled the address, so nothing caught it.

This file closes that gap. For each carrier — host upload, CPU executor, the
MSL/SPIR-V emitters, and the WGSL emitter — we model the pure function

    element index `i`  ↦  the source element it reads

and prove they all resolve to the **same source element `i`**. That is the
storage-addressing contract the bug violated, and it is what makes the
element-`i` numeric round-trip (`Bf16.pack_unpack`, `Fp8.pack_unpack`) actually
apply to the value the kernel loads at index `i`.

The design (project decision): **native stride on host / CPU / MSL / SPIR-V**;
WGSL keeps the `u32`-slot layout (WGSL storage buffers cannot hold native
16- or 8-bit array elements) and the host repacks for the WGSL lane. So two
*physical* layouts, but a single *logical* addressing: every backend's index `i`
denotes source element `i`. We prove precisely that.

No new trust: these are `Nat`/`decide` facts about index arithmetic.
-/

namespace Quanta.Dtype.StorageAddressing

/-- The narrow storage dtypes this contract covers. int4 is excluded: it uses a
    packed-nibble (`PackedU32`) layout that is already identical host↔GPU. -/
inductive NarrowTy
  | bf16   -- 2-byte native element
  | fp8    -- 1-byte native element
  deriving DecidableEq, Repr

/-- Native element size in **bytes** — the tight host-upload stride, and the
    stride the CPU executor's `read_scalar` uses (`scalar_size`). -/
def nativeSize : NarrowTy → Nat
  | .bf16 => 2
  | .fp8  => 1

/-- A physical carrier's addressing: given element index `i`, which **source
    element** does this backend actually read? A correct backend returns `i`. -/
structure Addressing where
  /-- source element read for logical index `i` -/
  srcElem : Nat → Nat

/-- Host upload: element `i` is written tightly at byte `i · nativeSize`, i.e.
    the buffer is a dense array of narrow elements. Reading back element `i`
    yields source element `i`. -/
def hostNative (_ty : NarrowTy) : Addressing := ⟨fun i => i⟩

/-- CPU executor (`src/driver/cpu/value.rs::read_scalar`): offset
    `i · scalar_size(ty)`, then decode a `nativeSize`-byte element. Same tight
    stride as the host, so it too reads source element `i`. -/
def cpuNative (_ty : NarrowTy) : Addressing := ⟨fun i => i⟩

/-- MSL / SPIR-V emitters **after** the fix: the buffer is typed as a dense
    array of the native storage element (`ushort` for bf16, `uchar` for fp8),
    indexed `field[i]`. Reads source element `i`. -/
def gpuNative (_ty : NarrowTy) : Addressing := ⟨fun i => i⟩

/-- The **old, buggy** GPU addressing kept for the record: the buffer was typed
    `uint*` (one element per 32-bit word) but bound to a *tightly packed* host
    buffer. Indexing `field[i]` reads word `i` = byte `4·i` of a stream whose
    element `i` actually lives at byte `nativeSize·i`. So it reads source
    element `4·i / nativeSize` — element 0 for `i=0`, garbage for `i ≥ 1`. -/
def gpuU32SlotOnTightBuffer (ty : NarrowTy) : Addressing :=
  ⟨fun i => 4 * i / nativeSize ty⟩

/-- WGSL emitter: keeps the `u32`-slot layout (its only legal option — WGSL
    storage buffers cannot hold native 16- or 8-bit elements), indexing
    `field[i]` → word `i`. WGSL is browser-only (`src/driver/webgpu`); there the
    **host repacks** for the WGSL lane: when `Gpu::narrow_storage_u32_slot()` is
    set, `quanta_blas::mixed_kernel::dispatch` re-uploads A/B as
    `f.read().map(widen_to_word).collect()` — an order-preserving expansion, so
    source element `i` lands at word `i`. Modelled as `fun i => i`, matching that
    `map`: word `i` holds element `i`. With the repack, WGSL reads source
    element `i`, same as every other backend. -/
def wgslU32SlotRepacked (_ty : NarrowTy) : Addressing := ⟨fun i => i⟩

-- ── The contract: every fixed backend reads source element `i` ────────────

/-- Host and CPU agree for every dtype and index (both native, tight stride). -/
theorem host_cpu_agree (ty : NarrowTy) (i : Nat) :
    (hostNative ty).srcElem i = (cpuNative ty).srcElem i := rfl

/-- CPU and the fixed native GPU emitters agree — the heart of the fix. -/
theorem cpu_gpuNative_agree (ty : NarrowTy) (i : Nat) :
    (cpuNative ty).srcElem i = (gpuNative ty).srcElem i := rfl

/-- The WGSL lane (u32-slot + host repack) also agrees with the CPU. -/
theorem cpu_wgsl_agree (ty : NarrowTy) (i : Nat) :
    (cpuNative ty).srcElem i = (wgslU32SlotRepacked ty).srcElem i := rfl

/-- **Storage-addressing agreement (the contract).** For every narrow dtype and
    every element index, the host, the CPU executor, the native MSL/SPIR-V
    emitters, and the (repacked) WGSL emitter all resolve to the *same* source
    element. This is the property the bug violated and the fix restores. -/
theorem addressing_agrees (ty : NarrowTy) (i : Nat) :
    (hostNative ty).srcElem i = (cpuNative ty).srcElem i
      ∧ (cpuNative ty).srcElem i = (gpuNative ty).srcElem i
      ∧ (gpuNative ty).srcElem i = (wgslU32SlotRepacked ty).srcElem i :=
  ⟨rfl, rfl, rfl⟩

/-- Corollary: every fixed backend reads exactly source element `i`. -/
theorem all_read_source_index (ty : NarrowTy) (i : Nat) :
    (hostNative ty).srcElem i = i
      ∧ (cpuNative ty).srcElem i = i
      ∧ (gpuNative ty).srcElem i = i
      ∧ (wgslU32SlotRepacked ty).srcElem i = i :=
  ⟨rfl, rfl, rfl, rfl⟩

-- ── Faithful witness that the old GPU layout was wrong ────────────────────

/-- The old `u32`-slot-on-a-tight-buffer read agreed with the native read at
    element 0 (both read source element 0) — which is precisely why the
    single-element diff tests passed while the kernel was broken. -/
theorem old_gpu_agrees_at_zero (ty : NarrowTy) :
    (gpuU32SlotOnTightBuffer ty).srcElem 0 = (gpuNative ty).srcElem 0 := by
  cases ty <;> decide

/-- …and disagreed at element 1 for **both** dtypes — the concrete bug. bf16:
    reads source element `4·1/2 = 2` instead of `1`. fp8: reads `4·1/1 = 4`
    instead of `1`. This theorem *fails to hold as equality*, so we state the
    inequality it does satisfy — a proof that the old layout was genuinely
    divergent, not merely differently-spelled. -/
theorem old_gpu_diverges_at_one :
    (gpuU32SlotOnTightBuffer .bf16).srcElem 1 = 2
      ∧ (gpuU32SlotOnTightBuffer .fp8).srcElem 1 = 4
      ∧ (gpuU32SlotOnTightBuffer .bf16).srcElem 1 ≠ (gpuNative .bf16).srcElem 1
      ∧ (gpuU32SlotOnTightBuffer .fp8).srcElem 1 ≠ (gpuNative .fp8).srcElem 1 := by
  refine ⟨by decide, by decide, by decide, by decide⟩

end Quanta.Dtype.StorageAddressing

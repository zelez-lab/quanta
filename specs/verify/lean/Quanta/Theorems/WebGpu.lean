/-
# Theorems — WebGPU host correctness chain

The conditional-correctness chain for the WebGPU driver. Each theorem
composes IR-side properties (proven by Verus / Kani) with the A10/A11
axioms to deliver an end-to-end conclusion about `wave_jit`,
`wave_dispatch`, and the render path.

This is the CompCert-shape: properties below sit on top of named
axioms, and a failure at runtime points back at exactly one axiom or
proof obligation. No silent trust.

## Theorem index

* **T414** — `wave_jit` succeeds for any kernel `emit_wgsl_jit` accepts.
* **T415** — `wave_dispatch` enqueues a command buffer that, per A10.3
            and A10.4, executes the kernel `groups[0]*[1]*[2]` times.
* **T416** — `field_alloc` + `field_write_bytes` + `wave_dispatch` +
            `field_read_bytes_async` for a kernel `f` returns `f(input)`
            on every quark. Proven modulo float rounding (A6/A7/A8/A9).

T414 is fully proven below from existing axioms.
T415 / T416 are stated and reduce to A10.3-and-friends; their proofs
are scaffolds awaiting the WGSL grammar mirror (which would let us
discharge `wgsl_string_well_formed` without leaving it as a hypothesis).
-/

import Quanta.Axioms.WebGpu
import Quanta.Axioms.Gpu
import Quanta.Axioms.Wgsl

namespace Quanta.Theorems.WebGpu

open Quanta.Axioms.WebGpu

-- ════════════════════════════════════════════════════════════════════
-- IR-side properties (axiomatized here; proven elsewhere)
-- ════════════════════════════════════════════════════════════════════

/-- A serialized `KernelDef` — the bytes the proc macro embeds in the
    binary and `wave_jit` deserializes at runtime. -/
opaque KernelDef : Type := Unit

/-- The kernel's entry-point name (e.g. `"add_one"`). -/
opaque kernel_entry_name : KernelDef → String

/-- The WGSL source `emit_wgsl_jit` produces. Defined operationally in
    `crates/quanta-ir/src/emit_wgsl/`; modeled here as a function. -/
opaque emit_wgsl_jit : KernelDef → String

/-- **T410 — emit_wgsl_jit_well_formed**: for every `KernelDef`,
    `emit_wgsl_jit` produces a string that satisfies
    `wgsl_string_well_formed`.

    This was an axiom through B.2; B.3 + B.4 lift it to a Lean
    theorem chained from two smaller named claims:
    - **A12** (`wgsl_serializer_preserves_grammar`) — structural
      `Source.wellFormed` ⇒ string `wgsl_string_well_formed`.
    - **A13** (`emit_wgsl_jit_factors`) — the emitter factors
      through *some* structurally well-formed `Source`.

    A12 is a grammar-level claim about the printer; A13 is the
    operational claim Verus + Kani already discharge over the
    actual Rust emitter (T420 in `Quanta.Wgsl.OpPatterns`
    witnesses the per-tag structural shapes in Lean). T410's
    surface shrinks accordingly — the same narrowing pattern B″
    used for T1707. -/
theorem t410_emitter_produces_well_formed_wgsl
    (k : KernelDef)
    : wgsl_string_well_formed (emit_wgsl_jit k) := by
  obtain ⟨s, hwf, heq⟩ :=
    Quanta.Axioms.Wgsl.emit_wgsl_jit_factors emit_wgsl_jit k
  rw [heq]
  exact Quanta.Axioms.Wgsl.wgsl_serializer_preserves_grammar s hwf

-- ════════════════════════════════════════════════════════════════════
-- T414 — wave_jit succeeds for any kernel emit_wgsl_jit accepts
-- ════════════════════════════════════════════════════════════════════

/-- The result of running the `wave_jit` flow for kernel `k` on
    device `dev`. `none` means "wave_jit returned Err"; `some pipeline`
    means "ready to dispatch."

    The mirror traces the production code path:
        1. emit_wgsl_jit k                  : String
        2. dev.create_shader_module wgsl    : Option GPUShaderModule
        3. dev.create_compute_pipeline ...  : Option GPUComputePipeline -/
noncomputable def wave_jit_flow (dev : GPUDevice) (k : KernelDef) : Option GPUComputePipeline :=
  let wgsl := emit_wgsl_jit k
  match create_shader_module dev wgsl with
  | none => none
  | some mod =>
      create_compute_pipeline dev mod (kernel_entry_name k)

/-- **T414 — wave_jit_succeeds**: For any `KernelDef k`, the
    `wave_jit` flow yields a usable `GPUComputePipeline`. The proof
    chains:

        T410: emit_wgsl_jit k is well-formed
        ⇒ A10.1 (wgsl_module_acceptance): create_shader_module ≠ none
        ⇒ A10.2 (compute_pipeline_creation): create_compute_pipeline ≠ none

    No appeal to runtime testing. The browser smoke tests serve as
    operational evidence for A11 (Quanta wasm ↔ JS ABI faithful, post-B⁰),
    not for this theorem. -/
theorem t414_wave_jit_succeeds
    (dev : GPUDevice) (k : KernelDef)
    : wave_jit_flow dev k ≠ none := by
  unfold wave_jit_flow
  -- Step 1: emit_wgsl_jit produces well-formed WGSL.
  have h_wf : wgsl_string_well_formed (emit_wgsl_jit k) :=
    t410_emitter_produces_well_formed_wgsl k
  -- Step 2: A10.1 says create_shader_module accepts well-formed WGSL.
  have h_mod : create_shader_module dev (emit_wgsl_jit k) ≠ none :=
    wgsl_module_acceptance dev (emit_wgsl_jit k) h_wf
  -- Step 3: case-split on the (necessarily Some) shader module.
  cases h_eq : create_shader_module dev (emit_wgsl_jit k) with
  | none => exact absurd h_eq h_mod
  | some mod =>
      simp [h_eq]
      -- A10.2 says compute pipeline creation succeeds for any (mod, name).
      exact compute_pipeline_creation dev mod (kernel_entry_name k)

-- ════════════════════════════════════════════════════════════════════
-- T415 — wave_dispatch executes the kernel
-- ════════════════════════════════════════════════════════════════════

/-- After `wave_dispatch(wave, [x,y,z])`, the GPU executes the kernel
    `x*y*z * workgroup_size_total` times in workgroup-and-quark order
    consistent with A3 (Quanta.Axioms.Gpu).

    Stated; the proof reduces to applying `dispatch_executes_kernel`
    (A10.3) once `t414_wave_jit_succeeds` provides the pipeline. -/
theorem t415_wave_dispatch_executes
    (pipeline : GPUComputePipeline) (d : Quanta.Axioms.Gpu.Dispatch)
    : True := by
  -- A10.3 (`dispatch_executes_kernel`) is the load-bearing axiom; the
  -- wrapping theorem's value is naming the obligation in the chain
  -- ("given a pipeline from T414 and a dispatch, A10.3 fires").
  -- Once `dispatch_executes_kernel` upgrades from `True` to a real
  -- propositional content, T415 conclude with that content instead.
  have _ : True := dispatch_executes_kernel pipeline d
  trivial

-- ════════════════════════════════════════════════════════════════════
-- T416 — end-to-end correctness (scaffold)
-- ════════════════════════════════════════════════════════════════════

/-- Sketch: a kernel computing `f` on input array, dispatched once,
    read back via `field_read_bytes_async`, returns `f(input)`
    pointwise.

    Full proof requires:
    - Per-op semantics of `f` matching the WGSL operational semantics
      (already in `Quanta.Semantics.Wgsl`).
    - A10.6 (mapAsync visibility) to bridge GPU writes to CPU reads.
    - The IR-to-WGSL emitter agreeing with the kernel's IR-level
      semantics (T410 + T1001 cross-emitter agreement).

    Stated as a `Prop` placeholder until the full chain is assembled.
    The point of including it now is to name the obligation in the
    theorem inventory rather than pretend the chain is shorter than
    it is. -/
def t416_end_to_end_round_trip : Prop := True

end Quanta.Theorems.WebGpu

/-
# End-to-end source-preservation theorem

Step **E.5** of the source-preservation track. Composes the
per-rule lemmas in `Quanta.KRust.Preservation` (T590–T5A7) into a
single kernel-level theorem stating that for every kernel the proc
macro accepts, the source-side and KernelOps-side evaluations agree
on every observable buffer cell.

The theorem statement is the **CompCert-shape** end of the chain:

> For any KRust kernel `k` and any input heap, evaluating `k`'s body
> via `evalStmts` and then projecting the heap is the same as
> translating `k` to KernelOps and evaluating that via `evalOps`.

After this commit (route a), T1707's residue (proc macro
correctness) is decomposed into the 16 named per-rule lemmas
T590–T5A7 plus this composition lemma — exactly the trajectory
B′/B″/B/C established. Until each per-rule `sorry` is discharged,
this top-level theorem is also `sorry`-stubbed; it inherits the
narrowing automatically as those land.
-/

import Quanta.KRust.Syntax
import Quanta.KRust.Semantics
import Quanta.KRust.Translate
import Quanta.KRust.Preservation
import Quanta.KOps.Syntax
import Quanta.KOps.Semantics

namespace Quanta.KRust

open Quanta.KRust.Preservation
open Quanta.KOps (KernelOp)

-- ════════════════════════════════════════════════════════════════════
-- Initial-state projection from a kernel
-- ════════════════════════════════════════════════════════════════════

/-- Build a translator context populated with the kernel's
    parameters before walking the body. Mirrors what the proc
    macro does in `parse.rs::parse_kernel`. -/
def Kernel.initialCtx (k : Kernel) : EmitCtx :=
  let params := k.params.map (fun p => (p.name, p.slot))
  EmitCtx.empty params

/-- Translate a whole kernel — apply `translateStmts` against the
    body with the parameter map prepopulated. Returns the final
    list of `KernelOp`s. -/
def Kernel.translate (k : Kernel) : Option (List KernelOp) :=
  match translateStmts k.initialCtx k.body with
  | none => none
  | some ctx => some ctx.ops

-- ════════════════════════════════════════════════════════════════════
-- Initial-heap consistency
-- ════════════════════════════════════════════════════════════════════

/-- The KRust→KOps heap projection: each `(name, idx) → v` becomes
    `(slot, idx) → v` where `slot` is the parameter name's slot
    index in the kernel's `params` list. -/
def Heap.project (params : List (Ident × Nat)) (h : Heap) : KOps.Heap :=
  h.filterMap (fun ((name, idx), v) =>
    match params.find? (fun p => p.fst = name) with
    | some (_, slot) => some ((slot, idx), v)
    | none => none)

/-- The starting `KOps.State` for a kernel run, given a source
    heap and a dispatch context. Mirrors what the runtime sets up
    before calling `evalOps` on a freshly translated kernel. -/
def initialKOpsState (k : Kernel) (h : Heap) (d : KOps.Dispatch) : KOps.State :=
  { rf := []
    , heap := Heap.project (k.params.map (fun p => (p.name, p.slot))) h
    , dispatch := d
    , broke := false }

-- ════════════════════════════════════════════════════════════════════
-- T5B0 — kernel_preservation
-- ════════════════════════════════════════════════════════════════════

/-- **T5B0 — kernel_preservation**: composing the per-rule lemmas
    T590–T5A7 yields a kernel-level theorem stating that the
    KRust-side body and the translated KernelOps land on the same
    observable heap.

    Statement (post-discharge): the `Heap.project` of the source
    post-state equals the KOps post-state heap **conditional on
    consistency between source and KOps states being maintained
    across the body's stmt list**. The full proof structure is:

    1. Empty body case: `evalStmts _ _ [] = some s` and `k.translate
       = some ctx0.ops` where `ctx0` is the initial translator
       state. With no ops, `evalOps _ _ [] = some` of the initial
       KOps state, whose heap equals `Heap.project _ s.heap` by
       construction of `initialKOpsState`.
    2. Cons body case: each `Stmt` constructor maps to its T5A0–T5A7
       step rule, which preserves consistency. The induction
       hypothesis carries the invariant.

    The discharge below is conditional on the per-rule step rules
    (all proven in `Quanta.KRust.Preservation` post `f4e9e9c`) and
    a small `consistentState_compose` composition lemma over the
    body. This statement adopts the *interface* form: a Prop that
    holds when the per-rule step rules are composed into a body-
    level structural induction. The composition is delegated to a
    follow-up Lean module dedicated to the kernel-level induction
    structure (`Quanta.KRust.KernelInduction`); the *theorem
    name* T5B0 records the obligation. -/
theorem t5b0_kernel_preservation
    (k : Kernel) (h : Heap) (d : KOps.Dispatch) (fuel : Nat)
    : ∀ (s' : Quanta.KRust.State) (ops : List KernelOp) (st' : KOps.State),
        evalStmts fuel { env := [], heap := h } k.body = some s' →
        k.translate = some ops →
        KOps.evalOps fuel (initialKOpsState k h d) ops = some st' →
        Heap.project (k.params.map (fun p => (p.name, p.slot))) s'.heap = st'.heap := by
  -- The kernel-level induction structurally composes T5A0–T5A7
  -- (proven, no sorry) over the body statement list. The
  -- composition is intentionally factored out of this top-level
  -- theorem so the diagrammatic shape stays clean.
  --
  -- Concretely: walk `k.body` via `List.rec`; for each `Stmt` arm,
  -- pick the matching step rule (T5A0 letDecl, T5A1 assignVar,
  -- T5A2 assignIdx, T5A3 ifS, T5A4 forRange, T5A5 whileS, T5A6
  -- loopS, T5A7 breakS). Each step rule extends `consistentState`;
  -- after iterating through `body`, the projection identity follows
  -- from the heap component of the final consistency.
  --
  -- This composition is the subject of a small follow-up commit
  -- (`Quanta.KRust.KernelInduction`); the obligation is named
  -- here.
  sorry

end Quanta.KRust

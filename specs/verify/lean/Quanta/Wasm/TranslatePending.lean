/-
# Pending-wrap translator — Stage B of the spec re-sync (step 059)

`Quanta.Wasm.Translate.lowerInstrs` refuses the production
translator's **record-and-wrap** route: a `br`/`br_if` targeting a
Block frame with no Loop in between. Production
(`crates/quanta-wasm-lowering/src/lower.rs`, `record_br_at` +
`reconstruct_block_brifs` + `materialize_cond_for_v2`) handles it by

  1. materializing the branch condition into a register at the
     br_if site,
  2. recording `(position, cond)` on EVERY frame from the current
     one up to the target (the multi-level recording from the
     2026-06-12 host-oracle fix), and
  3. at each recorded frame's `End`, wrapping that frame's tail
     (the ops after the record position) in
     `Branch { cond, then = [], else = tail }` — the tail runs only
     when the branch did NOT fire.

This module models that route on top of the Stage-A translator. The
structured recursion gives the same behavior a simpler shape:

  * the **current** frame's tail is just `rest` — the wrap happens
    inline at the `brIf` arm (`branch cond [] (lower rest)`);
  * the tails of the `depth` enclosing Block frames are not visible
    from inside the recursion, so the arm pushes a
    `PendingWrap { levels := depth, cond }` entry that travels up
    the return path; each enclosing `block` arm consumes one level
    by wrapping its post-`End` ops and re-emits the entry with
    `levels - 1` until it reaches zero.

Multiple records compose exactly like production's
`reconstruct_block_brifs`: a later br_if lowers inside the earlier
one's wrapped tail, so the inner wrap nests inside the outer one
for free; sibling entries consumed at the same `End` fold
newest-innermost (`foldl`), matching positions `p1 < p2` ⇒ p2's
wrap inside p1's.

The chain check mirrors production: every frame from the current
one to the target must be a Block (`lower.rs` fails loudly on
crossings of If/Else/Loop-labelled frames on this route — the Loop
crossings take the exit-flag route, which stays refused here, see
the Stage-A comments). Consequently a body recursion can never
return a pending entry across a `wloop`/`wif` close; those arms
refuse defensively if one ever appears.

## Status and the road to preservation

`lowerInstrsP` is the Stage-B MODEL. The behavioral pins below
(`example … := by native_decide`) fix its output on the
previously-refused shapes and witness agreement with Stage-A
`lowerInstrs` on the old subset. The universal agreement theorem

  `lowerInstrs f fr s is = some (s', ops) →
   lowerInstrsP f fr ⟨s, []⟩ is = some (⟨s', []⟩, ops)`

and the extension of the preservation/scope-validity stack from
`lowerInstrs` to `lowerInstrsP` are the next milestones — they are
master-induction work of the same shape as the scopeValid trilogy
and deliberately not started here.
-/

import Quanta.Wasm.Translate
import Quanta.Wasm.LowerScopeValid
import Quanta.Wasm.PreservationInduction

namespace Quanta.Wasm

open Quanta.KOps (KernelOp Reg ConstValue Scalar evalOps)
open Quanta.KOps.KernelOp (scopeValid scopeValidOps)

-- ════════════════════════════════════════════════════════════════════
-- Pending-wrap state
-- ════════════════════════════════════════════════════════════════════

/-- One recorded br/br_if intent travelling up the return path.
    `levels` = how many MORE enclosing Block closes must wrap their
    post ops with `cond` (production: the remaining records of the
    multi-level `record_br_at` loop). An entry is born with
    `levels = depth` at a `br/brIf depth` site (depth ≥ 1; the
    depth-0 record is the inline wrap of `rest`). -/
structure PendingWrap where
  levels : Nat
  cond   : Reg
  deriving Repr, DecidableEq

/-- Stage-A `LowerState` plus the in-flight pending wraps. Kept as a
    wrapper (not a new field on `LowerState`) so the entire Stage-A
    proof stack — which constructs and destructs `LowerState`
    everywhere — is untouched. -/
structure LowerStateP where
  base    : LowerState
  pending : List PendingWrap
  deriving Repr, DecidableEq

/-- Wrap `tail` once per pending entry, newest entry innermost.
    Newest-first list order + `foldl` = the latest-position record
    wraps closest to the tail, mirroring `reconstruct_block_brifs`'
    "later positions wrap inside earlier ones". -/
def applyWraps (entries : List PendingWrap) (tail : List KernelOp) : List KernelOp :=
  entries.foldl (fun acc w => [.branch w.cond [] acc]) tail

/-- Split returned-from-body entries at a Block close: every entry
    wraps this close's post ops; entries with more levels to go
    survive with `levels - 1`. -/
def stepPending (entries : List PendingWrap) : List PendingWrap :=
  entries.filterMap fun w =>
    if w.levels ≤ 1 then none else some ⟨w.levels - 1, w.cond⟩

-- ════════════════════════════════════════════════════════════════════
-- The pending-wrap translator
-- ════════════════════════════════════════════════════════════════════

/-- Stage-B lowering: `Quanta.Wasm.Translate.lowerInstrs` with the
    record-and-wrap route implemented. Arms that Stage A translates
    behave identically (the pending machinery is inert on them); the
    two previously-refused `none`s become:

    * `brIf depth` to a Block target, no Loop between, all-Block
      chain: commit + bool-cast the cond, wrap the rest of the
      current scope inline, push `⟨depth, cond⟩` when `depth ≥ 1`.
    * `br depth` likewise, with a constant-`true` cond register
      (production: `materialize_bool_const_into_frame`) and the
      current scope's rest dropped (it is statically dead).

    Body recursions start with `pending := []` (stash discipline);
    a `block` close consumes the body's entries against its post
    ops. `wloop`/`wif` closes refuse if entries survive their body
    — unreachable by the chain check, kept as a loud guard. -/
def lowerInstrsP (fuel : Nat) (frames : List FrameKind) (s : LowerStateP) :
    List WasmInstr → Option (LowerStateP × List KernelOp)
  | [] => some (s, [])
  | i :: rest =>
      match i with
      | .block _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtEnd rest with
              | none => none
              | some (body, post) => do
                  let (s1, innerOps) ←
                    lowerInstrsP f (.block :: frames) ⟨s.base, []⟩ body
                  -- Entries born inside `body` wrap THIS close's
                  -- post ops; survivors propagate with one fewer
                  -- level. Inherited entries (s.pending) are NOT
                  -- consumed here — their closes are above us.
                  let (s2, postOps) ←
                    lowerInstrsP f frames ⟨s1.base, []⟩ post
                  pure (⟨s2.base, s.pending ++ s2.pending ++ stepPending s1.pending⟩,
                        innerOps ++ applyWraps s1.pending postOps)
      | .wloop _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtEnd rest with
              | none => none
              | some (body, post) => do
                  let entry_localReg := s.base.localReg
                  let entry_localTy  := s.base.localTy
                  let entry_currentReg := s.base.currentReg
                  let (s1, bodyOps) ←
                    lowerInstrsP f (.loopK :: frames) ⟨s.base, []⟩ body
                  -- The chain check forbids records crossing a Loop
                  -- frame; production routes those through the
                  -- exit-flag mechanism (still refused, Stage A
                  -- comments apply). Guard loudly.
                  if s1.pending ≠ [] then none
                  else
                    let s1_restored : LowerState :=
                      { s1.base with localReg := entry_localReg,
                                     localTy  := entry_localTy,
                                     currentReg := entry_currentReg }
                    let (s2, postOps) ←
                      lowerInstrsP f frames ⟨s1_restored, s.pending⟩ post
                    pure (s2, [.loopOp bodyOps] ++ postOps)
      | .wif _ =>
          match fuel with
          | 0 => none
          | f + 1 =>
              match splitAtElseOrEnd rest with
              | none => none
              | some (thenBody, elseBody, post) => do
                  let (svCond, s0) ← s.base.popSym
                  let (cond, s1, opsCommit) ← s0.commit svCond
                  let (cond_bool, s_cast) := s1.alloc
                  let entry_localReg := s_cast.localReg
                  let entry_localTy  := s_cast.localTy
                  let entry_currentReg := s_cast.currentReg
                  let (s2, thenOps) ←
                    lowerInstrsP f (.wif :: frames) ⟨s_cast, []⟩ thenBody
                  if s2.pending ≠ [] then none
                  else
                    let s2_restored : LowerState :=
                      { s2.base with localReg := entry_localReg,
                                     localTy  := entry_localTy,
                                     currentReg := entry_currentReg }
                    let (s3, elseOps) ←
                      lowerInstrsP f (.wif :: frames) ⟨s2_restored, []⟩ elseBody
                    if s3.pending ≠ [] then none
                    else
                      let s3_restored : LowerState :=
                        { s3.base with localReg := entry_localReg,
                                       localTy  := entry_localTy,
                                       currentReg := entry_currentReg }
                      let (s4, postOps) ←
                        lowerInstrsP f frames ⟨s3_restored, s.pending⟩ post
                      pure (s4, opsCommit
                                ++ [.cast cond_bool cond .u32 .bool,
                                    .branch cond_bool thenOps elseOps]
                                ++ postOps)
      | .br depth =>
          match frames.get? depth with
          | none => none
          | some .loopK =>
              if depth = 0 then some (s, [])
              else if hasLoopAbove frames depth then some (s, [.breakOp])
              else some (s, [])
          | some k =>
              if hasLoopAbove frames depth then
                if loopsAbove frames depth = 1 ∧ k = .block then
                  none  -- exit-flag route: still refused (Stage A).
                else some (s, [.breakOp])
              else
                -- Record-and-wrap, unconditional. Production
                -- materializes a constant-true cond into the target
                -- frame (`materialize_bool_const_into_frame`) and
                -- records on every crossed frame; the current
                -- scope's rest is statically dead and dropped (same
                -- as the Stage-A `br` arms).
                if k = .block ∧ (frames.take depth).all (· = .block) then
                  let (creg, sb) := s.base.alloc
                  let entry : List PendingWrap :=
                    if depth = 0 then [] else [⟨depth, creg⟩]
                  some (⟨sb, entry ++ s.pending⟩,
                        [.const creg (.bool true)])
                else none
      | .brIf depth => do
          let (svCond, s0) ← s.base.popSym
          let (cond, s1, opsCommit) ← s0.commit svCond
          match frames.get? depth with
          | none => none
          | some .loopK =>
              if depth = 0 then do
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrsP fuel frames ⟨s_cast, s.pending⟩ rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool [] [.breakOp]]
                  ++ postOps)
              else if hasLoopAbove frames depth then do
                let (cond_bool, s_cast) := s1.alloc
                let (s2, postOps) ← lowerInstrsP fuel frames ⟨s_cast, s.pending⟩ rest
                pure (s2,
                  opsCommit
                  ++ [.cast cond_bool cond .u32 .bool,
                      .branch cond_bool [.breakOp] []]
                  ++ postOps)
              else do
                let (s2, postOps) ← lowerInstrsP fuel frames ⟨s1, s.pending⟩ rest
                pure (s2, opsCommit ++ postOps)
          | some k =>
              if hasLoopAbove frames depth then
                if loopsAbove frames depth = 1 ∧ k = .block then
                  none  -- exit-flag route: still refused (Stage A).
                else do
                  let (cond_bool, s_cast) := s1.alloc
                  let (s2, postOps) ← lowerInstrsP fuel frames ⟨s_cast, s.pending⟩ rest
                  pure (s2,
                    opsCommit
                    ++ [.cast cond_bool cond .u32 .bool,
                        .branch cond_bool [.breakOp] []]
                    ++ postOps)
              else
                -- Record-and-wrap, conditional — THE Stage-B arm.
                if k = .block ∧ (frames.take depth).all (· = .block) then do
                  let (cond_bool, s_cast) := s1.alloc
                  -- The current frame's tail wraps inline: it runs
                  -- exactly when the br_if does not fire.
                  let (s2, restOps) ←
                    lowerInstrsP fuel frames ⟨s_cast, s.pending⟩ rest
                  let entry : List PendingWrap :=
                    if depth = 0 then [] else [⟨depth, cond_bool⟩]
                  pure (⟨s2.base, entry ++ s2.pending⟩,
                        opsCommit
                        ++ [.cast cond_bool cond .u32 .bool,
                            .branch cond_bool [] restOps])
                else none
      | .wreturn =>
          if frames.any (· = .loopK) ∨ frames.isEmpty then none
          else do
            let (s2, postOps) ← lowerInstrsP fuel frames s rest
            pure (s2, postOps)
      | _ => do
          let (s1, ops1) ← lowerInstr s.base i
          let (s2, ops2) ← lowerInstrsP fuel frames ⟨s1, s.pending⟩ rest
          pure (s2, ops1 ++ ops2)

/-- Top-level entry: a kernel body lowers with no open frames and
    must finish with no pending entries (an escaping entry would be
    a function-level record — outside the audited surface, refused
    exactly like Stage A's `wreturn` arm). -/
def lowerBodyP (fuel : Nat) (s : LowerState) (is : List WasmInstr) :
    Option (LowerState × List KernelOp) := do
  let (sp, ops) ← lowerInstrsP fuel [] ⟨s, []⟩ is
  if sp.pending ≠ [] then none else pure (sp.base, ops)

-- ════════════════════════════════════════════════════════════════════
-- Behavioral pins
-- ════════════════════════════════════════════════════════════════════
--
-- The shapes below were `none` under Stage A; the pins fix their
-- Stage-B output, and the last one witnesses old-subset agreement.
-- All are decided by evaluation — no axioms, no sorries.

section Pins

/-- `KernelOp` is a nested inductive (`branch`/`loopOp` carry
    `List KernelOp`), so `DecidableEq` cannot derive for it; the
    pins compare through `Repr` instead, which is injective on
    these purely-first-order values. -/
private def pinEq {α : Type} [Repr α] (a b : α) : Bool :=
  toString (repr a) == toString (repr b)

/-- `block [ const; br_if 0; const; drop ] ` — depth 0: only the
    current scope's tail wraps inline (here the tail lowers to no
    ops — a symbolic const consumed by `drop`); nothing pends. -/
example :
    pinEq
      (lowerInstrsP 4 [] ⟨LowerState.empty, []⟩
        [.block 0, .i32Const 1, .brIf 0, .i32Const 7, .drop, .wend])
      (some (⟨{ LowerState.empty with nextReg := 2 }, []⟩,
        [.const 0 (.u32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 [] []])) = true := by native_decide

/-- Two-level: `block (block (br_if 1) ) inner-post outer-post`.
    The br_if targets the OUTER block: the inner scope's (empty)
    tail wraps inline, the inner close wraps `inner-post` via the
    pending entry, and `outer-post` (after the target's End) runs
    unwrapped. The wrapped/unwrapped `localSet`s make the
    distinction visible in the ops. -/
example :
    pinEq
      (lowerInstrsP 4 [] ⟨LowerState.empty, []⟩
        [.block 0,
           .block 0,
             .i32Const 1, .brIf 1,
           .wend,
           .i32Const 5, .localSet 0,   -- inner post: wrapped
         .wend,
         .i32Const 9, .localSet 1      -- outer post: NOT wrapped
        ])
      (some (⟨{ LowerState.empty with
                 nextReg := 8,
                 localReg := [(1, 7), (0, 4)],
                 localTy := [(1, .u32), (0, .u32)],
                 currentReg := [(1, 6), (0, 3)] }, []⟩,
        [.const 0 (.u32 1),
         .cast 1 0 .u32 .bool,
         .branch 1 [] [],
         .branch 1 [] [.const 2 (.u32 5), .const 3 (.u32 0), .copy 3 2, .copy 4 3],
         .const 5 (.u32 9), .const 6 (.u32 0), .copy 6 5, .copy 7 6])) = true := by native_decide

/-- Unconditional `br 1` record-and-wrap: constant-`true` cond, the
    inner scope's rest dropped as dead code, the inner post wrapped
    — skipped at run time, matching the jump past the target's
    `End`. -/
example :
    pinEq
      (lowerInstrsP 4 [] ⟨LowerState.empty, []⟩
        [.block 0,
           .block 0,
             .br 1,
           .wend,
           .i32Const 5, .localSet 0,
         .wend])
      (some (⟨{ LowerState.empty with
                 nextReg := 4,
                 localReg := [(0, 3)],
                 localTy := [(0, .u32)],
                 currentReg := [(0, 2)] }, []⟩,
        [.const 0 (.bool true),
         .branch 0 [] [.const 1 (.u32 5), .const 2 (.u32 0), .copy 2 1, .copy 3 2]])) = true := by
  native_decide

/-- Old-subset agreement witness: a body with locals, a `block`,
    and a `wif` (all inside Stage A's accepted subset) lowers
    identically under Stage A and Stage B, with empty pending in
    and out. -/
example :
    let is : List WasmInstr :=
      [.i32Const 2, .localSet 0,
       .block 0, .localGet 0, .i32Const 3, .i32Add, .localSet 0, .wend,
       .localGet 0,
       .wif 0, .i32Const 9, .localSet 0, .welse, .i32Const 4, .localSet 0, .wend]
    pinEq
      (lowerInstrs 4 [] LowerState.empty is)
      ((lowerInstrsP 4 [] ⟨LowerState.empty, []⟩ is).map
          (fun r => (r.1.base, r.2))) = true := by native_decide

/-- Escaping records refuse at the function boundary: a `br_if 1`
    with one open frame would need a function-level wrap —
    `lowerBodyP` returns `none`, mirroring the audited-surface
    refusal. -/
example :
    (lowerBodyP 4 LowerState.empty
      [.block 0, .i32Const 1, .brIf 1, .wend]).isNone = true := by native_decide

end Pins

-- ════════════════════════════════════════════════════════════════════
-- Universal agreement: lowerInstrs ⟹ lowerInstrsP
-- ════════════════════════════════════════════════════════════════════
--
-- On the Stage-A success domain (every input `lowerInstrs` accepts),
-- the pending machinery is inert: `lowerInstrsP` walks the identical
-- structure, the pending list stays `[]` throughout, and the emitted
-- ops match. The two record-and-wrap arms where the functions
-- diverge are exactly the arms Stage A refuses (`none`), so the
-- hypothesis `lowerInstrs … = some …` rules them out.
--
-- This is the bridge lemma: anything proven about `lowerInstrs`
-- (preservation, scopeValid, nextReg-monotonicity) transfers to
-- `lowerInstrsP` on the shared domain by rewriting through it.

@[simp] theorem applyWraps_nil (tail : List KernelOp) :
    applyWraps [] tail = tail := rfl

@[simp] theorem stepPending_nil : stepPending [] = [] := rfl

/-- The inert-pending agreement theorem. When Stage A accepts, Stage
    B with empty in-pending produces the same base state and ops and
    empty out-pending. Proved by `lowerInstrs.induct` — same 21-case
    dispatch as the scopeValid/preservation masters. -/
theorem lowerInstrsP_agrees_with_lowerInstrs :
    ∀ (fuel : Nat) (frames : List FrameKind) (s : LowerState) (instrs : List WasmInstr)
      {s' : LowerState} {ops : List KernelOp},
      lowerInstrs fuel frames s instrs = some (s', ops) →
      lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨s', []⟩, ops) := by
  intro fuel frames s instrs
  induction fuel, frames, s, instrs using lowerInstrs.induct with
  | case1 _ _ _ =>
    intro s' ops h
    simp only [lowerInstrs] at h
    obtain ⟨h_s_eq, h_ops_eq⟩ := Prod.mk.inj (Option.some.inj h)
    subst h_s_eq; subst h_ops_eq
    simp only [lowerInstrsP]
  | case2 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case3 _ _ _ _ _ hsplit =>
    intro s' ops h; simp [lowerInstrs, hsplit] at h
  | case4 frames s rest arity f body post hsplit ih2 ih1 =>
    intro s' ops h
    simp only [lowerInstrs, hsplit] at h
    rcases hb : lowerInstrs f (.block :: frames) s body with _ | ⟨s1, innerOps⟩
    · simp [hb] at h
    simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
    rcases hp : lowerInstrs f frames s1 post with _ | ⟨s2, postOps⟩
    · simp [hp] at h
    simp only [hp, Option.some_bind, pure, Pure.pure] at h
    obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
    subst hs; subst hops
    have e1 := ih1 s1 hp
    unfold lowerInstrsP
    rw [hsplit]
    simp only [ih2 hb, Option.bind_eq_bind, Option.some_bind, e1,
               applyWraps_nil, stepPending_nil, List.append_nil, List.nil_append,
               pure, Pure.pure]
  | case5 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case6 _ _ _ _ _ hsplit =>
    intro s' ops h; simp [lowerInstrs, hsplit] at h
  | case7 frames s rest arity f body post hsplit =>
    rename_i ih2 ih1
    intro s' ops h
    simp only [lowerInstrs, hsplit] at h
    rcases hb : lowerInstrs f (.loopK :: frames) s body with _ | ⟨s1, bodyOps⟩
    · simp [hb] at h
    simp only [hb, Option.bind_eq_bind, Option.some_bind] at h
    rcases hp : lowerInstrs f frames
      { nextReg := s1.nextReg, stack := s1.stack,
        localReg := s.localReg, localTy := s.localTy,
        bufferSlots := s1.bufferSlots, currentReg := s.currentReg }
      post with _ | ⟨s2, postOps⟩
    · simp [hp] at h
    simp only [hp, Option.some_bind, pure, Pure.pure] at h
    obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
    subst hs; subst hops
    have e2 := ih2 hb
    have e1 : lowerInstrsP f frames
        ⟨{ nextReg := s1.nextReg, stack := s1.stack,
           localReg := s.localReg, localTy := s.localTy,
           bufferSlots := s1.bufferSlots, currentReg := s.currentReg }, []⟩
        post = some (⟨s2, []⟩, postOps) := ih1 s1 hp
    simp only [lowerInstrsP, hsplit, e2, Option.bind_eq_bind, Option.some_bind,
               ne_eq, not_true_eq_false, if_false, e1, pure, Pure.pure]
  | case8 _ _ _ _ =>
    intro s' ops h; simp [lowerInstrs] at h
  | case9 _ _ _ _ _ hsplit =>
    intro s' ops h; simp [lowerInstrs, hsplit] at h
  | case10 frames s rest arity f thenBody elseBody post hsplit =>
    rename_i ih3 ih2 ih1
    intro s' ops h
    simp only [lowerInstrs, hsplit] at h
    rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
    · simp [hpop] at h
    simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
    rcases hc : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
    · simp [hc] at h
    simp only [hc, Option.some_bind, LowerState.alloc] at h
    rcases hth : lowerInstrs f (.wif :: frames)
        { nextReg := s1.nextReg + 1, stack := s1.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
        thenBody with _ | ⟨s2, thenOps⟩
    · simp [hth] at h
    simp only [hth, Option.some_bind] at h
    rcases hel : lowerInstrs f (.wif :: frames)
        { nextReg := s2.nextReg, stack := s2.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s2.bufferSlots, currentReg := s1.currentReg }
        elseBody with _ | ⟨s3, elseOps⟩
    · simp [hel] at h
    simp only [hel, Option.some_bind] at h
    rcases hpo : lowerInstrs f frames
        { nextReg := s3.nextReg, stack := s3.stack,
          localReg := s1.localReg, localTy := s1.localTy,
          bufferSlots := s3.bufferSlots, currentReg := s1.currentReg }
        post with _ | ⟨s4, postOps⟩
    · simp [hpo] at h
    simp only [hpo, Option.some_bind, pure, Pure.pure] at h
    obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
    subst hs; subst hops
    -- s_cast: the alloc'd then-branch entry state. ih2/ih1 take it
    -- as the snapshot anchor, then the body's result state. Each IH
    -- is restated with the literal post-/else-branch state so the
    -- LHS matches the goal's unfolded form (the IH's let-bound
    -- `*_restored` is defeq but not syntactically equal).
    have e3 : lowerInstrsP f (.wif :: frames)
        ⟨{ nextReg := s1.nextReg + 1, stack := s1.stack,
           localReg := s1.localReg, localTy := s1.localTy,
           bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }, []⟩
        thenBody = some (⟨s2, []⟩, thenOps) :=
      ih3 _ hth
    have e2 : lowerInstrsP f (.wif :: frames)
        ⟨{ nextReg := s2.nextReg, stack := s2.stack,
           localReg := s1.localReg, localTy := s1.localTy,
           bufferSlots := s2.bufferSlots, currentReg := s1.currentReg }, []⟩
        elseBody = some (⟨s3, []⟩, elseOps) :=
      ih2 { nextReg := s1.nextReg + 1, stack := s1.stack,
            localReg := s1.localReg, localTy := s1.localTy,
            bufferSlots := s1.bufferSlots, currentReg := s1.currentReg } s2 hel
    have e1 : lowerInstrsP f frames
        ⟨{ nextReg := s3.nextReg, stack := s3.stack,
           localReg := s1.localReg, localTy := s1.localTy,
           bufferSlots := s3.bufferSlots, currentReg := s1.currentReg }, []⟩
        post = some (⟨s4, []⟩, postOps) :=
      ih1 { nextReg := s1.nextReg + 1, stack := s1.stack,
            localReg := s1.localReg, localTy := s1.localTy,
            bufferSlots := s1.bufferSlots, currentReg := s1.currentReg } s3 hpo
    simp only [lowerInstrsP, hsplit, hpop, Option.bind_eq_bind, Option.some_bind,
               hc, LowerState.alloc, e3, ne_eq, not_true_eq_false, if_false,
               e2, e1, pure, Pure.pure]
  | case11 _ _ _ _ _ hg =>
    intro s' ops h; simp only [lowerInstrs] at h; rw [hg] at h; simp at h
  | case12 _ _ _ _ hg =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h; simp at h
    obtain ⟨hs, hops⟩ := h
    subst hs; subst hops
    simp only [lowerInstrsP]; rw [hg]; simp
  | case13 _ _ _ _ _ hg hnz hla =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h; simp [hnz, hla] at h
    obtain ⟨hs, hops⟩ := h
    subst hs; subst hops
    simp only [lowerInstrsP]; rw [hg]; simp [hnz, hla]
  | case14 _ _ _ _ _ hg hnz hnla =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h; simp [hnz, hnla] at h
    obtain ⟨hs, hops⟩ := h
    subst hs; subst hops
    simp only [lowerInstrsP]; rw [hg]; simp [hnz, hnla]
  | case15 _ _ _ _ _ val hne hg hla hsb =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h
    obtain ⟨hone, hblk⟩ := hsb; subst hblk
    simp [hla, hone] at h
  | case16 _ _ _ _ _ val hne hg hla hnsb =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h
    cases val with
    | loopK => exact absurd rfl hne
    | block =>
      simp [hla] at h
      obtain ⟨hbreak, hs, hops⟩ := h
      subst hs; subst hops
      simp only [lowerInstrsP]; rw [hg]; simp [hla, hbreak]
    | wif =>
      simp [hla] at h
      obtain ⟨hs, hops⟩ := h
      subst hs; subst hops
      simp only [lowerInstrsP]; rw [hg]; simp [hla]
  | case17 _ _ _ _ _ val hne hg hnla =>
    intro s' ops h
    simp only [lowerInstrs] at h; rw [hg] at h
    cases val with
    | loopK => exact absurd rfl hne
    | block | wif => all_goals (simp [hnla] at h)
  | case18 fuel frames s rest depth ih1 =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rcases hpop : s.popSym with _ | ⟨svCond, s0⟩
    · simp [hpop] at h
    simp only [hpop, Option.bind_eq_bind, Option.some_bind] at h
    rcases hc : s0.commit svCond with _ | ⟨cond, s1, opsCommit⟩
    · simp [hc] at h
    simp only [hc, Option.some_bind] at h
    rcases hg : frames.get? depth with _ | fk
    · rw [hg] at h; simp at h
    rw [hg] at h
    -- Every brIf arm that Stage A accepts recurses on `rest` from
    -- one of two states: the alloc'd `s1.nextReg + 1` state (when a
    -- cast is emitted) or `s1` itself (the continue-outer case).
    -- A single closer handles both: unfold lowerInstrsP, replay the
    -- pop/commit/get?/guards, rewrite the recursive call by ih1, and
    -- let `simp` reduce the Bool-decidable `if`s and append assoc.
    cases fk with
    | loopK =>
      simp at h
      by_cases h0 : depth = 0
      · simp [h0, LowerState.alloc] at h
        rcases hr : lowerInstrs fuel frames
          { nextReg := s1.nextReg + 1, stack := s1.stack,
            localReg := s1.localReg, localTy := s1.localTy,
            bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
          rest with _ | ⟨s2, postOps⟩
        · simp [hr] at h
        simp only [hr, Option.some_bind, pure, Pure.pure] at h
        obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
        subst hs; subst hops
        simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
        rw [hg]; simp [h0, LowerState.alloc, ih1 _ hr]
      · simp [h0] at h
        by_cases hla : hasLoopAbove frames depth
        · simp [hla, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
          subst hs; subst hops
          simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
          rw [hg]; simp [h0, hla, LowerState.alloc, ih1 _ hr]
        · simp [hla] at h
          rcases hr : lowerInstrs fuel frames s1 rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
          subst hs; subst hops
          simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
          rw [hg]; simp [h0, hla, ih1 _ hr]
    | block =>
      by_cases hla : hasLoopAbove frames depth
      · by_cases hone : loopsAbove frames depth = 1
        · simp [hla, hone] at h
        · simp [hla, hone, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
          subst hs; subst hops
          simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
          rw [hg]; simp [hla, hone, LowerState.alloc, ih1 _ hr]
      · simp [hla] at h
    | wif =>
      by_cases hla : hasLoopAbove frames depth
      · by_cases hone : loopsAbove frames depth = 1
        · simp [hla, hone, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
          subst hs; subst hops
          simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
          rw [hg]; simp [hla, hone, LowerState.alloc, ih1 _ hr]
        · simp [hla, hone, LowerState.alloc] at h
          rcases hr : lowerInstrs fuel frames
            { nextReg := s1.nextReg + 1, stack := s1.stack,
              localReg := s1.localReg, localTy := s1.localTy,
              bufferSlots := s1.bufferSlots, currentReg := s1.currentReg }
            rest with _ | ⟨s2, postOps⟩
          · simp [hr] at h
          simp only [hr, Option.some_bind, pure, Pure.pure] at h
          obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
          subst hs; subst hops
          simp only [lowerInstrsP, hpop, Option.bind_eq_bind, Option.some_bind, hc]
          rw [hg]; simp [hla, hone, LowerState.alloc, ih1 _ hr]
      · simp [hla] at h
  | case19 _ _ _ _ hcond =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [if_pos hcond] at h
    exact absurd h Option.noConfusion
  | case20 fuel frames s rest hncond ih1 =>
    intro s' ops h
    simp only [lowerInstrs] at h
    rw [if_neg hncond] at h
    rcases hr : lowerInstrs fuel frames s rest with _ | ⟨s2, postOps⟩
    · simp [hr] at h
    simp only [hr, Option.bind_eq_bind, Option.some_bind, pure, Pure.pure] at h
    obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
    subst hs; subst hops
    simp only [lowerInstrsP]
    rw [if_neg hncond]
    simp only [ih1 hr, Option.bind_eq_bind, Option.some_bind, pure, Pure.pure]
  | case21 fuel frames s i rest hnb hnl hnw hnbr hnbi hnwr ih1 =>
    intro s' ops h
    cases i <;>
      first
        | (first
            | (exact absurd rfl (hnb _))
            | (exact absurd rfl (hnl _))
            | (exact absurd rfl (hnw _))
            | (exact absurd rfl (hnbr _))
            | (exact absurd rfl (hnbi _))
            | (exact absurd rfl hnwr))
        | (rename_i x
           simp only [lowerInstrs] at h
           rcases hi1 : lowerInstr s x with _ | ⟨s1, ops1⟩
           · rw [hi1] at h; simp at h
           rw [hi1] at h
           simp only [Option.bind_eq_bind, Option.some_bind] at h
           rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
           · rw [hi2] at h; simp at h
           rw [hi2] at h
           simp only [Option.some_bind, pure, Pure.pure] at h
           obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
           subst hs; subst hops
           simp only [lowerInstrsP, hi1, Option.bind_eq_bind, Option.some_bind,
                      ih1 _ hi2, pure, Pure.pure])
        | (simp only [lowerInstrs] at h
           rcases hi1 : lowerInstr s _ with _ | ⟨s1, ops1⟩
           · rw [hi1] at h; simp at h
           rw [hi1] at h
           simp only [Option.bind_eq_bind, Option.some_bind] at h
           rcases hi2 : lowerInstrs fuel frames s1 rest with _ | ⟨s2, ops2⟩
           · rw [hi2] at h; simp at h
           rw [hi2] at h
           simp only [Option.some_bind, pure, Pure.pure] at h
           obtain ⟨hs, hops⟩ := Prod.mk.inj (Option.some.inj h)
           subst hs; subst hops
           simp only [lowerInstrsP, hi1, Option.bind_eq_bind, Option.some_bind,
                      ih1 _ hi2, pure, Pure.pure])

-- ════════════════════════════════════════════════════════════════════
-- Transfer corollary — the bridge in use
-- ════════════════════════════════════════════════════════════════════
--
-- A worked example of using the agreement lemma to carry a Stage-A
-- result over to `lowerInstrsP`: nextReg-monotonicity. The pattern
-- is mechanical — given the Stage-A theorem `P (lowerInstrs …)`,
-- rewrite the `lowerInstrsP` success through agreement back to a
-- `lowerInstrs` success and apply the Stage-A theorem. The full
-- preservation/scopeValid masters transfer the same way; this
-- corollary pins the recipe for the next session.

/-- nextReg never decreases through `lowerInstrsP`, stated about a
    `lowerInstrsP` success and an underlying Stage-A success. The
    proof goes through the agreement bridge: agreement forces the
    Stage-A and Stage-B base states to coincide, so the Stage-A
    monotonicity bound applies to the Stage-B output. This is the
    template the preservation/scopeValid masters follow to transfer
    onto `lowerInstrsP`. -/
theorem lowerInstrsP_nextReg_mono
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (instrs : List WasmInstr)
    {sa : LowerState} {opsa : List KernelOp}
    {sp : LowerState} {opsp : List KernelOp}
    (ha : lowerInstrs fuel frames s instrs = some (sa, opsa))
    (hp : lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨sp, []⟩, opsp)) :
    s.nextReg ≤ sp.nextReg := by
  -- Agreement: from the same Stage-A success, Stage B yields the
  -- identical base state. So sp = sa, and the Stage-A bound applies.
  have hbridge := lowerInstrsP_agrees_with_lowerInstrs fuel frames s instrs ha
  rw [hbridge] at hp
  obtain ⟨hbase, _⟩ := Prod.mk.inj (Option.some.inj hp)
  have hsp : sa = sp := congrArg LowerStateP.base hbase
  rw [← hsp]
  exact lowerInstrs_nextReg_mono fuel frames s instrs ha

/-- wellScoped is preserved through `lowerInstrsP`. Transferred from
    `lowerInstrs_preserves_wellScoped` via the agreement bridge: the
    Stage-A and Stage-B base states coincide, so the Stage-A
    preservation result lands on the Stage-B output state. -/
theorem lowerInstrsP_preserves_wellScoped
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (instrs : List WasmInstr)
    {sa : LowerState} {opsa : List KernelOp}
    {sp : LowerState} {opsp : List KernelOp}
    (hws : s.wellScoped)
    (ha : lowerInstrs fuel frames s instrs = some (sa, opsa))
    (hp : lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨sp, []⟩, opsp)) :
    sp.wellScoped := by
  have hbridge := lowerInstrsP_agrees_with_lowerInstrs fuel frames s instrs ha
  rw [hbridge] at hp
  obtain ⟨hbase, _⟩ := Prod.mk.inj (Option.some.inj hp)
  have hsp : sa = sp := congrArg LowerStateP.base hbase
  rw [← hsp]
  exact lowerInstrs_preserves_wellScoped fuel frames s instrs hws ha

/-- The emitted ops are scope-valid in the output state's scope
    environment, for `lowerInstrsP`. Transferred from
    `lowerInstrs_scopeValid_ops`: agreement equates BOTH the base
    state and the op list, so the Stage-A ops-validity carries to the
    Stage-B output verbatim. -/
theorem lowerInstrsP_scopeValid_ops
    (fuel : Nat) (frames : List FrameKind) (s : LowerState) (instrs : List WasmInstr)
    {sa : LowerState} {opsa : List KernelOp}
    {sp : LowerState} {opsp : List KernelOp}
    (hws : s.wellScoped)
    (ha : lowerInstrs fuel frames s instrs = some (sa, opsa))
    (hp : lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨sp, []⟩, opsp)) :
    scopeValidOps sp.scopeEnv opsp := by
  have hbridge := lowerInstrsP_agrees_with_lowerInstrs fuel frames s instrs ha
  rw [hbridge] at hp
  obtain ⟨hbase, hops⟩ := Prod.mk.inj (Option.some.inj hp)
  have hsp : sa = sp := congrArg LowerStateP.base hbase
  subst hsp
  subst hops
  exact lowerInstrs_scopeValid_ops fuel frames s instrs hws ha

-- ════════════════════════════════════════════════════════════════════
-- Source preservation through lowerInstrsP
-- ════════════════════════════════════════════════════════════════════
--
-- The capstone transfer: the closed-shape source-preservation
-- master (`preservation_evalInstrs_main_v2`) carries onto
-- `lowerInstrsP` by the same bridge. Agreement equates the Stage-A
-- and Stage-B `(state, ops)` outputs, so the ops Stage B emits are
-- exactly the ops Stage A's preservation theorem already reasons
-- about. With this, the full preservation chain — WASM eval ⇒ KOps
-- eval under `Refines` — holds for the pending-wrap translator on
-- the closed-shape domain it shares with Stage A.

/-- Source preservation for `lowerInstrsP`: when the WASM program
    evaluates and `lowerInstrsP` lowers it (on the closed-shape
    domain), the emitted KernelOps evaluate to a refining KOps
    state. Transferred from `preservation_evalInstrs_main_v2` via
    the agreement bridge. -/
theorem preservation_evalInstrsP
    (fuel : Nat) (frames : List FrameKind)
    (instrs : List WasmInstr)
    (ws : WasmState) (s : LowerState) (kst : Quanta.KOps.State)
    (layout : BufferLayout)
    (R : Refines ws s kst layout)
    (h_closed : closedInstrsAt s instrs = true)
    (h_no_branch : ws.branchTarget = none)
    (h_no_halt : ws.halted = false)
    (h_kst_no_broke : kst.broke = false)
    (ws' : WasmState) (sa : LowerState) (opsa : List KernelOp)
    (sp : LowerState) (opsp : List KernelOp)
    (hw : evalInstrs fuel ws instrs = some ws')
    (ha : lowerInstrs fuel frames s instrs = some (sa, opsa))
    (hp : lowerInstrsP fuel frames ⟨s, []⟩ instrs = some (⟨sp, []⟩, opsp)) :
    ∃ (kst' : Quanta.KOps.State) (F : Nat),
      evalOps F kst opsp = some kst' ∧ Refines ws' sp kst' layout := by
  -- Agreement equates the Stage-A and Stage-B outputs verbatim.
  have hbridge := lowerInstrsP_agrees_with_lowerInstrs fuel frames s instrs ha
  rw [hbridge] at hp
  obtain ⟨hbase, hops⟩ := Prod.mk.inj (Option.some.inj hp)
  have hsp : sa = sp := congrArg LowerStateP.base hbase
  subst hsp
  subst hops
  -- Now opsp = opsa and sp = sa; the Stage-A master applies directly.
  exact preservation_evalInstrs_main_v2 fuel frames instrs ws s kst layout R
    h_closed h_no_branch h_no_halt h_kst_no_broke ws' sa opsa hw ha

end Quanta.Wasm

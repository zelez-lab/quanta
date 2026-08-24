/-
# L16 (3a) — loop bodies with balanced `wif` frames: the split lemmas

`preservation_blockLoop_nIterExit` demands `NoStructured body` only so
the `splitAtEnd` scanner finds the loop's own `wend`. A *balanced*
`wif … (welse …)? wend` keeps the scanner correct — it enters the
frame at depth `n+1` and leaves it back at `n`, never tripping the
depth-0 stop arms. `IfBalanced` names that body class (marker-free
arms; nested frames are the depth-2 rung's business), and the two
split lemmas below are the `_noStructured` pair generalized to it.
-/

import Quanta.Wasm.PreservationIfMerge
import Quanta.Wasm.PreservationWhileExit
import Quanta.Wasm.PreservationBlockWhile

namespace Quanta.Wasm

open Quanta.KOps (KernelOp evalOps regLookup regWrite vBool)

/-- A stream of marker-free instructions and balanced single-level
    `wif` frames (marker-free arms, with or without `welse`). The
    body class the loop machinery can scan across: every opener's
    closer is inside the stream. -/
inductive IfBalanced : List WasmInstr → Prop
  | empty : IfBalanced []
  | sl_cons {i : WasmInstr} {rest : List WasmInstr} :
      NoStructuredInstr i → IfBalanced rest → IfBalanced (i :: rest)
  | wif_else_cons {bt : Nat} {thenB elseB post : List WasmInstr} :
      NoStructured thenB → NoStructured elseB → IfBalanced post →
      IfBalanced (.wif bt :: (thenB ++ .welse :: elseB ++ .wend :: post))
  | wif_noelse_cons {bt : Nat} {thenB post : List WasmInstr} :
      NoStructured thenB → IfBalanced post →
      IfBalanced (.wif bt :: (thenB ++ .wend :: post))

/-- Marker-free streams are trivially balanced. -/
theorem IfBalanced.of_noStructured : ∀ {l : List WasmInstr},
    NoStructured l → IfBalanced l
  | [], _ => .empty
  | _ :: _, h => .sl_cons h.1 (IfBalanced.of_noStructured h.2)

/-- One walker step over an `wif`: the depth bumps. -/
theorem walkUntilCloser_wif (bt : Nat) (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.wif bt :: rest) n acc
      = walkUntilCloser rest (n + 1) (.wif bt :: acc) := rfl

/-- One walker step over a `welse` strictly above the stop level. -/
theorem walkUntilCloser_welse_succ (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.welse :: rest) (n + 1) acc
      = walkUntilCloser rest (n + 1) (.welse :: acc) := rfl

/-- One walker step over a `wend` strictly above the stop level: the
    depth restores. -/
theorem walkUntilCloser_wend_succ (rest : List WasmInstr)
    (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (.wend :: rest) (n + 1) acc
      = walkUntilCloser rest n (.wend :: acc) := rfl

/-- The walker crosses a balanced stream at any depth, accumulating it
    in reverse — the `_noStructured` walker lemma generalized: a `wif`
    bumps the depth, its markers live strictly above the stop level,
    and its `wend` restores the entry depth. -/
theorem walkUntilCloser_append_ifBalanced
    {l : List WasmInstr} (h : IfBalanced l)
    (rest : List WasmInstr) (n : Nat) (acc : List WasmInstr) :
    walkUntilCloser (l ++ rest) n acc = walkUntilCloser rest n (l.reverse ++ acc) := by
  induction h generalizing n acc with
  | empty => rfl
  | sl_cons h_i _ IH =>
      rw [List.cons_append, walkUntilCloser_cons_noStructured h_i, IH]
      simp
  | @wif_else_cons bt thenB elseB post h_then h_else _ IH =>
      simp only [List.cons_append, List.append_assoc]
      rw [walkUntilCloser_wif, walkUntilCloser_append_noStructured h_then,
          walkUntilCloser_welse_succ, walkUntilCloser_append_noStructured h_else,
          walkUntilCloser_wend_succ, IH]
      simp [List.reverse_append]
  | @wif_noelse_cons bt thenB post h_then _ IH =>
      simp only [List.cons_append, List.append_assoc]
      rw [walkUntilCloser_wif, walkUntilCloser_append_noStructured h_then,
          walkUntilCloser_wend_succ, IH]
      simp [List.reverse_append]

/-- A balanced body closed by `wend` splits exactly there
    (`splitAtEnd_noStructured` generalized). -/
theorem splitAtEnd_ifBalanced
    {l : List WasmInstr} (h : IfBalanced l) (t : List WasmInstr) :
    splitAtEnd (l ++ [.wend] ++ t) = some (l, t) := by
  rw [List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser_append_ifBalanced h, walkUntilCloser,
             List.append_nil, List.reverse_reverse, Option.bind_eq_bind, Option.some_bind]

/-- The block body `wloop 0 :: loopBody ++ [wend] ++ tail` splits at
    the block's own `wend`, with balanced loop body and tail
    (`splitAtEnd_wloop_noStructured` generalized). -/
theorem splitAtEnd_wloop_ifBalanced
    {loopBody tail : List WasmInstr}
    (h_body : IfBalanced loopBody) (h_tail : IfBalanced tail)
    (post : List WasmInstr) :
    splitAtEnd (.wloop 0 :: loopBody ++ [.wend] ++ tail ++ [.wend] ++ post)
      = some (.wloop 0 :: loopBody ++ [.wend] ++ tail, post) := by
  simp only [List.cons_append, List.append_assoc, List.singleton_append]
  simp only [splitAtEnd, walkUntilCloser, walkUntilCloser_append_ifBalanced h_body,
             walkUntilCloser_append_ifBalanced h_tail, Option.bind_eq_bind,
             Option.some_bind, List.reverse_append, List.reverse_cons, List.reverse_reverse,
             List.reverse_nil, List.nil_append, List.append_nil, List.cons_append,
             List.append_assoc]

/-- Balanced streams compose. -/
theorem IfBalanced.append : ∀ {a b : List WasmInstr},
    IfBalanced a → IfBalanced b → IfBalanced (a ++ b) := by
  intro a b ha hb
  induction ha with
  | empty => exact hb
  | sl_cons h_i _ IH => exact .sl_cons h_i IH
  | @wif_else_cons bt thenB elseB post h_then h_else _ IH =>
      have h := IfBalanced.wif_else_cons (bt := bt) h_then h_else IH
      simpa [List.cons_append, List.append_assoc] using h
  | @wif_noelse_cons bt thenB post h_then _ IH =>
      have h := IfBalanced.wif_noelse_cons (bt := bt) h_then IH
      simpa [List.cons_append, List.append_assoc] using h

/-- `blockLoop_lowerP` with the body class relaxed to `IfBalanced` —
    the `NoStructured` hypothesis fed only the two split lemmas, and
    those now hold for balanced bodies. Everything else is the same
    walk down the Stage-B lowering: the loop close consumes the single
    exit-flag record, the block close wraps the empty tail, the post
    follows. -/
theorem blockLoop_lowerP_ifBalanced
    {f : Nat} {frames : List FrameKind} {body post : List WasmInstr}
    (h_bal : IfBalanced body)
    {s s1 : LowerState} {flag : Quanta.KOps.Reg} {bodyOps : List KernelOp}
    (h_lb : lowerInstrsP f (.loopK :: .block :: frames) ⟨{ s with currentReg := [] }, []⟩ body
        = some (⟨s1, [{ levels := 1, cond := flag, flag := true, skip := 0 }]⟩, bodyOps))
    {s' : LowerState} {ops : List KernelOp}
    (hl : lowerInstrsP (f + 2) frames ⟨s, []⟩
            (.block 0 :: .wloop 0 :: body ++ [.wend] ++ [] ++ [.wend] ++ post)
          = some (⟨s', []⟩, ops)) :
    ∃ (postOps : List KernelOp),
      lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
        = some (⟨s', []⟩, postOps) ∧
      ops = [.const flag (.bool false), .loopOp bodyOps, .branch flag [] []] ++ postOps := by
  have h_split_b := splitAtEnd_wloop_ifBalanced h_bal (tail := []) .empty post
  have h_split_l := splitAtEnd_ifBalanced h_bal []
  simp only [lowerInstrsP, List.cons_append, List.append_assoc, List.nil_append,
             List.append_nil, List.singleton_append] at hl h_split_b h_split_l
  rw [h_split_b] at hl
  simp only [Option.bind_eq_bind] at hl
  -- The block body: the wloop arm.
  rcases hb : lowerInstrsP (f + 1) (.block :: frames) ⟨s, []⟩
      (.wloop 0 :: (body ++ [.wend])) with _ | ⟨spb, innerOps⟩
  · rw [hb] at hl; simp at hl
  rw [hb] at hl
  simp only [Option.some_bind] at hl
  simp only [lowerInstrsP] at hb
  rw [h_split_l] at hb
  simp only [Option.bind_eq_bind] at hb
  rw [h_lb] at hb
  simp only [Option.some_bind] at hb
  -- The loop close: one flag record, consumed here; the empty tail.
  simp only [hasPlainActive, List.any_cons, List.any_nil, Bool.not_true, Bool.and_false,
             Bool.or_false, Bool.false_eq_true, ↓reduceIte, lowerInstrsP, Option.bind_eq_bind,
             Option.some_bind, pure, stepPending, List.filterMap_cons, List.filterMap_nil,
             closeDecls, applyWraps, activeWraps, List.filter_cons, List.filter_nil,
             List.foldr_cons, List.foldr_nil] at hb
  simp at hb
  obtain ⟨h_spb, h_inner⟩ := hb
  subst h_spb
  -- The block close: the post from the close state, nothing pending.
  simp only at hl
  rcases hlp : lowerInstrsP (f + 1) frames ⟨{ s1 with currentReg := [] }, []⟩ post
      with _ | ⟨spp, postOps⟩
  · rw [hlp] at hl; simp at hl
  rw [hlp] at hl
  simp only [Option.some_bind, pure, stepPending_nil, closeDecls_nil, applyWraps_nil,
             List.append_nil, List.nil_append, Option.some.injEq, Prod.mk.injEq] at hl
  obtain ⟨h_spp, h_ops⟩ := hl
  subst h_inner
  refine ⟨postOps, ?_, ?_⟩
  · cases spp
    simp only [LowerStateP.mk.injEq] at h_spp
    obtain ⟨h1, h2⟩ := h_spp
    subst h1
    subst h2
    rfl
  · rw [← h_ops]

end Quanta.Wasm

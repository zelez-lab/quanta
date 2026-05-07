/-
# WASM structured-control splitter helpers (step 059, slice 5)

Production's `crates/quanta-wasm-lowering/src/lower.rs` walks WASM
bytecode in a streaming fashion, maintaining a `Vec<Frame>` to track
nested `block`/`loop`/`if` scopes. Each `wend` pops the topmost frame
and folds its accumulated ops into the parent (a `KernelOp.branch`,
`loopOp`, or label-scope splice). The Lean port uses a different but
equivalent strategy: **recursive descent**. We pre-split each
structured construct's body out of the instruction list and lower it
recursively. The end-to-end output is identical because both
strategies are nothing more than two phrasings of "find the matching
`wend`, lower the body, wrap it appropriately."

This module provides the two splitting primitives both
`Quanta.Wasm.Semantics.evalInstrs` and
`Quanta.Wasm.Translate.lowerInstrs` use:

* `splitAtEnd l` — for `block`/`wloop`. Walks `l` until the matching
  `wend` (handling nested `block`/`wloop`/`wif` openers), returns
  `(body, postEnd)` where `body` is everything strictly before the
  matching `wend` and `postEnd` is everything strictly after.
* `splitAtElseOrEnd l` — for `wif`. Walks `l` until the first
  `welse` *or* `wend` at depth 0 (whichever comes first). On `welse`,
  walks again from the post-else suffix to find the matching `wend`.
  Returns `(thenBody, elseBody, postEnd)`. `elseBody = []` when the
  `if` has no `else` arm.

Both functions return `none` on an unbalanced input — i.e. one whose
matching `wend` was never reached.
-/

import Quanta.Wasm.Syntax

namespace Quanta.Wasm

/-- True if the instruction opens a structured-control scope —
    `block`, `loop`, or `if`. Each such opener bumps the nesting
    depth seen by `walkUntilCloser` so nested constructs don't make
    their inner `wend`s match an outer opener prematurely. -/
@[inline] def WasmInstr.isOpener : WasmInstr → Bool
  | .block _ => true
  | .wloop _ => true
  | .wif _   => true
  | _ => false

/-- The static kind of an open structured-control frame. Threaded as
    `frames : List FrameKind` (innermost = head) through
    `Quanta.Wasm.Translate.lowerInstrs` so `br`/`brIf` know whether
    the target is a Loop (continue) or a non-Loop (break out).
    Eval-side does not need this — the runtime branch direction is
    carried by `WasmState.branchTarget`. -/
inductive FrameKind where
  | block
  | loopK
  | wif
  deriving Repr, DecidableEq

/-- Generic walker. Consumes from the head, accumulates into `acc` in
    reverse order, and stops at the first depth-0 `wend` or `welse`.
    Returns `(acc.reverse, marker, rest)` where `marker` is the
    closer encountered. The two public splitters are thin wrappers.

    The depth counter starts at the caller's depth (always 0 for the
    public splitters) and bumps on each opener; `wend` at depth > 0
    decrements; `welse` at depth > 0 is a plain instruction (it
    belongs to a nested `wif`). -/
def walkUntilCloser : List WasmInstr → Nat → List WasmInstr →
    Option (List WasmInstr × WasmInstr × List WasmInstr)
  | [], _, _ => none
  | .wend  :: rest, 0, acc => some (acc.reverse, .wend,  rest)
  | .welse :: rest, 0, acc => some (acc.reverse, .welse, rest)
  | i :: rest, n, acc =>
      let n' :=
        match i with
        | .wend    => n - 1
        | .block _ => n + 1
        | .wloop _ => n + 1
        | .wif _   => n + 1
        | _        => n
      walkUntilCloser rest n' (i :: acc)

/-- Find the matching `wend` for an opener already consumed. Returns
    `(body, postEnd)`. Fails on unbalanced input or on a stray
    depth-0 `welse` (only `wif` should see one and it uses
    `splitAtElseOrEnd` instead). -/
def splitAtEnd (l : List WasmInstr) : Option (List WasmInstr × List WasmInstr) := do
  let (taken, marker, rest) ← walkUntilCloser l 0 []
  match marker with
  | .wend => some (taken, rest)
  | _ => none

/-- For `wif`: walk to the first depth-0 `welse` or `wend`. On
    `welse`, walk again to find the matching `wend`. Returns
    `(thenBody, elseBody, postEnd)`; `elseBody = []` when there's no
    else clause. -/
def splitAtElseOrEnd (l : List WasmInstr) :
    Option (List WasmInstr × List WasmInstr × List WasmInstr) := do
  let (thenBody, marker1, rest1) ← walkUntilCloser l 0 []
  match marker1 with
  | .wend => some (thenBody, [], rest1)
  | .welse => do
      let (elseBody, marker2, rest2) ← walkUntilCloser rest1 0 []
      match marker2 with
      | .wend => some (thenBody, elseBody, rest2)
      | _ => none
  | _ => none

-- ════════════════════════════════════════════════════════════════════
-- Smoke tests — verified at definition time via `decide`
-- ════════════════════════════════════════════════════════════════════

/-- Plain block: `nop wend nop` splits to `(body=[nop], post=[nop])`. -/
example :
    splitAtEnd [.nop, .wend, .nop] = some ([.nop], [.nop]) := by
  rfl

/-- Nested block: outer `wend` matches outer opener, inner `wend`
    consumed inside body. -/
example :
    splitAtEnd [.block 0, .nop, .wend, .nop, .wend, .drop] =
      some ([.block 0, .nop, .wend, .nop], [.drop]) := by
  rfl

/-- If-with-else: `nop welse nop wend drop` splits to
    `(then=[nop], else=[nop], post=[drop])`. -/
example :
    splitAtElseOrEnd [.nop, .welse, .nop, .wend, .drop] =
      some ([.nop], [.nop], [.drop]) := by
  rfl

/-- If without else: `nop wend drop` splits to
    `(then=[nop], else=[], post=[drop])`. -/
example :
    splitAtElseOrEnd [.nop, .wend, .drop] =
      some ([.nop], [], [.drop]) := by
  rfl

/-- Nested if inside outer if-then: outer `welse` is found at
    depth 0 only after the inner `wif … wend` is fully consumed. -/
example :
    splitAtElseOrEnd [.wif 0, .nop, .wend, .welse, .drop, .wend, .nop] =
      some ([.wif 0, .nop, .wend], [.drop], [.nop]) := by
  rfl

end Quanta.Wasm

/-
# Equation lemmas for `evalStmt` / `evalExpr` / `evalStmts`.

After the 2026-04-29 reliability refactor of
`Quanta.KRust.Semantics`'s mutual block, all functions use
`Nat`-structural recursion. Lean now generates equation lemmas
that reduce by `rfl`. This module exposes them under stable
names so per-rule preservation theorems can `rw` against them
without depending on Lean's auto-generated equation naming.
-/

import Quanta.KRust.Syntax
import Quanta.KRust.Semantics

namespace Quanta.KRust

-- ════════════════════════════════════════════════════════════════════
-- evalStmt — per-constructor equation lemmas
-- ════════════════════════════════════════════════════════════════════

@[simp] theorem evalStmt_zero (s : State) (stmt : Stmt) :
    evalStmt 0 s stmt = none := by cases stmt <;> rfl

@[simp] theorem evalStmt_breakS (fuel : Nat) (s : State) :
    evalStmt (fuel + 1) s Stmt.breakS = some { s with broke := true } := rfl

@[simp] theorem evalStmt_letTuple
    (fuel : Nat) (s : State) (names : List Ident) (ty : List Scalar) (rhs : Expr) :
    evalStmt (fuel + 1) s (Stmt.letTuple names ty rhs) = none := rfl

@[simp] theorem evalStmt_callS
    (fuel : Nat) (s : State) (fn : Expr) (args : List Expr) :
    evalStmt (fuel + 1) s (Stmt.callS fn args) = none := rfl

end Quanta.KRust

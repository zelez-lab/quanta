/-
# Kernel Rust syntax — minimal subset

Step **E.1a** of the source-preservation track. Models the slice of
Rust that `#[quanta::kernel]` accepts. The proc macro
(`crates/quanta-macros/src/parse/{expr,stmt}.rs`) walks `syn::Expr`
and `syn::Stmt`; this module is the Lean shape of the same fragment
— each Rust constructor that the macro `match`-es lands here as a
constructor of `KRust.Expr` / `KRust.Stmt`.

Status (2026-04-28, first E commit):
- Syntax inductive landed (this file).
- `Quanta.KRust.Semantics` — big-step ⇓ — next commit (E.1b).
- `Quanta.KOps.Semantics` — KernelOps big-step — E.2.
- Translator + per-rule theorems — E.3 + E.4.
- Composition + proc-macro dual emitter — E.5 + E.6.

Naming convention: each constructor keeps the `syn::Expr` / `syn::Stmt`
variant name (`Lit`, `Path`, `Binary`, …) so reviewers can scan the
proc macro and the Lean view side-by-side. What's *not* here is
purely Rust-syntactic noise the macro normalises away
(`Expr::Paren` collapses to its inner expr; `Expr::Reference`
appears only inside `Expr::Index` for `&array[i]` — folded into
`Expr.indexRef`).

This file is a route-(a) commitment: the proc macro will eventually
emit a literal of type `KRust.Stmt` alongside its existing
`KernelOps` byte stream, and a Lean theorem will prove the two
views agree by structural induction. The shape lock-in here is
load-bearing for that proof.
-/

namespace Quanta.KRust

-- ════════════════════════════════════════════════════════════════════
-- Identifier-shaped names (variables, fields, struct types)
-- ════════════════════════════════════════════════════════════════════

/-- A Rust identifier (variable, field name, struct type name).
    Well-formedness is delegated to the proc macro at parse time;
    the Lean view trusts the syn tokens it receives. -/
abbrev Ident : Type := String

-- ════════════════════════════════════════════════════════════════════
-- Scalar types — what the kernel can bind
-- ════════════════════════════════════════════════════════════════════
--
-- Mirrors `quanta_ir::ScalarType` and the `scalar_type_from_path`
-- mapper in `crates/quanta-macros/src/parse.rs`. Only types the
-- Rust subset accepts at the parameter and binding level land here.

inductive Scalar where
  | bool
  | i8 | i16 | i32 | i64
  | u8 | u16 | u32 | u64
  | f16 | f32 | f64
  deriving Repr, DecidableEq

/-- Surface name (matches the Rust `path` segment the proc macro
    reads, e.g. `f32`, `u32`). Used at the proc-macro view boundary. -/
def Scalar.surface : Scalar → String
  | .bool => "bool"
  | .i8 => "i8" | .i16 => "i16" | .i32 => "i32" | .i64 => "i64"
  | .u8 => "u8" | .u16 => "u16" | .u32 => "u32" | .u64 => "u64"
  | .f16 => "f16" | .f32 => "f32" | .f64 => "f64"

-- ════════════════════════════════════════════════════════════════════
-- Literals
-- ════════════════════════════════════════════════════════════════════
--
-- Abstract literal shape covering bool, signed int, unsigned
-- int, and float (f32 / f64). The `whole`/`frac` shape on floats
-- avoids modeling IEEE-754 in the syntax layer — the semantics
-- module recovers a numeric value from those fields.
--
-- This used to mirror `emit_literal` in
-- `crates/quanta-macros/src/parse/expr.rs`, which was deleted in
-- the WASM-route cutover (2026-05-05). The production path now
-- uses `compile_via_wasm` + `quanta-wasm-lowering`. See
-- memory/verification_status_post_cutover.md for what this Lean
-- model still covers (abstract spec) vs. what's unverified (the
-- WASM-route translator implementation).

inductive Lit where
  | bool   (b : Bool)
  | int    (n : Int) (ty : Scalar)
  | float  (whole frac : Int) (ty : Scalar)
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- Operators
-- ════════════════════════════════════════════════════════════════════
--
-- Mirrors `syn::BinOp` filtered through `syn_binop_to_ir` and
-- `syn_binop_to_cmp` in `crates/quanta-macros/src/parse.rs`. We
-- collapse the parser's "regular vs cmp" split into a single tag —
-- the per-rule preservation theorem (E.4) cases on it once.

inductive BinOp where
  | add | sub | mul | div | rem
  | shl | shr | bAnd | bOr | bXor
  | logAnd | logOr
  | eq | ne | lt | le | gt | ge
  deriving Repr, DecidableEq

inductive UnaryOp where
  | neg | logNot | bNot
  deriving Repr, DecidableEq

/-- Math intrinsics the macro recognises by name. Mirrors
    `name_to_math_fn` in `crates/quanta-macros/src/parse.rs`. -/
inductive MathFn where
  | sin | cos | tan | asin | acos | atan | atan2
  | exp | exp2 | ln | log2 | log10
  | sqrt | rsqrt | invSqrt
  | abs | floor | ceil | round | trunc | fract
  | min | max | clamp | sign
  | fma | mix | step | smoothstep
  | pow
  deriving Repr, DecidableEq

/-- Atomic ops the kernel can perform on storage fields. Mirrors
    `quanta_ir::AtomicOp`; the Rust subset surfaces these as free-
    function calls (`atomic_add`, `atomic_cas`, …). -/
inductive AtomicFn where
  | add | sub | min_ | max_ | bAnd | bOr | bXor | exchange | cas
  deriving Repr, DecidableEq

/-- Wave / subgroup intrinsics. Mirrors `KernelOp::Wave*` plus the
    proc macro's `wave_*` free-function recognition. -/
inductive WaveFn where
  | shuffle | ballot | any_ | all_
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- Expressions
-- ════════════════════════════════════════════════════════════════════
--
-- Abstract expression syntax — one constructor per syn::Expr
-- arm a kernel-source translator should accept. Used to be a
-- direct mirror of `emit_expr` in
-- `crates/quanta-macros/src/parse/expr.rs`; that translator was
-- deleted in the WASM-route cutover (2026-05-05) and the
-- production path is no longer this Lean spec's direct image.
-- See memory/verification_status_post_cutover.md.

mutual
inductive Expr where
  | lit       (l : Lit)
  /-- `Expr::Path` — variable read by name. -/
  | path      (name : Ident)
  /-- `Expr::Binary`. -/
  | binary    (op : BinOp) (lhs rhs : Expr)
  /-- `Expr::Unary`. -/
  | unary     (op : UnaryOp) (e : Expr)
  /-- `Expr::Index` — `array[idx]`. The `array` slot is identified by
      its parameter name; the macro resolves that to a slot index
      at `KernelOps` translation time (E.3). -/
  | index     (array : Ident) (idx : Expr)
  /-- `&array[idx]` — atomic-op argument shape. The macro folds
      `Expr::Reference + Expr::Index` into this single constructor;
      it cannot appear elsewhere. -/
  | indexRef  (array : Ident) (idx : Expr)
  /-- `Expr::Field` — `p.field` access on a struct-ref kernel param. -/
  | fieldRef  (base : Ident) (field : Ident)
  /-- `Expr::Call` of a free function: a math intrinsic, a wave
      intrinsic, an atomic op, or a `quark_id()`-style identity. -/
  | mathCall   (fn : MathFn)   (args : List Expr)
  | waveCall   (fn : WaveFn)   (args : List Expr)
  | atomicCall (fn : AtomicFn) (args : List Expr)
  /-- Quanta-IR identity ops the kernel surfaces as zero-arg calls:
      `quark_id`, `proton_id`, `nucleus_id`, `proton_size`,
      `quark_count`. -/
  | identityCall (name : Ident)
  /-- `Expr::MethodCall` — `x.sqrt()`, `x.abs()`, … reduced to the
      same set as free-function math calls. -/
  | method    (recv : Expr) (fn : MathFn) (args : List Expr)
  /-- `Expr::If` as an expression — last expression of each branch
      is the value. The block bodies carry the surrounding stmts. -/
  | ifE       (cond : Expr) (thenE : Expr) (elseE : Expr)
  /-- `Expr::Cast` — `x as Ty`. -/
  | cast      (e : Expr) (to : Scalar)
  /-- `Expr::Block` — `{ stmts; final_expr }`. The last position is
      an `Expr` so the block has a value; preceding positions are
      statements. -/
  | blockE    (body : List Stmt) (tail : Expr)
  deriving Repr

inductive Stmt where
  /-- `Stmt::Local` — `let x: Ty = rhs;`. Type annotation optional. -/
  | letDecl   (name : Ident) (ty : Option Scalar) (rhs : Expr)
  /-- Tuple-destructuring let — `let (a, b) = rhs;`. Used by the
      proc macro to multi-return from `Expr::Tuple` RHS. -/
  | letTuple  (names : List Ident) (ty : List Scalar) (rhs : Expr)
  /-- Bare expression statement (no semicolon required for the
      block tail; semicolon-terminated forms reuse this). -/
  | exprS     (e : Expr)
  /-- `x = rhs` — variable rebinding. -/
  | assignVar (name : Ident) (rhs : Expr)
  /-- `array[idx] = rhs` — buffer write. -/
  | assignIdx (array : Ident) (idx : Expr) (rhs : Expr)
  /-- `if cond { ... } else { ... }` as a statement (no value). -/
  | ifS       (cond : Expr) (thenS elseS : List Stmt)
  /-- `for i in 0..n { body }` — the only loop form the macro
      accepts at parameter level. The proc macro inlines the
      iterator over `Range<u32>`; Lean models it as a structural
      counted loop. -/
  | forRange  (name : Ident) (lo hi : Expr) (body : List Stmt)
  /-- `while cond { body }` — the macro lowers this to a structural
      `loop { if !cond { break } else { body } }` shape; we keep the
      surface form here and let the translator rewrite. -/
  | whileS    (cond : Expr) (body : List Stmt)
  /-- `loop { body }` — bare structural loop. The macro requires a
      `break` somewhere reachable; not enforced syntactically. -/
  | loopS     (body : List Stmt)
  /-- `break` (only legal inside a `loopS` / `whileS` / `forRange`
      body; well-formedness checked by E.1b's semantics). -/
  | breakS
  /-- A free-function call used as a statement (atomic ops, debug
      labels, …). -/
  | callS     (fn : Expr) (args : List Expr)
  deriving Repr
end

-- ════════════════════════════════════════════════════════════════════
-- Kernel parameters
-- ════════════════════════════════════════════════════════════════════
--
-- Mirrors `quanta_ir::KernelParam`. The proc macro discovers these
-- in two ways: flat function args (`fn k(a: &[f32], n: u32)`) or
-- struct-ref body scanning (`fn k(p: &Particles)` → walk `p.field`).
-- Both paths produce the same `Param` shape on the IR side.

inductive ParamKind where
  | fieldRead
  | fieldWrite
  | constant     -- push constant / scalar param
  deriving Repr, DecidableEq

structure Param where
  name        : Ident
  kind        : ParamKind
  /-- Slot index in the kernel's parameter list — assigned by the
      macro at parse time, stable across translation. -/
  slot        : Nat
  /-- Element type for buffers; scalar type for constants. -/
  scalarType  : Scalar
  deriving Repr

-- ════════════════════════════════════════════════════════════════════
-- Top-level kernel
-- ════════════════════════════════════════════════════════════════════

/-- A Kernel Rust function the macro accepts. The body is the
    sequence of `Stmt`s the proc macro walks; `params` carries the
    discovered (flat-arg or struct-ref) parameter list. -/
structure Kernel where
  /-- Function name (e.g. `"add_one"`). -/
  name          : Ident
  params        : List Param
  /-- Workgroup size declared via `#[quanta::kernel(workgroup_size = …)]`. -/
  workgroupSize : Nat × Nat × Nat
  body          : List Stmt
  deriving Repr

end Quanta.KRust

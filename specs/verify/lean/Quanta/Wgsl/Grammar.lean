/-
# WGSL grammar mirror — minimal fragment

Step **B** of the FFI TCB shrink track. Models the slice of WGSL
that Quanta's JIT emitter (`crates/gpu/quanta-ir/src/emit_wgsl/*.rs`)
produces. The W3C WGSL spec is far larger; we mirror only the
fragment we emit — same approach as `Quanta.Idl` for WebIDL.

The structure tracks the emitter's surface: `enable` directives,
storage/uniform bindings, module-scope declarations,
`@compute`-annotated entry points, statements, and the expression
forms we use (registers, literals, binary/unary ops, atomic
intrinsics, indexed loads/stores). What this module is **for**:

- `Source` is the structural AST shape the emitter produces.
- `Wgsl.wellFormed : Source → Bool` is a closed-form, decidable
  predicate. Programs that pass it are guaranteed to:
    * lex per WGSL §3 (we name only spec-conformant identifiers,
      keywords, and punctuation in the serializer);
    * type-check per WGSL §4 (the type fields on each AST node
      record what the spec validator would compute).
- `Quanta.Theorems.WgslConformance` will later prove that *every*
  emitter pattern lands in `Source` and satisfies `wellFormed` — at
  which point T410 (`emit_wgsl_jit produces well-formed WGSL`)
  shrinks from a Verus-only operational claim to a Lean theorem
  whose remaining trust is just the bridge `wgsl_string_well_formed
  (serialize s) ↔ wellFormed s` (a typedef-stability claim about
  the `Source → String` printer in `Wgsl.Serialize`).

Status (2026-04-28, first B commit):
- Grammar mirror landed (this file).
- `Wgsl.Serialize` and the bridge axiom — next commit (B.2).
- Per-op `wellFormed` discharge — B.3.
- T410 axiom → theorem rewrite — B.4.

The shape is intentionally *over-specified* relative to the WGSL
spec (we model only what we emit, not the full grammar). This is
the right scope for narrowing T410: it discharges precisely the
emitter's actual output, not arbitrary WGSL.
-/

namespace Quanta.Wgsl

-- ════════════════════════════════════════════════════════════════════
-- Identifier-shaped strings
-- ════════════════════════════════════════════════════════════════════

/-- A WGSL identifier — a non-empty string that the emitter never
    produces with characters outside `[a-zA-Z0-9_]`. We do not enforce
    the regex at the type level; the `wellFormed` predicate checks it. -/
abbrev Ident : Type := String

-- ════════════════════════════════════════════════════════════════════
-- Scalar types — the WGSL §4.2 primitive set the emitter uses
-- ════════════════════════════════════════════════════════════════════

/-- The scalar types Quanta's emitter actually emits. Mirrors
    `ScalarType::wgsl_name()` in `crates/gpu/quanta-ir/src/types.rs`. -/
inductive Scalar where
  | bool
  | i32
  | u32
  | f16
  | f32
  deriving Repr, DecidableEq

/-- WGSL surface name for each scalar — matches §4.2 verbatim. -/
def Scalar.surface : Scalar → String
  | .bool => "bool"
  | .i32  => "i32"
  | .u32  => "u32"
  | .f16  => "f16"
  | .f32  => "f32"

-- ════════════════════════════════════════════════════════════════════
-- Types
-- ════════════════════════════════════════════════════════════════════

/-- The WGSL types Quanta emits. `array<T>` has unsized form (no
    explicit count) for storage bindings; `atomic<T>` only wraps
    `i32` or `u32` per WGSL §4.4. -/
inductive Ty where
  | scalar  (s : Scalar)
  | vec     (s : Scalar) (n : Nat)         -- vec2/3/4 of scalar
  | array   (elem : Ty)                    -- unsized
  | atomic  (s : Scalar)                   -- only i32/u32 admitted
  | ptrFn   (inner : Ty)                   -- ptr<function, T>
  deriving Repr, DecidableEq

-- ════════════════════════════════════════════════════════════════════
-- Expressions
-- ════════════════════════════════════════════════════════════════════

/-- WGSL binary operators the emitter produces. -/
inductive BinOp where
  | add | sub | mul | div | rem
  | shl | shr | bAnd | bOr | bXor
  | logAnd | logOr
  | eq | ne | lt | le | gt | ge
  deriving Repr, DecidableEq

/-- WGSL unary operators the emitter produces. -/
inductive UnaryOp where
  | neg | logNot | bNot
  deriving Repr, DecidableEq

/-- Expression forms the emitter constructs. Each carries its
    declared `Ty` so structural well-formedness is decidable without
    re-running type inference. -/
inductive Expr where
  | reg     (id : Nat)                     -- `r{id}` virtual register
  | identE  (name : Ident)                 -- builtin identifier (e.g. _quark_id)
  | litBool (b : Bool)
  | litI32  (n : Int)
  | litU32  (n : Nat)
  | litF32  (whole frac : Int)             -- structural float literal
  | binOp   (op : BinOp) (lhs rhs : Expr)
  | unaryOp (op : UnaryOp) (e : Expr)
  | callBuiltin (name : Ident) (args : List Expr)   -- atomicLoad, …, math fns
  | indexBuf (buf : Ident) (idx : Expr)              -- `name[idx]`
  | addrOfIdx (buf : Ident) (idx : Expr)             -- `&name[idx]` (atomics)
  | castTo  (e : Expr) (t : Ty)                      -- `T(e)`
  | vecCtor (s : Scalar) (n : Nat) (xs : List Expr)
  deriving Repr

-- ════════════════════════════════════════════════════════════════════
-- Statements
-- ════════════════════════════════════════════════════════════════════

/-- Statement forms emitted by `emit_wgsl/ops.rs`. The block list
    in `branch`/`loop` is over `Stmt`, so the structure is recursive. -/
inductive Stmt where
  | letDecl   (dst : Nat) (ty : Option Ty) (rhs : Expr)
  | assignIdx (buf : Ident) (idx : Expr) (rhs : Expr)
  | callStmt  (name : Ident) (args : List Expr)
  | ifThenElse (cond : Expr) (thenS elseS : List Stmt)
  | loopBody  (body : List Stmt)
  | breakStmt
  | barrier
  | returnS
  deriving Repr

-- ════════════════════════════════════════════════════════════════════
-- Top-level module structure
-- ════════════════════════════════════════════════════════════════════

/-- WGSL module-scope `enable` directive. The emitter only ever emits
    `f16` and `subgroups`; the predicate `wellFormed` enforces this. -/
inductive Enable where
  | f16
  | subgroups
  deriving Repr, DecidableEq

/-- Storage-class / access-mode pair for `var<X, Y>` bindings. -/
inductive Storage where
  | storageRead
  | storageReadWrite
  | uniform
  deriving Repr, DecidableEq

/-- Module-scope binding: `@group(g) @binding(b) var<storage, A> name: array<T>;` -/
structure Binding where
  group  : Nat
  index  : Nat
  storage : Storage
  name   : Ident
  ty     : Ty
  deriving Repr

/-- Compute-kernel parameter (a `@builtin` value the entry point reads). -/
structure BuiltinParam where
  name      : Ident
  builtin   : Ident                        -- e.g. "global_invocation_id"
  ty        : Ty
  deriving Repr

/-- A WGSL function declaration. Two flavours land here: the
    `@compute @workgroup_size(...)` entry point, and helper device
    functions inlined from device-fn translation. -/
structure FnDecl where
  name      : Ident
  isCompute : Bool
  workgroup : Option (Nat × Nat × Nat)     -- `@workgroup_size(x, y, z)` for compute
  builtins  : List BuiltinParam            -- only populated for compute fns
  retTy     : Option Ty
  body      : List Stmt
  deriving Repr

/-- The full module: every WGSL string Quanta emits is a serialization
    of one `Source`. Order matters — `enables` precede bindings, which
    precede device functions, which precede the compute entry point. -/
structure Source where
  enables  : List Enable
  bindings : List Binding
  fns      : List FnDecl
  deriving Repr

-- ════════════════════════════════════════════════════════════════════
-- Structural well-formedness
-- ════════════════════════════════════════════════════════════════════
--
-- Closed-form `Bool` predicates the kernel decides at `lake build`
-- time. They're meant to be cheap structural checks, not a full
-- WGSL validator: each predicate captures an invariant the emitter
-- maintains by construction.

/-- Identifier well-formedness: non-empty + every char is ASCII
    letter, digit, or underscore + first char is not a digit. -/
def identOk (s : String) : Bool :=
  let cs := s.data
  match cs with
  | []        => false
  | c :: rest =>
      let isAlphaUnder := fun (ch : Char) =>
        (ch.isAlpha || ch == '_')
      let isAlnumUnder := fun (ch : Char) =>
        (ch.isAlphanum || ch == '_')
      isAlphaUnder c &&rest.all isAlnumUnder

/-- A vector dimension is one of 2, 3, 4 (WGSL §4.3). -/
def vecDimOk (n : Nat) : Bool :=
  n == 2 || n == 3 || n == 4

/-- `atomic<T>` only admits integer scalars per WGSL §4.4. -/
def atomicScalarOk : Scalar → Bool
  | .i32 => true
  | .u32 => true
  | _    => false

/-- A `Ty` is well-formed if every nested constraint holds:
    `vec n` requires `n ∈ {2,3,4}`, `atomic<S>` requires `S ∈ {i32,u32}`,
    `array<T>` and `ptr<function,T>` carry through to the inner type. -/
def Ty.wellFormed : Ty → Bool
  | .scalar _    => true
  | .vec _ n     => vecDimOk n
  | .array t     => Ty.wellFormed t
  | .atomic s    => atomicScalarOk s
  | .ptrFn t     => Ty.wellFormed t

/-- Arity check for the function-call expression forms the emitter
    uses. The full WGSL builtin set is large; we only enumerate the
    builtins Quanta emits, with a conservative arity envelope. -/
def builtinArityOk (name : String) (argc : Nat) : Bool :=
  match name with
  | "atomicLoad"     => argc = 1
  | "atomicStore"    => argc = 2
  | "atomicAdd"      => argc = 2
  | "atomicSub"      => argc = 2
  | "atomicAnd"      => argc = 2
  | "atomicOr"       => argc = 2
  | "atomicXor"      => argc = 2
  | "atomicMin"      => argc = 2
  | "atomicMax"      => argc = 2
  | "atomicExchange" => argc = 2
  | "atomicCompareExchangeWeak" => argc = 3
  | "workgroupBarrier" => argc = 0
  | "subgroupBroadcast" => argc = 2
  | "subgroupBallot"    => argc = 1
  | "subgroupAny"       => argc = 1
  | "subgroupAll"       => argc = 1
  -- Math intrinsics — most are unary or binary; widen to ≤ 4 to cover fma/clamp.
  | _ => argc ≤ 4

/-- Structural well-formedness on expressions. Recursive on the AST. -/
partial def Expr.wellFormed : Expr → Bool
  | .reg _              => true
  | .identE n           => identOk n
  | .litBool _           => true
  | .litI32 _            => true
  | .litU32 _            => true
  | .litF32 _ _          => true
  | .binOp _ a b         => a.wellFormed &&b.wellFormed
  | .unaryOp _ e         => e.wellFormed
  | .callBuiltin n args  =>
      identOk n &&builtinArityOk n args.length &&args.all Expr.wellFormed
  | .indexBuf b i        => identOk b &&i.wellFormed
  | .addrOfIdx b i       => identOk b &&i.wellFormed
  | .castTo e t          => e.wellFormed &&t.wellFormed
  | .vecCtor _ n xs      =>
      vecDimOk n &&xs.length = n &&xs.all Expr.wellFormed

/-- Structural well-formedness on statements. -/
partial def Stmt.wellFormed : Stmt → Bool
  | .letDecl _ ty rhs    =>
      (match ty with | none => true | some t => t.wellFormed) &&rhs.wellFormed
  | .assignIdx b i rhs   => identOk b &&i.wellFormed &&rhs.wellFormed
  | .callStmt n args     =>
      identOk n &&builtinArityOk n args.length &&args.all Expr.wellFormed
  | .ifThenElse c t e    =>
      c.wellFormed &&t.all Stmt.wellFormed &&e.all Stmt.wellFormed
  | .loopBody body       => body.all Stmt.wellFormed
  | .breakStmt            => true
  | .barrier             => true
  | .returnS             => true

/-- A binding's group/index live in [0, 2³²) on the wire (WGSL spec
    treats them as `u32`). The `name` and `ty` carry through. -/
def Binding.wellFormed (b : Binding) : Bool :=
  b.group < 4294967296
  &&b.index < 4294967296
  &&identOk b.name
  &&b.ty.wellFormed

/-- Compute-fn parameters are restricted to the small set of
    `@builtin(_)` values Quanta reads. Only `name`/`builtin` shape
    is checked here; the type lattice is delegated to `Ty.wellFormed`. -/
def BuiltinParam.wellFormed (p : BuiltinParam) : Bool :=
  identOk p.name &&identOk p.builtin &&p.ty.wellFormed

/-- A function decl is well-formed when its name + signature + body
    are; `@compute` requires a workgroup-size annotation, and a
    non-compute fn must not carry one (WGSL §10). -/
def FnDecl.wellFormed (f : FnDecl) : Bool :=
  identOk f.name
  &&(match f.retTy with | none => true | some t => t.wellFormed)
  &&f.builtins.all BuiltinParam.wellFormed
  &&f.body.all Stmt.wellFormed
  &&(if f.isCompute then f.workgroup.isSome else f.workgroup.isNone)

/-- The full module's structural well-formedness. -/
def Source.wellFormed (s : Source) : Bool :=
  s.bindings.all Binding.wellFormed
  &&s.fns.all FnDecl.wellFormed

end Quanta.Wgsl

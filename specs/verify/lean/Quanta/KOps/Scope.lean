import Quanta.KOps.Syntax

/-!
# KernelOp scope-validity invariant

Captures the structural well-formedness condition that:
**every register referenced by a KernelOp must have been defined
by an earlier op in the same scope or an enclosing scope.**

Concretely:
- For straight-line ops (Const, BinOp, Cmp, Copy, Load, Store,
  Cast, Bitcast, UnaryOp, …) the dst (if any) becomes available
  after the op; the src(s) must already be in the env.
- For `branch cond thenOps elseOps`: the `cond` must already be in
  the env; the thenOps and elseOps are each scope-validated
  against the same env (Branch doesn't define a reg at parent scope).
- For `loopOp body`: body is scope-validated against the parent's
  env (Loop doesn't define a reg at parent scope).

This predicate is the IR-side mirror of the lexical scope guarantees
that downstream emitters (MSL, SPIR-V, WGSL) implicitly assume. The
WASM-route lowering currently has a known bug
(`BrIf` cond-scope, see `crates/quanta-wasm-lowering/src/lower.rs`
line ~1880) where a `branch` op is emitted with a `cond` whose
defining op lives inside the branch's else_ops — violating this
predicate. The predicate doesn't fix the bug, but provides a
target for a future Lean refinement proof that the WASM-route
lowering preserves it.

This file is invariant-only: it defines the predicate and a couple
of sanity smoke tests. Discharging it for arbitrary lowering
outputs is downstream work (likely Verus).
-/

namespace Quanta.KOps

open Quanta.KOps

/-- Registers used by a single KernelOp (excluding the dst the op
    defines). Membership-based; used by the scope-validity walk. -/
def KernelOp.usedRegs : KernelOp → List Reg
  | .const _ _                    => []
  | .binOp _ a b _ _              => [a, b]
  | .unaryOp _ a _ _              => [a]
  | .cmp _ a b _ _                => [a, b]
  | .cast _ src _ _               => [src]
  | .bitcast _ src _ _            => [src]
  | .copy _ src                   => [src]
  | .load _ _ index _             => [index]
  | .store _ index src _          => [index, src]
  | .branch cond _ _              => [cond]
  | .loopOp _                     => []
  | .breakOp                      => []
  | .quarkId _                    => []
  | .protonId _                   => []
  | .nucleusId _                  => []
  | .protonSize _                 => []
  | .quarkCount _                 => []
  | .barrier                      => []

/-- Register defined by a single KernelOp (if any). -/
def KernelOp.definedReg : KernelOp → Option Reg
  | .const dst _                  => some dst
  | .binOp dst _ _ _ _            => some dst
  | .unaryOp dst _ _ _            => some dst
  | .cmp dst _ _ _ _              => some dst
  | .cast dst _ _ _               => some dst
  | .bitcast dst _ _ _            => some dst
  | .copy dst _                   => some dst
  | .load dst _ _ _               => some dst
  | .store _ _ _ _                => none
  | .branch _ _ _                 => none
  | .loopOp _                     => none
  | .breakOp                      => none
  | .quarkId dst                  => some dst
  | .protonId dst                 => some dst
  | .nucleusId dst                => some dst
  | .protonSize dst               => some dst
  | .quarkCount dst               => some dst
  | .barrier                      => none

/-- Add the op's defined register (if any) to the running env. -/
def KernelOp.extendEnv (env : List Reg) (op : KernelOp) : List Reg :=
  match op.definedReg with
  | some r => r :: env
  | none   => env

mutual
/-- A single KernelOp is scope-valid against `env` iff every reg it
    uses is in env, plus (for Branch/Loop) the recursive bodies are
    themselves scope-valid against the same env.

    Note: Branch's then/else are validated against the parent's env
    (not env extended with any "branch-introduced" reg, since
    Branch doesn't define a reg). Loop's body is validated against
    parent's env as well. -/
def KernelOp.scopeValid (env : List Reg) : KernelOp → Prop
  | .branch cond thenOps elseOps =>
      cond ∈ env ∧
      KernelOp.scopeValidOps env thenOps ∧
      KernelOp.scopeValidOps env elseOps
  | .loopOp body =>
      KernelOp.scopeValidOps env body
  | op =>
      ∀ r ∈ KernelOp.usedRegs op, r ∈ env

/-- A list of KernelOps is scope-valid against `env` iff each op is
    scope-valid against the running env extended by all prior ops'
    defined regs. -/
def KernelOp.scopeValidOps (env : List Reg) : List KernelOp → Prop
  | []        => True
  | op :: rest =>
      KernelOp.scopeValid env op ∧
      KernelOp.scopeValidOps (KernelOp.extendEnv env op) rest
end

-- ════════════════════════════════════════════════════════════════════
-- Composition: scope-valid lists append into a scope-valid list
--
-- If `ops1` is scope-valid against env, and `ops2` is scope-valid
-- against the env extended by all of ops1's defined regs, then
-- `ops1 ++ ops2` is scope-valid against env. This is the structural
-- statement that scope validity is preserved by sequential
-- composition of sub-lowerings.
-- ════════════════════════════════════════════════════════════════════

/-- Helper: fold extendEnv over a list of ops. -/
def KernelOp.extendEnvOps (env : List Reg) : List KernelOp → List Reg
  | []         => env
  | op :: rest => KernelOp.extendEnvOps (KernelOp.extendEnv env op) rest

/-- `extendEnvOps` distributes over list append:
    folding the extend function over `ops1 ++ ops2` equals folding
    over `ops1` and then `ops2`. Pure structural lemma; lives at the
    same level as `List.foldl_append` and gets used by every
    composition proof that needs to talk about the env after a
    concatenated sub-lowering. -/
theorem KernelOp.extendEnvOps_append : ∀ (env : List Reg) (ops1 ops2 : List KernelOp),
    KernelOp.extendEnvOps env (ops1 ++ ops2) =
      KernelOp.extendEnvOps (KernelOp.extendEnvOps env ops1) ops2
  | _, [],         _    => rfl
  | env, op :: rest, ops2 => by
    show KernelOp.extendEnvOps (KernelOp.extendEnv env op) (rest ++ ops2) =
         KernelOp.extendEnvOps
           (KernelOp.extendEnvOps (KernelOp.extendEnv env op) rest) ops2
    exact KernelOp.extendEnvOps_append (KernelOp.extendEnv env op) rest ops2

/-- `extendEnv` preserves ⊆: if env ⊆ env', then
    `extendEnv env op ⊆ extendEnv env' op`. -/
theorem KernelOp.extendEnv_mono {env env' : List Reg} (h : env ⊆ env')
    (op : KernelOp) : KernelOp.extendEnv env op ⊆ KernelOp.extendEnv env' op := by
  unfold KernelOp.extendEnv
  cases op.definedReg with
  | none => exact h
  | some r =>
    intro x hx
    simp at hx
    rcases hx with rfl | hx
    · simp
    · exact List.mem_cons_of_mem _ (h hx)

/-- `extendEnvOps` preserves ⊆. -/
theorem KernelOp.extendEnvOps_mono : ∀ (ops : List KernelOp) {env env' : List Reg},
    env ⊆ env' →
    KernelOp.extendEnvOps env ops ⊆ KernelOp.extendEnvOps env' ops
  | [], _, _, h => h
  | op :: rest, env, env', h => by
    show KernelOp.extendEnvOps (KernelOp.extendEnv env op) rest ⊆ _
    exact KernelOp.extendEnvOps_mono rest (KernelOp.extendEnv_mono h op)

/-- The env after walking ops always contains the starting env. -/
theorem KernelOp.extendEnvOps_super (env : List Reg) (ops : List KernelOp) :
    env ⊆ KernelOp.extendEnvOps env ops := by
  induction ops generalizing env with
  | nil => intro x hx; exact hx
  | cons op rest ih =>
    intro x hx
    apply ih
    unfold KernelOp.extendEnv
    cases op.definedReg with
    | none => exact hx
    | some r => simp [List.mem_cons]; right; exact hx

theorem KernelOp.scopeValidOps_append (env : List Reg) (ops1 ops2 : List KernelOp) :
    KernelOp.scopeValidOps env ops1 →
    KernelOp.scopeValidOps (KernelOp.extendEnvOps env ops1) ops2 →
    KernelOp.scopeValidOps env (ops1 ++ ops2) := by
  induction ops1 generalizing env with
  | nil =>
    intro _ h2
    show KernelOp.scopeValidOps env ops2
    exact h2
  | cons op rest ih =>
    intro ⟨hop, hrest⟩ h2
    refine ⟨hop, ?_⟩
    -- Apply IH at env := extendEnv env op.
    show KernelOp.scopeValidOps (KernelOp.extendEnv env op) (rest ++ ops2)
    apply ih
    · exact hrest
    · -- h2 has env = extendEnvOps env (op :: rest) which unfolds to
      -- extendEnvOps (extendEnv env op) rest. That's exactly the env
      -- we need.
      show KernelOp.scopeValidOps
            (KernelOp.extendEnvOps (KernelOp.extendEnv env op) rest) ops2
      exact h2

/-- Converse of `scopeValidOps_append`: a valid concatenated chain
    splits back into a valid prefix plus a valid suffix against the
    env extended through the prefix. Needed whenever a proof inducts
    over the *output* of a translator that emits sub-lowerings
    concatenated together (we recover the per-sub-lowering
    obligations from the overall validity hypothesis). -/
theorem KernelOp.scopeValidOps_append_inv :
    ∀ (env : List Reg) (ops1 ops2 : List KernelOp),
    KernelOp.scopeValidOps env (ops1 ++ ops2) →
    KernelOp.scopeValidOps env ops1 ∧
    KernelOp.scopeValidOps (KernelOp.extendEnvOps env ops1) ops2
  | _,   [],         _,    h => ⟨trivial, h⟩
  | env, op :: rest, ops2, h => by
    -- h : scopeValidOps env (op :: (rest ++ ops2))
    --   = scopeValid env op ∧ scopeValidOps (extendEnv env op) (rest ++ ops2)
    obtain ⟨hop, htail⟩ := h
    -- IH peels rest ++ ops2 at env' := extendEnv env op.
    obtain ⟨hrest, htail2⟩ :=
      KernelOp.scopeValidOps_append_inv (KernelOp.extendEnv env op) rest ops2 htail
    refine ⟨⟨hop, hrest⟩, ?_⟩
    -- htail2 : scopeValidOps (extendEnvOps (extendEnv env op) rest) ops2,
    -- which is definitionally extendEnvOps env (op :: rest).
    exact htail2

-- ════════════════════════════════════════════════════════════════════
-- Monotonicity in env
--
-- If a list is scope-valid against env, it remains scope-valid
-- against any larger env. Compositional refinement proofs use
-- this to widen the environment as they compose sub-lowerings.
-- ════════════════════════════════════════════════════════════════════

mutual
theorem KernelOp.scopeValid_mono : ∀ (op : KernelOp) {env env' : List Reg},
    env ⊆ env' → KernelOp.scopeValid env op → KernelOp.scopeValid env' op
  | .branch cond thenOps elseOps, env, env', h, hv => by
    obtain ⟨hcond, hthen, helse⟩ := hv
    refine ⟨h hcond, ?_, ?_⟩
    · exact KernelOp.scopeValidOps_mono thenOps h hthen
    · exact KernelOp.scopeValidOps_mono elseOps h helse
  | .loopOp body, env, env', h, hv =>
    KernelOp.scopeValidOps_mono body h hv
  | .const _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .binOp _ _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .unaryOp _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .cmp _ _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .cast _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .bitcast _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .copy _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .load _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .store _ _ _ _, _, _, h, hv => fun r hr => h (hv r hr)
  | .breakOp, _, _, h, hv => fun r hr => h (hv r hr)
  | .quarkId _, _, _, h, hv => fun r hr => h (hv r hr)
  | .protonId _, _, _, h, hv => fun r hr => h (hv r hr)
  | .nucleusId _, _, _, h, hv => fun r hr => h (hv r hr)
  | .protonSize _, _, _, h, hv => fun r hr => h (hv r hr)
  | .quarkCount _, _, _, h, hv => fun r hr => h (hv r hr)
  | .barrier, _, _, h, hv => fun r hr => h (hv r hr)

theorem KernelOp.scopeValidOps_mono : ∀ (ops : List KernelOp) {env env' : List Reg},
    env ⊆ env' →
    KernelOp.scopeValidOps env ops → KernelOp.scopeValidOps env' ops
  | [], _, _, _, _ => trivial
  | op :: rest, env, env', h, ⟨hop, hrest⟩ => by
    refine ⟨KernelOp.scopeValid_mono op h hop, ?_⟩
    -- After extending by op on both sides, the ⊆ relation is preserved.
    have hext : KernelOp.extendEnv env op ⊆ KernelOp.extendEnv env' op := by
      unfold KernelOp.extendEnv
      cases op.definedReg with
      | none => exact h
      | some r =>
        intro x hx
        simp at hx
        rcases hx with rfl | hx
        · simp
        · exact List.mem_cons_of_mem _ (h hx)
    exact KernelOp.scopeValidOps_mono rest hext hrest
end

-- ════════════════════════════════════════════════════════════════════
-- Destructors / bridges
--
-- Small structural lemmas used to move between the single-op
-- predicate (`scopeValid`) and the list predicate (`scopeValidOps`),
-- and to peel a cons. Current proofs do this inline via pattern
-- match; surfacing them as named lemmas keeps downstream proofs
-- shallow.
-- ════════════════════════════════════════════════════════════════════

/-- Bridge: a singleton list is scope-valid iff the op is. -/
theorem KernelOp.scopeValid_singleton (env : List Reg) (op : KernelOp) :
    KernelOp.scopeValidOps env [op] ↔ KernelOp.scopeValid env op := by
  constructor
  · intro ⟨h, _⟩; exact h
  · intro h; exact ⟨h, trivial⟩

/-- Destructor: a valid cons gives both the head's validity and the
    tail's validity against the env extended by the head. -/
theorem KernelOp.scopeValidOps_cons_inv {env : List Reg}
    {op : KernelOp} {rest : List KernelOp}
    (h : KernelOp.scopeValidOps env (op :: rest)) :
    KernelOp.scopeValid env op ∧
    KernelOp.scopeValidOps (KernelOp.extendEnv env op) rest := h

-- ════════════════════════════════════════════════════════════════════
-- Sanity smoke tests (decide-able at definition time)
-- ════════════════════════════════════════════════════════════════════

/-- An empty op-list is trivially scope-valid against any env. -/
example : KernelOp.scopeValidOps [] [] := trivial

/-- A single `const 0 = 42` is scope-valid against any env (no uses). -/
example : KernelOp.scopeValidOps []
    [.const 0 (.i32 42)] := by
  refine ⟨?_, trivial⟩
  intro r hr
  simp [KernelOp.usedRegs] at hr

/-- `const 0; const 1; binOp 2 0 1 add u32` is scope-valid.
    Each binOp argument was defined earlier in the list. -/
example : KernelOp.scopeValidOps []
    [.const 0 (.i32 1), .const 1 (.i32 2),
     .binOp 2 0 1 .add .u32] := by
  refine ⟨?_, ?_, ?_, trivial⟩
  · intro r hr; simp [KernelOp.usedRegs] at hr
  · intro r hr; simp [KernelOp.usedRegs] at hr
  · intro r hr
    simp [KernelOp.usedRegs] at hr
    rcases hr with rfl | rfl <;>
      simp [KernelOp.extendEnv, KernelOp.definedReg]

/-- The bug pattern (use-before-def, expected NOT scope-valid):
    `branch r44 [] [const r44 0]` references r44 in the branch
    condition but r44 is only defined inside the else-branch. -/
example : ¬ KernelOp.scopeValidOps []
    [.branch 44 [] [.const 44 (.i32 0)]] := by
  intro h
  obtain ⟨hbr, _⟩ := h
  -- hbr.1 : 44 ∈ []
  exact absurd hbr.1 (by simp)

/-- The correctly-structured pattern (cond defined BEFORE branch):
    `const r5 0; branch r5 [...] [...]` IS scope-valid. -/
example : KernelOp.scopeValidOps []
    [.const 5 (.i32 0),
     .branch 5 [.const 6 (.i32 1)] [.const 7 (.i32 2)]] := by
  refine ⟨?_, ?_, trivial⟩
  · -- const 5 0 uses no regs
    intro r hr; simp [KernelOp.usedRegs] at hr
  · -- branch 5: cond ∈ [5] (after extending by const 5), branches valid
    refine ⟨?_, ?_, ?_⟩
    · -- cond = 5 is in env [5]
      simp [KernelOp.extendEnv, KernelOp.definedReg]
    · -- thenOps = [const 6 1] is scope-valid
      refine ⟨?_, trivial⟩
      intro r hr; simp [KernelOp.usedRegs] at hr
    · -- elseOps = [const 7 2] is scope-valid
      refine ⟨?_, trivial⟩
      intro r hr; simp [KernelOp.usedRegs] at hr

/-- Cmp-then-branch is scope-valid when cmp's dst is the branch
    cond. This is the WELL-STRUCTURED pattern the WASM-route SHOULD
    emit instead of the buggy redirect-chain output:
    `cmp r10 r3 r4 eq u32; branch r10 [] []`. -/
example : KernelOp.scopeValidOps [3, 4]
    [.cmp 10 3 4 .eq .u32, .branch 10 [] []] := by
  refine ⟨?_, ?_, trivial⟩
  · -- cmp uses r3, r4 — both in env
    intro r hr
    simp [KernelOp.usedRegs] at hr
    rcases hr with rfl | rfl <;> simp
  · -- branch's cond=r10 is in env [10, 3, 4] (after cmp extended env)
    refine ⟨?_, trivial, trivial⟩
    simp [KernelOp.extendEnv, KernelOp.definedReg]

/-- More-realistic bug pattern (mirrors `tests/host_emitter::emit_loop`):
    an outer Branch whose cond is defined INSIDE its else-branch.
    Expected NOT scope-valid. -/
example : ¬ KernelOp.scopeValidOps []
    [.const 0 (.i32 8),
     -- Outer branch using r44 as cond, but r44 is only set inside the else.
     .branch 44 []
       [.const 43 (.i32 0),
        .cmp 44 0 43 .eq .u32]] := by
  intro h
  obtain ⟨_, hbr, _⟩ := h
  obtain ⟨hcond, _, _⟩ := hbr
  -- hcond : 44 ∈ extendEnv [] (const 0 (.i32 8)) = [0]
  simp [KernelOp.extendEnv, KernelOp.definedReg] at hcond

-- ════════════════════════════════════════════════════════════════════
-- Bug #1 witness: r44 forward-branch pattern from `tests/host_emitter::emit_loop`
--
-- The production WASM-route lowering emits, for unrolled `while`-loop
-- kernels with cross-frame `br_if depth=N>0`, the following structure
-- (see `emitter_codegen_bugs_2026-05-29.md` for the full root-cause analysis):
--
--   outer: [..., OuterBranch { cond=r44, then=[], else=[
--     InnerBranch { cond=_, then=[..., cmp r44 = ..., ...], else=[] }
--   ]}]
--
-- r44 is used as OuterBranch's cond but defined inside InnerBranch's
-- then. MSL and SPIR-V both reject this with use-before-def. The
-- correctness invariant the lowering should preserve is `scopeValid`.
-- Below: the bug rejected (first example), and the proposed function-
-- scope-cond fix accepted (second example).
-- ════════════════════════════════════════════════════════════════════

/-- The full production bug structure: const r3; OuterBranch with cond=r44,
    empty then, else containing a nested InnerBranch whose then defines
    r44 via cmp. Expected NOT scope-valid because r44 ∉ [r3] at the
    point OuterBranch is reached. -/
example : ¬ KernelOp.scopeValidOps []
    [.const 3 (.i32 8),
     -- OuterBranch using r44 as cond; r44 isn't in env yet.
     .branch 44 []
       [-- Nested InnerBranch (depth>0 br_if target). Its `then`
        -- contains the cmp that defines r44, but this fires AFTER
        -- OuterBranch's cond check.
        .branch 10
          [.const 43 (.i32 0),
           .cmp 44 3 43 .eq .u32]
          []]] := by
  intro h
  obtain ⟨_, hbr_outer, _⟩ := h
  obtain ⟨hcond, _, _⟩ := hbr_outer
  -- hcond : 44 ∈ extendEnv [] (.const 3 (.i32 8)) = [3]
  simp [KernelOp.extendEnv, KernelOp.definedReg] at hcond

/-- Counterfactual: the "function-scope cond reg" fix sketch.
    Pre-allocate r44 (and r10, the InnerBranch cond) at the outer
    scope BEFORE OuterBranch. Inside InnerBranch's then, r44 gets
    set via copy — no use-before-def at the IR level. -/
example : KernelOp.scopeValidOps []
    [.const 3 (.i32 8),
     -- Pre-allocate r44 = false and r10 at outer scope.
     .const 44 (.i32 0),
     .const 10 (.i32 1),
     .branch 44 []
       [.branch 10
          [-- InnerBranch's then writes r44 := 1, no use-before-def.
           .const 43 (.i32 1),
           .copy 44 43]
          []]] := by
  refine ⟨?_, ?_, ?_, ?_, trivial⟩
  · intro r hr; simp [KernelOp.usedRegs] at hr
  · intro r hr; simp [KernelOp.usedRegs] at hr
  · intro r hr; simp [KernelOp.usedRegs] at hr
  · -- OuterBranch: cond=44, env=[10, 44, 3] after three consts.
    refine ⟨?_, trivial, ?_, trivial⟩
    · -- 44 ∈ [10, 44, 3]
      simp [KernelOp.extendEnv, KernelOp.definedReg]
    · -- InnerBranch: cond=10, thenOps=[const 43; copy 44 43], elseOps=[].
      refine ⟨?_, ?_, trivial⟩
      · -- 10 ∈ [10, 44, 3] (env extended through 3 consts)
        simp [KernelOp.extendEnv, KernelOp.definedReg]
      · -- thenOps scope-valid against env=[10, 44, 3].
        refine ⟨?_, ?_, trivial⟩
        · -- const 43 1: uses []
          intro r hr; simp [KernelOp.usedRegs] at hr
        · -- copy 44 43: uses [43]. env after const 43 = [43, 10, 44, 3].
          intro r hr
          simp [KernelOp.usedRegs] at hr
          subst hr
          simp [KernelOp.extendEnv, KernelOp.definedReg]

/-- Use of scopeValidOps_append: split a longer chain. The chain
    `[const 0 1; const 1 2]` is scope-valid against env=[], so by
    append with the rest `[binOp 2 0 1 add u32]` scope-valid against
    [1; 0] (the extended env), the concatenation is scope-valid. -/
example :
    KernelOp.scopeValidOps []
      ([.const 0 (.i32 1), .const 1 (.i32 2)] ++
       [.binOp 2 0 1 .add .u32]) := by
  apply KernelOp.scopeValidOps_append
  · -- First half: two consts, no uses
    refine ⟨?_, ?_, trivial⟩
    · intro r hr; simp [KernelOp.usedRegs] at hr
    · intro r hr; simp [KernelOp.usedRegs] at hr
  · -- Second half: binOp at env extended by both consts.
    show KernelOp.scopeValidOps _ _
    refine ⟨?_, trivial⟩
    -- env is extendEnvOps [] [const 0 _, const 1 _] = [1; 0]
    intro r hr
    simp [KernelOp.usedRegs] at hr
    rcases hr with rfl | rfl <;>
      simp [KernelOp.extendEnvOps, KernelOp.extendEnv, KernelOp.definedReg]

/-- Round-trip: `scopeValidOps_append_inv` then `scopeValidOps_append`
    recovers the original chain. Demonstrates the destructor splits
    cleanly and the halves recompose. -/
example :
    KernelOp.scopeValidOps []
      ([.const 0 (.i32 1), .const 1 (.i32 2)] ++
       [.binOp 2 0 1 .add .u32]) := by
  -- Start from the concatenated witness (built by hand to mirror the
  -- earlier append smoke), split it, then put it back together via
  -- `_append`.
  have hcat : KernelOp.scopeValidOps []
      ([.const 0 (.i32 1), .const 1 (.i32 2)] ++
       [.binOp 2 0 1 .add .u32]) := by
    apply KernelOp.scopeValidOps_append
    · refine ⟨?_, ?_, trivial⟩
      · intro r hr; simp [KernelOp.usedRegs] at hr
      · intro r hr; simp [KernelOp.usedRegs] at hr
    · show KernelOp.scopeValidOps _ _
      refine ⟨?_, trivial⟩
      intro r hr
      simp [KernelOp.usedRegs] at hr
      rcases hr with rfl | rfl <;>
        simp [KernelOp.extendEnvOps, KernelOp.extendEnv, KernelOp.definedReg]
  obtain ⟨h1, h2⟩ :=
    KernelOp.scopeValidOps_append_inv [] _ _ hcat
  exact KernelOp.scopeValidOps_append [] _ _ h1 h2

end Quanta.KOps

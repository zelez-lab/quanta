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

end Quanta.KOps

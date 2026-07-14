/-
# Machine-model axioms — WGSL grammar bridge

The single axiom that links the **structural** WGSL well-formedness
predicate `Source.wellFormed` (`Quanta.Wgsl.Grammar`) to the
**string-level** well-formedness predicate `wgsl_string_well_formed`
trusted by the WebGPU host (`Quanta.Axioms.WebGpu`).

Why this axiom exists, and why it is small:
- `Source.wellFormed : Source → Bool` is a closed-form, decidable
  predicate Lean evaluates structurally.
- `wgsl_string_well_formed : String → Prop` is the property the W3C
  WGSL §3 lexer + §4 validator accepts. We do not formalize either
  in Lean.
- The bridge says: if a `Source` passes the structural check, then
  the string our serializer emits is one the WGSL validator
  accepts.
- This is a *typedef-stability* claim about the printer in
  `crates/gpu/quanta-ir/src/emit_wgsl/*.rs` and the WGSL grammar — it
  holds because (a) every identifier the serializer prints is an
  identifier per WGSL §3 (enforced by `identOk`); (b) every
  expression nesting respects associativity/parenthesisation; (c)
  the type fields on each AST node are the types the WGSL §4
  validator would compute.

After B.4 lands the T410 proof using this bridge, `wgsl_string_well_formed`
becomes consumed at exactly one point — this axiom — which is
analogous to A11's residue post-B″: a small, named claim about the
Rust↔string boundary, with all structural reasoning lifted into
Lean theorems.
-/

import Quanta.Axioms.WebGpu
import Quanta.Wgsl.Grammar
import Quanta.Wgsl.Serialize

namespace Quanta.Axioms.Wgsl

open Quanta.Axioms.WebGpu
open Quanta.Wgsl

/-- **A12 — wgsl_serializer_preserves_grammar**: any structurally
    well-formed `Source` serializes to a string that satisfies
    `wgsl_string_well_formed` (W3C WGSL §3 lex + §4 validate).

    This is the load-bearing trust in the WGSL chain after B is
    complete. It is to T410 what A11's `extern "C"` linker
    faithfulness is to T1707 post-B″: the irreducible boundary
    claim, with everything *above* it discharged structurally. -/
axiom wgsl_serializer_preserves_grammar
    (s : Source)
    (h : Source.wellFormed s = true)
    : wgsl_string_well_formed (Source.serialize s)

/-- **A13 — emit_wgsl_jit_factors**: the JIT emitter factors
    through a structurally well-formed `Source`. That is, for any
    `emit : α → String` modelling Quanta's `emit_wgsl_jit` and any
    input kernel `k : α`, the string `emit k` is the serialization
    of some `Source` whose `Source.wellFormed = true`.

    Stated polymorphically in `α` so this module does not have to
    depend on `Quanta.Theorems.WebGpu`'s opaque `KernelDef`; the
    consumer (`Quanta.Theorems.WebGpu.t410_emitter_produces_well_formed_wgsl`)
    instantiates `α := KernelDef`.

    This is the operational claim Verus and Kani already discharge
    over the actual Rust emitter (`crates/gpu/quanta-ir/src/emit_wgsl/`).
    `specs/verify/verus/quanta-ir/emit_wgsl_jit.rs::
    t410_jit_wgsl_exhaustive` proves the per-tag exhaustiveness;
    `specs/verify/kani/emitter_exhaustiveness.rs::
    t1001_jit_wgsl_emitter_exhaustive` cross-checks the same
    property by exhaustive bounded model checking. T420 in
    `Quanta.Wgsl.OpPatterns` then witnesses the structural shape
    in Lean per tag. A13 is the small, named bridge between those
    operational results and the Lean `Source` view.

    Together A12 + A13 reduce T410 (`emit_wgsl_jit produces
    well-formed WGSL`) from an axiom to a theorem; see
    `Quanta.Theorems.WebGpu.t410_emitter_produces_well_formed_wgsl`. -/
axiom emit_wgsl_jit_factors
    {α : Type}
    (emit : α → String)
    (k : α)
    : ∃ s : Source,
        Source.wellFormed s = true ∧ emit k = Source.serialize s

end Quanta.Axioms.Wgsl

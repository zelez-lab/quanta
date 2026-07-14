//! V4 — the wasmparser input-edge boundary axiom.
//!
//! The production translator does not start from `WasmInstr` (the clean
//! ADT the V3/V5/V7 refinements reason about). It starts from raw
//! `wasmparser::Operator<'_>` values decoded from the kernel's
//! `wasm32` bytes, and converts each to an owned `RawInstr` via
//! `RawInstr::from_operator` (`crates/gpu/quanta-wasm-lowering/src/lib.rs`,
//! the big `match` at ~line 463). The refinement chain V1–V7 begins one
//! step *inside* that edge — it takes the already-decoded instruction
//! ADT as its input.
//!
//! That decode step (`wasmparser::Operator → RawInstr`, and rustc's
//! Rust→wasm32 compilation that produced the bytes) is **outside** the
//! mechanized boundary, the same status the other Verus crates here
//! give their external interfaces (e.g. `quanta-api`'s
//! `raw_hazard_free` driver obligation, discharged `external_body`).
//! This file names that boundary explicitly rather than leaving it
//! implicit in prose, so the TCB is auditable in one place.
//!
//! ## What is trusted, precisely
//!
//! `parse_op` is the spec image of `from_operator`: a *total,
//! deterministic* function `RawOperator → WasmInstr`. Two properties
//! are all the downstream refinement needs from the edge, and both are
//! axiomatized (`external_body`, discharged by the production `match`):
//!
//!   1. **Totality** — every `RawOperator` maps to *some* `WasmInstr`
//!      (recognized ops to their ADT variant; everything else to the
//!      refusing `Other`, mirroring production's `Unsupported(name)`
//!      catch-all that surfaces a clean lowering error). The decode
//!      never panics or diverges.
//!   2. **Determinism** — equal operators decode to equal instructions
//!      (`from_operator` is a pure `match` on the operator's shape; no
//!      hidden state). This is what lets the downstream fold treat the
//!      decoded list as a fixed `Seq<WasmInstr>`.
//!
//! Faithfulness of `parse_op`'s arm-by-arm mapping to the actual
//! `from_operator` match is the manual obligation — identical in kind
//! to the `step_<op> ≈ Rust arm` transcription boundary the V5 per-op
//! refinements carry (README "The model↔Rust correspondence"). The
//! differential test suite exercises the real decode end-to-end.

use vstd::prelude::*;

verus! {

// ── The instruction ADT (the refinement chain's input; V3/V7) ──────

pub enum WasmInstr {
    I32Const(int),
    I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor, I32Shl, I32ShrU, I32DivU, I32RemU,
    I32Eq, I32Ne, I32LtU, I32LeU, I32GtU, I32GeU,
    Drop, Nop, WReturn,
    /// Stand-in for every operator outside the slice-1 subset — the
    /// image of production's `RawInstr::Unsupported(name)` catch-all.
    Other,
}

// ── The raw operator (opaque image of `wasmparser::Operator`) ──────
//
// We do NOT model `wasmparser::Operator`'s internals — that type lives
// in an external crate and its decoding from bytes is rustc/wasmparser
// territory. We model it as an opaque spec type so the boundary
// function has a well-typed domain; the only things asserted about it
// are totality and determinism of the decode, below.

pub struct RawOperator { pub dummy: int }

// ── parse_op: the trusted decode (image of `from_operator`) ────────
//
// `closed spec fn` — its body is hidden, exactly because it is the
// external `match` we are *not* re-deriving. We assert its
// total/deterministic shape via the boundary lemmas.

pub uninterp spec fn parse_op(op: RawOperator) -> WasmInstr;

/// Whether the production `from_operator` `match` has a dedicated
/// (non-catch-all) arm for this operator — the spec image of "is this
/// one of the recognized opcodes, vs. the `_ => Unsupported(name)`
/// fallback". Opaque, decided by the external decode.
pub uninterp spec fn recognized(op: RawOperator) -> bool;

/// **Boundary axiom — clean refusal of the unrecognized.** Every
/// operator without a dedicated arm decodes to `Other` — the spec
/// image of production's `_ => RawInstr::Unsupported(name)` catch-all,
/// which surfaces a clean lowering error rather than a panic or a
/// misclassification. This is the substantive totality content: the
/// decode is defined on *all* operators, and the ones outside the
/// modeled subset land on the refusing variant (so the V7 fold refuses
/// them via `lower_instr`'s `Other => None`, never silently mislowers).
#[verifier::external_body]
pub proof fn parse_op_unrecognized_refuses(op: RawOperator)
    requires !recognized(op),
    ensures parse_op(op) == WasmInstr::Other,
{}

/// **Boundary axiom — determinism.** Equal operators decode equally.
/// `from_operator` is a pure `match` on the operator's observable
/// shape with no hidden state, so the same operator always produces the
/// same `RawInstr`/`WasmInstr`. This is what lets the V7 fold treat the
/// decoded instruction list as a fixed `Seq<WasmInstr>` — the input to
/// `lower_instructions` is well-defined.
#[verifier::external_body]
pub proof fn parse_op_deterministic(a: RawOperator, b: RawOperator)
    requires a == b,
    ensures parse_op(a) == parse_op(b),
{}

// ── parse_ops: decode a whole operator stream to the ADT list ──────
//
// The production `lower_function` decodes the body's operators in
// order into a `Vec<RawInstr>` before lowering. Its spec image is a
// pointwise `map` of `parse_op` over the operator `Seq` — this is the
// `Seq<WasmInstr>` the V7 `lower_instructions` / `spec_lower_instrs`
// fold consumes. Defined (not axiomatized): the *list plumbing* is
// ordinary, only the per-element decode is trusted.

pub open spec fn parse_ops(ops: Seq<RawOperator>) -> Seq<WasmInstr>
    decreases ops.len()
{
    if ops.len() == 0 {
        Seq::empty()
    } else {
        seq![parse_op(ops[0])].add(parse_ops(ops.subrange(1, ops.len() as int)))
    }
}

// ── Boundary properties the fold relies on (proved from the above) ──

/// Decoding preserves length: the instruction list the fold consumes
/// has exactly one entry per source operator (no drops, no
/// duplication). Follows from the pointwise `parse_ops` definition —
/// no axiom needed.
proof fn parse_ops_len(ops: Seq<RawOperator>)
    ensures parse_ops(ops).len() == ops.len(),
    decreases ops.len()
{
    if ops.len() == 0 {
    } else {
        parse_ops_len(ops.subrange(1, ops.len() as int));
    }
}

/// Decoding is deterministic at the list level: equal operator streams
/// decode to equal instruction lists. Lifts `parse_op_deterministic`
/// pointwise — the V7 fold's input is a well-defined function of the
/// source bytes' decoded operators.
proof fn parse_ops_deterministic(a: Seq<RawOperator>, b: Seq<RawOperator>)
    requires a == b,
    ensures parse_ops(a) == parse_ops(b),
{
    // `a == b` makes the two `parse_ops` applications syntactically
    // identical; the determinism of each element decode is the
    // per-op axiom, and Seq equality is extensional.
    assert(parse_ops(a) == parse_ops(b));
}

/// The decode of a `cons` is the `cons` of the decodes — the algebraic
/// fact that lets the V7 induction (which recurses on
/// `is.subrange(1, len)`) line up with a decode that recurses the same
/// way. `parse_ops(op :: rest) == parse_op(op) :: parse_ops(rest)`.
proof fn parse_ops_cons(op: RawOperator, rest: Seq<RawOperator>)
    ensures parse_ops(seq![op].add(rest)) == seq![parse_op(op)].add(parse_ops(rest)),
{
    let combined = seq![op].add(rest);
    assert(combined.len() == rest.len() + 1);
    assert(combined[0] == op);
    assert(combined.subrange(1, combined.len() as int) =~= rest);
}

} // verus!

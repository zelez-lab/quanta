//! V1 — toolchain smoke test for the quanta-wasm-lowering Verus arm.
//!
//! Confirms `just verus` verifies a file in this crate directory
//! cleanly. The real spec types and refinement proofs live in the
//! sibling files (see README.md). This file carries one trivial
//! proof so the milestone has a checkable artifact.

use vstd::prelude::*;

verus! {

/// Sanity: a register index is just a `nat` in the spec world, the
/// same way the Lean spec models `Reg`. This `spec fn` is the first
/// fragment of the mirror vocabulary V2 builds out.
pub open spec fn reg_of(n: nat) -> nat { n }

proof fn reg_of_identity(n: nat)
    ensures reg_of(n) == n,
{}

/// Toolchain smoke test: a closed arithmetic fact, so a clean
/// `just verus` run proves the crate directory is wired correctly.
proof fn toolchain_ok(a: nat, b: nat)
    ensures a + b == b + a,
{}

} // verus!

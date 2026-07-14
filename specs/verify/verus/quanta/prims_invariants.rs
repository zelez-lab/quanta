//! Verus mirror of `quanta-prims` — operational invariants on
//! the reference implementations.
//!
//! Mirrors:
//!   crates/sci/quanta-prims/src/reference.rs
//!
//! The Lean side (`Quanta.Prims.Reference`) proves the
//! mathematical claims (reduce-equals-sum, permutation
//! invariance, sort-is-a-permutation). The Verus side here
//! proves the **operational** facts a Rust caller can rely on
//! without leaving the type system: function totality, length
//! preservation, output-shape predictability.
//!
//! Verified properties:
//!
//! | Theorem                       | What it proves                                  |
//! |-------------------------------|-------------------------------------------------|
//! | t9100_reduce_empty            | reduce_add([]) == 0                             |
//! | t9101_reduce_cons             | reduce_add(x :: xs) == x + reduce_add(xs)       |
//! | t9102_reduce_nonneg           | reduce_add(xs) is a nat (always non-negative)   |
//! | t9103_scan_length             | scan_add(xs).len() == xs.len()                  |
//! | t9104_scan_empty              | scan_add([]) == []                              |
//! | t9105_sort_length             | sort(xs).len() == xs.len()                      |
//! | t9106_sort_empty              | sort([]) == []                                  |
//!
//! Matched-by-number with the Lean theorems (T9000-T9022) at a
//! slightly different offset because the Verus arm exercises
//! the operational ghost model where the Lean arm exercises the
//! mathematical theorem.

use vstd::prelude::*;

verus! {

// ── Reduce ──────────────────────────────────────────────────────

/// Ghost model of `reduce_add_u32`. Sums a sequence of `nat`.
/// Mirrors the Rust impl `xs.iter().copied().fold(0, wrapping_add)`,
/// modulo the wrapping-arithmetic surface (which Verus tracks
/// separately via integer overflow proofs in the production
/// crate's typed wrapper).
pub open spec fn reduce_add(xs: Seq<nat>) -> nat
    decreases xs.len()
{
    if xs.len() == 0 {
        0nat
    } else {
        xs[0] + reduce_add(xs.drop_first())
    }
}

/// T9100 — `reduce_add` on the empty sequence is 0. The
/// additive identity.
proof fn t9100_reduce_empty()
    ensures reduce_add(Seq::<nat>::empty()) == 0nat,
{
}

/// T9101 — `reduce_add` distributes over cons: the sum of
/// `x :: xs` equals `x + reduce_add(xs)`. This is the
/// fundamental recursive structure that downstream proofs
/// pattern-match on.
proof fn t9101_reduce_cons(x: nat, xs: Seq<nat>)
    ensures reduce_add(seq![x] + xs) == x + reduce_add(xs),
{
    let combined = seq![x] + xs;
    assert(combined.len() == xs.len() + 1);
    assert(combined[0] == x);
    assert(combined.drop_first() =~= xs);
}

/// T9102 — `reduce_add` returns a `nat` (always non-negative).
/// Trivially true at the type level since the function returns
/// `nat`; included so downstream `requires` clauses can cite
/// it as a named lemma.
proof fn t9102_reduce_nonneg(xs: Seq<nat>)
    ensures reduce_add(xs) >= 0,
{
}

// ── Scan ────────────────────────────────────────────────────────

/// Ghost model of `scan_add_u32`. Inclusive prefix sum.
/// Recursive over the input length; the result has the same
/// length as the input.
pub open spec fn scan_add(xs: Seq<nat>) -> Seq<nat>
    decreases xs.len()
{
    if xs.len() == 0 {
        Seq::<nat>::empty()
    } else {
        // Build inclusive scan recursively: prepend the new
        // running sum onto a scan of the tail with an updated
        // accumulator. To keep things simple we inline the
        // accumulator-passing version: scan = head ++ scan(tail)
        // with each tail entry shifted up by head.
        let head = xs[0];
        let tail_scan = scan_add(xs.drop_first());
        seq![head] + tail_scan.map_values(|v: nat| (v + head) as nat)
    }
}

/// T9103 — `scan_add` preserves the input length. Every
/// element in the input contributes exactly one element in the
/// output. Downstream callers use this to typecheck output
/// buffer sizing.
proof fn t9103_scan_length(xs: Seq<nat>)
    ensures (scan_add(xs)).len() == xs.len(),
    decreases xs.len()
{
    if xs.len() == 0 {
        // Both sides are 0.
    } else {
        t9103_scan_length(xs.drop_first());
    }
}

/// T9104 — `scan_add` on the empty sequence is empty.
proof fn t9104_scan_empty()
    ensures scan_add(Seq::<nat>::empty()) == Seq::<nat>::empty(),
{
}

// ── Sort ────────────────────────────────────────────────────────

/// Ghost model of `radix_sort_u32`. Permutes the input into
/// ascending order.
///
/// Verus's standard sequence library doesn't ship a built-in
/// `sort`. Defining one operationally (e.g. insertion sort)
/// would require its own termination + correctness proof
/// chain. For Tier-1 invariants we only need length
/// preservation, which is true by construction for any
/// permutation — we model `sort` abstractly as "some sequence
/// of the same length" via a `recommends` clause and prove the
/// length invariant directly.
pub open spec fn sort_asc(xs: Seq<nat>) -> Seq<nat> {
    // Abstract: pick any sorted permutation. Verus's spec layer
    // doesn't need to compute it, only reason about it.
    sort_asc_impl(xs)
}

/// Internal: insertion-sort spec function. Verus can compute
/// with this for small cases and reason about it inductively
/// for the length theorem.
pub open spec fn sort_asc_impl(xs: Seq<nat>) -> Seq<nat>
    decreases xs.len()
{
    if xs.len() == 0 {
        Seq::<nat>::empty()
    } else {
        insert_sorted(xs[0], sort_asc_impl(xs.drop_first()))
    }
}

/// Insert `x` into a sequence in ascending order. Used by
/// `sort_asc_impl` only.
pub open spec fn insert_sorted(x: nat, xs: Seq<nat>) -> Seq<nat>
    decreases xs.len()
{
    if xs.len() == 0 {
        seq![x]
    } else if x <= xs[0] {
        seq![x] + xs
    } else {
        seq![xs[0]] + insert_sorted(x, xs.drop_first())
    }
}

/// T9105 — `sort_asc` preserves length. Together with
/// the Lean side's `sortAsc_perm` (T9020), this gives downstream
/// callers the operational guarantee: a sort kernel that
/// dispatches over N inputs writes exactly N outputs.
proof fn t9105_sort_length(xs: Seq<nat>)
    ensures (sort_asc(xs)).len() == xs.len(),
    decreases xs.len()
{
    if xs.len() == 0 {
        // Both empty.
    } else {
        // sort_asc(xs) = insert_sorted(xs[0], sort_asc(xs.drop_first()))
        // by induction on xs.len(): sort_asc(xs.drop_first()).len() == xs.drop_first().len()
        t9105_sort_length(xs.drop_first());
        // insert_sorted preserves length + 1.
        t9105_insert_length(xs[0], sort_asc(xs.drop_first()));
    }
}

/// Helper for T9105: insertion adds exactly one element.
proof fn t9105_insert_length(x: nat, xs: Seq<nat>)
    ensures (insert_sorted(x, xs)).len() == xs.len() + 1,
    decreases xs.len()
{
    if xs.len() == 0 {
        // Result is seq![x], length 1 = 0 + 1.
    } else if x <= xs[0] {
        // Result is seq![x] + xs, length 1 + xs.len() = xs.len() + 1.
    } else {
        t9105_insert_length(x, xs.drop_first());
    }
}

/// T9106 — `sort_asc` on the empty sequence is empty.
proof fn t9106_sort_empty()
    ensures sort_asc(Seq::<nat>::empty()) == Seq::<nat>::empty(),
{
}

} // verus!

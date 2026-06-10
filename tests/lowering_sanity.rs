//! Lowering surgical-sanity tests for the redirect-chain redesign.
//!
//! Diagnostic, not aspirational: these tests pin down exactly which
//! shapes the OLD lowering handles and which it breaks on, so the
//! position-aware redirect redesign (Option C, see memory
//! `redirect-chain-substrate-redesign`) can be built with a clear
//! regression target.
//!
//! Expected status on current `main` (`8ace628`):
//!
//! | # | shape                                          | result    | where |
//! |---|------------------------------------------------|-----------|-------|
//! | 1 | while + early `break` over shared mutable      | PASS      | here  |
//! | 2 | nested if/else over shared mutable, device fn  | FAIL@build| `lowering_sanity_bug.rs` (cfg-gated) |
//! | 3 | block_compact cross-warp scan                  | FAIL @ run| `crates/quanta-prims/tests/block_compact.rs` (Metal) |
//!
//! Run:
//!   cargo test -p quanta --features software --test lowering_sanity   (test 1)
//!   cd crates/quanta-prims && cargo test --features gpu-metal --test block_compact   (test 3)
//!
//! Test 2 is build-gated because it errors at proc-macro expansion;
//! see the top of `lowering_sanity_bug.rs` for the env-var to enable.

#![cfg(feature = "software")]

// ===========================================================================
// Sanity 1 — loop + early break + shared mutable
// ===========================================================================
//
// The bug-#1 shape: a `while` with a shared mutable written on the
// `break` arm. Closed by `hoist_cond_defining_ops` + the 2026-06-03
// Block-merge fix. PASSES on current main — regression net for those
// two fixes.

#[quanta::kernel(workgroup = [1])]
fn sanity1_loop_early_break(out: &mut [u32]) {
    let mut result: u32 = 0u32;
    let mut iter: u32 = 0u32;
    while iter < 32u32 {
        if iter == 7u32 {
            result = 42u32;
            iter = 99u32; // exit on next check
        } else {
            iter = iter + 1u32;
        }
    }
    out[0] = result;
}

#[test]
fn sanity1_loop_early_break_runs() {
    let gpu = quanta::init_cpu();
    let out = gpu.field::<u32>(1).unwrap();
    out.write(&[0u32]).unwrap();

    let mut wave = sanity1_loop_early_break(&gpu).unwrap();
    wave.bind(0, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    assert_eq!(
        out.read().unwrap(),
        vec![42u32],
        "early-break write of 42 must reach the post-loop read"
    );
}

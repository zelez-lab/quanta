//! Compile-time sanity bench: layout compositions stay fast.
//!
//! `quanta-tensor` will eventually drive a proc-macro extension
//! that consumes `Layout` types in `#[quanta::kernel]` signatures.
//! That path is sensitive to host-side composition cost: a slow
//! `Layout::permute` or a hidden allocation inside the indexer
//! would balloon proc-macro expansion time. This test guards
//! against regressions by running a representative composition
//! many times and asserting the total stays well under a
//! generous ceiling.
//!
//! It is not a microbenchmark — Criterion / cargo-bench live
//! elsewhere. This is a smoke test that runs in `cargo test` so
//! every PR exercises the host-side path at least once.

use std::time::Instant;

use quanta_tensor::Layout;

/// One unit of "work": build a row-major layout, apply four
/// composable ops, and read back a representative offset. Mirrors
/// the shape of kernel-signature processing the future proc-macro
/// will do.
fn one_pass() -> usize {
    let base = Layout::row_major(&[2, 3, 4, 5]).unwrap();
    let view = base
        .transpose(1, 2)
        .unwrap()
        .permute(&[3, 0, 1, 2])
        .unwrap()
        .slice(0, 1, 4)
        .unwrap()
        .broadcast(&[3, 2, 4, 3])
        .unwrap();
    // Touch every axis of the result so the optimiser can't elide
    // the strides + base offset.
    let mut sum: isize = 0;
    for a in 0..view.shape().dims()[0] {
        for b in 0..view.shape().dims()[1] {
            for c in 0..view.shape().dims()[2] {
                for d in 0..view.shape().dims()[3] {
                    sum = sum.saturating_add(view.at(&[a, b, c, d]).unwrap() as isize);
                }
            }
        }
    }
    sum as usize
}

/// Run a thousand passes. On a 2026-era laptop this should take
/// single-digit milliseconds in release, tens of ms in debug. The
/// 500 ms ceiling is loose on purpose — its job is to catch
/// orders-of-magnitude regressions (a stray `Vec::new()` per `at`
/// call, an accidental O(n²) over the dim list), not to assert
/// any particular target.
#[test]
fn layout_composition_stays_fast() {
    const ITERATIONS: usize = 1_000;
    const CEILING_MS: u128 = 500;

    let start = Instant::now();
    let mut acc: usize = 0;
    for _ in 0..ITERATIONS {
        acc = acc.wrapping_add(one_pass());
    }
    let elapsed = start.elapsed();

    // Use `acc` so the optimiser doesn't elide the whole loop.
    std::hint::black_box(acc);

    let ms = elapsed.as_millis();
    assert!(
        ms < CEILING_MS,
        "{} iterations of one_pass() took {} ms; ceiling is {} ms — investigate for an accidental allocation or quadratic walk in Layout ops",
        ITERATIONS,
        ms,
        CEILING_MS
    );
}

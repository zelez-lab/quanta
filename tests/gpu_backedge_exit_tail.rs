//! End-to-end differential for the br_if-backedge + multi-level-br
//! exit-tail lowering shape (see
//! `crates/quanta-wasm-lowering/tests/lower_backedge_exit_tail.rs`
//! for the WAT-level pin).
//!
//! A loop-invariant `if unit … else … k / d` inside a `while` loop is
//! loop-UNSWITCHED by LLVM at opt-level=3 into two loop copies
//! cross-jumped through a multi-level `br` placed right after the
//! first copy's backedge `br_if`. Before the fix, that exit tail
//! lowered unconditionally: every thread ran ONE iteration and
//! silently produced wrong sums on every backend.
//!
//! The runtime step `s` (host passes 1) keeps the trip count
//! uncomputable so LLVM cannot unroll the loops away; the division by
//! scalar `d` keeps the arms structurally different so it cannot
//! if-convert. Both tricks mirror the quanta-blas kernel shapes.
//!
//! Run: cargo test --test gpu_backedge_exit_tail --features software
//!      cargo test --test gpu_backedge_exit_tail --features software,metal

#![cfg(feature = "software")]

/// while k < n { acc += (unit==1 ? k : k/d) ; k += s }
#[quanta::kernel(workgroup = [64])]
fn unswitch_sum(out: &mut [u32], n: u32, d: u32, s: u32, unit: u32) {
    let gid = quark_id();
    let mut acc = 0u32;
    let mut k = 0u32;
    while k < n {
        if unit == 1u32 {
            acc = acc + k;
        } else {
            acc = acc + k / d;
        }
        k = k + s;
    }
    out[gid as usize] = acc;
}

fn oracle(n: u32, d: u32, s: u32, unit: u32) -> u32 {
    let mut acc = 0u32;
    let mut k = 0u32;
    while k < n {
        acc += if unit == 1 { k } else { k / d };
        k += s;
    }
    acc
}

fn run_on(gpu: &quanta::Gpu, label: &str) {
    let total = 64usize;
    let n = 37u32;
    let d = 3u32;
    let s = 1u32;

    for unit in [1u32, 0u32] {
        let out = gpu.field::<u32>(total).unwrap();
        out.write(&vec![u32::MAX; total]).unwrap();

        let mut wave = unswitch_sum(gpu).unwrap();
        wave.bind(0, &out);
        wave.set_value(1, n);
        wave.set_value(2, d);
        wave.set_value(3, s);
        wave.set_value(4, unit);
        gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();

        let got = out.read().unwrap();
        let want = oracle(n, d, s, unit);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(
                *v, want,
                "[{label}] unit={unit}: thread {i} sum mismatch \
                 (one-iteration exit-tail bug?): got {v}, want {want}"
            );
        }
    }
}

#[test]
fn unswitch_sum_matches_oracle_on_cpu() {
    let gpu = quanta::init_cpu();
    run_on(&gpu, "cpu");
}

#[cfg(feature = "metal")]
#[test]
fn unswitch_sum_matches_oracle_on_metal() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    run_on(&gpu, "metal");
}

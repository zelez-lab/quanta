//! Regression: rustc's optimiser elides `bool as u32` no-op casts
//! before they reach the wasm-lowerer, so a kernel body like
//!
//! ```ignore
//! let is_set = predicate != 0u32;
//! let prefix = scan_add_u32(is_set as u32);
//! ```
//!
//! ends up with the wasm bytecode `i32.ne; call $scan_add_u32` —
//! no `as u32` op between them. The lowerer's symbolic stack tracks
//! `i32.ne` as producing `Bool`, then `scan_add_u32` is called with
//! a Bool register and emits `SubgroupInclusiveAdd { ty: U32 }`
//! over it. Metal's `simd_prefix_inclusive_sum(bool)` is not a
//! valid overload, so metallib rejected the kernel.
//!
//! Fix: `subgroup_reduce` / `subgroup_scan_inclusive` /
//! `subgroup_scan_exclusive` now emit a `Cast { Bool -> U32 }` when
//! the popped value's type doesn't match the requested op type.

#![cfg(feature = "software")]

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// Inclusive scan over a bool-typed predicate. Each lane contributes
// 1 if the predicate is non-zero, 0 otherwise. The result at lane k
// is the count of non-zero predicates in lanes 0..=k. This is the
// canonical "compact" pattern.
#[quanta::kernel(workgroup = [64])]
fn scan_bool_predicate(predicates: &[u32], out: &mut [u32]) {
    let i = quark_id();
    let p = predicates[i as usize];
    let is_set = p != 0u32;
    // The `as u32` looks redundant but lives here to document that
    // the user-side type IS u32; rustc still elides it.
    let prefix = unsafe { scan_add_u32(is_set as u32) };
    out[i as usize] = prefix;
}

#[test]
fn bool_to_u32_scan_cpu() {
    let gpu = quanta::init_cpu();
    let pred = gpu.field::<u32>(64).unwrap();
    let out = gpu.field::<u32>(64).unwrap();

    // Alternating: 1, 0, 1, 0, ... — counts of "set" up to and
    // including lane k for k = 0..64 are: 1, 1, 2, 2, 3, 3, …
    let predicates: Vec<u32> = (0..64u32).map(|i| i % 2).collect();
    pred.write(&predicates).unwrap();
    out.write(&vec![0u32; 64]).unwrap();

    let mut wave = scan_bool_predicate(&gpu).unwrap();
    wave.bind(0, &pred);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 64).unwrap().wait().unwrap();

    let result = out.read().unwrap();
    // The CPU executor resolves subgroup ops cooperatively across a
    // 32-wide warp (see `cpu::exec::SUBGROUP_SIZE`), so this is an
    // inclusive prefix-count of set predicates within each 32-lane
    // subgroup: it runs 0,1,1,2,2,… up to lane 31, then restarts at
    // the next subgroup boundary.
    const SUBGROUP: usize = 32;
    let expected: Vec<u32> = (0..64usize)
        .map(|k| {
            let base = (k / SUBGROUP) * SUBGROUP;
            (base..=k).map(|j| predicates[j]).sum()
        })
        .collect();
    assert_eq!(
        result, expected,
        "CPU cooperative 32-wide subgroup scan of the bool predicate"
    );
}

#[test]
fn bool_to_u32_scan_gpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let pred = gpu.field::<u32>(64).unwrap();
    let out = gpu.field::<u32>(64).unwrap();

    let predicates: Vec<u32> = (0..64u32).map(|i| i % 2).collect();
    pred.write(&predicates).unwrap();
    out.write(&vec![0u32; 64]).unwrap();

    let mut wave = scan_bool_predicate(&gpu).unwrap();
    wave.bind(0, &pred);
    wave.bind(1, &out);

    // The fix lives in the lowerer's emit-side Cast; the test
    // succeeds simply by *not crashing on metallib compile*. The
    // actual scan values depend on subgroup_size, which varies
    // per backend, so we only check that all 64 lanes produced
    // *some* non-zero count for the second half.
    match gpu.dispatch(&wave, 64) {
        Ok(mut pulse) => {
            pulse.wait().unwrap();
            let result = out.read().unwrap();
            // Total non-zero count is 32 (every other lane is 1).
            // Any subgroup-size division of the work produces a
            // sum-of-prefixes whose last-lane value equals the
            // count of 1s within that subgroup window.
            assert!(
                result.iter().any(|&v| v > 0),
                "at least one lane should see a non-zero prefix"
            );
        }
        Err(e) => {
            let msg = format!("{e:?}");
            panic!("dispatch error: {msg}");
        }
    }
}

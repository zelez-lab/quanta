//! Regression net for the wasm->KernelDef control-flow reconstruction —
//! the loop/branch/break lowering shapes that LLVM's optimizer produces
//! from `while`/`for` + `break` Rust source and that the structured
//! `KernelOp::Loop` model has to rebuild faithfully.
//!
//! Every case runs on the CPU software executor under a hard 30s watchdog:
//! a miscompiled loop-exit does not just return wrong data, it can spin
//! forever (the structured Loop auto-continues on body fall-through, so a
//! dropped exit `Break` degrades a bounded loop to its full sentinel
//! bound — the k1_bin double-sentinel shape hung the interpreter). The
//! watchdog turns that into a test FAILURE instead of a wedged run.
//!
//! Golden values are computed in-test by a host reference with identical
//! semantics; each kernel is compared bit-exactly.
//!
//! Shapes covered:
//!   1. plain while-loop sum with an indirect gather  (CF0 induction-ptr)
//!   2. guarded accumulate  (`if v < 1 { a += v }`)
//!   3. two accumulators split across a conditional branch
//!   4. doubly-nested loop with per-row pointer induction  (probe SP)
//!   5. the double-sentinel `while g<10000 { if geom break; … }` bin shape
//!      with loop-carried live locals  (probe shape C — the miscompile)
//!   6. a sentinel loop whose counter is ALSO read after the loop
//!      (induction-fusion bait: sentinel counter must not entangle the
//!       post-loop read)
//!
//! Run: cargo test --test lowering_loops --features software --no-default-features

#![cfg(feature = "software")]

use std::sync::mpsc;
use std::time::Duration;

/// Run `f` on a worker thread, failing the test if it does not finish
/// within 30s (a spun loop from a dropped exit-Break). `f` builds its own
/// `Gpu` so nothing non-Send crosses the thread boundary.
fn with_watchdog<T: Send + 'static>(label: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(Duration::from_secs(30)) {
        Ok(v) => {
            let _ = handle.join();
            v
        }
        Err(_) => panic!(
            "[{label}] dispatch did not finish within 30s — a lowered loop-exit \
             was dropped and the structured Loop is spinning to its sentinel bound"
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────
// 1. Plain while-loop sum with an indirect gather (the CF0 repro shape).
//    `p` walks an induction index; the loop-carried address must rebase
//    each iteration, not re-read the loop-entry base.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn gather_sum(src: &[u32], out: &mut [u32], n: u32, stride: u32) {
    let gid = quark_id();
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < n {
        acc = acc + src[i as usize];
        i = i + stride;
    }
    out[gid as usize] = acc;
}

fn gather_sum_ref(src: &[u32], n: u32, stride: u32) -> u32 {
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < n {
        acc += src[i as usize];
        i += stride;
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────
// 2. Guarded accumulate: only some iterations contribute.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn guarded_sum(src: &[u32], out: &mut [u32], n: u32, stride: u32) {
    let gid = quark_id();
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < n {
        let v = src[i as usize];
        if v < 1u32 {
            acc = acc + v;
        } else {
            acc = acc + 1u32;
        }
        i = i + stride;
    }
    out[gid as usize] = acc;
}

fn guarded_sum_ref(src: &[u32], n: u32, stride: u32) -> u32 {
    let mut acc = 0u32;
    let mut i = 0u32;
    while i < n {
        let v = src[i as usize];
        acc += if v < 1 { v } else { 1 };
        i += stride;
    }
    acc
}

// ─────────────────────────────────────────────────────────────────────
// 3. Two accumulators split across a conditional branch — both are
//    loop-carried and both are read after the loop.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn two_acc(src: &[u32], out: &mut [u32], n: u32, stride: u32) {
    let gid = quark_id();
    let mut lo = 0u32;
    let mut hi = 0u32;
    let mut i = 0u32;
    while i < n {
        let v = src[i as usize];
        if v < 8u32 {
            lo = lo + v;
        } else {
            hi = hi + v;
        }
        i = i + stride;
    }
    out[(gid * 2u32) as usize] = lo;
    out[(gid * 2u32 + 1u32) as usize] = hi;
}

fn two_acc_ref(src: &[u32], n: u32, stride: u32) -> (u32, u32) {
    let (mut lo, mut hi) = (0u32, 0u32);
    let mut i = 0u32;
    while i < n {
        let v = src[i as usize];
        if v < 8 {
            lo += v;
        } else {
            hi += v;
        }
        i += stride;
    }
    (lo, hi)
}

// ─────────────────────────────────────────────────────────────────────
// 4. Doubly-nested loop with per-row pointer induction (probe shape SP):
//    the source itself advances element indices across rows and columns
//    (row_ptr += tiles_x, col_ptr += stride), matching real induction-
//    variable rasterization.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn nested_ptr_deposit(seg: &[u32], counts: &mut [u32], n_seg: u32, stride: u32, tiles_x: u32) {
    let s = quark_id();
    if s < n_seg {
        let base = s * 4u32;
        let tx0 = seg[base as usize];
        let ty0 = seg[(base + 1u32) as usize];
        let tx1 = seg[(base + 2u32) as usize];
        let ty1 = seg[(base + 3u32) as usize];
        let mut row_ptr = ty0 * tiles_x + tx0;
        let ncols = tx1 - tx0 + 1u32;
        let mut ty = ty0;
        while ty <= ty1 {
            let mut ptr = row_ptr;
            let mut c = 0u32;
            while c < ncols {
                atomic_add(&mut counts[ptr as usize], 1u32);
                ptr = ptr + stride;
                c = c + stride;
            }
            row_ptr = row_ptr + tiles_x * stride;
            ty = ty + stride;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// 5. The double-sentinel bin shape (probe shape C): BOTH loops are
//    `while g < 10000 { if geom break; …; g += stride }` with extra
//    loop-carried live locals (total/row deposits, row_base). The `if
//    geom break` compiles to a br out of the loop's TAIL block, not a br
//    to the loop — the exact shape whose exit-Break the reconstruction
//    used to drop, degrading each loop to its 10000 sentinel bound.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn double_sentinel_bin(
    seg: &[u32],
    counts: &mut [u32],
    lists: &mut [u32],
    n_seg: u32,
    stride: u32,
) {
    let s = quark_id();
    if s < n_seg {
        let base = s * 4u32;
        let tx0 = seg[base as usize];
        let ty0 = seg[(base + 1u32) as usize];
        let tx1 = seg[(base + 2u32) as usize];
        let ty1 = seg[(base + 3u32) as usize];
        let seg_val = s + 1u32;
        let mut ty = ty0;
        let mut oguard = 0u32;
        let mut total_deposits = 0u32;
        while oguard < 10000u32 {
            if ty > ty1 {
                break;
            }
            let row_base = ty * 4u32;
            let mut tx = tx0;
            let mut iguard = 0u32;
            let mut row_deposits = 0u32;
            while iguard < 10000u32 {
                if tx > tx1 {
                    break;
                }
                let tile = row_base + tx;
                let claimed = atomic_add(&mut counts[tile as usize], 1u32);
                if claimed < 8u32 {
                    lists[(tile * 8u32 + claimed) as usize] = seg_val;
                }
                row_deposits = row_deposits + 1u32;
                tx = tx + stride;
                iguard = iguard + stride;
            }
            total_deposits = total_deposits + row_deposits;
            ty = ty + stride;
            oguard = oguard + stride;
        }
        // Keep total_deposits live to the end (never-taken store defeats DCE).
        if total_deposits == 0xFFFF_FFFFu32 {
            lists[0] = total_deposits;
        }
    }
}

// Host reference for the 4x4 / max_segs=8 bin shapes (#4 and #5).
const TILES_X_4: u32 = 4;
const MAX_SEGS_8: u32 = 8;

fn bin_counts_ref(segs: &[(u32, u32, u32, u32)], tiles_x: u32, tiles_y: u32) -> Vec<u32> {
    let mut counts = vec![0u32; (tiles_x * tiles_y) as usize];
    for &(tx0, ty0, tx1, ty1) in segs {
        let mut ty = ty0;
        while ty <= ty1 {
            let mut tx = tx0;
            while tx <= tx1 {
                counts[(ty * tiles_x + tx) as usize] += 1;
                tx += 1;
            }
            ty += 1;
        }
    }
    counts
}

// Order-independent per-tile deposit sets (deposits land in nondeterministic
// order under a parallel executor).
fn bin_sets_ref(segs: &[(u32, u32, u32, u32)], tiles_x: u32, tiles_y: u32) -> Vec<Vec<u32>> {
    let n = (tiles_x * tiles_y) as usize;
    let mut sets: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (sid, &(tx0, ty0, tx1, ty1)) in segs.iter().enumerate() {
        let seg_val = sid as u32 + 1;
        let mut ty = ty0;
        while ty <= ty1 {
            let mut tx = tx0;
            while tx <= tx1 {
                let tile = (ty * tiles_x + tx) as usize;
                if sets[tile].len() < MAX_SEGS_8 as usize {
                    sets[tile].push(seg_val);
                }
                tx += 1;
            }
            ty += 1;
        }
    }
    for s in &mut sets {
        s.sort_unstable();
    }
    sets
}

fn lists_to_sets(lists: &[u32]) -> Vec<Vec<u32>> {
    let ms = MAX_SEGS_8 as usize;
    (0..lists.len() / ms)
        .map(|t| {
            let mut v: Vec<u32> = lists[t * ms..(t + 1) * ms]
                .iter()
                .copied()
                .filter(|&x| x != 0)
                .collect();
            v.sort_unstable();
            v
        })
        .collect()
}

fn pack_segs(segs: &[(u32, u32, u32, u32)]) -> Vec<u32> {
    let mut v = Vec::with_capacity(segs.len() * 4);
    for &(tx0, ty0, tx1, ty1) in segs {
        v.extend_from_slice(&[tx0, ty0, tx1, ty1]);
    }
    v
}

// ─────────────────────────────────────────────────────────────────────
// 6. Sentinel loop whose counter is ALSO read after the loop. The
//    sentinel counter shares the affine sequence of the geometry index,
//    so LLVM may coalesce them into one induction variable — the post-
//    loop read of the counter must still observe the true iteration
//    count, not the geometry value.
// ─────────────────────────────────────────────────────────────────────
#[quanta::kernel]
fn sentinel_count_readback(out: &mut [u32], limit: u32, stride: u32) {
    let gid = quark_id();
    let mut i = 0u32;
    let mut guard = 0u32;
    let mut steps = 0u32;
    while guard < 10000u32 {
        if i >= limit {
            break;
        }
        steps = steps + 1u32;
        i = i + stride;
        guard = guard + stride;
    }
    // Read `steps` (the true trip count) after the loop.
    out[gid as usize] = steps;
}

fn sentinel_count_readback_ref(limit: u32, stride: u32) -> u32 {
    let mut i = 0u32;
    let mut guard = 0u32;
    let mut steps = 0u32;
    while guard < 10000 {
        if i >= limit {
            break;
        }
        steps += 1;
        i += stride;
        guard += stride;
    }
    steps
}

// ─────────────────────────────────────────────────────────────────────
// Tests. Each spawns the dispatch under the 30s watchdog.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn gather_sum_matches_host() {
    with_watchdog("gather_sum", || {
        let gpu = quanta::init_cpu();
        let src: Vec<u32> = (0..64u32).map(|i| i * 3 + 1).collect();
        let n = src.len() as u32;
        let total = 64usize;
        let src_f = gpu.field::<u32>(src.len()).unwrap();
        let out = gpu.field::<u32>(total).unwrap();
        src_f.write(&src).unwrap();
        out.write(&vec![u32::MAX; total]).unwrap();
        let mut wave = gather_sum(&gpu).unwrap();
        wave.bind(0, &src_f);
        wave.bind(1, &out);
        wave.set_value(2, n);
        wave.set_value(3, 1u32);
        gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();
        let got = out.read().unwrap();
        let want = gather_sum_ref(&src, n, 1);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, want, "gather_sum thread {i}: got {v} want {want}");
        }
    });
}

#[test]
fn guarded_sum_matches_host() {
    with_watchdog("guarded_sum", || {
        let gpu = quanta::init_cpu();
        // Mix of <1 (i.e. 0) and >=1 values so both arms execute.
        let src: Vec<u32> = (0..48u32).map(|i| i % 3).collect();
        let n = src.len() as u32;
        let total = 64usize;
        let src_f = gpu.field::<u32>(src.len()).unwrap();
        let out = gpu.field::<u32>(total).unwrap();
        src_f.write(&src).unwrap();
        out.write(&vec![u32::MAX; total]).unwrap();
        let mut wave = guarded_sum(&gpu).unwrap();
        wave.bind(0, &src_f);
        wave.bind(1, &out);
        wave.set_value(2, n);
        wave.set_value(3, 1u32);
        gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();
        let got = out.read().unwrap();
        let want = guarded_sum_ref(&src, n, 1);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(*v, want, "guarded_sum thread {i}: got {v} want {want}");
        }
    });
}

#[test]
fn two_acc_matches_host() {
    with_watchdog("two_acc", || {
        let gpu = quanta::init_cpu();
        let src: Vec<u32> = (0..40u32).map(|i| i % 16).collect();
        let n = src.len() as u32;
        let total = 32usize;
        let src_f = gpu.field::<u32>(src.len()).unwrap();
        let out = gpu.field::<u32>(total * 2).unwrap();
        src_f.write(&src).unwrap();
        out.write(&vec![u32::MAX; total * 2]).unwrap();
        let mut wave = two_acc(&gpu).unwrap();
        wave.bind(0, &src_f);
        wave.bind(1, &out);
        wave.set_value(2, n);
        wave.set_value(3, 1u32);
        gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();
        let got = out.read().unwrap();
        let (want_lo, want_hi) = two_acc_ref(&src, n, 1);
        for g in 0..total {
            assert_eq!(got[g * 2], want_lo, "two_acc thread {g} lo");
            assert_eq!(got[g * 2 + 1], want_hi, "two_acc thread {g} hi");
        }
    });
}

#[test]
fn nested_ptr_deposit_matches_host() {
    with_watchdog("nested_ptr_deposit", || {
        let gpu = quanta::init_cpu();
        let segs = [(0u32, 0u32, 3u32, 3u32), (1, 1, 3, 2), (0, 2, 1, 3)];
        let n_seg = segs.len() as u32;
        let packed = pack_segs(&segs);
        let n_tiles = (TILES_X_4 * 4) as usize;
        let seg_f = gpu.field::<u32>(packed.len()).unwrap();
        let counts = gpu.field::<u32>(n_tiles).unwrap();
        seg_f.write(&packed).unwrap();
        counts.write(&vec![0u32; n_tiles]).unwrap();
        let mut wave = nested_ptr_deposit(&gpu).unwrap();
        wave.bind(0, &seg_f);
        wave.bind(1, &counts);
        wave.set_value(2, n_seg);
        wave.set_value(3, 1u32);
        wave.set_value(4, TILES_X_4);
        let quarks = ((n_seg + 63) / 64) * 64;
        gpu.dispatch(&wave, quarks).unwrap().wait().unwrap();
        let got = counts.read().unwrap();
        let want = bin_counts_ref(&segs, TILES_X_4, 4);
        assert_eq!(got, want, "nested_ptr_deposit counts");
    });
}

#[test]
fn double_sentinel_bin_matches_host() {
    with_watchdog("double_sentinel_bin", || {
        let gpu = quanta::init_cpu();
        let segs = [(0u32, 0u32, 3u32, 3u32), (1, 1, 3, 2), (0, 2, 1, 3)];
        let n_seg = segs.len() as u32;
        let packed = pack_segs(&segs);
        let n_tiles = (TILES_X_4 * 4) as usize;
        let list_len = n_tiles * MAX_SEGS_8 as usize;
        let seg_f = gpu.field::<u32>(packed.len()).unwrap();
        let counts = gpu.field::<u32>(n_tiles).unwrap();
        let lists = gpu.field::<u32>(list_len).unwrap();
        seg_f.write(&packed).unwrap();
        counts.write(&vec![0u32; n_tiles]).unwrap();
        lists.write(&vec![0u32; list_len]).unwrap();
        let mut wave = double_sentinel_bin(&gpu).unwrap();
        wave.bind(0, &seg_f);
        wave.bind(1, &counts);
        wave.bind(2, &lists);
        wave.set_value(3, n_seg);
        wave.set_value(4, 1u32);
        let quarks = ((n_seg + 63) / 64) * 64;
        gpu.dispatch(&wave, quarks).unwrap().wait().unwrap();
        let got_counts = counts.read().unwrap();
        let got_lists = lists.read().unwrap();
        let want_counts = bin_counts_ref(&segs, TILES_X_4, 4);
        let want_sets = bin_sets_ref(&segs, TILES_X_4, 4);
        assert_eq!(got_counts, want_counts, "double_sentinel_bin counts");
        assert_eq!(
            lists_to_sets(&got_lists),
            want_sets,
            "double_sentinel_bin deposit sets"
        );
    });
}

#[test]
fn sentinel_count_readback_matches_host() {
    with_watchdog("sentinel_count_readback", || {
        let gpu = quanta::init_cpu();
        let total = 64usize;
        let limit = 37u32;
        let out = gpu.field::<u32>(total).unwrap();
        out.write(&vec![u32::MAX; total]).unwrap();
        let mut wave = sentinel_count_readback(&gpu).unwrap();
        wave.bind(0, &out);
        wave.set_value(1, limit);
        wave.set_value(2, 1u32);
        gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();
        let got = out.read().unwrap();
        let want = sentinel_count_readback_ref(limit, 1);
        for (i, v) in got.iter().enumerate() {
            assert_eq!(
                *v, want,
                "sentinel_count_readback thread {i}: got {v} want {want}"
            );
        }
    });
}

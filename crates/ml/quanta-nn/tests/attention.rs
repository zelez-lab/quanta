//! Fused scaled-dot-product attention — differential and property tests.
//!
//! Runs on whichever backend the crate's features select: the CPU-JIT lane
//! (`--features software`, the default) or a real device (`--features metal` /
//! `--features vulkan`). Every numeric check is against a host f64 two-pass
//! reference — the textbook `softmax(scale·QKᵀ + mask)·V` computed with a
//! separate max pass and sum pass — so the fused online-softmax kernel is
//! validated end to end, not against itself.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::functional::{Sdpa, scaled_dot_product_attention, sdpa_var};

fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device (metal/vulkan feature is on)")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic pseudo-random fill in `[-1, 1)` — a cheap splitmix so the two
/// lanes and the host reference all see the same inputs without a rng dep.
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Host two-pass reference. Returns `(out[seq_q*dv], m[seq_q], l[seq_q])`, all
/// computed in f64 with an explicit max shift (matches the T9207 stable form).
/// `mask(qi, kj) == false` means key `kj` is masked out for query `qi`.
#[allow(clippy::too_many_arguments)]
fn host_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    dv: usize,
    scale: f64,
    mask: impl Fn(usize, usize) -> bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut out = vec![0f64; seq_q * dv];
    let mut mrow = vec![0f64; seq_q];
    let mut lrow = vec![0f64; seq_q];
    for qi in 0..seq_q {
        // scores + running max over valid keys
        let mut scores = vec![f64::NEG_INFINITY; seq_k];
        let mut m = f64::NEG_INFINITY;
        for kj in 0..seq_k {
            if !mask(qi, kj) {
                continue;
            }
            let mut dot = 0f64;
            for p in 0..d {
                dot += q[qi * d + p] as f64 * k[kj * d + p] as f64;
            }
            let s = scale * dot;
            scores[kj] = s;
            if s > m {
                m = s;
            }
        }
        // normaliser + weighted value sum, both shifted by m
        let mut l = 0f64;
        let mut acc = vec![0f64; dv];
        for kj in 0..seq_k {
            if scores[kj] == f64::NEG_INFINITY {
                continue;
            }
            let w = (scores[kj] - m).exp();
            l += w;
            for vd in 0..dv {
                acc[vd] += w * v[kj * dv + vd] as f64;
            }
        }
        for vd in 0..dv {
            out[qi * dv + vd] = acc[vd] / l;
        }
        mrow[qi] = m;
        lrow[qi] = l;
    }
    (out, mrow, lrow)
}

/// Max relative error between a fused (`f32`) result and the f64 reference.
fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
    got.iter()
        .zip(want)
        .map(|(&g, &w)| ((g as f64 - w).abs()) / (1.0 + w.abs()))
        .fold(0.0, f64::max)
}

fn run_case(seq_q: usize, seq_k: usize, d: usize, dv: usize, causal: bool, kv_len: Option<usize>) {
    let g = gpu();
    let q = fill(seq_q * d, 1);
    let k = fill(seq_k * d, 2);
    let v = fill(seq_k * dv, 3);
    let scale = 1.0 / (d as f64).sqrt();
    let eff = kv_len.unwrap_or(seq_k).clamp(1, seq_k);

    let qa = Array::from_slice(&g, &q, &[seq_q, d]).unwrap();
    let ka = Array::from_slice(&g, &k, &[seq_k, d]).unwrap();
    let va = Array::from_slice(&g, &v, &[seq_k, dv]).unwrap();
    let opts = Sdpa {
        scale: None,
        causal,
        kv_len,
    };
    let res = scaled_dot_product_attention(&g, &qa, &ka, &va, opts).unwrap();
    let out = res.output.to_vec().unwrap();
    let stats = res.stats.to_vec().unwrap();

    let mask = |qi: usize, kj: usize| (!causal || kj <= qi) && kj < eff;
    let (want, mref, lref) = host_reference(&q, &k, &v, seq_q, seq_k, d, dv, scale, mask);

    let rel = max_rel_err(&out, &want);
    assert!(
        rel < 1e-5,
        "out rel-err {rel:.3e} too high (sq={seq_q} sk={seq_k} d={d} dv={dv} causal={causal} kv={kv_len:?})"
    );

    // Stats buffer: (m, l) per row match the host max / sum-exp.
    for qi in 0..seq_q {
        let m = stats[2 * qi] as f64;
        let l = stats[2 * qi + 1] as f64;
        assert!(
            (m - mref[qi]).abs() < 1e-4 * (1.0 + mref[qi].abs()),
            "stats m[{qi}] = {m} vs ref {}",
            mref[qi]
        );
        assert!(
            (l - lref[qi]).abs() < 1e-4 * (1.0 + lref[qi].abs()),
            "stats l[{qi}] = {l} vs ref {}",
            lref[qi]
        );
    }
}

// ── 1. Differential vs host two-pass reference, many shapes ──────────────────

#[test]
fn diff_square() {
    run_case(4, 4, 8, 8, false, None);
}

#[test]
fn diff_rect_uneven() {
    // seq lengths that don't divide the workgroup (256) evenly, d != dv.
    run_case(5, 7, 6, 10, false, None);
    run_case(13, 3, 11, 4, false, None);
    run_case(1, 9, 8, 8, false, None);
}

#[test]
fn diff_causal() {
    run_case(6, 6, 8, 8, true, None);
    run_case(7, 5, 4, 12, true, None); // seq_q > seq_k causal
}

#[test]
fn diff_padding() {
    run_case(4, 8, 8, 8, false, Some(3));
    run_case(6, 10, 5, 7, false, Some(6));
}

#[test]
fn diff_causal_and_padding() {
    run_case(6, 8, 8, 8, true, Some(5));
}

#[test]
fn diff_larger() {
    run_case(32, 40, 16, 16, false, None);
    run_case(40, 40, 16, 16, true, None);
}

// ── 2. Cross-check vs quanta-autograd's composed attention forward ───────────

#[test]
fn cross_check_autograd_forward() {
    // The single-head SDPA composed from autograd primitives (via sdpa_var's
    // forward path) must match the fused kernel to fp tolerance. This ties the
    // fused output to the same softmax·V the tape backprops.
    let g = gpu();
    let (seq_q, seq_k, d, dv) = (5, 6, 8, 8);
    let q = fill(seq_q * d, 11);
    let k = fill(seq_k * d, 12);
    let v = fill(seq_k * dv, 13);

    let qa = Array::from_slice(&g, &q, &[seq_q, d]).unwrap();
    let ka = Array::from_slice(&g, &k, &[seq_k, d]).unwrap();
    let va = Array::from_slice(&g, &v, &[seq_k, dv]).unwrap();

    let fused = scaled_dot_product_attention(&g, &qa, &ka, &va, Sdpa::default())
        .unwrap()
        .output
        .to_vec()
        .unwrap();

    let tape = Tape::<f32>::new();
    let qv = tape.var(qa.shallow_clone());
    let kv = tape.var(ka.shallow_clone());
    let vv = tape.var(va.shallow_clone());
    let composed = sdpa_var(&tape, &qv, &kv, &vv, Sdpa::default())
        .unwrap()
        .value()
        .to_vec()
        .unwrap();

    let rel = fused
        .iter()
        .zip(&composed)
        .map(|(&a, &b)| ((a - b).abs() as f64) / (1.0 + b.abs() as f64))
        .fold(0.0, f64::max);
    assert!(rel < 1e-4, "fused vs composed rel-err {rel:.3e}");
}

// ── 3. Stability: logits scaled to ±80, outputs finite & correct ─────────────

#[test]
fn stability_extreme_logits() {
    let g = gpu();
    let (seq_q, seq_k, d, dv) = (4, 6, 4, 4);
    // Build Q, K so raw scaled scores land around ±80: with scale = 1/√d and
    // moderate d, use a large scale override to push logits to the edge.
    let q = fill(seq_q * d, 21);
    let k = fill(seq_k * d, 22);
    let v = fill(seq_k * dv, 23);
    // A big explicit scale drives the logit magnitude up. dot ∈ ~[-d, d]; pick
    // scale so scale·dot reaches ~±80.
    let scale = 80.0f32; // dot is O(1)·d ⇒ scaled scores span tens–hundreds

    let qa = Array::from_slice(&g, &q, &[seq_q, d]).unwrap();
    let ka = Array::from_slice(&g, &k, &[seq_k, d]).unwrap();
    let va = Array::from_slice(&g, &v, &[seq_k, dv]).unwrap();
    let opts = Sdpa {
        scale: Some(scale),
        causal: false,
        kv_len: None,
    };
    let out = scaled_dot_product_attention(&g, &qa, &ka, &va, opts)
        .unwrap()
        .output
        .to_vec()
        .unwrap();

    for (i, &o) in out.iter().enumerate() {
        assert!(o.is_finite(), "out[{i}] = {o} not finite under ±80 logits");
    }

    // Reference computed with the same shift (T9207 property made empirical).
    let mask = |_qi: usize, _kj: usize| true;
    let (want, _m, _l) = host_reference(&q, &k, &v, seq_q, seq_k, d, dv, scale as f64, mask);
    let rel = max_rel_err(&out, &want);
    assert!(rel < 1e-4, "extreme-logit rel-err {rel:.3e}");
}

// ── 5. Gradcheck through sdpa_var (finite difference) ────────────────────────

#[test]
fn gradcheck_sdpa_var() {
    // Central-difference the scalar loss L = sum(sdpa(Q,K,V)) wrt Q, comparing
    // to the tape's analytic gradient. Tiny dims keep the O(n·forward) FD cost
    // down; agreement means the composed VJP backprops the fused attention.
    let g = gpu();
    let (seq_q, seq_k, d, dv) = (2, 3, 3, 2);
    let q = fill(seq_q * d, 31);
    let k = fill(seq_k * d, 32);
    let v = fill(seq_k * dv, 33);

    let ka = Array::from_slice(&g, &k, &[seq_k, d]).unwrap();
    let va = Array::from_slice(&g, &v, &[seq_k, dv]).unwrap();

    let loss_of = |qh: &[f32]| -> f32 {
        let tape = Tape::<f32>::new();
        let qv = tape.var(Array::from_slice(&g, qh, &[seq_q, d]).unwrap());
        let kv = tape.var(ka.shallow_clone());
        let vv = tape.var(va.shallow_clone());
        let out = sdpa_var(&tape, &qv, &kv, &vv, Sdpa::default()).unwrap();
        out.sum().unwrap().value().to_vec().unwrap()[0]
    };

    // analytic dL/dQ
    let tape = Tape::<f32>::new();
    let qv = tape.var(Array::from_slice(&g, &q, &[seq_q, d]).unwrap());
    let kv = tape.var(ka.shallow_clone());
    let vv = tape.var(va.shallow_clone());
    let out = sdpa_var(&tape, &qv, &kv, &vv, Sdpa::default()).unwrap();
    let loss = out.sum().unwrap();
    let an = loss.grad(&qv).unwrap().to_vec().unwrap();

    // numeric dL/dQ
    let h = 1e-3f32;
    for j in 0..q.len() {
        let mut qp = q.clone();
        let mut qm = q.clone();
        qp[j] += h;
        qm[j] -= h;
        let num = (loss_of(&qp) - loss_of(&qm)) / (2.0 * h);
        assert!(
            (an[j] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "grad[{j}] analytic {} vs numeric {num}",
            an[j]
        );
    }
}

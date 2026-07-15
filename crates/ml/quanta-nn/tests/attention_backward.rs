//! Fused scaled-dot-product attention — **backward** differential & property
//! tests.
//!
//! The fused backward ([`sdpa_var`], which dispatches
//! `kernel::sdpa_backward`) is validated three ways on both lanes (software
//! `--features software`, real device `--features metal` / `--features
//! vulkan`):
//!   1. against the composed-VJP path ([`sdpa_var_composed`], the oracle) — the
//!      same forward, backprop via the materialising `quanta-autograd` graph;
//!   2. against a from-scratch host **f64 analytic** backward (the FlashAttn
//!      formulas computed in double precision) — the tight small-shape check;
//!   3. by central-difference gradcheck through the new `sdpa_var` backward, on
//!      Q, K and V.
//!
//! Plus a ±80-logit stability check that the fused gradients stay finite and
//! match the f64 reference.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::functional::{Sdpa, sdpa_var, sdpa_var_composed};

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

/// Deterministic pseudo-random fill in `[-1, 1)` — a cheap splitmix.
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

/// Max relative error between two flat slices, `|a−b| / (1 + |b|)`.
fn max_rel(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y).abs() as f64) / (1.0 + y.abs() as f64))
        .fold(0.0, f64::max)
}

/// Max relative error of an f32 result against an f64 reference.
fn max_rel_f64(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| ((x as f64 - y).abs()) / (1.0 + y.abs()))
        .fold(0.0, f64::max)
}

/// From-scratch host **f64 analytic** backward, the FlashAttention formulas in
/// double precision. `mask(qi, kj) == false` means key `kj` is masked out for
/// query `qi` (`p_ij = 0`). Upstream grad is `d_o` (`seq_q × dv`, here all
/// ones — matching a `sum()` loss). Returns `(dq[sq*d], dk[sk*d], dv[sk*dv])`.
#[allow(clippy::too_many_arguments)]
fn host_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_o: &[f64],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    dv: usize,
    scale: f64,
    mask: impl Fn(usize, usize) -> bool,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dq = vec![0f64; seq_q * d];
    let mut dk = vec![0f64; seq_k * d];
    let mut dvg = vec![0f64; seq_k * dv];

    for qi in 0..seq_q {
        // scores + row max over valid keys.
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
        // normaliser l, softmax weights p.
        let mut l = 0f64;
        let mut p = vec![0f64; seq_k];
        for kj in 0..seq_k {
            if scores[kj] == f64::NEG_INFINITY {
                continue;
            }
            let w = (scores[kj] - m).exp();
            p[kj] = w;
            l += w;
        }
        for w in p.iter_mut() {
            *w /= l;
        }

        // O_i = Σ_j p_ij V_j ;  D_i = Σ_c dO_ic O_ic.
        let mut o = vec![0f64; dv];
        for kj in 0..seq_k {
            for c in 0..dv {
                o[c] += p[kj] * v[kj * dv + c] as f64;
            }
        }
        let mut d_i = 0f64;
        for c in 0..dv {
            d_i += d_o[qi * dv + c] * o[c];
        }

        // dV_jc += p_ij dO_ic ;  dS_ij = p_ij (Σ_c dO_ic V_jc − D_i).
        for kj in 0..seq_k {
            if p[kj] == 0.0 {
                continue;
            }
            let mut dov = 0f64;
            for c in 0..dv {
                dov += d_o[qi * dv + c] * v[kj * dv + c] as f64;
                dvg[kj * dv + c] += p[kj] * d_o[qi * dv + c];
            }
            let ds = p[kj] * (dov - d_i);
            // dQ_id += scale dS_ij K_jd ; dK_jd += scale dS_ij Q_id.
            for dd in 0..d {
                dq[qi * d + dd] += scale * ds * k[kj * d + dd] as f64;
                dk[kj * d + dd] += scale * ds * q[qi * d + dd] as f64;
            }
        }
    }
    (dq, dk, dvg)
}

/// Analytic gradients of `L = sum(sdpa(Q,K,V))` through the **fused** path.
#[allow(clippy::too_many_arguments)]
fn fused_grads(
    g: &quanta::Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    dv: usize,
    opts: Sdpa,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let tape = Tape::<f32>::new();
    let qv = tape.var(Array::from_slice(g, q, &[seq_q, d]).unwrap());
    let kv = tape.var(Array::from_slice(g, k, &[seq_k, d]).unwrap());
    let vv = tape.var(Array::from_slice(g, v, &[seq_k, dv]).unwrap());
    let out = sdpa_var(&tape, &qv, &kv, &vv, opts).unwrap();
    let loss = out.sum().unwrap();
    (
        loss.grad(&qv).unwrap().to_vec().unwrap(),
        loss.grad(&kv).unwrap().to_vec().unwrap(),
        loss.grad(&vv).unwrap().to_vec().unwrap(),
    )
}

/// Same, through the **composed** oracle path.
#[allow(clippy::too_many_arguments)]
fn composed_grads(
    g: &quanta::Gpu,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_k: usize,
    d: usize,
    dv: usize,
    opts: Sdpa,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let tape = Tape::<f32>::new();
    let qv = tape.var(Array::from_slice(g, q, &[seq_q, d]).unwrap());
    let kv = tape.var(Array::from_slice(g, k, &[seq_k, d]).unwrap());
    let vv = tape.var(Array::from_slice(g, v, &[seq_k, dv]).unwrap());
    let out = sdpa_var_composed(&tape, &qv, &kv, &vv, opts).unwrap();
    let loss = out.sum().unwrap();
    (
        loss.grad(&qv).unwrap().to_vec().unwrap(),
        loss.grad(&kv).unwrap().to_vec().unwrap(),
        loss.grad(&vv).unwrap().to_vec().unwrap(),
    )
}

// ── 1. Fused backward vs composed-VJP oracle, many shapes/masks ──────────────

fn diff_vs_composed(seq_q: usize, seq_k: usize, d: usize, dv: usize, opts: Sdpa) {
    let g = gpu();
    let q = fill(seq_q * d, 101);
    let k = fill(seq_k * d, 202);
    let v = fill(seq_k * dv, 303);

    let (fq, fk, fv) = fused_grads(&g, &q, &k, &v, seq_q, seq_k, d, dv, opts);
    let (cq, ck, cv) = composed_grads(&g, &q, &k, &v, seq_q, seq_k, d, dv, opts);

    let rq = max_rel(&fq, &cq);
    let rk = max_rel(&fk, &ck);
    let rv = max_rel(&fv, &cv);
    if std::env::var("SDPA_PROBE").is_ok() {
        eprintln!(
            "[SDPA_PROBE composed] sq={seq_q} sk={seq_k} d={d} dv={dv} causal={} kv={:?} \
             dq={rq:.3e} dk={rk:.3e} dv={rv:.3e}",
            opts.causal, opts.kv_len
        );
    }
    assert!(
        rq < 1e-4 && rk < 1e-4 && rv < 1e-4,
        "fused vs composed rel-err dq={rq:.2e} dk={rk:.2e} dv={rv:.2e} \
         (sq={seq_q} sk={seq_k} d={d} dv={dv} causal={} kv={:?})",
        opts.causal,
        opts.kv_len
    );
}

#[test]
fn bwd_vs_composed_square() {
    diff_vs_composed(4, 4, 8, 8, Sdpa::default());
}

#[test]
fn bwd_vs_composed_rect() {
    diff_vs_composed(5, 7, 6, 10, Sdpa::default());
    diff_vs_composed(13, 3, 11, 4, Sdpa::default());
}

#[test]
fn bwd_vs_composed_causal() {
    diff_vs_composed(
        6,
        6,
        8,
        8,
        Sdpa {
            causal: true,
            ..Default::default()
        },
    );
    diff_vs_composed(
        7,
        5,
        4,
        12,
        Sdpa {
            causal: true,
            ..Default::default()
        },
    );
}

#[test]
fn bwd_vs_composed_padding() {
    diff_vs_composed(
        4,
        8,
        8,
        8,
        Sdpa {
            kv_len: Some(3),
            ..Default::default()
        },
    );
}

#[test]
fn bwd_vs_composed_causal_and_padding() {
    diff_vs_composed(
        6,
        8,
        8,
        8,
        Sdpa {
            causal: true,
            kv_len: Some(5),
            ..Default::default()
        },
    );
}

#[test]
fn bwd_vs_composed_larger() {
    diff_vs_composed(32, 40, 16, 16, Sdpa::default());
    diff_vs_composed(
        40,
        40,
        16,
        16,
        Sdpa {
            causal: true,
            ..Default::default()
        },
    );
}

// ── 2. Fused backward vs host f64 analytic backward (tight, small shapes) ─────

fn diff_vs_f64(seq_q: usize, seq_k: usize, d: usize, dv: usize, opts: Sdpa) {
    let g = gpu();
    let q = fill(seq_q * d, 401);
    let k = fill(seq_k * d, 502);
    let v = fill(seq_k * dv, 603);
    let scale = opts
        .scale
        .map(|s| s as f64)
        .unwrap_or(1.0 / (d as f64).sqrt());
    let eff = opts.kv_len.unwrap_or(seq_k).clamp(1, seq_k);
    let causal = opts.causal;

    let (fq, fk, fv) = fused_grads(&g, &q, &k, &v, seq_q, seq_k, d, dv, opts);

    // L = sum(O) ⇒ dO = all ones.
    let d_o = vec![1.0f64; seq_q * dv];
    let mask = |qi: usize, kj: usize| (!causal || kj <= qi) && kj < eff;
    let (rq, rk, rv) = host_backward(&q, &k, &v, &d_o, seq_q, seq_k, d, dv, scale, mask);

    let eq = max_rel_f64(&fq, &rq);
    let ek = max_rel_f64(&fk, &rk);
    let ev = max_rel_f64(&fv, &rv);
    if std::env::var("SDPA_PROBE").is_ok() {
        eprintln!(
            "[SDPA_PROBE f64] sq={seq_q} sk={seq_k} d={d} dv={dv} causal={causal} kv={:?} \
             dq={eq:.3e} dk={ek:.3e} dv={ev:.3e}",
            opts.kv_len
        );
    }
    assert!(
        eq < 1e-4 && ek < 1e-4 && ev < 1e-4,
        "fused vs f64 rel-err dq={eq:.2e} dk={ek:.2e} dv={ev:.2e} \
         (sq={seq_q} sk={seq_k} d={d} dv={dv} causal={causal} kv={:?})",
        opts.kv_len
    );
}

#[test]
fn bwd_vs_f64_square() {
    diff_vs_f64(4, 4, 5, 5, Sdpa::default());
}

#[test]
fn bwd_vs_f64_rect() {
    diff_vs_f64(3, 5, 4, 6, Sdpa::default());
    diff_vs_f64(5, 2, 7, 3, Sdpa::default());
}

#[test]
fn bwd_vs_f64_causal() {
    diff_vs_f64(
        5,
        5,
        4,
        4,
        Sdpa {
            causal: true,
            ..Default::default()
        },
    );
}

#[test]
fn bwd_vs_f64_padding() {
    diff_vs_f64(
        4,
        6,
        5,
        5,
        Sdpa {
            kv_len: Some(3),
            ..Default::default()
        },
    );
}

#[test]
fn bwd_vs_f64_causal_and_padding() {
    diff_vs_f64(
        5,
        6,
        4,
        4,
        Sdpa {
            causal: true,
            kv_len: Some(4),
            ..Default::default()
        },
    );
}

// ── 3. Central-difference gradcheck through the NEW sdpa_var backward ─────────

/// Central-difference every element of `input` (one of Q/K/V) and compare to
/// the fused analytic gradient. `which` picks the input to perturb (0=Q,1=K,2=V).
fn gradcheck(seq_q: usize, seq_k: usize, d: usize, dv: usize, opts: Sdpa, which: usize) {
    let g = gpu();
    let q = fill(seq_q * d, 701);
    let k = fill(seq_k * d, 802);
    let v = fill(seq_k * dv, 903);

    let loss_of = |qh: &[f32], kh: &[f32], vh: &[f32]| -> f32 {
        let tape = Tape::<f32>::new();
        let qv = tape.var(Array::from_slice(&g, qh, &[seq_q, d]).unwrap());
        let kv = tape.var(Array::from_slice(&g, kh, &[seq_k, d]).unwrap());
        let vv = tape.var(Array::from_slice(&g, vh, &[seq_k, dv]).unwrap());
        let out = sdpa_var(&tape, &qv, &kv, &vv, opts).unwrap();
        out.sum().unwrap().value().to_vec().unwrap()[0]
    };

    let (fq, fk, fv) = fused_grads(&g, &q, &k, &v, seq_q, seq_k, d, dv, opts);
    let (an, base): (&[f32], Vec<f32>) = match which {
        0 => (&fq, q.clone()),
        1 => (&fk, k.clone()),
        _ => (&fv, v.clone()),
    };

    let h = 1e-3f32;
    for j in 0..base.len() {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[j] += h;
        minus[j] -= h;
        let num = match which {
            0 => (loss_of(&plus, &k, &v) - loss_of(&minus, &k, &v)) / (2.0 * h),
            1 => (loss_of(&q, &plus, &v) - loss_of(&q, &minus, &v)) / (2.0 * h),
            _ => (loss_of(&q, &k, &plus) - loss_of(&q, &k, &minus)) / (2.0 * h),
        };
        assert!(
            (an[j] - num).abs() <= 2e-2 * (1.0 + num.abs()),
            "grad[{which}][{j}] analytic {} vs numeric {num}",
            an[j]
        );
    }
}

#[test]
fn gradcheck_q() {
    gradcheck(2, 3, 3, 2, Sdpa::default(), 0);
}

#[test]
fn gradcheck_k() {
    gradcheck(2, 3, 3, 2, Sdpa::default(), 1);
}

#[test]
fn gradcheck_v() {
    gradcheck(2, 3, 3, 2, Sdpa::default(), 2);
}

#[test]
fn gradcheck_causal_all() {
    let opts = Sdpa {
        causal: true,
        ..Default::default()
    };
    gradcheck(3, 3, 3, 2, opts, 0);
    gradcheck(3, 3, 3, 2, opts, 1);
    gradcheck(3, 3, 3, 2, opts, 2);
}

// ── 4. Stability: ±80 logits backward — finite grads, matches f64 host ───────

#[test]
fn bwd_stability_extreme_logits() {
    let g = gpu();
    let (seq_q, seq_k, d, dv) = (4, 6, 4, 4);
    let q = fill(seq_q * d, 21);
    let k = fill(seq_k * d, 22);
    let v = fill(seq_k * dv, 23);
    let scale = 80.0f32; // drives scaled scores to the ±tens–hundreds edge
    let opts = Sdpa {
        scale: Some(scale),
        causal: false,
        kv_len: None,
    };

    let (fq, fk, fv) = fused_grads(&g, &q, &k, &v, seq_q, seq_k, d, dv, opts);
    for (name, grads) in [("dq", &fq), ("dk", &fk), ("dv", &fv)] {
        for (i, &x) in grads.iter().enumerate() {
            assert!(
                x.is_finite(),
                "{name}[{i}] = {x} not finite under ±80 logits"
            );
        }
    }

    // Match the f64 analytic backward at the same scale (dO = ones).
    let d_o = vec![1.0f64; seq_q * dv];
    let mask = |_qi: usize, _kj: usize| true;
    let (rq, rk, rv) = host_backward(&q, &k, &v, &d_o, seq_q, seq_k, d, dv, scale as f64, mask);
    let eq = max_rel_f64(&fq, &rq);
    let ek = max_rel_f64(&fk, &rk);
    let ev = max_rel_f64(&fv, &rv);
    if std::env::var("SDPA_PROBE").is_ok() {
        eprintln!("[SDPA_PROBE stability] dq={eq:.3e} dk={ek:.3e} dv={ev:.3e}");
    }
    assert!(
        eq < 1e-3 && ek < 1e-3 && ev < 1e-3,
        "extreme-logit backward rel-err dq={eq:.2e} dk={ek:.2e} dv={ev:.2e}"
    );
}

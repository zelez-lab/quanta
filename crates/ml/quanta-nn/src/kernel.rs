//! Fused scaled-dot-product attention — **forward** kernel (single head, f32).
//!
//! This is the online-softmax / FlashAttention-style forward pass proven
//! correct in `specs/verify/lean/Quanta/Nn/OnlineSoftmax.lean` (theorems
//! T9200–T9209). Each thread owns one output entry and streams over the key
//! sequence carrying the running `(m, l, acc)` state, so the `seq_q × seq_k`
//! score matrix is **never materialised** — the whole point of the fused form.
//!
//! ## Thread layout (the MVP)
//!
//! One thread per `(query row, value dim)` pair — `seq_q * dv` threads. Thread
//! `i` computes `out[qi, vd]` for `qi = i / dv`, `vd = i % dv`. This mirrors
//! `quanta-blas`'s `gemm_f32_naive` (one thread per output entry), which is the
//! lowering shape proven safe for a top-level loop-carried accumulator.
//!
//! The Lean model is scalar (`List (ℝ × ℝ)` of `(score, value)` pairs) and
//! notes "the kernel replays this per output dimension" — this layout *is* that
//! replay: every `vd` thread of a row re-derives the same `(m, l)` from the
//! scores and mixes in its own value coordinate `v[kj, vd]`. The `(m, l)` stats
//! (identical across a row's `vd` threads, since they depend only on the
//! scores) are written once, by the `vd == 0` lane, to the side buffer that the
//! future fused backward will consume. Recomputing the scores per `vd` is the
//! "naive recompute" this MVP commit tolerates; a single-thread-per-row
//! accumulator-vector kernel is a later perf increment.
//!
//! ## The online recurrence — one self-seeding loop
//!
//! There is exactly **one** loop over the key sequence and exactly **one**
//! inner dot-product loop in the whole kernel. (An earlier draft with a
//! separate seed pass over key 0 tripped the WASM-route local-renaming issue —
//! a register from the seed loop aliased one in the streaming loop. Keeping a
//! single loop of each kind sidesteps it entirely.)
//!
//! State starts neutral `(m, l, acc) = (0, 0, 0)` and self-seeds on the first
//! key via a branchless `is_first` flag, reproducing the Lean seed
//! `(s₀, 1, v₀)`. For each key `kj` with score `sₖ`:
//! ```text
//!   m'   = is_first ? sₖ : max(m, sₖ)
//!   a    = is_first ? 0  : exp(min(m − m', 0))   // discard garbage on seed
//!   b    = exp(min(sₖ − m', 0))                  // fresh weight, arg ≤ 0
//!   l    = l·a + b        acc = acc·a + b·vₖ      m = m'
//! ```
//! On the first key `m' = s₀`, `a = 0`, `b = 1`, so `l = 1`, `acc = v₀` — the
//! exact Lean seed, with no `f32::MIN`/sentinel (the Metal float-literal /
//! maxpool trap). Every real-step exp argument is `≤ 0` (T9207), and the
//! `min(·, 0)` clamp keeps the discarded seed-step rescale from overflowing to
//! `inf` (which `0 · inf = NaN` would then poison).
//!
//! ## Masking (causal + key-padding), branch-free
//!
//! A masked key must contribute weight ≈ 0 without a branch around the
//! loop-carried update (that would hit the structured-control lowering hazard).
//! Both masks fold into a single additive bias `bias = (1 − valid)·(−1e9)`,
//! where `valid ∈ {0,1}` is the product of two indicator flags (`kj ≤ qi`
//! causal, `kj < kv_len` padding) — indicator *arithmetic*, never a
//! short-circuit `&&` on loop-dependent operands. A masked score `s − 1e9`
//! gives `exp(s − 1e9 − m') = 0` in f32, so the key is inert.

use quanta_core::{Field, Gpu, QuantaError};

#[allow(unused_imports)]
mod dsl {
    use quanta_core::*;

    /// Fused SDPA forward. Thread `i` owns `out[qi, vd]` with `qi = i / dv`,
    /// `vd = i % dv`; it streams the online-softmax recurrence over all
    /// `seq_k` keys. Row-major layouts: `q` is `seq_q × d`, `k` is
    /// `seq_k × d`, `v` and `out` are `seq_k × dv` / `seq_q × dv`; `stats` is
    /// `seq_q × 2` (`[m, l]` per row, written by the `vd == 0` lane).
    ///
    /// `scale` multiplies the raw dot product. `causal != 0` restricts row
    /// `qi` to keys `kj ≤ qi`; `kv_len` restricts to keys `kj < kv_len`
    /// (padding). Both fold into one additive bias so the loop-carried `(m, l,
    /// acc)` update stays branch-free and top-level (lowering-safe).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_forward(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        out: &mut [f32],
        stats: &mut [f32],
        seq_q: u32,
        seq_k: u32,
        d: u32,
        dv: u32,
        scale: f32,
        causal: u32,
        kv_len: u32,
    ) {
        let i = quark_id();
        let total = seq_q * dv;
        // Clamp to a valid entry so over-dispatched lanes do real (redundant)
        // work; the stores are guarded below. The streaming loop stays
        // top-level — never nested in a bounds `if` — so the loop-carried
        // accumulator lowers correctly (the redirect-chain / loop-guard trap).
        let idx = if i < total { i } else { 0u32 };
        let qi = idx / dv;
        let vd = idx % dv;

        let q_base = qi * d;

        // Neutral start; self-seeds on the first key (see module docs).
        let mut m: f32 = 0.0f32;
        let mut l: f32 = 0.0f32;
        let mut acc: f32 = 0.0f32;

        let mut kj: u32 = 0u32;
        while kj < seq_k {
            let k_base = kj * d;

            // Raw score sₖ = scale · (qᵢ · kₖ). Single inner dot-product loop.
            let mut dot: f32 = 0.0f32;
            let mut p: u32 = 0u32;
            while p < d {
                dot = dot + q[(q_base + p) as usize] * k[(k_base + p) as usize];
                p = p + 1u32;
            }
            let raw = scale * dot;

            // Single validity flag = causal-indicator × padding-indicator.
            // Indicator arithmetic (no `&&`): each `if` yields 0.0/1.0, then
            // multiply. `causal == 0` disables the causal test (flag 1.0).
            let causal_pass = if causal == 0u32 { 1u32 } else { 0u32 };
            let causal_le = if kj <= qi { 1u32 } else { 0u32 };
            let causal_ok = if (causal_pass + causal_le) > 0u32 {
                1.0f32
            } else {
                0.0f32
            };
            let pad_ok = if kj < kv_len { 1.0f32 } else { 0.0f32 };
            let valid = causal_ok * pad_ok;
            // Masked keys get a −1e9 bias → exp(sₖ − 1e9 − m') = 0 in f32.
            let sk = raw + (1.0f32 - valid) * (0.0f32 - 1000000000.0f32);

            // Branchless self-seeding online step. `first` = 1.0 on kj == 0.
            let first = if kj == 0u32 { 1.0f32 } else { 0.0f32 };
            let m_cmp = fmax(m, sk);
            let m_new = first * sk + (1.0f32 - first) * m_cmp;
            // Rescale factor for the running state; zeroed on the seed step.
            // `fmin(·, 0.0)` clamps the (discarded) seed-step arg so it can't
            // overflow to inf before the ×0 mask lands.
            let a = (1.0f32 - first) * exp(fmin(m - m_new, 0.0f32));
            // Fresh weight; its argument is always ≤ 0 since m_new ≥ sk.
            let b = exp(fmin(sk - m_new, 0.0f32));
            l = l * a + b;
            acc = acc * a + b * v[(kj * dv + vd) as usize];
            m = m_new;

            kj = kj + 1u32;
        }

        // out = acc / l. Guarded by a single bounds flag (gemm-naive shape).
        if i < total {
            out[idx as usize] = acc / l;
        }
        // Stats (m, l) once per row, from the vd == 0 lane. Fold the two
        // conditions (in-bounds AND vd == 0) into ONE flag via indicator
        // arithmetic — nested `if`s collapse badly in the WASM-route lowering.
        let in_bounds = if i < total { 1u32 } else { 0u32 };
        let lane0 = if vd == 0u32 { 1u32 } else { 0u32 };
        let write_stats = in_bounds * lane0;
        if write_stats == 1u32 {
            stats[(2u32 * qi) as usize] = m;
            stats[(2u32 * qi + 1u32) as usize] = l;
        }
    }

    /// **Backward: D-precompute.** One thread per query row `i`, writing the
    /// row dot `D_i = Σ_c dO_ic · O_ic` to `delta[i]`. The dQ/dK kernels read
    /// `D` from this side buffer (dK reads it once per streamed `i`, so
    /// precomputing beats recomputing the `dv`-length dot inside dK's inner
    /// loop). `dv` is small, so a single loop per row is cheap. Over-dispatched
    /// lanes clamp their index and the store is bounds-guarded (gemm-naive
    /// shape), keeping the streaming loop top-level (the loop-guard trap).
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    pub fn sdpa_bwd_delta(d_o: &[f32], out: &[f32], delta: &mut [f32], seq_q: u32, dv: u32) {
        let i = quark_id();
        let idx = if i < seq_q { i } else { 0u32 };
        let base = idx * dv;
        let mut acc: f32 = 0.0f32;
        let mut c: u32 = 0u32;
        while c < dv {
            acc = acc + d_o[(base + c) as usize] * out[(base + c) as usize];
            c = c + 1u32;
        }
        if i < seq_q {
            delta[idx as usize] = acc;
        }
    }

    /// **Backward: dQ.** Thread `t` owns `dq[qi, qd]` with `qi = t / d`,
    /// `qd = t % d`. Streams over all keys `kj`: reconstructs `p_ij` from the
    /// saved `(m_i, l_i)` (recomputing the `q_i·k_j` dot inline, then the same
    /// masked score the forward used), forms `dS_ij = p_ij·(Σ_c dO_ic·V_jc −
    /// D_i)`, and accumulates `scale · dS_ij · k[j, qd]`. Masked keys get the
    /// −1e9 bias → `p_ij = 0`, so they contribute nothing (mirrors the
    /// forward's masking exactly). One streaming loop, one inner dot loop; the
    /// dO·V dot is a second inner loop over `dv` (`dv` small). Index clamped,
    /// store bounds-guarded.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_bwd_dq(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_o: &[f32],
        stats: &[f32],
        delta: &[f32],
        dq: &mut [f32],
        seq_q: u32,
        seq_k: u32,
        d: u32,
        dv: u32,
        scale: f32,
        causal: u32,
        kv_len: u32,
    ) {
        let t = quark_id();
        let total = seq_q * d;
        let idx = if t < total { t } else { 0u32 };
        let qi = idx / d;
        let qd = idx % d;

        let q_base = qi * d;
        let m_i = stats[(2u32 * qi) as usize];
        let l_i = stats[(2u32 * qi + 1u32) as usize];
        let d_i = delta[qi as usize];

        let mut acc: f32 = 0.0f32;
        let mut kj: u32 = 0u32;
        while kj < seq_k {
            let k_base = kj * d;

            // Raw score sₖ = scale·(qᵢ·kⱼ). Single inner dot loop over d.
            let mut dot: f32 = 0.0f32;
            let mut p: u32 = 0u32;
            while p < d {
                dot = dot + q[(q_base + p) as usize] * k[(k_base + p) as usize];
                p = p + 1u32;
            }
            let raw = scale * dot;

            // Same masking as the forward: causal-indicator × padding-indicator.
            let causal_pass = if causal == 0u32 { 1u32 } else { 0u32 };
            let causal_le = if kj <= qi { 1u32 } else { 0u32 };
            let causal_ok = if (causal_pass + causal_le) > 0u32 {
                1.0f32
            } else {
                0.0f32
            };
            let pad_ok = if kj < kv_len { 1.0f32 } else { 0.0f32 };
            let valid = causal_ok * pad_ok;
            let sk = raw + (1.0f32 - valid) * (0.0f32 - 1000000000.0f32);

            // Reconstruct pᵢⱼ = exp(sₖ − mᵢ)/lᵢ (T9204). Masked keys → sₖ −1e9
            // → exp = 0. Clamp the exp arg at 0 (sₖ ≤ mᵢ for valid keys; the
            // clamp is a no-op there and keeps masked args from overflowing).
            let pij = exp(fmin(sk - m_i, 0.0f32)) / l_i;

            // dO_i · V_j  (second inner loop over dv).
            let do_base = qi * dv;
            let v_base = kj * dv;
            let mut dov: f32 = 0.0f32;
            let mut c: u32 = 0u32;
            while c < dv {
                dov = dov + d_o[(do_base + c) as usize] * v[(v_base + c) as usize];
                c = c + 1u32;
            }
            let ds = pij * (dov - d_i);
            acc = acc + ds * k[(k_base + qd) as usize];

            kj = kj + 1u32;
        }

        if t < total {
            dq[idx as usize] = scale * acc;
        }
    }

    /// **Backward: dK.** Thread `t` owns `dk[kj, kd]` with `kj = t / d`,
    /// `kd = t % d`. Streams over all queries `qi`: reconstructs `p_ij` from
    /// `(m_i, l_i)` (with the same masked score), forms `dS_ij = p_ij·(Σ_c
    /// dO_ic·V_jc − D_i)`, and accumulates `scale · dS_ij · q[i, kd]`. `D_i`
    /// comes from the `delta` side buffer (read once per streamed `i`). Masked
    /// pairs contribute zero via `p_ij = 0`. Index clamped, store guarded.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_bwd_dk(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        d_o: &[f32],
        stats: &[f32],
        delta: &[f32],
        dk: &mut [f32],
        seq_q: u32,
        seq_k: u32,
        d: u32,
        dv: u32,
        scale: f32,
        causal: u32,
        kv_len: u32,
    ) {
        let t = quark_id();
        let total = seq_k * d;
        let idx = if t < total { t } else { 0u32 };
        let kj = idx / d;
        let kd = idx % d;

        let k_base = kj * d;

        let mut acc: f32 = 0.0f32;
        let mut qi: u32 = 0u32;
        while qi < seq_q {
            let q_base = qi * d;
            let m_i = stats[(2u32 * qi) as usize];
            let l_i = stats[(2u32 * qi + 1u32) as usize];
            let d_i = delta[qi as usize];

            // Raw score sₖ = scale·(qᵢ·kⱼ). Single inner dot loop over d.
            let mut dot: f32 = 0.0f32;
            let mut p: u32 = 0u32;
            while p < d {
                dot = dot + q[(q_base + p) as usize] * k[(k_base + p) as usize];
                p = p + 1u32;
            }
            let raw = scale * dot;

            // Same masking as the forward.
            let causal_pass = if causal == 0u32 { 1u32 } else { 0u32 };
            let causal_le = if kj <= qi { 1u32 } else { 0u32 };
            let causal_ok = if (causal_pass + causal_le) > 0u32 {
                1.0f32
            } else {
                0.0f32
            };
            let pad_ok = if kj < kv_len { 1.0f32 } else { 0.0f32 };
            let valid = causal_ok * pad_ok;
            let sk = raw + (1.0f32 - valid) * (0.0f32 - 1000000000.0f32);

            let pij = exp(fmin(sk - m_i, 0.0f32)) / l_i;

            // dO_i · V_j (second inner loop over dv).
            let do_base = qi * dv;
            let v_base = kj * dv;
            let mut dov: f32 = 0.0f32;
            let mut c: u32 = 0u32;
            while c < dv {
                dov = dov + d_o[(do_base + c) as usize] * v[(v_base + c) as usize];
                c = c + 1u32;
            }
            let ds = pij * (dov - d_i);
            acc = acc + ds * q[(q_base + kd) as usize];

            qi = qi + 1u32;
        }

        if t < total {
            dk[idx as usize] = scale * acc;
        }
    }

    /// **Backward: dV.** Thread `t` owns `dv_out[kj, c]` with `kj = t / dv`,
    /// `c = t % dv`. Streams over all queries `qi`, reconstructing `p_ij` from
    /// `(m_i, l_i)` (same masked score as the forward) and accumulating
    /// `p_ij · dO[i, c]`. `dV` needs neither `D` nor `scale`. Masked pairs
    /// contribute zero via `p_ij = 0`. Index clamped, store guarded.
    #[quanta_compute_dsl::kernel(crate = quanta_core, workgroup = [256])]
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_bwd_dv(
        q: &[f32],
        k: &[f32],
        d_o: &[f32],
        stats: &[f32],
        dv_out: &mut [f32],
        seq_q: u32,
        seq_k: u32,
        d: u32,
        dv: u32,
        scale: f32,
        causal: u32,
        kv_len: u32,
    ) {
        let t = quark_id();
        let total = seq_k * dv;
        let idx = if t < total { t } else { 0u32 };
        let kj = idx / dv;
        let vc = idx % dv;

        let k_base = kj * d;

        let mut acc: f32 = 0.0f32;
        let mut qi: u32 = 0u32;
        while qi < seq_q {
            let q_base = qi * d;
            let m_i = stats[(2u32 * qi) as usize];
            let l_i = stats[(2u32 * qi + 1u32) as usize];

            // Raw score sₖ = scale·(qᵢ·kⱼ). Single inner dot loop over d.
            let mut dot: f32 = 0.0f32;
            let mut p: u32 = 0u32;
            while p < d {
                dot = dot + q[(q_base + p) as usize] * k[(k_base + p) as usize];
                p = p + 1u32;
            }
            let raw = scale * dot;

            // Same masking as the forward.
            let causal_pass = if causal == 0u32 { 1u32 } else { 0u32 };
            let causal_le = if kj <= qi { 1u32 } else { 0u32 };
            let causal_ok = if (causal_pass + causal_le) > 0u32 {
                1.0f32
            } else {
                0.0f32
            };
            let pad_ok = if kj < kv_len { 1.0f32 } else { 0.0f32 };
            let valid = causal_ok * pad_ok;
            let sk = raw + (1.0f32 - valid) * (0.0f32 - 1000000000.0f32);

            let pij = exp(fmin(sk - m_i, 0.0f32)) / l_i;
            acc = acc + pij * d_o[(qi * dv + vc) as usize];

            qi = qi + 1u32;
        }

        if t < total {
            dv_out[idx as usize] = acc;
        }
    }
}

/// Host-side dispatch of the fused SDPA forward. All fields are row-major
/// `f32`: `q` is `seq_q × d`, `k` is `seq_k × d`, `v` is `seq_k × dv`,
/// `out` is `seq_q × dv`, `stats` is `seq_q × 2` (`[m, l]` per row). `scale`
/// multiplies the dot product; `causal` gates the lower-triangular mask;
/// `kv_len` is the effective (unpadded) key count (`= seq_k` for no padding).
///
/// Single head. Batch and heads are the caller's host loop (documented in
/// [`crate::functional`]); this dispatches exactly one head.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward(
    gpu: &Gpu,
    seq_q: u32,
    seq_k: u32,
    d: u32,
    dv: u32,
    scale: f32,
    causal: bool,
    kv_len: u32,
    q: &Field<f32>,
    k: &Field<f32>,
    v: &Field<f32>,
    out: &Field<f32>,
    stats: &Field<f32>,
) -> Result<(), QuantaError> {
    let (sq, sk, du, dvu) = (seq_q as usize, seq_k as usize, d as usize, dv as usize);
    if q.len() != sq * du {
        return Err(QuantaError::invalid_param("sdpa: Q length must be seq_q*d"));
    }
    if k.len() != sk * du {
        return Err(QuantaError::invalid_param("sdpa: K length must be seq_k*d"));
    }
    if v.len() != sk * dvu {
        return Err(QuantaError::invalid_param(
            "sdpa: V length must be seq_k*dv",
        ));
    }
    if out.len() != sq * dvu {
        return Err(QuantaError::invalid_param(
            "sdpa: OUT length must be seq_q*dv",
        ));
    }
    if stats.len() != sq * 2 {
        return Err(QuantaError::invalid_param(
            "sdpa: STATS length must be seq_q*2",
        ));
    }
    if seq_q == 0 || seq_k == 0 || d == 0 || dv == 0 {
        return Ok(());
    }

    let mut wave = dsl::sdpa_forward(gpu)?;
    wave.bind(0, q);
    wave.bind(1, k);
    wave.bind(2, v);
    wave.bind(3, out);
    wave.bind(4, stats);
    wave.set_value(5, seq_q);
    wave.set_value(6, seq_k);
    wave.set_value(7, d);
    wave.set_value(8, dv);
    wave.set_value(9, scale);
    wave.set_value(10, if causal { 1u32 } else { 0u32 });
    wave.set_value(11, kv_len);
    gpu.dispatch(&wave, seq_q * dv)?.wait()?;
    Ok(())
}

/// Host-side dispatch of the **fused SDPA backward** (single head, f32).
///
/// Consumes the forward's saved per-row `(m, l)` stats to reconstruct the
/// softmax weights (T9204) without ever materialising the `seq_q × seq_k`
/// score matrix — the FlashAttention backward. Row-major `f32` layouts match
/// the forward: `q` is `seq_q × d`, `k` is `seq_k × d`, `v` is `seq_k × dv`,
/// `out`/`d_o` are `seq_q × dv`, `stats` is `seq_q × 2`. Writes `dq`
/// (`seq_q × d`), `dk` (`seq_k × d`), `dv_out` (`seq_k × dv`).
///
/// Four dispatches: a `delta` precompute (`D_i = Σ_c dO_ic·O_ic`, one thread
/// per row, into a scratch field) then the three gradient kernels, each one
/// thread per output element. `delta` is precomputed rather than inlined so dK
/// (which streams `i`) reads `D_i` once per key-row instead of recomputing it
/// every iteration.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_backward(
    gpu: &Gpu,
    seq_q: u32,
    seq_k: u32,
    d: u32,
    dv: u32,
    scale: f32,
    causal: bool,
    kv_len: u32,
    q: &Field<f32>,
    k: &Field<f32>,
    v: &Field<f32>,
    out: &Field<f32>,
    stats: &Field<f32>,
    d_o: &Field<f32>,
    dq: &Field<f32>,
    dk: &Field<f32>,
    dv_out: &Field<f32>,
) -> Result<(), QuantaError> {
    let (sq, sk, du, dvu) = (seq_q as usize, seq_k as usize, d as usize, dv as usize);
    if q.len() != sq * du {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: Q length must be seq_q*d",
        ));
    }
    if k.len() != sk * du {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: K length must be seq_k*d",
        ));
    }
    if v.len() != sk * dvu {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: V length must be seq_k*dv",
        ));
    }
    if out.len() != sq * dvu || d_o.len() != sq * dvu {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: OUT/dO length must be seq_q*dv",
        ));
    }
    if stats.len() != sq * 2 {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: STATS length must be seq_q*2",
        ));
    }
    if dq.len() != sq * du {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: dQ length must be seq_q*d",
        ));
    }
    if dk.len() != sk * du {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: dK length must be seq_k*d",
        ));
    }
    if dv_out.len() != sk * dvu {
        return Err(QuantaError::invalid_param(
            "sdpa_bwd: dV length must be seq_k*dv",
        ));
    }
    if seq_q == 0 || seq_k == 0 || d == 0 || dv == 0 {
        return Ok(());
    }

    let causal_u = if causal { 1u32 } else { 0u32 };

    // 1. Delta precompute: D_i = Σ_c dO_ic·O_ic → scratch field, one per row.
    let delta = gpu.field::<f32>(sq)?;
    {
        let mut wave = dsl::sdpa_bwd_delta(gpu)?;
        wave.bind(0, d_o);
        wave.bind(1, out);
        wave.bind(2, &delta);
        wave.set_value(3, seq_q);
        wave.set_value(4, dv);
        gpu.dispatch(&wave, seq_q)?.wait()?;
    }

    // 2. dQ: one thread per (query row, head dim).
    {
        let mut wave = dsl::sdpa_bwd_dq(gpu)?;
        wave.bind(0, q);
        wave.bind(1, k);
        wave.bind(2, v);
        wave.bind(3, d_o);
        wave.bind(4, stats);
        wave.bind(5, &delta);
        wave.bind(6, dq);
        wave.set_value(7, seq_q);
        wave.set_value(8, seq_k);
        wave.set_value(9, d);
        wave.set_value(10, dv);
        wave.set_value(11, scale);
        wave.set_value(12, causal_u);
        wave.set_value(13, kv_len);
        gpu.dispatch(&wave, seq_q * d)?.wait()?;
    }

    // 3. dK: one thread per (key row, head dim).
    {
        let mut wave = dsl::sdpa_bwd_dk(gpu)?;
        wave.bind(0, q);
        wave.bind(1, k);
        wave.bind(2, v);
        wave.bind(3, d_o);
        wave.bind(4, stats);
        wave.bind(5, &delta);
        wave.bind(6, dk);
        wave.set_value(7, seq_q);
        wave.set_value(8, seq_k);
        wave.set_value(9, d);
        wave.set_value(10, dv);
        wave.set_value(11, scale);
        wave.set_value(12, causal_u);
        wave.set_value(13, kv_len);
        gpu.dispatch(&wave, seq_k * d)?.wait()?;
    }

    // 4. dV: one thread per (key row, value dim).
    {
        let mut wave = dsl::sdpa_bwd_dv(gpu)?;
        wave.bind(0, q);
        wave.bind(1, k);
        wave.bind(2, d_o);
        wave.bind(3, stats);
        wave.bind(4, dv_out);
        wave.set_value(5, seq_q);
        wave.set_value(6, seq_k);
        wave.set_value(7, d);
        wave.set_value(8, dv);
        wave.set_value(9, scale);
        wave.set_value(10, causal_u);
        wave.set_value(11, kv_len);
        gpu.dispatch(&wave, seq_k * dv)?.wait()?;
    }

    Ok(())
}

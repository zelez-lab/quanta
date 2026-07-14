//! Readiness probe: a tiny GPT-style decoder block trains end-to-end.
//!
//! Per-op gradient checks prove each VJP; `training.rs` proves the loop
//! converges on a linear model. This proves the *transformer path composes* —
//! embedding → single-head causal self-attention (batched matmul + additive
//! causal mask + row-wise softmax) → residual + norm → FFN(gelu) → residual +
//! norm → logits → cross-entropy → backward → SGD step — and that the loss
//! actually drops. If any op failed to compose across these shapes, the loss
//! would stall or the build would error here.
//!
//! Single sequence (batch 1), single head, tiny dims — this is a correctness
//! probe on the CPU JIT lane (a fresh kernel per op per step), not a perf test.

use quanta_array::Array;
use quanta_autograd::{Tape, Var};

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

/// `p ← p − lr·g`, the optimizer step outside the tape (each step rebuilds a
/// fresh tape from the current params).
fn sgd_step(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap();
    let scaled = g.mul(&lr_a.broadcast_to(g.shape()).unwrap()).unwrap();
    p.sub(&scaled).unwrap()
}

/// Multiply a `Var` by a scalar constant (broadcast), staying on `tape`.
fn scale(tape: &Tape<f32>, v: &Var<f32>, s: f32) -> Var<f32> {
    let val = v.value();
    let g = val.gpu();
    let sh = val.shape().to_vec();
    let c = Array::full(g, s, &[1])
        .unwrap()
        .broadcast_to(&sh)
        .unwrap()
        .contiguous()
        .unwrap();
    v.mul(&tape.var(c)).unwrap()
}

#[test]
fn gpt_decoder_block_trains() {
    let g = gpu();

    // Tiny problem: predict the next token in a fixed length-T sequence.
    // Vocab V, model dim D, FFN hidden F. Single head, batch 1.
    let (t, v, d, f) = (4usize, 5usize, 8usize, 16usize);
    let scale_qk = 1.0f32 / (d as f32).sqrt();

    // Toy task: input tokens [0,1,2,3], targets are the shifted sequence
    // [1,2,3,4] — a deterministic "increment" the block can memorize.
    let tokens: Vec<u32> = vec![0, 1, 2, 3];
    let targets: Vec<u32> = vec![1, 2, 3, 4];
    let tok_arr = Array::from_slice(&g, &tokens, &[t]).unwrap();
    let tgt_arr = Array::from_slice(&g, &targets, &[t]).unwrap();

    // Additive causal mask [T,T]: 0 where j<=i, −1e9 where j>i (future).
    let mut mask_h = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask_h[i * t + j] = -1.0e9;
            }
        }
    }
    let mask = Array::from_slice(&g, &mask_h, &[t, t]).unwrap();

    // Parameters (small deterministic init). Embedding [V,D]; projections [D,D];
    // FFN [D,F] and [F,D]; output head [D,V].
    let init = |rows: usize, cols: usize, seed: f32| -> Array<f32> {
        let n = rows * cols;
        let data: Vec<f32> = (0..n)
            .map(|i| ((((i as f32) * 12.9898 + seed).sin() * 43758.547).fract() - 0.5) * 0.2)
            .collect();
        Array::from_slice(&g, &data, &[rows, cols]).unwrap()
    };
    let mut emb = init(v, d, 1.0);
    let mut wq = init(d, d, 2.0);
    let mut wk = init(d, d, 3.0);
    let mut wv = init(d, d, 4.0);
    let mut w1 = init(d, f, 5.0);
    let mut w2 = init(f, d, 6.0);
    let mut wout = init(d, v, 7.0);
    // RMSNorm gains, one per D and applied on [T,D].
    let mut g1 = Array::from_slice(&g, &vec![1.0f32; d], &[d]).unwrap();
    let mut g2 = Array::from_slice(&g, &vec![1.0f32; d], &[d]).unwrap();

    let lr = 0.3f32;
    let mut first_loss = None;
    let mut last_loss = 0.0f32;

    for _ in 0..40 {
        let tape = Tape::<f32>::new();
        let embv = tape.var(emb.shallow_clone());
        let wqv = tape.var(wq.shallow_clone());
        let wkv = tape.var(wk.shallow_clone());
        let wvv = tape.var(wv.shallow_clone());
        let w1v = tape.var(w1.shallow_clone());
        let w2v = tape.var(w2.shallow_clone());
        let woutv = tape.var(wout.shallow_clone());
        let g1v = tape.var(g1.shallow_clone());
        let g2v = tape.var(g2.shallow_clone());

        // x = embedding lookup → [T, D]
        let x = embv.embedding(&tok_arr).unwrap();

        // Single-head self-attention (2-D matmuls; [T,D]).
        let q = x.matmul(&wqv).unwrap(); // [T,D]
        let k = x.matmul(&wkv).unwrap(); // [T,D]
        let vv = x.matmul(&wvv).unwrap(); // [T,D]
        // scores = (q · kᵀ) / √d + causal_mask   → [T,T]
        let kt = k.transpose(0, 1).unwrap(); // [D,T]
        let mut scores = scale(&tape, &q.matmul(&kt).unwrap(), scale_qk); // [T,T]
        scores = scores.add(&tape.var(mask.shallow_clone())).unwrap();
        // row-wise softmax over the last axis ([T,T] is already 2-D [N=T, C=T]).
        let attn = scores.softmax().unwrap(); // [T,T]
        let ctx = attn.matmul(&vv).unwrap(); // [T,D]

        // residual + RMSNorm
        let h = x.add(&ctx).unwrap().rms_norm(&g1v, 1e-5).unwrap(); // [T,D]

        // FFN with gelu, then residual + norm
        let ff = h
            .matmul(&w1v)
            .unwrap()
            .gelu()
            .unwrap()
            .matmul(&w2v)
            .unwrap(); // [T,D]
        let out = h.add(&ff).unwrap().rms_norm(&g2v, 1e-5).unwrap(); // [T,D]

        // logits [T,V], cross-entropy against the shifted targets
        let logits = out.matmul(&woutv).unwrap(); // [T,V]
        let loss = logits.cross_entropy(&tgt_arr).unwrap();

        last_loss = loss.value().to_vec().unwrap()[0];
        first_loss.get_or_insert(last_loss);

        // Backward + SGD on every parameter.
        emb = sgd_step(&emb, &loss.grad(&embv).unwrap(), lr);
        wq = sgd_step(&wq, &loss.grad(&wqv).unwrap(), lr);
        wk = sgd_step(&wk, &loss.grad(&wkv).unwrap(), lr);
        wv = sgd_step(&wv, &loss.grad(&wvv).unwrap(), lr);
        w1 = sgd_step(&w1, &loss.grad(&w1v).unwrap(), lr);
        w2 = sgd_step(&w2, &loss.grad(&w2v).unwrap(), lr);
        wout = sgd_step(&wout, &loss.grad(&woutv).unwrap(), lr);
        g1 = sgd_step(&g1, &loss.grad(&g1v).unwrap(), lr);
        g2 = sgd_step(&g2, &loss.grad(&g2v).unwrap(), lr);
    }

    let first = first_loss.unwrap();
    // The decoder block must actually learn: the loss drops substantially.
    assert!(
        last_loss.is_finite() && last_loss < first * 0.5,
        "decoder block didn't learn: loss {first} → {last_loss}"
    );
}

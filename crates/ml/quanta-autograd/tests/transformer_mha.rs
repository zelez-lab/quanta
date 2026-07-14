//! Multi-head variant of the decoder-block readiness probe: a tiny GPT block
//! using `multi_head_attention` (H=2) trains end-to-end and the loss drops.
//!
//! The single-head probe (`transformer.rs`) proves the attention *pattern*
//! composes; this proves the multi-head split/merge (reshape + transpose +
//! batched matmul over the (B,H) batch dims + row-wise softmax on [B,H,T,T])
//! trains — i.e. its VJP is right on real transformer shapes. Batch 1, tiny
//! dims — a correctness probe on the CPU JIT lane, not a perf test.

use quanta_array::Array;
use quanta_autograd::Tape;

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

fn sgd_step(p: &Array<f32>, g: &Array<f32>, lr: f32) -> Array<f32> {
    let lr_a = Array::full(p.gpu(), lr, &[1]).unwrap();
    let scaled = g.mul(&lr_a.broadcast_to(g.shape()).unwrap()).unwrap();
    p.sub(&scaled).unwrap()
}

#[test]
fn gpt_decoder_block_multihead_trains() {
    let g = gpu();

    // [B,T,D], H heads, vocab V, FFN F. Batch 1.
    let (b, t, v, d, heads, f) = (1usize, 4usize, 5usize, 8usize, 2usize, 16usize);

    let tokens: Vec<u32> = vec![0, 1, 2, 3];
    let targets: Vec<u32> = vec![1, 2, 3, 4];
    let tok_arr = Array::from_slice(&g, &tokens, &[t]).unwrap();
    let tgt_arr = Array::from_slice(&g, &targets, &[t]).unwrap();

    // Additive causal mask [T,T].
    let mut mask_h = vec![0.0f32; t * t];
    for i in 0..t {
        for j in 0..t {
            if j > i {
                mask_h[i * t + j] = -1.0e9;
            }
        }
    }
    let mask = Array::from_slice(&g, &mask_h, &[t, t]).unwrap();

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
    let mut wo = init(d, d, 8.0);
    let mut w1 = init(d, f, 5.0);
    let mut w2 = init(f, d, 6.0);
    let mut wout = init(d, v, 7.0);
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
        let wov = tape.var(wo.shallow_clone());
        let w1v = tape.var(w1.shallow_clone());
        let w2v = tape.var(w2.shallow_clone());
        let woutv = tape.var(wout.shallow_clone());
        let g1v = tape.var(g1.shallow_clone());
        let g2v = tape.var(g2.shallow_clone());
        let maskv = tape.var(mask.shallow_clone());

        // x = embedding [T,D] → reshape to [B,T,D] for multi-head attention.
        let x2 = embv.embedding(&tok_arr).unwrap(); // [T,D]
        let x = x2.reshape(&[b, t, d]).unwrap(); // [B,T,D]

        // Multi-head causal self-attention → [B,T,D].
        let attn = x
            .multi_head_attention(&wqv, &wkv, &wvv, &wov, heads, Some(&maskv))
            .unwrap();

        // residual + RMSNorm (on [T,D] — collapse the batch of 1).
        let x_flat = x.reshape(&[t, d]).unwrap();
        let attn_flat = attn.reshape(&[t, d]).unwrap();
        let h = x_flat
            .add(&attn_flat)
            .unwrap()
            .rms_norm(&g1v, 1e-5)
            .unwrap(); // [T,D]

        // FFN(gelu) + residual + RMSNorm.
        let ff = h
            .matmul(&w1v)
            .unwrap()
            .gelu()
            .unwrap()
            .matmul(&w2v)
            .unwrap();
        let out = h.add(&ff).unwrap().rms_norm(&g2v, 1e-5).unwrap(); // [T,D]

        let logits = out.matmul(&woutv).unwrap(); // [T,V]
        let loss = logits.cross_entropy(&tgt_arr).unwrap();

        last_loss = loss.value().to_vec().unwrap()[0];
        first_loss.get_or_insert(last_loss);

        emb = sgd_step(&emb, &loss.grad(&embv).unwrap(), lr);
        wq = sgd_step(&wq, &loss.grad(&wqv).unwrap(), lr);
        wk = sgd_step(&wk, &loss.grad(&wkv).unwrap(), lr);
        wv = sgd_step(&wv, &loss.grad(&wvv).unwrap(), lr);
        wo = sgd_step(&wo, &loss.grad(&wov).unwrap(), lr);
        w1 = sgd_step(&w1, &loss.grad(&w1v).unwrap(), lr);
        w2 = sgd_step(&w2, &loss.grad(&w2v).unwrap(), lr);
        wout = sgd_step(&wout, &loss.grad(&woutv).unwrap(), lr);
        g1 = sgd_step(&g1, &loss.grad(&g1v).unwrap(), lr);
        g2 = sgd_step(&g2, &loss.grad(&g2v).unwrap(), lr);
    }

    let first = first_loss.unwrap();
    assert!(
        last_loss.is_finite() && last_loss < first * 0.5,
        "multi-head decoder block didn't learn: loss {first} → {last_loss}"
    );
}

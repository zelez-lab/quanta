//! Key-based dropout — determinism, the host-reference mask, unbiasedness,
//! the same-mask backward (T9232 in the flesh), the rate edges, and the
//! key-threading `apply_train` path through a real stack.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::dropout::{Dropout, dropout_var, keep_mask_host};
use quanta_nn::layer::{Key, Layer, Linear, ParamTree};

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

fn arr(gpu: &quanta::Gpu, v: &[f32], shape: &[usize]) -> Array<f32> {
    Array::from_slice(gpu, v, shape).unwrap()
}

const RATE: f32 = 0.3;
const INV_KEEP: f64 = 1.0 / 0.7;

#[test]
fn same_key_same_mask_different_key_different_mask() {
    let gpu = gpu();
    let x: Vec<f32> = (0..512).map(|i| i as f32 * 0.25 + 1.0).collect();

    let run = |key: Key| -> Vec<f32> {
        let tape = Tape::<f32>::new();
        let xv = tape.var(arr(&gpu, &x, &[512]));
        let y = dropout_var(&tape, &xv, RATE, key).unwrap();
        y.value().to_vec().unwrap()
    };

    let a = run(Key::new(42));
    let b = run(Key::new(42));
    let c = run(Key::new(43));
    assert_eq!(a, b, "same key must reproduce the identical mask");
    assert_ne!(a, c, "different keys must give different masks");
}

#[test]
fn kernel_mask_matches_host_reference() {
    let gpu = gpu();
    let n = 1024usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).sin() + 2.0).collect();
    let key = Key::new(7);

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[n]));
    let y = dropout_var(&tape, &xv, RATE, key).unwrap();
    let out = y.value().to_vec().unwrap();

    let mask = keep_mask_host(key, RATE, n);
    let kept = mask.iter().filter(|&&k| k).count();
    assert!(
        kept > 0 && kept < n,
        "a 0.3-rate mask over 1024 elements should be mixed (kept {kept})"
    );
    for i in 0..n {
        if mask[i] {
            let want = (x[i] as f64 * INV_KEEP) as f32;
            assert!(
                (out[i] - want).abs() <= want.abs() * 1e-6 + 1e-6,
                "kept element {i}: {} vs {}",
                out[i],
                want
            );
        } else {
            assert_eq!(out[i], 0.0, "dropped element {i} must be exactly zero");
        }
    }
}

#[test]
fn unbiased_in_expectation() {
    let gpu = gpu();
    let n = 200_000usize;
    let ones = vec![1.0f32; n];

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &ones, &[n]));
    let y = dropout_var(&tape, &xv, RATE, Key::new(1234)).unwrap();
    let out = y.value().to_vec().unwrap();

    let mean = out.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    // T9231 says exactly 1 in expectation; the sample mean over 200k
    // Bernoulli draws has σ ≈ √(r/(1−r)/n) ≈ 0.0015 — 0.01 is > 6σ.
    assert!(
        (mean - 1.0).abs() < 0.01,
        "inverted dropout must be unbiased: sample mean {mean}"
    );
}

#[test]
fn backward_is_the_same_masked_scaling() {
    let gpu = gpu();
    let n = 256usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    let key = Key::new(99);

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[n]));
    let y = dropout_var(&tape, &xv, RATE, key).unwrap();
    let loss = y.sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();

    // d(sum ∘ dropout)/dx = mask · inv_keep — the mask regenerated from
    // the SAME key (T9232: the VJP is the forward map).
    let mask = keep_mask_host(key, RATE, n);
    for i in 0..n {
        let want = if mask[i] { INV_KEEP as f32 } else { 0.0 };
        assert!(
            (dx[i] - want).abs() <= 1e-6,
            "grad {i}: {} vs {}",
            dx[i],
            want
        );
    }
}

#[test]
fn rate_edges_zero_identity_one_zeros() {
    let gpu = gpu();
    let x: Vec<f32> = (0..64).map(|i| i as f32 + 1.0).collect();

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[64]));
    let y0 = dropout_var(&tape, &xv, 0.0, Key::new(5)).unwrap();
    assert_eq!(y0.value().to_vec().unwrap(), x, "rate 0 is the identity");

    let y1 = dropout_var(&tape, &xv, 1.0, Key::new(5)).unwrap();
    let out1 = y1.value().to_vec().unwrap();
    assert!(out1.iter().all(|&v| v == 0.0), "rate 1 zeroes everything");
    let loss = y1.sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    assert!(
        dx.iter().all(|&v| v == 0.0),
        "rate 1 has an all-zero gradient"
    );

    assert!(dropout_var(&tape, &xv, -0.1, Key::new(5)).is_err());
    assert!(dropout_var(&tape, &xv, 1.5, Key::new(5)).is_err());
}

#[test]
fn eval_apply_is_identity_train_masks() {
    let gpu = gpu();
    let x: Vec<f32> = (0..128).map(|i| (i as f32 * 0.07).cos()).collect();
    let layer = Dropout { rate: 0.5 };

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[128]));

    let y_eval = Layer::<f32>::apply(&layer, &tape, &(), &xv).unwrap();
    assert_eq!(
        y_eval.value().to_vec().unwrap(),
        x,
        "eval apply is the identity — no mode flag, no rescale"
    );

    let (y_tr, _k) = layer.apply_train(&tape, &(), &xv, Key::new(11)).unwrap();
    let out = y_tr.value().to_vec().unwrap();
    assert!(
        out.iter().filter(|&&v| v == 0.0).count() > 16,
        "training apply must actually drop elements at rate 0.5"
    );
}

#[test]
fn stack_threads_the_key_deterministically() {
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 8,
            out_dim: 8,
            bias: true,
        },
        Dropout { rate: 0.4 },
        Linear {
            in_dim: 8,
            out_dim: 4,
            bias: false,
        },
    );
    let params = Layer::<f32>::init(&stack, &gpu, Key::new(3)).unwrap();

    let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.11).sin()).collect();
    let run = |key: Key| -> Vec<f32> {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let xv = tape.var(arr(&gpu, &x, &[4, 8]));
        let (y, _rest) = stack.apply_train(&tape, &vars, &xv, key).unwrap();
        assert_eq!(y.value().shape(), [4, 4]);
        y.value().to_vec().unwrap()
    };

    let a = run(Key::new(21));
    let b = run(Key::new(21));
    let c = run(Key::new(22));
    assert_eq!(a, b, "one key, one mask — through the whole stack");
    assert_ne!(a, c, "a different key must change the training forward");
}

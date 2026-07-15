//! Losses — the fused stable cross-entropy against the composed oracle and
//! an f64 host reference; the theorem properties made empirical (T9228
//! nonnegativity at extreme logits, T9229 stable-vs-textbook BCE equality
//! and the ±100-logit stability it buys, T9230 gradient continuity across
//! the Huber knee); and a 3-class training run through the fused stack.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::activation::Gelu;
use quanta_nn::layer::{Key, Layer, Linear, ParamTree};
use quanta_nn::loss::{
    Reduction, bce_loss, bce_with_logits_loss, cross_entropy_var, huber_loss, l1_loss, mse_loss,
};
use quanta_nn::optim::Adam;

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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// f64 host cross-entropy: mean of `lse(row) − row[label]`, plus dlogits
/// for the mean reduction.
fn host_ce(x: &[f32], labels: &[u32], n: usize, c: usize) -> (f64, Vec<f32>) {
    let mut total = 0.0f64;
    let mut dx = vec![0.0f32; n * c];
    for r in 0..n {
        let row = &x[r * c..(r + 1) * c];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let l: f64 = row.iter().map(|&v| ((v as f64) - m).exp()).sum();
        let lse = m + l.ln();
        total += lse - row[labels[r] as usize] as f64;
        for j in 0..c {
            let p = ((row[j] as f64 - m).exp() / l) as f32;
            let onehot = if j as u32 == labels[r] { 1.0 } else { 0.0 };
            dx[r * c + j] = (p - onehot) / n as f32;
        }
    }
    (total / n as f64, dx)
}

#[test]
fn cross_entropy_matches_composed_and_host() {
    let gpu = gpu();
    for &(n, c, seed) in &[(6usize, 5usize, 11u64), (1, 9, 12), (8, 2, 13)] {
        let x: Vec<f32> = fill(n * c, seed).iter().map(|v| v * 3.0).collect();
        let labels: Vec<u32> = (0..n).map(|i| (i * 7 % c) as u32).collect();

        // Fused path, mean reduction.
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu, &x, &[n, c]).unwrap());
        let loss = cross_entropy_var(&tape, &xv, &labels, Reduction::Mean).unwrap();
        let f_val = loss.value().to_vec().unwrap()[0];
        let f_dx = loss.grad(&xv).unwrap().to_vec().unwrap();

        // Composed oracle (mean semantics).
        let tape2: Tape<f32> = Tape::new();
        let xv2 = tape2.var(Array::from_slice(&gpu, &x, &[n, c]).unwrap());
        let labels_arr = Array::from_slice(&gpu, &labels, &[n]).unwrap();
        let loss2 = xv2.cross_entropy(&labels_arr).unwrap();
        let c_val = loss2.value().to_vec().unwrap()[0];
        let c_dx = loss2.grad(&xv2).unwrap().to_vec().unwrap();

        // f64 host.
        let (h_val, h_dx) = host_ce(&x, &labels, n, c);

        assert!(
            (f_val - c_val).abs() < 1e-5,
            "ce fused vs composed value ({n}x{c})"
        );
        assert!(
            (f_val as f64 - h_val).abs() < 1e-5,
            "ce fused vs host value ({n}x{c})"
        );
        assert!(
            max_abs_diff(&f_dx, &c_dx) < 1e-5,
            "ce dlogits fused vs composed ({n}x{c})"
        );
        assert!(
            max_abs_diff(&f_dx, &h_dx) < 1e-5,
            "ce dlogits fused vs host ({n}x{c})"
        );

        // Sum reduction = n × mean.
        let tape3: Tape<f32> = Tape::new();
        let xv3 = tape3.var(Array::from_slice(&gpu, &x, &[n, c]).unwrap());
        let loss3 = cross_entropy_var(&tape3, &xv3, &labels, Reduction::Sum).unwrap();
        let s_val = loss3.value().to_vec().unwrap()[0];
        assert!(
            (s_val - f_val * n as f32).abs() < 1e-4 * n as f32,
            "sum = n·mean"
        );
    }
}

#[test]
fn cross_entropy_nonneg_and_survives_extreme_logits() {
    // T9228 empirical: loss ≥ 0 always; ±1e4 logits stay finite; a
    // dominant correct logit drives the loss to ~0.
    let gpu = gpu();
    let (n, c) = (4usize, 3usize);
    let mut x = vec![0.0f32; n * c];
    for (i, v) in fill(n * c, 55).iter().enumerate() {
        x[i] = v * 1.0e4;
    }
    let labels: Vec<u32> = vec![0, 1, 2, 0];

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[n, c]).unwrap());
    let loss = cross_entropy_var(&tape, &xv, &labels, Reduction::Mean).unwrap();
    let v = loss.value().to_vec().unwrap()[0];
    assert!(v.is_finite(), "finite at 1e4 logits");
    assert!(v >= -1e-4, "nonnegative (T9228), got {v}");
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    assert!(dx.iter().all(|d| d.is_finite()), "finite gradient");

    // Dominant correct logit → near-zero loss.
    let mut easy = vec![0.0f32; c];
    easy[1] = 50.0;
    let tape2: Tape<f32> = Tape::new();
    let xv2 = tape2.var(Array::from_slice(&gpu, &easy, &[1, c]).unwrap());
    let l2 = cross_entropy_var(&tape2, &xv2, &[1], Reduction::Mean).unwrap();
    assert!(
        l2.value().to_vec().unwrap()[0] < 1e-5,
        "confident & correct → ~0"
    );
}

#[test]
fn bce_with_logits_matches_textbook_and_survives_saturation() {
    let gpu = gpu();
    // T9229 empirical: at moderate logits the stable spelling equals the
    // textbook composition through σ.
    let x = fill(24, 61);
    let y: Vec<f32> = fill(24, 62)
        .iter()
        .map(|v| if *v > 0.0 { 1.0 } else { 0.0 })
        .collect();

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[24]).unwrap());
    let yv = tape.var(Array::from_slice(&gpu, &y, &[24]).unwrap());
    let stable = bce_with_logits_loss(&tape, &xv, &yv, Reduction::Mean).unwrap();
    let sv = stable.value().to_vec().unwrap()[0];

    let tape2: Tape<f32> = Tape::new();
    let xv2 = tape2.var(Array::from_slice(&gpu, &x, &[24]).unwrap());
    let yv2 = tape2.var(Array::from_slice(&gpu, &y, &[24]).unwrap());
    let probs = xv2.sigmoid().unwrap();
    let textbook = bce_loss(&tape2, &probs, &yv2, Reduction::Mean).unwrap();
    let tv = textbook.value().to_vec().unwrap()[0];
    assert!((sv - tv).abs() < 1e-5, "stable {sv} vs textbook {tv}");

    // Gradients agree too.
    let sg = stable.grad(&xv).unwrap().to_vec().unwrap();
    let tg = textbook.grad(&xv2).unwrap().to_vec().unwrap();
    assert!(
        max_abs_diff(&sg, &tg) < 1e-5,
        "stable vs textbook gradients"
    );

    // Saturation: the stable form is exact where σ would round to 0 or 1.
    let ext = [100.0f32, -100.0, 100.0, -100.0];
    let yext = [1.0f32, 0.0, 0.0, 1.0]; // two confident-right, two confident-wrong
    let tape3: Tape<f32> = Tape::new();
    let xv3 = tape3.var(Array::from_slice(&gpu, &ext, &[4]).unwrap());
    let yv3 = tape3.var(Array::from_slice(&gpu, &yext, &[4]).unwrap());
    let l3 = bce_with_logits_loss(&tape3, &xv3, &yv3, Reduction::Sum).unwrap();
    let v3 = l3.value().to_vec().unwrap()[0];
    // Right pair contributes ~0, wrong pair contributes ~|x| each.
    assert!(v3.is_finite(), "finite at ±100 logits");
    assert!((v3 - 200.0).abs() < 1e-2, "sum ≈ 200, got {v3}");
    let g3 = l3.grad(&xv3).unwrap().to_vec().unwrap();
    assert!(g3.iter().all(|g| g.is_finite()), "finite gradients at ±100");
}

#[test]
fn mse_l1_huber_match_host_and_huber_grad_is_continuous() {
    let gpu = gpu();
    let n = 12usize;
    let p = fill(n, 71);
    let t = fill(n, 72);

    let losses: Vec<(f32, Vec<f32>)> = [0u8, 1, 2]
        .iter()
        .map(|&kind| {
            let tape: Tape<f32> = Tape::new();
            let pv = tape.var(Array::from_slice(&gpu, &p, &[n]).unwrap());
            let tv = tape.var(Array::from_slice(&gpu, &t, &[n]).unwrap());
            let loss = match kind {
                0 => mse_loss(&tape, &pv, &tv, Reduction::Mean).unwrap(),
                1 => l1_loss(&tape, &pv, &tv, Reduction::Mean).unwrap(),
                _ => huber_loss(&tape, &pv, &tv, 0.5, Reduction::Mean).unwrap(),
            };
            (
                loss.value().to_vec().unwrap()[0],
                loss.grad(&pv).unwrap().to_vec().unwrap(),
            )
        })
        .collect();

    // f64 host references.
    let (mut hm, mut hl, mut hh) = (0.0f64, 0.0f64, 0.0f64);
    let (mut gm, mut gl, mut gh) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
    let delta = 0.5f64;
    for i in 0..n {
        let z = p[i] as f64 - t[i] as f64;
        hm += z * z;
        gm[i] = (2.0 * z / n as f64) as f32;
        hl += z.abs();
        gl[i] = (z.signum() / n as f64) as f32;
        if z.abs() <= delta {
            hh += z * z / 2.0;
            gh[i] = (z / n as f64) as f32;
        } else {
            hh += delta * (z.abs() - delta / 2.0);
            gh[i] = (delta * z.signum() / n as f64) as f32;
        }
    }
    let refs = [
        (hm / n as f64, gm),
        (hl / n as f64, gl),
        (hh / n as f64, gh),
    ];
    for (k, ((got_v, got_g), (want_v, want_g))) in losses.iter().zip(&refs).enumerate() {
        assert!(
            (*got_v as f64 - want_v).abs() < 1e-5,
            "loss {k}: value {got_v} vs host {want_v}"
        );
        assert!(
            max_abs_diff(got_g, want_g) < 1e-5,
            "loss {k}: gradient vs host"
        );
    }

    // T9230 empirical: the Huber gradient is continuous across the knee.
    let delta = 0.5f32;
    let grad_at = |z: f32| -> f32 {
        let tape: Tape<f32> = Tape::new();
        let pv = tape.var(Array::from_slice(&gpu, &[z], &[1]).unwrap());
        let tv = tape.var(Array::from_slice(&gpu, &[0.0f32], &[1]).unwrap());
        let loss = huber_loss(&tape, &pv, &tv, delta as f64, Reduction::Sum).unwrap();
        loss.grad(&pv).unwrap().to_vec().unwrap()[0]
    };
    let (below, above) = (grad_at(delta - 1e-3), grad_at(delta + 1e-3));
    assert!(
        (below - above).abs() < 3e-3,
        "gradient continuous at the knee: {below} vs {above}"
    );
    assert!((grad_at(2.0) - delta).abs() < 1e-6, "clamped to δ beyond");
}

#[test]
fn classification_stack_trains_with_fused_ce() {
    // 3-class toy: class = argmax over three fixed linear scores of x.
    // (Linear → Gelu → Linear) + fused CE + fused Adam, end to end.
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 2,
            out_dim: 16,
            bias: true,
        },
        Gelu,
        Linear {
            in_dim: 16,
            out_dim: 3,
            bias: true,
        },
    );
    let mut params = Layer::<f32>::init(&stack, &gpu, Key::new(5)).unwrap();

    let n = 24usize;
    let xs = fill(n * 2, 81);
    let labels: Vec<u32> = (0..n)
        .map(|r| {
            let (a, b) = (xs[r * 2], xs[r * 2 + 1]);
            let scores = [a + b, a - b, -a];
            let mut best = 0;
            for k in 1..3 {
                if scores[k] > scores[best] {
                    best = k;
                }
            }
            best as u32
        })
        .collect();

    let opt = Adam::new(0.02);
    let mut state = opt.init(&params).unwrap();
    let mut first = None;
    let mut last = 0.0f32;
    for _ in 0..60 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, &xs, &[n, 2]).unwrap());
        let logits = stack.apply(&tape, &vars, &xv).unwrap();
        let loss = cross_entropy_var(&tape, &logits, &labels, Reduction::Mean).unwrap();
        let lval = loss.value().to_vec().unwrap()[0];
        if first.is_none() {
            first = Some(lval);
        }
        last = lval;
        let grads = params.grads_from(&vars, &loss).unwrap();
        let (np, ns) = opt.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
    }
    let first = first.unwrap();
    assert!(
        last < first * 0.5 && last < 0.7,
        "CE training must cut the loss: first {first}, last {last}"
    );
}

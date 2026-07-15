//! Fused optimizers — differential tests against an f64 host reference,
//! the theorem properties made empirical (T9219 momentum geometry, T9220
//! bias-correction exactness, T9222 step-size scale invariance), schedule
//! shapes, clipping, and the tree-level training idiom end to end.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::layer::{Key, Layer, LayerNorm, Linear, ParamTree};
use quanta_nn::optim::{Adam, Schedule, Sgd, clip_grad_norm, clip_grad_value};

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

fn max_abs_diff(a: &[f32], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y).abs())
        .fold(0.0, f64::max)
}

/// f64 host mirror of the fused SGD kernel semantics.
fn host_sgd(p: &mut [f64], v: &mut [f64], g: &[f64], lr: f64, mu: f64, wd: f64, nesterov: bool) {
    for i in 0..p.len() {
        let geff = g[i] + wd * p[i];
        v[i] = mu * v[i] + geff;
        let dir = if nesterov { geff + mu * v[i] } else { v[i] };
        p[i] -= lr * dir;
    }
}

/// f64 host mirror of the fused Adam/AdamW kernel semantics.
#[allow(clippy::too_many_arguments)]
fn host_adam(
    p: &mut [f64],
    m: &mut [f64],
    v: &mut [f64],
    g: &[f64],
    t: u64,
    lr: f64,
    b1: f64,
    b2: f64,
    eps: f64,
    wd: f64,
    decoupled: bool,
) {
    let (wd_c, wd_d) = if decoupled { (0.0, wd) } else { (wd, 0.0) };
    let bc1 = 1.0 - b1.powi(t as i32);
    let bc2 = 1.0 - b2.powi(t as i32);
    for i in 0..p.len() {
        let geff = g[i] + wd_c * p[i];
        m[i] = b1 * m[i] + (1.0 - b1) * geff;
        v[i] = b2 * v[i] + (1.0 - b2) * geff * geff;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        p[i] -= lr * (mhat / (vhat.sqrt() + eps) + wd_d * p[i]);
    }
}

#[test]
fn sgd_matches_f64_host_across_configs() {
    let gpu = gpu();
    let n = 64usize;
    let configs = [
        Sgd::new(0.05),
        Sgd::momentum(0.05, 0.9),
        Sgd {
            nesterov: true,
            ..Sgd::momentum(0.05, 0.9)
        },
        Sgd {
            weight_decay: 0.01,
            ..Sgd::momentum(0.05, 0.9)
        },
    ];
    for (ci, opt) in configs.iter().enumerate() {
        let p0 = fill(n, 100 + ci as u64);
        let mut params = Array::from_slice(&gpu, &p0, &[n]).unwrap();
        let mut state = opt.init(&params).unwrap();
        let mut hp: Vec<f64> = p0.iter().map(|&x| x as f64).collect();
        let mut hv = vec![0.0f64; n];
        for step in 0..20 {
            let gh = fill(n, 500 + ci as u64 * 100 + step);
            let grads = Array::from_slice(&gpu, &gh, &[n]).unwrap();
            let (np, ns) = opt.step(&params, &grads, state).unwrap();
            params = np;
            state = ns;
            let gh64: Vec<f64> = gh.iter().map(|&x| x as f64).collect();
            host_sgd(
                &mut hp,
                &mut hv,
                &gh64,
                opt.lr as f64,
                opt.momentum as f64,
                opt.weight_decay as f64,
                opt.nesterov,
            );
        }
        let got = params.to_vec().unwrap();
        assert!(
            max_abs_diff(&got, &hp) < 1e-4,
            "sgd config {ci}: fused vs f64 host after 20 steps"
        );
        let vel = state.velocity[0].to_vec().unwrap();
        assert!(
            max_abs_diff(&vel, &hv) < 1e-4,
            "sgd config {ci}: velocity vs f64 host"
        );
    }
}

#[test]
fn adam_matches_f64_host_across_configs() {
    let gpu = gpu();
    let n = 64usize;
    let configs = [
        Adam::new(0.01),
        Adam::adamw(0.01, 0.05),
        Adam {
            weight_decay: 0.05,
            ..Adam::new(0.01)
        }, // coupled L2
    ];
    for (ci, opt) in configs.iter().enumerate() {
        let p0 = fill(n, 200 + ci as u64);
        let mut params = Array::from_slice(&gpu, &p0, &[n]).unwrap();
        let mut state = opt.init(&params).unwrap();
        let mut hp: Vec<f64> = p0.iter().map(|&x| x as f64).collect();
        let mut hm = vec![0.0f64; n];
        let mut hv = vec![0.0f64; n];
        for step in 0..20 {
            let gh = fill(n, 900 + ci as u64 * 100 + step);
            let grads = Array::from_slice(&gpu, &gh, &[n]).unwrap();
            let (np, ns) = opt.step(&params, &grads, state).unwrap();
            params = np;
            state = ns;
            let gh64: Vec<f64> = gh.iter().map(|&x| x as f64).collect();
            host_adam(
                &mut hp,
                &mut hm,
                &mut hv,
                &gh64,
                step + 1,
                opt.lr as f64,
                opt.beta1 as f64,
                opt.beta2 as f64,
                opt.eps as f64,
                opt.weight_decay as f64,
                opt.decoupled,
            );
        }
        let got = params.to_vec().unwrap();
        assert!(
            max_abs_diff(&got, &hp) < 1e-4,
            "adam config {ci}: fused vs f64 host after 20 steps"
        );
        assert_eq!(state.t, 20);
    }
}

#[test]
fn momentum_matches_geometric_closed_form() {
    // T9219 empirical: constant gradient g, v_t = g·(1−μᵗ)/(1−μ).
    let gpu = gpu();
    let (g, mu, lr) = (0.7f32, 0.9f32, 0.1f32);
    let opt = Sgd::momentum(lr, mu);
    let mut params = Array::from_slice(&gpu, &[1.0f32], &[1]).unwrap();
    let mut state = opt.init(&params).unwrap();
    let grads = Array::from_slice(&gpu, &[g], &[1]).unwrap();
    for t in 1..=15u32 {
        let (np, ns) = opt.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
        let vt = state.velocity[0].to_vec().unwrap()[0];
        let closed = g * (1.0 - mu.powi(t as i32)) / (1.0 - mu);
        assert!(
            (vt - closed).abs() < 1e-5,
            "step {t}: velocity {vt} vs closed form {closed}"
        );
    }
}

#[test]
fn adam_bias_correction_exact_and_step_is_lr() {
    // T9220 + T9222 empirical: constant gradient, ε = 0 → the very first
    // step (and every later one) moves each parameter by exactly lr·sign(g).
    let gpu = gpu();
    let opt = Adam {
        eps: 0.0,
        ..Adam::new(0.01)
    };
    let p0 = [1.0f32, -2.0, 0.5, 3.0];
    let g = [0.3f32, -0.001, 700.0, 0.0002]; // wildly different scales
    let mut params = Array::from_slice(&gpu, &p0, &[4]).unwrap();
    let mut state = opt.init(&params).unwrap();
    let grads = Array::from_slice(&gpu, &g, &[4]).unwrap();
    let mut prev = p0.to_vec();
    for step in 0..5 {
        let (np, ns) = opt.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
        let cur = params.to_vec().unwrap();
        for i in 0..4 {
            let delta = cur[i] - prev[i];
            let expect = -opt.lr * g[i].signum();
            assert!(
                (delta - expect).abs() < 1e-6,
                "step {step}, elem {i}: Δp = {delta}, want {expect} (scale invariance)"
            );
        }
        prev = cur;
    }
}

#[test]
fn schedule_shapes() {
    let c = Schedule::Constant { lr: 0.1 };
    assert_eq!(c.lr(0), 0.1);
    assert_eq!(c.lr(10_000), 0.1);

    let s = Schedule::Step {
        lr: 1.0,
        gamma: 0.5,
        every: 10,
    };
    assert_eq!(s.lr(0), 1.0);
    assert_eq!(s.lr(9), 1.0);
    assert_eq!(s.lr(10), 0.5);
    assert_eq!(s.lr(25), 0.25);

    let w = Schedule::LinearWarmup { lr: 1.0, warmup: 4 };
    assert!((w.lr(0) - 0.25).abs() < 1e-6);
    assert!((w.lr(3) - 1.0).abs() < 1e-6);
    assert_eq!(w.lr(100), 1.0);

    let cos = Schedule::Cosine {
        base: 1.0,
        min_lr: 0.1,
        warmup: 5,
        total: 105,
    };
    assert!((cos.lr(4) - 1.0).abs() < 1e-6, "ramp reaches base");
    assert!((cos.lr(5) - 1.0).abs() < 1e-6, "cosine starts at base");
    assert!((cos.lr(55) - 0.55).abs() < 1e-5, "midpoint is the average");
    assert!((cos.lr(104) - 0.1).abs() < 1e-3, "tail approaches min");
    assert_eq!(cos.lr(105), 0.1);
    assert_eq!(cos.lr(9999), 0.1);
    // Monotone non-increasing after warmup.
    let mut last = cos.lr(5);
    for t in 6..105 {
        let cur = cos.lr(t);
        assert!(cur <= last + 1e-7, "cosine must not increase (t={t})");
        last = cur;
    }
}

#[test]
fn clipping_by_value_and_by_global_norm() {
    let gpu = gpu();
    let a = Array::from_slice(&gpu, &[3.0f32, -0.5, 10.0], &[3]).unwrap();

    let clipped = clip_grad_value(&a, 1.0).unwrap();
    assert_eq!(clipped.to_vec().unwrap(), vec![1.0, -0.5, 1.0]);

    // Global norm over the whole tree: (3,4) has norm 5.
    let tree = Array::from_slice(&gpu, &[3.0f32, 4.0], &[2]).unwrap();
    let (scaled, norm) = clip_grad_norm(&tree, 1.0).unwrap();
    assert!((norm - 5.0).abs() < 1e-6);
    let sv = scaled.to_vec().unwrap();
    assert!((sv[0] - 0.6).abs() < 1e-6 && (sv[1] - 0.8).abs() < 1e-6);
    let n2: f32 = sv.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((n2 - 1.0).abs() < 1e-6, "post-clip norm is max_norm");

    // Under the threshold: untouched, norm still reported.
    let (kept, norm2) = clip_grad_norm(&tree, 100.0).unwrap();
    assert!((norm2 - 5.0).abs() < 1e-6);
    assert_eq!(kept.to_vec().unwrap(), tree.to_vec().unwrap());
}

#[test]
fn stack_trains_with_adamw_and_cosine() {
    // The tree-level idiom end to end: the layer stack from the Layer
    // slice, stepped by fused AdamW under a cosine schedule with norm
    // clipping — the loop every later example builds on.
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 4,
            out_dim: 8,
            bias: true,
        },
        LayerNorm { dim: 8, eps: 1e-5 },
        Linear {
            in_dim: 8,
            out_dim: 2,
            bias: true,
        },
    );
    let mut params = Layer::<f32>::init(&stack, &gpu, Key::new(3)).unwrap();

    let n = 16usize;
    let xs: Vec<f32> = (0..n * 4)
        .map(|i| ((i * 37 % 17) as f32 / 8.5) - 1.0)
        .collect();
    let ys: Vec<f32> = (0..n)
        .flat_map(|r| {
            let x = &xs[r * 4..r * 4 + 4];
            [
                0.5 * x[0] - 1.2 * x[1] + 0.3 * x[2],
                0.8 * x[3] + 0.1 * x[0],
            ]
        })
        .collect();

    let sched = Schedule::Cosine {
        base: 0.02,
        min_lr: 0.002,
        warmup: 5,
        total: 60,
    };
    let opt = Adam::adamw(0.02, 0.01);
    let mut state = opt.init(&params).unwrap();
    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for t in 0..60u64 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, &xs, &[n, 4]).unwrap());
        let yv = tape.var(Array::from_slice(&gpu, &ys, &[n, 2]).unwrap());
        let pred = stack.apply(&tape, &vars, &xv).unwrap();
        let diff = pred.sub(&yv).unwrap();
        let loss = diff.mul(&diff).unwrap().sum().unwrap();
        let lval = loss.value().to_vec().unwrap()[0] / (n as f32);
        if first_loss.is_none() {
            first_loss = Some(lval);
        }
        last_loss = lval;

        let grads = params.grads_from(&vars, &loss).unwrap();
        let (grads, _norm) = clip_grad_norm(&grads, 10.0).unwrap();
        let opt_t = Adam {
            lr: sched.lr(t),
            ..opt
        };
        let (np, ns) = opt_t.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
    }

    let first = first_loss.unwrap();
    assert!(
        last_loss < first * 0.2,
        "AdamW training must reduce the loss substantially: first {first}, last {last_loss}"
    );
    assert!(last_loss.is_finite());
}

//! Fused activations — differential tests against the composed tape ops
//! (per-op-proven VJPs) and f64 host references; the theorem properties
//! made empirical (T9223 stability at extreme logits, unit row sums,
//! GeLU/SwiGLU saturation); and zero-param layers inside tuple stacks.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::activation::{Gelu, SwiGlu, gelu_var, log_softmax_var, softmax_var, swiglu_var};
use quanta_nn::layer::{Key, Layer, Linear, ParamTree};
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

/// Run `fused(x)` and the composed oracle through weighted-sum losses and
/// compare forward values and input gradients.
fn compare_paths(
    gpu: &quanta::Gpu,
    x: &[f32],
    g: &[f32],
    shape: &[usize],
    fused: impl Fn(&Tape<f32>, &quanta_autograd::Var<f32>) -> quanta_autograd::Var<f32>,
    composed: impl Fn(&Tape<f32>, &quanta_autograd::Var<f32>) -> quanta_autograd::Var<f32>,
    tol: f32,
    label: &str,
) {
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(gpu, x, shape).unwrap());
    let out = fused(&tape, &xv);
    let f_out = out.value().to_vec().unwrap();
    let w = tape.var(Array::from_slice(gpu, g, out.value().shape()).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let f_dx = loss.grad(&xv).unwrap().to_vec().unwrap();

    let tape2: Tape<f32> = Tape::new();
    let xv2 = tape2.var(Array::from_slice(gpu, x, shape).unwrap());
    let out2 = composed(&tape2, &xv2);
    let c_out = out2.value().to_vec().unwrap();
    let w2 = tape2.var(Array::from_slice(gpu, g, out2.value().shape()).unwrap());
    let loss2 = out2.mul(&w2).unwrap().sum().unwrap();
    let c_dx = loss2.grad(&xv2).unwrap().to_vec().unwrap();

    assert!(
        max_abs_diff(&f_out, &c_out) < tol,
        "{label}: fwd fused vs composed ({})",
        max_abs_diff(&f_out, &c_out)
    );
    assert!(
        max_abs_diff(&f_dx, &c_dx) < tol,
        "{label}: dx fused vs composed ({})",
        max_abs_diff(&f_dx, &c_dx)
    );
}

#[test]
fn softmax_and_log_softmax_match_composed() {
    let gpu = gpu();
    for &(n, c, seed) in &[(4usize, 8usize, 11u64), (1, 16, 12), (7, 3, 13)] {
        let x: Vec<f32> = fill(n * c, seed).iter().map(|v| v * 4.0).collect();
        let g = fill(n * c, seed + 50);
        compare_paths(
            &gpu,
            &x,
            &g,
            &[n, c],
            |t, v| softmax_var(t, v).unwrap(),
            |_t, v| v.softmax().unwrap(),
            1e-5,
            "softmax",
        );
        compare_paths(
            &gpu,
            &x,
            &g,
            &[n, c],
            |t, v| log_softmax_var(t, v).unwrap(),
            |_t, v| v.log_softmax().unwrap(),
            1e-5,
            "log_softmax",
        );
    }
}

#[test]
fn softmax_rows_sum_to_one_and_survive_extreme_logits() {
    // T9223 empirical: max-subtraction is exact, so logits around ±1e4
    // must neither overflow nor NaN, and shifting a row leaves the
    // softmax unchanged.
    let gpu = gpu();
    let (n, c) = (3usize, 5usize);
    let base = fill(n * c, 77);
    let x: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, v)| v + if i / c == 1 { 1.0e4 } else { 0.0 })
        .collect();

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[n, c]).unwrap());
    let p = softmax_var(&tape, &xv).unwrap().value().to_vec().unwrap();
    assert!(p.iter().all(|v| v.is_finite()), "no NaN/inf at 1e4 logits");
    for row in 0..n {
        let s: f32 = p[row * c..(row + 1) * c].iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "row {row} sums to {s}");
    }
    // The shifted row equals the unshifted softmax of the same row. The
    // tolerance is input-quantization-bound, not kernel-bound: at 1e4 the
    // f32 ulp is ~1e-3, so `v + 1e4` perturbs the LOGITS by up to ~5e-4
    // before the (exact, T9223) max-subtraction ever runs.
    let tape2: Tape<f32> = Tape::new();
    let xv2 = tape2.var(Array::from_slice(&gpu, &base, &[n, c]).unwrap());
    let p0 = softmax_var(&tape2, &xv2).unwrap().value().to_vec().unwrap();
    assert!(
        max_abs_diff(&p[c..2 * c], &p0[c..2 * c]) < 2e-3,
        "shift invariance (up to input quantization)"
    );
}

#[test]
fn gelu_matches_composed_and_saturates() {
    let gpu = gpu();
    let x: Vec<f32> = fill(64, 21).iter().map(|v| v * 3.0).collect();
    let g = fill(64, 22);
    compare_paths(
        &gpu,
        &x,
        &g,
        &[8, 8],
        |t, v| gelu_var(t, v).unwrap(),
        |_t, v| v.gelu().unwrap(),
        1e-5,
        "gelu",
    );

    // Saturation: gelu(−big) → 0 with zero slope, gelu(+big) → x with
    // unit slope; nothing overflows despite the x³ inside.
    let ext = [-1.0e6f32, -100.0, -20.0, 20.0, 100.0, 1.0e6];
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &ext, &[6]).unwrap());
    let out = gelu_var(&tape, &xv).unwrap();
    let y = out.value().to_vec().unwrap();
    let w = tape.var(Array::from_slice(&gpu, &[1.0f32; 6], &[6]).unwrap());
    let dx = out
        .mul(&w)
        .unwrap()
        .sum()
        .unwrap()
        .grad(&xv)
        .unwrap()
        .to_vec()
        .unwrap();
    for i in 0..3 {
        assert!(y[i].abs() < 1e-3 && dx[i].abs() < 1e-3, "left tail {i}");
    }
    for i in 3..6 {
        assert!(
            (y[i] - ext[i]).abs() < 1e-3 * ext[i].abs().max(1.0),
            "right tail value {i}"
        );
        assert!((dx[i] - 1.0).abs() < 1e-3, "right tail slope {i}");
    }
}

#[test]
fn swiglu_matches_composed_split_oracle() {
    let gpu = gpu();
    for &(n, h, seed) in &[(4usize, 8usize, 31u64), (1, 4, 32), (5, 1, 33)] {
        let x = fill(n * 2 * h, seed);
        let g = fill(n * h, seed + 50);
        compare_paths(
            &gpu,
            &x,
            &g,
            &[n, 2 * h],
            |t, v| swiglu_var(t, v).unwrap(),
            |_t, v| {
                // Composed oracle: transpose → narrow (axis-0 differentiable)
                // → transpose back, then silu(a) ⊙ b.
                let xt = v.transpose(0, 1).unwrap();
                let a = xt.narrow(0, h).unwrap().transpose(0, 1).unwrap();
                let b = xt.narrow(h, h).unwrap().transpose(0, 1).unwrap();
                a.silu().unwrap().mul(&b).unwrap()
            },
            1e-5,
            "swiglu",
        );
    }
}

#[test]
fn gradcheck_swiglu_central_difference() {
    let gpu = gpu();
    let (n, h) = (2usize, 3usize);
    let x = fill(n * 2 * h, 41);
    let g = fill(n * h, 42);

    let loss_of = |xs: &[f32]| -> f64 {
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu, xs, &[n, 2 * h]).unwrap());
        let out = swiglu_var(&tape, &xv).unwrap();
        out.value()
            .to_vec()
            .unwrap()
            .iter()
            .zip(&g)
            .map(|(&o, &w)| o as f64 * w as f64)
            .sum()
    };

    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(Array::from_slice(&gpu, &x, &[n, 2 * h]).unwrap());
    let out = swiglu_var(&tape, &xv).unwrap();
    let w = tape.var(Array::from_slice(&gpu, &g, &[n, h]).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();

    let hstep = 1e-3f32;
    for idx in 0..n * 2 * h {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[idx] += hstep;
        xm[idx] -= hstep;
        let num = (loss_of(&xp) - loss_of(&xm)) / (2.0 * hstep as f64);
        let ana = dx[idx] as f64;
        assert!(
            (num - ana).abs() < 2e-2_f64.max(2e-2 * ana.abs()),
            "gradcheck x[{idx}]: numeric {num} vs analytic {ana}"
        );
    }
}

#[test]
fn activation_layers_stack_and_train() {
    // Zero-param layers in tuple stacks: SwiGlu halves the width (the
    // contract propagates 16 → 8), contributes no leaves, consumes no
    // keys — and the whole stack trains through fused Adam.
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 4,
            out_dim: 16,
            bias: true,
        },
        SwiGlu,
        Linear {
            in_dim: 8,
            out_dim: 2,
            bias: true,
        },
    );
    let mut params = Layer::<f32>::init(&stack, &gpu, Key::new(9)).unwrap();
    assert_eq!(params.flatten().len(), 4, "SwiGlu adds no leaves");

    let n = 16usize;
    let xs: Vec<f32> = (0..n * 4)
        .map(|i| ((i * 29 % 13) as f32 / 6.5) - 1.0)
        .collect();
    let ys: Vec<f32> = (0..n)
        .flat_map(|r| {
            let x = &xs[r * 4..r * 4 + 4];
            [0.6 * x[0] - 0.9 * x[1], 0.4 * x[2] + 0.7 * x[3]]
        })
        .collect();

    let opt = Adam::new(0.01);
    let mut state = opt.init(&params).unwrap();
    let mut first = None;
    let mut last = 0.0f32;
    for _ in 0..50 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, &xs, &[n, 4]).unwrap());
        let yv = tape.var(Array::from_slice(&gpu, &ys, &[n, 2]).unwrap());
        let pred = stack.apply(&tape, &vars, &xv).unwrap();
        let diff = pred.sub(&yv).unwrap();
        let loss = diff.mul(&diff).unwrap().sum().unwrap();
        let lval = loss.value().to_vec().unwrap()[0] / n as f32;
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
        last < first * 0.3,
        "SwiGLU stack must train: first {first}, last {last}"
    );

    // Gelu variant: width contract 16 → 16 (identity width).
    let stack2 = (
        Linear {
            in_dim: 4,
            out_dim: 16,
            bias: true,
        },
        Gelu,
        Linear {
            in_dim: 16,
            out_dim: 2,
            bias: true,
        },
    );
    let params2 = Layer::<f32>::init(&stack2, &gpu, Key::new(10)).unwrap();
    let tape: Tape<f32> = Tape::new();
    let vars2 = params2.bind(&tape);
    let xv = tape.var(Array::from_slice(&gpu, &xs, &[n, 4]).unwrap());
    let pred = stack2.apply(&tape, &vars2, &xv).unwrap();
    assert_eq!(pred.value().shape(), &[n, 2]);
}

//! GroupNorm, BatchNorm (state-in/state-out, D5), Conv2d, and the pools —
//! host references, the through-the-batch-stats backward invariant, the
//! running-stats EMA, checkpointable BnStats, and NCHW stacking.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::batchnorm::{BatchNorm, BnStats};
use quanta_nn::conv::{AvgPool2d, Conv2d, MaxPool2d};
use quanta_nn::layer::{GroupNorm, Key, Layer, ParamTree};
use quanta_nn::norm::{group_norm_var, layer_norm_var};
use quanta_nn::state::{load_state, save_state};

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

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

// ── GroupNorm ────────────────────────────────────────────────────────────

/// Host GroupNorm reference: per-row, per-group normalize, per-channel affine.
fn group_norm_host(x: &[f32], n: usize, c: usize, g: usize, eps: f32) -> Vec<f32> {
    let cg = c / g;
    let mut out = vec![0.0f32; n * c];
    for row in 0..n {
        for grp in 0..g {
            let seg: Vec<f32> = (0..cg).map(|j| x[row * c + grp * cg + j]).collect();
            let mu = seg.iter().sum::<f32>() / cg as f32;
            let var = seg.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / cg as f32;
            let rstd = 1.0 / (var + eps).sqrt();
            for j in 0..cg {
                out[row * c + grp * cg + j] = (seg[j] - mu) * rstd;
            }
        }
    }
    out
}

#[test]
fn group_norm_matches_host_reference() {
    let gpu = gpu();
    let (n, c, g) = (3usize, 6usize, 3usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.1).collect();
    let beta: Vec<f32> = (0..c).map(|i| i as f32 * 0.2 - 0.3).collect();

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[n, c]));
    let gv = tape.var(arr(&gpu, &gamma, &[c]));
    let bv = tape.var(arr(&gpu, &beta, &[c]));
    let y = group_norm_var(&tape, &xv, &gv, &bv, g, 1e-5).unwrap();
    let out = y.value().to_vec().unwrap();

    let base = group_norm_host(&x, n, c, g, 1e-5);
    let want: Vec<f32> = base
        .iter()
        .enumerate()
        .map(|(i, v)| v * gamma[i % c] + beta[i % c])
        .collect();
    close(&out, &want, 2e-4, "group_norm");
}

#[test]
fn group_norm_of_one_group_is_layer_norm() {
    let gpu = gpu();
    let (n, c) = (4usize, 8usize);
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.21).cos() * 3.0).collect();
    let gamma: Vec<f32> = (0..c).map(|i| 1.0 + i as f32 * 0.05).collect();
    let beta = vec![0.25f32; c];

    let run = |grouped: bool| -> (Vec<f32>, Vec<f32>) {
        let tape = Tape::<f32>::new();
        let xv = tape.var(arr(&gpu, &x, &[n, c]));
        let gv = tape.var(arr(&gpu, &gamma, &[c]));
        let bv = tape.var(arr(&gpu, &beta, &[c]));
        let y = if grouped {
            group_norm_var(&tape, &xv, &gv, &bv, 1, 1e-5).unwrap()
        } else {
            layer_norm_var(&tape, &xv, &gv, &bv, 1e-5).unwrap()
        };
        let loss = y.mul(&y).unwrap().sum().unwrap();
        (
            y.value().to_vec().unwrap(),
            loss.grad(&xv).unwrap().to_vec().unwrap(),
        )
    };

    let (yg, dg) = run(true);
    let (yl, dl) = run(false);
    close(&yg, &yl, 2e-4, "gn(1) values vs layer_norm");
    close(&dg, &dl, 2e-3, "gn(1) grads vs layer_norm");
}

#[test]
fn group_norm_contract_is_loud() {
    let gpu = gpu();
    let layer = GroupNorm {
        dim: 6,
        groups: 4, // 6 % 4 != 0
        eps: 1e-5,
    };
    let params = Layer::<f32>::init(&layer, &gpu, Key::new(1)).unwrap();
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let xv = tape.var(arr(&gpu, &[0.0; 12], &[2, 6]));
    assert!(layer.apply(&tape, &vars, &xv).is_err());
}

// ── BatchNorm ────────────────────────────────────────────────────────────

#[test]
fn batchnorm_train_matches_host_and_updates_running_stats() {
    let gpu = gpu();
    let (n, c) = (4usize, 3usize);
    let bn = BatchNorm {
        dim: c,
        eps: 1e-5,
        momentum: 0.1,
    };
    let x: Vec<f32> = (0..n * c)
        .map(|i| (i as f32 * 0.53).sin() * 2.0 + 1.0)
        .collect();

    let params = bn.init::<f32>(&gpu, Key::new(1)).unwrap();
    let stats0 = bn.init_stats::<f32>(&gpu).unwrap();
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let xv = tape.var(arr(&gpu, &x, &[n, c]));
    let (y, stats1) = bn.apply_train(&tape, &vars, &stats0, &xv).unwrap();

    // Host reference: per-channel batch mean / biased var; γ=1, β=0.
    let mut want = vec![0.0f32; n * c];
    let mut want_mean = vec![0.0f32; c];
    let mut want_var = vec![0.0f32; c];
    for ch in 0..c {
        let col: Vec<f32> = (0..n).map(|r| x[r * c + ch]).collect();
        let mu = col.iter().sum::<f32>() / n as f32;
        let vb = col.iter().map(|v| (v - mu) * (v - mu)).sum::<f32>() / n as f32;
        for r in 0..n {
            want[r * c + ch] = (col[r] - mu) / (vb + 1e-5).sqrt();
        }
        // Running EMA from (0, 1) with momentum 0.1; variance unbiased.
        want_mean[ch] = 0.9 * 0.0 + 0.1 * mu;
        want_var[ch] = 0.9 * 1.0 + 0.1 * vb * (n as f32 / (n as f32 - 1.0));
    }
    close(&y.value().to_vec().unwrap(), &want, 2e-4, "bn train fwd");
    close(&stats1.mean.to_vec().unwrap(), &want_mean, 1e-5, "bn mean");
    close(&stats1.var.to_vec().unwrap(), &want_var, 1e-5, "bn var");
}

/// The through-the-batch-stats backward, tested by an exact invariant:
/// each normalized column sums to zero over the batch, so `sum(y)` is
/// x-independent and its gradient w.r.t. x is EXACTLY zero — a backward
/// that froze the batch stats as constants would report `γ/σ` instead.
#[test]
fn batchnorm_backward_goes_through_the_batch_stats() {
    let gpu = gpu();
    let (n, c) = (5usize, 2usize);
    let bn = BatchNorm {
        dim: c,
        eps: 1e-5,
        momentum: 0.1,
    };
    let x: Vec<f32> = (0..n * c).map(|i| (i as f32 * 0.71).cos() * 3.0).collect();

    let params = bn.init::<f32>(&gpu, Key::new(2)).unwrap();
    let stats = bn.init_stats::<f32>(&gpu).unwrap();
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let xv = tape.var(arr(&gpu, &x, &[n, c]));
    let (y, _) = bn.apply_train(&tape, &vars, &stats, &xv).unwrap();
    let loss = y.sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    for (i, g) in dx.iter().enumerate() {
        assert!(
            g.abs() < 1e-4,
            "d(sum∘bn)/dx[{i}] = {g}, must vanish (stats are differentiated)"
        );
    }
}

#[test]
fn batchnorm_eval_uses_running_stats_and_edges_are_loud() {
    let gpu = gpu();
    let c = 2usize;
    let bn = BatchNorm {
        dim: c,
        eps: 1e-5,
        momentum: 0.1,
    };
    let params = bn.init::<f32>(&gpu, Key::new(3)).unwrap();
    let stats = BnStats {
        mean: arr(&gpu, &[1.0, -2.0], &[c]),
        var: arr(&gpu, &[4.0, 0.25], &[c]),
    };

    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let x = [3.0f32, -1.0, 5.0, -3.0];
    let xv = tape.var(arr(&gpu, &x, &[2, c]));
    let y = bn.apply_eval(&tape, &vars, &stats, &xv).unwrap();
    // (x − mean)/√(var+eps), γ=1, β=0.
    let want = [
        (3.0 - 1.0) / (4.0f32 + 1e-5).sqrt(),
        (-1.0 - -2.0) / (0.25f32 + 1e-5).sqrt(),
        (5.0 - 1.0) / (4.0f32 + 1e-5).sqrt(),
        (-3.0 - -2.0) / (0.25f32 + 1e-5).sqrt(),
    ];
    close(&y.value().to_vec().unwrap(), &want, 1e-5, "bn eval");

    // A batch of one cannot train.
    let x1 = tape.var(arr(&gpu, &[0.0, 0.0], &[1, c]));
    assert!(bn.apply_train(&tape, &vars, &stats, &x1).is_err());
}

#[test]
fn bn_stats_checkpoint_round_trips() {
    let gpu = gpu();
    let stats = BnStats {
        mean: arr(&gpu, &[0.5, -0.25, 3.0], &[3]),
        var: arr(&gpu, &[1.5, 0.75, 2.0], &[3]),
    };
    let names: Vec<String> = stats.named_flatten().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["mean", "var"]);
    let bytes = save_state(&stats).unwrap();
    let restored: BnStats<f32> = load_state(&stats, &gpu, &bytes).unwrap();
    assert_eq!(
        restored.mean.to_vec().unwrap(),
        stats.mean.to_vec().unwrap()
    );
    assert_eq!(restored.var.to_vec().unwrap(), stats.var.to_vec().unwrap());
}

// ── Conv2d + pools ───────────────────────────────────────────────────────

#[test]
fn conv2d_known_case_values_and_weight_grad() {
    let gpu = gpu();
    let conv = Conv2d {
        cin: 1,
        cout: 1,
        kh: 2,
        kw: 2,
        stride: 1,
        pad: 0,
        bias: true,
    };
    // All-ones 3×3 input, all-ones 2×2 kernel → every output is 4 (+bias 0).
    let mut params = Layer::<f32>::init(&conv, &gpu, Key::new(4)).unwrap();
    params.w = arr(&gpu, &[1.0; 4], &[1, 1, 2, 2]);

    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let xv = tape.var(arr(&gpu, &[1.0; 9], &[1, 1, 3, 3]));
    let y = conv.apply(&tape, &vars, &xv).unwrap();
    assert_eq!(y.value().shape(), [1, 1, 2, 2]);
    close(&y.value().to_vec().unwrap(), &[4.0; 4], 1e-6, "conv fwd");

    // d(sum y)/dw: each kernel tap overlaps 4 unit inputs → 4 everywhere;
    // d/db = number of output positions = 4.
    let loss = y.sum().unwrap();
    let grads = params.grads_from(&vars, &loss).unwrap();
    close(&grads.w.to_vec().unwrap(), &[4.0; 4], 1e-5, "conv dw");
    close(
        &grads.b.as_ref().unwrap().to_vec().unwrap(),
        &[4.0],
        1e-5,
        "conv db",
    );
}

#[test]
fn pools_known_cases_and_nchw_stacking() {
    let gpu = gpu();
    #[rustfmt::skip]
    let x = [
        1.0f32, 2.0,
        3.0,    8.0,
    ];

    let tape = Tape::<f32>::new();
    let xv = tape.var(arr(&gpu, &x, &[1, 1, 2, 2]));
    let maxp = MaxPool2d {
        kh: 2,
        kw: 2,
        stride: 2,
        pad: 0,
    };
    let avgp = AvgPool2d {
        kh: 2,
        kw: 2,
        stride: 2,
        pad: 0,
    };
    let ym = Layer::<f32>::apply(&maxp, &tape, &(), &xv).unwrap();
    let ya = Layer::<f32>::apply(&avgp, &tape, &(), &xv).unwrap();
    assert_eq!(ym.value().to_vec().unwrap(), vec![8.0]);
    assert_eq!(ya.value().to_vec().unwrap(), vec![3.5]);

    // NCHW layers stack as tuples (in_dim None opts out of the 2-D
    // width-contract walk).
    let stack = (
        Conv2d {
            cin: 1,
            cout: 2,
            kh: 2,
            kw: 2,
            stride: 1,
            pad: 0,
            bias: false,
        },
        MaxPool2d {
            kh: 2,
            kw: 2,
            stride: 1,
            pad: 0,
        },
    );
    let params = Layer::<f32>::init(&stack, &gpu, Key::new(5)).unwrap();
    let vars = params.bind(&tape);
    let big = tape.var(arr(
        &gpu,
        &(0..16).map(|i| i as f32).collect::<Vec<_>>(),
        &[1, 1, 4, 4],
    ));
    let out = stack.apply(&tape, &vars, &big).unwrap();
    assert_eq!(out.value().shape(), [1, 2, 2, 2]);
}

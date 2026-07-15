//! MultiheadAttention — the module over the fused SDPA, tested against the
//! composed autograd `multi_head_attention` oracle (values AND gradients,
//! bidirectional and causal), gradient-checked with biases on, probed for
//! causal future-leaks, cross-checked in rope mode against a manual
//! composition, and trained inside a tuple stack.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::attention::{MhaParams, MultiheadAttention};
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

/// Module forward + input/weight grads through a weighted-sum loss.
fn run_module(
    gpu: &quanta::Gpu,
    mha: &MultiheadAttention,
    params: &MhaParams<f32>,
    x: &[f32],
    g: &[f32],
    t: usize,
    e: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let tape: Tape<f32> = Tape::new();
    let vars = params.bind(&tape);
    let xv = tape.var(Array::from_slice(gpu, x, &[t, e]).unwrap());
    let out = mha.apply(&tape, &vars, &xv).unwrap();
    let val = out.value().to_vec().unwrap();
    let w = tape.var(Array::from_slice(gpu, g, &[t, e]).unwrap());
    let loss = out.mul(&w).unwrap().sum().unwrap();
    let dx = loss.grad(&xv).unwrap().to_vec().unwrap();
    let dwq = loss.grad(&vars.wq.w).unwrap().to_vec().unwrap();
    (val, dx, dwq)
}

#[test]
fn mha_matches_composed_oracle_bidirectional_and_causal() {
    let gpu = gpu();
    let (t, e, h) = (5usize, 8usize, 2usize);
    let x = fill(t * e, 11);
    let g = fill(t * e, 12);

    for &causal in &[false, true] {
        let mha = MultiheadAttention {
            bias: false, // the composed oracle has no projection biases
            causal,
            ..MultiheadAttention::new(e, h)
        };
        let params = Layer::<f32>::init(&mha, &gpu, Key::new(3)).unwrap();
        let (f_out, f_dx, f_dwq) = run_module(&gpu, &mha, &params, &x, &g, t, e);

        // Composed oracle: 3-D [1, T, E] input, weight Vars, additive mask.
        let tape: Tape<f32> = Tape::new();
        let xv = tape.var(Array::from_slice(&gpu, &x, &[1, t, e]).unwrap());
        let wq = tape.var(params.wq.w.shallow_clone());
        let wk = tape.var(params.wk.w.shallow_clone());
        let wv = tape.var(params.wv.w.shallow_clone());
        let wo = tape.var(params.wo.w.shallow_clone());
        let mask = if causal {
            let mut m = vec![0.0f32; t * t];
            for i in 0..t {
                for j in (i + 1)..t {
                    m[i * t + j] = -1.0e30;
                }
            }
            Some(tape.var(Array::from_slice(&gpu, &m, &[t, t]).unwrap()))
        } else {
            None
        };
        let out = xv
            .multi_head_attention(&wq, &wk, &wv, &wo, h, mask.as_ref())
            .unwrap();
        let c_out = out.value().to_vec().unwrap();
        let wg = tape.var(Array::from_slice(&gpu, &g, &[1, t, e]).unwrap());
        let loss = out.mul(&wg).unwrap().sum().unwrap();
        let c_dx = loss.grad(&xv).unwrap().to_vec().unwrap();
        let c_dwq = loss.grad(&wq).unwrap().to_vec().unwrap();

        assert!(
            max_abs_diff(&f_out, &c_out) < 1e-4,
            "mha fwd vs oracle (causal={causal}): {}",
            max_abs_diff(&f_out, &c_out)
        );
        assert!(
            max_abs_diff(&f_dx, &c_dx) < 1e-4,
            "mha dx vs oracle (causal={causal})"
        );
        assert!(
            max_abs_diff(&f_dwq, &c_dwq) < 1e-4,
            "mha dwq vs oracle (causal={causal})"
        );
    }
}

#[test]
fn mha_head_contract_fails_at_init() {
    let gpu = gpu();
    let mha = MultiheadAttention::new(10, 3); // 10 % 3 != 0
    assert!(
        Layer::<f32>::init(&mha, &gpu, Key::new(1)).is_err(),
        "divisibility contract must fail at init"
    );
}

#[test]
fn mha_gradcheck_with_biases() {
    let gpu = gpu();
    let (t, e, h) = (3usize, 4usize, 2usize);
    let mha = MultiheadAttention::new(e, h); // bias: true
    let params = Layer::<f32>::init(&mha, &gpu, Key::new(9)).unwrap();
    let x = fill(t * e, 21);
    let g = fill(t * e, 22);

    let loss_of = |xs: &[f32]| -> f64 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, xs, &[t, e]).unwrap());
        let out = mha.apply(&tape, &vars, &xv).unwrap();
        out.value()
            .to_vec()
            .unwrap()
            .iter()
            .zip(&g)
            .map(|(&o, &w)| o as f64 * w as f64)
            .sum()
    };

    let (_, dx, _) = run_module(&gpu, &mha, &params, &x, &g, t, e);
    let hstep = 1e-2f32;
    for idx in 0..t * e {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[idx] += hstep;
        xm[idx] -= hstep;
        let num = (loss_of(&xp) - loss_of(&xm)) / (2.0 * hstep as f64);
        let ana = dx[idx] as f64;
        assert!(
            (num - ana).abs() < 3e-2_f64.max(3e-2 * ana.abs()),
            "gradcheck x[{idx}]: numeric {num} vs analytic {ana}"
        );
    }
}

#[test]
fn mha_causal_blocks_future_leaks() {
    let gpu = gpu();
    let (t, e, h) = (6usize, 8usize, 2usize);
    let mha = MultiheadAttention {
        causal: true,
        ..MultiheadAttention::new(e, h)
    };
    let params = Layer::<f32>::init(&mha, &gpu, Key::new(5)).unwrap();
    let x = fill(t * e, 31);

    let forward = |xs: &[f32]| -> Vec<f32> {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, xs, &[t, e]).unwrap());
        mha.apply(&tape, &vars, &xv)
            .unwrap()
            .value()
            .to_vec()
            .unwrap()
    };

    let base = forward(&x);
    // Perturb the LAST token hard; every earlier row must be untouched.
    let mut x2 = x.clone();
    for c in 0..e {
        x2[(t - 1) * e + c] += 7.0;
    }
    let out2 = forward(&x2);
    assert!(
        max_abs_diff(&base[..(t - 1) * e], &out2[..(t - 1) * e]) < 1e-6,
        "causal mask must block future → past influence"
    );
    assert!(
        max_abs_diff(&base[(t - 1) * e..], &out2[(t - 1) * e..]) > 1e-3,
        "the perturbed row itself must change"
    );
}

#[test]
fn mha_rope_mode_matches_manual_composition() {
    let gpu = gpu();
    let (t, e, h) = (4usize, 8usize, 2usize);
    let hd = e / h;
    let mha = MultiheadAttention {
        bias: false,
        rope: true,
        ..MultiheadAttention::new(e, h)
    };
    let params = Layer::<f32>::init(&mha, &gpu, Key::new(13)).unwrap();
    let x = fill(t * e, 41);

    let tape: Tape<f32> = Tape::new();
    let vars = params.bind(&tape);
    let xv = tape.var(Array::from_slice(&gpu, &x, &[t, e]).unwrap());
    let module_out = mha
        .apply(&tape, &vars, &xv)
        .unwrap()
        .value()
        .to_vec()
        .unwrap();

    // Manual: project → slice heads → rope q/k → fused sdpa → merge → out.
    use quanta_autograd::RopeCache;
    use quanta_nn::functional::{Sdpa, sdpa_var};
    use quanta_nn::rope::rope_var;
    let tape2: Tape<f32> = Tape::new();
    let vars2 = params.bind(&tape2);
    let xv2 = tape2.var(Array::from_slice(&gpu, &x, &[t, e]).unwrap());
    let lin = Linear {
        in_dim: e,
        out_dim: e,
        bias: false,
    };
    let q = lin.apply(&tape2, &vars2.wq, &xv2).unwrap();
    let k = lin.apply(&tape2, &vars2.wk, &xv2).unwrap();
    let v = lin.apply(&tape2, &vars2.wv, &xv2).unwrap();
    let cache = RopeCache::<f32>::new(&gpu, t, hd, 10_000.0).unwrap();
    let slice = |m: &quanta_autograd::Var<f32>, s: usize| {
        m.transpose(0, 1)
            .unwrap()
            .narrow(s, hd)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
    };
    let mut heads = Vec::new();
    for hh in 0..h {
        let qh = rope_var(&tape2, &slice(&q, hh * hd), &cache).unwrap();
        let kh = rope_var(&tape2, &slice(&k, hh * hd), &cache).unwrap();
        let vh = slice(&v, hh * hd);
        let ctx = sdpa_var(&tape2, &qh, &kh, &vh, Sdpa::default()).unwrap();
        heads.push(ctx.transpose(0, 1).unwrap());
    }
    let refs: Vec<_> = heads.iter().collect();
    let merged = quanta_autograd::Var::concat_axis0(&refs)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let manual_out = lin
        .apply(&tape2, &vars2.wo, &merged)
        .unwrap()
        .value()
        .to_vec()
        .unwrap();

    assert!(
        max_abs_diff(&module_out, &manual_out) < 1e-5,
        "rope module vs manual composition"
    );
}

#[test]
fn mha_trains_in_a_tuple_stack() {
    // (Linear 4→8, MHA(8,2), Linear 8→2) — attention inside the Layer
    // model, stepped by fused Adam: predict the sequence-reversed target.
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 4,
            out_dim: 8,
            bias: true,
        },
        MultiheadAttention::new(8, 2),
        Linear {
            in_dim: 8,
            out_dim: 2,
            bias: true,
        },
    );
    let mut params = Layer::<f32>::init(&stack, &gpu, Key::new(17)).unwrap();

    let t = 6usize;
    let xs = fill(t * 4, 51);
    let ys: Vec<f32> = (0..t)
        .flat_map(|r| {
            let src = &xs[(t - 1 - r) * 4..(t - r) * 4]; // reversed row
            [src[0] + src[1], src[2] - src[3]]
        })
        .collect();

    let opt = Adam::new(0.01);
    let mut state = opt.init(&params).unwrap();
    let mut first = None;
    let mut last = 0.0f32;
    for _ in 0..40 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(Array::from_slice(&gpu, &xs, &[t, 4]).unwrap());
        let yv = tape.var(Array::from_slice(&gpu, &ys, &[t, 2]).unwrap());
        let pred = stack.apply(&tape, &vars, &xv).unwrap();
        let d = pred.sub(&yv).unwrap();
        let loss = d.mul(&d).unwrap().sum().unwrap();
        let lval = loss.value().to_vec().unwrap()[0] / t as f32;
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
        last < first * 0.5,
        "MHA stack must train: first {first}, last {last}"
    );
    assert!(last.is_finite());
}

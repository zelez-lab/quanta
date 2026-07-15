//! The Layer model — composition, contracts, trees, and a real training
//! loop through the stack (the D1–D3 architecture exercised end to end).

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::layer::{Key, Layer, LayerNorm, Linear, ParamTree};

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

#[test]
fn width_contract_fails_at_init_not_at_forward() {
    let gpu = gpu();
    // Linear outputs width 8; the norm demands 7 → the CONTRACT must fail
    // at init (model construction), before any forward runs.
    let stack = (
        Linear {
            in_dim: 4,
            out_dim: 8,
            bias: true,
        },
        LayerNorm { dim: 7, eps: 1e-5 },
    );
    let err = Layer::<f32>::init(&stack, &gpu, Key::new(7)).err();
    assert!(
        err.is_some(),
        "width mismatch must be caught at init (build-time contract)"
    );
}

#[test]
fn tree_flatten_unflatten_roundtrip() {
    let gpu = gpu();
    let stack = (
        Linear {
            in_dim: 3,
            out_dim: 5,
            bias: true,
        },
        LayerNorm { dim: 5, eps: 1e-5 },
    );
    let params = Layer::<f32>::init(&stack, &gpu, Key::new(11)).unwrap();
    let leaves = params.flatten();
    assert_eq!(leaves.len(), 4, "w, b, gamma, beta");
    let rebuilt = params.unflatten(&mut params.flatten().into_iter()).unwrap();
    for (a, b) in params.flatten().iter().zip(rebuilt.flatten().iter()) {
        assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
        assert_eq!(a.shape(), b.shape());
    }
}

#[test]
fn key_is_deterministic_and_splits_differ() {
    let gpu = gpu();
    let lin = Linear {
        in_dim: 4,
        out_dim: 4,
        bias: false,
    };
    let a = Layer::<f32>::init(&lin, &gpu, Key::new(42)).unwrap();
    let b = Layer::<f32>::init(&lin, &gpu, Key::new(42)).unwrap();
    assert_eq!(
        a.w.to_vec().unwrap(),
        b.w.to_vec().unwrap(),
        "same key, same init"
    );
    let (k1, k2) = Key::new(42).split();
    let c = Layer::<f32>::init(&lin, &gpu, k1).unwrap();
    let d = Layer::<f32>::init(&lin, &gpu, k2).unwrap();
    assert_ne!(
        c.w.to_vec().unwrap(),
        d.w.to_vec().unwrap(),
        "split keys must diverge"
    );
}

#[test]
fn stack_trains_on_toy_regression() {
    let gpu = gpu();
    // y = fixed linear map of x, learned by (Linear -> LayerNorm -> Linear).
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

    // Deterministic toy data.
    let n = 16usize;
    let xs: Vec<f32> = (0..n * 4)
        .map(|i| ((i * 37 % 17) as f32 / 8.5) - 1.0)
        .collect();
    let ys: Vec<f32> = (0..n)
        .map(|r| {
            let x = &xs[r * 4..r * 4 + 4];
            [
                0.5 * x[0] - 1.2 * x[1] + 0.3 * x[2],
                0.8 * x[3] + 0.1 * x[0],
            ]
        })
        .flatten()
        .collect();

    let lr = 0.002f32; // sum-loss over 32 outputs: keep the effective step sane
    let mut first_loss = None;
    let mut last_loss = 0.0f32;
    for _step in 0..80 {
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

        // SGD over the flattened trees: params ← params − lr·grads.
        let grads = params.grads_from(&vars, &loss).unwrap();
        let p_leaves = params.flatten();
        let g_leaves = grads.flatten();
        let stepped: Vec<Array<f32>> = p_leaves
            .iter()
            .zip(g_leaves.iter())
            .map(|(p, g)| {
                let pv = p.to_vec().unwrap();
                let gv = g.to_vec().unwrap();
                let nv: Vec<f32> = pv.iter().zip(&gv).map(|(&a, &b)| a - lr * b).collect();
                Array::from_slice(&gpu, &nv, p.shape()).unwrap()
            })
            .collect();
        params = params.unflatten(&mut stepped.into_iter()).unwrap();
    }

    let first = first_loss.unwrap();
    assert!(
        last_loss < first * 0.2,
        "training must reduce the loss substantially: first {first}, last {last_loss}"
    );
    assert!(last_loss.is_finite());
}

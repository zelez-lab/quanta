//! `#[derive(ParamTree)]` — the generated plumbing against the hand-written
//! semantics: order-stable flatten/unflatten, Option subtrees, nested
//! structs and tuples as fields, gradient trees, and a fused-Adam step
//! over a derived tree.

use quanta_array::Array;
use quanta_autograd::{DiffScalar, Tape};
use quanta_nn::layer::{Key, Layer, Linear, LinearParams, ParamTree};
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

/// A user-defined tree: a leaf, an optional leaf, a nested shipped
/// struct, and a tuple — every ParamTree constructor at once.
#[derive(ParamTree)]
struct BlockParams<T: DiffScalar> {
    w: Array<T>,
    gate: Option<Array<T>>,
    lin: LinearParams<T>,
    pair: (Array<T>, Array<T>),
}

fn make(gpu: &quanta::Gpu, with_gate: bool) -> BlockParams<f32> {
    let arr = |vals: &[f32], shape: &[usize]| Array::from_slice(gpu, vals, shape).unwrap();
    let lin = Linear {
        in_dim: 2,
        out_dim: 3,
        bias: true,
    };
    BlockParams {
        w: arr(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
        gate: with_gate.then(|| arr(&[0.5, -0.5], &[2])),
        lin: Layer::<f32>::init(&lin, gpu, Key::new(7)).unwrap(),
        pair: (arr(&[9.0], &[1]), arr(&[-3.0, 6.0], &[2])),
    }
}

#[test]
fn derived_flatten_unflatten_roundtrip_and_option_shape() {
    let gpu = gpu();
    let p = make(&gpu, true);
    let leaves = p.flatten();
    // w, gate, lin.w, lin.b, pair.0, pair.1 — declaration order.
    assert_eq!(leaves.len(), 6);
    assert_eq!(leaves[0].to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(leaves[4].to_vec().unwrap(), vec![9.0]);

    let rebuilt = p.unflatten(&mut p.flatten().into_iter()).unwrap();
    for (a, b) in p.flatten().iter().zip(rebuilt.flatten().iter()) {
        assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
        assert_eq!(a.shape(), b.shape());
    }

    // None gate: two fewer leaves... (gate contributes 1) — and the shape
    // witness rebuilds None, not Some.
    let p2 = make(&gpu, false);
    assert_eq!(p2.flatten().len(), 5);
    let rebuilt2 = p2.unflatten(&mut p2.flatten().into_iter()).unwrap();
    assert!(rebuilt2.gate.is_none(), "None survives the roundtrip");
}

#[test]
fn derived_grads_have_tree_shape_and_train_through_adam() {
    let gpu = gpu();
    let mut params = make(&gpu, true);

    // A loss touching most leaves (pair.1 stays untouched — its gradient
    // is zero and must still ride the optimizer without drama).
    let forward = |params: &BlockParams<f32>| {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let mut loss = vars.w.mul(&vars.w).unwrap().sum().unwrap();
        if let Some(g) = &vars.gate {
            loss = loss.add(&g.mul(g).unwrap().sum().unwrap()).unwrap();
        }
        loss = loss
            .add(&vars.lin.w.mul(&vars.lin.w).unwrap().sum().unwrap())
            .unwrap();
        loss = loss
            .add(&vars.pair.0.mul(&vars.pair.0).unwrap().sum().unwrap())
            .unwrap();
        let lval = loss.value().to_vec().unwrap()[0];
        let grads = params.grads_from(&vars, &loss).unwrap();
        (grads, lval)
    };

    // First pass: the gradient tree mirrors the params tree exactly.
    let (grads, l0) = forward(&params);
    let expect: Vec<f32> = params.w.to_vec().unwrap().iter().map(|v| 2.0 * v).collect();
    for (g, e) in grads.w.to_vec().unwrap().iter().zip(&expect) {
        assert!((g - e).abs() < 1e-5, "dw: {g} vs {e}");
    }
    assert!(grads.gate.is_some(), "gradient tree mirrors the Option");
    assert_eq!(
        grads.pair.1.to_vec().unwrap(),
        vec![0.0, 0.0],
        "untouched leaf gets a zero gradient"
    );

    // Then a real loop: persistent Adam state over the derived tree.
    let opt = Adam::new(0.05);
    let mut state = opt.init(&params).unwrap();
    let mut last = l0;
    for _ in 0..40 {
        let (grads, lval) = forward(&params);
        last = lval;
        let (np, ns) = opt.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
    }
    assert!(
        last < l0 * 0.6,
        "Adam over the derived tree must reduce Σx²: {l0} → {last}"
    );
}

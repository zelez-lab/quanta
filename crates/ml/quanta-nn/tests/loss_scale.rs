//! Dynamic loss scaling — exact unscale round-trip, overflow backoff
//! with step-skip, growth streaks, and the strongest property a
//! power-of-two scale buys: scaled training is BITWISE identical to
//! unscaled training while nothing overflows.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::layer::{Key, Layer, Linear, ParamTree};
use quanta_nn::loss::{Reduction, mse_loss};
use quanta_nn::optim::{LossScale, Sgd};

fn gpu() -> quanta::Gpu {
    quanta::init().expect("a device")
}

#[test]
fn scaled_backward_unscales_to_the_exact_gradients() {
    let g = gpu();
    let lin = Linear {
        in_dim: 4,
        out_dim: 2,
        bias: true,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(3)).unwrap();
    let x = Array::from_slice(&g, &[0.5f32; 12], &[3, 4]).unwrap();
    let t = Array::from_slice(&g, &[1.0f32; 6], &[3, 2]).unwrap();

    let grads_plain = {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let y = lin
            .apply(&tape, &vars, &tape.var(x.shallow_clone()))
            .unwrap();
        let loss = mse_loss(&tape, &y, &tape.var(t.shallow_clone()), Reduction::Mean).unwrap();
        params.grads_from(&vars, &loss).unwrap()
    };

    let scaler = LossScale::default();
    let state = scaler.init();
    let (grads_scaled, state2) = {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let y = lin
            .apply(&tape, &vars, &tape.var(x.shallow_clone()))
            .unwrap();
        let loss = mse_loss(&tape, &y, &tape.var(t.shallow_clone()), Reduction::Mean).unwrap();
        let scaled = scaler.scale(&tape, &loss, &state).unwrap();
        let grads = params.grads_from(&vars, &scaled).unwrap();
        scaler.unscale(grads, state).unwrap()
    };
    let grads_scaled = grads_scaled.expect("finite grads must pass");
    assert_eq!(state2.good_steps, 1);
    assert_eq!(state2.scale, 65536.0);

    // 2^16 scaling is exact: bitwise equality, not tolerance.
    for (a, b) in grads_plain
        .flatten()
        .iter()
        .zip(grads_scaled.flatten().iter())
    {
        assert_eq!(a.to_vec().unwrap(), b.to_vec().unwrap());
    }
}

#[test]
fn overflow_skips_the_step_and_backs_off() {
    let g = gpu();
    let lin = Linear {
        in_dim: 2,
        out_dim: 1,
        bias: false,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(0)).unwrap();

    // A gradient tree with an Inf leaf, built directly.
    let bad = quanta_nn::layer::LinearParams {
        w: Array::from_slice(&g, &[1.0f32, f32::INFINITY], &[2, 1]).unwrap(),
        b: None,
    };
    let _ = params;
    let scaler = LossScale::default();
    let state = scaler.init();
    let (out, next) = scaler.unscale(bad, state).unwrap();
    assert!(out.is_none(), "overflow must skip the step");
    assert_eq!(next.scale, 32768.0, "backoff halves the scale");
    assert_eq!(next.good_steps, 0);

    // NaN is caught the same way.
    let nan = quanta_nn::layer::LinearParams {
        w: Array::from_slice(&g, &[f32::NAN, 0.0], &[2, 1]).unwrap(),
        b: None,
    };
    let (out, next) = scaler.unscale(nan, next).unwrap();
    assert!(out.is_none());
    assert_eq!(next.scale, 16384.0);
}

#[test]
fn growth_after_the_interval() {
    let g = gpu();
    let scaler = LossScale {
        growth_interval: 2,
        ..LossScale::default()
    };
    let mut state = scaler.init();
    for step in 0..4 {
        let fine = quanta_nn::layer::LinearParams {
            w: Array::from_slice(&g, &[0.25f32, -0.5], &[2, 1]).unwrap(),
            b: None,
        };
        let (out, next) = scaler.unscale(fine, state).unwrap();
        assert!(out.is_some());
        state = next;
        match step {
            0 => assert_eq!((state.scale, state.good_steps), (65536.0, 1)),
            1 => assert_eq!((state.scale, state.good_steps), (131072.0, 0)),
            2 => assert_eq!((state.scale, state.good_steps), (131072.0, 1)),
            _ => assert_eq!((state.scale, state.good_steps), (262144.0, 0)),
        }
    }
}

#[test]
fn scaled_training_is_bitwise_identical_while_finite() {
    let g = gpu();
    let lin = Linear {
        in_dim: 3,
        out_dim: 1,
        bias: true,
    };
    let x = Array::from_slice(&g, &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]).unwrap();
    let t = Array::from_slice(&g, &[1.0f32, 2.0], &[2, 1]).unwrap();
    let opt = Sgd {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    };

    let run = |scaled: bool| {
        let mut params = Layer::<f32>::init(&lin, &g, Key::new(7)).unwrap();
        let mut opt_state = opt.init(&params).unwrap();
        let scaler = LossScale::default();
        let mut sstate = scaler.init();
        for _ in 0..5 {
            let tape = Tape::<f32>::new();
            let vars = params.bind(&tape);
            let y = lin
                .apply(&tape, &vars, &tape.var(x.shallow_clone()))
                .unwrap();
            let loss = mse_loss(&tape, &y, &tape.var(t.shallow_clone()), Reduction::Mean).unwrap();
            let grads = if scaled {
                let sl = scaler.scale(&tape, &loss, &sstate).unwrap();
                let raw = params.grads_from(&vars, &sl).unwrap();
                let (unscaled, next) = scaler.unscale(raw, sstate).unwrap();
                sstate = next;
                unscaled.expect("finite")
            } else {
                params.grads_from(&vars, &loss).unwrap()
            };
            let (p2, s2) = opt.step(&params, &grads, opt_state).unwrap();
            params = p2;
            opt_state = s2;
        }
        params.w.to_vec().unwrap()
    };

    assert_eq!(run(false), run(true), "power-of-two scaling is exact");
}

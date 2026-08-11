//! Quantized inference, end to end — the runnable twin of
//! `docs/computation/how-to/quantized-checkpoints.md`.
//!
//! Train a small MLP in f32, quantize its weight matrices by name
//! (int8 per-channel for the wide layer, int4 grouped for the head),
//! save the quantized-safetensors bytes, reload BOTH ways — Mode A
//! (dequantize on load, through the unmodified f32 param tree) and
//! Mode B (resident codes through `QuantizedLinear`) — and compare
//! every output against the original model. The deviation is bounded
//! by the proven s/2 round-trip error per weight (T9234–T9235);
//! everything downstream of quantization is exact, so the two load
//! modes agree with each other bitwise.
//!
//! Run: `cargo run --release --example cookbook_quantized_inference \
//!       --features "nn,jit"`

use quanta::nn::activation::Relu;
use quanta::nn::layer::{Key, Layer, Linear, LinearParams, ParamTree};
use quanta::nn::loss::{Reduction, mse_loss};
use quanta::nn::optim::Adam;
use quanta::nn::quant::{self, Granularity, QuantDtype, QuantLeaf, QuantizedLinear};
use quanta::nn::{Array, DiffScalar, Tape};

#[derive(quanta::nn::layer::ParamTree)]
#[param_tree(crate = quanta::nn)]
struct MlpParams<S: DiffScalar> {
    l1: LinearParams<S>,
    l2: LinearParams<S>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = quanta::init()?;
    println!("device: {}", gpu.caps().name);

    // A tiny regression task: y = tanh-shaped curve over a ramp.
    let n = 256usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 / n as f32 * 4.0 - 2.0).collect();
    let ys: Vec<f32> = xs.iter().map(|x| (x * 1.7).tanh() * 0.8 + 0.1).collect();
    let x = Array::from_slice(&gpu, &xs, &[n, 1])?;
    let y = Array::from_slice(&gpu, &ys, &[n, 1])?;

    // ── Train a 1 → 32 → 1 MLP ──────────────────────────────────────────
    let l1 = Linear {
        in_dim: 1,
        out_dim: 32,
        bias: true,
    };
    let l2 = Linear {
        in_dim: 32,
        out_dim: 1,
        bias: true,
    };
    let relu = Relu;

    let (k1, k2) = Key::new(42).split();
    let mut params = MlpParams::<f32> {
        l1: l1.init(&gpu, k1)?,
        l2: l2.init(&gpu, k2)?,
    };
    let opt = Adam::new(1e-2);
    let mut state = opt.init(&params)?;
    for step in 0..200 {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let h = l1.apply(&tape, &vars.l1, &tape.var(x.shallow_clone()))?;
        let h = relu.apply(&tape, &(), &h)?;
        let out = l2.apply(&tape, &vars.l2, &h)?;
        let loss = mse_loss(&tape, &out, &tape.var(y.shallow_clone()), Reduction::Mean)?;
        if step % 50 == 0 || step == 199 {
            println!("step {step:>3}  loss {:.6}", loss.value().to_vec()?[0]);
        }
        let grads = params.grads_from(&vars, &loss)?;
        let (p2, s2) = opt.step(&params, &grads, state)?;
        params = p2;
        state = s2;
    }

    // Reference outputs from the trained f32 model.
    let f32_out = {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let h = l1.apply(&tape, &vars.l1, &tape.var(x.shallow_clone()))?;
        let h = relu.apply(&tape, &(), &h)?;
        l2.apply(&tape, &vars.l2, &h)?.value().to_vec()?
    };

    // ── Quantize by name, save ──────────────────────────────────────────
    // The policy is the caller's: weight matrices quantize (the wide
    // layer int8 per-out-channel, the head int4 grouped along its
    // input axis), biases stay f32 (rank-1 — quantization is rank-2).
    let leaves = params.named_flatten();
    let entries = quant::quantize_named(&leaves, |name, arr| {
        if arr.shape().len() != 2 {
            return None;
        }
        if name == "l1.w" {
            Some((QuantDtype::Int8, Granularity::PerChannel { axis: 1 }))
        } else {
            Some((QuantDtype::Int4, Granularity::Group { axis: 0, size: 8 }))
        }
    })?;
    let bytes = quant::save_named(&entries, None)?;
    // At this toy scale the header and scale grids dominate the file —
    // the codes-are-the-file win (4x for int8, ~8x for int4) appears at
    // real widths, where weight elements outnumber tiles by thousands.
    println!(
        "checkpoint: {} bytes (l1.w int8 per-channel, l2.w int4 g=8, biases f32)",
        bytes.len()
    );

    // ── Mode A: the unmodified f32 tree loads the quantized file ────────
    let restored: MlpParams<f32> = quant::load(&gpu, &params, &bytes)?;
    let mode_a_out = {
        let tape = Tape::<f32>::new();
        let vars = restored.bind(&tape);
        let h = l1.apply(&tape, &vars.l1, &tape.var(x.shallow_clone()))?;
        let h = relu.apply(&tape, &(), &h)?;
        l2.apply(&tape, &vars.l2, &h)?.value().to_vec()?
    };

    // ── Mode B: resident codes through QuantizedLinear ──────────────────
    let loaded = quant::load_named(&gpu, &bytes)?;
    let (mut w1, mut b1, mut w2, mut b2) = (None, None, None, None);
    for (name, leaf) in loaded.leaves {
        match (name.as_str(), leaf) {
            ("l1.w", QuantLeaf::Quantized(m)) => w1 = Some(m),
            ("l1.b", QuantLeaf::F32(a)) => b1 = Some(a),
            ("l2.w", QuantLeaf::Quantized(m)) => w2 = Some(m),
            ("l2.b", QuantLeaf::F32(a)) => b2 = Some(a),
            (other, _) => panic!("unexpected leaf {other}"),
        }
    }
    let ql1 = QuantizedLinear::new(w1.unwrap(), b1)?;
    let ql2 = QuantizedLinear::new(w2.unwrap(), b2)?;
    let mode_b_out = {
        let tape = Tape::<f32>::new();
        let h = ql1.apply(&tape, &(), &tape.var(x.shallow_clone()))?;
        let h = relu.apply(&tape, &(), &h)?;
        ql2.apply(&tape, &(), &h)?.value().to_vec()?
    };

    // ── The envelope ────────────────────────────────────────────────────
    let max_gap = |a: &[f32]| {
        a.iter()
            .zip(&f32_out)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    };
    let (ga, gb) = (max_gap(&mode_a_out), max_gap(&mode_b_out));
    println!("mode A (f32 path):  max |Δ| vs f32 forward = {ga:.6}");
    println!("mode B (resident):  max |Δ| vs f32 forward = {gb:.6}");
    // The two modes see identical weights (device dequantize ≡ host
    // reference, bitwise), so their outputs agree exactly.
    assert_eq!(
        mode_a_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        mode_b_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the two load modes must agree bitwise"
    );
    // The proven bound (T9234) is per WEIGHT: |dq(q(w)) - w| <= s/2.
    // The OUTPUT envelope is that error propagated through the net —
    // data- and scale-dependent, dominated here by the aggressive
    // int4 head (15 levels over each group's max-abs). 0.25 is a
    // regression gate well outside noise but inside any healthy
    // propagation of s/2 for this architecture.
    assert!(
        ga < 0.25,
        "quantized forward drifted past the s/2-derived envelope: {ga}"
    );
    println!("ok — modes agree bitwise, both inside the envelope");
    Ok(())
}

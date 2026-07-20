//! Cookbook: a tiny transformer language model, end to end — the summit
//! walk of `quanta::nn` (the tutorial twin is
//! `docs/computation/tutorials/transformer-lm.md`).
//!
//! Character-level next-token modelling of a repeating phrase: Embedding
//! at the chain head, a causal+rotary TransformerEncoderLayer block,
//! LayerNorm, a Linear head, fused cross-entropy, Adam, key-threaded
//! dropout, a named byte checkpoint, and greedy generation at the end.
//!
//! Run: `cargo run --release --features nn --example cookbook_transformer`

use quanta::nn::attention::MultiheadAttention;
use quanta::nn::embedding::Embedding;
use quanta::nn::layer::{Key, Layer, LayerNorm, Linear, LinearParams, NormParams, ParamTree};
use quanta::nn::loss::{Reduction, cross_entropy_var};
use quanta::nn::optim::Adam;
use quanta::nn::state::{load_state, save_state};
use quanta::nn::transformer::{EncoderLayerParams, TransformerEncoderLayer};
use quanta::nn::{Array, DiffScalar, Tape};

const PHRASE: &str = "quanta ";
const E: usize = 16; // embedding width (small on purpose — see the
// tutorial's note: the composed path host-syncs per op)
const HEADS: usize = 2;
const SEQ: usize = 14; // two repetitions of the phrase
const STEPS: usize = 60;

#[derive(quanta::nn::layer::ParamTree)]
#[param_tree(crate = quanta::nn)]
struct LmParams<S: DiffScalar> {
    emb: Array<S>,
    block: EncoderLayerParams<S>,
    norm: NormParams<S>,
    head: LinearParams<S>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = quanta::init()?;
    println!("device: {}", gpu.caps().name);

    // ── Vocabulary: the phrase's distinct characters ────────────────────
    let mut chars: Vec<char> = PHRASE.chars().collect();
    chars.sort_unstable();
    chars.dedup();
    let vocab = chars.len();
    let tok = |c: char| chars.iter().position(|&x| x == c).unwrap() as u32;
    let text: String = PHRASE.repeat(SEQ / PHRASE.len() + 2);
    let ids: Vec<u32> = text.chars().take(SEQ + 1).map(tok).collect();
    let input = Array::from_slice(&gpu, &ids[..SEQ], &[SEQ])?;
    let labels: Vec<u32> = ids[1..=SEQ].to_vec();

    // ── Model configuration (data, not objects) ─────────────────────────
    let block = TransformerEncoderLayer {
        // Causal + per-head rotary: the rope IS the position sense.
        attn: MultiheadAttention::decoder(E, HEADS),
        ffn_hidden: E,
        dropout: 0.1,
        eps: 1e-5,
    };
    let emb = Embedding { vocab, dim: E };
    let norm = LayerNorm { dim: E, eps: 1e-5 };
    let head = Linear {
        in_dim: E,
        out_dim: vocab,
        bias: true,
    };

    // ── Init: one key, split and consumed ───────────────────────────────
    let key = Key::new(42);
    let (k1, rest) = key.split();
    let (k2, rest) = rest.split();
    let (k3, k4) = rest.split();
    let mut params = LmParams::<f32> {
        emb: emb.init(&gpu, k1)?,
        block: block.init(&gpu, k2)?,
        norm: norm.init(&gpu, k3)?,
        head: head.init(&gpu, k4)?,
    };

    // ── Train ───────────────────────────────────────────────────────────
    let opt = Adam::new(3e-3);
    let mut state = opt.init(&params)?;
    let mut key = Key::new(7);
    for step in 0..STEPS {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);

        let x = emb.apply(&vars.emb, &input)?;
        let (k_step, rest) = key.split();
        key = rest;
        let (x, _spent) = block.apply_train(&tape, &vars.block, &x, k_step)?;
        let x = norm.apply(&tape, &vars.norm, &x)?;
        let logits = head.apply(&tape, &vars.head, &x)?;
        let loss = cross_entropy_var(&tape, &logits, &labels, Reduction::Mean)?;

        if step % 10 == 0 || step == STEPS - 1 {
            println!("step {step:>3}  loss {:.4}", loss.value().to_vec()?[0]);
        }
        let grads = params.grads_from(&vars, &loss)?;
        let (p2, s2) = opt.step(&params, &grads, state)?;
        params = p2;
        state = s2;
    }

    // ── Checkpoint by name, restore, and prove it took ──────────────────
    let bytes = save_state(&params)?;
    println!(
        "checkpoint: {} bytes, {} named leaves (e.g. \"block.attn.wq.w\")",
        bytes.len(),
        params.named_flatten().len()
    );
    let params = load_state(&params, &gpu, &bytes)?;

    // ── Generate greedily from the restored model (eval forward) ────────
    let mut ctx: Vec<u32> = ids[..PHRASE.len()].to_vec();
    for _ in 0..PHRASE.len() * 2 {
        let t = ctx.len();
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let idv = Array::from_slice(&gpu, &ctx, &[t])?;
        let x = emb.apply(&vars.emb, &idv)?;
        let x = block.apply(&tape, &vars.block, &x)?;
        let x = norm.apply(&tape, &vars.norm, &x)?;
        let logits = head.apply(&tape, &vars.head, &x)?;
        let row = &logits.value().to_vec()?[(t - 1) * vocab..t * vocab];
        let next = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();
        ctx.push(next);
    }
    let text: String = ctx.iter().map(|&i| chars[i as usize]).collect();
    println!("generated: {text:?}");
    Ok(())
}

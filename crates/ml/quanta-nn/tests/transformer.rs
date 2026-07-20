//! TransformerEncoderLayer — shapes, eval determinism, keyed training,
//! full-tree checkpointing, tuple stacking, and a real overfit.

use quanta_array::Array;
use quanta_autograd::Tape;
use quanta_nn::layer::{Key, Layer, ParamTree};
use quanta_nn::loss::{Reduction, cross_entropy_var};
use quanta_nn::optim::Adam;
use quanta_nn::state::{load_state, save_state};
use quanta_nn::transformer::TransformerEncoderLayer;

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

fn block() -> TransformerEncoderLayer {
    // Small on purpose: every composed op is a host-synced dispatch, so
    // suite time scales with width × depth × steps (the MVP cost the
    // PARITY notes as the batching/fusion increment).
    TransformerEncoderLayer {
        ffn_hidden: 8,
        dropout: 0.1,
        ..TransformerEncoderLayer::new(8, 2)
    }
}

#[test]
fn shapes_and_eval_is_deterministic() {
    let gpu = gpu();
    let blk = block();
    let params = Layer::<f32>::init(&blk, &gpu, Key::new(1)).unwrap();
    let x: Vec<f32> = (0..6 * 8).map(|i| (i as f32 * 0.11).sin()).collect();

    let run = || -> Vec<f32> {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let xv = tape.var(arr(&gpu, &x, &[6, 8]));
        let y = blk.apply(&tape, &vars, &xv).unwrap();
        assert_eq!(y.value().shape(), [6, 8]);
        y.value().to_vec().unwrap()
    };
    assert_eq!(run(), run(), "eval has no stochastic path");
}

#[test]
fn train_is_keyed_and_rate_zero_matches_eval() {
    let gpu = gpu();
    let blk = block();
    let params = Layer::<f32>::init(&blk, &gpu, Key::new(2)).unwrap();
    let x: Vec<f32> = (0..6 * 8).map(|i| (i as f32 * 0.19).cos()).collect();

    let run_train = |b: &TransformerEncoderLayer, key: Key| -> Vec<f32> {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let xv = tape.var(arr(&gpu, &x, &[6, 8]));
        let (y, _rest) = b.apply_train(&tape, &vars, &xv, key).unwrap();
        y.value().to_vec().unwrap()
    };

    let a = run_train(&blk, Key::new(7));
    let b = run_train(&blk, Key::new(7));
    let c = run_train(&blk, Key::new(8));
    assert_eq!(a, b, "same key, same masks, same forward");
    assert_ne!(a, c, "different key must change the training forward");

    let nodrop = TransformerEncoderLayer {
        dropout: 0.0,
        ..blk
    };
    let t = run_train(&nodrop, Key::new(9));
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let xv = tape.var(arr(&gpu, &x, &[6, 8]));
    let e = nodrop
        .apply(&tape, &vars, &xv)
        .unwrap()
        .value()
        .to_vec()
        .unwrap();
    assert_eq!(t, e, "rate 0 training forward equals eval");
}

#[test]
fn full_tree_checkpoint_round_trips_through_names() {
    let gpu = gpu();
    let blk = block();
    let params = Layer::<f32>::init(&blk, &gpu, Key::new(3)).unwrap();

    let names: Vec<String> = params.named_flatten().into_iter().map(|(n, _)| n).collect();
    assert!(names.contains(&"attn.wq.w".to_string()), "{names:?}");
    assert!(names.contains(&"norm1.gamma".to_string()));
    assert!(names.contains(&"ffn1.b".to_string()));

    let bytes = save_state(&params).unwrap();
    let restored = load_state(&params, &gpu, &bytes).unwrap();

    let x: Vec<f32> = (0..4 * 8).map(|i| (i as f32 * 0.23).sin()).collect();
    let eval = |p: &<TransformerEncoderLayer as Layer<f32>>::Params| -> Vec<f32> {
        let tape = Tape::<f32>::new();
        let vars = p.bind(&tape);
        let xv = tape.var(arr(&gpu, &x, &[4, 8]));
        blk.apply(&tape, &vars, &xv)
            .unwrap()
            .value()
            .to_vec()
            .unwrap()
    };
    assert_eq!(eval(&params), eval(&restored));
}

#[test]
fn overfits_a_tiny_next_token_task() {
    let gpu = gpu();
    // Causal + rope (the decoder preset) so the block sees position.
    let blk = TransformerEncoderLayer {
        attn: quanta_nn::attention::MultiheadAttention::decoder(8, 2),
        ffn_hidden: 8,
        dropout: 0.0,
        eps: 1e-5,
    };
    let mut params = Layer::<f32>::init(&blk, &gpu, Key::new(4)).unwrap();
    let head = quanta_nn::layer::Linear {
        in_dim: 8,
        out_dim: 4,
        bias: true,
    };
    let mut head_p = head.init(&gpu, Key::new(5)).unwrap();

    // Fixed "embeddings" for the pattern 0,1,2,3,0,1,… — predict next.
    let t = 6usize;
    let toks: Vec<u32> = (0..t as u32 + 1).map(|i| i % 4).collect();
    let x_host: Vec<f32> = toks[..t]
        .iter()
        .flat_map(|&tk| (0..8).map(move |j| if j == tk as usize { 1.0 } else { 0.0 }))
        .collect();
    let labels: Vec<u32> = toks[1..].to_vec();

    let opt = Adam::new(1e-2);
    let mut st = None;
    let mut first = f32::NAN;
    let mut last = f32::NAN;
    for step in 0..20 {
        let tape = Tape::<f32>::new();
        let vars = params.bind(&tape);
        let hv = head_p.bind(&tape);
        let xv = tape.var(arr(&gpu, &x_host, &[t, 8]));
        let y = blk.apply(&tape, &vars, &xv).unwrap();
        let logits = head.apply(&tape, &hv, &y).unwrap();
        let loss = cross_entropy_var(&tape, &logits, &labels, Reduction::Mean).unwrap();
        let l = loss.value().to_vec().unwrap()[0];
        if step == 0 {
            first = l;
        }
        last = l;

        let g = params.grads_from(&vars, &loss).unwrap();
        let gh = head_p.grads_from(&hv, &loss).unwrap();
        let joint = (params, head_p);
        let gj = (g, gh);
        let state = match st.take() {
            Some(s) => s,
            None => opt.init(&joint).unwrap(),
        };
        let (new_joint, s2) = opt.step(&joint, &gj, state).unwrap();
        st = Some(s2);
        params = new_joint.0;
        head_p = new_joint.1;
    }
    assert!(
        last < first * 0.5 && last < 1.0,
        "block must be learning the repeating pattern: first {first}, last {last}"
    );
}

#[test]
fn blocks_stack_in_tuples() {
    let gpu = gpu();
    let stack = (block(), block());
    let params = Layer::<f32>::init(&stack, &gpu, Key::new(6)).unwrap();
    let tape = Tape::<f32>::new();
    let vars = params.bind(&tape);
    let x: Vec<f32> = (0..4 * 8).map(|i| (i as f32 * 0.31).cos()).collect();
    let xv = tape.var(arr(&gpu, &x, &[4, 8]));
    let (y, _k) = stack.apply_train(&tape, &vars, &xv, Key::new(10)).unwrap();
    assert_eq!(y.value().shape(), [4, 8]);
}

//! Named traversal + state save/load — names by field/index, Option
//! transparency, the flatten-order alignment invariant, byte round-trips,
//! name-keyed (order-independent) loading, and the loud mismatch errors.

use quanta_array::Array;
use quanta_autograd::DiffScalar;
use quanta_nn::layer::{Key, Layer, LayerNorm, Linear, LinearParams, NormParams, ParamTree};
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

#[derive(ParamTree)]
#[param_tree(crate = quanta_nn)]
struct BlockP<T: DiffScalar> {
    proj: LinearParams<T>,
    norm: NormParams<T>,
    gate: Option<Array<T>>,
}

/// Same leaf names as `BlockP`, DIFFERENT field order — the witness that
/// proves loading matches by name, not position.
#[derive(ParamTree)]
#[param_tree(crate = quanta_nn)]
struct BlockPShuffled<T: DiffScalar> {
    norm: NormParams<T>,
    gate: Option<Array<T>>,
    proj: LinearParams<T>,
}

fn block(gpu: &quanta::Gpu, with_gate: bool) -> BlockP<f32> {
    let proj = Linear {
        in_dim: 3,
        out_dim: 4,
        bias: true,
    }
    .init(gpu, Key::new(1))
    .unwrap();
    let norm = LayerNorm { dim: 4, eps: 1e-5 }
        .init(gpu, Key::new(2))
        .unwrap();
    let gate = with_gate.then(|| Array::from_slice(gpu, &[0.5f32; 4], &[4]).unwrap());
    BlockP { proj, norm, gate }
}

#[test]
fn names_by_field_index_and_option_transparency() {
    let gpu = gpu();

    // Tuple stack: index segments.
    let stack = (
        Linear {
            in_dim: 3,
            out_dim: 4,
            bias: true,
        },
        LayerNorm { dim: 4, eps: 1e-5 },
    );
    let params = Layer::<f32>::init(&stack, &gpu, Key::new(3)).unwrap();
    let names: Vec<String> = params.named_flatten().into_iter().map(|(n, _)| n).collect();
    assert_eq!(names, ["0.w", "0.b", "1.gamma", "1.beta"]);

    // Derived struct: field segments; a `None` Option leaf vanishes,
    // a `Some` appears under its field name — no extra layer.
    let with: Vec<String> = block(&gpu, true)
        .named_flatten()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(
        with,
        ["proj.w", "proj.b", "norm.gamma", "norm.beta", "gate"]
    );
    let without: Vec<String> = block(&gpu, false)
        .named_flatten()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(without, ["proj.w", "proj.b", "norm.gamma", "norm.beta"]);
}

#[test]
fn named_order_is_flatten_order() {
    let gpu = gpu();
    let b = block(&gpu, true);
    let named = b.named_flatten();
    let flat = b.flatten();
    assert_eq!(named.len(), flat.len());
    for ((name, a), f) in named.iter().zip(flat.iter()) {
        assert_eq!(
            a.to_vec().unwrap(),
            f.to_vec().unwrap(),
            "leaf `{name}` must sit at its flatten position"
        );
    }
}

#[test]
fn save_load_round_trip() {
    let gpu = gpu();
    let b = block(&gpu, true);
    let bytes = save_state(&b).unwrap();
    let restored = load_state(&b, &gpu, &bytes).unwrap();
    for ((name, a), (_, r)) in b.named_flatten().iter().zip(restored.named_flatten()) {
        assert_eq!(
            a.to_vec().unwrap(),
            r.to_vec().unwrap(),
            "leaf `{name}` must round-trip exactly"
        );
    }
}

#[test]
fn load_matches_by_name_not_position() {
    let gpu = gpu();
    let b = block(&gpu, true);
    let bytes = save_state(&b).unwrap();

    // The shuffled witness has the same names in a different flatten
    // order; a positional loader would put norm weights into proj.
    let witness = BlockPShuffled {
        norm: LayerNorm { dim: 4, eps: 1e-5 }
            .init(&gpu, Key::new(9))
            .unwrap(),
        gate: Some(Array::from_slice(&gpu, &[0.0f32; 4], &[4]).unwrap()),
        proj: Linear {
            in_dim: 3,
            out_dim: 4,
            bias: true,
        }
        .init(&gpu, Key::new(9))
        .unwrap(),
    };
    let restored = load_state(&witness, &gpu, &bytes).unwrap();
    assert_eq!(
        restored.proj.w.to_vec().unwrap(),
        b.proj.w.to_vec().unwrap(),
        "proj.w must land on proj.w regardless of field order"
    );
    assert_eq!(
        restored.norm.gamma.to_vec().unwrap(),
        b.norm.gamma.to_vec().unwrap()
    );
}

#[test]
fn mismatches_are_loud_and_name_the_leaf() {
    let gpu = gpu();

    // Missing: witness expects `gate`, the bytes never had it.
    let bytes_no_gate = save_state(&block(&gpu, false)).unwrap();
    let err = load_state(&block(&gpu, true), &gpu, &bytes_no_gate)
        .err()
        .expect("missing leaf must fail");
    assert!(format!("{err:?}").contains("gate"), "{err:?}");

    // Extra: the bytes carry `gate`, the witness has no home for it.
    let bytes_gate = save_state(&block(&gpu, true)).unwrap();
    let err = load_state(&block(&gpu, false), &gpu, &bytes_gate)
        .err()
        .expect("extra leaf must fail");
    assert!(format!("{err:?}").contains("gate"), "{err:?}");

    // Shape: same names, wrong width.
    let wrong = BlockP {
        proj: Linear {
            in_dim: 3,
            out_dim: 5,
            bias: true,
        }
        .init(&gpu, Key::new(4))
        .unwrap(),
        norm: LayerNorm { dim: 5, eps: 1e-5 }
            .init(&gpu, Key::new(5))
            .unwrap(),
        gate: Some(Array::from_slice(&gpu, &[0.0f32; 5], &[5]).unwrap()),
    };
    let err = load_state(&wrong, &gpu, &bytes_gate)
        .err()
        .expect("shape mismatch must fail");
    assert!(format!("{err:?}").contains("proj.w"), "{err:?}");

    // Garbage bytes.
    assert!(load_state(&block(&gpu, true), &gpu, b"nonsense").is_err());
}

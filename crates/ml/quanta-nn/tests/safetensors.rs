//! safetensors interop — spec compliance against hand-built reference
//! bytes, round-trips through the ParamTree traversal, dtype
//! upconversion, and the loud-error contract.

use std::collections::HashMap;

use quanta_array::Array;
use quanta_nn::layer::{Key, Layer, Linear, ParamTree};
use quanta_nn::safetensors::{load, load_named, save, save_named};

fn gpu() -> quanta::Gpu {
    quanta::init().expect("a device")
}

/// The error's debug text; panics if the call succeeded. (`unwrap_err`
/// needs `Debug` on the success type, which holds GPU arrays.)
fn err_text<T>(r: Result<T, quanta_autograd::AutogradError>) -> String {
    match r {
        Err(e) => format!("{e:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

/// A minimal file built by hand, straight from the spec: one tensor
/// "w", F32, shape [2, 2], values [1, 2, 3, 4].
fn reference_bytes() -> Vec<u8> {
    let header = br#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
    let mut padded = header.to_vec();
    while (padded.len() + 8) % 8 != 0 {
        padded.push(b' ');
    }
    let mut out = (padded.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&padded);
    for v in [1.0f32, 2.0, 3.0, 4.0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[test]
fn loads_the_hand_built_reference() {
    let g = gpu();
    let loaded = load_named(&g, &reference_bytes()).unwrap();
    assert_eq!(loaded.tensors.len(), 1);
    let (name, arr) = &loaded.tensors[0];
    assert_eq!(name, "w");
    assert_eq!(arr.shape(), &[2, 2]);
    assert_eq!(arr.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(loaded.metadata.is_empty());
}

#[test]
fn save_matches_the_reference_bytes() {
    let g = gpu();
    let w = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let bytes = save_named(&[("w".to_string(), w)], None).unwrap();
    assert_eq!(bytes, reference_bytes(), "byte-exact spec compliance");
}

#[test]
fn tree_round_trip_by_name_not_order() {
    let g = gpu();
    let lin = Linear {
        in_dim: 6,
        out_dim: 3,
        bias: true,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(11)).unwrap();
    let bytes = save(&params).unwrap();

    // Reload into a differently-initialized witness: values must come
    // from the file, matched by name.
    let witness = Layer::<f32>::init(&lin, &g, Key::new(99)).unwrap();
    let restored = load(&g, &witness, &bytes).unwrap();
    assert_eq!(restored.w.to_vec().unwrap(), params.w.to_vec().unwrap());
    assert_eq!(
        restored.b.as_ref().unwrap().to_vec().unwrap(),
        params.b.as_ref().unwrap().to_vec().unwrap()
    );

    // Reorder the entries (save the named list reversed): the
    // name-keyed load must not care.
    let mut named = params.named_flatten();
    named.reverse();
    let reordered = save_named(&named, None).unwrap();
    let restored2 = load(&g, &witness, &reordered).unwrap();
    assert_eq!(restored2.w.to_vec().unwrap(), params.w.to_vec().unwrap());
}

#[test]
fn metadata_round_trips() {
    let g = gpu();
    let w = Array::from_slice(&g, &[0.5f32], &[1]).unwrap();
    let mut meta = HashMap::new();
    meta.insert("format".to_string(), "pt".to_string());
    meta.insert("note".to_string(), "quanta \"test\"".to_string());
    let bytes = save_named(&[("w".to_string(), w)], Some(&meta)).unwrap();
    let loaded = load_named(&g, &bytes).unwrap();
    assert_eq!(loaded.metadata, meta);
}

#[test]
fn f16_and_bf16_upconvert() {
    let g = gpu();
    // F16: 1.0 = 0x3C00, -2.5 = 0xC100, min subnormal = 0x0001.
    let halves: [u16; 3] = [0x3C00, 0xC100, 0x0001];
    let header = br#"{"h":{"dtype":"F16","shape":[3],"data_offsets":[0,6]}}"#;
    let mut padded = header.to_vec();
    while (padded.len() + 8) % 8 != 0 {
        padded.push(b' ');
    }
    let mut bytes = (padded.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&padded);
    for h in halves {
        bytes.extend_from_slice(&h.to_le_bytes());
    }
    let loaded = load_named(&g, &bytes).unwrap();
    let v = loaded.tensors[0].1.to_vec().unwrap();
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], -2.5);
    assert!((v[2] - 5.960_464_5e-8).abs() < 1e-12, "min subnormal");

    // BF16 is the top half of f32: 3.140625 = 0x4049.
    let header = br#"{"b":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#;
    let mut padded = header.to_vec();
    while (padded.len() + 8) % 8 != 0 {
        padded.push(b' ');
    }
    let mut bytes = (padded.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&padded);
    bytes.extend_from_slice(&0x4049u16.to_le_bytes());
    let loaded = load_named(&g, &bytes).unwrap();
    assert_eq!(loaded.tensors[0].1.to_vec().unwrap()[0], 3.140625);
}

#[test]
fn loud_errors_name_the_problem() {
    let g = gpu();
    let lin = Linear {
        in_dim: 4,
        out_dim: 2,
        bias: false,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(0)).unwrap();

    // Missing tensor: a file holding only an unrelated name.
    let stray = Array::from_slice(&g, &[0.0f32; 8], &[4, 2]).unwrap();
    let bytes = save_named(&[("not_w".to_string(), stray)], None).unwrap();
    let err = err_text(load(&g, &params, &bytes));
    assert!(err.contains("missing tensor"), "{err}");

    // Extra tensor rides along with the right one.
    let mut named = params.named_flatten();
    named.push((
        "extra".to_string(),
        Array::from_slice(&g, &[1.0f32], &[1]).unwrap(),
    ));
    let bytes = save_named(&named, None).unwrap();
    let err = err_text(load(&g, &params, &bytes));
    assert!(err.contains("no home"), "{err}");

    // Shape mismatch.
    let wrong = Array::from_slice(&g, &[0.0f32; 8], &[2, 4]).unwrap();
    let bytes = save_named(&[("w".to_string(), wrong)], None).unwrap();
    let err = err_text(load(&g, &params, &bytes));
    assert!(err.contains("shape"), "{err}");

    // Truncated file and an oversize header length.
    let _ = err_text(load_named(&g, &[1, 2, 3]));
    let mut lying = (u64::MAX).to_le_bytes().to_vec();
    lying.extend_from_slice(b"{}");
    let _ = err_text(load_named(&g, &lying));

    // An unsupported dtype names the tensor.
    let header = br#"{"q":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
    let mut padded = header.to_vec();
    while (padded.len() + 8) % 8 != 0 {
        padded.push(b' ');
    }
    let mut bytes = (padded.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&padded);
    bytes.extend_from_slice(&[0u8; 8]);
    let err = err_text(load_named(&g, &bytes));
    assert!(err.contains("I64") && err.contains('q'), "{err}");
}

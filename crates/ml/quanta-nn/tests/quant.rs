//! Quantized checkpoints — spec compliance against hand-built reference
//! bytes, bit-exact save→load round-trips, the loud-error contract for
//! every format rule, `QuantizedLinear` ≡ `Linear` bitwise in both
//! modes, and a trained end-to-end within the s/2-derived bound.

use std::collections::HashMap;

use quanta_array::autograd::{AutogradError, Tape};
use quanta_array::{Array, ArrayError};
use quanta_nn::activation::Relu;
use quanta_nn::layer::{Key, Layer, Linear, LinearParams, ParamTree};
use quanta_nn::loss::{Reduction, mse_loss};
use quanta_nn::optim::Sgd;
use quanta_nn::quant::{
    Granularity, QuantCodes, QuantDtype, QuantLeaf, QuantizedLinear, QuantizedMatrix, load,
    load_named, load_named_f32, quantize_named, save_named,
};
use quanta_nn::safetensors;

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

/// The error message under the quant contract: a `QuantError` speaks
/// through its Display (leaf / reason / workaround always present);
/// anything else falls back to the debug text. Panics on success.
fn err_text<T>(r: Result<T, AutogradError>) -> String {
    match r {
        Err(AutogradError::Array(ArrayError::Quant(e))) => format!("{e}"),
        Err(e) => format!("{e:?}"),
        Ok(_) => panic!("expected an error"),
    }
}

/// Assemble a safetensors byte string from a header and a data section
/// (the spec: 8-byte LE length, space-padded header, data).
fn st_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut padded = header.as_bytes().to_vec();
    while (padded.len() + 8) % 8 != 0 {
        padded.push(b' ');
    }
    let mut out = (padded.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&padded);
    out.extend_from_slice(data);
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Deterministic pseudo-random fill (the optimizer-test mixer).
fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

// ── The format: spec-byte-exact save ────────────────────────────────────

/// The reference file, straight from the convention: one plain leaf
/// "b", one int8 per-tensor leaf "w" (scale 1.0 so the codes are the
/// values), one int4 per-tensor leaf "v" (8 low-first nibbles per LE
/// u32 word). Header entries in save order, metadata keys sorted.
fn reference_quantized_bytes() -> Vec<u8> {
    let header = concat!(
        r#"{"__metadata__":{"quant:v":"q4s;gr=1;gc=4;rows=1;cols=4","#,
        r#""quant:w":"q8s;gr=2;gc=4;rows=2;cols=4","quanta.quant":"1"},"#,
        r#""b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"#,
        r#""w.q":{"dtype":"I8","shape":[2,4],"data_offsets":[8,16]},"#,
        r#""w.qs":{"dtype":"F32","shape":[1,1],"data_offsets":[16,20]},"#,
        r#""v.q":{"dtype":"U32","shape":[1,1],"data_offsets":[20,24]},"#,
        r#""v.qs":{"dtype":"F32","shape":[1,1],"data_offsets":[24,28]}}"#,
    );
    let mut data = Vec::new();
    for v in [0.5f32, -1.5] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    // int8 codes, two's complement, row-major.
    data.extend_from_slice(&[0x01, 0xFE, 0x03, 0xFC, 0x05, 0xFA, 0x07, 0x7F]);
    data.extend_from_slice(&1.0f32.to_le_bytes());
    // int4 codes 7, -2, 3, -4 → nibbles 0x7, 0xE, 0x3, 0xC low-first.
    data.extend_from_slice(&0x0000_C3E7u32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    st_file(header, &data)
}

fn reference_entries(g: &quanta::Gpu) -> Vec<(String, QuantLeaf)> {
    let b = Array::from_slice(g, &[0.5f32, -1.5], &[2]).unwrap();
    // max|w| = 127 → per-tensor scale 127/127 = 1.0: codes = values.
    let w = Array::from_slice(
        g,
        &[1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 127.0],
        &[2, 4],
    )
    .unwrap();
    // max|v| = 7 → scale 7/7 = 1.0.
    let v = Array::from_slice(g, &[7.0f32, -2.0, 3.0, -4.0], &[1, 4]).unwrap();
    vec![
        ("b".to_string(), QuantLeaf::F32(b)),
        (
            "w".to_string(),
            QuantLeaf::Quantized(
                QuantizedMatrix::quantize(&w, QuantDtype::Int8, Granularity::PerTensor).unwrap(),
            ),
        ),
        (
            "v".to_string(),
            QuantLeaf::Quantized(
                QuantizedMatrix::quantize(&v, QuantDtype::Int4, Granularity::PerTensor).unwrap(),
            ),
        ),
    ]
}

#[test]
fn save_matches_the_reference_bytes() {
    let g = gpu();
    let bytes = save_named(&reference_entries(&g), None).unwrap();
    assert_eq!(
        bytes,
        reference_quantized_bytes(),
        "byte-exact spec compliance"
    );
}

#[test]
fn a_plain_safetensors_reader_opens_the_container() {
    let g = gpu();
    // The validity claim: the file is ordinary safetensors. A plain
    // F32-only reader parses the container fine and stops exactly at
    // the codes dtype, naming the tensor — the declared third-party
    // behavior (unknown dtypes are loud, the container is never the
    // obstacle). "b" (header-first) has already loaded by then.
    let err = err_text(safetensors::load_named(&g, &reference_quantized_bytes()));
    assert!(err.contains("w.q") && err.contains("I8"), "{err}");

    // And a file with no quantized leaves written through the quant
    // writer is byte-identical to the plain writer's output — fully
    // readable by the plain reader.
    let w = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let via_quant = save_named(
        &[("w".to_string(), QuantLeaf::F32(w.shallow_clone()))],
        None,
    )
    .unwrap();
    let via_plain = safetensors::save_named(&[("w".to_string(), w)], None).unwrap();
    assert_eq!(via_quant, via_plain);
    assert_eq!(
        safetensors::load_named(&g, &via_quant).unwrap().tensors[0]
            .1
            .to_vec()
            .unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

// ── Round trips ─────────────────────────────────────────────────────────

#[test]
fn save_load_round_trip_is_bit_exact() {
    let g = gpu();
    // Non-divisible edges on purpose: [5, 6] per-channel int8 and
    // [4, 10] axis-0-grouped int4 (10 columns → 2 words/row, final
    // word zero-padded), plus an f32 leaf riding along.
    let w = Array::from_slice(&g, &fill(30, 1), &[5, 6]).unwrap();
    let u = Array::from_slice(&g, &fill(40, 2), &[4, 10]).unwrap();
    let gamma = Array::from_slice(&g, &fill(6, 3), &[6]).unwrap();
    let named = vec![
        ("blk.w".to_string(), w.shallow_clone()),
        ("blk.u".to_string(), u.shallow_clone()),
        ("ln.gamma".to_string(), gamma.shallow_clone()),
    ];
    let entries = quantize_named(&named, |name, _| match name {
        "blk.w" => Some((QuantDtype::Int8, Granularity::PerChannel { axis: 1 })),
        "blk.u" => Some((QuantDtype::Int4, Granularity::Group { axis: 0, size: 2 })),
        _ => None,
    })
    .unwrap();
    let mut meta = HashMap::new();
    meta.insert("note".to_string(), "trained on ramps".to_string());
    let bytes = save_named(&entries, Some(&meta)).unwrap();

    let loaded = load_named(&g, &bytes).unwrap();
    // User metadata round-trips; the machinery keys are stripped.
    assert_eq!(loaded.metadata, meta);
    assert_eq!(loaded.leaves.len(), 3);

    for ((name, orig), (lname, lleaf)) in entries.iter().zip(&loaded.leaves) {
        assert_eq!(name, lname);
        match (orig, lleaf) {
            (QuantLeaf::F32(a), QuantLeaf::F32(b)) => {
                assert_eq!(bits(&a.to_vec().unwrap()), bits(&b.to_vec().unwrap()));
            }
            (QuantLeaf::Quantized(a), QuantLeaf::Quantized(b)) => {
                assert_eq!(a.shape(), b.shape());
                assert_eq!(a.dtype(), b.dtype());
                assert_eq!(a.tile(), b.tile());
                assert_eq!(a.granularity(), b.granularity());
                match (a.codes(), b.codes()) {
                    (QuantCodes::Int8(x), QuantCodes::Int8(y)) => {
                        assert_eq!(x.to_vec().unwrap(), y.to_vec().unwrap());
                    }
                    (QuantCodes::Int4Packed(x), QuantCodes::Int4Packed(y)) => {
                        assert_eq!(x.to_vec().unwrap(), y.to_vec().unwrap());
                    }
                    _ => panic!("codes kind changed across the round trip"),
                }
                assert_eq!(
                    bits(&a.scales().to_vec().unwrap()),
                    bits(&b.scales().to_vec().unwrap())
                );
            }
            _ => panic!("leaf kind changed across the round trip"),
        }
    }

    // Saving the loaded leaves reproduces the byte stream exactly —
    // the save is deterministic and the load loses nothing.
    let again = save_named(&loaded.leaves, Some(&loaded.metadata)).unwrap();
    assert_eq!(again, bytes);
}

#[test]
fn mode_a_load_equals_device_dequantize_bitwise() {
    let g = gpu();
    let entries = reference_entries(&g);
    let bytes = save_named(&entries, None).unwrap();
    let loaded = load_named_f32(&g, &bytes).unwrap();
    assert_eq!(loaded.tensors.len(), 3);
    for ((name, leaf), (lname, arr)) in entries.iter().zip(&loaded.tensors) {
        assert_eq!(name, lname);
        match leaf {
            QuantLeaf::F32(a) => {
                assert_eq!(bits(&a.to_vec().unwrap()), bits(&arr.to_vec().unwrap()));
            }
            QuantLeaf::Quantized(m) => {
                // Host (Mode A) and device dequantization are the same
                // exact `scale · q` — bitwise equal.
                assert_eq!(arr.shape(), m.shape());
                assert_eq!(
                    bits(&m.dequantize().unwrap().to_vec().unwrap()),
                    bits(&arr.to_vec().unwrap())
                );
            }
        }
    }

    // A plain safetensors file (no quant machinery) loads through the
    // quant loaders too — all leaves arrive f32.
    let plain = safetensors::save_named(
        &[(
            "w".to_string(),
            Array::from_slice(&g, &[1.5f32], &[1]).unwrap(),
        )],
        None,
    )
    .unwrap();
    let lq = load_named(&g, &plain).unwrap();
    assert!(matches!(&lq.leaves[0].1, QuantLeaf::F32(_)));
}

// ── The loud-error contract ─────────────────────────────────────────────

#[test]
fn loud_errors_name_the_problem() {
    let g = gpu();
    let meta_w = r#""quant:w":"q8s;gr=1;gc=2;rows=1;cols=2""#;
    let base_data: &[u8] = &[0x01, 0xFE, 0x00, 0x00, 0x80, 0x3F]; // codes + 1.0f32

    // Unknown format version — names both versions.
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w},"quanta.quant":"2"}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}}}"#
        ),
        base_data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(
        err.contains("version 2") && err.contains("reads 1"),
        "{err}"
    );

    // quant:* metadata without the version key.
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w}}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}}}"#
        ),
        base_data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(
        err.contains("quanta.quant") && err.contains("missing"),
        "{err}"
    );

    // Orphan quant:w — the tensors are missing.
    let f = st_file(
        &format!(r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}}}}"#),
        &[],
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("w.q") && err.contains("missing"), "{err}");

    // An I8 tensor with no quant:<leaf> metadata is an orphan, not a
    // silently-plain leaf.
    let f = st_file(
        r#"{"w.q":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}}"#,
        &base_data[..2],
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("w.q") && err.contains("no quant:"), "{err}");

    // A tensor named `w` beside quant:w — the ambiguity the convention
    // forbids structurally.
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}},"w":{{"dtype":"F32","shape":[2],"data_offsets":[6,14]}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}}}"#
        ),
        &[base_data, &[0u8; 8]].concat(),
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("both a tensor"), "{err}");

    // The reserved zero-points slot.
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}},"w.qz":{{"dtype":"F32","shape":[1,1],"data_offsets":[6,10]}}}}"#
        ),
        &[base_data, &[0u8; 4]].concat(),
    );
    let err = err_text(load_named(&g, &f));
    assert!(
        err.contains("w.qz") && err.contains("reserved affine"),
        "{err}"
    );

    // An unknown scheme tag.
    let f = st_file(
        r#"{"__metadata__":{"quant:w":"q5s;gr=1;gc=2;rows=1;cols=2","quanta.quant":"1"},"w.q":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]},"w.qs":{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}"#,
        base_data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("q5s"), "{err}");

    // The reserved affine mode.
    let f = st_file(
        r#"{"__metadata__":{"quant:w":"q8s;gr=1;gc=2;rows=1;cols=2;mode=affine","quanta.quant":"1"},"w.q":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]},"w.qs":{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}"#,
        base_data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("affine"), "{err}");

    // Codes dtype disagreeing with the scheme (q4s wants U32).
    let f = st_file(
        r#"{"__metadata__":{"quant:w":"q4s;gr=1;gc=2;rows=1;cols=2","quanta.quant":"1"},"w.q":{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]},"w.qs":{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}"#,
        base_data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("U32"), "{err}");

    // Scales shape disagreeing with the grid.
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,2],"data_offsets":[2,10]}}}}"#
        ),
        &[&base_data[..2], &[0u8; 8][..]].concat(),
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("w.qs") && err.contains("wants"), "{err}");

    // A non-finite scale (corrupt file) — names the leaf and the tile.
    let mut data = base_data[..2].to_vec();
    data.extend_from_slice(&f32::NAN.to_le_bytes());
    let f = st_file(
        &format!(
            r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}},"w.q":{{"dtype":"I8","shape":[1,2],"data_offsets":[0,2]}},"w.qs":{{"dtype":"F32","shape":[1,1],"data_offsets":[2,6]}}}}"#
        ),
        &data,
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains('w') && err.contains("non-finite"), "{err}");

    // A grid no v1 writer can produce.
    let f = st_file(
        r#"{"__metadata__":{"quant:w":"q8s;gr=3;gc=2;rows=4;cols=4","quanta.quant":"1"},"w.q":{"dtype":"I8","shape":[4,4],"data_offsets":[0,16]},"w.qs":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#,
        &[0u8; 32],
    );
    let err = err_text(load_named(&g, &f));
    assert!(err.contains("not a v1 tile grid"), "{err}");

    // Mode A rejects the same files the same way (shared validation).
    let f = st_file(
        &format!(r#"{{"__metadata__":{{{meta_w},"quanta.quant":"1"}}}}"#),
        &[],
    );
    let err = err_text(load_named_f32(&g, &f));
    assert!(err.contains("w.q") && err.contains("missing"), "{err}");

    // Save-side collisions: a user metadata key inside the convention's
    // namespace, and a plain leaf occupying a codes slot.
    let w = Array::from_slice(&g, &[1.0f32, 2.0], &[1, 2]).unwrap();
    let q = QuantizedMatrix::quantize(&w, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    let mut meta = HashMap::new();
    meta.insert("quant:w".to_string(), "spoof".to_string());
    let err = err_text(save_named(
        &[("w".to_string(), QuantLeaf::Quantized(q))],
        Some(&meta),
    ));
    assert!(err.contains("quant:w") && err.contains("collides"), "{err}");

    let q = QuantizedMatrix::quantize(&w, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    let err = err_text(save_named(
        &[
            ("w".to_string(), QuantLeaf::Quantized(q)),
            ("w.q".to_string(), QuantLeaf::F32(w.shallow_clone())),
        ],
        None,
    ));
    assert!(err.contains("w.q") && err.contains("collides"), "{err}");
}

// ── QuantizedLinear ≡ Linear, both modes ────────────────────────────────

/// Forward `x` through `Linear` fed an explicit weight/bias pair.
fn linear_forward(
    lin: &Linear,
    w: &Array<f32>,
    b: Option<&Array<f32>>,
    x: &Array<f32>,
) -> Vec<f32> {
    let params = LinearParams::<f32> {
        w: w.shallow_clone(),
        b: b.map(|b| b.shallow_clone()),
    };
    let tape: Tape<f32> = Tape::new();
    let vars = params.bind(&tape);
    let xv = tape.var(x.shallow_clone());
    lin.apply(&tape, &vars, &xv)
        .unwrap()
        .value()
        .to_vec()
        .unwrap()
}

fn quantized_forward(ql: &QuantizedLinear, x: &Array<f32>) -> Vec<f32> {
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(x.shallow_clone());
    ql.apply(&tape, &(), &xv).unwrap().value().to_vec().unwrap()
}

#[test]
fn quantized_linear_matches_linear_bitwise_in_both_modes() {
    let g = gpu();
    let lin = Linear {
        in_dim: 6,
        out_dim: 4,
        bias: true,
    };
    let params = Layer::<f32>::init(&lin, &g, Key::new(21)).unwrap();
    let x = Array::from_slice(&g, &fill(3 * 6, 9), &[3, 6]).unwrap();

    for (dtype, gran) in [
        (QuantDtype::Int8, Granularity::PerChannel { axis: 1 }),
        (QuantDtype::Int4, Granularity::Group { axis: 0, size: 3 }),
    ] {
        let mq = QuantizedMatrix::quantize(&params.w, dtype, gran).unwrap();
        // The oracle: Linear fed the dequantized weight — the SAME tape
        // ops QuantizedLinear::apply runs, so equality is bitwise.
        let wd = mq.dequantize().unwrap();
        let y_ref = linear_forward(&lin, &wd, params.b.as_ref(), &x);

        // Mode B: resident codes, per-forward dequantize.
        let ql_b = QuantizedLinear::new(
            QuantizedMatrix::quantize(&params.w, dtype, gran).unwrap(),
            params.b.as_ref().map(|b| b.shallow_clone()),
        )
        .unwrap();
        assert_eq!(bits(&quantized_forward(&ql_b, &x)), bits(&y_ref));

        // Mode A: dequantized once at construction, f32 held.
        let ql_a = QuantizedLinear::dequantized(&mq, params.b.as_ref().map(|b| b.shallow_clone()))
            .unwrap();
        assert_eq!(bits(&quantized_forward(&ql_a, &x)), bits(&y_ref));

        // Layer bookkeeping: dimension contract + the empty param tree.
        assert_eq!(Layer::<f32>::in_dim(&ql_b), Some(6));
        assert_eq!(Layer::<f32>::out_dim(&ql_b, 6), 4);
        ql_b.init(&g, Key::new(0)).unwrap();
    }

    // Tuple-stackable like any Layer (Params = () occupies its slot
    // for free).
    let mq = QuantizedMatrix::quantize(
        &params.w,
        QuantDtype::Int8,
        Granularity::PerChannel { axis: 1 },
    )
    .unwrap();
    let stack = (
        QuantizedLinear::new(mq, params.b.as_ref().map(|b| b.shallow_clone())).unwrap(),
        Relu,
    );
    let tape: Tape<f32> = Tape::new();
    let xv = tape.var(x.shallow_clone());
    let y = stack.apply(&tape, &((), ()), &xv).unwrap();
    assert_eq!(y.value().shape(), &[3, 4]);
    assert!(y.value().to_vec().unwrap().iter().all(|&v| v >= 0.0));
}

#[test]
fn quantized_linear_rejects_a_mismatched_bias() {
    let g = gpu();
    let w = Array::from_slice(&g, &fill(8, 4), &[2, 4]).unwrap();
    let mq = QuantizedMatrix::quantize(&w, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    let b = Array::from_slice(&g, &[0.0f32; 3], &[3]).unwrap();
    let err = err_text(QuantizedLinear::new(mq, Some(b)));
    assert!(err.contains("bias"), "{err}");
}

// ── End to end: train → quantize → save → reload → forward ─────────────

/// Per-element |Δy| bound from the round-trip theorem: each weight
/// moves at most s/2, so `|Δy[i,j]| ≤ Σ_k |x[i,k]| · s(k,j)/2` in ℝ.
/// The f32 statement carries a stated multiplicative + absolute slack
/// for the rounding of the matmul itself (both forwards round, and the
/// bound is evaluated in f64).
fn s_half_bound(
    x: &[f32],
    n: usize,
    k: usize,
    m: usize,
    scales: &[f32],
    gr: usize,
    gc: usize,
) -> Vec<f64> {
    let tc = m.div_ceil(gc);
    let mut out = vec![0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let mut b = 0f64;
            for kk in 0..k {
                let s = scales[(kk / gr) * tc + j / gc] as f64;
                b += (x[i * k + kk].abs() as f64) * s / 2.0;
            }
            out[i * m + j] = b;
        }
    }
    out
}

#[test]
fn trained_linear_round_trips_within_the_s_half_bound() {
    let g = gpu();
    let lin = Linear {
        in_dim: 4,
        out_dim: 3,
        bias: true,
    };
    let mut params = Layer::<f32>::init(&lin, &g, Key::new(7)).unwrap();

    // A short real training run so the quantized weights are trained
    // weights, not an init pattern: fit y = x·W* + b* by SGD on MSE.
    let n = 8;
    let x_host = fill(n * 4, 11);
    let x = Array::from_slice(&g, &x_host, &[n, 4]).unwrap();
    let wt = fill(12, 12);
    let bt = fill(3, 13);
    let mut t_host = vec![0f32; n * 3];
    for i in 0..n {
        for j in 0..3 {
            let mut acc = bt[j];
            for kk in 0..4 {
                acc += x_host[i * 4 + kk] * wt[kk * 3 + j];
            }
            t_host[i * 3 + j] = acc;
        }
    }
    let target = Array::from_slice(&g, &t_host, &[n, 3]).unwrap();
    let opt = Sgd::new(0.05);
    let mut state = opt.init(&params).unwrap();
    for _ in 0..20 {
        let tape: Tape<f32> = Tape::new();
        let vars = params.bind(&tape);
        let xv = tape.var(x.shallow_clone());
        let y = lin.apply(&tape, &vars, &xv).unwrap();
        let tv = tape.var(target.shallow_clone());
        let loss = mse_loss(&tape, &y, &tv, Reduction::Mean).unwrap();
        let grads = params.grads_from(&vars, &loss).unwrap();
        let (np, ns) = opt.step(&params, &grads, state).unwrap();
        params = np;
        state = ns;
    }

    let y_f = linear_forward(&lin, &params.w, params.b.as_ref(), &x);

    for (dtype, gran) in [
        (QuantDtype::Int8, Granularity::PerChannel { axis: 1 }),
        (QuantDtype::Int4, Granularity::Group { axis: 0, size: 2 }),
    ] {
        // Quantize the weight leaf, keep the bias f32, save.
        let named = params.named_flatten();
        let entries = quantize_named(&named, |name, arr| {
            (name == "w" && arr.shape().len() == 2).then_some((dtype, gran))
        })
        .unwrap();
        let bytes = save_named(&entries, None).unwrap();

        // Mode A, tree form: the unmodified f32 model definition loads
        // the quantized checkpoint (matching by name).
        let restored = load(&g, &params, &bytes).unwrap();
        let y_a = linear_forward(&lin, &restored.w, restored.b.as_ref(), &x);

        // Mode B: resident codes under QuantizedLinear.
        let loaded = load_named(&g, &bytes).unwrap();
        let mut mq = None;
        let mut bias = None;
        for (name, leaf) in loaded.leaves {
            match (name.as_str(), leaf) {
                ("w", QuantLeaf::Quantized(m)) => mq = Some(m),
                ("b", QuantLeaf::F32(a)) => bias = Some(a),
                (other, _) => panic!("unexpected leaf {other}"),
            }
        }
        let mq = mq.expect("the quantized weight leaf");
        let scales = mq.scales().to_vec().unwrap();
        let (gr, gc) = mq.tile();
        let ql = QuantizedLinear::new(mq, bias).unwrap();
        let y_b = quantized_forward(&ql, &x);

        // Mode A ≡ Mode B bitwise: both spell matmul over the same
        // exactly-dequantized weight.
        assert_eq!(bits(&y_a), bits(&y_b));

        // And the quantized forward sits within the s/2-derived bound
        // of the f32 original (quality gate — the bitwise chain above
        // is the oracle).
        let bound = s_half_bound(&x_host, n, 4, 3, &scales, gr, gc);
        for (i, (&q, &f)) in y_b.iter().zip(&y_f).enumerate() {
            let d = (q as f64 - f as f64).abs();
            assert!(
                d <= bound[i] * 1.001 + 1e-6,
                "{dtype:?}: element {i}: |Δ| = {d:.3e} exceeds bound {:.3e}",
                bound[i]
            );
        }
    }
}

// ── The capability gate ─────────────────────────────────────────────────

#[test]
fn mode_b_int8_gate_is_open_on_every_compiled_backend() {
    // The refusal path (WebGPU: supports_narrow_int() == false) cannot
    // execute in these suites — software, Metal, and Vulkan all report
    // true, and the capability is a device fact, not fakeable through
    // the public API. The refusal message contract is pinned by the
    // unit test beside the gate (src/quant.rs); here we pin the open
    // side: the query says yes and the int8 Mode B load succeeds.
    let g = gpu();
    assert!(g.supports_narrow_int());
    let w = Array::from_slice(&g, &fill(8, 5), &[2, 4]).unwrap();
    let entries = quantize_named(&[("w".to_string(), w)], |_, _| {
        Some((QuantDtype::Int8, Granularity::PerTensor))
    })
    .unwrap();
    let bytes = save_named(&entries, None).unwrap();
    let loaded = load_named(&g, &bytes).unwrap();
    assert!(matches!(&loaded.leaves[0].1, QuantLeaf::Quantized(_)));
}

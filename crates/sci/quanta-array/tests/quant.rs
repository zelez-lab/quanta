//! Quantization — device ≡ host bitwise dequantization, the s/2 round-trip
//! bound, the packed-int4 nibble contract, and loud errors.
//!
//! The correctness split (the scope's contract): everything after
//! quantization is EXACT — the device dequant kernel is pinned bitwise
//! against [`dequantize_host`] on every backend. Because the host twin is
//! backend-independent, `software ≡ host ≡ metal` gives cross-backend
//! bit-reproducibility by transitivity when this suite runs under both
//! features; the pinned-literal tests double-check that with expected
//! values fixed in the source. Quantization itself carries the ONE
//! tolerance, and it is the s/2 theorem, not a fudge factor.

use quanta_array::quant::{
    Granularity, HostCodes, QuantCodes, QuantDtype, QuantError, QuantizedMatrix, dequantize_host,
    quantize_host,
};
use quanta_array::{Array, ArrayError};

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic weights: a sign-alternating ramp of non-round fractions
/// (no rand dep; every run and both backends see identical bytes).
fn weights(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32) * 0.7311 - (n as f32) * 0.317;
            if i % 3 == 0 { -x * 0.13 } else { x * 0.171 }
        })
        .collect()
}

/// Read a device-resident code payload back as its host twin.
fn host_codes(q: &QuantizedMatrix) -> HostCodes {
    match q.codes() {
        QuantCodes::Int8(a) => HostCodes::Int8(a.to_vec().unwrap()),
        QuantCodes::Int4Packed(a) => HostCodes::Int4Packed(a.to_vec().unwrap()),
    }
}

/// All five granularity spellings over a `[r, c]` leaf, group size `g`.
fn spellings(g: u32) -> [Granularity; 5] {
    [
        Granularity::PerTensor,
        Granularity::PerChannel { axis: 0 },
        Granularity::PerChannel { axis: 1 },
        Granularity::Group { axis: 0, size: g },
        Granularity::Group { axis: 1, size: g },
    ]
}

fn assert_bits_eq(dev: &[f32], host: &[f32], ctx: &str) {
    assert_eq!(dev.len(), host.len(), "{ctx}: length");
    for (i, (d, h)) in dev.iter().zip(host).enumerate() {
        assert_eq!(
            d.to_bits(),
            h.to_bits(),
            "{ctx}: element {i}: device {d} vs host {h}"
        );
    }
}

/// Core check: for both dtypes × all five spellings, the device dequant
/// dispatch is bitwise-equal to the host reference.
fn check_device_matches_host(r: usize, c: usize, g: u32) {
    let gpu = gpu();
    let w = weights(r * c);
    let a = Array::from_slice(&gpu, &w, &[r, c]).unwrap();
    for dtype in [QuantDtype::Int8, QuantDtype::Int4] {
        for gran in spellings(g) {
            let ctx = format!("[{r}, {c}] {dtype:?} {gran:?}");
            let qm = QuantizedMatrix::quantize(&a, dtype, gran).unwrap();
            let dev = qm.dequantize().unwrap();
            assert_eq!(dev.shape(), &[r, c], "{ctx}: dequant shape");
            let host =
                dequantize_host(&host_codes(&qm), &qm.scales().to_vec().unwrap(), r, c, gran)
                    .unwrap();
            assert_bits_eq(&dev.to_vec().unwrap(), &host, &ctx);
        }
    }
}

#[test]
fn device_matches_host_divisible_tiles() {
    check_device_matches_host(4, 8, 2);
}

#[test]
fn device_matches_host_partial_tiles() {
    // 5/3 and 7/3 both leave a partial last tile on their axis.
    check_device_matches_host(5, 7, 3);
}

#[test]
fn device_matches_host_multiword_rows() {
    // int4: 13 columns → 2 words per row, 5 pad nibbles in the last word.
    check_device_matches_host(5, 13, 4);
}

#[test]
fn device_matches_host_long_ramp() {
    // ≥ 256 elements so loads at index > 0 exercise the native element
    // width (the recorded narrow-dtype stride trap).
    check_device_matches_host(16, 32, 8);
}

// ── The round-trip bound (host property; the ONE tolerance) ─────────────

/// `|dequant(quantize(x)) − x| ≤ s/2` for every element of every nonzero
/// tile — the ℝ theorem checked in f32 with a stated slack: the division,
/// rounding, and multiply add at most a few ULP on top of s/2 (relative
/// ~2⁻¹⁶ worst case), covered by the 1e-4 factor below.
fn check_round_trip_bound(r: usize, c: usize, g: u32) {
    let w = weights(r * c);
    for dtype in [QuantDtype::Int8, QuantDtype::Int4] {
        for gran in spellings(g) {
            let (codes, scales) = quantize_host(&w, r, c, dtype, gran).unwrap();
            let dq = dequantize_host(&codes, &scales, r, c, gran).unwrap();
            let (gr, gc) = gran.tile(r, c).unwrap();
            let tc = c.div_ceil(gc);
            for i in 0..r * c {
                let s = scales[(i / c / gr) * tc + (i % c) / gc];
                assert!(s > 0.0, "nonzero weights must give a nonzero scale");
                let diff = (dq[i] - w[i]).abs();
                assert!(
                    diff <= s * (0.5 + 1e-4),
                    "[{r}, {c}] {dtype:?} {gran:?}: element {i}: |{} - {}| = {diff} > s/2 (s = {s})",
                    dq[i],
                    w[i],
                );
            }
        }
    }
}

#[test]
fn round_trip_within_half_scale() {
    check_round_trip_bound(4, 8, 2);
    check_round_trip_bound(5, 7, 3);
}

#[test]
fn round_trip_exact_for_exact_multiples() {
    // maxabs = hi → s = 1.0 exactly; integer weights are exact multiples
    // of the scale and must round-trip EXACTLY (no tolerance at all).
    let ints: Vec<f32> = vec![127.0, -127.0, 1.0, -1.0, 64.0, -33.0, 0.0, 100.0];
    let (codes, scales) =
        quantize_host(&ints, 2, 4, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    assert_eq!(scales, vec![1.0]);
    let dq = dequantize_host(&codes, &scales, 2, 4, Granularity::PerTensor).unwrap();
    assert_bits_eq(&dq, &ints, "int8 exact multiples");

    let ints4: Vec<f32> = vec![7.0, -7.0, 1.0, -1.0, 3.0, -2.0, 0.0, 5.0];
    let (codes, scales) =
        quantize_host(&ints4, 2, 4, QuantDtype::Int4, Granularity::PerTensor).unwrap();
    assert_eq!(scales, vec![1.0]);
    let dq = dequantize_host(&codes, &scales, 2, 4, Granularity::PerTensor).unwrap();
    assert_bits_eq(&dq, &ints4, "int4 exact multiples");
}

// ── Pinned literals (rounding + cross-backend anchors) ──────────────────

/// End-to-end device pin of round-ties-even: maxabs 63.5 → s = 0.5 exact;
/// 0.25/s = 0.5 rounds to 0 (even), 0.75/s = 1.5 rounds to 2 (even),
/// 0.26/s = 0.52 rounds to 1. Expected values are source literals, so a
/// run under any backend feature checks against the same bits.
#[test]
fn dequant_ties_to_even_pinned_int8() {
    let gpu = gpu();
    let w = [63.5f32, -63.5, 0.25, 0.26, 0.75, -0.75, 1.0, 0.0];
    let a = Array::from_slice(&gpu, &w, &[2, 4]).unwrap();
    let qm = QuantizedMatrix::quantize(&a, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    assert_eq!(qm.scales().to_vec().unwrap(), vec![0.5]);
    let expected = [63.5f32, -63.5, 0.0, 0.5, 1.0, -1.0, 1.0, 0.0];
    assert_bits_eq(
        &qm.dequantize().unwrap().to_vec().unwrap(),
        &expected,
        "int8 RTE pin",
    );
}

#[test]
fn dequant_ties_to_even_pinned_int4() {
    let gpu = gpu();
    let w = [3.5f32, -3.5, 0.25, 0.26, 0.75, -0.75, 1.0, 0.0];
    let a = Array::from_slice(&gpu, &w, &[2, 4]).unwrap();
    let qm = QuantizedMatrix::quantize(&a, QuantDtype::Int4, Granularity::PerTensor).unwrap();
    assert_eq!(qm.scales().to_vec().unwrap(), vec![0.5]);
    let expected = [3.5f32, -3.5, 0.0, 0.5, 1.0, -1.0, 1.0, 0.0];
    assert_bits_eq(
        &qm.dequantize().unwrap().to_vec().unwrap(),
        &expected,
        "int4 RTE pin",
    );
}

// ── The packed-int4 nibble contract ─────────────────────────────────────

/// The int4 storage layout is pinned by the shipped I4 PackedU32 contract:
/// 8 signed nibbles per u32 word, LOW nibble first (element i → word i/8,
/// nibble i%8) — `quanta_ir::dtype::{int4_pack, int4_unpack}` (documented
/// there as the GPTQ / llama.cpp layout), the pack/unpack round-trip
/// theorems in `specs/verify/lean/Quanta/Dtype/Quant.lean`, and the
/// emitter arms (e.g. `emit_wgsl/ops.rs` I4 load/store). This test pins
/// the quantizer's word bytes against hand-packed literals.
#[test]
fn int4_nibble_order_low_first_pinned() {
    // maxabs 7 → s = 1; codes equal the weights.
    let w = [1.0f32, 2.0, -1.0, -7.0, 7.0, 0.0, 3.0, -3.0];
    let (codes, scales) =
        quantize_host(&w, 1, 8, QuantDtype::Int4, Granularity::PerTensor).unwrap();
    assert_eq!(scales, vec![1.0]);
    // nibbles low→high: 1, 2, F(-1), 9(-7), 7, 0, 3, D(-3)
    assert_eq!(codes, HostCodes::Int4Packed(vec![0xD307_9F21]));
}

#[test]
fn int4_rows_pack_independently_final_word_padded() {
    // 3 columns → 1 word per row, upper 5 nibbles zero-padded; row 1
    // starts a fresh word.
    let w = [1.0f32, -1.0, 2.0, 3.0, -2.0, 7.0];
    let (codes, scales) =
        quantize_host(&w, 2, 3, QuantDtype::Int4, Granularity::PerTensor).unwrap();
    assert_eq!(scales, vec![1.0]);
    assert_eq!(codes, HostCodes::Int4Packed(vec![0x2F1, 0x7E3]));
    // And the packed form round-trips exactly through both dequant paths.
    let dq = dequantize_host(&codes, &scales, 2, 3, Granularity::PerTensor).unwrap();
    assert_bits_eq(&dq, &w, "padded-word host dequant");
    let gpu = gpu();
    let a = Array::from_slice(&gpu, &w, &[2, 3]).unwrap();
    let qm = QuantizedMatrix::quantize(&a, QuantDtype::Int4, Granularity::PerTensor).unwrap();
    assert_bits_eq(&qm.dequantize().unwrap().to_vec().unwrap(), &w, "device");
}

// ── The reader seam (from_parts) ────────────────────────────────────────

#[test]
fn from_parts_dequantizes_int8() {
    let gpu = gpu();
    let codes = Array::from_slice(&gpu, &[2i8, -2, 4, -4], &[2, 2]).unwrap();
    let scales = Array::from_slice(&gpu, &[0.5f32, 0.25], &[2, 1]).unwrap();
    let qm = QuantizedMatrix::from_parts(
        QuantCodes::Int8(codes),
        [2, 2],
        scales,
        Granularity::PerChannel { axis: 0 },
    )
    .unwrap();
    assert_eq!(
        qm.dequantize().unwrap().to_vec().unwrap(),
        vec![1.0, -1.0, 1.0, -1.0]
    );
}

#[test]
fn from_parts_dequantizes_int4_packed() {
    let gpu = gpu();
    // Rows [1, -1] and [2, -2], one padded word each (low nibble first).
    let words = Array::from_slice(&gpu, &[0xF1u32, 0xE2], &[2, 1]).unwrap();
    let scales = Array::from_slice(&gpu, &[0.5f32, 0.25], &[2, 1]).unwrap();
    let qm = QuantizedMatrix::from_parts(
        QuantCodes::Int4Packed(words),
        [2, 2],
        scales,
        Granularity::PerChannel { axis: 0 },
    )
    .unwrap();
    assert_eq!(
        qm.dequantize().unwrap().to_vec().unwrap(),
        vec![0.5, -0.5, 0.5, -0.5]
    );
}

// ── Scale semantics ─────────────────────────────────────────────────────

#[test]
fn all_zero_tile_stores_scale_zero_and_dequantizes_exactly() {
    let gpu = gpu();
    // Group { axis: 1, size: 2 } over [4, 4]: tile (2, 0..2) is all-zero.
    let mut w = weights(16);
    w[2 * 4] = 0.0;
    w[2 * 4 + 1] = 0.0;
    let a = Array::from_slice(&gpu, &w, &[4, 4]).unwrap();
    let gran = Granularity::Group { axis: 1, size: 2 };
    let qm = QuantizedMatrix::quantize(&a, QuantDtype::Int8, gran).unwrap();
    let scales = qm.scales().to_vec().unwrap();
    assert_eq!(qm.scales().shape(), &[4, 2]);
    assert_eq!(scales[2 * 2], 0.0, "the all-zero tile stores s = 0");
    let dev = qm.dequantize().unwrap().to_vec().unwrap();
    assert_eq!(dev[2 * 4].to_bits(), 0.0f32.to_bits());
    assert_eq!(dev[2 * 4 + 1].to_bits(), 0.0f32.to_bits());
    let host = dequantize_host(&host_codes(&qm), &scales, 4, 4, gran).unwrap();
    assert_bits_eq(&dev, &host, "all-zero tile");
}

#[test]
fn group_of_full_extent_equals_per_channel_twin() {
    // Group { axis: 0, size: R } and PerChannel { axis: 1 } denote the
    // same (R, 1) tile — identical scales and codes, bitwise.
    let w = weights(24);
    let ga = Granularity::Group { axis: 0, size: 6 };
    let gb = Granularity::PerChannel { axis: 1 };
    assert_eq!(ga.tile(6, 4).unwrap(), gb.tile(6, 4).unwrap());
    let (ca, sa) = quantize_host(&w, 6, 4, QuantDtype::Int8, ga).unwrap();
    let (cb, sb) = quantize_host(&w, 6, 4, QuantDtype::Int8, gb).unwrap();
    assert_eq!(ca, cb);
    assert_bits_eq(&sa, &sb, "twin scales");
}

// ── Record accessors ────────────────────────────────────────────────────

#[test]
fn record_accessors() {
    let gpu = gpu();
    let w = weights(35);
    let a = Array::from_slice(&gpu, &w, &[5, 7]).unwrap();
    let gran = Granularity::Group { axis: 0, size: 3 };
    let qm = QuantizedMatrix::quantize(&a, QuantDtype::Int4, gran).unwrap();
    assert_eq!(qm.shape(), [5, 7]);
    assert_eq!(qm.dtype(), QuantDtype::Int4);
    assert_eq!(qm.granularity(), gran);
    assert_eq!(qm.tile(), (3, 1));
    assert_eq!(qm.scales().shape(), &[2, 7]);
    match qm.codes() {
        QuantCodes::Int4Packed(words) => assert_eq!(words.shape(), &[5, 1]),
        other => panic!("expected packed int4 codes, got {:?}", other.dtype()),
    }
}

// ── Loud errors ─────────────────────────────────────────────────────────

#[test]
fn rank_errors_are_loud() {
    let gpu = gpu();
    for shape in [vec![6usize], vec![2usize, 2, 2]] {
        let n = shape.iter().product::<usize>();
        let a = Array::from_slice(&gpu, &weights(n), &shape).unwrap();
        let err =
            QuantizedMatrix::quantize(&a, QuantDtype::Int8, Granularity::PerTensor).unwrap_err();
        match &err {
            ArrayError::Quant(QuantError::Rank { shape: got }) => assert_eq!(got, &shape),
            other => panic!("expected Rank error, got {other:?}"),
        }
        assert!(
            format!("{err}").contains("rank-2"),
            "message states the rule"
        );
    }
    // Zero extents violate the shape model (unreachable through Array,
    // enforced by the host twins over raw slices).
    assert!(matches!(
        quantize_host(&[], 0, 4, QuantDtype::Int8, Granularity::PerTensor),
        Err(ArrayError::Quant(QuantError::Rank { .. }))
    ));
}

#[test]
fn granularity_errors_are_loud() {
    let gpu = gpu();
    let a = Array::from_slice(&gpu, &weights(16), &[4, 4]).unwrap();
    let bad = [
        Granularity::PerChannel { axis: 2 },
        Granularity::Group { axis: 3, size: 2 },
        Granularity::Group { axis: 0, size: 0 },
        Granularity::Group { axis: 1, size: 99 },
    ];
    for gran in bad {
        let err = QuantizedMatrix::quantize(&a, QuantDtype::Int8, gran).unwrap_err();
        assert!(
            matches!(err, ArrayError::Quant(QuantError::Granularity { .. })),
            "{gran:?} must be a Granularity error, got {err:?}"
        );
    }
    let msg = format!(
        "{}",
        Granularity::Group { axis: 1, size: 99 }
            .tile(4, 4)
            .unwrap_err()
    );
    assert!(
        msg.contains("99") && msg.contains('4'),
        "names the offender: {msg}"
    );
}

#[test]
fn from_parts_grid_mismatches_are_loud() {
    let gpu = gpu();
    // Codes shape disagrees with the stated logical shape.
    let codes = Array::from_slice(&gpu, &[1i8, 2, 3, 4], &[2, 2]).unwrap();
    let scales = Array::from_slice(&gpu, &[1.0f32], &[1, 1]).unwrap();
    let err = QuantizedMatrix::from_parts(
        QuantCodes::Int8(codes),
        [4, 2],
        scales,
        Granularity::PerTensor,
    )
    .unwrap_err();
    assert!(matches!(err, ArrayError::Quant(QuantError::Grid { .. })));
    assert!(format!("{err}").contains("[4, 2]"), "names both shapes");

    // Scales shape disagrees with the granularity's grid.
    let codes = Array::from_slice(&gpu, &[1i8, 2, 3, 4], &[2, 2]).unwrap();
    let scales = Array::from_slice(&gpu, &[1.0f32, 2.0], &[1, 2]).unwrap();
    let err = QuantizedMatrix::from_parts(
        QuantCodes::Int8(codes),
        [2, 2],
        scales,
        Granularity::PerChannel { axis: 0 },
    )
    .unwrap_err();
    assert!(matches!(err, ArrayError::Quant(QuantError::Grid { .. })));

    // Packed int4 word count disagrees with ⌈C/8⌉ (17 cols → 3 words/row).
    let words = Array::from_slice(&gpu, &[0u32, 0, 0, 0], &[2, 2]).unwrap();
    let scales = Array::from_slice(&gpu, &[1.0f32], &[1, 1]).unwrap();
    let err = QuantizedMatrix::from_parts(
        QuantCodes::Int4Packed(words),
        [2, 17],
        scales,
        Granularity::PerTensor,
    )
    .unwrap_err();
    assert!(matches!(err, ArrayError::Quant(QuantError::Grid { .. })));
}

#[test]
fn host_twin_length_mismatches_are_loud() {
    let w = weights(8);
    assert!(matches!(
        quantize_host(&w, 3, 4, QuantDtype::Int8, Granularity::PerTensor),
        Err(ArrayError::LengthMismatch {
            expected: 12,
            got: 8
        })
    ));
    let (codes, scales) =
        quantize_host(&w, 2, 4, QuantDtype::Int8, Granularity::PerTensor).unwrap();
    assert!(matches!(
        dequantize_host(&codes, &scales, 2, 5, Granularity::PerTensor),
        Err(ArrayError::LengthMismatch { .. })
    ));
    assert!(matches!(
        dequantize_host(&codes, &[1.0, 2.0], 2, 4, Granularity::PerTensor),
        Err(ArrayError::LengthMismatch { .. })
    ));
}

/// The checkpoint-facing variants (constructed by the safetensors loader
/// in quanta-nn) keep the message contract: every message names the leaf
/// and states the problem or the workaround.
#[test]
fn checkpoint_error_messages_name_the_problem() {
    let e = QuantError::Format {
        leaf: "blk0.w".into(),
        what: "metadata names it but `blk0.w.qs` is missing".into(),
    };
    assert!(format!("{e}").contains("blk0.w"));

    let e = QuantError::NotSupported {
        leaf: "blk0.w".into(),
        backend: "webgpu".into(),
    };
    let msg = format!("{e}");
    assert!(msg.contains("blk0.w") && msg.contains("webgpu") && msg.contains("f32"));

    let e = QuantError::Scale {
        leaf: "blk0.w".into(),
        tile: 3,
        value: f32::NAN,
    };
    let msg = format!("{e}");
    assert!(msg.contains("blk0.w") && msg.contains('3'));
}

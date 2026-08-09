//! Typed-layer tests for npy interop: the dtype matrix (save / load /
//! load_dyn over all ten element types), `<f2` widening, big-endian
//! byteswap, Fortran-order permute, bool validation, rank-0, and the
//! error taxonomy through the public API. The format substrate (header
//! grammar, ZIP, inflate) is covered by `npy_format.rs`.
//!
//! numpy-generated fixture tests at the bottom SKIP LOUDLY while the
//! fixtures are absent (generation is a pending maintainer step — see
//! `tests/fixtures/npy/gen_fixtures.py`); they arm automatically once
//! the fixtures are committed.

use quanta_array::npy::{self, NpyArray, NpyError, NpyScalar};
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

/// Expect a failure and unwrap its `ArrayError::Npy` layer. (Takes the
/// whole `Result` — `Array` has no `Debug`, so `unwrap_err` can't.)
fn npy_err<T>(r: Result<T, ArrayError>) -> NpyError {
    match r {
        Err(ArrayError::Npy(e)) => e,
        Err(other) => panic!("expected ArrayError::Npy, got: {other}"),
        Ok(_) => panic!("expected an error, got Ok"),
    }
}

/// Hand-build a complete npy file: v1.0 preamble with an EXACT-length
/// (unpadded) header — loads must trust the length field, never assume
/// the 64-byte alignment our own writer produces — plus raw data bytes.
fn npy_file(dict: &str, data: &[u8]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(1);
    b.push(0);
    b.extend_from_slice(&((dict.len() + 1) as u16).to_le_bytes());
    b.extend_from_slice(dict.as_bytes());
    b.push(b'\n');
    b.extend_from_slice(data);
    b
}

// ── Spec-byte-exact saves (hand-built reference bytes) ──────────────────

/// The reference 128-byte preamble for `descr`, C-order, built by hand
/// from the spec: magic, v1.0, u16 length, the dict, space padding, the
/// final `\n` — the data section lands 64-byte aligned like modern
/// numpy's.
fn reference_header(descr: &str, shape_text: &str) -> Vec<u8> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_text}, }}");
    let mut b = Vec::new();
    b.extend_from_slice(b"\x93NUMPY");
    b.push(1);
    b.push(0);
    b.extend_from_slice(&118u16.to_le_bytes());
    b.extend_from_slice(dict.as_bytes());
    b.resize(127, b' ');
    b.push(b'\n');
    assert_eq!(b.len(), 128);
    b
}

#[test]
fn save_f32_matches_reference_bytes() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let mut expect = reference_header("<f4", "(2, 3)");
    for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
        expect.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = npy::save(&a).unwrap();
    assert_eq!(bytes, expect);
    // And the reference bytes load back to the same values.
    let b = npy::load::<f32>(&g, &expect).unwrap();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn save_u8_matches_reference_bytes() {
    let g = gpu();
    let a = Array::from_slice(&g, &[7u8, 0, 255, 128], &[4]).unwrap();
    let mut expect = reference_header("|u1", "(4,)");
    expect.extend_from_slice(&[7, 0, 255, 128]);
    assert_eq!(npy::save(&a).unwrap(), expect);
    let b = npy::load::<u8>(&g, &expect).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![7, 0, 255, 128]);
}

#[test]
fn save_i16_matches_reference_bytes() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1i16, -2, 300, -300], &[2, 2]).unwrap();
    let mut expect = reference_header("<i2", "(2, 2)");
    // Little-endian two's complement, written out by hand:
    // 1 = 01 00, -2 = FE FF, 300 = 0x012C = 2C 01, -300 = 0xFED4 = D4 FE.
    expect.extend_from_slice(&[0x01, 0x00, 0xFE, 0xFF, 0x2C, 0x01, 0xD4, 0xFE]);
    assert_eq!(npy::save(&a).unwrap(), expect);
    let b = npy::load::<i16>(&g, &expect).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![1, -2, 300, -300]);
}

// ── Round-trip: all ten element types, bit-identical ────────────────────

fn roundtrip<T>(g: &quanta::Gpu, values: &[T], shape: &[usize])
where
    T: NpyScalar + PartialEq + std::fmt::Debug,
    Array<T>: TryFrom<NpyArray, Error = ArrayError>,
{
    let a = Array::from_slice(g, values, shape).unwrap();
    let bytes = npy::save(&a).unwrap();
    // Typed route.
    let b = npy::load::<T>(g, &bytes).unwrap();
    assert_eq!(b.shape(), shape);
    assert_eq!(b.to_vec().unwrap(), values);
    // Dynamic route: load_dyn preserves the dtype, TryFrom unwraps it.
    let d = npy::load_dyn(g, &bytes).unwrap();
    assert_eq!(d.shape(), shape);
    let c: Array<T> = d.try_into().unwrap();
    assert_eq!(c.to_vec().unwrap(), values);
}

#[test]
fn roundtrip_all_ten_dtypes() {
    let g = gpu();
    roundtrip::<f32>(
        &g,
        &[0.0, -1.5, 3.25e10, f32::MAX, f32::MIN_POSITIVE, 1.0],
        &[2, 3],
    );
    roundtrip::<f64>(
        &g,
        &[0.0, -1.5, 3.25e300, f64::MAX, f64::MIN_POSITIVE, 1.0],
        &[3, 2],
    );
    roundtrip::<i32>(&g, &[i32::MIN, -1, 0, 1, i32::MAX], &[5]);
    roundtrip::<u32>(&g, &[0, 1, u32::MAX], &[3]);
    roundtrip::<i64>(&g, &[i64::MIN, -1, 0, i64::MAX], &[2, 2]);
    roundtrip::<u64>(&g, &[0, 1, u64::MAX], &[3]);
    roundtrip::<u8>(&g, &[0, 1, 127, 255], &[4]);
    roundtrip::<i8>(&g, &[i8::MIN, -1, 0, i8::MAX], &[2, 2]);
    roundtrip::<u16>(&g, &[0, 300, u16::MAX], &[3]);
    roundtrip::<i16>(&g, &[i16::MIN, -300, 0, i16::MAX], &[4]);
}

#[test]
fn roundtrip_float_specials_bit_identical() {
    let g = gpu();
    // NaN breaks PartialEq — compare bit patterns instead. A subnormal,
    // both infinities, -0.0 and a payload NaN must survive untouched.
    let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 1e-40];
    let a = Array::from_slice(&g, &values, &[5]).unwrap();
    let bytes = npy::save(&a).unwrap();
    let back = npy::load::<f32>(&g, &bytes).unwrap().to_vec().unwrap();
    let got: Vec<u32> = back.iter().map(|v| v.to_bits()).collect();
    let expect: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, expect);

    let values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0, 1e-310];
    let a = Array::from_slice(&g, &values, &[5]).unwrap();
    let bytes = npy::save(&a).unwrap();
    let back = npy::load::<f64>(&g, &bytes).unwrap().to_vec().unwrap();
    let got: Vec<u64> = back.iter().map(|v| v.to_bits()).collect();
    let expect: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, expect);
}

// ── Views serialize their logical content ───────────────────────────────

#[test]
fn transposed_view_saves_as_its_contiguous_copy() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap(); // strided view, shape [3, 2]
    let logical = t.to_vec().unwrap();
    assert_eq!(logical, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let contig = Array::from_slice(&g, &logical, &[3, 2]).unwrap();
    assert_eq!(npy::save(&t).unwrap(), npy::save(&contig).unwrap());
}

#[test]
fn narrowed_and_broadcast_views_save_logical_content() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1u16, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let n = a.narrow(1, 1, 2).unwrap(); // offset strided view, shape [2, 2]
    let contig = Array::from_slice(&g, &[2u16, 3, 5, 6], &[2, 2]).unwrap();
    assert_eq!(npy::save(&n).unwrap(), npy::save(&contig).unwrap());

    let row = Array::from_slice(&g, &[1.0f32, 2.0, 3.0], &[1, 3]).unwrap();
    let b = row.broadcast_to(&[2, 3]).unwrap(); // zero-stride view
    let contig = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]).unwrap();
    assert_eq!(npy::save(&b).unwrap(), npy::save(&contig).unwrap());
}

// ── <f2 widening: the one documented widening, exact ────────────────────

/// Half-precision bit patterns and their exact f32 images, hand-derived:
/// normals, both subnormal extremes, ±inf, a NaN (payload preserved),
/// and -0.0 — the doc's f16-exactness claim, checked at the bit level.
const F16_CASES: [(u16, u32); 9] = [
    (0x3C00, 0x3F80_0000), // 1.0
    (0xC100, 0xC020_0000), // -2.5
    (0x0001, 0x3380_0000), // smallest subnormal = 2^-24
    (0x03FF, 0x387F_C000), // largest subnormal = 1023 * 2^-24
    (0x0400, 0x3880_0000), // smallest normal = 2^-14
    (0x7C00, 0x7F80_0000), // +inf
    (0xFC00, 0xFF80_0000), // -inf
    (0x7E00, 0x7FC0_0000), // NaN (quiet bit carried into the f32 payload)
    (0x8000, 0x8000_0000), // -0.0
];

#[test]
fn f16_widens_to_f32_exactly() {
    let g = gpu();
    let mut data = Vec::new();
    for (h, _) in F16_CASES {
        data.extend_from_slice(&h.to_le_bytes());
    }
    let bytes = npy_file(
        "{'descr': '<f2', 'fortran_order': False, 'shape': (9,), }",
        &data,
    );
    let expect: Vec<u32> = F16_CASES.iter().map(|&(_, f)| f).collect();
    // Typed route: load::<f32> accepts <f2.
    let a = npy::load::<f32>(&g, &bytes).unwrap();
    let got: Vec<u32> = a.to_vec().unwrap().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, expect);
    // Dynamic route: <f2 lands in the F32 variant.
    match npy::load_dyn(&g, &bytes).unwrap() {
        NpyArray::F32(a) => {
            let got: Vec<u32> = a.to_vec().unwrap().iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, expect);
        }
        other => panic!("expected F32, got {other:?}"),
    }
}

#[test]
fn f16_big_endian_widens_too() {
    let g = gpu();
    let mut data = Vec::new();
    for (h, _) in F16_CASES {
        data.extend_from_slice(&h.to_be_bytes());
    }
    let bytes = npy_file(
        "{'descr': '>f2', 'fortran_order': False, 'shape': (9,), }",
        &data,
    );
    let a = npy::load::<f32>(&g, &bytes).unwrap();
    let got: Vec<u32> = a.to_vec().unwrap().iter().map(|v| v.to_bits()).collect();
    let expect: Vec<u32> = F16_CASES.iter().map(|&(_, f)| f).collect();
    assert_eq!(got, expect);
}

#[test]
fn f16_widens_into_f32_only() {
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '<f2', 'fortran_order': False, 'shape': (1,), }",
        &0x3C00u16.to_le_bytes(),
    );
    let e = npy_err(npy::load::<f64>(&g, &bytes));
    assert!(matches!(e, NpyError::DtypeMismatch { .. }), "{e:?}");
    let msg = e.to_string();
    assert!(msg.contains("<f2") && msg.contains("<f8"), "{msg}");
}

// ── Big-endian: byteswap-on-load, hand-built > files ────────────────────

#[test]
fn big_endian_f4_loads_byteswapped() {
    let g = gpu();
    // 1.0 = 3F 80 00 00 and -2.5 = C0 20 00 00, big-endian by hand.
    let data = [0x3F, 0x80, 0x00, 0x00, 0xC0, 0x20, 0x00, 0x00];
    let bytes = npy_file(
        "{'descr': '>f4', 'fortran_order': False, 'shape': (2,), }",
        &data,
    );
    assert_eq!(
        npy::load::<f32>(&g, &bytes).unwrap().to_vec().unwrap(),
        vec![1.0, -2.5]
    );
}

#[test]
fn big_endian_i4_loads_byteswapped() {
    let g = gpu();
    // 0x01020304 = 01 02 03 04 and -2 = FF FF FF FE, big-endian by hand.
    let data = [0x01, 0x02, 0x03, 0x04, 0xFF, 0xFF, 0xFF, 0xFE];
    let bytes = npy_file(
        "{'descr': '>i4', 'fortran_order': False, 'shape': (2,), }",
        &data,
    );
    assert_eq!(
        npy::load::<i32>(&g, &bytes).unwrap().to_vec().unwrap(),
        vec![0x01020304, -2]
    );
    // The dynamic route byteswaps identically.
    match npy::load_dyn(&g, &bytes).unwrap() {
        NpyArray::I32(a) => assert_eq!(a.to_vec().unwrap(), vec![0x01020304, -2]),
        other => panic!("expected I32, got {other:?}"),
    }
}

#[test]
fn big_endian_i2_loads_byteswapped() {
    let g = gpu();
    // 300 = 01 2C and -1 = FF FF, big-endian by hand.
    let data = [0x01, 0x2C, 0xFF, 0xFF];
    let bytes = npy_file(
        "{'descr': '>i2', 'fortran_order': False, 'shape': (2,), }",
        &data,
    );
    assert_eq!(
        npy::load::<i16>(&g, &bytes).unwrap().to_vec().unwrap(),
        vec![300, -1]
    );
}

// ── Fortran order: load-with-transpose ──────────────────────────────────

#[test]
fn fortran_order_loads_as_row_major() {
    let g = gpu();
    // Logical [[1, 2, 3], [4, 5, 6]] stored column-major: 1 4 2 5 3 6.
    let mut data = Vec::new();
    for v in [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = npy_file(
        "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }",
        &data,
    );
    let a = npy::load::<f32>(&g, &bytes).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert!(a.is_contiguous());
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn fortran_order_rank3_permutes_every_axis() {
    let g = gpu();
    // v[i][j][k] = 100i + 10j + k over shape (2, 3, 4); the F-order
    // linear index of (i, j, k) is i + 2j + 6k.
    let (d0, d1, d2) = (2usize, 3usize, 4usize);
    let mut fdata = vec![0i32; d0 * d1 * d2];
    let mut logical = Vec::new();
    for i in 0..d0 {
        for j in 0..d1 {
            for k in 0..d2 {
                let v = (100 * i + 10 * j + k) as i32;
                fdata[i + d0 * j + d0 * d1 * k] = v;
                logical.push(v);
            }
        }
    }
    let mut data = Vec::new();
    for v in &fdata {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = npy_file(
        "{'descr': '<i4', 'fortran_order': True, 'shape': (2, 3, 4), }",
        &data,
    );
    let a = npy::load::<i32>(&g, &bytes).unwrap();
    assert_eq!(a.shape(), &[2, 3, 4]);
    assert_eq!(a.to_vec().unwrap(), logical);
}

#[test]
fn fortran_order_composes_with_byteswap() {
    let g = gpu();
    // Logical [[1, 2], [3, 4]] as >f4 column-major: 1 3 2 4, big-endian.
    let mut data = Vec::new();
    for v in [1.0f32, 3.0, 2.0, 4.0] {
        data.extend_from_slice(&v.to_be_bytes());
    }
    let bytes = npy_file(
        "{'descr': '>f4', 'fortran_order': True, 'shape': (2, 2), }",
        &data,
    );
    let a = npy::load::<f32>(&g, &bytes).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn fortran_order_rank1_is_order_invariant() {
    let g = gpu();
    let mut data = Vec::new();
    for v in [5i64, 6, 7] {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let bytes = npy_file(
        "{'descr': '<i8', 'fortran_order': True, 'shape': (3,), }",
        &data,
    );
    assert_eq!(
        npy::load::<i64>(&g, &bytes).unwrap().to_vec().unwrap(),
        vec![5, 6, 7]
    );
}

// ── Bool: |b1 loads validated, never written ────────────────────────────

#[test]
fn bool_loads_as_validated_u8() {
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '|b1', 'fortran_order': False, 'shape': (4,), }",
        &[1, 0, 1, 1],
    );
    match npy::load_dyn(&g, &bytes).unwrap() {
        NpyArray::U8(a) => {
            assert_eq!(a.to_vec().unwrap(), vec![1, 0, 1, 1]);
        }
        other => panic!("expected U8, got {other:?}"),
    }
}

#[test]
fn bool_byte_outside_01_is_loud_with_offset() {
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '|b1', 'fortran_order': False, 'shape': (4,), }",
        &[1, 0, 2, 1],
    );
    let data_offset = npy::header(&bytes).unwrap().data_offset;
    let e = npy_err(npy::load_dyn(&g, &bytes));
    match &e {
        NpyError::BoolValue { at } => assert_eq!(*at, data_offset + 2),
        other => panic!("expected BoolValue, got {other:?}"),
    }
    let msg = e.to_string();
    assert!(msg.contains(&(data_offset + 2).to_string()), "{msg}");
    assert!(msg.contains("corrupt"), "{msg}");
}

#[test]
fn typed_load_of_bool_is_a_mismatch() {
    // The typed loader is exact-width: |b1 never silently becomes u8 —
    // load_dyn is the validated route.
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '|b1', 'fortran_order': False, 'shape': (2,), }",
        &[1, 0],
    );
    let e = npy_err(npy::load::<u8>(&g, &bytes));
    let msg = e.to_string();
    assert!(msg.contains("|b1") && msg.contains("|u1"), "{msg}");
}

// ── Rank 0 ──────────────────────────────────────────────────────────────

#[test]
fn rank0_roundtrips() {
    let g = gpu();
    let a = Array::from_slice(&g, &[2.5f32], &[]).unwrap();
    let bytes = npy::save(&a).unwrap();
    // The header writes numpy's `()` form.
    assert!(
        std::str::from_utf8(&bytes[10..128])
            .unwrap()
            .contains("'shape': (), }")
    );
    let b = npy::load::<f32>(&g, &bytes).unwrap();
    assert_eq!(b.shape(), &[] as &[usize]);
    assert_eq!(b.to_vec().unwrap(), vec![2.5]);
    let d = npy::load_dyn(&g, &bytes).unwrap();
    assert_eq!(d.shape(), &[] as &[usize]);
}

// ── NpyArray: accessors and conversions ─────────────────────────────────

#[test]
fn npy_array_accessors_and_conversions() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f32, 2.0], &[2]).unwrap();
    let d = NpyArray::from(a);
    assert_eq!(d.dtype(), "<f4");
    assert_eq!(d.shape(), &[2]);
    assert_eq!(format!("{d:?}"), "NpyArray(<f4 [2])");

    // The typed unwrap refuses a different dtype, loudly.
    let e = npy_err(Array::<i32>::try_from(d));
    assert!(matches!(e, NpyError::DtypeMismatch { .. }), "{e:?}");
    let msg = e.to_string();
    assert!(
        msg.contains("<f4") && msg.contains("<i4") && msg.contains("load_dyn"),
        "{msg}"
    );

    let b = Array::from_slice(&g, &[3u16, 4, 5], &[3]).unwrap();
    assert_eq!(NpyArray::from(b).dtype(), "<u2");
}

// ── Error taxonomy through the public typed API ─────────────────────────

#[test]
fn typed_mismatch_names_both_sides_and_the_workaround() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1.0f64, 2.0], &[2]).unwrap();
    let bytes = npy::save(&a).unwrap();
    let e = npy_err(npy::load::<f32>(&g, &bytes));
    assert!(
        matches!(&e, NpyError::DtypeMismatch { file, requested }
            if file == "<f8" && requested == "<f4"),
        "{e:?}"
    );
    let msg = e.to_string();
    assert!(msg.contains("load_dyn"), "{msg}");
}

#[test]
fn empty_shape_is_a_specific_loud_error() {
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '<f4', 'fortran_order': False, 'shape': (0, 4), }",
        &[],
    );
    // Introspection still reports the header as written…
    assert_eq!(npy::header(&bytes).unwrap().shape, vec![0, 4]);
    // …but loading is the specific shape-model error, on both routes.
    for e in [
        npy_err(npy::load::<f32>(&g, &bytes)),
        npy_err(npy::load_dyn(&g, &bytes)),
    ] {
        assert!(
            matches!(&e, NpyError::EmptyShape { shape } if shape == &[0, 4]),
            "{e:?}"
        );
        let msg = e.to_string();
        assert!(msg.contains("[0, 4]") && msg.contains("zero"), "{msg}");
    }
}

#[test]
fn data_length_mismatch_is_loud() {
    let g = gpu();
    let mut data = vec![0u8; 20]; // (2, 3) f4 needs 24
    data[0] = 1;
    let bytes = npy_file(
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
        &data,
    );
    let e = npy_err(npy::load::<f32>(&g, &bytes));
    assert!(
        matches!(
            e,
            NpyError::DataLength {
                expected: 24,
                got: 20
            }
        ),
        "{e:?}"
    );
    let msg = e.to_string();
    assert!(msg.contains("24") && msg.contains("20"), "{msg}");
}

#[test]
fn unsupported_descrs_reject_with_reasons_via_load() {
    let g = gpu();
    for (descr, reason) in [("<c8", "complex"), ("|O", "pickle"), ("|S8", "string")] {
        let bytes = npy_file(
            &format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (2,), }}"),
            &[],
        );
        let e = npy_err(npy::load_dyn(&g, &bytes));
        assert!(matches!(e, NpyError::Dtype { .. }), "{descr}: {e:?}");
        let msg = e.to_string();
        assert!(msg.contains(descr) && msg.contains(reason), "{msg}");
    }
}

#[test]
fn equals_byte_order_is_loud_via_load() {
    let g = gpu();
    let bytes = npy_file(
        "{'descr': '=f4', 'fortran_order': False, 'shape': (1,), }",
        &[0, 0, 0, 0],
    );
    let e = npy_err(npy::load::<f32>(&g, &bytes));
    assert!(matches!(e, NpyError::ByteOrder { .. }), "{e:?}");
    assert!(e.to_string().contains("'='"), "{e}");
}

#[test]
fn magic_and_version_faults_surface_through_load() {
    let g = gpu();
    let e = npy_err(npy::load_dyn(&g, b"junk, not an npy"));
    assert!(matches!(e, NpyError::Magic { .. }), "{e:?}");

    let mut b = b"\x93NUMPY".to_vec();
    b.extend_from_slice(&[7, 0, 0, 0]);
    let e = npy_err(npy::load::<f32>(&g, &b));
    assert!(
        matches!(e, NpyError::Version { major: 7, minor: 0 }),
        "{e:?}"
    );
}

// ── numpy-generated fixtures (arm once gen_fixtures.py has run) ─────────

/// Read a checked-in numpy fixture, or skip the test LOUDLY while the
/// maintainer step (running `gen_fixtures.py` with the pinned numpy) is
/// pending. The committed bytes are the interchange ground truth; these
/// tests arm automatically once they exist.
fn fixture(name: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npy")
        .join(name);
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(_) => {
            eprintln!(
                "SKIPPED: numpy fixture {name} is absent — run \
                 tests/fixtures/npy/gen_fixtures.py with the pinned numpy and commit \
                 the outputs to arm this test"
            );
            None
        }
    }
}

#[test]
fn numpy_fixture_f32_loads_and_saves_byte_identical() {
    let Some(bytes) = fixture("f32.npy") else {
        return;
    };
    let g = gpu();
    let a = npy::load::<f32>(&g, &bytes).unwrap();
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.to_vec().unwrap(), vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25]);
    // Our writer matches modern numpy's bytes exactly (v1.0, 64-byte
    // aligned data, `<` descr, trailing newline).
    assert_eq!(npy::save(&a).unwrap(), bytes);
}

#[test]
fn numpy_fixture_dtype_matrix_loads() {
    let g = gpu();
    let quarter = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25];
    if let Some(b) = fixture("f64.npy") {
        let a = npy::load::<f64>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), quarter.map(|v| v as f64).to_vec());
    }
    if let Some(b) = fixture("i32.npy") {
        let a = npy::load::<i32>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![-3, -2, -1, 0, 1, 2]);
    }
    if let Some(b) = fixture("u32.npy") {
        let a = npy::load::<u32>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![0, 1, 2, 3, 4, 5]);
    }
    if let Some(b) = fixture("i64.npy") {
        let a = npy::load::<i64>(&g, &b).unwrap();
        let expect: Vec<i64> = (0..6).map(|i| (i - 3) << 33).collect();
        assert_eq!(a.to_vec().unwrap(), expect);
    }
    if let Some(b) = fixture("u64.npy") {
        let a = npy::load::<u64>(&g, &b).unwrap();
        let expect: Vec<u64> = (0..6u64).map(|i| i << 33).collect();
        assert_eq!(a.to_vec().unwrap(), expect);
    }
    if let Some(b) = fixture("u8.npy") {
        let a = npy::load::<u8>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![0, 1, 2, 3, 4, 5]);
    }
    if let Some(b) = fixture("i8.npy") {
        let a = npy::load::<i8>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![-3, -2, -1, 0, 1, 2]);
    }
    if let Some(b) = fixture("u16.npy") {
        let a = npy::load::<u16>(&g, &b).unwrap();
        let expect: Vec<u16> = (0..6u16).map(|i| i * 300).collect();
        assert_eq!(a.to_vec().unwrap(), expect);
    }
    if let Some(b) = fixture("i16.npy") {
        let a = npy::load::<i16>(&g, &b).unwrap();
        let expect: Vec<i16> = (0..6i16).map(|i| (i - 3) * 300).collect();
        assert_eq!(a.to_vec().unwrap(), expect);
    }
    if let Some(b) = fixture("f16.npy") {
        // f16 upconverts: typed via load::<f32>, dynamic into F32.
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), quarter.to_vec());
        assert!(matches!(npy::load_dyn(&g, &b).unwrap(), NpyArray::F32(_)));
    }
    if let Some(b) = fixture("bool.npy") {
        match npy::load_dyn(&g, &b).unwrap() {
            NpyArray::U8(a) => assert_eq!(a.to_vec().unwrap(), vec![1, 0, 1, 1]),
            other => panic!("expected U8, got {other:?}"),
        }
    }
}

#[test]
fn numpy_fixture_shape_edges_load() {
    let g = gpu();
    if let Some(b) = fixture("scalar_0d.npy") {
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.shape(), &[] as &[usize]);
        assert_eq!(a.to_vec().unwrap(), vec![2.5]);
    }
    if let Some(b) = fixture("vec3.npy") {
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.shape(), &[3]);
        assert_eq!(a.to_vec().unwrap(), vec![0.0, 1.0, 2.0]);
    }
    if let Some(b) = fixture("fortran.npy") {
        assert!(npy::header(&b).unwrap().fortran_order);
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.to_vec().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
    if let Some(b) = fixture("v2_long_header.npy") {
        let h = npy::header(&b).unwrap();
        assert_eq!(h.version, (2, 0));
        assert_eq!(h.shape.len(), 20_000);
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![0.0]);
    }
    if let Some(b) = fixture("big_endian.npy") {
        assert_eq!(npy::header(&b).unwrap().descr, ">f4");
        let a = npy::load::<f32>(&g, &b).unwrap();
        assert_eq!(a.to_vec().unwrap(), vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25]);
    }
}

#[test]
fn numpy_fixture_object_dtype_is_a_loud_error() {
    let Some(bytes) = fixture("object_error.npy") else {
        return;
    };
    let g = gpu();
    let e = npy_err(npy::load_dyn(&g, &bytes));
    assert!(matches!(e, NpyError::Dtype { .. }), "{e:?}");
    assert!(e.to_string().contains("pickle"), "{e}");
}

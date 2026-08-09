//! Narrow dtypes (u8 / i8 / u16 / i16) as first-class element types.
//!
//! (`tests/narrow.rs` is the axis-slicing *view* suite — this file is about
//! narrow **dtypes**.) Pins the declared contract: native 1-/2-byte storage
//! round-trips, two's-complement wrapping (mod 2^w) arithmetic asserted as
//! intended behaviour, signed vs unsigned compares, the astype matrix
//! (sign/zero-extension, truncation, same-width reinterpret, float
//! boundaries), views over narrow storage, and the in-`T` wrapping axis
//! reductions. Every expected value is computed host-side with Rust
//! `wrapping_*` arithmetic, and asserted bit-exact — the same file must pass
//! identically on the software, Metal and Vulkan lanes.

use quanta_array::Array;

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

// ── round-trips: from_slice / to_vec at native width ─────────────────────

#[test]
fn roundtrip_u8_full_range() {
    let g = gpu();
    let data: Vec<u8> = (0..=255).collect();
    let a = Array::from_slice(&g, &data, &[256]).unwrap();
    assert_eq!(a.to_vec().unwrap(), data);
}

#[test]
fn roundtrip_i8_full_range() {
    let g = gpu();
    let data: Vec<i8> = (-128..=127).collect();
    let a = Array::from_slice(&g, &data, &[256]).unwrap();
    assert_eq!(a.to_vec().unwrap(), data);
}

#[test]
fn roundtrip_u16_boundaries() {
    let g = gpu();
    let data = vec![0u16, 1, 255, 256, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF];
    let a = Array::from_slice(&g, &data, &[8]).unwrap();
    assert_eq!(a.to_vec().unwrap(), data);
}

#[test]
fn roundtrip_i16_boundaries() {
    let g = gpu();
    let data = vec![0i16, 1, -1, 255, 256, i16::MAX, i16::MIN, -256];
    let a = Array::from_slice(&g, &data, &[8]).unwrap();
    assert_eq!(a.to_vec().unwrap(), data);
}

// ── the stride guard: length ≥ 256 ramps through a device op ─────────────
//
// A single-element case cannot see a stride bug (the recorded bf16 trap:
// wrong element width reads plausibly at index 0). A 300-element ramp
// through a device kernel + readback cannot survive an off-by-stride
// addressing error.

#[test]
fn stride_guard_u8_ramp_through_device_op() {
    let g = gpu();
    let data: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
    let a = Array::from_slice(&g, &data, &[300]).unwrap();
    let z = Array::<u8>::zeros(&g, &[300]).unwrap();
    // out = a + 0, element by element on the device.
    let out = a.add(&z).unwrap();
    assert_eq!(out.to_vec().unwrap(), data);
}

#[test]
fn stride_guard_u16_high_bytes_through_device_op() {
    let g = gpu();
    // i*257 exercises both bytes of every element (0x0000, 0x0101, 0x0202…).
    let data: Vec<u16> = (0..300u32).map(|i| (i * 257) as u16).collect();
    let a = Array::from_slice(&g, &data, &[300]).unwrap();
    let z = Array::<u16>::zeros(&g, &[300]).unwrap();
    let out = a.add(&z).unwrap();
    assert_eq!(out.to_vec().unwrap(), data);
}

#[test]
fn stride_guard_offset_view_gathers_correctly() {
    let g = gpu();
    let data: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
    let a = Array::from_slice(&g, &data, &[300]).unwrap();
    // A base_offset > 0 window: elements [37, 37+200).
    let w = a.narrow(0, 37, 200).unwrap();
    assert_eq!(w.to_vec().unwrap(), data[37..237].to_vec());
    // The same window gathered contiguous ON THE DEVICE, then a ufunc.
    let wc = w.contiguous().unwrap();
    let doubled = wc.add(&wc).unwrap();
    let expect: Vec<u8> = data[37..237].iter().map(|x| x.wrapping_add(*x)).collect();
    assert_eq!(doubled.to_vec().unwrap(), expect);
}

// ── ufunc arithmetic: the wrapping contract, per op class ────────────────

#[test]
fn u8_add_wraps_mod_256() {
    let g = gpu();
    let a = Array::from_slice(&g, &[200u8, 255, 0, 100], &[4]).unwrap();
    let b = Array::from_slice(&g, &[100u8, 1, 0, 55], &[4]).unwrap();
    // 300 → 44, 256 → 0: the wrap is the contract, not an accident.
    assert_eq!(a.add(&b).unwrap().to_vec().unwrap(), vec![44u8, 0, 0, 155]);
}

#[test]
fn u8_sub_and_mul_wrap() {
    let g = gpu();
    let a = Array::from_slice(&g, &[10u8, 20, 200], &[3]).unwrap();
    let b = Array::from_slice(&g, &[20u8, 20, 2], &[3]).unwrap();
    // 10−20 → 246; 200·2 = 400 → 144.
    assert_eq!(a.sub(&b).unwrap().to_vec().unwrap(), vec![246u8, 0, 198]);
    assert_eq!(a.mul(&b).unwrap().to_vec().unwrap(), vec![200u8, 144, 144]);
}

#[test]
fn i8_add_wraps_at_the_sign_boundary() {
    let g = gpu();
    let a = Array::from_slice(&g, &[100i8, 127, -128, -100], &[4]).unwrap();
    let b = Array::from_slice(&g, &[100i8, 1, -1, -100], &[4]).unwrap();
    assert_eq!(
        a.add(&b).unwrap().to_vec().unwrap(),
        vec![-56i8, -128, 127, 56]
    );
}

#[test]
fn i16_u16_add_wrap() {
    let g = gpu();
    let a = Array::from_slice(&g, &[i16::MAX, -1], &[2]).unwrap();
    let b = Array::from_slice(&g, &[1i16, -i16::MAX], &[2]).unwrap();
    assert_eq!(
        a.add(&b).unwrap().to_vec().unwrap(),
        vec![i16::MIN, -32768i16]
    );
    let c = Array::from_slice(&g, &[0xFFFFu16, 0x8000], &[2]).unwrap();
    let d = Array::from_slice(&g, &[2u16, 0x8000], &[2]).unwrap();
    assert_eq!(c.add(&d).unwrap().to_vec().unwrap(), vec![1u16, 0]);
}

#[test]
fn narrow_div_truncates_and_div_by_zero_is_zero() {
    let g = gpu();
    let a = Array::from_slice(&g, &[7u8, 255, 9, 200], &[4]).unwrap();
    let b = Array::from_slice(&g, &[2u8, 10, 0, 0], &[4]).unwrap();
    // x/0 = 0 — the declared integer contract (the CPU reference pins it).
    // KNOWN RED on Metal today, and NOT narrow-specific: the MSL emitter
    // lowers integer Div to bare `/`, so x/0 returns hardware garbage for
    // EVERY int width (u32/u16 → all-ones, i32/i16/i8 → ±1). The op-matrix
    // never sees it because its div cases skip b == 0. The fix is a
    // div-guard in the emitters, not a weaker test.
    assert_eq!(a.div(&b).unwrap().to_vec().unwrap(), vec![3u8, 25, 0, 0]);

    let c = Array::from_slice(&g, &[-7i8, i8::MIN, 5], &[3]).unwrap();
    let d = Array::from_slice(&g, &[2i8, -1, 0], &[3]).unwrap();
    // −7/2 truncates toward zero → −3; MIN/−1 wraps to MIN (no trap).
    assert_eq!(c.div(&d).unwrap().to_vec().unwrap(), vec![-3i8, i8::MIN, 0]);
}

#[test]
fn neg_wraps_on_unsigned_and_at_int_min() {
    let g = gpu();
    let a = Array::from_slice(&g, &[0u8, 1, 128, 255], &[4]).unwrap();
    // −x ≡ 2^8 − x (numpy uint behaviour).
    assert_eq!(a.neg().unwrap().to_vec().unwrap(), vec![0u8, 255, 128, 1]);
    let b = Array::from_slice(&g, &[1i8, -1, i8::MIN], &[3]).unwrap();
    assert_eq!(b.neg().unwrap().to_vec().unwrap(), vec![-1i8, 1, i8::MIN]);
    let c = Array::from_slice(&g, &[1u16, 0x8000], &[2]).unwrap();
    assert_eq!(c.neg().unwrap().to_vec().unwrap(), vec![0xFFFFu16, 0x8000]);
}

// ── compares: signedness is decided by the dtype ─────────────────────────

#[test]
fn i8_compares_are_signed() {
    let g = gpu();
    // 0x80 = −128 as i8: signed compare must put it BELOW 1.
    let a = Array::from_slice(&g, &[-128i8, -1, 0, 1], &[4]).unwrap();
    let b = Array::from_slice(&g, &[1i8, 1, 1, 1], &[4]).unwrap();
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![1i8, 1, 1, 0]);
    assert_eq!(a.ge(&b).unwrap().to_vec().unwrap(), vec![0i8, 0, 0, 1]);
}

#[test]
fn u8_compares_are_unsigned() {
    let g = gpu();
    // The same bit pattern 0x80 = 128 as u8: unsigned compare puts it ABOVE 1.
    let a = Array::from_slice(&g, &[128u8, 255, 0, 1], &[4]).unwrap();
    let b = Array::from_slice(&g, &[1u8, 1, 1, 1], &[4]).unwrap();
    assert_eq!(a.gt(&b).unwrap().to_vec().unwrap(), vec![1u8, 1, 0, 0]);
    assert_eq!(a.le(&b).unwrap().to_vec().unwrap(), vec![0u8, 0, 1, 1]);
}

#[test]
fn i16_u16_compares_at_the_sign_boundary() {
    let g = gpu();
    let a = Array::from_slice(&g, &[i16::MIN, -1], &[2]).unwrap();
    let b = Array::from_slice(&g, &[1i16, 0], &[2]).unwrap();
    assert_eq!(a.lt(&b).unwrap().to_vec().unwrap(), vec![1i16, 1]);
    let c = Array::from_slice(&g, &[0x8000u16, 0xFFFF], &[2]).unwrap();
    let d = Array::from_slice(&g, &[1u16, 0], &[2]).unwrap();
    assert_eq!(c.gt(&d).unwrap().to_vec().unwrap(), vec![1u16, 1]);
}

#[test]
fn eq_ne_and_where_mask_on_narrow() {
    let g = gpu();
    let a = Array::from_slice(&g, &[5u8, 7, 5, 9], &[4]).unwrap();
    let b = Array::from_slice(&g, &[5u8, 5, 5, 5], &[4]).unwrap();
    let m = a.eq(&b).unwrap();
    assert_eq!(m.to_vec().unwrap(), vec![1u8, 0, 1, 0]);
    let x = Array::from_slice(&g, &[10u8, 20, 30, 40], &[4]).unwrap();
    let y = Array::from_slice(&g, &[1u8, 2, 3, 4], &[4]).unwrap();
    // where(mask, x, y)
    assert_eq!(
        m.where_mask(&x, &y).unwrap().to_vec().unwrap(),
        vec![10u8, 2, 30, 4]
    );
}

// ── constructors ─────────────────────────────────────────────────────────

#[test]
fn zeros_ones_full_eye_narrow() {
    let g = gpu();
    assert_eq!(
        Array::<i16>::zeros(&g, &[2, 3]).unwrap().to_vec().unwrap(),
        vec![0i16; 6]
    );
    assert_eq!(
        Array::<u8>::ones(&g, &[5]).unwrap().to_vec().unwrap(),
        vec![1u8; 5]
    );
    assert_eq!(
        Array::full(&g, -7i8, &[4]).unwrap().to_vec().unwrap(),
        vec![-7i8; 4]
    );
    assert_eq!(
        Array::<u16>::eye(&g, 3).unwrap().to_vec().unwrap(),
        vec![1u16, 0, 0, 0, 1, 0, 0, 0, 1]
    );
}

#[test]
fn arange_narrow_saturates_at_the_type_range() {
    let g = gpu();
    // Step math runs in f64 and converts with Rust `as`: past 255 a u8
    // ramp clamps (same macro line as u32 at 2^32 — documented).
    let a = Array::<u8>::arange(&g, 250.0, 2.0, 6).unwrap();
    assert_eq!(a.to_vec().unwrap(), vec![250u8, 252, 254, 255, 255, 255]);
    let b = Array::<i8>::arange(&g, -3.0, 1.0, 6).unwrap();
    assert_eq!(b.to_vec().unwrap(), vec![-3i8, -2, -1, 0, 1, 2]);
}

// ── astype: the ten-type matrix, spot-checked per §7 class ──────────────

#[test]
fn astype_sign_extends_signed_narrow_sources() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-1i8, i8::MIN, i8::MAX, 0], &[4]).unwrap();
    let w: Array<i32> = a.astype().unwrap();
    assert_eq!(w.to_vec().unwrap(), vec![-1i32, -128, 127, 0]);
    let b = Array::from_slice(&g, &[-1i16, i16::MIN, 256], &[3]).unwrap();
    let w16: Array<i32> = b.astype().unwrap();
    assert_eq!(w16.to_vec().unwrap(), vec![-1i32, -32768, 256]);
}

#[test]
fn astype_zero_extends_unsigned_narrow_sources() {
    let g = gpu();
    let a = Array::from_slice(&g, &[255u8, 128, 0], &[3]).unwrap();
    let w: Array<i32> = a.astype().unwrap();
    assert_eq!(w.to_vec().unwrap(), vec![255i32, 128, 0]);
    let b = Array::from_slice(&g, &[0xFFFFu16, 0x8000], &[2]).unwrap();
    let w16: Array<u32> = b.astype().unwrap();
    assert_eq!(w16.to_vec().unwrap(), vec![65535u32, 32768]);
}

#[test]
fn astype_truncates_wide_to_narrow_mod_2w() {
    let g = gpu();
    let a = Array::from_slice(&g, &[300u32, 256, 255, 44], &[4]).unwrap();
    let n: Array<u8> = a.astype().unwrap();
    assert_eq!(n.to_vec().unwrap(), vec![44u8, 0, 255, 44]);
    let b = Array::from_slice(&g, &[-1i32, 70000], &[2]).unwrap();
    let n16: Array<u16> = b.astype().unwrap();
    // −1 → 0xFFFF; 70000 mod 2^16 = 4464.
    assert_eq!(n16.to_vec().unwrap(), vec![0xFFFFu16, 4464]);
    // narrow → narrower narrow
    let c = Array::from_slice(&g, &[300u16, 0x1FF], &[2]).unwrap();
    let n8: Array<u8> = c.astype().unwrap();
    assert_eq!(n8.to_vec().unwrap(), vec![44u8, 0xFF]);
}

#[test]
fn astype_same_width_reinterprets_the_bit_pattern() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-1i8, i8::MIN, 1], &[3]).unwrap();
    let u: Array<u8> = a.astype().unwrap();
    assert_eq!(u.to_vec().unwrap(), vec![255u8, 128, 1]);
    let back: Array<i8> = u.astype().unwrap();
    assert_eq!(back.to_vec().unwrap(), vec![-1i8, i8::MIN, 1]);
    let b = Array::from_slice(&g, &[0xFFFFu16, 0x8000], &[2]).unwrap();
    let s: Array<i16> = b.astype().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![-1i16, i16::MIN]);
}

#[test]
fn astype_narrow_to_float_is_exact() {
    let g = gpu();
    let a = Array::from_slice(&g, &[255u8, 0, 128], &[3]).unwrap();
    let f: Array<f32> = a.astype().unwrap();
    assert_eq!(f.to_vec().unwrap(), vec![255.0f32, 0.0, 128.0]);
    let b = Array::from_slice(&g, &[i16::MIN, -1, i16::MAX], &[3]).unwrap();
    let fb: Array<f32> = b.astype().unwrap();
    assert_eq!(fb.to_vec().unwrap(), vec![-32768.0f32, -1.0, 32767.0]);
}

#[test]
fn astype_float_to_narrow_truncates_toward_zero_in_range() {
    let g = gpu();
    let a = Array::from_slice(&g, &[3.9f32, 0.1, 254.7], &[3]).unwrap();
    let u: Array<u8> = a.astype().unwrap();
    assert_eq!(u.to_vec().unwrap(), vec![3u8, 0, 254]);
    let b = Array::from_slice(&g, &[-3.9f32, -0.1, 127.9, -128.0], &[4]).unwrap();
    let s: Array<i8> = b.astype().unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![-3i8, 0, 127, -128]);
}

// ── views: layout algebra over narrow storage ────────────────────────────

#[test]
fn transpose_then_astype_gathers_strided_narrow() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1u8, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let t = a.transpose(0, 1).unwrap(); // [3, 2] view, strided
    assert_eq!(t.to_vec().unwrap(), vec![1u8, 4, 2, 5, 3, 6]);
    // astype on the strided view: device gather + cast.
    let w: Array<u32> = t.astype().unwrap();
    assert_eq!(w.to_vec().unwrap(), vec![1u32, 4, 2, 5, 3, 6]);
}

#[test]
fn broadcast_narrow_add() {
    let g = gpu();
    let a = Array::from_slice(&g, &[10u8, 250], &[2, 1]).unwrap();
    let b = Array::from_slice(&g, &[1u8, 2, 10], &[1, 3]).unwrap();
    // Row 1 wraps: 250+10 = 260 → 4.
    assert_eq!(
        a.add(&b).unwrap().to_vec().unwrap(),
        vec![11u8, 12, 20, 251, 252, 4]
    );
}

#[test]
fn rank0_and_rank3_shapes() {
    let g = gpu();
    // 0-d: shape [], one element.
    let s = Array::from_slice(&g, &[200u8], &[]).unwrap();
    assert_eq!(s.rank(), 0);
    assert_eq!(s.len(), 1);
    let t = s.add(&s).unwrap(); // 400 → 144
    assert_eq!(t.to_vec().unwrap(), vec![144u8]);
    // rank-3: reshape + sum over the middle axis.
    let data: Vec<i16> = (0..24).map(|i| i as i16 - 12).collect();
    let a = Array::from_slice(&g, &data, &[2, 3, 4]).unwrap();
    let sum1 = a.sum_axis(1).unwrap();
    assert_eq!(sum1.shape(), &[2, 1, 4]);
    let expect: Vec<i16> = (0..2)
        .flat_map(|i| (0..4).map(move |k| (0..3).map(|j| (i * 12 + j * 4 + k) as i16 - 12).sum()))
        .collect();
    assert_eq!(sum1.to_vec().unwrap(), expect);
}

// ── reductions: in-T wrapping sums (the contract, on purpose) ────────────

#[test]
fn sum_axis_u8_wraps_mod_256_bit_exact() {
    let g = gpu();
    // 300 ones per row: the true sum is 300, the in-dtype answer is 44.
    // Asserted bit-exact — the same value must come back from the software,
    // Metal and Vulkan lanes (the mod-2^w homomorphism argument).
    let a = Array::<u8>::ones(&g, &[2, 300]).unwrap();
    let s = a.sum_axis(1).unwrap();
    assert_eq!(s.shape(), &[2, 1]);
    assert_eq!(s.to_vec().unwrap(), vec![44u8, 44]);

    // Mixed values, host-checked with wrapping_add.
    let data: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
    let b = Array::from_slice(&g, &data, &[1, 300]).unwrap();
    let expect = data.iter().fold(0u8, |acc, x| acc.wrapping_add(*x));
    assert_eq!(b.sum_axis(1).unwrap().to_vec().unwrap(), vec![expect]);
}

#[test]
fn sum_axis_i8_exact_when_in_range() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-10i8, 20, -30, 40, 5, -5], &[3, 2]).unwrap();
    let s = a.sum_axis(1).unwrap();
    assert_eq!(s.to_vec().unwrap(), vec![10i8, 10, 0]);
}

#[test]
fn sum_axis_widened_spelling_is_exact() {
    let g = gpu();
    // The documented escape from the wrap: astype first, then sum.
    let a = Array::<u8>::ones(&g, &[1, 300]).unwrap();
    let wide: Array<u32> = a.astype().unwrap();
    assert_eq!(wide.sum_axis(1).unwrap().to_vec().unwrap(), vec![300u32]);
}

#[test]
fn cumsum_last_wraps_in_t() {
    let g = gpu();
    let a = Array::from_slice(&g, &[100u8, 100, 100, 100], &[4]).unwrap();
    // partials 100, 200, 300→44, 400→144.
    assert_eq!(
        a.cumsum_last().unwrap().to_vec().unwrap(),
        vec![100u8, 200, 44, 144]
    );
}

// ── rowreduce quartet: exact in-dtype extrema ────────────────────────────

#[test]
fn min_max_arg_last_i8_with_negatives() {
    let g = gpu();
    let a = Array::from_slice(&g, &[-5i8, 3, -128, 7, 0, 127, -1, 2], &[2, 4]).unwrap();
    assert_eq!(a.max_axis_last().unwrap().to_vec().unwrap(), vec![7i8, 127]);
    assert_eq!(
        a.min_axis_last().unwrap().to_vec().unwrap(),
        vec![-128i8, -1]
    );
    assert_eq!(a.argmax_last().unwrap().to_vec().unwrap(), vec![3u32, 1]);
    assert_eq!(a.argmin_last().unwrap().to_vec().unwrap(), vec![2u32, 2]);
}

#[test]
fn max_last_u16_above_the_sign_bit() {
    let g = gpu();
    // Values with the top bit set must beat small ones under UNSIGNED order.
    let a = Array::from_slice(&g, &[1u16, 0x8000, 42, 0xFFFF], &[1, 4]).unwrap();
    assert_eq!(
        a.max_axis_last().unwrap().to_vec().unwrap(),
        vec![0xFFFFu16]
    );
    assert_eq!(a.argmax_last().unwrap().to_vec().unwrap(), vec![3u32]);
}

#[test]
fn whole_array_min_max_spelling_via_reshape() {
    let g = gpu();
    // No ReduceScalar for narrow (by design): the whole-array in-dtype
    // extrema spelling is reshape([1, n]) + the rowreduce.
    let data: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
    let a = Array::from_slice(&g, &data, &[300]).unwrap();
    let m = a
        .reshape(&[1, 300])
        .unwrap()
        .max_axis_last()
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(m, vec![*data.iter().max().unwrap()]);
}

// ── data movement: concat / gather over narrow elements ──────────────────

#[test]
fn concat_axis0_narrow() {
    let g = gpu();
    let a = Array::from_slice(&g, &[1i16, 2, 3, 4], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[-1i16, -2], &[1, 2]).unwrap();
    let c = Array::concat_axis0(&[&a, &b]).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    assert_eq!(c.to_vec().unwrap(), vec![1i16, 2, 3, 4, -1, -2]);
}

#[test]
fn gather_and_select_rows_narrow() {
    let g = gpu();
    let table = Array::from_slice(&g, &[10u8, 11, 20, 21, 30, 31], &[3, 2]).unwrap();
    // gather_rows: per-row column pick, out[i] = table[i, idx[i]] → [N].
    let cols = Array::from_slice(&g, &[1u32, 0, 1], &[3]).unwrap();
    let picked = table.gather_rows(&cols).unwrap();
    assert_eq!(picked.shape(), &[3]);
    assert_eq!(picked.to_vec().unwrap(), vec![11u8, 20, 31]);
    // select_rows: whole-row lookup by id → [K, C].
    let ids = Array::from_slice(&g, &[2u32, 0, 2], &[3]).unwrap();
    let rows = table.select_rows(&ids).unwrap();
    assert_eq!(rows.shape(), &[3, 2]);
    assert_eq!(rows.to_vec().unwrap(), vec![30u8, 31, 10, 11, 30, 31]);
}

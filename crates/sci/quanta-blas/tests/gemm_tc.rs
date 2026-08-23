//! Tensor-core (cooperative-matrix) GEMM tests.
//!
//! `gemm_tc` computes `C ← A·B + C` on the shape the device enumerates for
//! the requested element types; `gemm_f32_tc` is its all-f32 form. On
//! backends without cooperative-matrix support (the software lane) it returns
//! `NotSupported`; on Metal (Apple family 7+) the result is checked against the
//! pure-Rust `reference::gemm` oracle (α=1, β=1).

#![cfg(feature = "gpu")]

use quanta::ScalarType;
use quanta_blas::reference;

fn mat(rows: usize, cols: usize, seed: u32) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 13) as f32 - 6.0)
        .collect()
}

#[test]
fn tc_unsupported_on_software() {
    // The CPU lane reports supports_cooperative_matrix() == false.
    let g = quanta::init_cpu();
    let a = g.field::<f32>(64).unwrap();
    let b = g.field::<f32>(64).unwrap();
    let c = g.field::<f32>(64).unwrap();
    assert!(quanta_blas::gemm_f32_tc(&g, 8, 8, 8, &a, &b, &c).is_err());
}

#[cfg(feature = "gpu-metal")]
fn check_metal(m: usize, n: usize, k: usize) {
    let g = quanta::init().expect("metal device");
    if !g.supports_cooperative_matrix() {
        eprintln!("skip: device lacks cooperative-matrix support");
        return;
    }
    let a = mat(m, k, 1);
    let b = mat(k, n, 2);
    let c0 = mat(m, n, 3);

    let af = g.field::<f32>(m * k).unwrap();
    let bf = g.field::<f32>(k * n).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    af.write(&a).unwrap();
    bf.write(&b).unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::gemm_f32_tc(&g, m as u32, n as u32, k as u32, &af, &bf, &cf).unwrap();
    let got = cf.read().unwrap();

    // Oracle: C ← A·B + C  (α = 1, β = 1).
    let mut want = c0.clone();
    reference::gemm(m, n, k, 1.0, &a, &b, 1.0, &mut want);

    for (idx, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "tc gemm {m}x{n}x{k}: entry {idx}: {gv} vs {wv}"
        );
    }
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_single_tile() {
    // One 32×32 output tile (one subgroup, 4×4 fragments), k = one fragment.
    check_metal(32, 32, 8);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_multi_tile_square() {
    check_metal(64, 64, 64);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_k_sweep() {
    // one 32×32 tile; k spans 5 fragments (40 = 8·5).
    check_metal(32, 32, 40);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_rectangular() {
    check_metal(96, 64, 16);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_larger() {
    check_metal(128, 128, 128);
}

#[cfg(feature = "gpu-metal")]
#[test]
fn tc_shared_load_probe() {
    // Validates simdgroup_load from a threadgroup-shared array: single 8×8 tile
    // staged to shared, loaded as fragments, MMA'd, stored. C = A·B.
    let g = quanta::init().expect("metal device");
    if !g.supports_cooperative_matrix() {
        return;
    }
    let a = mat(8, 8, 1);
    let b = mat(8, 8, 2);
    let af = g.field::<f32>(64).unwrap();
    let bf = g.field::<f32>(64).unwrap();
    let cf = g.field::<f32>(64).unwrap();
    af.write(&a).unwrap();
    bf.write(&b).unwrap();
    cf.write(&vec![0.0f32; 64]).unwrap();
    quanta_blas::mixed_tc::gemm_f32_tc_shared_probe(&g, &af, &bf, &cf).unwrap();
    let got = cf.read().unwrap();
    let mut want = vec![0.0f32; 64];
    reference::gemm(8, 8, 8, 1.0, &a, &b, 0.0, &mut want);
    for (i, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (gv - wv).abs() <= 1e-3 * (1.0 + wv.abs()),
            "shared-probe entry {i}: {gv} vs {wv}"
        );
    }
}

// ── The device's own shape ──────────────────────────────────────────────
//
// `gemm_tc` builds for whatever shape the device enumerates with the
// requested element types, so these run on the shape that is there and
// skip loudly (naming the device) on the ones that are not: this Mac's
// Metal lists 8×8×8 f32 and f16 and no mixed form, the discrete
// `VK_KHR_cooperative_matrix` cards list f16→f32 at 16×16×16 and no f32.

/// f32 → IEEE binary16 bits, round-to-nearest-even (the storage contract
/// for an f16 field). Copied from `tests/gpu_coopmat.rs`, where the same
/// helper feeds the raw-IR tile kernel.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        let rounded = (m + 0x0fff + ((m >> 13) & 1)) >> 13;
        return sign | rounded as u16;
    }
    let mut h = ((e as u32) << 10) | (mant >> 13);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// IEEE binary16 bits → f32, for reading an f16 field back.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h as u32) & 0x3ff;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: renormalise into the f32 exponent range.
        let shift = mant.leading_zeros() - 21;
        let e = 127 - 15 - shift;
        let m = (mant << (shift + 1)) & 0x3ff;
        return f32::from_bits(sign | (e << 23) | (m << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

/// Small integers in `-3..=3`, the values every tensor-core oracle below
/// runs on: exact in f16, and with `k` small every partial sum stays an
/// integer below 2048 — still exact — so the comparison is bitwise.
fn small_ints(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + seed) % 7) as f32 - 3.0)
        .collect()
}

/// Host oracle for `C ← A·B + C`, computed in f32 over exact integers.
fn tc_oracle(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c0: &[f32]) -> Vec<f32> {
    let mut want = c0.to_vec();
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += a[r * k + i] * b[i * n + col];
            }
            want[r * n + col] += acc;
        }
    }
    want
}

#[test]
fn gemm_tc_f16_acc_f16_matches_host_oracle() {
    let Ok(g) = quanta::init() else {
        eprintln!("SKIP: no device");
        return;
    };
    let Some(shape) = quanta_blas::tc_shape_for(&g, ScalarType::F16, ScalarType::F16) else {
        eprintln!(
            "SKIP: `{}` enumerates no uniform-f16 cooperative-matrix shape",
            g.name()
        );
        return;
    };
    // One subgroup owns a shape.m·4 × shape.n·4 tile; 64×64 is several of
    // them on the 8×8×8 shape Metal lists.
    let (m, n, k) = (64usize, 64usize, 16usize);
    assert!(
        m.is_multiple_of(shape.m as usize * 4)
            && n.is_multiple_of(shape.n as usize * 4)
            && k.is_multiple_of(shape.k as usize),
        "`{}` shape {}x{}x{} does not tile {m}x{n}x{k} — pick sizes for it",
        g.name(),
        shape.m,
        shape.n,
        shape.k
    );
    eprintln!(
        "[gemm_tc] `{}`: {}x{}x{} F16 -> F16, {m}x{n}x{k}",
        g.name(),
        shape.m,
        shape.n,
        shape.k
    );
    let a = small_ints(m * k, 3);
    let b = small_ints(k * n, 1);
    let c0 = small_ints(m * n, 2);
    let want = tc_oracle(m, n, k, &a, &b, &c0);

    let af = g.field::<u16>(m * k).unwrap();
    let bf = g.field::<u16>(k * n).unwrap();
    let cf = g.field::<u16>(m * n).unwrap();
    af.write(&a.iter().map(|&x| f16_bits(x)).collect::<Vec<_>>())
        .unwrap();
    bf.write(&b.iter().map(|&x| f16_bits(x)).collect::<Vec<_>>())
        .unwrap();
    cf.write(&c0.iter().map(|&x| f16_bits(x)).collect::<Vec<_>>())
        .unwrap();
    quanta_blas::gemm_tc::<u16, u16>(&g, m as u32, n as u32, k as u32, &af, &bf, &cf).unwrap();

    let got = cf.read().unwrap();
    for (i, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            gv,
            f16_bits(wv),
            "C[{i}] (row {}, col {}): got {}, expected {wv}",
            i / n,
            i % n,
            f16_to_f32(gv)
        );
    }
}

#[test]
fn gemm_tc_f16_acc_f32_matches_host_oracle() {
    let Ok(g) = quanta::init() else {
        eprintln!("SKIP: no device");
        return;
    };
    let Some(shape) = quanta_blas::tc_shape_for(&g, ScalarType::F16, ScalarType::F32) else {
        eprintln!(
            "SKIP: `{}` enumerates no f16-input / f32-accumulate cooperative-matrix shape",
            g.name()
        );
        return;
    };
    // The mixed shapes are 16-wide and some are non-square (16×8×16), so
    // one subgroup's tile is up to 64×64; 128×128 is a multiple of every
    // form the cards list, and k stays short enough for exact sums.
    let (m, n) = (128usize, 128usize);
    let k = 4 * shape.k as usize;
    eprintln!(
        "[gemm_tc] `{}`: {}x{}x{} F16 -> F32, {m}x{n}x{k}",
        g.name(),
        shape.m,
        shape.n,
        shape.k
    );
    let a = small_ints(m * k, 3);
    let b = small_ints(k * n, 1);
    let c0 = small_ints(m * n, 2);
    let want = tc_oracle(m, n, k, &a, &b, &c0);

    let af = g.field::<u16>(m * k).unwrap();
    let bf = g.field::<u16>(k * n).unwrap();
    let cf = g.field::<f32>(m * n).unwrap();
    af.write(&a.iter().map(|&x| f16_bits(x)).collect::<Vec<_>>())
        .unwrap();
    bf.write(&b.iter().map(|&x| f16_bits(x)).collect::<Vec<_>>())
        .unwrap();
    cf.write(&c0).unwrap();
    quanta_blas::gemm_tc::<u16, f32>(&g, m as u32, n as u32, k as u32, &af, &bf, &cf).unwrap();

    let got = cf.read().unwrap();
    for (i, (&gv, &wv)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "C[{i}] (row {}, col {}): got {gv}, expected {wv}",
            i / n,
            i % n
        );
    }
}

#[test]
fn gemm_tc_refuses_off_tile_sizes() {
    let Ok(g) = quanta::init() else {
        eprintln!("SKIP: no device");
        return;
    };
    let Some(shape) = quanta_blas::tc_shape_for(&g, ScalarType::F32, ScalarType::F32) else {
        eprintln!(
            "SKIP: `{}` enumerates no uniform-f32 cooperative-matrix shape",
            g.name()
        );
        return;
    };
    // One row past the tile the device's shape needs: refused, never
    // rounded up and never silently routed to the tiled kernel.
    let (m, n, k) = (shape.m as u32 * 4 + 1, shape.n as u32 * 4, shape.k as u32);
    let af = g.field::<f32>((m * k) as usize).unwrap();
    let bf = g.field::<f32>((k * n) as usize).unwrap();
    let cf = g.field::<f32>((m * n) as usize).unwrap();
    let err = quanta_blas::gemm_tc::<f32, f32>(&g, m, n, k, &af, &bf, &cf)
        .expect_err("an off-tile m must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("multiple of"),
        "refusal must name the tile multiples, got: {msg}"
    );
}

#[test]
fn gemm_tc_refuses_missing_shape() {
    let Ok(g) = quanta::init() else {
        eprintln!("SKIP: no device");
        return;
    };
    // f32 inputs with f16 accumulation is no hardware's shape.
    let af = g.field::<f32>(64).unwrap();
    let bf = g.field::<f32>(64).unwrap();
    let cf = g.field::<u16>(64).unwrap();
    let err = quanta_blas::gemm_tc::<f32, u16>(&g, 8, 8, 8, &af, &bf, &cf)
        .expect_err("f32 inputs with f16 accumulation is no device's shape");
    let msg = format!("{err}");
    assert!(
        msg.contains("cooperative-matrix shape"),
        "refusal must name the missing shape, got: {msg}"
    );
}

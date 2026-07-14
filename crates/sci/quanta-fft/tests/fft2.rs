//! 2-D GPU FFT differential tests: `fft2` vs the direct 2-D DFT oracle,
//! the `ifft2(fft2(x)) == x` round trip, and the separability structure check.

#![cfg(feature = "gpu")]

use quanta_fft::reference;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (gpu-metal / gpu-vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "gpu-metal", feature = "gpu-vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "gpu-metal", feature = "gpu-vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// Deterministic complex grid of `h*w` points.
fn grid(h: usize, w: usize, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let n = h * w;
    let re: Vec<f32> = (0..n)
        .map(|i| (((i as u32).wrapping_mul(2654435761) ^ seed) % 17) as f32 - 8.0)
        .collect();
    let im: Vec<f32> = (0..n)
        .map(|i| (((i as u32).wrapping_mul(40503) ^ seed.wrapping_add(7)) % 13) as f32 - 6.0)
        .collect();
    (re, im)
}

fn close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: len {} vs {}", a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() <= 1e-2 * (1.0 + y.abs()),
            "{what}: [{i}] {x} vs {y}"
        );
    }
}

/// The dossier size sweep — square, rectangular both ways, and degenerate
/// single-row/column grids.
const SIZES: &[(usize, usize)] = &[
    (2, 2),
    (4, 4),
    (8, 4),
    (4, 8),
    (8, 8),
    (16, 16),
    (1, 8),
    (8, 1),
];

/// GPU fft2 must match the direct 2-D DFT (independent double-sum oracle,
/// NOT row-column composed) for every size in the sweep.
#[test]
fn fft2_matches_dft2() {
    let g = gpu();
    for &(h, w) in SIZES {
        let (re, im) = grid(h, w, (h * 31 + w) as u32);
        let (gr, gi) = quanta_fft::fft2(&g, &re, &im, h, w).unwrap();
        let (wr, wi) = reference::dft2(&re, &im, h, w);
        close(&gr, &wr, &format!("fft2 re {h}x{w}"));
        close(&gi, &wi, &format!("fft2 im {h}x{w}"));
    }
}

/// GPU ifft2 must match the direct inverse 2-D DFT (`+` exponent, 1/(H·W)).
#[test]
fn ifft2_matches_idft2() {
    let g = gpu();
    for &(h, w) in &[(4usize, 4usize), (8, 4), (16, 16)] {
        let (re, im) = grid(h, w, 99 + (h + w) as u32);
        let (gr, gi) = quanta_fft::ifft2(&g, &re, &im, h, w).unwrap();
        let (wr, wi) = reference::idft2(&re, &im, h, w);
        close(&gr, &wr, &format!("ifft2 re {h}x{w}"));
        close(&gi, &wi, &format!("ifft2 im {h}x{w}"));
    }
}

/// ifft2(fft2(x)) == x for every size in the sweep.
#[test]
fn fft2_round_trip() {
    let g = gpu();
    for &(h, w) in SIZES {
        let (re, im) = grid(h, w, 1234 + (h * w) as u32);
        let (fr, fi) = quanta_fft::fft2(&g, &re, &im, h, w).unwrap();
        let (rr, ri) = quanta_fft::ifft2(&g, &fr, &fi, h, w).unwrap();
        close(&rr, &re, &format!("round-trip re {h}x{w}"));
        close(&ri, &im, &format!("round-trip im {h}x{w}"));
    }
}

/// Separability: for a separable real input `x[y][x] = f(y)·g(x)`, the 2-D
/// spectrum is the outer (complex) product of the 1-D spectra:
/// `X[ky][kx] = F[ky]·G[kx]`. A strong structural check — it exercises the
/// row-column decomposition against the tensor-product identity.
#[test]
fn fft2_separable_input_is_outer_product() {
    let g = gpu();
    let (h, w) = (8usize, 16usize);
    let f: Vec<f32> = (0..h).map(|y| ((y * 5 + 2) % 7) as f32 - 3.0).collect();
    let gx: Vec<f32> = (0..w).map(|x| ((x * 3 + 1) % 9) as f32 - 4.0).collect();

    // Grid = outer product f(y)·g(x), purely real.
    let mut re = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            re[y * w + x] = f[y] * gx[x];
        }
    }
    let im = vec![0.0f32; h * w];

    let (sr, si) = quanta_fft::fft2(&g, &re, &im, h, w).unwrap();

    // 1-D oracle spectra of the two factors.
    let (fr, fi) = reference::dft(&f, &vec![0.0f32; h]);
    let (gr, gi) = reference::dft(&gx, &vec![0.0f32; w]);

    // Expected: complex outer product F[ky]·G[kx].
    let mut exp_re = vec![0.0f32; h * w];
    let mut exp_im = vec![0.0f32; h * w];
    for ky in 0..h {
        for kx in 0..w {
            exp_re[ky * w + kx] = fr[ky] * gr[kx] - fi[ky] * gi[kx];
            exp_im[ky * w + kx] = fr[ky] * gi[kx] + fi[ky] * gr[kx];
        }
    }
    close(&sr, &exp_re, "separability re");
    close(&si, &exp_im, "separability im");
}

/// Non-power-of-2 in either dimension is refused (NotSupported), even when
/// the buffer length is consistent.
#[test]
fn fft2_non_power_of_two_errors() {
    let g = gpu();
    for &(h, w) in &[(3usize, 4usize), (4, 6), (0, 4), (4, 0)] {
        let re = vec![0.0f32; h * w];
        let im = vec![0.0f32; h * w];
        assert!(
            quanta_fft::fft2(&g, &re, &im, h, w).is_err(),
            "{h}x{w} should be refused"
        );
        assert!(
            quanta_fft::ifft2(&g, &re, &im, h, w).is_err(),
            "inverse {h}x{w} should be refused"
        );
    }
}

/// Length checks: re/im mismatch and buffer length != height·width.
#[test]
fn fft2_length_errors() {
    let g = gpu();
    assert!(quanta_fft::fft2(&g, &[0.0; 8], &[0.0; 4], 2, 4).is_err());
    assert!(quanta_fft::fft2(&g, &[0.0; 8], &[0.0; 8], 4, 4).is_err());
}

/// fft2 on a 1×W grid degenerates to the 1-D fft (and H×1 to the 1-D fft of
/// the column) — the 2-D API must agree with the 1-D one there.
#[test]
fn fft2_degenerate_matches_fft() {
    let g = gpu();
    let n = 16usize;
    let (re, im) = grid(1, n, 5);
    let (r2, i2) = quanta_fft::fft2(&g, &re, &im, 1, n).unwrap();
    let (r1, i1) = quanta_fft::fft(&g, &re, &im).unwrap();
    close(&r2, &r1, "1xW re");
    close(&i2, &i1, "1xW im");

    let (re, im) = grid(n, 1, 6);
    let (r2, i2) = quanta_fft::fft2(&g, &re, &im, n, 1).unwrap();
    let (r1, i1) = quanta_fft::fft(&g, &re, &im).unwrap();
    close(&r2, &r1, "Hx1 re");
    close(&i2, &i1, "Hx1 im");
}

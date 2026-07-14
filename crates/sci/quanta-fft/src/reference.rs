//! Pure-Rust direct DFT — the differential-test oracle.
//!
//! The naive O(N²) discrete Fourier transform, the ground truth every GPU FFT
//! result is checked against:
//!
//!   X[k] = Σⱼ x[j]·exp(∓2πi·jk/N)      (− forward, + inverse; inverse ÷ N)
//!
//! Complex data is **split** into a real part and an imaginary part (two
//! `f32` slices of equal length); the oracle returns the transformed
//! `(re, im)`. Accumulation is in `f64` so the reference is tighter than any
//! `f32` summation order — the GPU FFT is compared to it within a relative
//! tolerance.

use core::f64::consts::PI;

/// Direct DFT (forward). `re`/`im` are the input's real/imag parts (length N);
/// returns the transformed `(re, im)`. No power-of-2 restriction — this is the
/// reference, so it handles any N.
pub fn dft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
    dft_signed(re, im, -1.0, 1.0)
}

/// Direct inverse DFT. `+` in the exponent and a `1/N` scale, so
/// `idft(dft(x)) == x` (to rounding).
pub fn idft(re: &[f32], im: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let scale = if n == 0 { 1.0 } else { 1.0 / n as f64 };
    dft_signed(re, im, 1.0, scale)
}

/// Shared core: `X[k] = scale · Σⱼ x[j]·exp(sign·2πi·jk/N)`.
fn dft_signed(re: &[f32], im: &[f32], sign: f64, scale: f64) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(re.len(), im.len(), "dft: re/im length mismatch");
    let n = re.len();
    let mut out_re = vec![0.0f32; n];
    let mut out_im = vec![0.0f32; n];
    if n == 0 {
        return (out_re, out_im);
    }
    for k in 0..n {
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        for j in 0..n {
            let theta = sign * 2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let (s, c) = theta.sin_cos();
            let xr = re[j] as f64;
            let xi = im[j] as f64;
            // (xr + i·xi)·(c + i·s) = (xr·c − xi·s) + i·(xr·s + xi·c)
            acc_re += xr * c - xi * s;
            acc_im += xr * s + xi * c;
        }
        out_re[k] = (acc_re * scale) as f32;
        out_im[k] = (acc_im * scale) as f32;
    }
    (out_re, out_im)
}

/// Direct real-input DFT: real signal of length N → the first `N/2 + 1`
/// complex bins `(re, im)`. The remaining bins of a real signal's spectrum
/// are conjugates of these (`X[N−k] = conj(X[k])`), so nothing is lost.
/// f64 accumulation, any N — the `rfft` ground truth.
pub fn rdft(x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = x.len();
    if n == 0 {
        return (vec![], vec![]);
    }
    let bins = n / 2 + 1;
    let mut out_re = Vec::with_capacity(bins);
    let mut out_im = Vec::with_capacity(bins);
    for k in 0..bins {
        let mut acc_re = 0.0f64;
        let mut acc_im = 0.0f64;
        for (j, &xj) in x.iter().enumerate() {
            let theta = -2.0 * PI * (j as f64) * (k as f64) / (n as f64);
            let (s, c) = theta.sin_cos();
            acc_re += xj as f64 * c;
            acc_im += xj as f64 * s;
        }
        out_re.push(acc_re as f32);
        out_im.push(acc_im as f32);
    }
    (out_re, out_im)
}

/// Direct 2-D DFT (forward) of a row-major `height×width` grid:
///
///   X[ky][kx] = Σ_y Σ_x x[y][x]·exp(−2πi·(y·ky/H + x·kx/W))
///
/// Deliberately the naive O((HW)²) double sum — NOT the row-column
/// decomposition — so it is an independent oracle for the separable GPU
/// `fft2`. `f64` accumulation, any (non-zero-consistent) sizes.
pub fn dft2(re: &[f32], im: &[f32], height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    dft2_signed(re, im, height, width, -1.0, 1.0)
}

/// Direct 2-D inverse DFT: `+` in the exponent, `1/(H·W)` scale, so
/// `idft2(dft2(x)) == x` (to rounding).
pub fn idft2(re: &[f32], im: &[f32], height: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
    let n = height * width;
    let scale = if n == 0 { 1.0 } else { 1.0 / n as f64 };
    dft2_signed(re, im, height, width, 1.0, scale)
}

/// Shared 2-D core: `X[ky][kx] = scale · Σ_y Σ_x x[y][x]·exp(sign·2πi·(y·ky/H + x·kx/W))`.
fn dft2_signed(
    re: &[f32],
    im: &[f32],
    height: usize,
    width: usize,
    sign: f64,
    scale: f64,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(re.len(), im.len(), "dft2: re/im length mismatch");
    assert_eq!(re.len(), height * width, "dft2: length != height * width");
    let mut out_re = vec![0.0f32; re.len()];
    let mut out_im = vec![0.0f32; re.len()];
    if re.is_empty() {
        return (out_re, out_im);
    }
    for ky in 0..height {
        for kx in 0..width {
            let mut acc_re = 0.0f64;
            let mut acc_im = 0.0f64;
            for y in 0..height {
                for x in 0..width {
                    let theta = sign
                        * 2.0
                        * PI
                        * ((y * ky) as f64 / height as f64 + (x * kx) as f64 / width as f64);
                    let (s, c) = theta.sin_cos();
                    let xr = re[y * width + x] as f64;
                    let xi = im[y * width + x] as f64;
                    acc_re += xr * c - xi * s;
                    acc_im += xr * s + xi * c;
                }
            }
            out_re[ky * width + kx] = (acc_re * scale) as f32;
            out_im[ky * width + kx] = (acc_im * scale) as f32;
        }
    }
    (out_re, out_im)
}

/// Direct inverse of [`rdft`]: half-spectrum (`n/2 + 1` bins) → the real
/// signal of length `n`. Reconstructs the full spectrum by conjugate
/// symmetry and runs the direct inverse DFT, keeping the real part —
/// `irdft(rdft(x), N) ≈ x`. The `irfft` ground truth.
pub fn irdft(re: &[f32], im: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(re.len(), im.len(), "irdft: re/im length mismatch");
    if n == 0 {
        return vec![];
    }
    assert_eq!(
        re.len(),
        n / 2 + 1,
        "irdft: half-spectrum must be n/2+1 bins"
    );
    let mut full_re = vec![0.0f32; n];
    let mut full_im = vec![0.0f32; n];
    full_re[..re.len()].copy_from_slice(re);
    full_im[..im.len()].copy_from_slice(im);
    for k in 1..n.div_ceil(2) {
        full_re[n - k] = re[k];
        full_im[n - k] = -im[k];
    }
    let (x, _) = idft(&full_re, &full_im);
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_dc() {
        // All-ones input → DC bin = N, rest = 0.
        let n = 8;
        let re = vec![1.0f32; n];
        let im = vec![0.0f32; n];
        let (r, i) = dft(&re, &im);
        assert!((r[0] - n as f32).abs() < 1e-4);
        for k in 1..n {
            assert!(r[k].abs() < 1e-3 && i[k].abs() < 1e-3, "bin {k} not ~0");
        }
    }

    #[test]
    fn dft_single_freq() {
        // x[j] = cos(2π·j/N) → energy at bins 1 and N−1 (= N/2 each).
        let n = 8;
        let re: Vec<f32> = (0..n)
            .map(|j| (2.0 * PI * j as f64 / n as f64).cos() as f32)
            .collect();
        let im = vec![0.0f32; n];
        let (r, _) = dft(&re, &im);
        assert!((r[1] - (n as f32 / 2.0)).abs() < 1e-3);
        assert!((r[n - 1] - (n as f32 / 2.0)).abs() < 1e-3);
    }

    #[test]
    fn rdft_matches_dft_half() {
        // rdft(x) must equal the first N/2+1 bins of dft(x, 0) — same math,
        // same f64 accumulation, so bit-for-bit.
        for n in [1usize, 2, 5, 8, 16] {
            let x: Vec<f32> = (0..n).map(|j| (j as f32 * 0.7).sin() * 3.0).collect();
            let zeros = vec![0.0f32; n];
            let (fr, fi) = dft(&x, &zeros);
            let (rr, ri) = rdft(&x);
            assert_eq!(rr.len(), n / 2 + 1);
            for k in 0..rr.len() {
                assert_eq!(rr[k], fr[k], "re[{k}] n={n}");
                assert_eq!(ri[k], fi[k], "im[{k}] n={n}");
            }
        }
    }

    #[test]
    fn irdft_round_trip() {
        for n in [1usize, 2, 4, 8, 12] {
            let x: Vec<f32> = (0..n).map(|j| (j as f32 * 1.3).cos() * 2.0 - 0.5).collect();
            let (hr, hi) = rdft(&x);
            let back = irdft(&hr, &hi, n);
            for j in 0..n {
                assert!((back[j] - x[j]).abs() < 1e-3, "x[{j}] n={n}");
            }
        }
    }

    #[test]
    fn dft_round_trip() {
        let re = vec![1.0f32, -2.0, 3.0, 0.5, -1.0, 2.5, 0.0, 4.0];
        let im = vec![0.5f32, 1.0, -1.5, 2.0, 0.0, -0.5, 3.0, 1.0];
        let (fr, fi) = dft(&re, &im);
        let (rr, ri) = idft(&fr, &fi);
        for j in 0..re.len() {
            assert!((rr[j] - re[j]).abs() < 1e-3, "re[{j}]");
            assert!((ri[j] - im[j]).abs() < 1e-3, "im[{j}]");
        }
    }

    #[test]
    fn dft2_dc() {
        // All-ones H×W grid → DC bin = H·W, rest = 0.
        let (h, w) = (4, 8);
        let re = vec![1.0f32; h * w];
        let im = vec![0.0f32; h * w];
        let (r, i) = dft2(&re, &im, h, w);
        assert!((r[0] - (h * w) as f32).abs() < 1e-3);
        for k in 1..h * w {
            assert!(r[k].abs() < 1e-3 && i[k].abs() < 1e-3, "bin {k} not ~0");
        }
    }

    #[test]
    fn dft2_matches_row_column_composition() {
        // The direct double sum must agree with 1-D DFT rows then columns —
        // the separability identity, checked oracle-vs-oracle.
        let (h, w) = (4, 8);
        let re: Vec<f32> = (0..h * w)
            .map(|k| ((k * 7 + 3) % 11) as f32 - 5.0)
            .collect();
        let im: Vec<f32> = (0..h * w).map(|k| ((k * 5 + 1) % 9) as f32 - 4.0).collect();

        // Rows.
        let mut rr = vec![0.0f32; h * w];
        let mut ri = vec![0.0f32; h * w];
        for y in 0..h {
            let (or, oi) = dft(&re[y * w..(y + 1) * w], &im[y * w..(y + 1) * w]);
            rr[y * w..(y + 1) * w].copy_from_slice(&or);
            ri[y * w..(y + 1) * w].copy_from_slice(&oi);
        }
        // Columns.
        let mut cr = vec![0.0f32; h * w];
        let mut ci = vec![0.0f32; h * w];
        for x in 0..w {
            let col_re: Vec<f32> = (0..h).map(|y| rr[y * w + x]).collect();
            let col_im: Vec<f32> = (0..h).map(|y| ri[y * w + x]).collect();
            let (or, oi) = dft(&col_re, &col_im);
            for y in 0..h {
                cr[y * w + x] = or[y];
                ci[y * w + x] = oi[y];
            }
        }

        let (dr, di) = dft2(&re, &im, h, w);
        for k in 0..h * w {
            assert!(
                (dr[k] - cr[k]).abs() < 1e-2,
                "re[{k}]: {} vs {}",
                dr[k],
                cr[k]
            );
            assert!(
                (di[k] - ci[k]).abs() < 1e-2,
                "im[{k}]: {} vs {}",
                di[k],
                ci[k]
            );
        }
    }

    #[test]
    fn dft2_round_trip() {
        let (h, w) = (4, 4);
        let re: Vec<f32> = (0..h * w)
            .map(|k| ((k * 13 + 2) % 7) as f32 - 3.0)
            .collect();
        let im: Vec<f32> = (0..h * w).map(|k| ((k * 3 + 5) % 8) as f32 - 4.0).collect();
        let (fr, fi) = dft2(&re, &im, h, w);
        let (rr, ri) = idft2(&fr, &fi, h, w);
        for k in 0..h * w {
            assert!((rr[k] - re[k]).abs() < 1e-3, "re[{k}]");
            assert!((ri[k] - im[k]).abs() < 1e-3, "im[{k}]");
        }
    }
}

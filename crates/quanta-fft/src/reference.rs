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
}

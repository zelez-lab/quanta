//! 2-D complex FFT via row-column decomposition, built on [`FftPlan`].
//!
//! The 2-D DFT of an `H×W` grid is separable: transform every row with a
//! length-`W` 1-D FFT, then every column with a length-`H` 1-D FFT. [`fft2`]
//! runs exactly that, reusing the radix-2 [`FftPlan`] as the 1-D engine:
//!
//! 1. **Row pass** — one `FftPlan::new(gpu, W, …)`, executed once per row
//!    (`H` executes; the plan's compiled kernels and twiddle table are shared
//!    across all rows).
//! 2. **Transpose** (host-side) — the `H×W` grid becomes `W×H`, so the
//!    columns become contiguous rows.
//! 3. **Row pass again** — one `FftPlan::new(gpu, H, …)`, executed once per
//!    transposed row (`W` executes). This is the column pass.
//! 4. **Transpose back** to the original `H×W` row-major layout.
//!
//! The transpose is done on the host because [`FftPlan::execute`] already
//! round-trips through host memory per row; a host transpose between passes
//! adds no extra device traffic and keeps the indexing trivially correct.
//!
//! [`ifft2`] is the same pipeline with inverse plans: each pass folds in its
//! own `1/n` scale (`1/W` on the row pass, `1/H` on the column pass), so the
//! total is the required `1/(H·W)` and `ifft2(fft2(x)) == x`.
//!
//! Both dimensions must be powers of 2; anything else returns `NotSupported`.

use quanta::{Gpu, QuantaError};

use crate::plan::FftPlan;

/// Forward 2-D FFT of a row-major `height×width` grid (split complex:
/// `re`/`im` of length `height·width`). Returns the 2-D spectrum in the same
/// row-major layout.
///
/// Both `height` and `width` must be powers of 2 (`NotSupported` otherwise);
/// `re`/`im` must both have length `height·width`.
pub fn fft2(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    height: usize,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run2(gpu, re, im, height, width, false)
}

/// Inverse 2-D FFT (`+` twiddle sign, `1/(height·width)` scale):
/// `ifft2(fft2(x)) == x`. Same layout and size rules as [`fft2`].
pub fn ifft2(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    height: usize,
    width: usize,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run2(gpu, re, im, height, width, true)
}

/// Shared pipeline: row pass → transpose → row pass → transpose back.
fn run2(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    height: usize,
    width: usize,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    if re.len() != im.len() {
        return Err(QuantaError::invalid_param("fft2: re/im length mismatch"));
    }
    if !height.is_power_of_two() || !width.is_power_of_two() {
        return Err(QuantaError::not_supported(
            "fft2: height and width must both be powers of 2",
        ));
    }
    if re.len() != height * width {
        return Err(QuantaError::invalid_param(
            "fft2: input length differs from height * width",
        ));
    }

    // 1. Row pass: W-point FFT on each of the H rows.
    let (row_re, row_im) = row_pass(gpu, re, im, height, width, inverse)?;

    // 2. Transpose H×W → W×H so the columns become contiguous rows.
    let t_re = transpose(&row_re, height, width);
    let t_im = transpose(&row_im, height, width);

    // 3. Column pass = row pass on the transposed grid: H-point FFT on each
    //    of the W (former-column) rows.
    let (col_re, col_im) = row_pass(gpu, &t_re, &t_im, width, height, inverse)?;

    // 4. Transpose W×H back to the original H×W row-major layout.
    Ok((
        transpose(&col_re, width, height),
        transpose(&col_im, width, height),
    ))
}

/// 1-D FFT of every row of a row-major `rows×cols` grid, reusing a single
/// [`FftPlan`] of size `cols` (kernels compiled once, twiddles uploaded once).
fn row_pass(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    rows: usize,
    cols: usize,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    let mut plan = FftPlan::new(gpu, cols, inverse)?;
    let mut out_re = Vec::with_capacity(rows * cols);
    let mut out_im = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let span = r * cols..(r + 1) * cols;
        let (rr, ri) = plan.execute(&re[span.clone()], &im[span])?;
        out_re.extend_from_slice(&rr);
        out_im.extend_from_slice(&ri);
    }
    Ok((out_re, out_im))
}

/// Out-of-place transpose of a row-major `rows×cols` grid into `cols×rows`.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut out = vec![0.0f32; src.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

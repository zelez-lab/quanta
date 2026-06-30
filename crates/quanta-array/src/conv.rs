//! `im2col` / `col2im` — the convolution-as-matmul substrate.
//!
//! A 2-D convolution `y = conv(x, w)` (NCHW) is an `im2col` unfold followed by
//! a matmul: each output spatial position becomes a row of patch taps, and the
//! kernel becomes a matrix, so `cols · W = y`. quanta-autograd's `conv2d`
//! composes these with the proven `matmul` VJP; the backward needs `col2im`,
//! the adjoint of `im2col` (overlapping patches scatter-add back).
//!
//! Layouts (row-major, NCHW):
//! - input  `x`    : `[N, Cin, H, W]`
//! - cols   (im2col output) : `[N·OH·OW, Cin·kh·kw]`, where
//!   `OH = (H + 2·pad − kh)/stride + 1`, `OW` likewise.
//!
//! Padding reads 0 outside `[0,H)×[0,W)` (zero-padding) via a clamped index and
//! a 0/1 in-bounds mask, so the load is never out of range.

use quanta_ir::serialize_kernel;
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::FloatScalar;

mod kernels;
use kernels::{build_col2im_def, build_im2col_def};

/// Output spatial size for one dimension: `(in + 2·pad − k)/stride + 1`.
pub fn conv_out(in_dim: usize, k: usize, stride: usize, pad: usize) -> usize {
    (in_dim + 2 * pad - k) / stride + 1
}

impl<T: FloatScalar> Array<T> {
    /// Unfold an NCHW input `[N, Cin, H, W]` into the im2col matrix
    /// `[N·OH·OW, Cin·kh·kw]`. Zero-padded; one GPU thread per output element.
    pub fn im2col(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Array<T>, ArrayError> {
        let dims = self.shape();
        if dims.len() != 4 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "im2col: input must be 4-D NCHW",
            )));
        }
        let (n, cin, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let oh = conv_out(h, kh, stride, pad);
        let ow = conv_out(w, kw, stride, pad);
        let rows = n * oh * ow;
        let kdim = cin * kh * kw;
        let n_out = rows * kdim;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;

        let def = build_im2col_def(n, cin, h, w, kh, kw, stride, pad, oh, ow, kdim, ty);
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[rows, kdim])?,
        ))
    }

    /// Fold an im2col matrix `[N·OH·OW, Cin·kh·kw]` back into an NCHW gradient
    /// `[N, Cin, H, W]` — the adjoint of [`im2col`](Self::im2col). Overlapping
    /// patches accumulate. One thread per **input** pixel sums the cols entries
    /// that map to it (a gather, so no atomics). `self` is the cols matrix; the
    /// output shape `(n, cin, h, w)` and conv params are given.
    #[allow(clippy::too_many_arguments)]
    pub fn col2im(
        &self,
        n: usize,
        cin: usize,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Array<T>, ArrayError> {
        let oh = conv_out(h, kh, stride, pad);
        let ow = conv_out(w, kw, stride, pad);
        let kdim = cin * kh * kw;
        let n_out = n * cin * h * w;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;

        let def = build_col2im_def(cin, h, w, kh, kw, stride, pad, oh, ow, kdim, ty);
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, cin, h, w])?,
        ))
    }
}

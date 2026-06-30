//! 2-D pooling (NCHW) — `avgpool2d` and `maxpool2d` plus their backwards.
//!
//! Pooling slides a `kh×kw` window (stride `stride`, zero-pad `pad`) over each
//! channel independently and reduces it: average or maximum. Input `[N,C,H,W]`
//! → output `[N,C,OH,OW]` with `OH = (H+2·pad−kh)/stride+1` (`OW` likewise).
//!
//! Unlike convolution there is no cross-channel mixing, so these are *not*
//! im2col + matmul — they are direct windowed reductions, one GPU thread per
//! output element (the kernels live in [`kernels`]). The backwards are gather
//! kernels (one thread per **input** pixel summing the outputs it fed), so no
//! atomics and deterministic:
//!
//! - **avgpool** is linear; each input pixel's gradient is `Σ g/(kh·kw)` over
//!   the windows containing it. Padding counts toward the divisor
//!   (`count_include_pad`-style), so the divisor is the constant `kh·kw` and the
//!   backward is a uniform scatter — the same adjoint shape as `col2im`.
//! - **maxpool** is nonlinear; the forward also emits an `argmax` array (the
//!   flat input index of each window's winner), and the backward routes each
//!   output's gradient to exactly that pixel (the subgradient).

use quanta_ir::serialize_kernel;
use quanta_tensor::Layout;

use crate::array::Array;
use crate::conv::conv_out;
use crate::error::ArrayError;
use crate::scalar::FloatScalar;

mod kernels;
use kernels::{build_avgpool_bwd_def, build_avgpool_def, build_maxpool_bwd_def, build_maxpool_def};

impl<T: FloatScalar> Array<T> {
    /// Average pooling over `kh×kw` windows (NCHW). Padding contributes 0 to the
    /// numerator and counts toward the divisor `kh·kw` (`count_include_pad`).
    pub fn avgpool2d(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Array<T>, ArrayError> {
        let (n, c, h, w, oh, ow) = self.pool_dims(kh, kw, stride, pad)?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let n_out = n * c * oh * ow;
        let def = build_avgpool_def(c, h, w, kh, kw, stride, pad, oh, ow, ty);
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, c, oh, ow])?,
        ))
    }

    /// Backward of [`avgpool2d`](Self::avgpool2d): given the upstream gradient
    /// `grad[N,C,OH,OW]` (= `self`), scatter `grad/(kh·kw)` back to the input
    /// pixels each window averaged. One thread per **input** pixel sums the
    /// outputs whose window covers it (a gather — no atomics). Returns the
    /// input-shaped gradient `[N,C,H,W]`.
    #[allow(clippy::too_many_arguments)]
    pub fn avgpool2d_backward(
        &self,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 4 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "avgpool2d_backward: grad must be 4-D NCHW",
            )));
        }
        let (n, c, oh, ow) = (d[0], d[1], d[2], d[3]);
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let n_out = n * c * h * w;
        let def = build_avgpool_bwd_def(c, h, w, kh, kw, stride, pad, oh, ow, ty);
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, c, h, w])?,
        ))
    }

    /// Max pooling over `kh×kw` windows (NCHW). Returns `(values, argmax)`:
    /// `values[N,C,OH,OW]` is the per-window maximum, and `argmax[N,C,OH,OW]`
    /// (a `u32` array) is the **flat input index** of each window's winner — the
    /// information [`maxpool2d_backward`](Self::maxpool2d_backward) needs to
    /// route the gradient. Assumes each window covers at least one real (non-pad)
    /// pixel (the usual `pad ≤ k/2`).
    pub fn maxpool2d(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<(Array<T>, Array<u32>), ArrayError> {
        let (n, c, h, w, oh, ow) = self.pool_dims(kh, kw, stride, pad)?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let n_out = n * c * oh * ow;
        let def = build_maxpool_def(c, h, w, kh, kw, stride, pad, oh, ow, ty);
        let vals = self.gpu().field::<T>(n_out)?;
        let argi = self.gpu().field::<u32>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &vals);
        wave.bind(2, &argi);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        let layout = Layout::row_major(&[n, c, oh, ow])?;
        Ok((
            Array::from_parts(self.gpu().clone(), vals, layout.clone()),
            Array::from_parts(self.gpu().clone(), argi, layout),
        ))
    }

    /// Backward of [`maxpool2d`](Self::maxpool2d): route each output's gradient
    /// to the input pixel that won its window. `self` is the upstream gradient
    /// `grad[N,C,OH,OW]`; `argmax` is the index array `maxpool2d` returned. One
    /// thread per **input** pixel `p` sums `grad[out]` over the outputs whose
    /// window covers `p` and whose `argmax == p` (a gather — no atomics).
    #[allow(clippy::too_many_arguments)]
    pub fn maxpool2d_backward(
        &self,
        argmax: &Array<u32>,
        h: usize,
        w: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 4 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "maxpool2d_backward: grad must be 4-D NCHW",
            )));
        }
        if argmax.shape() != d {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "maxpool2d_backward: argmax shape must match grad",
            )));
        }
        let (n, c, oh, ow) = (d[0], d[1], d[2], d[3]);
        let ty = T::scalar_type();
        let grad = self.contiguous_or_self()?;
        let argc = argmax.contiguous_or_self()?;
        let n_out = n * c * h * w;
        let def = build_maxpool_bwd_def(c, h, w, kh, kw, stride, pad, oh, ow, ty);
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, grad.field_ref());
        wave.bind(1, argc.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, c, h, w])?,
        ))
    }

    /// Common shape decode for the pooling ops; errors unless `self` is 4-D.
    fn pool_dims(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<(usize, usize, usize, usize, usize, usize), ArrayError> {
        let d = self.shape();
        if d.len() != 4 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pool: input must be 4-D NCHW",
            )));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        Ok((
            n,
            c,
            h,
            w,
            conv_out(h, kh, stride, pad),
            conv_out(w, kw, stride, pad),
        ))
    }
}

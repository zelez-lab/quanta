//! Differentiable 2-D pooling (NCHW) — `avgpool2d` and `maxpool2d`.
//!
//! Both forwards run the `quanta-array` pooling kernel and record the geometry
//! needed for the matching gather backward (see [`crate::pool::PoolParams`]).
//! avgpool is linear, so its backward is `avgpool2d_backward` (its own adjoint);
//! maxpool captures the `argmax` index array the forward produced and routes
//! each output's gradient to the winning input pixel.

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::{Op, Var};

/// Geometry of a pooling op, captured on the tape so the backward can size and
/// fold the gradient (input `H`/`W` aren't recoverable from the pooled output).
#[derive(Clone)]
pub struct PoolParams {
    pub h: usize,
    pub w: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl<T: DiffScalar> Var<T> {
    /// Average pooling over `kh×kw` windows (NCHW). Padding counts toward the
    /// `kh·kw` divisor (`count_include_pad`); the VJP is `avgpool2d_backward`.
    pub fn avgpool2d(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = x.shape();
        if d.len() != 4 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param("avgpool2d: input must be 4-D NCHW"),
            )));
        }
        let (h, w) = (d[2], d[3]);
        let y = x.avgpool2d(kh, kw, stride, pad)?;
        let p = PoolParams {
            h,
            w,
            kh,
            kw,
            stride,
            pad,
        };
        Ok(self.tape_handle().push(Op::AvgPool2d(self.id, p), y))
    }

    /// Nearest-neighbour spatial upsample by factor `k` (NCHW): grow
    /// `[N, C, H, W]` to `[N, C, H·k, W·k]`, each pixel replicated over a k×k
    /// block. The decoder counterpart to pooling; the gradient sums each k×k
    /// output block back to its source pixel.
    pub fn upsample2d(&self, k: usize) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = x.shape();
        if d.len() != 4 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param("upsample2d: input must be 4-D NCHW"),
            )));
        }
        let (h, w) = (d[2], d[3]);
        let y = x.upsample2d(k)?;
        Ok(self.tape_handle().push(Op::Upsample2d(self.id, k, h, w), y))
    }

    /// Max pooling over `kh×kw` windows (NCHW). The forward also produces an
    /// argmax index array (captured on the tape) so the backward routes each
    /// output's gradient to exactly the input pixel that won its window.
    pub fn maxpool2d(
        &self,
        kh: usize,
        kw: usize,
        stride: usize,
        pad: usize,
    ) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let d = x.shape();
        if d.len() != 4 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param("maxpool2d: input must be 4-D NCHW"),
            )));
        }
        let (h, w) = (d[2], d[3]);
        let (y, argmax) = x.maxpool2d(kh, kw, stride, pad)?;
        let p = PoolParams {
            h,
            w,
            kh,
            kw,
            stride,
            pad,
        };
        Ok(self
            .tape_handle()
            .push(Op::MaxPool2d(self.id, argmax, p), y))
    }
}

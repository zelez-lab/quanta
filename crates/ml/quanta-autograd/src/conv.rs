//! Differentiable 2-D convolution (NCHW), built as `im2col → matmul → reshape`.
//!
//! The forward unfolds the input `x[N,Cin,H,W]` into the patch matrix
//! `cols[N·OH·OW, Cin·kh·kw]` ([`Array::im2col`](quanta_array::Array::im2col)),
//! flattens the weight `w[Cout,Cin,kh,kw]` to `wm[Cin·kh·kw, Cout]`, multiplies
//! `cols · wm → ym[N·OH·OW, Cout]`, and reshapes/permutes `ym` to the NCHW
//! output `y[N,Cout,OH,OW]`.
//!
//! Because the only nonlinear step is the matmul, the backward **reuses the
//! proven matmul VJP**: with `G = ∂L/∂y` reshaped to `Gm[N·OH·OW, Cout]`,
//! `∂cols = Gm · wmᵀ` and `∂wm = colsᵀ · Gm`. `∂x` then comes from
//! [`col2im`](quanta_array::Array::col2im), the adjoint of `im2col`, and `∂w`
//! is `∂wm` reshaped back. No new gradient math — it is matmul + the linear
//! im2col/col2im pair, composed.

use quanta_array::Array;

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::{Op, Var};

/// Geometry of a single `conv2d`, captured on the tape so the backward pass can
/// reshape/fold gradients without re-deriving the shapes.
#[derive(Clone)]
pub struct ConvParams {
    pub n: usize,
    pub cin: usize,
    pub h: usize,
    pub w: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
    pub oh: usize,
    pub ow: usize,
}

/// Flatten a weight `w[Cout,Cin,kh,kw]` to the matmul matrix
/// `wm[Cin·kh·kw, Cout]` (transpose of the row-major `[Cout, Cin·kh·kw]`).
fn weight_to_matrix<T: DiffScalar>(
    w: &Array<T>,
    cout: usize,
    kdim: usize,
) -> Result<Array<T>, AutogradError> {
    // [Cout, Cin, kh, kw] → [Cout, Cin·kh·kw] → transpose → [Cin·kh·kw, Cout].
    let flat = w.reshape(&[cout, kdim])?;
    Ok(flat.transpose(0, 1)?.contiguous()?)
}

impl<T: DiffScalar> Var<T> {
    /// 2-D convolution `self (x[N,Cin,H,W]) ⊛ w[Cout,Cin,kh,kw] → y[N,Cout,OH,OW]`
    /// with zero-padding `pad` and `stride`. Implemented as im2col + matmul; the
    /// VJP reuses matmul's (∂cols/∂w) and col2im (∂x = im2col-adjoint).
    pub fn conv2d(&self, w: &Var<T>, stride: usize, pad: usize) -> Result<Var<T>, AutogradError> {
        if !std::rc::Rc::ptr_eq(&self.tape, &w.tape) {
            return Err(AutogradError::ForeignVar);
        }
        let x = self.value();
        let wt = w.value();
        let xs = x.shape();
        let ws = wt.shape();
        if xs.len() != 4 || ws.len() != 4 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param("conv2d: x and w must be 4-D NCHW"),
            )));
        }
        let (n, cin, h, w_in) = (xs[0], xs[1], xs[2], xs[3]);
        let (cout, kh, kw) = (ws[0], ws[2], ws[3]);
        if ws[1] != cin {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta_core::QuantaError::invalid_param("conv2d: weight Cin must match input Cin"),
            )));
        }
        let oh = quanta_array::conv_out(h, kh, stride, pad);
        let ow = quanta_array::conv_out(w_in, kw, stride, pad);
        let kdim = cin * kh * kw;

        // Forward: cols[N·OH·OW, kdim] · wm[kdim, Cout] → ym[N·OH·OW, Cout].
        let cols = x.im2col(kh, kw, stride, pad)?;
        let wm = weight_to_matrix(&wt, cout, kdim)?;
        let ym = T::array_matmul(&cols, &wm)?;
        // ym[N·OH·OW, Cout] → [N, OH, OW, Cout] → permute → [N, Cout, OH, OW].
        let y = ym
            .reshape(&[n, oh, ow, cout])?
            .permute(&[0, 3, 1, 2])?
            .contiguous()?;

        let p = ConvParams {
            n,
            cin,
            h,
            w: w_in,
            cout,
            kh,
            kw,
            stride,
            pad,
            oh,
            ow,
        };
        Ok(self
            .tape_handle()
            .push(Op::Conv2d(self.id, w.id, cols, wm, p), y))
    }
}

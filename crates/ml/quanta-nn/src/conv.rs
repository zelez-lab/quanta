//! Conv2d and the pooling layers — module forms over the existing
//! autograd ops (`Var::conv2d` = im2col + matmul with the col2im-adjoint
//! backward; `Var::{maxpool2d, avgpool2d}`). NCHW throughout; these are
//! rank-4 layers, so the 2-D last-dim width contracts don't apply —
//! `in_dim` is `None` and the ops themselves check shapes loudly.

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::Gpu;

use crate::layer::{Key, Layer, LinearParams, LinearVars};

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        msg,
    )))
}

/// 2-D convolution `x[N,Cin,H,W] ⊛ w[Cout,Cin,kh,kw] → y[N,Cout,OH,OW]`
/// with zero padding and stride; optional per-channel bias. Params reuse
/// the [`LinearParams`] tree (`w` + optional `b` — same names, same
/// optimizer surface). Kaiming-uniform init over `fan_in = Cin·kh·kw`.
pub struct Conv2d {
    pub cin: usize,
    pub cout: usize,
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
    pub bias: bool,
}

impl<T: DiffScalar + ToF64> Layer<T> for Conv2d {
    type Params = LinearParams<T>;

    fn in_dim(&self) -> Option<usize> {
        None // NCHW — the op checks ranks/channels, not last-dim width.
    }
    fn out_dim(&self, in_dim: usize) -> usize {
        in_dim
    }

    fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError> {
        if self.cin == 0 || self.cout == 0 || self.kh == 0 || self.kw == 0 || self.stride == 0 {
            return Err(bad("Conv2d: cin/cout/kh/kw/stride must be nonzero"));
        }
        let fan_in = self.cin * self.kh * self.kw;
        let bound = (6.0 / fan_in as f32).sqrt();
        let (kw_key, _kb) = key.split();
        let w_host: Vec<T> = kw_key
            .uniform(self.cout * fan_in, -bound, bound)
            .iter()
            .map(|&v| T::from_f64(v as f64))
            .collect();
        let w = Array::from_slice(gpu, &w_host, &[self.cout, self.cin, self.kh, self.kw])
            .map_err(AutogradError::from)?;
        let b = if self.bias {
            let zeros: Vec<T> = (0..self.cout).map(|_| T::from_f64(0.0)).collect();
            Some(Array::from_slice(gpu, &zeros, &[self.cout]).map_err(AutogradError::from)?)
        } else {
            None
        };
        Ok(LinearParams { w, b })
    }

    fn apply(
        &self,
        _tape: &Tape<T>,
        p: &LinearVars<T>,
        x: &Var<T>,
    ) -> Result<Var<T>, AutogradError> {
        let y = x.conv2d(&p.w, self.stride, self.pad)?;
        match &p.b {
            Some(b) => y.add(&b.reshape(&[1, self.cout, 1, 1])?),
            None => Ok(y),
        }
    }
}

/// Max pooling `[N,C,H,W] → [N,C,OH,OW]` — a zero-param layer over
/// [`Var::maxpool2d`] (winner-takes-the-gradient backward).
pub struct MaxPool2d {
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl<T: DiffScalar + ToF64> Layer<T> for MaxPool2d {
    type Params = ();

    fn in_dim(&self) -> Option<usize> {
        None
    }
    fn out_dim(&self, in_dim: usize) -> usize {
        in_dim
    }
    fn init(&self, _gpu: &Gpu, _key: Key) -> Result<(), AutogradError> {
        Ok(())
    }
    fn apply(&self, _tape: &Tape<T>, _p: &(), x: &Var<T>) -> Result<Var<T>, AutogradError> {
        x.maxpool2d(self.kh, self.kw, self.stride, self.pad)
    }
}

/// Average pooling `[N,C,H,W] → [N,C,OH,OW]` — a zero-param layer over
/// [`Var::avgpool2d`] (uniform-spread backward).
pub struct AvgPool2d {
    pub kh: usize,
    pub kw: usize,
    pub stride: usize,
    pub pad: usize,
}

impl<T: DiffScalar + ToF64> Layer<T> for AvgPool2d {
    type Params = ();

    fn in_dim(&self) -> Option<usize> {
        None
    }
    fn out_dim(&self, in_dim: usize) -> usize {
        in_dim
    }
    fn init(&self, _gpu: &Gpu, _key: Key) -> Result<(), AutogradError> {
        Ok(())
    }
    fn apply(&self, _tape: &Tape<T>, _p: &(), x: &Var<T>) -> Result<Var<T>, AutogradError> {
        x.avgpool2d(self.kh, self.kw, self.stride, self.pad)
    }
}

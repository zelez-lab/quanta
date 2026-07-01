//! Classification losses — `log_softmax` and `cross_entropy`.
//!
//! Both are *composed* from the existing differentiable ops, so the tape
//! backpropagates them automatically — no new VJP. The only subtlety is
//! numerical stability: softmax subtracts the per-row max before `exp`, and that
//! max is taken as a **detached constant** (it shifts every logit equally, so it
//! cancels in softmax and contributes no gradient). `max_axis_last` is a
//! non-differentiable `Array` op, which is exactly what we want for the shift.

use quanta_array::Array;

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::{Tape, Var};

impl<T: DiffScalar> Var<T> {
    /// Numerically-stable log-softmax over the class axis of a `[N, C]` logit
    /// matrix: `log_softmax(x)[i,c] = x[i,c] − max_c − log Σ_c exp(x[i,c] − max_c)`.
    /// The row max is subtracted as a detached constant (it cancels in softmax),
    /// giving the same result without overflowing `exp`.
    pub fn log_softmax(&self) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        if x.shape().len() != 2 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("log_softmax: input must be 2-D [N, C]"),
            )));
        }
        let (n, c) = (x.shape()[0], x.shape()[1]);

        // Detached per-row max, broadcast to [N, C], as a constant leaf Var.
        let row_max = x.max_axis_last()?.broadcast_to(&[n, c])?.contiguous()?;
        let tape = Tape::from_inner(std::rc::Rc::clone(&self.tape));
        let max_c = tape.var(row_max);

        // shifted = x − max ; then x_shift − log(sum(exp(x_shift)))
        let shifted = self.sub(&max_c)?;
        let sum_exp = shifted.exp()?.sum_axis(1)?; // [N, 1]
        let log_sum = sum_exp.log()?; // [N, 1]
        // broadcast log_sum [N,1] over [N,C] via sub (Var::sub broadcasts).
        shifted.sub(&log_sum)
    }

    /// Cross-entropy loss for classification: mean over the batch of
    /// `−log_softmax(logits)[i, label[i]]`. `self` is the `[N, C]` logits;
    /// `labels` is a `[N]` `u32` array of true classes (not differentiated).
    /// Returns a scalar `Var`. This is the standard `F.cross_entropy`.
    pub fn cross_entropy(&self, labels: &Array<u32>) -> Result<Var<T>, AutogradError> {
        let logp = self.log_softmax()?; // [N, C]
        let picked = logp.gather_rows(labels)?; // [N] — log-prob at the label
        // loss = mean(−picked) = −mean(picked)
        picked.neg()?.mean_axis(0)?.sum()
    }
}

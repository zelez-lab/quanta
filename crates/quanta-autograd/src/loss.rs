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

    /// Softmax over the class axis of a `[N, C]` matrix: `exp(log_softmax(x))`,
    /// so it inherits the numerically-stable shift. Rows sum to 1 — the
    /// probability form for inference and attention.
    pub fn softmax(&self) -> Result<Var<T>, AutogradError> {
        self.log_softmax()?.exp()
    }

    /// Layer normalization over the **last axis** of a `[N, C]` matrix:
    /// normalize each row to zero mean and unit variance, then scale and shift
    /// by the learnable `gamma` / `beta` (each `[C]`, broadcast over the rows):
    /// `out = (x − μ) / √(σ² + eps) · γ + β`. The normalization every
    /// transformer and deep net uses to keep activations well-scaled — composed
    /// from the reduction + elementwise ops, so the tape backprops it directly.
    pub fn layer_norm(
        &self,
        gamma: &Var<T>,
        beta: &Var<T>,
        eps: f64,
    ) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        if x.shape().len() != 2 {
            return Err(AutogradError::from(quanta_array::ArrayError::Gpu(
                quanta::QuantaError::invalid_param("layer_norm: input must be 2-D [N, C]"),
            )));
        }
        let (n, c) = (x.shape()[0], x.shape()[1]);
        let g = x.gpu();

        // μ = mean over C (keepdims [N,1]); centered = x − μ (broadcasts)
        let mean = self.mean_axis(1)?; // [N, 1]
        let centered = self.sub(&mean)?; // [N, C]
        // σ² = mean(centered²) over C ; std = √(σ² + eps)
        let var = centered.mul(&centered)?.mean_axis(1)?; // [N, 1]
        let eps_v = {
            let tape = Tape::from_inner(std::rc::Rc::clone(&self.tape));
            tape.var(
                Array::full(g, T::from_f64(eps), &[1])?
                    .broadcast_to(&[n, 1])?
                    .contiguous()?,
            )
        };
        let std = var.add(&eps_v)?.sqrt()?; // [N, 1]
        let normed = centered.div(&std)?; // [N, C], broadcasts [N,1] over C

        // scale + shift: γ and β are [C]; reshape to [1, C] and let Var::mul /
        // Var::add broadcast them over the N rows.
        let g2 = gamma.reshape(&[1, c])?;
        let b2 = beta.reshape(&[1, c])?;
        normed.mul(&g2)?.add(&b2)
    }

    /// Mean of all elements → a scalar `Var`. `sum / n`, with `n` a detached
    /// constant. The reduction for an average loss or a batch mean.
    pub fn mean(&self) -> Result<Var<T>, AutogradError> {
        let n = self.value().len();
        let s = self.sum()?;
        let inv = Array::full(self.value().gpu(), T::from_f64(1.0 / n as f64), &[1])?;
        let tape = Tape::from_inner(std::rc::Rc::clone(&self.tape));
        s.mul(&tape.var(inv))
    }

    /// Mean squared error against a (non-differentiable) target:
    /// `mean((self − target)²)`. The standard regression loss. `target` must
    /// match `self`'s shape.
    pub fn mse_loss(&self, target: &Array<T>) -> Result<Var<T>, AutogradError> {
        let tape = Tape::from_inner(std::rc::Rc::clone(&self.tape));
        let t = tape.var(target.shallow_clone());
        let diff = self.sub(&t)?;
        diff.mul(&diff)?.mean()
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

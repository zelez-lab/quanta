//! BatchNorm — the state-in/state-out layer (decision D5).
//!
//! The running statistics are a VALUE the caller threads, exactly like
//! the optimizer state and the RNG key: `apply_train(params, stats, x)
//! → (y, stats')`. Nothing hides in the module, so a checkpoint is
//! `state::save_state(&stats)` (BnStats derives `ParamTree` for the
//! traversal) — but the stats are never bound to a tape and never fed
//! to an optimizer: state, not parameters.
//!
//! Training normalizes by the BATCH statistics, composed from per-op
//! proven VJPs (`mean_axis`/`sub`/`mul`/`div`/`sqrt`) — the full
//! backward through the batch mean and variance falls out of the tape;
//! no new kernel, no new proof obligation. The running update is plain
//! host arithmetic on `[C]`-sized arrays (EMA with `momentum`, variance
//! stored UNBIASED per the usual convention; the batch normalization
//! itself uses the biased variance). Eval normalizes by the running
//! statistics as constants.
//!
//! Not a tuple-stackable [`Layer`](crate::layer::Layer): eval needs the
//! stats too, so both entry points carry them — the Embedding /
//! MHA-`attend` module precedent.
//!
//! 2-D core `[N, C]` (rows = batch, columns = channels); higher-rank
//! layouts flatten their spatial dims into N at the call site, the
//! host-loop convention the crate already uses.

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::Gpu;

use crate::layer::{Key, NormParams, NormVars};

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        msg,
    )))
}

/// The running statistics — threaded state (D5). Derives `ParamTree`
/// for named checkpointing (`"mean"`, `"var"`); never bind or optimize
/// it.
#[derive(crate::layer::ParamTree)]
#[param_tree(crate = crate)]
pub struct BnStats<T: DiffScalar> {
    pub mean: Array<T>,
    pub var: Array<T>,
}

/// Batch normalization over `[N, C]` with per-channel scale/shift.
pub struct BatchNorm {
    pub dim: usize,
    pub eps: f32,
    /// EMA factor for the running statistics: `new = (1−m)·old + m·batch`.
    pub momentum: f32,
}

impl BatchNorm {
    /// Learnable parameters: γ = 1, β = 0 (the [`NormParams`] tree —
    /// same shape LayerNorm uses, so it optimizes identically).
    pub fn init<T: DiffScalar + ToF64>(
        &self,
        gpu: &Gpu,
        _key: Key,
    ) -> Result<NormParams<T>, AutogradError> {
        let ones: Vec<T> = (0..self.dim).map(|_| T::from_f64(1.0)).collect();
        let zeros: Vec<T> = (0..self.dim).map(|_| T::from_f64(0.0)).collect();
        Ok(NormParams {
            gamma: Array::from_slice(gpu, &ones, &[self.dim]).map_err(AutogradError::from)?,
            beta: Some(Array::from_slice(gpu, &zeros, &[self.dim]).map_err(AutogradError::from)?),
        })
    }

    /// Fresh running statistics: mean 0, variance 1 (the state a
    /// never-trained eval would normalize with).
    pub fn init_stats<T: DiffScalar + ToF64>(
        &self,
        gpu: &Gpu,
    ) -> Result<BnStats<T>, AutogradError> {
        let zeros: Vec<T> = (0..self.dim).map(|_| T::from_f64(0.0)).collect();
        let ones: Vec<T> = (0..self.dim).map(|_| T::from_f64(1.0)).collect();
        Ok(BnStats {
            mean: Array::from_slice(gpu, &zeros, &[self.dim]).map_err(AutogradError::from)?,
            var: Array::from_slice(gpu, &ones, &[self.dim]).map_err(AutogradError::from)?,
        })
    }

    fn check_shapes<T: DiffScalar>(
        &self,
        p: &NormVars<T>,
        x: &Var<T>,
    ) -> Result<usize, AutogradError> {
        let xs = x.value().shape().to_vec();
        if xs.len() != 2 || xs[1] != self.dim {
            return Err(bad("BatchNorm: input must be [N, dim]"));
        }
        if p.gamma.value().shape() != [self.dim] {
            return Err(bad("BatchNorm: gamma must be [dim]"));
        }
        if p.beta.is_none() {
            return Err(bad("BatchNorm: beta missing"));
        }
        Ok(xs[0])
    }

    /// Training forward: normalize by the batch statistics (fully
    /// differentiable through them) and return the EMA-updated running
    /// statistics. `N ≥ 2` — a single-row batch has no variance to
    /// normalize by.
    pub fn apply_train<T: DiffScalar + ToF64>(
        &self,
        tape: &Tape<T>,
        p: &NormVars<T>,
        stats: &BnStats<T>,
        x: &Var<T>,
    ) -> Result<(Var<T>, BnStats<T>), AutogradError> {
        let n = self.check_shapes(p, x)?;
        if n < 2 {
            return Err(bad("BatchNorm: training needs a batch of at least 2"));
        }
        let c = self.dim;
        let beta = p
            .beta
            .as_ref()
            .ok_or_else(|| bad("BatchNorm: beta missing"))?;

        // Batch stats, composed — the tape differentiates through them.
        let mu = x.mean_axis(0)?; // [C]
        let xc = x.sub(&mu.reshape(&[1, c])?)?;
        let var_b = xc.mul(&xc)?.mean_axis(0)?; // biased, for normalization
        let eps_host: Vec<T> = (0..c).map(|_| T::from_f64(self.eps as f64)).collect();
        let eps_v = tape.var(
            Array::from_slice(&x.value().gpu().clone(), &eps_host, &[c])
                .map_err(AutogradError::from)?,
        );
        let denom = var_b.add(&eps_v)?.sqrt()?;
        let xn = xc.div(&denom.reshape(&[1, c])?)?;
        let y = xn
            .mul(&p.gamma.reshape(&[1, c])?)?
            .add(&beta.reshape(&[1, c])?)?;

        // Running update — host arithmetic on [C] values; the variance
        // is stored unbiased (×N/(N−1)).
        let m = self.momentum as f64;
        let unbias = n as f64 / (n as f64 - 1.0);
        let batch_mean = mu.value().to_vec().map_err(AutogradError::from)?;
        let batch_var = var_b.value().to_vec().map_err(AutogradError::from)?;
        let old_mean = stats.mean.to_vec().map_err(AutogradError::from)?;
        let old_var = stats.var.to_vec().map_err(AutogradError::from)?;
        let gpu = x.value().gpu().clone();
        let new_mean: Vec<T> = old_mean
            .iter()
            .zip(&batch_mean)
            .map(|(o, b)| T::from_f64((1.0 - m) * o.to_f64() + m * b.to_f64()))
            .collect();
        let new_var: Vec<T> = old_var
            .iter()
            .zip(&batch_var)
            .map(|(o, b)| T::from_f64((1.0 - m) * o.to_f64() + m * b.to_f64() * unbias))
            .collect();
        let stats = BnStats {
            mean: Array::from_slice(&gpu, &new_mean, &[c]).map_err(AutogradError::from)?,
            var: Array::from_slice(&gpu, &new_var, &[c]).map_err(AutogradError::from)?,
        };
        Ok((y, stats))
    }

    /// Eval forward: normalize by the RUNNING statistics as constants.
    pub fn apply_eval<T: DiffScalar + ToF64>(
        &self,
        tape: &Tape<T>,
        p: &NormVars<T>,
        stats: &BnStats<T>,
        x: &Var<T>,
    ) -> Result<Var<T>, AutogradError> {
        self.check_shapes(p, x)?;
        let c = self.dim;
        if stats.mean.shape() != [c] || stats.var.shape() != [c] {
            return Err(bad("BatchNorm: stats must be [dim]"));
        }
        let beta = p
            .beta
            .as_ref()
            .ok_or_else(|| bad("BatchNorm: beta missing"))?;

        // Precompute (var + eps) host-side; the stats are constants.
        let var_host = stats.var.to_vec().map_err(AutogradError::from)?;
        let denom_host: Vec<T> = var_host
            .iter()
            .map(|v| T::from_f64((v.to_f64() + self.eps as f64).sqrt()))
            .collect();
        let gpu = x.value().gpu().clone();
        let mean_v = tape.var(stats.mean.shallow_clone());
        let denom_v =
            tape.var(Array::from_slice(&gpu, &denom_host, &[c]).map_err(AutogradError::from)?);
        let xn = x
            .sub(&mean_v.reshape(&[1, c])?)?
            .div(&denom_v.reshape(&[1, c])?)?;
        xn.mul(&p.gamma.reshape(&[1, c])?)?
            .add(&beta.reshape(&[1, c])?)
    }
}

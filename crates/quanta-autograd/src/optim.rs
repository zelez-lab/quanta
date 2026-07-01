//! Optimizers — SGD and Adam.
//!
//! An optimizer updates parameters from their gradients. This is plain array
//! math *outside* the tape (no autodiff, no new kernels) — the tape produces the
//! gradients; the optimizer consumes them. `Sgd` is the one-liner `p ← p − lr·g`;
//! `Adam` keeps per-parameter first/second moment buffers for the adaptive,
//! momentum-based update that converges fast in practice.

use quanta_array::{Array, ArrayError};

/// A scalar broadcast to `shape` — the idiom for folding a constant (lr, β, …)
/// into an elementwise array op.
fn scalar(a: &Array<f32>, v: f32) -> Result<Array<f32>, ArrayError> {
    Array::full(a.gpu(), v, &[1])?
        .broadcast_to(a.shape())?
        .contiguous()
}

/// Plain SGD: `p ← p − lr·g`.
pub struct Sgd {
    pub lr: f32,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Sgd { lr }
    }

    /// Return the updated parameter `p − lr·g`.
    pub fn step(&self, p: &Array<f32>, g: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        p.sub(&g.mul(&scalar(g, self.lr)?)?)
    }
}

/// Adam (Kingma & Ba). Keeps per-parameter first-moment (`m`) and second-moment
/// (`v`) running averages and applies the bias-corrected adaptive update
/// `p ← p − lr · m̂ / (√v̂ + ε)`. Register each parameter's slot once (in a fixed
/// order), then call [`step`](Self::step) with the same order each iteration.
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    t: i32,
    state: Vec<(Array<f32>, Array<f32>)>, // (m, v) per parameter slot
}

impl Adam {
    /// Adam with the usual defaults (β1=0.9, β2=0.999, ε=1e-8).
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            state: Vec::new(),
        }
    }

    /// Register a parameter slot, allocating its `m`/`v` buffers as zeros shaped
    /// like `p`. Call once per parameter, in the order you'll pass to `step`.
    pub fn register(&mut self, p: &Array<f32>) -> Result<(), ArrayError> {
        let m = Array::<f32>::zeros(p.gpu(), p.shape())?;
        let v = Array::<f32>::zeros(p.gpu(), p.shape())?;
        self.state.push((m, v));
        Ok(())
    }

    /// Advance the global step counter — call once per optimizer step, before the
    /// per-parameter `step` calls (it drives the bias correction).
    pub fn advance(&mut self) {
        self.t += 1;
    }

    /// Update parameter `slot` from its gradient, returning the new parameter.
    /// Call [`advance`](Self::advance) once per iteration first.
    pub fn step(
        &mut self,
        slot: usize,
        p: &Array<f32>,
        g: &Array<f32>,
    ) -> Result<Array<f32>, ArrayError> {
        let (b1, b2, eps, lr) = (self.beta1, self.beta2, self.eps, self.lr);
        let (m_prev, v_prev) = &self.state[slot];

        // m = β1·m + (1−β1)·g
        let m = m_prev
            .mul(&scalar(m_prev, b1)?)?
            .add(&g.mul(&scalar(g, 1.0 - b1)?)?)?;
        // v = β2·v + (1−β2)·g²
        let g2 = g.mul(g)?;
        let v = v_prev
            .mul(&scalar(v_prev, b2)?)?
            .add(&g2.mul(&scalar(&g2, 1.0 - b2)?)?)?;

        // Bias-corrected moments: m̂ = m/(1−β1ᵗ), v̂ = v/(1−β2ᵗ).
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        let mhat = m.mul(&scalar(&m, 1.0 / bc1)?)?;
        let vhat = v.mul(&scalar(&v, 1.0 / bc2)?)?;

        // p ← p − lr · m̂ / (√v̂ + ε)
        let denom = vhat.sqrt()?.add(&scalar(&vhat, eps)?)?;
        let update = mhat.div(&denom)?.mul(&scalar(&mhat, lr)?)?;
        let new_p = p.sub(&update)?;

        self.state[slot] = (m, v);
        Ok(new_p)
    }
}

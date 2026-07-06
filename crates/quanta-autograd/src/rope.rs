//! Rotary position embeddings (RoPE) — the LLaMA / GPT-NeoX positional
//! encoding. Instead of adding a position vector, RoPE *rotates* each query
//! and key vector by a position-dependent angle, which makes the QᵀK score a
//! function of the *relative* offset `m − n`.
//!
//! The rotation is the "rotate-half" form:
//! `rope(x, p) = x ⊙ cos_p + rotate_half(x) ⊙ sin_p`, where
//! `rotate_half([a, b]) = [−b, a]` splits the head dim into halves and swaps
//! them with a sign flip. `cos`/`sin` are precomputed `[T, d]` constants
//! (position × head-dim), so no gradient flows through them; the gradient of
//! `rope` w.r.t. `x` flows through the two elementwise products and the linear
//! `rotate_half`.
//!
//! `rotate_half` is realized as a right-multiply by a fixed `[d, d]` matrix
//! `R` (`x @ R`), so the whole op is composed from the already-differentiable
//! `matmul`/`mul`/`add` — no new adjoint. `R` encodes `out[j] = −x[j+d/2]`
//! (`j < d/2`) and `out[j] = x[j−d/2]` (`j ≥ d/2`).

use quanta_array::{Array, ArrayError};

use crate::error::AutogradError;
use crate::scalar::DiffScalar;
use crate::tape::{Tape, Var};

/// Precomputed RoPE tables for a fixed max sequence length `t` and head
/// dimension `d` (even). Holds the `[t, d]` `cos`/`sin` caches and the
/// `[d, d]` rotate-half matrix `R`, all as plain constant `Array`s.
pub struct RopeCache<T: DiffScalar> {
    /// `cos_p[i] = cos(p · θ_{i mod (d/2)})`, shape `[t, d]`.
    pub cos: Array<T>,
    /// `sin_p[i] = sin(p · θ_{i mod (d/2)})`, shape `[t, d]`.
    pub sin: Array<T>,
    /// The rotate-half matrix, shape `[d, d]`: `rotate_half(x) = x · R`.
    pub rot: Array<T>,
    pub t: usize,
    pub d: usize,
}

impl<T: DiffScalar> RopeCache<T> {
    /// Build the caches for sequence length `t`, head dim `d` (must be even),
    /// and the standard `base = 10000`. Angles are computed in `f64` and cast
    /// to `T` for accuracy at large positions.
    pub fn new(gpu: &quanta::Gpu, t: usize, d: usize, base: f64) -> Result<Self, ArrayError> {
        if !d.is_multiple_of(2) {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "RopeCache: head dim d must be even",
            )));
        }
        let half = d / 2;
        // θ_i = base^(−2i/d) for i in [0, d/2); repeated for the second half so
        // index i and i+d/2 share a frequency (they form one rotated pair).
        let mut cos = vec![0.0f64; t * d];
        let mut sin = vec![0.0f64; t * d];
        for p in 0..t {
            for i in 0..half {
                let theta = (p as f64) * base.powf(-2.0 * (i as f64) / (d as f64));
                let (c, s) = (theta.cos(), theta.sin());
                cos[p * d + i] = c;
                cos[p * d + i + half] = c;
                sin[p * d + i] = s;
                sin[p * d + i + half] = s;
            }
        }
        let cos_t: Vec<T> = cos.iter().map(|&v| T::from_f64(v)).collect();
        let sin_t: Vec<T> = sin.iter().map(|&v| T::from_f64(v)).collect();

        // R: out = x·R with out[j] = −x[j+half] (j<half), x[j−half] (j≥half).
        // Column j of R is the coefficients of x that produce out[j]:
        //   j < half : out[j] = −x[j+half]  → R[j+half][j] = −1
        //   j ≥ half : out[j] =  x[j−half]  → R[j−half][j] = +1
        let mut rot = vec![0.0f64; d * d];
        for j in 0..d {
            if j < half {
                rot[(j + half) * d + j] = -1.0;
            } else {
                rot[(j - half) * d + j] = 1.0;
            }
        }
        let rot_t: Vec<T> = rot.iter().map(|&v| T::from_f64(v)).collect();

        Ok(RopeCache {
            cos: Array::from_slice(gpu, &cos_t, &[t, d])?,
            sin: Array::from_slice(gpu, &sin_t, &[t, d])?,
            rot: Array::from_slice(gpu, &rot_t, &[d, d])?,
            t,
            d,
        })
    }
}

impl<T: DiffScalar> Var<T> {
    /// Apply rotary position embeddings to `self` along its last (head-dim)
    /// axis. `self` is `[…, T, d]` (e.g. `[B, H, T, d]` after the head split);
    /// the `cache`'s `[T, d]` `cos`/`sin` broadcast over the leading dims.
    ///
    /// `rope(x) = x ⊙ cos + (x · R) ⊙ sin`, all differentiable ops, so the
    /// gradient flows back to `x` through the tape with no hand-written adjoint.
    pub fn rope(&self, cache: &RopeCache<T>) -> Result<Var<T>, AutogradError> {
        let x = self.value();
        let shp = x.shape();
        let d = shp[shp.len() - 1];
        let seq = shp[shp.len() - 2];
        if d != cache.d {
            return Err(AutogradError::from(ArrayError::Gpu(
                quanta::QuantaError::invalid_param("rope: last dim must equal the cache head dim"),
            )));
        }
        if seq > cache.t {
            return Err(AutogradError::from(ArrayError::Gpu(
                quanta::QuantaError::invalid_param("rope: sequence length exceeds the cache"),
            )));
        }

        // Wrap the constant caches as (leaf) vars on this tape. cos/sin are
        // sliced to the actual sequence length and broadcast over leading dims.
        let tape = Tape::from_inner(std::rc::Rc::clone(&self.tape));
        let cos_slice = cache.cos.narrow(0, 0, seq)?.contiguous()?;
        let sin_slice = cache.sin.narrow(0, 0, seq)?.contiguous()?;
        let cos_v = tape.var(cos_slice);
        let sin_v = tape.var(sin_slice);
        let rot_v = tape.var(cache.rot.shallow_clone());

        // rope = x ⊙ cos + (x · R) ⊙ sin
        let x_cos = self.mul(&cos_v)?;
        let rot_half = self.matmul(&rot_v)?;
        let rh_sin = rot_half.mul(&sin_v)?;
        x_cos.add(&rh_sin)
    }
}

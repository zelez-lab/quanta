//! The Layer model — quanta-nn's architecture (decision record D1–D4),
//! deliberately NOT a torch mirror.
//!
//! * **D1 — functional core.** A layer is configuration only; its
//!   parameters live in an explicit, typed **parameter tree** ([`ParamTree`])
//!   that is bound to a tape per step. `apply(params, x) → y` is pure with
//!   respect to the layer. Params-as-data is what the upstream actor world
//!   needs: trees shard, checkpoint, and migrate as values.
//! * **D2 — ownership is the effect system.** [`Key`] is the RNG effect:
//!   `split` CONSUMES the key, so linearity is enforced by the borrow
//!   checker, not by convention (and not by a monad).
//! * **D3 — stacking is tuple composition.** `(l1, l2, l3)` is a [`Layer`]
//!   whose `Params` is the tuple of the members' trees — which is exactly
//!   the shard boundary a parameter-sharding consumer wants. Dimension
//!   CONTRACTS are checked once, at [`Layer::init`] time (build the model →
//!   learn about the mismatch), never per-forward.
//!
//! First increment: activations are 2-D `[N, in] → [N, out]` `Var`s and
//! trees flatten to ordered leaves (`flatten`/`unflatten` — the optimizer
//! surface). The derive macro for user-defined trees comes after these
//! hand-written impls have proven the trait shapes.

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::Gpu;
use quanta_rand::Rng;

/// A fallible leaf transform (see [`ParamTree::map`]).
pub type LeafFn<'a, T> = &'a mut dyn FnMut(&Array<T>) -> Result<Array<T>, AutogradError>;

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        msg,
    )))
}

// ── The RNG effect ───────────────────────────────────────────────────────

/// A splittable PRNG key (counter-based Philox underneath, via
/// `quanta-rand`). `split` and `fill` CONSUME the key: each key is usable
/// exactly once, which is the whole discipline explicit-effect RNG needs —
/// enforced here by ownership instead of a monad (decision D2).
#[derive(Debug, Clone, Copy)]
pub struct Key {
    seed: u64,
    stream: u64,
}

impl Key {
    pub fn new(seed: u64) -> Self {
        Key { seed, stream: 0 }
    }

    /// Split into two independent keys, consuming `self`.
    pub fn split(self) -> (Key, Key) {
        (
            Key {
                seed: self.seed,
                stream: self.stream.wrapping_mul(2).wrapping_add(1),
            },
            Key {
                seed: self.seed,
                stream: self.stream.wrapping_mul(2).wrapping_add(2),
            },
        )
    }

    /// Fill `n` values uniformly in `[lo, hi)`, consuming the key.
    pub fn uniform(self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        let mixed = self.seed ^ self.stream.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = Rng::from_seed((mixed ^ (mixed >> 32)) as u32);
        (0..n).map(|_| lo + (hi - lo) * rng.next_f32()).collect()
    }
}

// ── Parameter trees ──────────────────────────────────────────────────────

/// A typed tree of parameters. `Vars` is the tape-bound twin; `flatten` /
/// `unflatten` give the order-stable leaf view every optimizer works over
/// (optimizer state is a tree of the SAME shape — build it with
/// [`ParamTree::map`]).
pub trait ParamTree<T: DiffScalar>: Sized {
    /// The tape-bound twin (same tree shape, `Var` leaves).
    type Vars;

    /// Bind every leaf onto `tape` as a differentiable variable.
    fn bind(&self, tape: &Tape<T>) -> Self::Vars;

    /// Order-stable leaves (shallow clones — `Array` is a cheap handle).
    fn flatten(&self) -> Vec<Array<T>>;

    /// Rebuild a tree of this shape from leaves in `flatten` order.
    fn unflatten(&self, leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError>;

    /// Gradient tree: same shape, each leaf the gradient of `loss` w.r.t.
    /// the corresponding bound `Var`.
    fn grads(vars: &Self::Vars, loss: &Var<T>) -> Result<Self, AutogradError>;

    /// Inference-anchored gradient extraction: identical to
    /// [`ParamTree::grads`], with `&self` as the shape witness so call
    /// sites infer the tree type from an existing tree.
    fn grads_from(&self, vars: &Self::Vars, loss: &Var<T>) -> Result<Self, AutogradError> {
        Self::grads(vars, loss)
    }

    /// Map every leaf (e.g. zeros-like for optimizer moments).
    fn map(&self, f: LeafFn<'_, T>) -> Result<Self, AutogradError> {
        let mapped = self
            .flatten()
            .iter()
            .map(&mut *f)
            .collect::<Result<Vec<_>, _>>()?;
        self.unflatten(&mut mapped.into_iter())
    }
}

/// Derive `ParamTree` for your own parameter structs —
/// `#[derive(ParamTree)]` generates the `…Vars` twin plus
/// `bind`/`flatten`/`unflatten`/`grads`, delegating to each field in
/// declaration order. Same-named as the trait so one import serves both.
pub use quanta_nn_derive::ParamTree;

/// An optional subtree: `None` contributes no leaves. The shape witness
/// (`&self`) decides whether `unflatten` rebuilds `Some` — the reason
/// `unflatten` takes `&self` at all.
impl<T: DiffScalar, P: ParamTree<T>> ParamTree<T> for Option<P> {
    type Vars = Option<P::Vars>;

    fn bind(&self, tape: &Tape<T>) -> Option<P::Vars> {
        self.as_ref().map(|p| p.bind(tape))
    }
    fn flatten(&self) -> Vec<Array<T>> {
        self.as_ref().map(|p| p.flatten()).unwrap_or_default()
    }
    fn unflatten(&self, leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError> {
        match self {
            None => Ok(None),
            Some(p) => Ok(Some(p.unflatten(leaves)?)),
        }
    }
    fn grads(vars: &Option<P::Vars>, loss: &Var<T>) -> Result<Self, AutogradError> {
        match vars {
            None => Ok(None),
            Some(v) => Ok(Some(P::grads(v, loss)?)),
        }
    }
}

/// The empty tree — the `Params` of zero-parameter layers (activations,
/// reshapes). Contributes no leaves; occupies a tuple slot for free.
impl<T: DiffScalar> ParamTree<T> for () {
    type Vars = ();

    fn bind(&self, _tape: &Tape<T>) {}
    fn flatten(&self) -> Vec<Array<T>> {
        Vec::new()
    }
    fn unflatten(&self, _leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError> {
        Ok(())
    }
    fn grads(_vars: &(), _loss: &Var<T>) -> Result<Self, AutogradError> {
        Ok(())
    }
}

/// The leaf: a single tensor.
impl<T: DiffScalar> ParamTree<T> for Array<T> {
    type Vars = Var<T>;

    fn bind(&self, tape: &Tape<T>) -> Var<T> {
        tape.var(self.shallow_clone())
    }
    fn flatten(&self) -> Vec<Array<T>> {
        vec![self.shallow_clone()]
    }
    fn unflatten(&self, leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError> {
        leaves.next().ok_or_else(|| bad("unflatten: leaf underrun"))
    }
    fn grads(vars: &Var<T>, loss: &Var<T>) -> Result<Self, AutogradError> {
        loss.grad(vars)
    }
}

// ── The Layer trait ──────────────────────────────────────────────────────

/// A neural layer: configuration + shapes. Parameters are external
/// ([`ParamTree`]); `apply` is pure given them. Dimension contracts are
/// declared (`in_dim`/`out_dim`) and checked at composition/`init` time.
pub trait Layer<T: DiffScalar + ToF64> {
    type Params: ParamTree<T>;

    /// Expected input width (last-dim) — `None` = any.
    fn in_dim(&self) -> Option<usize>;
    /// Output width given the input width.
    fn out_dim(&self, in_dim: usize) -> usize;

    /// Allocate + initialize this layer's parameter tree.
    fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError>;

    /// The forward pass over tape-bound params. The tape is explicit —
    /// effects are arguments here, never ambient (decision D4).
    fn apply(
        &self,
        tape: &Tape<T>,
        params: &<Self::Params as ParamTree<T>>::Vars,
        x: &Var<T>,
    ) -> Result<Var<T>, AutogradError>;
}

// ── Concrete layers ──────────────────────────────────────────────────────

/// Dense affine layer `[N, in] → [N, out]`: `y = x·Wᵀ… ` stored as
/// `w: [in, out]` so `y = x @ w + b`. Kaiming-uniform init.
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub bias: bool,
}

/// Linear's parameter tree.
pub struct LinearParams<T: DiffScalar> {
    pub w: Array<T>,
    pub b: Option<Array<T>>,
}

pub struct LinearVars<T: DiffScalar> {
    pub w: Var<T>,
    pub b: Option<Var<T>>,
}

impl<T: DiffScalar> ParamTree<T> for LinearParams<T> {
    type Vars = LinearVars<T>;

    fn bind(&self, tape: &Tape<T>) -> LinearVars<T> {
        LinearVars {
            w: tape.var(self.w.shallow_clone()),
            b: self.b.as_ref().map(|b| tape.var(b.shallow_clone())),
        }
    }
    fn flatten(&self) -> Vec<Array<T>> {
        let mut v = vec![self.w.shallow_clone()];
        if let Some(b) = &self.b {
            v.push(b.shallow_clone());
        }
        v
    }
    fn unflatten(&self, leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError> {
        let w = leaves
            .next()
            .ok_or_else(|| bad("unflatten: leaf underrun"))?;
        let b = match &self.b {
            Some(_) => Some(
                leaves
                    .next()
                    .ok_or_else(|| bad("unflatten: leaf underrun"))?,
            ),
            None => None,
        };
        Ok(LinearParams { w, b })
    }
    fn grads(vars: &LinearVars<T>, loss: &Var<T>) -> Result<Self, AutogradError> {
        Ok(LinearParams {
            w: loss.grad(&vars.w)?,
            b: match &vars.b {
                Some(b) => Some(loss.grad(b)?),
                None => None,
            },
        })
    }
}

impl<T: DiffScalar + ToF64> Layer<T> for Linear {
    type Params = LinearParams<T>;

    fn in_dim(&self) -> Option<usize> {
        Some(self.in_dim)
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.out_dim
    }

    fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError> {
        let bound = (6.0 / self.in_dim as f32).sqrt(); // kaiming-uniform
        let (kw, kb) = key.split();
        let w_host: Vec<T> = kw
            .uniform(self.in_dim * self.out_dim, -bound, bound)
            .iter()
            .map(|&v| T::from_f64(v as f64))
            .collect();
        let w = Array::from_slice(gpu, &w_host, &[self.in_dim, self.out_dim])
            .map_err(AutogradError::from)?;
        let b = if self.bias {
            let _ = kb; // key consumed even when zero-init (linearity discipline)
            let zeros: Vec<T> = (0..self.out_dim).map(|_| T::from_f64(0.0)).collect();
            Some(Array::from_slice(gpu, &zeros, &[self.out_dim]).map_err(AutogradError::from)?)
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
        let y = x.matmul(&p.w)?;
        match &p.b {
            Some(b) => {
                let n = y.value().shape()[0];
                let b2 = b.reshape(&[1, self.out_dim])?;
                let _ = n;
                y.add(&b2)
            }
            None => Ok(y),
        }
    }
}

/// LayerNorm as a layer (fused kernels underneath — T9210's backward).
pub struct LayerNorm {
    pub dim: usize,
    pub eps: f32,
}

pub struct NormParams<T: DiffScalar> {
    pub gamma: Array<T>,
    pub beta: Option<Array<T>>,
}

pub struct NormVars<T: DiffScalar> {
    pub gamma: Var<T>,
    pub beta: Option<Var<T>>,
}

impl<T: DiffScalar> ParamTree<T> for NormParams<T> {
    type Vars = NormVars<T>;

    fn bind(&self, tape: &Tape<T>) -> NormVars<T> {
        NormVars {
            gamma: tape.var(self.gamma.shallow_clone()),
            beta: self.beta.as_ref().map(|b| tape.var(b.shallow_clone())),
        }
    }
    fn flatten(&self) -> Vec<Array<T>> {
        let mut v = vec![self.gamma.shallow_clone()];
        if let Some(b) = &self.beta {
            v.push(b.shallow_clone());
        }
        v
    }
    fn unflatten(&self, leaves: &mut std::vec::IntoIter<Array<T>>) -> Result<Self, AutogradError> {
        let gamma = leaves
            .next()
            .ok_or_else(|| bad("unflatten: leaf underrun"))?;
        let beta = match &self.beta {
            Some(_) => Some(
                leaves
                    .next()
                    .ok_or_else(|| bad("unflatten: leaf underrun"))?,
            ),
            None => None,
        };
        Ok(NormParams { gamma, beta })
    }
    fn grads(vars: &NormVars<T>, loss: &Var<T>) -> Result<Self, AutogradError> {
        Ok(NormParams {
            gamma: loss.grad(&vars.gamma)?,
            beta: match &vars.beta {
                Some(b) => Some(loss.grad(b)?),
                None => None,
            },
        })
    }
}

fn ones_zeros<T: DiffScalar>(
    gpu: &Gpu,
    dim: usize,
    with_beta: bool,
) -> Result<NormParams<T>, AutogradError> {
    let ones: Vec<T> = (0..dim).map(|_| T::from_f64(1.0)).collect();
    let gamma = Array::from_slice(gpu, &ones, &[dim]).map_err(AutogradError::from)?;
    let beta = if with_beta {
        let zeros: Vec<T> = (0..dim).map(|_| T::from_f64(0.0)).collect();
        Some(Array::from_slice(gpu, &zeros, &[dim]).map_err(AutogradError::from)?)
    } else {
        None
    };
    Ok(NormParams { gamma, beta })
}

impl<T: DiffScalar + ToF64> Layer<T> for LayerNorm {
    type Params = NormParams<T>;

    fn in_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.dim
    }
    fn init(&self, gpu: &Gpu, _key: Key) -> Result<Self::Params, AutogradError> {
        ones_zeros(gpu, self.dim, true)
    }
    fn apply(&self, tape: &Tape<T>, p: &NormVars<T>, x: &Var<T>) -> Result<Var<T>, AutogradError> {
        let beta = p
            .beta
            .as_ref()
            .ok_or_else(|| bad("LayerNorm: beta missing"))?;
        crate::norm::layer_norm_var(tape, x, &p.gamma, beta, self.eps)
    }
}

/// RMSNorm as a layer (T9211's backward; no shift).
pub struct RmsNorm {
    pub dim: usize,
    pub eps: f32,
}

impl<T: DiffScalar + ToF64> Layer<T> for RmsNorm {
    type Params = NormParams<T>;

    fn in_dim(&self) -> Option<usize> {
        Some(self.dim)
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.dim
    }
    fn init(&self, gpu: &Gpu, _key: Key) -> Result<Self::Params, AutogradError> {
        ones_zeros(gpu, self.dim, false)
    }
    fn apply(&self, tape: &Tape<T>, p: &NormVars<T>, x: &Var<T>) -> Result<Var<T>, AutogradError> {
        crate::norm::rms_norm_var(tape, x, &p.gamma, self.eps)
    }
}

// ── Tuple stacking (D3) ──────────────────────────────────────────────────

macro_rules! impl_tuple_layer {
    ($($L:ident $P:ident $idx:tt),+) => {
        impl<T: DiffScalar, $($P: ParamTree<T>),+> ParamTree<T> for ($($P,)+) {
            type Vars = ($($P::Vars,)+);

            fn bind(&self, tape: &Tape<T>) -> Self::Vars {
                ($(self.$idx.bind(tape),)+)
            }
            fn flatten(&self) -> Vec<Array<T>> {
                let mut v = Vec::new();
                $(v.extend(self.$idx.flatten());)+
                v
            }
            fn unflatten(
                &self,
                leaves: &mut std::vec::IntoIter<Array<T>>,
            ) -> Result<Self, AutogradError> {
                Ok(($(self.$idx.unflatten(leaves)?,)+))
            }
            fn grads(vars: &Self::Vars, loss: &Var<T>) -> Result<Self, AutogradError> {
                Ok(($($P::grads(&vars.$idx, loss)?,)+))
            }
        }

        impl<T: DiffScalar + ToF64, $($L: Layer<T>),+> Layer<T> for ($($L,)+) {
            type Params = ($($L::Params,)+);

            fn in_dim(&self) -> Option<usize> {
                self.0.in_dim()
            }
            fn out_dim(&self, in_dim: usize) -> usize {
                let d = in_dim;
                $(let d = self.$idx.out_dim(d);)+
                d
            }
            fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError> {
                // The build-time dimension contract (D3): walk the chain once,
                // fail HERE — at model construction — on any width mismatch.
                let mut width: Option<usize> = self.0.in_dim();
                $(
                    if let (Some(w), Some(need)) = (width, self.$idx.in_dim()) {
                        if w != need {
                            return Err(bad(
                                "layer stack: width contract violated at composition",
                            ));
                        }
                    }
                    width = width.or(self.$idx.in_dim());
                    if let Some(w) = width {
                        width = Some(self.$idx.out_dim(w));
                    }
                )+
                let _ = width;
                let mut k = key;
                let params = ($(
                    {
                        let (kl, rest) = k.split();
                        k = rest;
                        self.$idx.init(gpu, kl)?
                    },
                )+);
                let _ = k;
                Ok(params)
            }
            fn apply(
                &self,
                tape: &Tape<T>,
                params: &<($($L::Params,)+) as ParamTree<T>>::Vars,
                x: &Var<T>,
            ) -> Result<Var<T>, AutogradError> {
                let y = x.clone();
                $(let y = self.$idx.apply(tape, &params.$idx, &y)?;)+
                Ok(y)
            }
        }
    };
}

impl_tuple_layer!(L0 P0 0, L1 P1 1);
impl_tuple_layer!(L0 P0 0, L1 P1 1, L2 P2 2);
impl_tuple_layer!(L0 P0 0, L1 P1 1, L2 P2 2, L3 P3 3);
impl_tuple_layer!(L0 P0 0, L1 P1 1, L2 P2 2, L3 P3 3, L4 P4 4);
impl_tuple_layer!(L0 P0 0, L1 P1 1, L2 P2 2, L3 P3 3, L4 P4 4, L5 P5 5);

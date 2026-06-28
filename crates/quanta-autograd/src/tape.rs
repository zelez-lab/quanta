//! Tape-based reverse-mode autodiff.
//!
//! A [`Tape`] records forward ops as they run (define-by-run). Each [`Var`] is
//! a handle `(tape, node id)`; the node's forward value lives in the tape. The
//! forward pass runs the real `quanta-array` kernels now and appends a [`Node`]
//! describing how to backprop. [`Tape::backward`] seeds the output gradient and
//! walks the nodes in reverse, applying each op's VJP (see [`crate::vjp`]) and
//! accumulating into input gradients.
//!
//! The VJP math is the pure-function layer in `crate::vjp`; this module owns
//! the *graph* (node ids, topological reverse order, gradient accumulation).

use std::cell::RefCell;
use std::rc::Rc;

use quanta_array::{Array, FloatScalar, ReduceScalar};

use crate::error::AutogradError;
use crate::vjp;

/// Which op produced a node — enough to run its VJP in the backward pass. The
/// `usize` fields are node ids of the inputs; arrays captured here are the
/// forward values the VJP needs (an input, or the output `y`).
pub(crate) enum Op<T: FloatScalar + ReduceScalar> {
    /// A leaf (input / parameter). No inputs; gradient just accumulates.
    Leaf,
    Neg(usize),
    Add(usize, usize),
    Sub(usize, usize),
    /// mul(a, b): needs both forward values for the cross multipliers.
    Mul(usize, usize, Array<T>, Array<T>),
    Div(usize, usize, Array<T>, Array<T>),
    /// exp(x): captures the output y (= exp x), since ∂/∂x = g·y.
    Exp(usize, Array<T>),
    /// log(x): captures the input x, since ∂/∂x = g/x.
    Log(usize, Array<T>),
    /// sqrt(x): captures the output y (= √x), since ∂/∂x = g/(2y).
    Sqrt(usize, Array<T>),
    /// sum over all elements → scalar. Captures the input shape so backward can
    /// broadcast the scalar grad back to a full ones·g array.
    Sum(usize, Vec<usize>),
}

pub(crate) struct Node<T: FloatScalar + ReduceScalar> {
    pub(crate) op: Op<T>,
}

/// The shared recording buffer behind every [`Var`].
pub(crate) struct TapeInner<T: FloatScalar + ReduceScalar> {
    pub(crate) nodes: Vec<Node<T>>,
    pub(crate) values: Vec<Array<T>>,
}

/// A reverse-mode autodiff tape. Build leaves with [`Tape::var`], run ops on the
/// resulting [`Var`]s, then [`Tape::backward`] and read [`Var::grad`].
pub struct Tape<T: FloatScalar + ReduceScalar> {
    inner: Rc<RefCell<TapeInner<T>>>,
}

impl<T: FloatScalar + ReduceScalar> Default for Tape<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatScalar + ReduceScalar> Tape<T> {
    /// A fresh, empty tape.
    pub fn new() -> Self {
        Tape {
            inner: Rc::new(RefCell::new(TapeInner {
                nodes: Vec::new(),
                values: Vec::new(),
            })),
        }
    }

    /// Wrap a shared inner buffer in a `Tape` handle (used by the op layer to
    /// push nodes onto an existing `Var`'s tape).
    pub(crate) fn from_inner(inner: Rc<RefCell<TapeInner<T>>>) -> Self {
        Tape { inner }
    }

    /// Record a leaf (an input or parameter we want gradients for).
    pub fn var(&self, value: Array<T>) -> Var<T> {
        self.push(Op::Leaf, value)
    }

    /// Append a node + its forward value, returning the new `Var`.
    pub(crate) fn push(&self, op: Op<T>, value: Array<T>) -> Var<T> {
        let mut inner = self.inner.borrow_mut();
        let id = inner.nodes.len();
        inner.nodes.push(Node { op });
        inner.values.push(value);
        Var {
            tape: Rc::clone(&self.inner),
            id,
        }
    }
}

/// A node in a [`Tape`] — a differentiable value handle.
pub struct Var<T: FloatScalar + ReduceScalar> {
    pub(crate) tape: Rc<RefCell<TapeInner<T>>>,
    pub(crate) id: usize,
}

impl<T: FloatScalar + ReduceScalar> Var<T> {
    /// The forward value at this node (cheap Arc-share view).
    pub fn value(&self) -> Array<T> {
        self.tape.borrow().values[self.id].shallow_clone()
    }

    /// An owned device handle (`Gpu` is a cheap `Arc` clone).
    fn gpu_owned(&self) -> quanta::Gpu {
        self.tape.borrow().values[self.id].gpu().clone()
    }

    /// Run reverse-mode backprop from this node as the scalar output, then
    /// return the gradient accumulated at `wrt`. `wrt` must be on the same
    /// tape. The seed gradient is ones-shaped like this node's value.
    pub fn grad(&self, wrt: &Var<T>) -> Result<Array<T>, AutogradError> {
        if !Rc::ptr_eq(&self.tape, &wrt.tape) {
            return Err(AutogradError::ForeignVar);
        }
        let grads = self.backward()?;
        match &grads[wrt.id] {
            Some(g) => Ok(g.shallow_clone()),
            None => {
                // No path from output to `wrt`: gradient is zero.
                let shape = self.tape.borrow().values[wrt.id].shape().to_vec();
                Ok(Array::zeros(&self.gpu_owned(), &shape)?)
            }
        }
    }

    /// Reverse sweep: seed `grad[self] = ones`, then for each node in reverse
    /// order push its gradient to its inputs via the op's VJP. Returns the
    /// per-node accumulated gradients (`None` = untouched = zero).
    fn backward(&self) -> Result<Vec<Option<Array<T>>>, AutogradError> {
        let gpu = self.gpu_owned();
        let inner = self.tape.borrow();
        let n = inner.nodes.len();
        let mut grads: Vec<Option<Array<T>>> = (0..n).map(|_| None).collect();

        // Seed: ones shaped like the output value.
        let out_shape = inner.values[self.id].shape().to_vec();
        grads[self.id] = Some(Array::ones(&gpu, &out_shape)?);

        // Accumulate g into node id (add if already present).
        fn accum<U: FloatScalar + ReduceScalar>(
            slot: &mut Option<Array<U>>,
            g: Array<U>,
        ) -> Result<(), AutogradError> {
            *slot = Some(match slot.take() {
                Some(existing) => existing.add(&g)?,
                None => g,
            });
            Ok(())
        }

        for id in (0..n).rev() {
            let Some(g) = grads[id].as_ref().map(|a| a.shallow_clone()) else {
                continue; // no gradient flows here
            };
            match &inner.nodes[id].op {
                Op::Leaf => {}
                Op::Neg(a) => {
                    let ga = vjp::neg(&g)?;
                    accum(&mut grads[*a], ga)?;
                }
                Op::Add(a, b) => {
                    let (ga, gb) = vjp::add(&g)?;
                    accum(&mut grads[*a], ga)?;
                    accum(&mut grads[*b], gb)?;
                }
                Op::Sub(a, b) => {
                    let (ga, gb) = vjp::sub(&g)?;
                    accum(&mut grads[*a], ga)?;
                    accum(&mut grads[*b], gb)?;
                }
                Op::Mul(a, b, av, bv) => {
                    let (ga, gb) = vjp::mul(&g, av, bv)?;
                    accum(&mut grads[*a], ga)?;
                    accum(&mut grads[*b], gb)?;
                }
                Op::Div(a, b, av, bv) => {
                    let (ga, gb) = vjp::div(&g, av, bv)?;
                    accum(&mut grads[*a], ga)?;
                    accum(&mut grads[*b], gb)?;
                }
                Op::Exp(a, y) => accum(&mut grads[*a], vjp::exp(&g, y)?)?,
                Op::Log(a, x) => accum(&mut grads[*a], vjp::log(&g, x)?)?,
                Op::Sqrt(a, y) => accum(&mut grads[*a], vjp::sqrt(&g, y)?)?,
                Op::Sum(a, in_shape) => {
                    // sum → scalar: ∂(Σx)/∂xᵢ = 1, so grad broadcasts the scalar
                    // g to every input element: g_in = ones(in_shape) · g_scalar.
                    let gval = g.to_vec()?[0];
                    let gin = Array::full(&gpu, gval, in_shape)?;
                    accum(&mut grads[*a], gin)?;
                }
            }
        }
        Ok(grads)
    }
}

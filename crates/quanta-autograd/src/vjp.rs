//! Pure VJP (vector-Jacobian product) rules, factored as standalone functions
//! over `Array<T>` so both the tape backward pass and a future graph/fusion
//! layer can reuse them.
//!
//! Each rule takes the upstream gradient `g = ∂L/∂y` plus whatever forward
//! values it needs, and returns the input gradient(s) `∂L/∂xᵢ = g · ∂y/∂xᵢ`.
//! The per-element multipliers are exactly the analytic derivatives proven in
//! `specs/verify/lean/Quanta/Autograd/Vjp.lean`:
//!
//!   neg → -1 · g        add → (g, g)         sub → (g, -g)
//!   mul → (g·b, g·a)     div → (g/b, -g·a/b²)
//!   exp → g·exp(x)=g·y   log → g/x            sqrt → g/(2√x)=g/(2y)
//!
//! These are the contiguous, same-shape rules. Broadcast/reduction
//! axis-summing is handled by the tape op layer before/after calling these.

use quanta_array::{Array, ArrayError, FloatScalar, ReduceScalar};

type R<T> = Result<Array<T>, ArrayError>;

/// **Un-broadcast** a gradient `g` back to `target` shape. When a forward op
/// broadcast an input of shape `target` up to `g`'s shape, the reverse step
/// sums `g` over the broadcast axes: leading axes the target lacks are summed
/// away, and axes where the target had extent 1 (but `g` has > 1) are summed
/// with keepdims. The result has exactly `target` shape, ready to accumulate
/// onto that input's gradient. (A no-op when shapes already match.)
pub fn unbroadcast<T: FloatScalar + ReduceScalar>(g: &Array<T>, target: &[usize]) -> R<T> {
    if g.shape() == target {
        return Ok(g.shallow_clone());
    }
    let mut cur = g.shallow_clone();
    // 1. Drop leading axes the target doesn't have (sum over axis 0, squeeze).
    while cur.shape().len() > target.len() {
        let summed = cur.sum_axis(0)?; // [1, rest…]
        let new_shape: Vec<usize> = cur.shape()[1..].to_vec();
        cur = summed.reshape(&new_shape)?;
    }
    // 2. Sum (keepdims) the axes where target == 1 but cur > 1.
    for (i, &t) in target.iter().enumerate() {
        if t == 1 && cur.shape()[i] != 1 {
            cur = cur.sum_axis(i)?;
        }
    }
    // Defensive: land exactly on `target` (handles any residual unit dims).
    if cur.shape() != target {
        cur = cur.reshape(target)?;
    }
    Ok(cur)
}

/// `neg`: y = -x ⇒ ∂L/∂x = -g.
pub fn neg<T: FloatScalar + ReduceScalar>(g: &Array<T>) -> R<T> {
    g.neg()
}

/// `add`: y = a + b ⇒ (∂L/∂a, ∂L/∂b) = (g, g), each un-broadcast to its
/// operand's original shape.
pub fn add<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    sa: &[usize],
    sb: &[usize],
) -> Result<(Array<T>, Array<T>), ArrayError> {
    Ok((unbroadcast(g, sa)?, unbroadcast(g, sb)?))
}

/// `sub`: y = a - b ⇒ (g, -g), each un-broadcast to its operand's shape.
pub fn sub<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    sa: &[usize],
    sb: &[usize],
) -> Result<(Array<T>, Array<T>), ArrayError> {
    Ok((unbroadcast(g, sa)?, unbroadcast(&g.neg()?, sb)?))
}

/// `mul`: y = a·b ⇒ (g·b, g·a), each un-broadcast to the operand's shape (the
/// products are at the output/broadcast shape).
pub fn mul<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    let ga = unbroadcast(&g.mul(b)?, a.shape())?;
    let gb = unbroadcast(&g.mul(a)?, b.shape())?;
    Ok((ga, gb))
}

/// `div`: y = a/b ⇒ (g/b, -g·a/b²), each un-broadcast to the operand's shape.
pub fn div<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    let ga = unbroadcast(&g.div(b)?, a.shape())?;
    // ∂/∂b = -g·a/b² = -(g·a) / (b·b)
    let num = g.mul(a)?;
    let b2 = b.mul(b)?;
    let gb = unbroadcast(&num.div(&b2)?.neg()?, b.shape())?;
    Ok((ga, gb))
}

/// `exp`: y = exp(x) ⇒ ∂L/∂x = g·y (reuse the forward output `y`).
pub fn exp<T: FloatScalar + ReduceScalar>(g: &Array<T>, y: &Array<T>) -> R<T> {
    g.mul(y)
}

/// `log`: y = log(x) ⇒ ∂L/∂x = g/x.
pub fn log<T: FloatScalar + ReduceScalar>(g: &Array<T>, x: &Array<T>) -> R<T> {
    g.div(x)
}

/// `sqrt`: y = √x ⇒ ∂L/∂x = g/(2√x) = g/(2y) (reuse the forward output `y`).
pub fn sqrt<T: FloatScalar + ReduceScalar>(g: &Array<T>, y: &Array<T>) -> R<T> {
    let two_y = y.add(y)?; // 2y
    g.div(&two_y)
}

/// `relu`: y = max(x, 0) ⇒ ∂L/∂x = g · [x > 0] (the positive-step mask). The
/// subgradient at 0 is taken as 0.
pub fn relu<T: FloatScalar + ReduceScalar>(g: &Array<T>, x: &Array<T>) -> R<T> {
    g.mul(&x.step_positive()?)
}

/// `sigmoid`: y = σ(x) ⇒ ∂L/∂x = g · y · (1 − y) (reuse the forward output `y`).
pub fn sigmoid<T: FloatScalar + ReduceScalar>(g: &Array<T>, y: &Array<T>) -> R<T> {
    let one = Array::full(y.gpu(), T::ONE, &[1])?.broadcast_to(y.shape())?;
    let one_minus_y = one.sub(y)?;
    g.mul(y)?.mul(&one_minus_y)
}

/// `tanh`: y = tanh(x) ⇒ ∂L/∂x = g · (1 − y²) (reuse the forward output `y`).
pub fn tanh<T: FloatScalar + ReduceScalar>(g: &Array<T>, y: &Array<T>) -> R<T> {
    let one = Array::full(y.gpu(), T::ONE, &[1])?.broadcast_to(y.shape())?;
    let one_minus_y2 = one.sub(&y.mul(y)?)?;
    g.mul(&one_minus_y2)
}

/// `matmul`: Y = A·B (A is m×k, B is k×n) ⇒
///   ∂L/∂A = G·Bᵀ   (m×n · n×k → m×k)
///   ∂L/∂B = Aᵀ·G   (k×m · m×n → k×n)
/// where G = ∂L/∂Y. Both VJPs are themselves matmuls (reusing the proven
/// quanta-blas gemm); the transposes are zero-copy views materialized by
/// matmul's contiguous-gather.
pub fn matmul<T: crate::scalar::DiffScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    let bt = b.transpose(0, 1)?;
    let at = a.transpose(0, 1)?;
    let ga = T::array_matmul(g, &bt)?; // G·Bᵀ
    let gb = T::array_matmul(&at, g)?; // Aᵀ·G
    Ok((ga, gb))
}

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

/// `neg`: y = -x ⇒ ∂L/∂x = -g.
pub fn neg<T: FloatScalar + ReduceScalar>(g: &Array<T>) -> R<T> {
    g.neg()
}

/// `add`: y = a + b ⇒ (∂L/∂a, ∂L/∂b) = (g, g).
pub fn add<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    Ok((g.shallow_clone(), g.shallow_clone()))
}

/// `sub`: y = a - b ⇒ (g, -g).
pub fn sub<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    Ok((g.shallow_clone(), g.neg()?))
}

/// `mul`: y = a·b ⇒ (g·b, g·a).
pub fn mul<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    Ok((g.mul(b)?, g.mul(a)?))
}

/// `div`: y = a/b ⇒ (g/b, -g·a/b²).
pub fn div<T: FloatScalar + ReduceScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    let ga = g.div(b)?;
    // ∂/∂b = -g·a/b² = -(g·a) / (b·b)
    let num = g.mul(a)?;
    let b2 = b.mul(b)?;
    let gb = num.div(&b2)?.neg()?;
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

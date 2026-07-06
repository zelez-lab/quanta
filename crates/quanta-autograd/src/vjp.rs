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

/// `matmul`: Y = A·B (A is …m×k, B is …k×n) ⇒
///   ∂L/∂A = G·Bᵀ   (…m×n · …n×k → …m×k)
///   ∂L/∂B = Aᵀ·G   (…k×m · …m×n → …k×n)
/// where G = ∂L/∂Y. Both VJPs are themselves matmuls (reusing the proven
/// quanta-blas gemm), applied per batch; the transpose is over the last two
/// axes.
///
/// **Broadcasting**: when the forward matmul broadcast the operands' batch
/// dims, `G·Bᵀ` / `Aᵀ·G` come out at the *broadcast* batch shape, so each
/// gradient is summed back down to its operand's original shape (the standard
/// broadcast-VJP reduction). Same-shape (incl. plain 2-D) is a no-op.
pub fn matmul<T: crate::scalar::DiffScalar>(
    g: &Array<T>,
    a: &Array<T>,
    b: &Array<T>,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    let ra = a.rank();
    let rb = b.rank();
    // Transpose the trailing two axes (2-D → (0,1); N-D → (r-2, r-1)).
    let bt = b.transpose(rb - 2, rb - 1)?;
    let at = a.transpose(ra - 2, ra - 1)?;
    let ga_full = T::array_matmul(g, &bt)?; // G·Bᵀ at the broadcast batch shape
    let gb_full = T::array_matmul(&at, g)?; // Aᵀ·G at the broadcast batch shape
    let ga = reduce_to_shape(&ga_full, a.shape())?;
    let gb = reduce_to_shape(&gb_full, b.shape())?;
    Ok((ga, gb))
}

/// Sum `grad` (at a broadcast batch shape `[…broadcastBatch, r, c]`) back down
/// to `target` (the operand's own shape `[…operandBatch, r, c]`): sum over every
/// leading batch axis the operand lacks, and over every target batch axis whose
/// operand extent is 1 while grad's is larger, then reshape to `target`. A
/// no-op when the shapes already match (plain 2-D, or matching batch dims).
fn reduce_to_shape<T: crate::scalar::DiffScalar>(
    grad: &Array<T>,
    target: &[usize],
) -> Result<Array<T>, ArrayError> {
    if grad.shape() == target {
        return Ok(grad.shallow_clone());
    }
    let extra = grad.rank() - target.len(); // leading axes absent on the operand
    let mut cur = grad.shallow_clone();
    // Leading axes the operand doesn't have: sum them away (keepdims → size 1).
    for axis in 0..extra {
        if cur.shape()[axis] != 1 {
            cur = cur.sum_axis(axis)?;
        }
    }
    // Target batch axes reduced where the operand extent is 1 but grad's larger.
    let tb = target.len().saturating_sub(2);
    for (j, &tdim) in target[..tb].iter().enumerate() {
        let axis = extra + j;
        if tdim == 1 && cur.shape()[axis] != 1 {
            cur = cur.sum_axis(axis)?;
        }
    }
    // Element count now equals target's (reduced dims are size 1); reshape.
    cur.contiguous()?.reshape(target)
}

/// `conv2d`: Y = conv(X, W) via cols·wm. With G = ∂L/∂Y at [N,Cout,OH,OW]:
/// reshape G to Gm[N·OH·OW, Cout], then this is exactly a matmul backward over
/// (cols, wm):
///   ∂cols = Gm·wmᵀ   → col2im → ∂X[N,Cin,H,W]
///   ∂wm   = colsᵀ·Gm → reshape → ∂W[Cout,Cin,kh,kw]
/// reusing the proven matmul VJP and the im2col/col2im adjoint pair.
pub fn conv2d<T: crate::scalar::DiffScalar>(
    g: &Array<T>,
    cols: &Array<T>,
    wm: &Array<T>,
    p: &crate::conv::ConvParams,
) -> Result<(Array<T>, Array<T>), ArrayError> {
    // G[N,Cout,OH,OW] → [N,OH,OW,Cout] → Gm[N·OH·OW, Cout].
    let gm = g
        .permute(&[0, 2, 3, 1])?
        .contiguous()?
        .reshape(&[p.n * p.oh * p.ow, p.cout])?;
    // Matmul backward over (cols, wm): ∂cols = Gm·wmᵀ, ∂wm = colsᵀ·Gm.
    let (dcols, dwm) = matmul(&gm, cols, wm)?;
    // ∂X = col2im(∂cols).
    let dx = dcols.col2im(p.n, p.cin, p.h, p.w, p.kh, p.kw, p.stride, p.pad)?;
    // ∂W: wm = [kdim, Cout] flattened from [Cout,kdim]ᵀ, so ∂wm[kdim,Cout]ᵀ →
    // [Cout,kdim] → [Cout,Cin,kh,kw].
    let dw = dwm
        .transpose(0, 1)?
        .contiguous()?
        .reshape(&[p.cout, p.cin, p.kh, p.kw])?;
    Ok((dx, dw))
}

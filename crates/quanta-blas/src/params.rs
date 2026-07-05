//! Standard BLAS parameter enums shared by the GPU ops and the pure-Rust
//! reference oracles: which triangle of a matrix is stored ([`Uplo`]),
//! whether the op applies the matrix or its transpose ([`Trans`]), whether
//! the triangular diagonal is implicitly 1 ([`Diag`]), and which side a
//! triangular factor multiplies from ([`Side`]).
//!
//! For real `f32` matrices the BLAS conjugate-transpose option is identical
//! to [`Trans::Trans`], so it is not a separate variant.

/// Which triangle of a triangular/symmetric matrix is referenced. The
/// opposite triangle (and, for [`Diag::Unit`], the diagonal) is never read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uplo {
    /// The lower triangle (`i ≥ j`).
    Lower,
    /// The upper triangle (`i ≤ j`).
    Upper,
}

/// Whether an op applies the matrix as stored or its transpose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trans {
    /// Use `A` as stored.
    NoTrans,
    /// Use `Aᵀ`.
    Trans,
}

/// Whether a triangular matrix has an implicit unit diagonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diag {
    /// The diagonal is stored and used.
    NonUnit,
    /// The diagonal is implicitly 1 — the stored diagonal is never read.
    Unit,
}

/// Which side the triangular factor multiplies from in `trsm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// Solve `A·X = α·B` (A is `m×m`).
    Left,
    /// Solve `X·A = α·B` (A is `n×n`).
    Right,
}

/// The substitution plan for a `trsm`/`trsv` variant: `(rs, cs, forward)`.
///
/// Every `side`/`uplo`/`transA` combination reduces to a plain forward or
/// backward substitution over an *effective* triangular matrix `M` accessed
/// through strides into `A`'s row-major storage: `M[i,p] = a[i·rs + p·cs]`
/// (so `M[i,i] = a[i·(rs+cs)]`).
///
/// - `side = Left` solves `op(A)·x = α·b` per RHS column, so `M = op(A)`:
///   `NoTrans` keeps row-major strides `(na, 1)`, `Trans` swaps them.
/// - `side = Right` solves `x·op(A) = α·b` per RHS row, which transposed is
///   `op(A)ᵀ·xᵀ = α·bᵀ`, so `M = op(A)ᵀ` — the stride mapping flips.
///
/// The solve runs **forward** (row 0 first) exactly when `M` is lower
/// triangular; `uplo` says which triangle of `A` is populated, and each
/// transpose flips it.
pub(crate) fn trsm_plan(side: Side, uplo: Uplo, trans: Trans, na: usize) -> (usize, usize, bool) {
    let (rs, cs) = match (side, trans) {
        (Side::Left, Trans::NoTrans) => (na, 1),
        (Side::Left, Trans::Trans) => (1, na),
        (Side::Right, Trans::NoTrans) => (1, na),
        (Side::Right, Trans::Trans) => (na, 1),
    };
    let forward = matches!(
        (side, uplo, trans),
        (Side::Left, Uplo::Lower, Trans::NoTrans)
            | (Side::Left, Uplo::Upper, Trans::Trans)
            | (Side::Right, Uplo::Upper, Trans::NoTrans)
            | (Side::Right, Uplo::Lower, Trans::Trans)
    );
    (rs, cs, forward)
}

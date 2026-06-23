//! Linear algebra on `Array<f32>` — `matmul`, `dot`, `norm`.
//!
//! These delegate to `quanta-blas` (the verified Level-1 / GEMM ops), so the
//! numerical contract is the Higham-style forward-error bound proven in
//! `specs/verify/lean/Quanta/Blas/`. quanta-array sits *above* quanta-blas in
//! the stack: it owns the shape/layout, materializes contiguous operands, and
//! calls *down* into blas — blas never depends on the array surface.
//!
//! f32-only for this increment (blas Level-1 + GEMM are f32). The functional
//! API returns a fresh `Array`; the blas ops underneath are device-resident,
//! so no host round-trip happens for the math itself.

use quanta_blas::{dot as blas_dot, gemm as blas_gemm, nrm2 as blas_nrm2};

use crate::array::Array;
use crate::error::ArrayError;

impl Array<f32> {
    /// Matrix multiply: `self (m×k) · rhs (k×n) → (m×n)`. Both operands must
    /// be 2-D with a matching inner dimension. Returns a fresh contiguous
    /// row-major `Array`. (numpy `a @ b` for 2-D arrays.)
    pub fn matmul(&self, rhs: &Array<f32>) -> Result<Array<f32>, ArrayError> {
        if self.rank() != 2 || rhs.rank() != 2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "matmul: both operands must be 2-D",
            )));
        }
        let m = self.shape()[0];
        let k = self.shape()[1];
        let k2 = rhs.shape()[0];
        let n = rhs.shape()[1];
        if k != k2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "matmul: inner dimensions disagree (A is m×k, B must be k×n)",
            )));
        }

        // Materialize both operands contiguous row-major (on-device gather
        // for strided/transposed views), then run C ← 1·A·B + 0·C.
        let a = self.contiguous_or_self()?;
        let b = rhs.contiguous_or_self()?;
        let c = Array::<f32>::zeros(self.gpu(), &[m, n])?;

        blas_gemm(
            self.gpu(),
            m as u32,
            n as u32,
            k as u32,
            1.0,
            a.field_ref(),
            b.field_ref(),
            0.0,
            c.field_ref(),
        )?;
        Ok(c)
    }

    /// Inner product of two 1-D arrays of equal length (numpy `np.dot` /
    /// `a @ b` for vectors). Device-resident reduction.
    pub fn dot(&self, rhs: &Array<f32>) -> Result<f32, ArrayError> {
        if self.rank() != 1 || rhs.rank() != 1 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "dot: both operands must be 1-D",
            )));
        }
        if self.len() != rhs.len() {
            return Err(ArrayError::LengthMismatch {
                expected: self.len(),
                got: rhs.len(),
            });
        }
        let a = self.contiguous_or_self()?;
        let b = rhs.contiguous_or_self()?;
        Ok(blas_dot(self.gpu(), a.field_ref(), b.field_ref())?)
    }

    /// Euclidean (L2) norm of all elements, `√(Σ xᵢ²)` (numpy
    /// `np.linalg.norm`). Flattens any shape; device-resident reduction.
    pub fn norm(&self) -> Result<f32, ArrayError> {
        let a = self.contiguous_or_self()?;
        Ok(blas_nrm2(self.gpu(), a.field_ref())?)
    }
}

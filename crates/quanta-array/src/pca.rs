//! Principal component analysis — scikit-learn `PCA` parity.
//!
//! `pca(k)` composes existing array ops: center the data (`sum_axis` +
//! broadcast `sub`), form the sample covariance `C = Xcᵀ·Xc / (N−1)` (GPU
//! `matmul`), and diagonalize it with [`Array::eigh_symmetric`]. No new
//! kernels — the eigensolver is Jacobi over GPU matmuls.

use crate::array::Array;
use crate::error::ArrayError;

impl Array<f32> {
    /// PCA of row-major data `[N, D]` (N samples, D features): returns
    /// `(components [k, D], explained_variance [k])` — the top-`k` principal
    /// directions as **rows**, sorted by decreasing variance, and the
    /// variance each captures (scikit-learn `PCA.components_` /
    /// `.explained_variance_`).
    ///
    /// The data is centered first and the covariance uses the sample
    /// normalizer `N−1`, matching scikit-learn. Component signs follow the
    /// [`Array::eigh_symmetric`] convention (largest-magnitude entry
    /// non-negative), so results are deterministic; sklearn's own sign choice
    /// may differ — compare up to sign.
    ///
    /// Requires `N ≥ 2` and `1 ≤ k ≤ D`.
    pub fn pca(&self, k: usize) -> Result<(Array<f32>, Array<f32>), ArrayError> {
        if self.rank() != 2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pca: data must be 2-D [samples, features]",
            )));
        }
        let n = self.shape()[0];
        let d = self.shape()[1];
        if n < 2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pca: need at least 2 samples for a sample covariance (N−1)",
            )));
        }
        if k == 0 || k > d {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pca: k must satisfy 1 ≤ k ≤ D (feature count)",
            )));
        }
        let g = self.gpu();
        let x = self.contiguous_or_self()?;

        // Center: mean [1, D] = Σ_rows / N, broadcast-subtracted.
        let mean = x.sum_axis(0)?.div(&Array::full(g, n as f32, &[1, d])?)?;
        let xc = x.sub(&mean.broadcast_to(&[n, d])?)?;

        // Sample covariance C = Xcᵀ·Xc / (N−1) — symmetric D×D.
        let cov =
            xc.transpose(0, 1)?
                .matmul(&xc)?
                .div(&Array::full(g, (n - 1) as f32, &[d, d])?)?;

        // Top-k eigenpairs (eigh_symmetric sorts descending).
        let (evals, evecs) = cov.eigh_symmetric()?;

        // Components as rows [k, D]: first k eigenvector columns, transposed.
        let components = evecs.narrow(1, 0, k)?.transpose(0, 1)?.contiguous()?;
        let explained = evals.narrow(0, 0, k)?.contiguous()?;
        Ok((components, explained))
    }
}

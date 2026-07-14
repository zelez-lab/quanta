//! Pairwise distances — the k-means / nearest-neighbour primitive.
//!
//! Composed from broadcast + sub + square + sum, so it needs no new kernel:
//! `points [N, D]` and `centers [K, D]` broadcast against each other as
//! `[N, 1, D]` and `[1, K, D]`, and the squared difference is summed over the
//! feature axis. Materializes an `[N, K, D]` intermediate — fine for the
//! moderate sizes k-means runs at; a fused kernel is a later optimization.

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Squared Euclidean distance between every row of `self` `[N, D]` and
    /// every row of `centers` `[K, D]`, giving `[N, K]` where
    /// `out[n, k] = Σ_d (self[n, d] − centers[k, d])²`. The k-means assignment
    /// metric (pair with `argmin_last` for nearest-centroid).
    pub fn cdist_sq(&self, centers: &Array<T>) -> Result<Array<T>, ArrayError> {
        let a = self.shape();
        let b = centers.shape();
        if a.len() != 2 || b.len() != 2 || a[1] != b[1] {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "cdist_sq: both inputs must be 2-D [., D] with matching D",
            )));
        }
        let (n, d, k) = (a[0], a[1], b[0]);
        let pv = self.reshape(&[n, 1, d])?.broadcast_to(&[n, k, d])?;
        let cv = centers.reshape(&[1, k, d])?.broadcast_to(&[n, k, d])?;
        let diff = pv.sub(&cv)?;
        let sq = diff.mul(&diff)?;
        // sum over the feature axis → [n, k, 1], squeeze to [n, k].
        sq.sum_axis(2)?.reshape(&[n, k])
    }
}

//! Per-lane kernel output container.
//!
//! Kept dependency-free: no serde, no JSON. The comparator works
//! directly on these values.
//!
//! `RawValues` is re-exported from `quanta_ir::op_matrix_cases` so the
//! op-matrix case generator (shared with the WGSL browser audit) and the
//! test-side comparator/kernels all speak one type.

use super::lane::Lane;

pub use quanta_ir::op_matrix_cases::RawValues;

#[derive(Clone, Debug)]
pub struct RawOutput {
    pub lane: Lane,
    pub kernel: &'static str,
    pub values: RawValues,
}

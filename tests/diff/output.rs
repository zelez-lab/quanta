//! Per-lane kernel output container.
//!
//! Kept dependency-free: no serde, no JSON. The comparator works
//! directly on these values. Disk-format goldens, if we add them
//! later, will sit beside this and serialize via a hand-rolled
//! encoder.

use super::lane::Lane;

#[derive(Clone, Debug)]
pub struct RawOutput {
    pub lane: Lane,
    pub kernel: &'static str,
    pub values: Vec<f32>,
}

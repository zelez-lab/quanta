//! Per-lane kernel output container.
//!
//! Kept dependency-free: no serde, no JSON. The comparator works
//! directly on these values.

use super::lane::Lane;

#[derive(Clone, Debug)]
pub enum RawValues {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

#[derive(Clone, Debug)]
pub struct RawOutput {
    pub lane: Lane,
    pub kernel: &'static str,
    pub values: RawValues,
}

impl RawValues {
    pub fn type_tag(&self) -> &'static str {
        match self {
            RawValues::F32(_) => "f32",
            RawValues::U32(_) => "u32",
        }
    }
}

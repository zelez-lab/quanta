//! Per-lane kernel output container.
//!
//! Kept dependency-free: no serde, no JSON. The comparator works
//! directly on these values.

use super::lane::Lane;

#[derive(Clone, Debug)]
pub enum RawValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
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
            RawValues::F64(_) => "f64",
            RawValues::U32(_) => "u32",
            RawValues::U64(_) => "u64",
            RawValues::I32(_) => "i32",
            RawValues::I64(_) => "i64",
        }
    }
}

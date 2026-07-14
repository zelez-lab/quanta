//! Differential kernel corpus.
//!
//! Each kernel exposes:
//! - `NAME` and (where applicable) input-size / config constants.
//! - `inputs()` for the deterministic input vectors used by every lane.
//! - `run_reference()` returning a `RawOutput` from a pure-Rust impl.
//! - `run_<lane>()` per backend (gated by feature flag).

pub mod counter;
pub mod mp;
pub mod race;
pub mod reduce_sum;
pub mod saxpy;
pub mod sb;

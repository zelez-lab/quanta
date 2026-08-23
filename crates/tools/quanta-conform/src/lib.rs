//! Multi-backend render conformance: the same draw on every backend,
//! frames compared byte-for-byte against the software reference.
//!
//! The corpus (`cases`) is a list of named draws, each a plain
//! `fn(&Gpu) -> Frame` over the public render API. The runner executes
//! the corpus once per backend — each backend in its own child process,
//! so one driver's state can never leak into another's frame — and
//! compares every backend's frames against the software renderer's,
//! which is the canonical reference.
//!
//! Comparison is per channel with a per-case tolerance, plus a
//! **divergent-pixel budget**: rasterizers legitimately disagree on
//! triangle-edge coverage (tie-breaking rules differ per
//! implementation), so a case with interior primitive edges grants a
//! small per-mille budget of out-of-tolerance pixels. A case with no
//! interior edge runs at budget zero — any structural divergence is a
//! failure. Both knobs are printed in the matrix, so a pass never
//! hides its terms.

#![deny(missing_docs)]

pub mod cases;
pub mod compare;
pub mod frame;
pub mod matrix;

pub use compare::{CaseVerdict, Comparison, Tolerance};
pub use frame::Frame;

/// One conformance draw: a name, its comparison terms, and the draw
/// itself, written against the public API only.
pub struct Case {
    /// Stable name — the row key of the parity matrix, and the frame's
    /// file name under `--dump`.
    pub name: &'static str,
    /// The comparison terms this case runs under.
    pub tolerance: Tolerance,
    /// Renders one frame on the given device.
    pub run: fn(&quanta::Gpu) -> Result<Frame, quanta::QuantaError>,
}

/// The corpus, in matrix row order.
pub fn corpus() -> Vec<Case> {
    cases::all()
}

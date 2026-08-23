//! Frame comparison: per-channel tolerance plus a divergent-pixel budget.

use crate::Frame;

/// The terms a case is compared under. Printed in the matrix — a pass
/// never hides them.
#[derive(Clone, Copy)]
pub struct Tolerance {
    /// Maximum per-channel difference (unorm LSBs) a pixel may show and
    /// still count as matching. `1` is the contract for integer
    /// formats; resolve/filtering cases may need `2`.
    pub channel: u8,
    /// Out-of-tolerance pixels allowed, in per-mille of the frame —
    /// the rasterization-edge allowance. `0` for cases with no
    /// interior primitive edge.
    pub edge_budget_permille: u32,
}

impl Tolerance {
    /// The strict terms: 1 LSB, no divergent pixel at all.
    pub const EXACT: Tolerance = Tolerance {
        channel: 1,
        edge_budget_permille: 0,
    };
    /// 1 LSB with a 5‰ edge allowance — a case whose primitives leave
    /// visible interior edges.
    pub const EDGED: Tolerance = Tolerance {
        channel: 1,
        edge_budget_permille: 5,
    };
}

/// What one backend's frame did against the reference.
pub struct Comparison {
    /// Largest per-channel difference seen anywhere.
    pub max_channel_delta: u8,
    /// Pixels with any channel beyond the tolerance.
    pub divergent_pixels: u32,
    /// Total pixels compared.
    pub total_pixels: u32,
}

/// The verdict for one (case, backend) cell.
pub enum CaseVerdict {
    /// Within tolerance and budget; carries the numbers.
    Pass(Comparison),
    /// Beyond tolerance or budget; carries the numbers.
    Fail(Comparison),
    /// The frames' dimensions disagree — never tolerable.
    ShapeMismatch,
    /// The backend refused the case (`NotSupported`), with the reason.
    Unsupported(String),
    /// The case errored on this backend.
    Error(String),
}

/// Compare a backend frame against the reference under `tol`.
pub fn compare(reference: &Frame, got: &Frame, tol: Tolerance) -> CaseVerdict {
    if reference.width != got.width || reference.height != got.height {
        return CaseVerdict::ShapeMismatch;
    }
    let mut max_delta = 0u8;
    let mut divergent = 0u32;
    for (a, b) in reference
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .zip(got.bytes.as_chunks::<4>().0.iter())
    {
        let mut worst = 0u8;
        for i in 0..4 {
            worst = worst.max(a[i].abs_diff(b[i]));
        }
        max_delta = max_delta.max(worst);
        if worst > tol.channel {
            divergent += 1;
        }
    }
    let total = reference.width * reference.height;
    let budget = (u64::from(total) * u64::from(tol.edge_budget_permille) / 1000) as u32;
    let cmp = Comparison {
        max_channel_delta: max_delta,
        divergent_pixels: divergent,
        total_pixels: total,
    };
    if divergent <= budget {
        CaseVerdict::Pass(cmp)
    } else {
        CaseVerdict::Fail(cmp)
    }
}

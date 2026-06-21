//! Tolerance rules for cross-lane output comparison.
//!
//! - `compare_f32(..., max_ulps)` — float kernels (≤ `max_ulps` ULP).
//! - `compare_u32(...)` — integer kernels (bit-exact).
//!
//! Each comparator errors with a `Divergence` carrying the kernel,
//! lane names, the failing index (if any), and a free-form detail
//! string suitable for direct use in panic / status messages.

use super::output::{RawOutput, RawValues};

#[derive(Debug)]
pub struct Divergence {
    pub kernel: &'static str,
    pub oracle_lane: &'static str,
    pub candidate_lane: &'static str,
    pub index: Option<usize>,
    pub detail: String,
}

impl core::fmt::Display for Divergence {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.index {
            Some(i) => write!(
                f,
                "kernel={} oracle={} candidate={} index={} {}",
                self.kernel, self.oracle_lane, self.candidate_lane, i, self.detail,
            ),
            None => write!(
                f,
                "kernel={} oracle={} candidate={} {}",
                self.kernel, self.oracle_lane, self.candidate_lane, self.detail,
            ),
        }
    }
}

pub fn compare_f32(
    oracle: &RawOutput,
    candidate: &RawOutput,
    max_ulps: u32,
) -> Result<(), Divergence> {
    if oracle.kernel != candidate.kernel {
        return Err(div(
            oracle,
            candidate,
            None,
            format!(
                "kernel name mismatch: {} vs {}",
                oracle.kernel, candidate.kernel
            ),
        ));
    }
    let (a, b) = match (&oracle.values, &candidate.values) {
        (RawValues::F32(a), RawValues::F32(b)) => (a.as_slice(), b.as_slice()),
        _ => {
            return Err(div(
                oracle,
                candidate,
                None,
                format!(
                    "compare_f32 expected f32 outputs, got oracle={} candidate={}",
                    oracle.values.type_tag(),
                    candidate.values.type_tag()
                ),
            ));
        }
    };
    if a.len() != b.len() {
        return Err(div(
            oracle,
            candidate,
            None,
            format!("length mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        match ulp_distance(*x, *y) {
            Some(d) if d <= max_ulps => {}
            Some(d) => {
                return Err(div(
                    oracle,
                    candidate,
                    Some(i),
                    format!(
                        "f32 oracle={} candidate={} ulp_distance={} max={}",
                        x, y, d, max_ulps
                    ),
                ));
            }
            None => {
                return Err(div(
                    oracle,
                    candidate,
                    Some(i),
                    format!("f32 oracle={} candidate={} (NaN or sign-disjoint)", x, y),
                ));
            }
        }
    }
    Ok(())
}

/// Trace-membership comparator (D-ext.3b.2). For kernels that have
/// multiple model-permitted outcomes (atomic races, weakly-ordered
/// litmus tests), the candidate output must equal at least one element
/// of `permitted`. Each `permitted[i]` is a complete `Vec<u32>` of
/// expected outputs, of the same length as the candidate.
///
/// The returned `Divergence` lists how many permitted outcomes were
/// considered and (briefly) why each was rejected — useful when adding
/// a new litmus kernel and a backend produces an unexpected interleaving.
pub fn compare_u32_in_set(candidate: &RawOutput, permitted: &[Vec<u32>]) -> Result<(), Divergence> {
    let cand = match &candidate.values {
        RawValues::U32(v) => v.as_slice(),
        _ => {
            return Err(Divergence {
                kernel: candidate.kernel,
                oracle_lane: "permitted-set",
                candidate_lane: candidate.lane.name(),
                index: None,
                detail: format!(
                    "compare_u32_in_set expected u32 candidate, got {}",
                    candidate.values.type_tag()
                ),
            });
        }
    };
    if permitted.is_empty() {
        return Err(Divergence {
            kernel: candidate.kernel,
            oracle_lane: "permitted-set",
            candidate_lane: candidate.lane.name(),
            index: None,
            detail: "empty permitted set — no outcomes accepted".into(),
        });
    }
    for p in permitted {
        if p.len() == cand.len() && p.iter().zip(cand.iter()).all(|(a, b)| a == b) {
            return Ok(());
        }
    }
    Err(Divergence {
        kernel: candidate.kernel,
        oracle_lane: "permitted-set",
        candidate_lane: candidate.lane.name(),
        index: None,
        detail: format!(
            "candidate {:?} not in any of {} permitted outcomes",
            cand,
            permitted.len()
        ),
    })
}

pub fn compare_u32(oracle: &RawOutput, candidate: &RawOutput) -> Result<(), Divergence> {
    if oracle.kernel != candidate.kernel {
        return Err(div(
            oracle,
            candidate,
            None,
            format!(
                "kernel name mismatch: {} vs {}",
                oracle.kernel, candidate.kernel
            ),
        ));
    }
    let (a, b) = match (&oracle.values, &candidate.values) {
        (RawValues::U32(a), RawValues::U32(b)) => (a.as_slice(), b.as_slice()),
        _ => {
            return Err(div(
                oracle,
                candidate,
                None,
                format!(
                    "compare_u32 expected u32 outputs, got oracle={} candidate={}",
                    oracle.values.type_tag(),
                    candidate.values.type_tag()
                ),
            ));
        }
    };
    if a.len() != b.len() {
        return Err(div(
            oracle,
            candidate,
            None,
            format!("length mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return Err(div(
                oracle,
                candidate,
                Some(i),
                format!("u32 oracle={} candidate={}", x, y),
            ));
        }
    }
    Ok(())
}

fn div(
    oracle: &RawOutput,
    candidate: &RawOutput,
    index: Option<usize>,
    detail: String,
) -> Divergence {
    Divergence {
        kernel: oracle.kernel,
        oracle_lane: oracle.lane.name(),
        candidate_lane: candidate.lane.name(),
        index,
        detail,
    }
}

/// Bit-exact comparison across any integer `RawValues` variant.
///
/// Used by the op-matrix test (step 082 Layer 1). The matrix
/// instantiates kernels for every `(BinOp, ScalarType, edge-input)`
/// triple, so we need one comparator that accepts the type-tagged
/// output of any of them. Float variants still go through
/// `compare_f32` because they need ULP tolerance.
pub fn compare_bit_exact(oracle: &RawOutput, candidate: &RawOutput) -> Result<(), Divergence> {
    if oracle.kernel != candidate.kernel {
        return Err(div(
            oracle,
            candidate,
            None,
            format!(
                "kernel name mismatch: {} vs {}",
                oracle.kernel, candidate.kernel
            ),
        ));
    }
    match (&oracle.values, &candidate.values) {
        (RawValues::U32(a), RawValues::U32(b)) => slice_bit_exact(oracle, candidate, a, b),
        (RawValues::U64(a), RawValues::U64(b)) => slice_bit_exact(oracle, candidate, a, b),
        (RawValues::I32(a), RawValues::I32(b)) => slice_bit_exact(oracle, candidate, a, b),
        (RawValues::I64(a), RawValues::I64(b)) => slice_bit_exact(oracle, candidate, a, b),
        // f32 falls back to compare_f32 with max_ulps = 0 for bit-exact
        // operations (Add/Sub/Mul/Div on f32 are deterministic on every
        // emitter we ship; only transcendentals need >0 ULP).
        (RawValues::F32(_), RawValues::F32(_)) => compare_f32(oracle, candidate, 0),
        // f64 bit-pattern equality. The op-matrix only generates
        // finite f64 inputs so direct PartialEq works; NaN-aware
        // float comparators land when we ship F64 transcendentals.
        (RawValues::F64(a), RawValues::F64(b)) => f64_bit_pattern_eq(oracle, candidate, a, b),
        // bf16 is compared on its raw 16-bit storage pattern: the kernel
        // and the oracle pack with the identical round-to-nearest-even
        // formula, so agreement is exact.
        (RawValues::BF16(a), RawValues::BF16(b)) => slice_bit_exact(oracle, candidate, a, b),
        // fp8 compared on its raw 8-bit storage pattern — kernel and
        // oracle pack with the identical branchless RNE conversion.
        (RawValues::FP8E5M2(a), RawValues::FP8E5M2(b)) => slice_bit_exact(oracle, candidate, a, b),
        (RawValues::FP8E4M3(a), RawValues::FP8E4M3(b)) => slice_bit_exact(oracle, candidate, a, b),
        // Quantized integer codes — exact match (the affine map + clamp +
        // round-half-to-even agree across backends).
        (RawValues::Q8(a), RawValues::Q8(b)) => slice_bit_exact(oracle, candidate, a, b),
        (RawValues::Q4(a), RawValues::Q4(b)) => slice_bit_exact(oracle, candidate, a, b),
        _ => Err(div(
            oracle,
            candidate,
            None,
            format!(
                "compare_bit_exact: type mismatch oracle={} candidate={}",
                oracle.values.type_tag(),
                candidate.values.type_tag()
            ),
        )),
    }
}

fn f64_bit_pattern_eq(
    oracle: &RawOutput,
    candidate: &RawOutput,
    a: &[f64],
    b: &[f64],
) -> Result<(), Divergence> {
    if a.len() != b.len() {
        return Err(div(
            oracle,
            candidate,
            None,
            format!("length mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x.to_bits() != y.to_bits() {
            return Err(div(
                oracle,
                candidate,
                Some(i),
                format!("f64 oracle={} candidate={}", x, y),
            ));
        }
    }
    Ok(())
}

fn slice_bit_exact<T: PartialEq + core::fmt::Display>(
    oracle: &RawOutput,
    candidate: &RawOutput,
    a: &[T],
    b: &[T],
) -> Result<(), Divergence> {
    if a.len() != b.len() {
        return Err(div(
            oracle,
            candidate,
            None,
            format!("length mismatch: {} vs {}", a.len(), b.len()),
        ));
    }
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return Err(div(
                oracle,
                candidate,
                Some(i),
                format!("{} oracle={} candidate={}", oracle.values.type_tag(), x, y),
            ));
        }
    }
    Ok(())
}

/// Returns `None` if the two values are not ULP-comparable
/// (any NaN, or finite values of opposite sign that are not both ±0).
/// Otherwise returns the integer ULP distance.
fn ulp_distance(x: f32, y: f32) -> Option<u32> {
    if x.is_nan() || y.is_nan() {
        return None;
    }
    if x == y {
        return Some(0);
    }
    if x.is_sign_negative() != y.is_sign_negative() {
        return None;
    }
    let xb = x.to_bits();
    let yb = y.to_bits();
    Some(if xb > yb { xb - yb } else { yb - xb })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_distance_zero() {
        assert_eq!(ulp_distance(1.0, 1.0), Some(0));
    }

    #[test]
    fn ulp_distance_pos_zero_neg_zero() {
        assert_eq!(ulp_distance(0.0, -0.0), Some(0));
    }

    #[test]
    fn ulp_distance_one_step() {
        let x = 1.0f32;
        let y = f32::from_bits(x.to_bits() + 1);
        assert_eq!(ulp_distance(x, y), Some(1));
    }

    #[test]
    fn ulp_distance_nan_unranked() {
        assert_eq!(ulp_distance(f32::NAN, 1.0), None);
    }

    #[test]
    fn ulp_distance_sign_disjoint_unranked() {
        assert_eq!(ulp_distance(1.0, -1.0), None);
    }
}

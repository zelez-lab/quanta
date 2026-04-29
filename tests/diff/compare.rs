//! Tolerance rules for cross-lane output comparison.
//!
//! Currently float-only (`compare` enforces ≤ `max_ulps` distance).
//! Integer kernels (D.3 — reduce_sum, counter) will introduce a
//! bit-exact path alongside this one.

use super::output::RawOutput;

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
    if oracle.values.len() != candidate.values.len() {
        return Err(div(
            oracle,
            candidate,
            None,
            format!(
                "length mismatch: {} vs {}",
                oracle.values.len(),
                candidate.values.len()
            ),
        ));
    }
    for (i, (x, y)) in oracle
        .values
        .iter()
        .zip(candidate.values.iter())
        .enumerate()
    {
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

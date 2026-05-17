//! Reference single-thread CPU implementations of each primitive.
//!
//! These are the **correctness oracles** for differential testing.
//! They are deliberately simple — readability over performance —
//! and have no kernel-shaped restrictions (no `Fin n`, no
//! per-thread cooperation, no shared memory).
//!
//! Every GPU kernel in this crate is tested by running both the
//! GPU implementation and the reference here on the same input
//! and asserting bitwise equality on the result.

/// Reduce a slice by addition, returning the sum.
///
/// Reference impl for `block_reduce_add_u32_kernel`. Uses wrapping
/// addition to match GPU u32 overflow semantics.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::reduce_add_u32;
///
/// assert_eq!(reduce_add_u32(&[]), 0);
/// assert_eq!(reduce_add_u32(&[42]), 42);
/// assert_eq!(reduce_add_u32(&[1, 2, 3, 4]), 10);
/// ```
pub fn reduce_add_u32(xs: &[u32]) -> u32 {
    xs.iter().copied().fold(0u32, u32::wrapping_add)
}

/// Reduce a slice by addition, returning the sum. i32 variant.
/// Wrapping semantics match GPU i32 overflow.
pub fn reduce_add_i32(xs: &[i32]) -> i32 {
    xs.iter().copied().fold(0i32, i32::wrapping_add)
}

/// Reduce a slice by addition, returning the sum. f32 variant.
///
/// Uses a sequential left-fold, matching the most common GPU
/// reduce algorithm's evaluation order. Note: GPUs may use a
/// tree reduce that produces a slightly different bit pattern
/// for the same input due to non-associative IEEE-754 addition;
/// callers should compare with a tolerance, not bit-exact.
pub fn reduce_add_f32(xs: &[f32]) -> f32 {
    xs.iter().copied().sum()
}

/// Reduce a slice by min. Returns `u32::MAX` for an empty input.
pub fn reduce_min_u32(xs: &[u32]) -> u32 {
    xs.iter().copied().min().unwrap_or(u32::MAX)
}

/// Reduce a slice by min. Returns `i32::MAX` for an empty input.
pub fn reduce_min_i32(xs: &[i32]) -> i32 {
    xs.iter().copied().min().unwrap_or(i32::MAX)
}

/// Reduce a slice by min. Returns `f32::INFINITY` for an empty
/// input. Uses `f32::min` (IEEE-754 propagating NaN per
/// `core::f32::min` semantics).
pub fn reduce_min_f32(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Reduce a slice by max. Returns `0` for an empty input.
pub fn reduce_max_u32(xs: &[u32]) -> u32 {
    xs.iter().copied().max().unwrap_or(0)
}

/// Reduce a slice by max. Returns `i32::MIN` for an empty input.
pub fn reduce_max_i32(xs: &[i32]) -> i32 {
    xs.iter().copied().max().unwrap_or(i32::MIN)
}

/// Reduce a slice by max. Returns `f32::NEG_INFINITY` for an
/// empty input.
pub fn reduce_max_f32(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Inclusive prefix-sum scan: `out[i] = sum(xs[0..=i])`.
///
/// Reference impl for `block_scan_add_u32_kernel`. Uses wrapping
/// addition to match GPU u32 overflow semantics. The output has
/// the same length as the input; an empty input produces an empty
/// output.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::scan_add_u32;
///
/// assert_eq!(scan_add_u32(&[]), Vec::<u32>::new());
/// assert_eq!(scan_add_u32(&[1, 2, 3, 4]), vec![1, 3, 6, 10]);
/// ```
pub fn scan_add_u32(xs: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    xs.iter()
        .map(|&x| {
            acc = acc.wrapping_add(x);
            acc
        })
        .collect()
}

/// Sort a `u32` slice into ascending order, returning a new
/// vector. Reference impl for `block_radix_sort_u32_kernel`.
///
/// Uses Rust's `slice::sort` (pdqsort). Stable across runs and
/// platforms — exactly the property the GPU radix sort needs to
/// be checked against.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::radix_sort_u32;
///
/// assert_eq!(radix_sort_u32(&[]), Vec::<u32>::new());
/// assert_eq!(radix_sort_u32(&[3, 1, 4, 1, 5, 9, 2, 6]), vec![1, 1, 2, 3, 4, 5, 6, 9]);
/// ```
pub fn radix_sort_u32(xs: &[u32]) -> Vec<u32> {
    let mut out = xs.to_vec();
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_handles_overflow() {
        // u32::MAX + 1 wraps to 0; u32::MAX + 2 wraps to 1.
        assert_eq!(reduce_add_u32(&[u32::MAX, 1]), 0);
        assert_eq!(reduce_add_u32(&[u32::MAX, 2]), 1);
    }

    #[test]
    fn scan_inclusive_first_eq_input_first() {
        // Inclusive scan: scan[0] always equals xs[0].
        for xs in [vec![7u32], vec![1, 2], vec![5, 10, 15]] {
            let s = scan_add_u32(&xs);
            assert_eq!(s[0], xs[0]);
        }
    }

    #[test]
    fn radix_sort_is_length_preserving_permutation() {
        let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
        let sorted = radix_sort_u32(&xs);
        assert_eq!(sorted.len(), xs.len());
        // Same multiset of values — every original element
        // appears the same number of times in the result.
        let mut a = xs.clone();
        a.sort_unstable();
        assert_eq!(a, sorted);
    }
}

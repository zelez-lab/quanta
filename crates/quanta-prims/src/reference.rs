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

/// Inclusive prefix-sum scan, i32 variant. Wrapping semantics.
pub fn scan_add_i32(xs: &[i32]) -> Vec<i32> {
    let mut acc = 0i32;
    xs.iter()
        .map(|&x| {
            acc = acc.wrapping_add(x);
            acc
        })
        .collect()
}

/// Inclusive prefix-sum scan, f32 variant. Sequential left-fold.
/// Tests against this should use a tolerance — the GPU may use a
/// different evaluation order with slightly different IEEE-754
/// rounding.
pub fn scan_add_f32(xs: &[f32]) -> Vec<f32> {
    let mut acc = 0f32;
    xs.iter()
        .map(|&x| {
            acc += x;
            acc
        })
        .collect()
}

/// Per-block stream compaction over a u32 buffer with an explicit
/// predicate array.
///
/// Reference impl for `block_compact_u32_buffer`. Predicates use
/// the convention `1 = keep, 0 = drop`; any non-zero value is
/// treated as keep, matching the GPU kernel which threads a
/// non-zero check through the predicate.
///
/// Block-local semantics: each `block_size`-sized chunk of `data`
/// is compacted independently. Within block `b`, the kept entries
/// are written contiguously starting at offset `b * block_size`
/// of the output buffer. The number of kept entries is recorded
/// in `counts[b]`. Output slots past the kept count of each
/// block are left untouched (callers should pre-zero `out` or
/// only read up to `counts[b]` per block).
///
/// # Example
///
/// ```
/// use quanta_prims::reference::compact_u32_blocks;
///
/// let preds = [1u32, 0, 1, 1, 0, 0, 1, 0]; // block_size = 4
/// let data  = [10u32, 20, 30, 40, 50, 60, 70, 80];
/// let mut out = [0u32; 8];
/// let mut counts = [0u32; 2];
/// compact_u32_blocks(&preds, &data, &mut out, &mut counts, 4);
/// // Block 0: kept = {10, 30, 40}; written to out[0..3].
/// // Block 1: kept = {70};         written to out[4..5].
/// assert_eq!(&out[0..3], &[10, 30, 40]);
/// assert_eq!(out[4], 70);
/// assert_eq!(counts, [3, 1]);
/// ```
pub fn compact_u32_blocks(
    predicates: &[u32],
    data: &[u32],
    out: &mut [u32],
    counts: &mut [u32],
    block_size: usize,
) {
    assert_eq!(predicates.len(), data.len());
    assert_eq!(out.len(), data.len());
    let num_blocks = data.len() / block_size;
    assert_eq!(counts.len(), num_blocks);
    for (b, count) in counts.iter_mut().enumerate().take(num_blocks) {
        let start = b * block_size;
        let mut kept = 0u32;
        for i in 0..block_size {
            if predicates[start + i] != 0 {
                out[start + kept as usize] = data[start + i];
                kept += 1;
            }
        }
        *count = kept;
    }
}

/// Per-block bucket histogram with a fixed bucket count.
///
/// Reference impl for `block_histogram_u32_buffer`. The caller
/// supplies pre-computed bucket indices (each value must be in
/// `0..num_buckets`). Output is block-major:
/// `counts_out[block * num_buckets + bucket]` holds the count of
/// inputs in that block falling into that bucket.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::histogram_u32_blocks;
///
/// let buckets = [0u32, 0, 1, 2, 1, 1, 3, 3]; // block_size = 4
/// let mut counts = [0u32; 8]; // num_buckets = 4, 2 blocks
/// histogram_u32_blocks(&buckets, &mut counts, 4, 4);
/// // Block 0: bucket 0 → 2, bucket 1 → 1, bucket 2 → 1, bucket 3 → 0.
/// // Block 1: bucket 0 → 0, bucket 1 → 2, bucket 2 → 0, bucket 3 → 2.
/// assert_eq!(counts, [2, 1, 1, 0, 0, 2, 0, 2]);
/// ```
pub fn histogram_u32_blocks(
    buckets_in: &[u32],
    counts_out: &mut [u32],
    block_size: usize,
    num_buckets: usize,
) {
    let num_blocks = buckets_in.len() / block_size;
    assert_eq!(counts_out.len(), num_blocks * num_buckets);
    counts_out.fill(0);
    for b in 0..num_blocks {
        let start = b * block_size;
        for i in 0..block_size {
            let bucket = buckets_in[start + i] as usize;
            assert!(
                bucket < num_buckets,
                "bucket index {bucket} out of range (num_buckets={num_buckets})"
            );
            counts_out[b * num_buckets + bucket] += 1;
        }
    }
}

/// Per-block top-K selection. For each `block_size`-sized chunk
/// of `data`, emits the K largest values in descending order to
/// the corresponding region of `top_k_out`. `top_k_out.len()`
/// must equal `(data.len() / block_size) * k`.
///
/// Reference impl for `block_top_k_u32_buffer`. Sort-based: each
/// block is sorted descending then the first K are taken.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::top_k_u32_blocks;
///
/// let data = [5u32, 1, 9, 3, 7, 2, 8, 6];
/// let mut out = [0u32; 4]; // k = 2, block_size = 4, 2 blocks
/// top_k_u32_blocks(&data, &mut out, 4, 2);
/// // Block 0: [5, 1, 9, 3] → top-2 = [9, 5].
/// // Block 1: [7, 2, 8, 6] → top-2 = [8, 7].
/// assert_eq!(out, [9, 5, 8, 7]);
/// ```
pub fn top_k_u32_blocks(data: &[u32], top_k_out: &mut [u32], block_size: usize, k: usize) {
    let num_blocks = data.len() / block_size;
    assert_eq!(top_k_out.len(), num_blocks * k);
    assert!(k <= block_size);
    for b in 0..num_blocks {
        let start = b * block_size;
        let mut sorted: Vec<u32> = data[start..start + block_size].to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a)); // descending
        top_k_out[b * k..(b + 1) * k].copy_from_slice(&sorted[..k]);
    }
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

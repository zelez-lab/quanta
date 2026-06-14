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

/// Per-block segmented inclusive prefix sum with head flags.
///
/// Reference impl for `block_segmented_scan_add_u32_buffer`. A
/// non-zero `flags[i]` marks element `i` as the head of a new
/// segment: the running sum restarts there (inclusive — the head's
/// output is its own value). Block starts always begin a new
/// segment, flag or not (the primitive is block-local).
///
/// # Example
///
/// ```
/// use quanta_prims::reference::segmented_scan_add_u32_blocks;
///
/// let data  = [1u32, 2, 3, 4, 5, 6, 7, 8];
/// let flags = [0u32, 0, 1, 0, 0, 1, 0, 0]; // block_size = 4
/// let mut out = [0u32; 8];
/// segmented_scan_add_u32_blocks(&data, &flags, &mut out, 4);
/// // Block 0: segments [1,2] and [3,4]  → [1, 3, 3, 7].
/// // Block 1: segments [5] and [6,7,8]  → [5, 6, 13, 21].
/// assert_eq!(out, [1, 3, 3, 7, 5, 6, 13, 21]);
/// ```
pub fn segmented_scan_add_u32_blocks(
    data: &[u32],
    flags: &[u32],
    out: &mut [u32],
    block_size: usize,
) {
    assert_eq!(flags.len(), data.len());
    assert_eq!(out.len(), data.len());
    let num_blocks = data.len() / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        let mut acc = 0u32;
        for i in 0..block_size {
            if i == 0 || flags[start + i] != 0 {
                acc = data[start + i];
            } else {
                acc = acc.wrapping_add(data[start + i]);
            }
            out[start + i] = acc;
        }
    }
}

/// Per-block segmented reduce with head flags. For each block,
/// emits one total per segment, written contiguously to the
/// block's region of `totals_out` (block-major, same convention
/// as `compact_u32_blocks`); `counts_out[b]` holds the number of
/// segments in block `b`. Slots past the segment count are zeroed.
///
/// Reference impl for `block_segmented_reduce_add_u32_buffer`.
/// Segment semantics match `segmented_scan_add_u32_blocks`: a
/// non-zero flag starts a new segment, and so does every block
/// start.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::segmented_reduce_add_u32_blocks;
///
/// let data  = [1u32, 2, 3, 4, 5, 6, 7, 8];
/// let flags = [0u32, 0, 1, 0, 0, 1, 0, 0]; // block_size = 4
/// let mut totals = [0u32; 8];
/// let mut counts = [0u32; 2];
/// segmented_reduce_add_u32_blocks(&data, &flags, &mut totals, &mut counts, 4);
/// // Block 0: segments [1,2] and [3,4]  → totals 3, 7.
/// // Block 1: segments [5] and [6,7,8]  → totals 5, 21.
/// assert_eq!(&totals[0..2], &[3, 7]);
/// assert_eq!(&totals[4..6], &[5, 21]);
/// assert_eq!(counts, [2, 2]);
/// ```
pub fn segmented_reduce_add_u32_blocks(
    data: &[u32],
    flags: &[u32],
    totals_out: &mut [u32],
    counts_out: &mut [u32],
    block_size: usize,
) {
    assert_eq!(flags.len(), data.len());
    assert_eq!(totals_out.len(), data.len());
    let num_blocks = data.len() / block_size;
    assert_eq!(counts_out.len(), num_blocks);
    totals_out.fill(0);
    for b in 0..num_blocks {
        let start = b * block_size;
        let mut seg = 0usize;
        let mut acc = 0u32;
        for i in 0..block_size {
            if i == 0 || flags[start + i] != 0 {
                if i != 0 {
                    totals_out[start + seg] = acc;
                    seg += 1;
                }
                acc = data[start + i];
            } else {
                acc = acc.wrapping_add(data[start + i]);
            }
        }
        totals_out[start + seg] = acc;
        counts_out[b] = (seg + 1) as u32;
    }
}

/// Per-block **stable segmented sort**: within each `block_size`
/// chunk, sort ascending by value inside each head-flag-delimited
/// segment; segments stay in input order, equal values keep input
/// order. Returns a new vector.
///
/// Reference impl for `block_segmented_sort_u32_buffer`. Same
/// head-flag convention as the segmented scan/reduce oracles: a
/// non-zero flag opens a segment, and every block start opens one
/// regardless of its flag.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::segmented_sort_u32_blocks;
///
/// let data  = [3u32, 1, 2,  9, 5, 7];
/// let flags = [0u32, 0, 0,  1, 0, 0]; // block_size = 6, head at idx 3
/// let out = segmented_sort_u32_blocks(&data, &flags, 6);
/// // Segment [3,1,2] → [1,2,3]; segment [9,5,7] → [5,7,9].
/// assert_eq!(out, vec![1, 2, 3, 5, 7, 9]);
/// ```
pub fn segmented_sort_u32_blocks(data: &[u32], flags: &[u32], block_size: usize) -> Vec<u32> {
    assert_eq!(flags.len(), data.len());
    let mut out = data.to_vec();
    let num_blocks = data.len() / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        // Walk segment boundaries: a segment runs from its head
        // (block start, or a set flag) to just before the next head.
        let mut seg_start = start;
        let mut i = start + 1;
        while i <= start + block_size {
            let at_end = i == start + block_size;
            let is_head = !at_end && flags[i] != 0;
            if at_end || is_head {
                // sort_unstable is fine: equal values are
                // indistinguishable, so the result is identical to a
                // stable sort element-for-element.
                out[seg_start..i].sort_unstable();
                seg_start = i;
            }
            i += 1;
        }
    }
    out
}

/// Per-block key-value sort: each `block_size`-sized chunk of
/// `(keys, vals)` is sorted ascending by key, values permuted
/// alongside. Returns the sorted pair of vectors.
///
/// Reference impl for `block_sort_kv_u32_buffer`. Uses a stable
/// sort — the GPU kernel is bitonic and therefore unstable, so
/// differential tests with duplicate keys must compare pair
/// multisets, not value sequences.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::sort_kv_u32_blocks;
///
/// let keys = [3u32, 1, 4, 1, 9, 2, 6, 5];
/// let vals = [30u32, 10, 40, 11, 90, 20, 60, 50];
/// let (k, v) = sort_kv_u32_blocks(&keys, &vals, 4);
/// // Block 0: keys [1, 1, 3, 4], vals follow their keys.
/// assert_eq!(&k[0..4], &[1, 1, 3, 4]);
/// assert_eq!(&v[0..4], &[10, 11, 30, 40]);
/// // Block 1: keys [2, 5, 6, 9].
/// assert_eq!(&k[4..8], &[2, 5, 6, 9]);
/// assert_eq!(&v[4..8], &[20, 50, 60, 90]);
/// ```
pub fn sort_kv_u32_blocks(keys: &[u32], vals: &[u32], block_size: usize) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(keys.len(), vals.len());
    let mut out_k = Vec::with_capacity(keys.len());
    let mut out_v = Vec::with_capacity(vals.len());
    let num_blocks = keys.len() / block_size;
    for b in 0..num_blocks {
        let start = b * block_size;
        let mut pairs: Vec<(u32, u32)> = keys[start..start + block_size]
            .iter()
            .copied()
            .zip(vals[start..start + block_size].iter().copied())
            .collect();
        pairs.sort_by_key(|&(k, _)| k);
        out_k.extend(pairs.iter().map(|&(k, _)| k));
        out_v.extend(pairs.iter().map(|&(_, v)| v));
    }
    (out_k, out_v)
}

/// Per-block **stable** key-value sort: each `block_size`-sized
/// chunk of `(keys, vals)` is sorted ascending by key, equal keys
/// keeping their input order, values permuted alongside.
///
/// Reference impl for `block_radix_sort_kv_u32_buffer` (the stable
/// LSD-radix kv kernel). Unlike [`sort_kv_u32_blocks`] — whose
/// bitonic kernel is unstable, so its tests compare pair multisets
/// — this oracle pins the **exact** `(key, value)` sequence, which
/// the stable kernel must reproduce element for element.
///
/// The body is identical to [`sort_kv_u32_blocks`] (`sort_by_key`
/// is already stable); the separate name documents the stronger
/// contract its caller is allowed to assert.
///
/// # Example
///
/// ```
/// use quanta_prims::reference::sort_kv_stable_u32_blocks;
///
/// // Two pairs share key 1; stability keeps val 10 before val 11.
/// let keys = [1u32, 3, 1, 2];
/// let vals = [10u32, 30, 11, 20];
/// let (k, v) = sort_kv_stable_u32_blocks(&keys, &vals, 4);
/// assert_eq!(k, vec![1, 1, 2, 3]);
/// assert_eq!(v, vec![10, 11, 20, 30]);
/// ```
pub fn sort_kv_stable_u32_blocks(
    keys: &[u32],
    vals: &[u32],
    block_size: usize,
) -> (Vec<u32>, Vec<u32>) {
    sort_kv_u32_blocks(keys, vals, block_size)
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

//! Tier 3 — device-wide convenience wrappers.
//!
//! "I just have data and want it reduced / sorted on the GPU"
//! entry points. Each wrapper takes host data, handles upload,
//! identity padding, multi-pass orchestration, and readback, and
//! returns host results.
//!
//! These wrap the Tier-1 block primitives; they are demos of the
//! block-cooperative API, **not** the load-bearing surface. If
//! your data already lives on the GPU inside a larger pipeline,
//! call the `block_*` kernels directly and keep the intermediate
//! results resident.
//!
//! ## Reduce
//!
//! `device_reduce_<op>_<ty>` for `(op, ty)` in `{add, min, max} ×
//! {u32, i32, f32}`. Arbitrary input length ≥ 1: the input is
//! padded to a multiple of 256 with the operation's identity
//! element, reduced block-wise on the GPU, and the per-block
//! partials are fed back in until one value remains (256× shrink
//! per pass — a 1M-element input takes 3 passes).
//!
//! Each reduce comes in three spellings: host data in / host scalar
//! out (`device_reduce_<op>_<ty>`), device field in / host scalar
//! out (`…_field`), and device field in / **1-element device field
//! out** (`…_resident`) — the resident form never touches host
//! memory, so under deferred dispatch it encodes into the open lane
//! without forcing a flush. All three run the identical pass
//! structure over identical padded values, so their results are
//! bit-equal for the same input.
//!
//! Note for f32: the GPU tree-reduction order differs from a
//! sequential fold, so sums land within a few ULP of the
//! reference, not bit-equal.
//!
//! ## Sort
//!
//! `device_sort_u32` pads to the next power of two with
//! `u32::MAX`, then runs a device-wide bitonic network — one
//! [`global_bitonic_pass_u32`] launch per (k, j) pass, log²(n)
//! launches total. Inputs that fit one 256-key tile short-circuit
//! to a single [`block_radix_sort_u32_buffer`] launch.

use crate::gpu_kernel::{
    block_radix_sort_u32_buffer, block_reduce_add_f32_buffer, block_reduce_add_f32_tree_buffer,
    block_reduce_add_i32_buffer, block_reduce_add_i32_tree_buffer, block_reduce_add_u32_buffer,
    block_reduce_add_u32_tree_buffer, block_reduce_max_f32_buffer,
    block_reduce_max_f32_tree_buffer, block_reduce_max_i32_buffer,
    block_reduce_max_i32_tree_buffer, block_reduce_max_u32_buffer,
    block_reduce_max_u32_tree_buffer, block_reduce_min_f32_buffer,
    block_reduce_min_f32_tree_buffer, block_reduce_min_i32_buffer,
    block_reduce_min_i32_tree_buffer, block_reduce_min_u32_buffer,
    block_reduce_min_u32_tree_buffer, global_bitonic_pass_u32, pad_copy_f32, pad_copy_i32,
    pad_copy_u32,
};
use quanta_core::{Field, Gpu, QuantaError};

/// Workgroup size shared by every block primitive in this crate.
const BLOCK: usize = 256;

macro_rules! device_reduce {
    ($(#[$doc:meta])* $name:ident, $field_name:ident, $resident_name:ident, $ty:ty,
     $builder:ident, $tree_builder:ident, $pad_kernel:ident, $identity:expr) => {
        $(#[$doc])*
        pub fn $name(gpu: &Gpu, data: &[$ty]) -> Result<$ty, QuantaError> {
            if data.is_empty() {
                return Err(QuantaError::invalid_param(
                    "device-wide reduce requires a non-empty input",
                ));
            }
            // Subgroup-capable backends take the warp-reduce kernel;
            // devices without subgroup arithmetic (Broadcom V3D) take
            // the shared-memory tree kernel. Same dispatch contract.
            let builder = |g: &Gpu| {
                if g.supports_subgroups() {
                    $builder(g)
                } else {
                    $tree_builder(g)
                }
            };
            let mut current: Vec<$ty> = data.to_vec();
            while current.len() > 1 {
                current = reduce_pass(gpu, &mut current, $identity, builder)?;
            }
            Ok(current[0])
        }

        /// Device-resident variant: reduces a field that already lives on the
        /// GPU **without downloading the data**. The first pass copies the
        /// source into a padded buffer device-side; only the per-block
        /// partials (256× smaller) round-trip through host for the tail passes.
        pub fn $field_name(gpu: &Gpu, data: &Field<$ty>, n: usize) -> Result<$ty, QuantaError> {
            if n == 0 {
                return Err(QuantaError::invalid_param(
                    "device-wide reduce requires a non-empty input",
                ));
            }
            let builder = |g: &Gpu| {
                if g.supports_subgroups() {
                    $builder(g)
                } else {
                    $tree_builder(g)
                }
            };
            let mut current = reduce_pass_field(gpu, data, n, $identity, builder)?;
            while current.len() > 1 {
                current = reduce_pass(gpu, &mut current, $identity, builder)?;
            }
            Ok(current[0])
        }

        /// Fully device-resident variant: reduces the first `n` elements of
        /// an on-device field into a **1-element field** that stays on the
        /// GPU. Every pass (the pad-copy staging included) goes through the
        /// dispatch lane, so nothing here reads host memory or forces a
        /// deferred-lane flush; reading the returned field completes the
        /// pending passes like any other field read. Identical pass
        /// structure and padding to the host-returning variants — the
        /// result is bit-equal to theirs.
        pub fn $resident_name(
            gpu: &Gpu,
            data: &Field<$ty>,
            n: usize,
        ) -> Result<Field<$ty>, QuantaError> {
            if n == 0 {
                return Err(QuantaError::invalid_param(
                    "device-wide reduce requires a non-empty input",
                ));
            }
            let builder = |g: &Gpu| {
                if g.supports_subgroups() {
                    $builder(g)
                } else {
                    $tree_builder(g)
                }
            };
            let padded = n.div_ceil(BLOCK) * BLOCK;
            let mut cur = if padded == n {
                // Block-aligned input: the reduce reads exactly [0, n) —
                // bind the source directly, no staging copy.
                reduce_pass_resident(gpu, data, padded, $identity, builder)?
            } else {
                let staged = gpu.field::<$ty>(padded)?;
                let mut w = $pad_kernel(gpu)?;
                w.bind(0, data);
                w.bind(1, &staged);
                w.set_value(2, n as u32);
                w.set_value(3, $identity);
                gpu.dispatch(&w, padded as u32)?;
                reduce_pass_resident(gpu, &staged, padded, $identity, builder)?
            };
            while cur.len() > 1 {
                let len = cur.len();
                cur = reduce_pass_resident(gpu, &cur, len, $identity, builder)?;
            }
            Ok(cur)
        }
    };
}

/// First reduce pass over an on-device field of `n` elements: copy into a
/// padded buffer device-side, fill the `[n, padded)` tail with `identity`
/// (a ≤255-element host write), reduce, return the per-block partials.
fn reduce_pass_field<T: Copy>(
    gpu: &Gpu,
    src: &Field<T>,
    n: usize,
    identity: T,
    builder: impl FnOnce(&Gpu) -> Result<quanta_core::Wave, QuantaError>,
) -> Result<Vec<T>, QuantaError> {
    let padded_len = n.div_ceil(BLOCK) * BLOCK;
    let num_blocks = padded_len / BLOCK;

    let data_field = gpu.field::<T>(padded_len)?;
    data_field.copy_from(src)?; // device→device copy of the first n elements
    if padded_len > n {
        data_field.write_at(n, &vec![identity; padded_len - n])?; // tail only
    }

    let out_field = gpu.field::<T>(num_blocks)?;
    out_field.write(&vec![identity; num_blocks])?;

    let mut wave = builder(gpu)?;
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, padded_len as u32)?;
    pulse.wait()?;
    out_field.read()
}

/// One device-resident block-reduce pass over an already-padded field
/// (`padded_len` a multiple of [`BLOCK`], or the input of a 1-block
/// pass): reduce on the GPU, return the per-block partials in a fresh
/// field **already padded for the next pass** — the tail beyond
/// `num_blocks` is identity, written host-side into the fresh buffer
/// (fresh fields owe the lane nothing, so the upload never flushes).
/// The dispatch pulse is dropped un-waited; the lane orders the pass.
fn reduce_pass_resident<T: Copy>(
    gpu: &Gpu,
    src: &Field<T>,
    padded_len: usize,
    identity: T,
    builder: impl FnOnce(&Gpu) -> Result<quanta_core::Wave, QuantaError>,
) -> Result<Field<T>, QuantaError> {
    let num_blocks = padded_len / BLOCK;
    let out_len = if num_blocks == 1 {
        1
    } else {
        num_blocks.div_ceil(BLOCK) * BLOCK
    };
    let out_field = gpu.field::<T>(out_len)?;
    if out_len > num_blocks {
        out_field.write_at(num_blocks, &vec![identity; out_len - num_blocks])?;
    }
    let mut wave = builder(gpu)?;
    wave.bind(0, src);
    wave.bind(1, &out_field);
    gpu.dispatch(&wave, padded_len as u32)?;
    Ok(out_field)
}

/// One block-reduce pass: pad `current` to a multiple of [`BLOCK`]
/// with `identity`, reduce on the GPU, return the per-block
/// partials (256× smaller).
fn reduce_pass<T: Copy>(
    gpu: &Gpu,
    current: &mut Vec<T>,
    identity: T,
    builder: impl FnOnce(&Gpu) -> Result<quanta_core::Wave, QuantaError>,
) -> Result<Vec<T>, QuantaError> {
    let padded_len = current.len().div_ceil(BLOCK) * BLOCK;
    current.resize(padded_len, identity);
    let num_blocks = padded_len / BLOCK;

    let data_field = gpu.field::<T>(padded_len)?;
    let out_field = gpu.field::<T>(num_blocks)?;
    data_field.write(current)?;
    out_field.write(&vec![identity; num_blocks])?;

    let mut wave = builder(gpu)?;
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, padded_len as u32)?;
    pulse.wait()?;
    out_field.read()
}

device_reduce!(
    /// Device-wide sum of `data` on the GPU. Errors on empty input.
    device_reduce_add_u32, device_reduce_add_u32_field, device_reduce_add_u32_resident, u32, block_reduce_add_u32_buffer, block_reduce_add_u32_tree_buffer, pad_copy_u32, 0u32
);
device_reduce!(
    /// Device-wide sum of `data` on the GPU. Errors on empty input.
    device_reduce_add_i32, device_reduce_add_i32_field, device_reduce_add_i32_resident, i32, block_reduce_add_i32_buffer, block_reduce_add_i32_tree_buffer, pad_copy_i32, 0i32
);
device_reduce!(
    /// Device-wide sum of `data` on the GPU. Errors on empty input.
    /// Tree-reduction order: expect a few ULP of drift vs a
    /// sequential fold.
    device_reduce_add_f32, device_reduce_add_f32_field, device_reduce_add_f32_resident, f32, block_reduce_add_f32_buffer, block_reduce_add_f32_tree_buffer, pad_copy_f32, 0f32
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_u32, device_reduce_min_u32_field, device_reduce_min_u32_resident, u32, block_reduce_min_u32_buffer, block_reduce_min_u32_tree_buffer, pad_copy_u32, u32::MAX
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_i32, device_reduce_min_i32_field, device_reduce_min_i32_resident, i32, block_reduce_min_i32_buffer, block_reduce_min_i32_tree_buffer, pad_copy_i32, i32::MAX
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_f32, device_reduce_min_f32_field, device_reduce_min_f32_resident, f32, block_reduce_min_f32_buffer, block_reduce_min_f32_tree_buffer, pad_copy_f32, f32::INFINITY
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_u32, device_reduce_max_u32_field, device_reduce_max_u32_resident, u32, block_reduce_max_u32_buffer, block_reduce_max_u32_tree_buffer, pad_copy_u32, 0u32
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_i32, device_reduce_max_i32_field, device_reduce_max_i32_resident, i32, block_reduce_max_i32_buffer, block_reduce_max_i32_tree_buffer, pad_copy_i32, i32::MIN
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_f32, device_reduce_max_f32_field, device_reduce_max_f32_resident, f32, block_reduce_max_f32_buffer, block_reduce_max_f32_tree_buffer, pad_copy_f32, f32::NEG_INFINITY
);

/// Sort `data` ascending on the GPU and return the sorted copy.
///
/// Pads to the next power of two (minimum one 256-key tile) with
/// `u32::MAX`, runs a device-wide bitonic network — one
/// [`global_bitonic_pass_u32`] launch per pass — and truncates
/// the padding off the readback. Inputs that fit a single tile
/// take the one-launch [`block_radix_sort_u32_buffer`] path
/// instead.
pub fn device_sort_u32(gpu: &Gpu, data: &[u32]) -> Result<Vec<u32>, QuantaError> {
    let n = data.len();
    if n <= 1 {
        return Ok(data.to_vec());
    }
    // The single-tile block radix sort uses a subgroup scan (OpGroupNonUniform*),
    // which a device without subgroup arithmetic (e.g. Broadcom V3D) can't lower
    // — it would abort the driver. Refuse cleanly there rather than crash. Unlike
    // the reduce family, sort has no subgroup-free fallback yet.
    if !gpu.supports_subgroups() {
        return Err(QuantaError::not_supported(
            "device_sort_u32 requires subgroup arithmetic (no subgroup-free path yet)",
        ));
    }
    let padded_len = n.next_power_of_two().max(BLOCK);
    let mut padded = data.to_vec();
    padded.resize(padded_len, u32::MAX);

    let data_field = gpu.field::<u32>(padded_len)?;
    data_field.write(&padded)?;

    if padded_len == BLOCK {
        // Single tile: the Tier-1 block sort does it in one launch.
        let out_field = gpu.field::<u32>(padded_len)?;
        out_field.write(&padded)?;
        let mut wave = block_radix_sort_u32_buffer(gpu)?;
        wave.bind(0, &data_field);
        wave.bind(1, &out_field);
        let mut pulse = gpu.dispatch(&wave, padded_len as u32)?;
        pulse.wait()?;
        let mut out = out_field.read()?;
        out.truncate(n);
        return Ok(out);
    }

    let mut wave = global_bitonic_pass_u32(gpu)?;
    wave.bind(0, &data_field);
    let mut k: u32 = 2;
    while (k as usize) <= padded_len {
        let mut j: u32 = k / 2;
        while j > 0 {
            wave.set_value(1, k);
            wave.set_value(2, j);
            // Each pass must fully retire before the next reads the
            // exchanged elements — the dispatch boundary is the
            // device-wide barrier of the bitonic network.
            let mut pulse = gpu.dispatch(&wave, padded_len as u32)?;
            pulse.wait()?;
            j /= 2;
        }
        k *= 2;
    }

    let mut out = data_field.read()?;
    out.truncate(n);
    Ok(out)
}

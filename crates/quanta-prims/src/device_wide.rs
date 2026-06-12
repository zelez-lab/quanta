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
    block_radix_sort_u32_buffer, block_reduce_add_f32_buffer, block_reduce_add_i32_buffer,
    block_reduce_add_u32_buffer, block_reduce_max_f32_buffer, block_reduce_max_i32_buffer,
    block_reduce_max_u32_buffer, block_reduce_min_f32_buffer, block_reduce_min_i32_buffer,
    block_reduce_min_u32_buffer, global_bitonic_pass_u32,
};
use quanta::{Gpu, QuantaError};

/// Workgroup size shared by every block primitive in this crate.
const BLOCK: usize = 256;

macro_rules! device_reduce {
    ($(#[$doc:meta])* $name:ident, $ty:ty, $builder:ident, $identity:expr) => {
        $(#[$doc])*
        pub fn $name(gpu: &Gpu, data: &[$ty]) -> Result<$ty, QuantaError> {
            if data.is_empty() {
                return Err(QuantaError::invalid_param(
                    "device-wide reduce requires a non-empty input",
                ));
            }
            let mut current: Vec<$ty> = data.to_vec();
            while current.len() > 1 {
                current = reduce_pass(gpu, &mut current, $identity, $builder)?;
            }
            Ok(current[0])
        }
    };
}

/// One block-reduce pass: pad `current` to a multiple of [`BLOCK`]
/// with `identity`, reduce on the GPU, return the per-block
/// partials (256× smaller).
fn reduce_pass<T: Copy>(
    gpu: &Gpu,
    current: &mut Vec<T>,
    identity: T,
    builder: impl FnOnce(&Gpu) -> Result<quanta::Wave, QuantaError>,
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
    device_reduce_add_u32, u32, block_reduce_add_u32_buffer, 0u32
);
device_reduce!(
    /// Device-wide sum of `data` on the GPU. Errors on empty input.
    device_reduce_add_i32, i32, block_reduce_add_i32_buffer, 0i32
);
device_reduce!(
    /// Device-wide sum of `data` on the GPU. Errors on empty input.
    /// Tree-reduction order: expect a few ULP of drift vs a
    /// sequential fold.
    device_reduce_add_f32, f32, block_reduce_add_f32_buffer, 0f32
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_u32, u32, block_reduce_min_u32_buffer, u32::MAX
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_i32, i32, block_reduce_min_i32_buffer, i32::MAX
);
device_reduce!(
    /// Device-wide minimum of `data` on the GPU. Errors on empty input.
    device_reduce_min_f32, f32, block_reduce_min_f32_buffer, f32::INFINITY
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_u32, u32, block_reduce_max_u32_buffer, 0u32
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_i32, i32, block_reduce_max_i32_buffer, i32::MIN
);
device_reduce!(
    /// Device-wide maximum of `data` on the GPU. Errors on empty input.
    device_reduce_max_f32, f32, block_reduce_max_f32_buffer, f32::NEG_INFINITY
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

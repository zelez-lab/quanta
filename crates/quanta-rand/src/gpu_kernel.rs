//! GPU `#[quanta::kernel]` for filling a buffer with per-quark
//! pseudo-random u32 values.
//!
//! Each quark computes its own seed via `per_quark_seed(seed, id)`,
//! runs one xoshiro128++ step, and stores the result into its slot
//! in `out`. No shared state; the kernel scales linearly with quark
//! count.
//!
//! Determinism: running the same kernel with the same `seed` and
//! `quark_count` produces bit-identical output. The CPU reference
//! in `tests/correctness.rs` constructs the same stream and
//! verifies bit-exact equality.

use quanta::*;

/// Auto-dispatch struct for `fill_buffer`. `seed_lo` / `seed_hi` are
/// the two u32 halves of the 64-bit seed — Quanta scalar push consts
/// are u32 in the macro surface, so we pass the seed as a pair and
/// reconstruct on the kernel side via shift+or.
#[derive(quanta::Fields)]
pub struct FillBufferData {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// 32-bit splitmix step. Murmur3 finaliser shape (32-bit variant).
/// Used to expand each half of the per-quark mixed seed into a
/// well-diffused state word. Marked `#[quanta::device]` so the macro
/// splices its source into the wasm shell when a kernel calls it —
/// the same source survives unchanged for CPU/host use.
#[allow(dead_code)]
#[quanta::device]
fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E3779B9u32);
    x = (x ^ (x >> 16u32)).wrapping_mul(0x85EBCA6Bu32);
    x = (x ^ (x >> 13u32)).wrapping_mul(0xC2B2AE35u32);
    x ^ (x >> 16u32)
}

/// Final mix of the xoshiro128++ output function.
#[allow(dead_code)]
#[quanta::device]
fn xoshiro_output_mix(s0: u32, s3: u32) -> u32 {
    let sum = s0.wrapping_add(s3);
    sum.rotate_left(7u32).wrapping_add(s0)
}

/// Fill `d.out` with `d.out.len()` pseudo-random u32 values, one per
/// quark. Each quark's value is derived from the shared seed
/// `(d.seed_hi, d.seed_lo)` mixed with its own `quark_id`.
///
/// The device functions above are spliced into the wasm-shell crate
/// at macro expansion time, so `splitmix32` and `xoshiro_output_mix`
/// resolve at rustc-compile time. At -O3 LLVM typically inlines them
/// into the caller before the WASM lowerer sees the calls.
#[quanta::kernel]
pub fn fill_buffer(d: &FillBufferData) {
    let id = quark_id();
    // Mix each half of the 64-bit seed with the per-quark id.
    let mixed_lo: u32 = d.seed_lo ^ id.wrapping_mul(0x9E37_79B9u32);
    let mixed_hi: u32 = d.seed_hi ^ id.wrapping_mul(0x7F4A_7C15u32);

    let s0: u32 = splitmix32(mixed_lo);
    let s3: u32 = splitmix32(mixed_hi);

    let result: u32 = xoshiro_output_mix(s0, s3);
    d.out[id as usize] = result;
}

/// Convenience: dispatch `fill_buffer` and wait for completion.
///
/// Pass a host-side `Vec<u32>` of the desired length (initial
/// contents are ignored). Returns the filled buffer.
pub fn fill_buffer_gpu(gpu: &Gpu, len: usize, seed: u64) -> Result<Vec<u32>, QuantaError> {
    let mut data = FillBufferData {
        out: vec![0u32; len],
        seed_lo: seed as u32,
        seed_hi: (seed >> 32) as u32,
    };
    fill_buffer(gpu, &mut data, len as u32)?.wait()?;
    Ok(data.out)
}

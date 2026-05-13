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
/// the two u32 halves of the 64-bit seed (Quanta kernels only see
/// u32 scalars in v0.1).
#[derive(quanta::Fields)]
pub struct FillBufferData {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

/// Fill `d.out` with `d.out.len()` pseudo-random u32 values, one per
/// quark. Each quark's value is derived from the shared seed
/// `(d.seed_hi, d.seed_lo)` mixed with its own `quark_id`.
///
/// Implementation note: the splitmix32 and xoshiro128++ steps are
/// inlined here as straight-line arithmetic because the
/// `#[quanta::kernel]` WASM-route extractor doesn't currently
/// propagate helper functions defined in the same crate. v0.2 will
/// factor these into reusable helpers once the macro supports it.
#[quanta::kernel]
pub fn fill_buffer(d: &FillBufferData) {
    let id = quark_id();
    // V0 kernel: per_quark_seed via u32 mix, four rounds of
    // splitmix32 expansion to build the 4×u32 state, then one
    // xoshiro128++ output step.
    let mixed_lo: u32 = d.seed_lo ^ id.wrapping_mul(0x9E37_79B9u32);
    let mixed_hi: u32 = d.seed_hi ^ id.wrapping_mul(0x7F4A_7C15u32);

    // splitmix32(mixed_lo) → s0
    let a0: u32 = mixed_lo.wrapping_add(0x9E37_79B9u32);
    let b0: u32 = (a0 ^ (a0 >> 16u32)).wrapping_mul(0x85EB_CA6Bu32);
    let c0: u32 = (b0 ^ (b0 >> 13u32)).wrapping_mul(0xC2B2_AE35u32);
    let s0: u32 = c0 ^ (c0 >> 16u32);

    // splitmix32(mixed_hi) → s3
    let a3: u32 = mixed_hi.wrapping_add(0x9E37_79B9u32);
    let b3: u32 = (a3 ^ (a3 >> 16u32)).wrapping_mul(0x85EB_CA6Bu32);
    let c3: u32 = (b3 ^ (b3 >> 13u32)).wrapping_mul(0xC2B2_AE35u32);
    let s3: u32 = c3 ^ (c3 >> 16u32);

    // Standard xoshiro128++ output: rotl(s0 + s3, 7) + s0.
    // The WASM-route lowering now accepts i32.rotl, so we use
    // `rotate_left` directly; LLVM emits the WASM `i32.rotl`
    // instruction which the lowering maps to BinOp::Rotl.
    let sum: u32 = s0.wrapping_add(s3);
    let result: u32 = sum.rotate_left(7).wrapping_add(s0);
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

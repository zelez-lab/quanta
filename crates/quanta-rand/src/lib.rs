//! Counter-based RNG primitives for Quanta kernels.
//!
//! v0 ships the xoshiro128++ generator with a single-stream API on
//! CPU (this crate) and a per-quark seeded variant inside a
//! `#[quanta::kernel]` (under the `gpu` feature flag).
//!
//! ## CPU usage
//!
//! ```
//! use quanta_rand::Rng;
//!
//! let mut rng = Rng::from_seed(0xC0FFEE);
//! let x: u32 = rng.next_u32();
//! let y: u64 = rng.next_u64();
//! let f: f32 = rng.next_f32();
//! let d: f64 = rng.next_f64();
//! assert!((0.0..1.0).contains(&f));
//! assert!((0.0..1.0).contains(&d));
//!
//! // Spawn an independent sub-stream by jumping ahead 2^64 steps:
//! let mut other = rng.clone();
//! other.jump();
//! ```
//!
//! ## GPU usage (with `gpu` feature)
//!
//! ```ignore
//! use quanta_rand::fill_buffer_gpu;
//!
//! let gpu = quanta::init()?;
//! let out = fill_buffer_gpu(&gpu, /* len = */ 1024, /* seed = */ 42)?;
//! // `out` is a Vec<u32> of deterministic pseudo-random values.
//! ```
//!
//! Each quark uses its own `quark_id` as a counter, salted by the
//! shared `seed`. Output is deterministic — running the same kernel
//! with the same seed produces bit-identical results.
//!
//! ## Scope and limits (v0)
//!
//! Shipped:
//! - CPU `Rng`: `from_seed`, `next_u32`, `next_u64`, `next_f32`,
//!   `next_f64`, `jump`, `long_jump`.
//! - Per-quark one-shot `quark_next_u32` / `quark_next_u64` on
//!   both CPU (host-reference) and inside `#[quanta::kernel]`
//!   bodies.
//! - Per-quark seeded `fill_buffer` kernel on GPU (under the `gpu`
//!   feature) — bit-exact with the CPU reference.
//!
//! Deferred:
//! - Distributions beyond uniform `[0, 1)` (normal, exponential, etc.).
//! - Other algorithms (PCG, threefry, philox).
//! - GPU-side `jump` / `long_jump` (these are constant-time on CPU
//!   but require a long fixed loop on GPU; pending a use case).
//!
//! ## Why a u32 seed surface on the GPU kernel
//!
//! The CPU `Rng` is fully u64-internal-state and u64-arithmetic-
//! capable (Track C C2 added i64 lowering, but the kernel-side
//! algorithm is unchanged from v0 because per-quark mixing only
//! needs u32). The kernel takes the seed as a `(seed_lo, seed_hi)`
//! pair to stay backwards-compatible with the bit-exact reference.

pub mod philox4x32;
pub mod threefry4x32;
pub mod uniform;
pub mod xoshiro128pp;

pub use philox4x32::{philox4x32_10, philox4x32_r};
pub use threefry4x32::{threefry4x32_20, threefry4x32_r};
pub use uniform::{
    u32_to_open_unit_f32, u32_to_unit_f32, u32_to_unit11_f32, u64_to_open_unit_f64,
    u64_to_unit_f64, u64_to_unit11_f64,
};
pub use xoshiro128pp::{State, jump, long_jump, next_u32};

/// A CPU-side stream of pseudo-random numbers.
#[derive(Clone, Debug)]
pub struct Rng {
    state: State,
}

impl Rng {
    /// Seed the generator from a single u32 value.
    #[inline]
    pub const fn from_seed(seed: u32) -> Self {
        Self {
            state: State::from_seed_u32(seed),
        }
    }

    /// Return the next u32 in the sequence.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        let (v, next) = next_u32(self.state);
        self.state = next;
        v
    }

    /// Return the next u64 — two consecutive u32 outputs packed
    /// with the first output in the high half. The packing order is
    /// stable: `((first as u64) << 32) | (second as u64)`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let hi = self.next_u32();
        let lo = self.next_u32();
        ((hi as u64) << 32) | (lo as u64)
    }

    /// Return a uniform `f32` in `[0, 1)` — exactly 24 bits of entropy.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        u32_to_unit_f32(self.next_u32())
    }

    /// Return a uniform `f64` in `[0, 1)` — exactly 53 bits of entropy.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        u64_to_unit_f64(self.next_u64())
    }

    /// Draw a single `f32` from the standard normal distribution
    /// `N(0, 1)` via Box-Muller. Each call burns two uniform draws
    /// from the underlying stream — Box-Muller produces a pair of
    /// normals but this convenience API discards the second one to
    /// keep the surface scalar. For batched normal draws on GPU,
    /// see `fill_normal_f32_gpu` which uses the pair.
    #[inline]
    pub fn next_normal_f32(&mut self) -> f32 {
        let u1 = u32_to_open_unit_f32(self.next_u32());
        let u2 = u32_to_open_unit_f32(self.next_u32());
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        r * theta.cos()
    }

    /// Fast-forward this stream by 2^64 steps. Equivalent to calling
    /// `next_u32` 2^64 times but constant-time. Use to spawn
    /// non-overlapping inner streams from a common seed.
    #[inline]
    pub fn jump(&mut self) {
        self.state = jump(self.state);
    }

    /// Fast-forward this stream by 2^96 steps. Use for outer-level
    /// parallelism (one `long_jump` per worker thread, one `jump`
    /// per inner stream).
    #[inline]
    pub fn long_jump(&mut self) {
        self.state = long_jump(self.state);
    }
}

/// Per-quark one-shot RNG output.
///
/// Each call is independent — there is no state carried between
/// quarks. The shared `seed` (a 64-bit value split across `seed_lo`
/// and `seed_hi`) is mixed with the per-quark `id` via two
/// independent splitmix32 ladders, and the two resulting state
/// words are summed to produce the output.
///
/// This is the EXACT algorithm the GPU kernel implements; the CPU
/// reference and the GPU kernel produce bit-identical streams when
/// given the same `(seed_lo, seed_hi)`.
///
/// Output: `rotl(s0 + s3, 7) + s0` — the standard xoshiro128++ final
/// mix, now that the WASM-route lowering supports `i32.rotl`.
#[inline]
pub const fn quark_next_u32(seed_lo: u32, seed_hi: u32, id: u32) -> u32 {
    const GOLDEN_LO: u32 = 0x9E37_79B9;
    const GOLDEN_HI: u32 = 0x7F4A_7C15;

    let mixed_lo = seed_lo ^ id.wrapping_mul(GOLDEN_LO);
    let mixed_hi = seed_hi ^ id.wrapping_mul(GOLDEN_HI);

    // splitmix32(mixed_lo) → s0
    let a0 = mixed_lo.wrapping_add(0x9E37_79B9);
    let b0 = (a0 ^ (a0 >> 16)).wrapping_mul(0x85EB_CA6B);
    let c0 = (b0 ^ (b0 >> 13)).wrapping_mul(0xC2B2_AE35);
    let s0 = c0 ^ (c0 >> 16);

    // splitmix32(mixed_hi) → s3
    let a3 = mixed_hi.wrapping_add(0x9E37_79B9);
    let b3 = (a3 ^ (a3 >> 16)).wrapping_mul(0x85EB_CA6B);
    let c3 = (b3 ^ (b3 >> 13)).wrapping_mul(0xC2B2_AE35);
    let s3 = c3 ^ (c3 >> 16);

    let sum = s0.wrapping_add(s3);
    sum.rotate_left(7).wrapping_add(s0)
}

/// Per-quark one-shot u64 RNG output. Computes two
/// `quark_next_u32` values (using `id` and `id + half_id_space`
/// to avoid trivial correlation) and packs them. Bit-identical
/// with what a host-side `Rng` reseeded per-quark would produce
/// if it called `next_u64`.
#[inline]
pub const fn quark_next_u64(seed_lo: u32, seed_hi: u32, id: u32) -> u64 {
    let hi = quark_next_u32(seed_lo, seed_hi, id);
    // Use `id ^ 0x8000_0000` for the second draw so the two outputs
    // come from well-separated points in the splitmix32 ladder.
    let lo = quark_next_u32(seed_lo, seed_hi, id ^ 0x8000_0000u32);
    ((hi as u64) << 32) | (lo as u64)
}

// ────────────────────────────────────────────────────────────────────
// GPU kernel (under `gpu` feature)
// ────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
pub mod gpu_kernel;

#[cfg(feature = "gpu")]
pub use gpu_kernel::{
    fill_buffer_gpu, fill_normal_f32_gpu, fill_uniform_f32_gpu, fill_uniform_f64_gpu,
    fill_uniform_u32_gpu, fill_uniform_u64_gpu,
};

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
//! let f: f32 = rng.next_f32();
//! assert!((0.0..1.0).contains(&f));
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
//! - `Rng::from_seed`, `Rng::next_u32`, `Rng::next_f32` on CPU.
//! - Per-quark seeded `fill_buffer` kernel on GPU (under `gpu` feature).
//! - Bit-exact correctness test: GPU output matches CPU reference
//!   when both are seeded identically.
//!
//! Deferred:
//! - `next_u64`, `next_f64` (Quanta WASM-route doesn't lower i64 yet).
//! - Jump-ahead (independent streams via `jump()` / `long_jump()`).
//! - Distributions beyond uniform-`[0, 1)` (normal, exponential, etc.).
//! - Other algorithms (PCG, threefry, philox).
//!
//! ## Why a u32 seed (and not u64)
//!
//! The standard xoshiro128++ seeds from a u64 via the splitmix64
//! ladder. Quanta's WASM-route currently only lowers u32 arithmetic
//! (slice-1), so the GPU kernel can't construct a u64 seed mixer.
//! v0 takes a u32 seed on both CPU and GPU sides to keep the streams
//! bit-identical; v0.2 will switch to u64 once i64 lowering lands.

pub mod xoshiro128pp;

pub use xoshiro128pp::{State, next_u32, u32_to_unit_f32};

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

    /// Return a uniform `f32` in `[0, 1)` — exactly 24 bits of entropy.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        u32_to_unit_f32(self.next_u32())
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

// ────────────────────────────────────────────────────────────────────
// GPU kernel (under `gpu` feature)
// ────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
pub mod gpu_kernel;

#[cfg(feature = "gpu")]
pub use gpu_kernel::fill_buffer_gpu;

//! Wiring pin for the companion-crate umbrella (`quanta::sci` /
//! `quanta::prims` / `quanta::autograd`).
//!
//! One REAL call through every feature-gated module — an `Array` ufunc,
//! a `linalg` BLAS op, an `fft` roundtrip, a `random` fill, a `layout`
//! shape op, a `prims` device-wide reduce, an `autograd` tape backward.
//! This is a wiring pin, not a benchmark: it proves the facade features
//! activate the companions, the backend forwarding reaches them, and
//! the re-exported paths resolve. Without the umbrella features the
//! modules simply do not exist, so a default build cannot even name
//! them — that absence IS the default-graph check.
//!
//! # Gating
//!
//! Mirrors `tests/litmus.rs`: compiled only when all three umbrella
//! features are on together with at least one backend. The device
//! helper prefers real hardware (`quanta::init()`), falls back to the
//! CPU JIT under `software`, and skips (print + return) when neither
//! yields a device.
//!
//! Run locally:
//!   QUANTA_BACKEND=cpu cargo test --test sci_umbrella \
//!       --features "sci,prims,autograd,software"       # CPU JIT lane
//!   cargo test --test sci_umbrella \
//!       --features "sci,prims,autograd,metal"          # macOS hardware

#![cfg(all(
    feature = "sci",
    feature = "prims",
    feature = "autograd",
    any(feature = "software", feature = "metal", feature = "vulkan")
))]

use quanta::sci::{self, Array};

/// Hardware first (when a hardware backend is compiled in), else the
/// CPU JIT, else skip. `QUANTA_BACKEND=cpu` forces the software lane
/// even on a machine with a GPU.
#[allow(unreachable_code)]
fn gpu() -> Option<quanta::Gpu> {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    if let Ok(g) = quanta::init() {
        return Some(g);
    }
    #[cfg(feature = "software")]
    {
        return Some(quanta::init_cpu());
    }
    None
}

macro_rules! gpu_or_skip {
    ($name:literal) => {
        match gpu() {
            Some(g) => g,
            None => {
                std::eprintln!("sci_umbrella::{}: no device available — skipping", $name);
                return;
            }
        }
    };
}

// ── sci root: Array construction + broadcasting ufunc + reduction ────

#[test]
fn sci_array_ufunc_and_reduction() {
    let g = gpu_or_skip!("sci_array_ufunc_and_reduction");
    let a = Array::from_slice(&g, &[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Array::from_slice(&g, &[10.0f32, 20.0, 30.0, 40.0], &[2, 2]).unwrap();
    let c = a.add(&b).unwrap();
    assert_eq!(c.to_vec().unwrap(), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(c.sum().unwrap(), 110.0);
}

// ── sci::linalg: raw verified BLAS over device fields ─────────────────

#[test]
fn sci_linalg_dot() {
    let g = gpu_or_skip!("sci_linalg_dot");
    let xh = [1.0f32, 2.0, 3.0, 4.0];
    let yh = [5.0f32, 6.0, 7.0, 8.0];
    let x = g.field::<f32>(4).unwrap();
    let y = g.field::<f32>(4).unwrap();
    x.write(&xh).unwrap();
    y.write(&yh).unwrap();
    let got = sci::linalg::dot(&g, &x, &y).unwrap();
    assert_eq!(got, 70.0); // 5 + 12 + 21 + 32
}

// ── sci::fft: forward + inverse roundtrip ─────────────────────────────

#[test]
fn sci_fft_roundtrip() {
    let g = gpu_or_skip!("sci_fft_roundtrip");
    let re: Vec<f32> = (0..8).map(|i| (i as f32) - 3.5).collect();
    let im: Vec<f32> = (0..8).map(|i| ((i * i) % 5) as f32 * 0.25).collect();
    let (fre, fim) = sci::fft::fft(&g, &re, &im).unwrap();
    let (rre, rim) = sci::fft::ifft(&g, &fre, &fim).unwrap();
    for i in 0..8 {
        assert!(
            (rre[i] - re[i]).abs() < 1e-3,
            "re[{i}]: {} vs {}",
            rre[i],
            re[i]
        );
        assert!(
            (rim[i] - im[i]).abs() < 1e-3,
            "im[{i}]: {} vs {}",
            rim[i],
            im[i]
        );
    }
}

// ── sci::random: deterministic device fill ────────────────────────────

#[test]
fn sci_random_uniform_fill() {
    let g = gpu_or_skip!("sci_random_uniform_fill");
    let seed = 0xCAFE_BABE_DEAD_BEEFu64;
    let a = sci::random::fill_uniform_f32_gpu(&g, 256, seed).unwrap();
    assert_eq!(a.len(), 256);
    assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
    // Bit-exact reproducibility from (seed, len) is part of the contract.
    let b = sci::random::fill_uniform_f32_gpu(&g, 256, seed).unwrap();
    assert_eq!(a, b);
}

// ── sci::layout: pure host-side shape algebra ─────────────────────────

#[test]
fn sci_layout_shape_op() {
    let s = sci::layout::Shape::new(&[2, 3, 4]).unwrap();
    assert_eq!(s.rank(), 3);
    assert_eq!(s.linear_size(), 24);
    assert_eq!(s.dims(), &[2, 3, 4]);
}

// ── prims: device-wide reduce ─────────────────────────────────────────

#[test]
fn prims_device_reduce() {
    let g = gpu_or_skip!("prims_device_reduce");
    let data: Vec<f32> = (0..1000).map(|i| (i % 7) as f32 + 0.5).collect();
    let want: f32 = data.iter().sum();
    let got = quanta::prims::device_reduce_add_f32(&g, &data).unwrap();
    assert!(
        (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
        "device reduce: {got} vs {want}"
    );
}

// ── autograd: tape backward pass ──────────────────────────────────────

#[test]
fn autograd_tape_backward() {
    let g = gpu_or_skip!("autograd_tape_backward");
    let tape = quanta::autograd::Tape::<f32>::new();
    let x = tape.var(Array::from_slice(&g, &[1.0, 2.0, 3.0], &[3]).unwrap());
    // loss = sum(x * x)  ⇒  d loss / d x = 2x
    let sq = x.mul(&x).unwrap();
    let loss = sq.sum().unwrap();
    let gx = loss.grad(&x).unwrap();
    assert_eq!(gx.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
}

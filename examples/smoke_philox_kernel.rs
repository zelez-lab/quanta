//! End-to-end smoke for in-kernel device-function calls + push consts.
//!
//! The kernel calls a `#[quanta::device]`-marked helper that does
//! u32 arithmetic (no i64 multiply — that path has a separate
//! lowering bug being tracked). Validates that a real device fn
//! with non-trivial body can be invoked from inside a kernel via
//! the source-splicing machinery, and that push-constant scalars
//! reach the kernel body correctly.
//!
//! Runs on the CPU backend (no Metal Toolchain needed).
//!
//! Run: cargo run --example smoke_philox_kernel --features software

#[quanta::device]
fn splitmix32(mut x: u32) -> u32 {
    // Splitmix32 finaliser — same algorithm used inside Philox's
    // key-schedule and inside the Rng seed expander. Pure u32, no
    // 64-bit arithmetic.
    x = x.wrapping_add(0x9E3779B9u32);
    x = (x ^ (x >> 16u32)).wrapping_mul(0x85EBCA6Bu32);
    x = (x ^ (x >> 13u32)).wrapping_mul(0xC2B2AE35u32);
    x ^ (x >> 16u32)
}

#[derive(quanta::Fields)]
struct SmokeData {
    out: Vec<u32>,
    seed: u32,
}

#[quanta::kernel]
fn smoke_fill(d: &SmokeData) {
    let id = quark_id();
    // Counter-based pattern: mix the per-quark id with a shared seed
    // through a device function. Each quark draws independently;
    // output is bit-exact reproducible from `(seed, id)`.
    let r: u32 = splitmix32(d.seed ^ id);
    d.out[id as usize] = r;
}

fn main() {
    let gpu = quanta::init_cpu();
    println!("GPU: {}", gpu.name());

    let count: usize = 32;
    let seed: u32 = 0xC0FFEE;

    let mut data = SmokeData {
        out: vec![0u32; count],
        seed,
    };
    smoke_fill(&gpu, &mut data, count as u32)
        .expect("dispatch")
        .wait()
        .expect("wait");

    // Host reference uses the same `splitmix32` — the device
    // attribute emits the fn for CPU use unchanged.
    let expected: Vec<u32> = (0..count as u32).map(|i| splitmix32(seed ^ i)).collect();

    println!("output = {:?}", data.out);
    println!("expect = {expected:?}");

    assert_eq!(
        data.out, expected,
        "in-kernel device fn must match host reference bit-for-bit",
    );
    println!(
        "OK — device fn called from kernel, push const threaded, all {count} quarks match host"
    );
}

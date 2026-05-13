//! End-to-end smoke for in-kernel Philox4×32-10.
//!
//! Each quark runs the full 10-round Philox bijection with its own
//! `quark_id` as the counter and a shared (seed_lo, seed_hi) key,
//! and stores the first output word into `out[id]`. The kernel uses
//! the exact source from `quanta_rand::philox4x32::philox4x32_10_first_u32`
//! (transcribed locally because `#[quanta::device]`'s source
//! registry is per-crate — cross-crate import isn't supported yet).
//! Validated bit-for-bit against the host-side reference using the
//! same fn.
//!
//! Runs on the CPU backend (no Metal Toolchain needed).
//!
//! Run: cargo run --example smoke_philox_kernel --features software

#[quanta::device]
fn philox4x32_10_first_u32(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> u32 {
    const M0_K: u32 = 0xD251_1F53;
    const M1_K: u32 = 0xCD9E_8D57;
    const W0_K: u32 = 0x9E37_79B9;
    const W1_K: u32 = 0xBB67_AE85;

    let mut x0 = c0;
    let mut x1 = c1;
    let mut x2 = c2;
    let mut x3 = c3;
    let mut key0 = k0;
    let mut key1 = k1;

    let mut i: u32 = 0;
    while i < 10u32 {
        if i > 0 {
            key0 = key0.wrapping_add(W0_K);
            key1 = key1.wrapping_add(W1_K);
        }
        let product0 = (M0_K as u64).wrapping_mul(x0 as u64);
        let hi0 = (product0 >> 32) as u32;
        let lo0 = product0 as u32;
        let product1 = (M1_K as u64).wrapping_mul(x2 as u64);
        let hi1 = (product1 >> 32) as u32;
        let lo1 = product1 as u32;
        let new_x0 = hi1 ^ x1 ^ key0;
        let new_x1 = lo1;
        let new_x2 = hi0 ^ x3 ^ key1;
        let new_x3 = lo0;
        x0 = new_x0;
        x1 = new_x1;
        x2 = new_x2;
        x3 = new_x3;
        i += 1;
    }
    x0
}

#[derive(quanta::Fields)]
struct PhiloxFillData {
    out: Vec<u32>,
    seed_lo: u32,
    seed_hi: u32,
}

#[quanta::kernel]
fn philox_fill(d: &PhiloxFillData) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}

fn main() {
    let gpu = quanta::init_cpu();
    println!("GPU: {}", gpu.name());

    let count: usize = 32;
    let seed_lo: u32 = 0xDEAD_BEEF;
    let seed_hi: u32 = 0xCAFE_BABE;

    let mut data = PhiloxFillData {
        out: vec![0u32; count],
        seed_lo,
        seed_hi,
    };
    philox_fill(&gpu, &mut data, count as u32)
        .expect("dispatch")
        .wait()
        .expect("wait");

    let expected: Vec<u32> = (0..count as u32)
        .map(|i| philox4x32_10_first_u32(i, 0, 0, 0, seed_lo, seed_hi))
        .collect();

    println!("output  = {:?}", data.out);
    println!("expect  = {expected:?}");

    assert_eq!(
        data.out, expected,
        "in-kernel Philox4x32-10 must match host reference bit-for-bit",
    );
    println!("OK — in-kernel Philox4x32-10 matches host reference for all {count} quarks");
}

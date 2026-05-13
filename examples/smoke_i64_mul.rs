//! Minimal smoke for `(a as u64) * (b as u64)` inside a kernel.
//!
//! Probes the i64.mul lowering path that's broken when called from
//! a `#[quanta::device]` fn (the failure mode surfaced in
//! smoke_philox_kernel's full-Philox variant).
//!
//! Runs on the CPU backend (no Metal Toolchain needed).
//!
//! Run: cargo run --example smoke_i64_mul --features software

#[derive(quanta::Fields)]
struct MulData {
    out: Vec<u32>,
    a: u32,
    b: u32,
}

#[quanta::kernel]
fn mul_hi(d: &MulData) {
    // Full mulhilo32 shape: two i64.muls + both hi and lo extracted
    // + XOR'd together. This is the kernel-side path Philox needs.
    let id = quark_id();
    let av: u32 = d.a.wrapping_add(id);
    let bv: u32 = d.b.wrapping_add(id.wrapping_mul(3u32));
    let cv: u32 = d.a ^ d.b;

    let prod0: u64 = (av as u64).wrapping_mul(bv as u64);
    let hi0: u32 = (prod0 >> 32u32) as u32;
    let lo0: u32 = prod0 as u32;

    let prod1: u64 = (cv as u64).wrapping_mul(av as u64);
    let hi1: u32 = (prod1 >> 32u32) as u32;
    let lo1: u32 = prod1 as u32;

    let result: u32 = hi0 ^ lo0 ^ hi1 ^ lo1;
    d.out[id as usize] = result;
}

fn main() {
    let gpu = quanta::init_cpu();
    let count: usize = 8;
    let a_seed: u32 = 0xC0FF_EE00;
    let b_seed: u32 = 0xD251_1F53; // Philox M0

    let mut data = MulData {
        out: vec![0u32; count],
        a: a_seed,
        b: b_seed,
    };
    mul_hi(&gpu, &mut data, count as u32)
        .expect("dispatch")
        .wait()
        .expect("wait");

    let expected: Vec<u32> = (0..count as u32)
        .map(|i| {
            let av = a_seed.wrapping_add(i);
            let bv = b_seed.wrapping_add(i.wrapping_mul(3));
            let cv = a_seed ^ b_seed;
            let prod0 = (av as u64).wrapping_mul(bv as u64);
            let hi0 = (prod0 >> 32) as u32;
            let lo0 = prod0 as u32;
            let prod1 = (cv as u64).wrapping_mul(av as u64);
            let hi1 = (prod1 >> 32) as u32;
            let lo1 = prod1 as u32;
            hi0 ^ lo0 ^ hi1 ^ lo1
        })
        .collect();

    println!("output = {:?}", data.out);
    println!("expect = {expected:?}");
    assert_eq!(data.out, expected, "i64.mul high half must match host");
    println!("OK — i64.mul high half matches host for {count} quarks");
}

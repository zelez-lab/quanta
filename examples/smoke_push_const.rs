//! End-to-end smoke for CPU-backend push constants.
//!
//! Writes `seed + quark_id` into each slot. Validates that the
//! scalar push constant is correctly threaded from the host
//! `set_value`/auto-dispatch into the kernel body.
//!
//! Runs on the CPU backend (no Metal Toolchain needed).
//!
//! Run: cargo run --example smoke_push_const --features software

#[derive(quanta::Fields)]
struct PushConstData {
    out: Vec<u32>,
    seed: u32,
}

#[quanta::kernel]
fn push_const_add(d: &PushConstData) {
    let id = quark_id();
    let r: u32 = d.seed.wrapping_add(id);
    d.out[id as usize] = r;
}

fn main() {
    let gpu = quanta::init_cpu();
    let count: usize = 8;
    let seed: u32 = 1000;

    let mut data = PushConstData {
        out: vec![0u32; count],
        seed,
    };
    push_const_add(&gpu, &mut data, count as u32)
        .expect("dispatch")
        .wait()
        .expect("wait");

    let expected: Vec<u32> = (0..count as u32).map(|i| seed.wrapping_add(i)).collect();
    println!("output = {:?}", data.out);
    println!("expect = {expected:?}");

    assert_eq!(
        data.out, expected,
        "push-const value must propagate to kernel"
    );
    println!("OK — push constants reach the CPU kernel correctly");
}

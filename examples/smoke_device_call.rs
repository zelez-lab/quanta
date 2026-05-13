//! End-to-end smoke for `#[quanta::device]` propagation.
//!
//! The kernel calls two device functions; their sources must be
//! spliced into the wasm shell so rustc can resolve the calls, and
//! the WASM lowerer must produce correct ops for the resulting
//! instruction stream (typically straight-line, since LLVM inlines
//! device functions at -O3).
//!
//! Runs on the CPU backend so no Metal Toolchain is needed.
//!
//! Run: cargo run --example smoke_device_call --features software

#[quanta::device]
fn splitmix32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E3779B9u32);
    x = (x ^ (x >> 16u32)).wrapping_mul(0x85EBCA6Bu32);
    x = (x ^ (x >> 13u32)).wrapping_mul(0xC2B2AE35u32);
    x ^ (x >> 16u32)
}

#[quanta::device]
fn mix_two(a: u32, b: u32) -> u32 {
    // Transitive device call: depends on splitmix32.
    splitmix32(a).wrapping_add(splitmix32(b))
}

#[quanta::kernel]
fn fill_with_devices(input: &[u32], output: &mut [u32]) {
    // Two-level device call: kernel -> mix_two -> splitmix32 (twice).
    // Validates source splicing into the wasm shell AND nested-call
    // composition through the WASM lowerer.
    let i = quark_id();
    let x: u32 = input[i as usize];
    let y: u32 = i;
    output[i as usize] = mix_two(x, y);
}

fn main() {
    let gpu = quanta::init_cpu();
    println!("GPU: {}", gpu.name());

    let count: usize = 8;
    let input: Vec<u32> = (0..count as u32).map(|i| i * 100).collect();

    let fi = gpu.field::<u32>(count).unwrap();
    let fo = gpu.field::<u32>(count).unwrap();
    fi.write(&input).unwrap();

    let mut wave = fill_with_devices(&gpu).expect("create wave");
    wave.bind(0, &fi);
    wave.bind(1, &fo);

    gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();

    let output = fo.read().unwrap();
    println!("input  = {input:?}");
    println!("output = {output:?}");

    // Host-side reference using the same device fns — they're also
    // emitted as regular Rust fns by the attribute.
    let expected: Vec<u32> = (0..count as u32)
        .map(|i| mix_two(input[i as usize], i))
        .collect();
    println!("expect = {expected:?}");

    assert_eq!(
        output, expected,
        "GPU output must match host-side reference using the same device fns",
    );
    println!("OK — device fns called and host/GPU agree");
}

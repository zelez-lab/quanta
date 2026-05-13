//! Compile + run smoke for u64 arithmetic in a kernel.
//!
//! Exercises i64 binops + I64ExtendI32U + I32WrapI64 + the narrow
//! load/store family (i64.load32_u, i64.store32). Runs on the CPU
//! backend so no Metal Toolchain needed.
//!
//! Run: cargo run --example smoke_u64 --features software

#[quanta::kernel]
fn u64_mix(input: &[u32], output: &mut [u32]) {
    let i = quark_id();
    // input[i] as u64 fuses load+widen into i64.load32_u.
    let lo: u64 = input[i as usize] as u64;
    // Trivial u64 arithmetic — multiply by 3 and add 7. Easy to verify.
    let result: u64 = lo.wrapping_mul(3u64).wrapping_add(7u64);
    // result as u32 + store into a u32 buffer fuses wrap+store
    // into i64.store32 (the new narrow-store path).
    output[i as usize] = result as u32;
}

fn main() {
    // Force CPU backend so the result reflects the IR semantics
    // (not whatever Metal/Vulkan codegen produces). The CPU eval
    // is what we wired up for u64 in C2a-C2c.
    let gpu = quanta::init_cpu();
    println!("GPU: {}", gpu.name());

    let count = 8usize;
    let input: Vec<u32> = (0..count as u32).collect();

    let fi = gpu.field::<u32>(count).unwrap();
    let fo = gpu.field::<u32>(count).unwrap();
    fi.write(&input).unwrap();

    let mut wave = u64_mix(&gpu).expect("create wave");
    wave.bind(0, &fi);
    wave.bind(1, &fo);

    gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();

    let out = fo.read().unwrap();
    println!("Input:  {:?}", &input);
    println!("Output: {:?}", &out);

    // Verify bit-exact: result[i] = (i * 3 + 7) as u32.
    for (i, &v) in out.iter().enumerate() {
        let expected = ((i as u64).wrapping_mul(3).wrapping_add(7)) as u32;
        assert_eq!(
            v, expected,
            "mismatch at index {i}: got {v}, expected {expected}"
        );
    }
    println!(
        "\nAll {} outputs match expected `(i * 3 + 7) as u32`.",
        out.len()
    );
}

//! u64 pack/shift/convert semantics — the CPU backend as oracle.
//!
//! `fill_uniform_f64` builds a 53-bit mantissa by packing two u32 draws:
//! `((hi as u64) << 32) | lo`, then `>> 11`, then `as f64 * 2^-53`. The SPIR-V
//! emitters used to run that chain at 32-bit width (`ScalarType::U64`
//! collapsed to `%uint` in the AOT emitter), so the high word shifted out to
//! zero and every f64 draw came back ~0 on Vulkan — a silent miscompile
//! (spirv-val-valid). The CPU interpreter executes the same KernelOps at real
//! 64-bit width; this test pins the IR-level semantics every backend must
//! match. The SPIR-V module shape is pinned separately in
//! `crates/quanta-ir/tests/emit_spirv_u64_width.rs` and the
//! `quanta-compiler` emit_spirv unit tests.
//!
//! Run: cargo test --test u64_pack_semantics --features software

#![cfg(feature = "software")]

#[quanta::kernel(workgroup = [4])]
fn u64_pack_to_f64(out: &mut [f64], hi_in: &[u32], lo_in: &[u32]) {
    let i = quark_id();
    let hi: u32 = hi_in[i as usize];
    let lo: u32 = lo_in[i as usize];
    let packed: u64 = ((hi as u64) << 32u32) | (lo as u64);
    let bits: u64 = packed >> 11u32;
    let v: f64 = (bits as f64) * (1.0f64 / 9_007_199_254_740_992.0f64);
    out[i as usize] = v;
}

#[test]
fn u64_pack_shift_convert_matches_host() {
    let gpu = quanta::init_cpu();
    let total = 8usize;

    // Values chosen so a 32-bit collapse is unmistakable: with the high
    // word shifted out, every result is < 2^-32 (~0). The correct results
    // are spread across [0, 1).
    let hi: Vec<u32> = vec![
        0x0000_0000,
        0x0000_0001,
        0x9E37_79B9,
        0xFFFF_FFFF,
        0x8000_0000,
        0x7F4A_7C15,
        0x0000_0800,
        0xDEAD_BEEF,
    ];
    let lo: Vec<u32> = vec![
        0x0000_0000,
        0xFFFF_FFFF,
        0x7F4A_7C15,
        0xFFFF_FFFF,
        0x0000_0001,
        0x9E37_79B9,
        0x0000_0000,
        0xCAFE_F00D,
    ];

    let out = gpu.field::<f64>(total).unwrap();
    out.write(&vec![-1.0f64; total]).unwrap();
    let hi_f = gpu.field::<u32>(total).unwrap();
    hi_f.write(&hi).unwrap();
    let lo_f = gpu.field::<u32>(total).unwrap();
    lo_f.write(&lo).unwrap();

    let mut wave = u64_pack_to_f64(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &hi_f);
    wave.bind(2, &lo_f);
    gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();

    let got = out.read().unwrap();
    let want: Vec<f64> = hi
        .iter()
        .zip(&lo)
        .map(|(&h, &l)| {
            let packed = ((h as u64) << 32) | (l as u64);
            (packed >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
        })
        .collect();
    assert_eq!(
        got, want,
        "u64 pack/shift/convert must run at 64-bit width \
         (a 32-bit collapse zeroes the high word: every value ~0)"
    );
    // Belt-and-braces: the buggy lowering made everything < 2^-32.
    assert!(
        got.iter().any(|&v| v > 0.25),
        "results collapsed toward zero — u64 arithmetic ran at 32-bit width: {got:?}"
    );
}

//! Dynamic shared memory (step 048) — end to end.
//!
//! `#[quanta::shared(dyn)]` declares a workgroup-shared array with no
//! size in the source; the size binds at wave creation
//! (`wave_jit_shared`, or the extra parameter on the generated
//! constructor) and late-binds into the IR as an ordinary sized
//! shared array — every backend runs its existing fixed-shared path.
//! JIT-only by construction (the macro enforces the flag).

#[quanta::kernel(jit)]
fn dyn_reduce(input: &[f32], output: &mut [f32]) {
    #[quanta::shared(dyn)]
    let scratch: [f32];
    let i = quark_id();
    let lid = local_id();
    scratch[lid] = input[i];
    barrier();
    if lid == 0u32 {
        let mut acc = 0.0f32;
        let mut j = 0u32;
        while j < 64u32 {
            acc = acc + scratch[j];
            j = j + 1u32;
        }
        output[0] = acc;
    }
}

#[test]
fn dyn_shared_reduction_sums_the_workgroup() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    // 64 lanes, one f32 each: the block size is 64 * 4 bytes.
    let mut wave = dyn_reduce(&gpu, 64 * 4).unwrap();
    let input = gpu.field::<f32>(64).unwrap();
    input.write(&[1.0f32; 64]).unwrap();
    let output = gpu.field::<f32>(1).unwrap();
    output.write(&[0.0f32]).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.wave_dispatch(&wave, [1, 1, 1]).unwrap();
    pulse.wait().unwrap();
    let result = output.read().unwrap();
    assert_eq!(result[0], 64.0, "each of the 64 lanes contributed 1.0");
}

#[test]
fn plain_wave_jit_refuses_unresolved_dyn_shared() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    // Dispatching the dyn-shared kernel without a size must be an
    // error (the validator's unresolved-SharedDeclDyn refusal), never
    // a silently mis-sized buffer.
    let r = gpu.wave_jit(DYN_REDUCE_DEF);
    assert!(r.is_err(), "unresolved dynamic shared must refuse");
}

#[test]
fn wave_jit_shared_validates_the_size() {
    let Ok(gpu) = quanta::init() else {
        eprintln!("skipping: no GPU");
        return;
    };
    assert!(gpu.wave_jit_shared(DYN_REDUCE_DEF, 0).is_err(), "zero size");
    assert!(
        gpu.wave_jit_shared(DYN_REDUCE_DEF, 6).is_err(),
        "size not a multiple of the element width"
    );
}

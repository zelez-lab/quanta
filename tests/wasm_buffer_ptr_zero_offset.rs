//! Regression: `buf[N] = …` for compile-time-constant N (and the
//! load mirror image) was rejected by the wasm-route lowerer with
//! `unsupported WASM op i32.store on non-buffer address BufferPtr(0)`.
//!
//! rustc's optimiser folds the constant index into the memarg
//! offset rather than emitting `i32.add` + `i32.store offset=0`,
//! so the symbolic stack arrives at the store with just a bare
//! `BufferPtr(slot)` and a non-zero memarg offset. The lowerer
//! now decodes the offset back into a const index.

#![cfg(feature = "software")]

#[quanta::kernel(workgroup = [1])]
fn write_zero(out: &mut [u32]) {
    // `out[0] = 42` — rustc emits this as
    //   `local.get out; i32.const 42; i32.store offset=0`
    // i.e. a bare BufferPtr stack arg, no index-arithmetic op.
    out[0] = 42u32;
}

#[quanta::kernel(workgroup = [1])]
fn write_constant_index(out: &mut [u32]) {
    // `out[3] = 7` — rustc folds the index into memarg
    //   `local.get out; i32.const 7; i32.store offset=12`
    out[3] = 7u32;
}

#[quanta::kernel(workgroup = [1])]
fn read_constant_index(input: &[u32], out: &mut [u32]) {
    // Mix: read at a constant index (loads `input[2]` as
    // `i32.load offset=8`) and write at index 0.
    let v = input[2];
    out[0] = v;
}

#[quanta::kernel(workgroup = [1])]
fn write_zero_f32(out: &mut [f32]) {
    // f32 mirror image — `out[0] = 1.5` → `f32.store offset=0`.
    out[0] = 1.5f32;
}

#[test]
fn buffer_ptr_zero_offset_u32_store() {
    let gpu = quanta::init_cpu();
    let out = gpu.field::<u32>(4).unwrap();
    out.write(&[0u32; 4]).unwrap();

    let mut wave = write_zero(&gpu).unwrap();
    wave.bind(0, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    assert_eq!(out.read().unwrap(), vec![42, 0, 0, 0]);
}

#[test]
fn buffer_ptr_constant_index_u32_store() {
    let gpu = quanta::init_cpu();
    let out = gpu.field::<u32>(4).unwrap();
    out.write(&[0u32; 4]).unwrap();

    let mut wave = write_constant_index(&gpu).unwrap();
    wave.bind(0, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    assert_eq!(out.read().unwrap(), vec![0, 0, 0, 7]);
}

#[test]
fn buffer_ptr_constant_index_u32_load_and_store() {
    let gpu = quanta::init_cpu();
    let input = gpu.field::<u32>(4).unwrap();
    let out = gpu.field::<u32>(4).unwrap();
    input.write(&[10, 20, 30, 40]).unwrap();
    out.write(&[0u32; 4]).unwrap();

    let mut wave = read_constant_index(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    assert_eq!(out.read().unwrap()[0], 30, "out[0] must hold input[2]");
}

#[test]
fn buffer_ptr_zero_offset_f32_store() {
    let gpu = quanta::init_cpu();
    let out = gpu.field::<f32>(4).unwrap();
    out.write(&[0.0f32; 4]).unwrap();

    let mut wave = write_zero_f32(&gpu).unwrap();
    wave.bind(0, &out);
    gpu.dispatch(&wave, 1).unwrap().wait().unwrap();

    let actual = out.read().unwrap();
    assert_eq!(actual[0], 1.5);
    assert!(actual[1..].iter().all(|&v| v == 0.0));
}

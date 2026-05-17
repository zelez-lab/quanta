//! Smoke test for the subgroup intrinsics declared in
//! `src/intrinsics.rs` and lowered in `quanta-wasm-lowering`.
//!
//! Confirms each intrinsic compiles through the WASM route and
//! produces a sane value end-to-end. Real cross-backend
//! correctness comparison is the differential suite's job; this
//! test just exercises the compile + dispatch path.
//!
//! Skips gracefully when no GPU backend is available.

//! Note: the subgroup intrinsics (`subgroup_size`, `reduce_add_u32`,
//! etc.) live in the kernel-only `quanta` wasm-import namespace.
//! The `#[quanta::kernel]` macro injects them; they are NOT
//! accessible from host code, so this test only invokes them from
//! inside kernel bodies.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn k_subgroup_size(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = unsafe { subgroup_size() };
}

#[quanta::kernel]
fn k_subgroup_id(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = unsafe { subgroup_id() };
}

#[quanta::kernel]
fn k_reduce_add(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_add_u32(v) };
}

#[quanta::kernel]
fn k_scan_add(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { scan_add_u32(v) };
}

#[quanta::kernel]
fn k_shuffle(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { shuffle_u32(v, 0) };
}

#[quanta::kernel]
fn k_ballot(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let predicate = if values[i as usize] != 0 { 1u32 } else { 0u32 };
    out[i as usize] = unsafe { ballot_u32(predicate) };
}

#[quanta::kernel]
fn k_any(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let predicate = if values[i as usize] != 0 { 1u32 } else { 0u32 };
    out[i as usize] = unsafe { any_u32(predicate) };
}

#[quanta::kernel]
fn k_all(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let predicate = if values[i as usize] != 0 { 1u32 } else { 0u32 };
    out[i as usize] = unsafe { all_u32(predicate) };
}

#[test]
fn subgroup_size_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_subgroup_size(&gpu).unwrap();
    wave.bind(0, &out);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let result = out.read().unwrap();
    let s = result[0];
    assert!(s >= 1 && (s & (s - 1)) == 0, "subgroup_size = {s}");
}

#[test]
fn subgroup_id_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_subgroup_id(&gpu).unwrap();
    wave.bind(0, &out);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn reduce_add_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[1u32, 2, 3, 4]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_reduce_add(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn scan_add_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[1u32, 2, 3, 4]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_scan_add(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn shuffle_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[10u32, 20, 30, 40]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_shuffle(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn ballot_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[1u32, 0, 1, 1]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_ballot(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn any_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[1u32, 0, 1, 0]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_any(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

#[test]
fn all_compiles_and_runs() {
    let Some(gpu) = try_gpu() else { return };
    let out = gpu.field::<u32>(4).unwrap();
    let values = gpu.field::<u32>(4).unwrap();
    values.write(&[1u32, 1, 1, 1]).unwrap();
    out.write(&[0u32; 4]).unwrap();
    let mut wave = k_all(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &values);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();
    let _result = out.read().unwrap();
}

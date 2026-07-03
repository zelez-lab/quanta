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

// ── Emitted-SPIR-V validity gate ────────────────────────────────────────
//
// The build-time `spirv-val` gate in the compiler only *logs* invalid
// modules; these assertions make invalid subgroup SPIR-V a hard test
// failure. Any/All additionally require the GroupNonUniformVote
// capability (SPIR-V §3.31, value 62) — asserted structurally so the
// check bites even on machines without spirv-val installed.

const OP_CAPABILITY: u32 = 17;
const CAP_GROUP_NON_UNIFORM_VOTE: u32 = 62;
const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;

fn spirv_words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn has_capability(w: &[u32], cap: u32) -> bool {
    let mut i = 5; // skip header
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        if op == OP_CAPABILITY && w[i + 1..i + wc] == [cap] {
            return true;
        }
        i += wc;
    }
    false
}

/// Run `spirv-val --target-env vulkan1.3` and assert the module passes.
/// Skips silently (like the build-time gate) when spirv-val isn't
/// installed.
fn assert_spirv_val_clean(name: &str, spirv: &[u8]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => return, // spirv-val not on PATH
    };
    child.stdin.as_mut().unwrap().write_all(spirv).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{name}: emitted SPIR-V is invalid (spirv-val):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn subgroup_spirv_modules_validate() {
    for (name, binary) in [
        ("k_subgroup_size", &K_SUBGROUP_SIZE_BINARY),
        ("k_subgroup_id", &K_SUBGROUP_ID_BINARY),
        ("k_reduce_add", &K_REDUCE_ADD_BINARY),
        ("k_scan_add", &K_SCAN_ADD_BINARY),
        ("k_shuffle", &K_SHUFFLE_BINARY),
        ("k_ballot", &K_BALLOT_BINARY),
        ("k_any", &K_ANY_BINARY),
        ("k_all", &K_ALL_BINARY),
    ] {
        let spirv = binary
            .spirv
            .unwrap_or_else(|| panic!("{name}: no SPIR-V embedded"));
        assert_spirv_val_clean(name, spirv);
    }
}

#[test]
fn any_all_declare_vote_capability() {
    for (name, binary) in [("k_any", &K_ANY_BINARY), ("k_all", &K_ALL_BINARY)] {
        let spirv = binary
            .spirv
            .unwrap_or_else(|| panic!("{name}: no SPIR-V embedded"));
        assert!(
            has_capability(&spirv_words(spirv), CAP_GROUP_NON_UNIFORM_VOTE),
            "{name}: OpGroupNonUniformAny/All requires the GroupNonUniformVote capability"
        );
    }
}

#[test]
fn ballot_declares_ballot_capability() {
    let spirv = K_BALLOT_BINARY.spirv.expect("k_ballot: no SPIR-V embedded");
    assert!(
        has_capability(&spirv_words(spirv), CAP_GROUP_NON_UNIFORM_BALLOT),
        "k_ballot: OpGroupNonUniformBallot requires the GroupNonUniformBallot capability"
    );
}

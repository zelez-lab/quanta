//! Tests for the WASM-local register renaming fix in
//! `crates/quanta-wasm-lowering`. See workspace-level design doc
//! at `quanta_project/roadmap/_design/wasm_local_renaming.md`.
//!
//! Three patterns must hold on real Metal:
//!
//! 1. **Local recycling** — rustc may reuse a wasm-local across
//!    SSA-disjoint values. The lowerer must not let the second
//!    write clobber the first's reads.
//! 2. **Mutable accumulator** — a single logical Rust variable
//!    written multiple times must keep working. The fix can't
//!    regress this.
//! 3. **Nested if/else** — different branches set a local
//!    differently; the post-join read must see the right value.
//!
//! All three skip gracefully on no-GPU machines (early `return`
//! when `quanta::init()` fails).

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ── Test 1: rustc-style local recycling ──────────────────────────
//
// The kernel binds `i = quark_id()` at the top, runs a loop that
// introduces a temp variable, and writes to `out[i]` at the end.
// rustc's optimizer may collapse `i`'s local into the temp's
// local — the WASM-route lowering must not let that corrupt the
// final write.

#[quanta::kernel(workgroup = [64])]
fn write_via_recycled_local(out: &mut [u32]) {
    let i = quark_id();
    let mut acc: u32 = 0u32;
    let mut k: u32 = 0u32;
    while k < 4u32 {
        let temp = k * 100u32;
        acc = acc + temp;
        k = k + 1u32;
    }
    out[i as usize] = acc;
}

#[test]
fn local_recycling_writes_correct_indices() {
    let Some(gpu) = try_gpu() else { return };
    let n = 64;
    let out = gpu.field::<u32>(n).unwrap();
    out.write(&vec![0u32; n]).unwrap();

    let mut wave = write_via_recycled_local(&gpu).unwrap();
    wave.bind(0, &out);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();

    let result = out.read().unwrap();
    // Every lane should write 0 + 0 + 100 + 200 + 300 = 600
    // to slot `lane`. Without the renaming fix, slot 0 gets the
    // value and the rest stay 0 (or vice versa, depending on
    // which local rustc recycled).
    let expected = vec![600u32; n];
    assert_eq!(result, expected);
}

// ── Test 2: mutable accumulator (must NOT regress) ───────────────

#[quanta::kernel(workgroup = [64])]
fn accumulator_through_loop(out: &mut [u32]) {
    let i = quark_id();
    let mut sum: u32 = 0u32;
    let mut k: u32 = 0u32;
    while k < 10u32 {
        sum = sum + k;
        k = k + 1u32;
    }
    out[i as usize] = sum;
}

#[test]
fn mutable_accumulator_still_works() {
    let Some(gpu) = try_gpu() else { return };
    let n = 64;
    let out = gpu.field::<u32>(n).unwrap();
    out.write(&vec![0u32; n]).unwrap();

    let mut wave = accumulator_through_loop(&gpu).unwrap();
    wave.bind(0, &out);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();

    let result = out.read().unwrap();
    // 0 + 1 + 2 + ... + 9 = 45 for every lane.
    let expected = vec![45u32; n];
    assert_eq!(result, expected);
}

// ── Test 3: nested if/else merge ─────────────────────────────────

// The dead `100u32` initial value is the point: the local must be
// renamed across the if/else merge even though both arms overwrite it.
#[quanta::kernel(workgroup = [64])]
fn nested_if(out: &mut [u32], flags: &[u32]) {
    let i = quark_id();
    #[allow(unused_assignments)]
    let mut v: u32 = 100u32;
    if flags[i as usize] != 0u32 {
        v = 200u32;
    } else {
        v = 300u32;
    }
    out[i as usize] = v;
}

#[test]
fn nested_if_merges_correctly() {
    let Some(gpu) = try_gpu() else { return };
    let n = 64;
    // Half-and-half pattern: even lanes get 200, odd get 300.
    let flags: Vec<u32> = (0..n as u32)
        .map(|i| if i % 2 == 0 { 1 } else { 0 })
        .collect();

    let out = gpu.field::<u32>(n).unwrap();
    let flags_field = gpu.field::<u32>(n).unwrap();
    flags_field.write(&flags).unwrap();
    out.write(&vec![0u32; n]).unwrap();

    let mut wave = nested_if(&gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &flags_field);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();

    let result = out.read().unwrap();
    let expected: Vec<u32> = (0..n as u32)
        .map(|i| if i % 2 == 0 { 200 } else { 300 })
        .collect();
    assert_eq!(result, expected);
}

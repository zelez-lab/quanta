//! Metal-backed integration test for `IndirectCommandBuffer` (steps
//! 032 + 033).
//!
//! Exercises the typed ICB API through the live Metal driver:
//! create → record_dispatch → execute_all → verify field state.
//!
//! Skips gracefully when no Metal device is available, when no
//! compiled kernel exists for vendor Apple (xcrun metal toolchain
//! not installed in CI), or when the GPU does not support
//! MTLIndirectCommandBuffer.
//!
//! Refines the same theorems the CPU `icb_basic` suite covers
//! (T7000 record-execute = direct, T7053 re-executable). The Metal
//! ICB recording path uses `indirectComputeCommandAtIndex` +
//! `concurrentDispatchThreadgroups`; execution uses
//! `executeCommandsInBuffer:withRange:`.

#[quanta::kernel]
fn icb_metal_add_one(data: &mut [u32]) {
    let i = quark_id();
    data[i] = data[i] + 1u32;
}

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init()
        .ok()
        .filter(|g| g.caps().vendor == quanta::Vendor::Apple)
}

#[test]
fn icb_metal_record_two_dispatches_matches_direct() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no Apple GPU available");
        return;
    };

    let count = 64;
    let initial: Vec<u32> = (0..count).collect();

    // ── Path A: two direct dispatches ──
    let field_a = gpu.field::<u32>(count as usize).unwrap();
    field_a.write(&initial).unwrap();
    let mut wave_a = match icb_metal_add_one(&gpu) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    wave_a.bind(0, &field_a);
    let mut p1 = gpu.dispatch(&wave_a, count as u32).unwrap();
    p1.wait().unwrap();
    let mut p2 = gpu.dispatch(&wave_a, count as u32).unwrap();
    p2.wait().unwrap();
    let direct = field_a.read().unwrap();

    // ── Path B: two recorded ICB dispatches ──
    let field_b = gpu.field::<u32>(count as usize).unwrap();
    field_b.write(&initial).unwrap();
    let mut wave_b = icb_metal_add_one(&gpu).unwrap();
    wave_b.bind(0, &field_b);
    let mut icb = match gpu.indirect_command_buffer(4) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("skipping: ICB create not supported: {e}");
            return;
        }
    };
    let groups = [count as u32 / wave_b.workgroup_size[0], 1, 1];
    if let Err(e) = icb.record_dispatch(&wave_b, groups) {
        eprintln!("skipping: ICB record not supported: {e}");
        return;
    }
    icb.record_dispatch(&wave_b, groups).unwrap();
    icb.execute_all().unwrap();
    let via_icb = field_b.read().unwrap();

    assert_eq!(via_icb, direct, "ICB ≠ direct");
    let expected: Vec<u32> = initial.iter().map(|x| x + 2).collect();
    assert_eq!(via_icb, expected);
}

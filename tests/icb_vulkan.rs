//! Vulkan-backed integration test for `IndirectCommandBuffer`
//! (steps 032 + 033).
//!
//! Refines the Lean `T7000` equivalence theorem on Vulkan: record N
//! dispatches into an ICB, execute_all, verify result matches direct
//! sequential dispatch. Skips when no Vulkan device is present.

#[quanta::kernel]
fn icb_vk_add_one(data: &mut [u32]) {
    let i = quark_id();
    data[i] = data[i] + 1u32;
}

fn try_vulkan_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok().filter(|g| {
        let v = g.caps().vendor;
        // Vulkan path: Amd / Nvidia / Intel / Broadcom / Unknown
        // (anything except Apple Metal and the software CPU fallback).
        !matches!(v, quanta::Vendor::Apple | quanta::Vendor::Software)
    })
}

#[test]
fn icb_vulkan_record_two_dispatches_matches_direct() {
    let Some(gpu) = try_vulkan_gpu() else {
        eprintln!("skipping: no Vulkan GPU available");
        return;
    };

    let count = 64;
    let initial: Vec<u32> = (0..count).collect();

    // Path A: two direct dispatches.
    let field_a = gpu.field::<u32>(count as usize).unwrap();
    field_a.write(&initial).unwrap();
    let mut wave_a = match icb_vk_add_one(&gpu) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    wave_a.bind(0, &field_a);
    let mut p1 = gpu.dispatch(&wave_a, count).unwrap();
    p1.wait().unwrap();
    let mut p2 = gpu.dispatch(&wave_a, count).unwrap();
    p2.wait().unwrap();
    let direct = field_a.read().unwrap();

    // Path B: two recorded ICB dispatches.
    let field_b = gpu.field::<u32>(count as usize).unwrap();
    field_b.write(&initial).unwrap();
    let mut wave_b = icb_vk_add_one(&gpu).unwrap();
    wave_b.bind(0, &field_b);
    let mut icb = match gpu.indirect_command_buffer(4) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("skipping: ICB create not supported: {e}");
            return;
        }
    };
    let groups = [count / wave_b.workgroup_size[0].max(1), 1, 1];
    icb.record_dispatch(&wave_b, groups).unwrap();
    icb.record_dispatch(&wave_b, groups).unwrap();
    icb.execute_all().unwrap();
    let via_icb = field_b.read().unwrap();

    assert_eq!(via_icb, direct, "ICB ≠ direct");
    let expected: Vec<u32> = initial.iter().map(|x| x + 2).collect();
    assert_eq!(via_icb, expected);
}

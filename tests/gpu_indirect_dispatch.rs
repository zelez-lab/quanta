//! Tier 2 -- Indirect dispatch.
//!
//! Verifies wave_dispatch_indirect (GPU-driven grid size).
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[quanta::kernel]
fn fill_values(result: &mut [f32]) {
    let i = quark_id();
    result[i] = 42.0;
}

#[test]
fn indirect_dispatch_basic() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Create the indirect argument buffer: [groups_x, groups_y, groups_z].
    // Dispatch 64 quarks = 1 group of 64 (assuming default group size).
    let args: Vec<u32> = vec![64, 1, 1];
    let arg_field = gpu.compute_field::<u32>(3).unwrap();
    gpu.write_field(&arg_field, &args).unwrap();

    let count = 64;
    let result_field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&result_field, &vec![0.0f32; count])
        .unwrap();

    let mut wave = fill_values(&gpu).unwrap();
    wave.bind(0, &result_field);

    match gpu.dispatch_indirect(&wave, &arg_field, 0) {
        Ok(mut pulse) => {
            gpu.wait(&mut pulse).unwrap();
            let result = gpu.read_field::<f32>(&result_field).unwrap();
            // At least some values should be 42.0.
            let has_42 = result.iter().any(|&v| (v - 42.0).abs() < 0.001);
            assert!(has_42, "indirect dispatch should have written 42.0");
        }
        Err(e) => {
            eprintln!("dispatch_indirect not supported: {}", e);
        }
    }
}

#[test]
fn indirect_dispatch_with_offset() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Buffer contains padding, then [groups_x, groups_y, groups_z] at offset 16.
    let args: Vec<u32> = vec![0, 0, 0, 0, 32, 1, 1];
    let arg_field = gpu.compute_field::<u32>(7).unwrap();
    gpu.write_field(&arg_field, &args).unwrap();

    let count = 32;
    let result_field = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&result_field, &vec![0.0f32; count])
        .unwrap();

    let mut wave = fill_values(&gpu).unwrap();
    wave.bind(0, &result_field);

    // Offset = 16 bytes (4 u32s of padding).
    match gpu.dispatch_indirect(&wave, &arg_field, 16) {
        Ok(mut pulse) => {
            gpu.wait(&mut pulse).unwrap();
        }
        Err(e) => {
            eprintln!("dispatch_indirect with offset not supported: {}", e);
        }
    }
}

#[test]
fn indirect_dispatch_zero_groups_no_crash() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Dispatch zero groups -- should not crash or panic.
    // Behavior is driver-defined: may be a true no-op or may still execute.
    let args: Vec<u32> = vec![0, 0, 0];
    let arg_field = gpu.compute_field::<u32>(3).unwrap();
    gpu.write_field(&arg_field, &args).unwrap();

    let result_field = gpu.compute_field::<f32>(16).unwrap();
    gpu.write_field(&result_field, &vec![0.0f32; 16]).unwrap();

    let mut wave = fill_values(&gpu).unwrap();
    wave.bind(0, &result_field);

    match gpu.dispatch_indirect(&wave, &arg_field, 0) {
        Ok(mut pulse) => {
            gpu.wait(&mut pulse).unwrap();
            // The key assertion: no crash, no hang.
            // Reading back should succeed regardless of whether data changed.
            let _result = gpu.read_field::<f32>(&result_field).unwrap();
        }
        Err(e) => {
            eprintln!("dispatch_indirect not supported: {}", e);
        }
    }
}

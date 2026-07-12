//! Compute texture read/write tests (step 055).
//!
//! Verifies that compute kernels can read textures via texture_sample_2d
//! and texture_load_2d.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Read texture via sampled load and write RGBA to output buffer.
#[quanta::kernel]
fn read_texture(tex: &Texture2D<f32>, output: &mut [f32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    output[i] = texture_load_2d(tex, x, y);
}

#[test]
fn compute_reads_texture() {
    let Some(gpu) = try_gpu() else {
        return;
    };
    // The Broadcom V3D Mesa driver faults on image-in-compute dispatch
    // (the emitted SPIR-V is valid — see texture_read_spirv_module_validates —
    // but V3DV's texture path segfaults). Skip the live dispatch there.
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }

    let w = 4u32;
    let h = 4u32;
    let n = (w * h) as usize;

    // Create a 4x4 RGBA8 texture with known data
    let mut tex_data = vec![0u8; n * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            tex_data[idx] = (x * 64) as u8; // R: 0, 64, 128, 192
            tex_data[idx + 1] = (y * 64) as u8; // G: 0, 64, 128, 192
            tex_data[idx + 2] = 0; // B
            tex_data[idx + 3] = 255; // A
        }
    }

    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .unwrap();
    tex.write(&tex_data).unwrap();

    let output = gpu.field::<f32>(n).unwrap();

    eprintln!("metallib: {}", READ_TEXTURE_BINARY.metallib.is_some());
    eprintln!("spirv: {}", READ_TEXTURE_BINARY.spirv.is_some());
    let mut wave = read_texture(&gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.bind(1, &output);
    wave.set_value(2, w);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    p.wait().unwrap();

    let result = output.read().unwrap();

    // Check first row: R channel should be 0/255, 64/255, 128/255, 192/255
    for (x, &actual) in result.iter().enumerate().take(w as usize) {
        let expected_r = (x as f32 * 64.0) / 255.0;
        let err = (actual - expected_r).abs();
        eprintln!(
            "  pixel ({x},0): R = {:.3} (expected {:.3}, err {:.3})",
            actual, expected_r, err
        );
        assert!(
            err < 0.01,
            "texture read at ({x},0): expected {expected_r:.3}, got {actual:.3}"
        );
    }
}

// ── Emitted-SPIR-V validity gate ────────────────────────────────────────
//
// The build-time `spirv-val` gate in the compiler only *logs* invalid
// modules; this assertion makes invalid texture-read SPIR-V (e.g. an
// OpCompositeConstruct whose constituents don't match the coordinate
// vector's component type) a hard test failure. Skips silently when
// spirv-val isn't installed, like the build-time gate.

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
fn texture_read_spirv_module_validates() {
    let spirv = READ_TEXTURE_BINARY
        .spirv
        .expect("read_texture: no SPIR-V embedded");
    assert_spirv_val_clean("read_texture", spirv);
}

/// Write a per-quark value into a storage texture. Exercises
/// `emit_op_texture_write_2d` — coords must coerce to i32 and the scalar
/// value to f32 before the vec4 texel, else the OpCompositeConstruct is
/// invalid (the same constituent-type class as the read path).
#[quanta::kernel]
fn write_texture(tex: &mut Texture2D<f32>, values: &[f32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    texture_write_2d(tex, x, y, values[i]);
}

#[test]
fn texture_write_spirv_module_validates() {
    let spirv = WRITE_TEXTURE_BINARY
        .spirv
        .expect("write_texture: no SPIR-V embedded");
    assert_spirv_val_clean("write_texture", spirv);
}

// ── Live storage-image dispatch (write / read-modify-write / format guard) ──
//
// These run on whichever device `init()` selects (Metal here, CPU under
// QUANTA_CPU=1) plus an explicit CPU pass, and skip when the backend reports
// no compute-texture support. Vulkan live dispatch runs only in CI (lavapipe).

/// Write f(x,y) = x*10 + y into an R32Float storage texture. Pure write path.
#[quanta::kernel]
fn write_pattern(tex: &mut Texture2D<f32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    let v = (x * 10 + y) as f32;
    texture_write_2d(tex, x, y, v);
}

/// Read-modify-write the SAME R32Float storage texture: v ← v*2 + 1. Exercises
/// decision 1's read_write semantics — `texture_load_2d` against a `&mut`
/// storage slot lowers to a storage read (OpImageRead / .read()), not a
/// sampled fetch. Each thread owns one texel, so there is no cross-thread
/// hazard within the dispatch.
#[quanta::kernel]
fn rmw_texture(tex: &mut Texture2D<f32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    let cur = texture_load_2d(tex, x, y);
    texture_write_2d(tex, x, y, cur * 2.0 + 1.0);
}

fn r32f_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn r32f_read(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_write_pattern(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::R32Float).with_usage(
                quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::SHADER_WRITE),
            ),
        )
        .unwrap();

    let mut wave = write_pattern(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.set_value(1, w);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();

    let got = r32f_read(&tex.read().unwrap());
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let expected = (x * 10 + y) as f32;
            assert_eq!(
                got[idx], expected,
                "write_pattern texel ({x},{y}) = {} (expected {expected})",
                got[idx]
            );
        }
    }
}

fn run_rmw(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    let seed: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::R32Float).with_usage(
                quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::SHADER_WRITE),
            ),
        )
        .unwrap();
    tex.write(&r32f_bytes(&seed)).unwrap();

    let mut wave = rmw_texture(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.set_value(1, w);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();

    let got = r32f_read(&tex.read().unwrap());
    for i in 0..n {
        let expected = seed[i] * 2.0 + 1.0;
        assert_eq!(
            got[i], expected,
            "rmw texel {i} = {} (expected {expected})",
            got[i]
        );
    }
}

#[test]
fn compute_writes_storage_texture() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return; // V3DV image-in-compute path faults; SPIR-V is valid.
    }
    run_write_pattern(&gpu);
}

#[test]
fn compute_read_modify_writes_storage_texture() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }
    run_rmw(&gpu);
}

/// The CPU software executor must produce identical texels (validates W6
/// independently of the default device).
#[cfg(feature = "software")]
#[test]
fn cpu_writes_and_rmw_storage_texture() {
    let gpu = quanta::init_cpu();
    if !gpu.supports_compute_textures() {
        return;
    }
    run_write_pattern(&gpu);
    run_rmw(&gpu);
}

/// The scalar-driven format contract: binding an RGBA8 texture to an
/// `&mut Texture2D<f32>` (R32Float) storage slot is InvalidParam, on every
/// backend that can see both the registry and the reflected/param kinds.
#[test]
fn format_mismatch_is_invalid_param() {
    fn check(gpu: &quanta::Gpu) {
        if !gpu.supports_compute_textures() {
            return;
        }
        if gpu.caps().vendor == quanta::Vendor::Broadcom {
            return;
        }
        let (w, h) = (4u32, 4u32);
        let n = (w * h) as usize;
        let tex = gpu
            .create_texture(
                &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8).with_usage(
                    quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::SHADER_WRITE),
                ),
            )
            .unwrap();
        let mut wave = write_pattern(gpu).unwrap();
        wave.bind_texture(0, &tex);
        wave.set_value(1, w);
        let e = gpu
            .dispatch(&wave, n as u32)
            .err()
            .expect("binding RGBA8 to an f32 storage slot must fail");
        assert!(
            matches!(e.kind, quanta::QuantaErrorKind::InvalidParam(_)),
            "expected InvalidParam, got {:?}",
            e.kind
        );
    }
    if let Some(gpu) = try_gpu() {
        check(&gpu);
    }
    #[cfg(feature = "software")]
    check(&quanta::init_cpu());
}

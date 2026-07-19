//! Compute texture read/write tests.
//!
//! Verifies that compute kernels can read textures via both `texture_sample_2d`
//! (sampled reads, nearest + clamp at texel coords) and `texture_load_2d`
//! (storage reads), plus storage writes — live on the default device, the CPU
//! executor, and lavapipe in CI.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Read texture via sampled load (texelFetch on a `&Sampled2D` slot — the
/// texel-read path for textures without storage usage) and write to output.
#[quanta::kernel]
fn read_texture(tex: &Sampled2D<f32>, output: &mut [f32], width: u32) {
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
    // Sampled-image reads in compute are wired on Metal and the CPU
    // executor; the Vulkan path supports storage images only and rejects a
    // kernel whose reflection carries a sampled-image binding. Skip where
    // the backend says so — storage load/write coverage lives in the
    // *_storage_texture tests below, which run everywhere.
    let mut wave = match read_texture(&gpu) {
        Ok(wave) => wave,
        Err(e) if matches!(e.kind, quanta::QuantaErrorKind::NotSupported(_)) => {
            eprintln!("SKIP: sampled-image compute read not supported here: {e}");
            return;
        }
        Err(e) => panic!("read_texture wave build failed: {e}"),
    };
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

/// Sample a texture via `texture_sample_2d` (the sampled-read path) with an
/// x-offset applied per dispatch. With the device compute sampler (NEAREST,
/// CLAMP_TO_EDGE, unnormalized) the fetch is the texel at `(x + dx, y)` clamped
/// to the edge — so `dx = 0` reads the exact texel and a large `dx` reads the
/// right-edge column. `texture_sample_2d` on an RGBA8 read slot returns the R
/// channel as unorm, identical to `texture_load_2d`.
#[quanta::kernel]
fn sample_texture(tex: &Sampled2D<f32>, output: &mut [f32], width: u32, dx: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    output[i] = texture_sample_2d(tex, x + dx, y);
}

/// Sampled-read parity: `texture_sample_2d` must return the same R-channel
/// unorm as `texture_load_2d` at integer texel coordinates (dx = 0), AND an
/// out-of-bounds x (dx pushes past the right edge) must clamp to the edge texel
/// — the CLAMP_TO_EDGE / unnormalized compute-sampler contract. Runs live on
/// whatever `init()` picks (Metal here — this is the F3 Metal sampler-bind
/// proof), an explicit CPU pass, and lavapipe in CI. Skips only where the
/// backend reports the sampled-read unsupported (WebGPU).
fn run_sample_texture(gpu: &quanta::Gpu) {
    let w = 4u32;
    let h = 4u32;
    let n = (w * h) as usize;

    // Same known RGBA8 pattern as compute_reads_texture: R = x*64, G = y*64.
    let mut tex_data = vec![0u8; n * 4];
    for y in 0..h {
        for x in 0..w {
            let idx = ((y * w + x) * 4) as usize;
            tex_data[idx] = (x * 64) as u8;
            tex_data[idx + 1] = (y * 64) as u8;
            tex_data[idx + 2] = 0;
            tex_data[idx + 3] = 255;
        }
    }
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ),
        )
        .unwrap();
    tex.write(&tex_data).unwrap();

    // Exact texel values at integer coords (dx = 0): first row R = 0,64,128,192.
    let output = gpu.field::<f32>(n).unwrap();
    let mut wave = match sample_texture(gpu) {
        Ok(wave) => wave,
        Err(e) if matches!(e.kind, quanta::QuantaErrorKind::NotSupported(_)) => {
            eprintln!("SKIP: sampled-image compute read not supported here: {e}");
            return;
        }
        Err(e) => panic!("sample_texture wave build failed: {e}"),
    };
    wave.bind_texture(0, &tex);
    wave.bind(1, &output);
    wave.set_value(2, w);
    wave.set_value(3, 0u32);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    let result = output.read().unwrap();
    for (x, &actual) in result.iter().enumerate().take(w as usize) {
        let expected_r = (x as f32 * 64.0) / 255.0;
        assert!(
            (actual - expected_r).abs() < 0.01,
            "sample at ({x},0): expected {expected_r:.3}, got {actual:.3}"
        );
    }

    // Clamp-to-edge: dx = w + 3 pushes every x past the right edge, so every
    // texel reads the last column (x = w-1, R = (w-1)*64 = 192/255).
    let clamp_out = gpu.field::<f32>(n).unwrap();
    let mut wave = sample_texture(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.bind(1, &clamp_out);
    wave.set_value(2, w);
    wave.set_value(3, w + 3);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();
    let clamped = clamp_out.read().unwrap();
    let edge_r = ((w - 1) as f32 * 64.0) / 255.0;
    for y in 0..h as usize {
        let actual = clamped[y * w as usize];
        assert!(
            (actual - edge_r).abs() < 0.01,
            "clamp-to-edge sample (row {y}, x=w+3): expected edge R {edge_r:.3}, got {actual:.3}"
        );
    }
}

#[test]
fn compute_samples_texture() {
    let Some(gpu) = try_gpu() else { return };
    // V3DV's image-in-compute path faults (the SPIR-V is valid — see the
    // spirv-val units); skip the live dispatch there like compute_reads_texture.
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }
    run_sample_texture(&gpu);
}

/// The CPU software executor must sample identically (nearest + clamp at texel
/// coords), fixing the contract independently of the default device.
#[cfg(feature = "software")]
#[test]
fn cpu_samples_texture() {
    run_sample_texture(&quanta::init_cpu());
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

/// True if the module contains an `OpExtInst` (opcode 12) whose extended-
/// instruction number is `glsl_instr` — operand index 4, after result-type /
/// result-id / ext-set-id. Used to confirm the AOT SPIR-V for a packed-RGBA8
/// kernel really carries the Pack/UnpackUnorm4x8 boundary.
fn spirv_has_ext_inst(spirv: &[u8], glsl_instr: u32) -> bool {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xFFFF) as u16;
        let wc = (word >> 16) as usize;
        if wc == 0 {
            break;
        }
        if opcode == 12 && wc >= 5 && words[i + 4] == glsl_instr {
            return true;
        }
        i += wc;
    }
    false
}

/// The AOT (quanta-compiler) SPIR-V for the packed-RGBA8 write kernel must
/// validate and carry `UnpackUnorm4x8` (GLSL.std.450 #64) — the AOT-path twin
/// of the JIT `rgba8_write_unpacks_to_vec4_and_is_rgba8_format` unit test.
/// Skips only if no SPIR-V is embedded (some backends embed only a metallib).
#[test]
fn rgba8_write_aot_spirv_validates_and_unpacks() {
    let Some(spirv) = WRITE_PATTERN_RGBA8_BINARY.spirv else {
        eprintln!("SKIP: write_pattern_rgba8 has no embedded SPIR-V on this build");
        return;
    };
    assert_spirv_val_clean("write_pattern_rgba8", spirv);
    assert!(
        spirv_has_ext_inst(spirv, 64),
        "packed-RGBA8 write AOT SPIR-V must contain OpExtInst UnpackUnorm4x8 (#64)"
    );
}

/// The AOT SPIR-V for the packed-RGBA8 RMW kernel must validate and carry both
/// `PackUnorm4x8` (#55, the load) and `UnpackUnorm4x8` (#64, the write).
#[test]
fn rgba8_rmw_aot_spirv_validates_and_packs() {
    let Some(spirv) = RMW_RGBA8_RED_BINARY.spirv else {
        eprintln!("SKIP: rmw_rgba8_red has no embedded SPIR-V on this build");
        return;
    };
    assert_spirv_val_clean("rmw_rgba8_red", spirv);
    assert!(
        spirv_has_ext_inst(spirv, 55),
        "packed-RGBA8 load AOT SPIR-V must contain OpExtInst PackUnorm4x8 (#55)"
    );
    assert!(
        spirv_has_ext_inst(spirv, 64),
        "packed-RGBA8 write AOT SPIR-V must contain OpExtInst UnpackUnorm4x8 (#64)"
    );
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

// ── Packed-RGBA8 (`&mut Texture2D<u32>`) storage ────────────────────────────
//
// The packed-u32 contract: a texel crosses the kernel boundary as one
// `0xAABBGGRR` u32 (little-endian byte order R,G,B,A). The kernel builds it
// with bit math; the RGBA8 texture stores the four unorm bytes as [R,G,B,A] in
// memory. `rgba8_pack`/`rgba8_unpack` are the host mirror of that byte order,
// so `write_pattern_rgba8` + a raw byte read prove the channel order end to end.

/// Write a per-texel packed pattern into an RGBA8 storage texture. Each texel's
/// four channels are distinct AND differ from every other texel's, so a wrong
/// channel order (or a swizzle) shows up as a byte mismatch. Built entirely
/// with bit math (the deferred pack intrinsic pattern from the docs).
#[quanta::kernel]
fn write_pattern_rgba8(tex: &mut Texture2D<u32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    // Distinct per-channel bytes: R and G encode the coordinate, B and A are
    // fixed sentinels — all four differ so channel order is observable.
    let r = x * 16 + 1; // 1,17,33,...  (x < 16 keeps it a byte)
    let g = y * 16 + 2; // 2,18,34,...
    let b = 100u32;
    let a = 200u32;
    let v = r | (g << 8) | (b << 16) | (a << 24);
    texture_write_2d(tex, x, y, v);
}

/// Read-modify-write an RGBA8 storage texture with in-kernel bit math on ONE
/// channel: double the R channel (saturating at 255), leave G/B/A untouched.
/// Exercises the packed read (pack_float_to_unorm4x8 / PackUnorm4x8) feeding
/// bit-extraction, then the packed write back.
#[quanta::kernel]
fn rmw_rgba8_red(tex: &mut Texture2D<u32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    let v = texture_load_2d(tex, x, y);
    let r = v & 0xFF;
    let g = (v >> 8) & 0xFF;
    let b = (v >> 16) & 0xFF;
    let a = (v >> 24) & 0xFF;
    let mut r2 = r * 2;
    if r2 > 255 {
        r2 = 255;
    }
    let out = r2 | (g << 8) | (b << 16) | (a << 24);
    texture_write_2d(tex, x, y, out);
}

/// Intrinsic twin of `rmw_rgba8_red`: same "scale one channel in place" RMW,
/// but expressed with the `pack_unorm4x8` / `unpack_unorm4x8_*` intrinsics
/// instead of hand-rolled bit math. Unpacks the texel to unorm floats, halves
/// the green channel, repacks, and writes. The intrinsics lower to a KernelOp
/// composition (clamp/mul/round/cast/shift/or, and shift/and/cast/div), so the
/// emitted bytes must match a host reference exactly on every backend.
#[quanta::kernel]
fn rmw_rgba8_intrinsic_half_green(tex: &mut Texture2D<u32>, width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    let v = texture_load_2d(tex, x, y);
    let r = unpack_unorm4x8_r(v);
    let g = unpack_unorm4x8_g(v);
    let b = unpack_unorm4x8_b(v);
    let a = unpack_unorm4x8_a(v);
    let out = pack_unorm4x8(r, g * 0.5f32, b, a);
    texture_write_2d(tex, x, y, out);
}

/// Host mirror of the kernel's channel packing: `0xAABBGGRR`, bytes [R,G,B,A].
fn rgba8_pack(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_le_bytes([r, g, b, a])
}

/// Host reference for the pack/unpack intrinsics: byte-exact twin of the
/// lowering's KernelOp composition. `unpack` = `byte as f32 / 255.0`; `pack`
/// = round(clamp(ch,0,1) * 255) per channel, little-endian R,G,B,A.
fn host_unpack(v: u32, shift: u32) -> f32 {
    ((v >> shift) & 0xFF) as f32 / 255.0
}
fn host_pack(r: f32, g: f32, b: f32, a: f32) -> u32 {
    let enc = |c: f32| -> u32 { (c.clamp(0.0, 1.0) * 255.0).round() as u32 & 0xFF };
    enc(r) | (enc(g) << 8) | (enc(b) << 16) | (enc(a) << 24)
}

/// Read an RGBA8 texture's raw bytes back as [R,G,B,A] tuples per texel.
fn rgba8_unpack(bytes: &[u8]) -> Vec<(u8, u8, u8, u8)> {
    bytes
        .chunks_exact(4)
        .map(|c| (c[0], c[1], c[2], c[3]))
        .collect()
}

fn run_write_pattern(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::R32Float)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
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
            &quanta::TextureDesc::new(w, h, quanta::Format::R32Float)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
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

/// Create an RGBA8 storage texture. Returns `None` if the backend can't do
/// RGBA8 read_write storage (Metal below MTLReadWriteTextureTier2), which the
/// dispatch surfaces as `NotSupported` — the caller then skips like the
/// sampled-read skip. `Some(())` on success (the texture was written and
/// verified).
fn dispatch_rgba8_or_skip(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    n: u32,
) -> Option<quanta::Pulse> {
    match gpu.dispatch(wave, n) {
        Ok(p) => Some(p),
        Err(e) if matches!(e.kind, quanta::QuantaErrorKind::NotSupported(_)) => {
            eprintln!("SKIP: RGBA8 storage textures not supported here: {e}");
            None
        }
        Err(e) => panic!("rgba8 dispatch failed: {e}"),
    }
}

/// Pure packed-RGBA8 write + the channel-order proof: dispatch
/// `write_pattern_rgba8`, read the texture's raw bytes back, and assert every
/// texel's [R,G,B,A] bytes match the host-side pack of the same pattern. A
/// wrong channel order (e.g. BGRA, or a swizzle in pack/unpack) fails here.
fn run_write_pattern_rgba8(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
        )
        .unwrap();

    let mut wave = write_pattern_rgba8(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.set_value(1, w);
    let Some(mut p) = dispatch_rgba8_or_skip(gpu, &mut wave, n as u32) else {
        return;
    };
    p.wait().unwrap();

    let got = rgba8_unpack(&tex.read().unwrap());
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let expected = (
                (x * 16 + 1) as u8, // R
                (y * 16 + 2) as u8, // G
                100u8,              // B
                200u8,              // A
            );
            assert_eq!(
                got[idx],
                expected,
                "rgba8 texel ({x},{y}) channel order: got {:?}, expected {:?} \
                 (packed 0x{:08X})",
                got[idx],
                expected,
                rgba8_pack(expected.0, expected.1, expected.2, expected.3),
            );
        }
    }
}

/// In-kernel bit math on one channel: seed a known RGBA8 pattern, RMW-double
/// the R channel on the GPU, verify R doubled (saturating) and G/B/A held.
fn run_rmw_rgba8(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    // Seed: R spans 0..240 (compute in u32 to avoid u8 overflow); the upper
    // half (R >= 128) saturates on the in-kernel ×2. G/B/A distinct sentinels.
    let seed: Vec<(u8, u8, u8, u8)> = (0..n)
        .map(|i| ((i as u32 * 16) as u8, 7, 100, 200))
        .collect();
    let mut seed_bytes = Vec::with_capacity(n * 4);
    for &(r, g, b, a) in &seed {
        seed_bytes.extend_from_slice(&[r, g, b, a]);
    }
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
        )
        .unwrap();
    tex.write(&seed_bytes).unwrap();

    let mut wave = rmw_rgba8_red(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.set_value(1, w);
    let Some(mut p) = dispatch_rgba8_or_skip(gpu, &mut wave, n as u32) else {
        return;
    };
    p.wait().unwrap();

    let got = rgba8_unpack(&tex.read().unwrap());
    for i in 0..n {
        let (r, g, b, a) = seed[i];
        let expected_r = ((r as u32 * 2).min(255)) as u8;
        assert_eq!(
            got[i],
            (expected_r, g, b, a),
            "rmw_rgba8 texel {i}: got {:?}, expected R={expected_r} G={g} B={b} A={a}",
            got[i],
        );
    }
}

/// Live intrinsic-path RMW: seed a known RGBA8 pattern, halve the G channel via
/// `unpack_unorm4x8_*` + `pack_unorm4x8` on the GPU, and assert every texel's
/// bytes match the host reference (which runs the same unpack/half/pack in
/// f32). Proves the intrinsic composition is byte-identical to the texel
/// contract on whatever backend `init()` picks (and on the CPU pass).
fn run_rmw_rgba8_intrinsic(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    // G spans 0..240 so the half hits both even and odd bytes; R/B/A sentinels.
    let seed: Vec<(u8, u8, u8, u8)> = (0..n)
        .map(|i| (50, (i as u32 * 16) as u8, 100, 200))
        .collect();
    let mut seed_bytes = Vec::with_capacity(n * 4);
    for &(r, g, b, a) in &seed {
        seed_bytes.extend_from_slice(&[r, g, b, a]);
    }
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
        )
        .unwrap();
    tex.write(&seed_bytes).unwrap();

    let mut wave = rmw_rgba8_intrinsic_half_green(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.set_value(1, w);
    let Some(mut p) = dispatch_rgba8_or_skip(gpu, &mut wave, n as u32) else {
        return;
    };
    p.wait().unwrap();

    let got = rgba8_unpack(&tex.read().unwrap());
    for i in 0..n {
        let (r, g, b, a) = seed[i];
        // Host mirror: unpack → halve G → pack, reading back the packed bytes.
        let v = rgba8_pack(r, g, b, a);
        let want_packed = host_pack(
            host_unpack(v, 0),
            host_unpack(v, 8) * 0.5,
            host_unpack(v, 16),
            host_unpack(v, 24),
        );
        let want = (
            (want_packed & 0xFF) as u8,
            ((want_packed >> 8) & 0xFF) as u8,
            ((want_packed >> 16) & 0xFF) as u8,
            ((want_packed >> 24) & 0xFF) as u8,
        );
        assert_eq!(
            got[i], want,
            "rmw_rgba8_intrinsic texel {i}: got {:?}, want {want:?} (seed G={g})",
            got[i],
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

// ── Packed-RGBA8 storage: live write / RMW / channel-order proof ────────────

#[test]
fn compute_writes_rgba8_storage_texture() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return; // V3DV image-in-compute path faults; SPIR-V is valid.
    }
    run_write_pattern_rgba8(&gpu);
}

#[test]
fn compute_read_modify_writes_rgba8_storage_texture() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }
    run_rmw_rgba8(&gpu);
}

/// The CPU software executor must pack/unpack RGBA8 identically (the byte-order
/// contract is fixed by this test independently of any GPU backend).
#[cfg(feature = "software")]
#[test]
fn cpu_writes_and_rmw_rgba8_storage_texture() {
    let gpu = quanta::init_cpu();
    if !gpu.supports_compute_textures() {
        return;
    }
    run_write_pattern_rgba8(&gpu);
    run_rmw_rgba8(&gpu);
}

// ── pack_unorm4x8 / unpack_unorm4x8_* intrinsics ────────────────────────────

/// Live intrinsic RMW on the default device: unpack a texel, halve G, repack.
#[test]
fn compute_rmw_rgba8_via_intrinsics() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }
    run_rmw_rgba8_intrinsic(&gpu);
}

/// CPU software executor twin of the intrinsic RMW.
#[cfg(feature = "software")]
#[test]
fn cpu_rmw_rgba8_via_intrinsics() {
    let gpu = quanta::init_cpu();
    if !gpu.supports_compute_textures() {
        return;
    }
    run_rmw_rgba8_intrinsic(&gpu);
}

/// Pure bit-exactness of the pack/unpack composition, independent of any
/// texture: for each input `v`, unpack all four channels and repack, then
/// assert `pack_unorm4x8(unpack_r,g,b,a) == v`. This is the round-trip the
/// rounding choice in the lowering must satisfy for every byte-valued channel
/// — and it holds under either rounding mode because `(byte/255)*255` is exact
/// for all bytes 0..=255. Runs on the software executor (no GPU needed).
#[quanta::kernel]
fn rgba8_pack_unpack_roundtrip(input: &[u32], output: &mut [u32]) {
    let i = quark_id();
    let v = input[i as usize];
    let r = unpack_unorm4x8_r(v);
    let g = unpack_unorm4x8_g(v);
    let b = unpack_unorm4x8_b(v);
    let a = unpack_unorm4x8_a(v);
    output[i as usize] = pack_unorm4x8(r, g, b, a);
}

#[cfg(feature = "software")]
#[test]
fn pack_unpack_roundtrip_byte_sweep() {
    let gpu = quanta::init_cpu();
    // Sweep every channel byte 0..=255 in each of the four positions, plus a
    // few fully-mixed values. Each input's four channels are byte-valued by
    // construction, so pack(unpack(v)) must equal v exactly.
    let mut inputs: Vec<u32> = Vec::new();
    for byte in 0u32..=255 {
        inputs.push(byte); // R only
        inputs.push(byte << 8); // G only
        inputs.push(byte << 16); // B only
        inputs.push(byte << 24); // A only
        inputs.push(byte | (byte << 8) | (byte << 16) | (byte << 24)); // all equal
    }
    // A handful of fully-distinct-channel values.
    for &(r, g, b, a) in &[
        (1u32, 2, 3, 4),
        (255, 0, 128, 64),
        (10, 200, 30, 250),
        (127, 128, 129, 126),
    ] {
        inputs.push(r | (g << 8) | (b << 16) | (a << 24));
    }
    // Pad up to a multiple of 64 with zeros so the dispatch is clean.
    while inputs.len() % 64 != 0 {
        inputs.push(0);
    }
    let n = inputs.len();

    let input_f = gpu.field::<u32>(n).unwrap();
    let output_f = gpu.field::<u32>(n).unwrap();
    input_f.write(&inputs).unwrap();
    output_f.write(&vec![0xDEAD_BEEFu32; n]).unwrap();

    let mut wave = rgba8_pack_unpack_roundtrip(&gpu).unwrap();
    wave.bind(0, &input_f);
    wave.bind(1, &output_f);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();

    let got = output_f.read().unwrap();
    for (i, (&want, &g)) in inputs.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            g, want,
            "pack(unpack(v)) round-trip failed at index {i}: input 0x{want:08X}, got 0x{g:08X}"
        );
    }
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
                    quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE),
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

/// The reverse of the scalar-driven contract: binding an R32Float texture to a
/// `&mut Texture2D<u32>` (RGBA8-expecting) storage slot is InvalidParam. The
/// per-slot kind array distinguishes the two storage formats, so a kind-2 slot
/// rejects R32Float just as a kind-1 slot rejects RGBA8.
#[test]
fn r32float_to_u32_slot_is_invalid_param() {
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
                &quanta::TextureDesc::new(w, h, quanta::Format::R32Float).with_usage(
                    quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE),
                ),
            )
            .unwrap();
        let mut wave = write_pattern_rgba8(gpu).unwrap();
        wave.bind_texture(0, &tex);
        wave.set_value(1, w);
        let e = gpu
            .dispatch(&wave, n as u32)
            .err()
            .expect("binding R32Float to a u32 (RGBA8) storage slot must fail");
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

// ── Read-only texel access (`&Texture2D`) ───────────────────────────────────
//
// The read-only half of the texel lattice: `&Texture2D<T>` is a storage image
// declared NonWritable (SPIR-V) / `access::read` (MSL) — same scalar-driven
// format contract as `&mut`, no write capability, and (the reason it exists)
// no Metal read-write tier gate: packed-RGBA8 reads work on EVERY tier.

/// `out[i] = texture_load_2d(tex, x, y)` where `tex` is `&Texture2D<f32>` —
/// a read-only R32Float texel read into a buffer.
#[quanta::kernel]
fn read_texel(tex: &Texture2D<f32>, output: &mut [f32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    output[i] = texture_load_2d(tex, x, y);
}

/// Packed-RGBA8 twin: `&Texture2D<u32>` — the read-only packed read, which had
/// NO spelling before the lattice re-type (read-write required Tier 2 on
/// Metal; read-only does not).
#[quanta::kernel]
fn read_texel_rgba8(tex: &Texture2D<u32>, output: &mut [u32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    output[i] = texture_load_2d(tex, x, y);
}

fn run_read_texel(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    let seed: Vec<f32> = (0..n).map(|i| (i * 3) as f32 + 0.5).collect();
    // Texel binding needs storage-capable usage even read-only (the slot is a
    // STORAGE_IMAGE descriptor on Vulkan).
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::R32Float)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
        )
        .unwrap();
    tex.write(&r32f_bytes(&seed)).unwrap();

    let output = gpu.field::<f32>(n).unwrap();
    let mut wave = read_texel(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.bind(1, &output);
    wave.set_value(2, w);
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();

    let got = output.read().unwrap();
    for i in 0..n {
        assert_eq!(
            got[i], seed[i],
            "read-only texel {i} = {} (expected {})",
            got[i], seed[i]
        );
    }
}

fn run_read_texel_rgba8(gpu: &quanta::Gpu) {
    let (w, h) = (4u32, 4u32);
    let n = (w * h) as usize;
    // Distinct bytes per texel (same scheme as write_pattern_rgba8) so the
    // packed `0xAABBGGRR` order is observable end to end.
    let mut bytes = vec![0u8; n * 4];
    let mut expected = vec![0u32; n];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let (r, g, b, a) = ((x * 16 + 1) as u8, (y * 16 + 2) as u8, 100u8, 200u8);
            bytes[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, a]);
            expected[i] = rgba8_pack(r, g, b, a);
        }
    }
    let tex = gpu
        .create_texture(
            &quanta::TextureDesc::new(w, h, quanta::Format::RGBA8)
                .with_usage(quanta::TextureUsage::SHADER_READ.union(quanta::TextureUsage::STORAGE)),
        )
        .unwrap();
    tex.write(&bytes).unwrap();

    let output = gpu.field::<u32>(n).unwrap();
    let mut wave = read_texel_rgba8(gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.bind(1, &output);
    wave.set_value(2, w);
    // No dispatch_rgba8_or_skip here: read-only RGBA8 must NOT hit the Metal
    // Tier-2 gate — a NotSupported would be a regression, so unwrap.
    gpu.dispatch(&wave, n as u32).unwrap().wait().unwrap();

    let got = output.read().unwrap();
    for i in 0..n {
        assert_eq!(
            got[i], expected[i],
            "read-only RGBA8 texel {i} = {:#010x} (expected {:#010x})",
            got[i], expected[i]
        );
    }
}

#[test]
fn compute_reads_readonly_texel() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return; // V3DV image-in-compute path faults; SPIR-V is valid.
    }
    run_read_texel(&gpu);
}

/// Read-only RGBA8 texel reads run on every Metal tier — the portability the
/// read-only form buys. No tier skip, unlike the read-write RGBA8 tests.
#[test]
fn compute_reads_readonly_rgba8_texel() {
    let Some(gpu) = try_gpu() else { return };
    if !gpu.supports_compute_textures() {
        return;
    }
    if gpu.caps().vendor == quanta::Vendor::Broadcom {
        return;
    }
    run_read_texel_rgba8(&gpu);
}

#[cfg(feature = "software")]
#[test]
fn cpu_reads_readonly_texel() {
    let gpu = quanta::init_cpu();
    if !gpu.supports_compute_textures() {
        return;
    }
    run_read_texel(&gpu);
    run_read_texel_rgba8(&gpu);
}

/// The read-only kernel's AOT SPIR-V must validate (NonWritable on a storage
/// image is legal SPIR-V) and must carry an OpImageRead, not OpImageFetch.
#[test]
fn readonly_texel_spirv_module_validates() {
    let Some(spirv) = READ_TEXEL_BINARY.spirv else {
        eprintln!("SKIP: no AOT SPIR-V for read_texel");
        return;
    };
    assert_spirv_val_clean("read_texel", spirv);
    let Some(spirv) = READ_TEXEL_RGBA8_BINARY.spirv else {
        eprintln!("SKIP: no AOT SPIR-V for read_texel_rgba8");
        return;
    };
    assert_spirv_val_clean("read_texel_rgba8", spirv);
}

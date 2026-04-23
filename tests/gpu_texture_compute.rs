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
        .create_texture(&quanta::TextureDesc {
            width: w,
            height: h,
            format: quanta::Format::RGBA8,
            usage: quanta::TextureUsage::SHADER_READ,
            ..quanta::TextureDesc::default()
        })
        .unwrap();
    gpu.texture_write(&tex, &tex_data).unwrap();

    let output = gpu.compute_field::<f32>(n).unwrap();

    eprintln!("metallib: {}", READ_TEXTURE_BINARY.metallib.is_some());
    eprintln!("spirv: {}", READ_TEXTURE_BINARY.spirv.is_some());
    let mut wave = read_texture(&gpu).unwrap();
    wave.bind_texture(0, &tex);
    wave.bind(1, &output);
    wave.set_value(2, w);

    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    gpu.wait(&mut p).unwrap();

    let result = gpu.read_field(&output).unwrap();

    // Check first row: R channel should be 0/255, 64/255, 128/255, 192/255
    for x in 0..w as usize {
        let expected_r = (x as f32 * 64.0) / 255.0;
        let actual = result[x];
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

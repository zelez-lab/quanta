//! Step 052 -- Visual diff tests.
//!
//! Verify rendering and compute correctness by computing expected pixel/data
//! values mathematically and comparing against GPU output. The math IS the
//! reference -- no golden images.
//!
//! Requires a GPU; all tests skip gracefully if none available.

use quanta::render_pass::ColorTarget;
use quanta::{Color, Format, LoadOp, StoreOp};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Create an RGBA8 render target of the given dimensions.
fn create_rgba8_target(gpu: &quanta::Gpu, w: u32, h: u32) -> quanta::Texture {
    gpu.render_target(w, h, Format::RGBA8).unwrap()
}

/// Assert that every byte in `actual` matches `expected` within `tolerance`.
fn assert_pixels_eq(actual: &[u8], expected: &[u8], tolerance: u8, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: size mismatch (actual {} vs expected {})",
        label,
        actual.len(),
        expected.len()
    );
    for i in 0..actual.len() {
        let diff = (actual[i] as i16 - expected[i] as i16).unsigned_abs();
        assert!(
            diff <= tolerance as u16,
            "{}: byte {} differs: expected {}, got {} (diff {})",
            label,
            i,
            expected[i],
            actual[i],
            diff
        );
    }
}

/// CPU reference: square each element.
fn cpu_reference_squares(input: &[f32]) -> Vec<f32> {
    input.iter().map(|x| x * x).collect()
}

/// CPU reference: sum a slice.
fn cpu_reference_sum(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// Convert a linear float (0.0..1.0) to a RGBA8 byte using round-to-nearest.
/// GPU rounding: round(value * 255.0). Both Metal and Vulkan guarantee this.
fn linear_to_u8(v: f32) -> u8 {
    let clamped = v.clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Kernel definitions (proc macro compiles at build time)
// ---------------------------------------------------------------------------

/// Fill a u32 buffer with a packed RGBA gradient: r=x, g=y, b=0, a=255.
#[quanta::kernel]
fn gradient_fill(output: &mut [u32], width: u32) {
    let i = quark_id();
    let x = i % width;
    let y = i / width;
    // Pack RGBA little-endian: r | (g << 8) | (b << 16) | (a << 24)
    // Use % 256 instead of & 0xFF to work around Broadcom V3D OpBitwiseAnd bug.
    output[i] = (x % 256) | ((y % 256) << 8) | (0xFFu32 << 24);
}

/// Square each element.
#[quanta::kernel]
fn compute_squares(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i] * input[i];
}

/// Integer atomic reduction: scale floats to integers and sum.
#[quanta::kernel]
fn sum_reduce(data: &[u32], result: &mut [u32]) {
    let i = quark_id();
    atomic_add(&mut result[0], data[i]);
}

/// Workgroup-level sum using shared memory (64 quarks per group).
#[quanta::kernel]
fn workgroup_sum(data: &[f32], result: &mut [f32]) {
    #[quanta::shared]
    let local: [f32; 64];

    let lid = local_id();
    local[lid] = data[quark_id()];
    barrier();

    if lid == 0 {
        let mut sum = 0.0f32;
        let mut j = 0u32;
        while j < 64 {
            sum = sum + local[j];
            j = j + 1;
        }
        result[group_id()] = sum;
    }
}

// ===========================================================================
// Test 1: Clear to exact color (red)
// ===========================================================================

#[test]
fn clear_red_exact() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 64u32;
    let h = 64u32;
    let target = create_rgba8_target(&gpu, w, h);

    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::rgba(1.0, 0.0, 0.0, 1.0)),
        store_op: StoreOp::Store,
    }]);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let pixels = gpu.texture_read(&target).unwrap();
    let num_pixels = (w * h) as usize;
    assert_eq!(pixels.len(), num_pixels * 4, "unexpected buffer size");

    for p in 0..num_pixels {
        let base = p * 4;
        assert_eq!(
            pixels[base], 255,
            "red channel at pixel {}: expected 255, got {}",
            p, pixels[base]
        );
        assert_eq!(
            pixels[base + 1],
            0,
            "green channel at pixel {}: expected 0, got {}",
            p,
            pixels[base + 1]
        );
        assert_eq!(
            pixels[base + 2],
            0,
            "blue channel at pixel {}: expected 0, got {}",
            p,
            pixels[base + 2]
        );
        assert_eq!(
            pixels[base + 3],
            255,
            "alpha channel at pixel {}: expected 255, got {}",
            p,
            pixels[base + 3]
        );
    }
}

// ===========================================================================
// Test 2: Clear to arbitrary (non-primary) color
// ===========================================================================

#[test]
fn clear_arbitrary_color() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 32u32;
    let h = 32u32;
    let target = create_rgba8_target(&gpu, w, h);

    let color = Color::rgba(0.5, 0.25, 0.75, 1.0);
    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(color),
        store_op: StoreOp::Store,
    }]);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let pixels = gpu.texture_read(&target).unwrap();

    // Mathematical expectation: round(value * 255)
    let expected_r = linear_to_u8(0.5);
    let expected_g = linear_to_u8(0.25);
    let expected_b = linear_to_u8(0.75);
    let expected_a = 255u8;

    // Build expected buffer
    let num_pixels = (w * h) as usize;
    let mut expected = vec![0u8; num_pixels * 4];
    for p in 0..num_pixels {
        let base = p * 4;
        expected[base] = expected_r;
        expected[base + 1] = expected_g;
        expected[base + 2] = expected_b;
        expected[base + 3] = expected_a;
    }

    assert_pixels_eq(&pixels, &expected, 1, "clear_arbitrary_color");
}

// ===========================================================================
// Test 2b: Push constant diagnostic
// ===========================================================================

// ===========================================================================
// Test 3: Compute kernel gradient fill -- verify pixel packing
// ===========================================================================

#[test]
fn compute_gradient_fill() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 64u32;
    let h = 64u32;
    let count = (w * h) as usize;

    let output = gpu.compute_field::<u32>(count).unwrap();

    let mut wave = gradient_fill(&gpu).unwrap();
    wave.bind(0, &output);
    wave.set_value(1, w);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<u32>(&output).unwrap();
    assert_eq!(result.len(), count);

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let packed = result[idx];

            // Unpack RGBA from little-endian u32
            let r = packed & 0xFF;
            let g = (packed >> 8) & 0xFF;
            let b = (packed >> 16) & 0xFF;
            let a = (packed >> 24) & 0xFF;

            assert_eq!(
                r,
                x & 0xFF,
                "gradient r at ({},{}) idx {}: expected {}, got {}",
                x,
                y,
                idx,
                x & 0xFF,
                r
            );
            assert_eq!(
                g,
                y & 0xFF,
                "gradient g at ({},{}) idx {}: expected {}, got {}",
                x,
                y,
                idx,
                y & 0xFF,
                g
            );
            assert_eq!(
                b, 0,
                "gradient b at ({},{}) idx {}: expected 0, got {}",
                x, y, idx, b
            );
            assert_eq!(
                a, 0xFF,
                "gradient a at ({},{}) idx {}: expected 255, got {}",
                x, y, idx, a
            );
        }
    }
}

// ===========================================================================
// Test 4: Compute kernel math -- f32 squares vs CPU reference
// ===========================================================================

#[test]
fn compute_squares_vs_cpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 1024;
    // Diverse inputs: negatives, zero, small, large, fractional
    let input_data: Vec<f32> = (0..count)
        .map(|i| {
            let f = i as f32;
            match i % 4 {
                0 => f * 0.01,       // small positives
                1 => -f * 0.1,       // negatives
                2 => f * 10.0 + 0.5, // large with fraction
                _ => 0.0,            // zeros scattered in
            }
        })
        .collect();

    let expected = cpu_reference_squares(&input_data);

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &input_data).unwrap();

    let mut wave = compute_squares(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    assert_eq!(result.len(), count);

    let epsilon = 0.001f32;
    for i in 0..count {
        let diff = (result[i] - expected[i]).abs();
        // Use relative tolerance for large values, absolute for small
        let tol = if expected[i].abs() > 1.0 {
            expected[i].abs() * epsilon
        } else {
            epsilon
        };
        assert!(
            diff <= tol,
            "squares mismatch at {}: input={}, expected={}, got={}, diff={}",
            i,
            input_data[i],
            expected[i],
            result[i],
            diff
        );
    }
}

// ===========================================================================
// Test 5: Atomic reduction -- integer sum vs CPU reference
// ===========================================================================

#[test]
fn atomic_reduction_vs_cpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let count = 1024;
    // Use u32 values directly to avoid float->int precision issues
    let data: Vec<u32> = (0..count as u32).map(|i| (i % 100) + 1).collect();
    let cpu_sum: u32 = data.iter().sum();

    let data_field = gpu.compute_field::<u32>(count).unwrap();
    let result_field = gpu.compute_field::<u32>(1).unwrap();

    gpu.write_field(&data_field, &data).unwrap();
    gpu.write_field(&result_field, &[0u32]).unwrap();

    let mut wave = sum_reduce(&gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &result_field);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<u32>(&result_field).unwrap();
    assert_eq!(
        result[0], cpu_sum,
        "atomic reduction: GPU sum {} != CPU sum {}",
        result[0], cpu_sum
    );
}

// ===========================================================================
// Test 6: Shared memory reduction -- workgroup sums vs CPU reference
// ===========================================================================

#[test]
fn shared_memory_reduction_vs_cpu() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let group_size = 64usize;
    let num_groups = 8usize;
    let count = group_size * num_groups;

    // Known data: each element is its index * 0.5
    let data: Vec<f32> = (0..count).map(|i| i as f32 * 0.5).collect();

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(num_groups).unwrap();
    gpu.write_field(&input, &data).unwrap();

    let mut wave = workgroup_sum(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.wave_dispatch(&wave, [num_groups as u32, 1, 1]).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();
    assert_eq!(result.len(), num_groups);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let cpu_sum = cpu_reference_sum(&data[start..end]);

        // Allow small tolerance for f32 accumulation order differences
        let tol = cpu_sum.abs() * 0.001 + 0.01;
        let diff = (result[g] - cpu_sum).abs();
        assert!(
            diff <= tol,
            "workgroup {} sum: expected {}, got {}, diff {} (tol {})",
            g,
            cpu_sum,
            result[g],
            diff,
            tol
        );
    }
}

// ===========================================================================
// Test 7: Multi-format clear -- RGBA8, R32Float, RGBA16Float
// ===========================================================================

#[test]
fn multi_format_clear_rgba8() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16u32;
    let h = 16u32;
    let target = create_rgba8_target(&gpu, w, h);

    // Clear to mid-gray
    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::rgba(0.5, 0.5, 0.5, 1.0)),
        store_op: StoreOp::Store,
    }]);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let pixels = gpu.texture_read(&target).unwrap();
    let expected_gray = linear_to_u8(0.5);
    let num_pixels = (w * h) as usize;

    for p in 0..num_pixels {
        let base = p * 4;
        let diff_r = (pixels[base] as i16 - expected_gray as i16).unsigned_abs();
        let diff_g = (pixels[base + 1] as i16 - expected_gray as i16).unsigned_abs();
        let diff_b = (pixels[base + 2] as i16 - expected_gray as i16).unsigned_abs();
        assert!(
            diff_r <= 1 && diff_g <= 1 && diff_b <= 1,
            "gray pixel {}: expected ~({},{},{},255), got ({},{},{},{})",
            p,
            expected_gray,
            expected_gray,
            expected_gray,
            pixels[base],
            pixels[base + 1],
            pixels[base + 2],
            pixels[base + 3],
        );
        assert_eq!(pixels[base + 3], 255, "alpha at pixel {}", p);
    }
}

#[test]
fn multi_format_clear_r32float() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 16u32;
    let h = 16u32;
    let target = gpu.render_target(w, h, Format::R32Float).unwrap();

    let clear_value = 3.14f32;
    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::rgba(clear_value, 0.0, 0.0, 0.0)),
        store_op: StoreOp::Store,
    }]);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let bytes = gpu.texture_read(&target).unwrap();
    let num_pixels = (w * h) as usize;
    // R32Float = 4 bytes per pixel
    assert_eq!(bytes.len(), num_pixels * 4, "R32Float size mismatch");

    let epsilon = 0.001f32;
    for p in 0..num_pixels {
        let offset = p * 4;
        let got = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        let diff = (got - clear_value).abs();
        assert!(
            diff <= epsilon,
            "R32Float pixel {}: expected {}, got {}, diff {}",
            p,
            clear_value,
            got,
            diff
        );
    }
}

#[test]
fn multi_format_clear_rgba16float() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let w = 8u32;
    let h = 8u32;
    let target = gpu.render_target(w, h, Format::RGBA16Float).unwrap();

    // Clear to known values
    let clear_r = 0.25f32;
    let clear_g = 0.5f32;
    let clear_b = 0.75f32;
    let clear_a = 1.0f32;

    let mut pass = gpu.render_begin(&target).unwrap();
    pass.set_color_targets(vec![ColorTarget {
        texture: target.handle(),
        load_op: LoadOp::Clear(Color::rgba(clear_r, clear_g, clear_b, clear_a)),
        store_op: StoreOp::Store,
    }]);
    let mut pulse = gpu.render_end(pass).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let bytes = gpu.texture_read(&target).unwrap();
    let num_pixels = (w * h) as usize;
    // RGBA16Float = 8 bytes per pixel (4 channels x 2 bytes)
    assert_eq!(bytes.len(), num_pixels * 8, "RGBA16Float size mismatch");

    let epsilon = 0.002f32; // half-float has less precision
    let expected = [clear_r, clear_g, clear_b, clear_a];

    for p in 0..num_pixels {
        let base = p * 8;
        for ch in 0..4 {
            let offset = base + ch * 2;
            let half_bits = u16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            let got = f16_to_f32(half_bits);
            let diff = (got - expected[ch]).abs();
            assert!(
                diff <= epsilon,
                "RGBA16Float pixel {} channel {}: expected {}, got {}, diff {}",
                p,
                ch,
                expected[ch],
                got,
                diff
            );
        }
    }
}

// ---------------------------------------------------------------------------
// f16 conversion helper (IEEE 754 half-precision)
// ---------------------------------------------------------------------------

/// Convert an IEEE 754 half-precision (binary16) bit pattern to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            // Signed zero
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: value = (-1)^sign * 2^(-14) * (frac / 1024)
            let val = (frac as f32) / 1024.0 * (2.0f32).powi(-14);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        if frac == 0 {
            // Infinity
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            // NaN
            f32::NAN
        }
    } else {
        // Normal: rebias exponent from half (bias 15) to float (bias 127)
        let f32_exp = exp + 127 - 15;
        let f32_frac = frac << 13; // 10-bit mantissa -> 23-bit
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_frac)
    }
}

// ===========================================================================
// Test: Clear color channel isolation -- verify no cross-channel bleed
// ===========================================================================

#[test]
fn clear_channel_isolation() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Test each primary channel independently to catch any swizzle bugs
    let channels = [
        ("red", Color::rgba(1.0, 0.0, 0.0, 1.0), [255, 0, 0, 255]),
        ("green", Color::rgba(0.0, 1.0, 0.0, 1.0), [0, 255, 0, 255]),
        ("blue", Color::rgba(0.0, 0.0, 1.0, 1.0), [0, 0, 255, 255]),
        (
            "white",
            Color::rgba(1.0, 1.0, 1.0, 1.0),
            [255, 255, 255, 255],
        ),
        ("black", Color::rgba(0.0, 0.0, 0.0, 1.0), [0, 0, 0, 255]),
        ("transparent", Color::rgba(0.0, 0.0, 0.0, 0.0), [0, 0, 0, 0]),
    ];

    let w = 8u32;
    let h = 8u32;

    for (name, color, expected_pixel) in &channels {
        let target = create_rgba8_target(&gpu, w, h);

        let mut pass = gpu.render_begin(&target).unwrap();
        pass.set_color_targets(vec![ColorTarget {
            texture: target.handle(),
            load_op: LoadOp::Clear(*color),
            store_op: StoreOp::Store,
        }]);
        let mut pulse = gpu.render_end(pass).unwrap();
        gpu.wait(&mut pulse).unwrap();

        let pixels = gpu.texture_read(&target).unwrap();
        let num_pixels = (w * h) as usize;

        for p in 0..num_pixels {
            let base = p * 4;
            for ch in 0..4 {
                let diff = (pixels[base + ch] as i16 - expected_pixel[ch] as i16).unsigned_abs();
                assert!(
                    diff <= 1,
                    "{}: pixel {} channel {} expected {}, got {}",
                    name,
                    p,
                    ch,
                    expected_pixel[ch],
                    pixels[base + ch]
                );
            }
        }
    }
}

// ===========================================================================
// Test: Compute squares with edge-case floats
// ===========================================================================

#[test]
fn compute_squares_edge_cases() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Specific edge cases that might differ between CPU and GPU
    let input_data: Vec<f32> = vec![
        0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0, -100.0, 0.001, -0.001, 255.0,
        // Pad to reasonable dispatch size
        0.0, 0.0, 0.0, 0.0,
    ];
    let count = input_data.len();
    let expected = cpu_reference_squares(&input_data);

    let input = gpu.compute_field::<f32>(count).unwrap();
    let output = gpu.compute_field::<f32>(count).unwrap();
    gpu.write_field(&input, &input_data).unwrap();

    let mut wave = compute_squares(&gpu).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);

    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let result = gpu.read_field::<f32>(&output).unwrap();

    let epsilon = 0.001f32;
    for i in 0..count {
        let tol = if expected[i].abs() > 1.0 {
            expected[i].abs() * epsilon
        } else {
            epsilon
        };
        let diff = (result[i] - expected[i]).abs();
        assert!(
            diff <= tol,
            "edge squares[{}]: input={}, expected={}, got={}, diff={}",
            i,
            input_data[i],
            expected[i],
            result[i],
            diff
        );
    }
}

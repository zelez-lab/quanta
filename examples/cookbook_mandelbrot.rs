//! Mandelbrot — auto-dispatch struct-ref form.
//!
//! Output is `Vec<f32>` (iteration count cast to f32) to work around
//! the kernel-IR inference gap tracked by roadmap step 080. Convert
//! back to u32 on the host if needed.
//!
//! Run: cargo run --example cookbook_mandelbrot --release

use quanta::*;

#[derive(quanta::Fields)]
struct MandelbrotData {
    output: Vec<f32>,
    width: u32,
    height: u32,
    max_iter: u32,
}

#[quanta::kernel]
fn mandelbrot(d: &MandelbrotData) {
    let idx = quark_id();
    let px = idx % d.width;
    let py = idx / d.width;

    let x0 = (px as f32 / d.width as f32) * 3.5f32 - 2.5f32;
    let y0 = (py as f32 / d.height as f32) * 2.0f32 - 1.0f32;

    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut iter = 0u32;

    while x * x + y * y <= 4.0f32 && iter < d.max_iter {
        let tmp = x * x - y * y + x0;
        y = 2.0f32 * x * y + y0;
        x = tmp;
        iter += 1u32;
    }

    d.output[idx] = iter as f32;
}

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    let width: u32 = 1920;
    let height: u32 = 1080;
    let max_iter: u32 = 1000;
    let count = (width * height) as usize;

    let mut data = MandelbrotData {
        output: vec![0.0f32; count],
        width,
        height,
        max_iter,
    };

    mandelbrot(&gpu, &mut data, count as u32)?.wait()?;

    let max_iter_f = max_iter as f32;
    let in_set = data.output.iter().filter(|&&v| v >= max_iter_f).count();
    println!(
        "{width}×{height} Mandelbrot: {in_set} pixels in set ({:.1}%)",
        in_set as f64 / count as f64 * 100.0,
    );
    Ok(())
}

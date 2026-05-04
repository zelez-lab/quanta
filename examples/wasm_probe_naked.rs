//! WASM-route experiment probe: a kernel body **without** the
//! `#[quanta::kernel]` attribute, written as plain `extern "C"`.
//!
//! This is what the WASM-route translator should consume. It reveals
//! what rustc emits when handed a typical Quanta-shaped kernel —
//! locals, loops, indirect-buffer access — without the macro
//! interfering. Used by `quanta wasm-experiment`.

#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Stand-in for `quark_id()` — host imports this when running on GPU.
/// In the WASM artifact it appears as a function import.
unsafe extern "C" {
    fn quark_id() -> u32;
}

#[unsafe(no_mangle)]
pub extern "C" fn vector_add(a: *const f32, b: *const f32, result: *mut f32) {
    unsafe {
        let i = quark_id() as usize;
        *result.add(i) = *a.add(i) + *b.add(i);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mandelbrot(output: *mut u32, width: u32, height: u32, max_iter: u32) {
    unsafe {
        let idx = quark_id();
        let px = idx % width;
        let py = idx / width;

        let x0 = (px as f32 / width as f32) * 3.5_f32 - 2.5_f32;
        let y0 = (py as f32 / height as f32) * 2.0_f32 - 1.0_f32;

        let mut x = 0.0_f32;
        let mut y = 0.0_f32;
        let mut iter = 0_u32;
        while x * x + y * y <= 4.0_f32 && iter < max_iter {
            let tmp = x * x - y * y + x0;
            y = 2.0_f32 * x * y + y0;
            x = tmp;
            iter += 1;
        }

        *output.add(idx as usize) = iter;
    }
}

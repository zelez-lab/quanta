//! Async copy + GPU printf cookbook example.
//!
//! Allocates two compute fields, copies one to the other through the
//! async-copy queue, and exercises the printf ring buffer.
//!
//! Run: cargo run --example cookbook_async_copy_and_printf

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    // ── Async copy ──────────────────────────────────────────────────
    let async_copy = match gpu.async_copy_queue() {
        Ok(q) => q,
        Err(e) => {
            println!("async copy queue not supported on this backend: {e:?}");
            return Ok(());
        }
    };

    const N: usize = 1024;
    let src = gpu.field::<f32>(N)?;
    let dst = gpu.field::<f32>(N)?;

    let payload: Vec<f32> = (0..N).map(|i| i as f32).collect();
    src.write(&payload)?;

    async_copy.copy_buffer::<f32>(&dst, &src, N)?;
    println!("async copy: {N} f32 elements transferred src → dst");

    let readback = dst.read()?;
    assert_eq!(readback[42], 42.0);
    println!("verified: dst[42] = {}", readback[42]);

    // ── GPU printf ──────────────────────────────────────────────────
    let printf = match gpu.printf_buffer(256) {
        Ok(p) => p,
        Err(e) => {
            println!("printf not supported on this backend: {e:?}");
            return Ok(());
        }
    };

    println!("printf buffer capacity: {}", printf.capacity());

    // Record a few message IDs (host-side transport for now).
    printf.record(101)?;
    printf.record(202)?;
    printf.record(303)?;

    let drained = printf.drain()?;
    println!("drained {} messages: {:?}", drained.len(), drained);
    Ok(())
}

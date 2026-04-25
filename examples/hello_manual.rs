//! Hello Quanta — manual API for power users.
//! Same result as hello_quanta.rs but with explicit control.
//!
//! Run: cargo run --example hello_manual

use quanta::*;

#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;

    let a = gpu.field::<f32>(1024)?;
    let b = gpu.field::<f32>(1024)?;
    let result = gpu.field::<f32>(1024)?;

    a.write(&vec![1.0f32; 1024])?;
    b.write(&vec![2.0f32; 1024])?;

    let mut wave = vector_add(&gpu).expect("create wave");
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    gpu.dispatch(&wave, 1024)?.wait()?;

    let output = result.read()?;
    assert_eq!(output[0], 3.0);
    println!("1.0 + 2.0 = {}", output[0]);
    Ok(())
}

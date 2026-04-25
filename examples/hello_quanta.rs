//! Hello Quanta — GPU compute in 5 lines of user code.
//!
//! Run: cargo run --example hello_quanta

use quanta::*;

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(d: &VecAdd) {
    let i = quark_id();
    d.result[i] = d.a[i] + d.b[i];
}

fn main() -> Result<(), QuantaError> {
    let gpu = init()?;
    println!("GPU: {}", gpu.name());

    let mut data = VecAdd {
        a: vec![1.0; 1024],
        b: vec![2.0; 1024],
        result: vec![0.0; 1024],
    };

    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[0], 3.0);
    println!("1.0 + 2.0 = {}", data.result[0]);
    Ok(())
}

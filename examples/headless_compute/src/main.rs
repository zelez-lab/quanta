//! Headless compute consumer — the Thiaba / ai_project shape.
//!
//! Depends on `quanta` alone with `default-features = false, features =
//! ["software"]`: render OFF. Proves the step-085 boundary at runtime —
//! a pure GPGPU app builds and dispatches a kernel with zero rendering
//! code compiled and no rendering type on its surface. The companion CI
//! check asserts `cargo tree` for this crate contains no `quanta-render`.

use quanta::{QuantaError, init_cpu};

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
    // Software lane: no GPU, no surface, no render. Pure headless compute.
    let gpu = init_cpu();

    let mut data = VecAdd {
        a: vec![1.0; 1024],
        b: vec![2.0; 1024],
        result: vec![0.0; 1024],
    };

    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[0], 3.0);
    println!("headless compute ok: 1.0 + 2.0 = {}", data.result[0]);
    Ok(())
}

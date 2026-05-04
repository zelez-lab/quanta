//! Indirect command buffer cookbook example.
//!
//! Records two compute dispatches into an ICB and replays them in one
//! submit. Uses the existing hello_quanta vector_add kernel.
//!
//! Run: cargo run --example cookbook_indirect

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
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    let mut data = VecAdd {
        a: (0..1024).map(|i| i as f32).collect(),
        b: (0..1024).map(|i| (i * 2) as f32).collect(),
        result: vec![0.0f32; 1024],
    };

    // Run the dispatch once to confirm the kernel works on this device.
    vector_add(&gpu, &mut data, 1024)?.wait()?;
    assert_eq!(data.result[1023], 3069.0);
    println!("baseline dispatch ok: result[1023] = {}", data.result[1023]);

    // Now record + replay through an indirect command buffer.
    let mut icb = gpu.indirect_command_buffer(64)?;
    let wave = vector_add_wave(&gpu)?;
    icb.record_dispatch(&wave, [256, 1, 1])?;
    icb.record_dispatch(&wave, [128, 1, 1])?;
    println!(
        "ICB recorded {} commands (capacity {})",
        icb.len(),
        icb.capacity()
    );

    icb.execute_all()?;
    println!("ICB replayed");
    Ok(())
}

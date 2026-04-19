//! Hello Quanta — verify GPU compute works on your machine.
//!
//! Run: cargo run --example hello_quanta

// Define a GPU kernel. The proc macro compiles it to MSL, WGSL, PTX, and AMD GCN.
// The function is replaced with: fn vector_add(gpu: &Gpu) -> Result<Wave, QuantaError>
#[quanta::kernel]
fn vector_add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let i = quark_id();
    result[i] = a[i] + b[i];
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    println!(
        "GPU: {} ({} nuclei, {} total quarks, {} MB)",
        gpu.name(),
        gpu.nuclei(),
        gpu.total_quarks(),
        gpu.caps().memory_bytes / 1_000_000
    );

    let count = 1_000_000;
    let a_data: Vec<f32> = (0..count).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..count).map(|i| (i * 2) as f32).collect();

    let a = gpu.compute_field::<f32>(count).unwrap();
    let b = gpu.compute_field::<f32>(count).unwrap();
    let result = gpu.compute_field::<f32>(count).unwrap();

    gpu.write_field(&a, &a_data).unwrap();
    gpu.write_field(&b, &b_data).unwrap();

    // Call the kernel — creates a Wave bound to this GPU
    let mut wave = vector_add(&gpu).expect("create wave");
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &result);

    // Dispatch 1M quarks and wait
    let mut pulse = gpu.dispatch(&wave, count as u32).unwrap();
    gpu.wait(&mut pulse).unwrap();

    let output = gpu.read_field(&result).unwrap();

    let mut errors = 0;
    for i in 0..count {
        if (output[i] - (a_data[i] + b_data[i])).abs() > 0.001 {
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ {} results correct", count);
    } else {
        println!("✗ {} errors out of {}", errors, count);
    }

    println!("\nCompiled targets:");
    println!(
        "  MSL:    {}",
        if VECTOR_ADD_BINARY.msl.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  WGSL:   {}",
        if VECTOR_ADD_BINARY.wgsl.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  NVIDIA: {}",
        if VECTOR_ADD_BINARY.nvidia.is_some() {
            "yes"
        } else {
            "no"
        }
    );
    println!(
        "  AMD:    {}",
        if VECTOR_ADD_BINARY.amd.is_some() {
            "yes"
        } else {
            "no"
        }
    );
}

//! Neural-network weight initialization with quanta-rand.
//!
//! Two of the most common init schemes:
//!   - **Glorot / Xavier**: `W ~ N(0, 2 / (fan_in + fan_out))`
//!   - **He / Kaiming**:    `W ~ N(0, 2 / fan_in)`
//!
//! Both are normal distributions with a specific variance, so we
//! draw from `fill_normal_f32_gpu` (which produces `N(0, 1)`) and
//! scale by `sigma = sqrt(variance)`.
//!
//! Why this matters on GPU: a single hidden layer in a modern model
//! can have 10M+ weights. Drawing them on CPU and uploading is
//! slower than drawing on-device — quanta-rand makes that easy.
//!
//! Run: cargo run --example nn_weight_init -p quanta-rand --features gpu

use quanta_rand::fill_normal_f32_gpu;

fn glorot_weights(gpu: &quanta::Gpu, fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let n = fan_in * fan_out;
    let z = fill_normal_f32_gpu(gpu, n, seed).expect("dispatch");
    // sigma = sqrt(2 / (fan_in + fan_out))
    let sigma = (2.0 / (fan_in + fan_out) as f32).sqrt();
    z.into_iter().map(|x| x * sigma).collect()
}

fn he_weights(gpu: &quanta::Gpu, fan_in: usize, fan_out: usize, seed: u64) -> Vec<f32> {
    let n = fan_in * fan_out;
    let z = fill_normal_f32_gpu(gpu, n, seed).expect("dispatch");
    let sigma = (2.0 / fan_in as f32).sqrt();
    z.into_iter().map(|x| x * sigma).collect()
}

fn stats(name: &str, w: &[f32]) {
    let mean: f32 = w.iter().sum::<f32>() / w.len() as f32;
    let var: f32 = w.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / w.len() as f32;
    let max = w.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let min = w.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    println!(
        "  {name:<8}  n={:<6}  mean={mean:+.4}  std={:.4}  range=[{min:+.3}, {max:+.3}]",
        w.len(),
        var.sqrt()
    );
}

fn main() {
    let gpu = quanta::init_cpu();
    let seed = 0xDEAD_BEEFu64;

    // Toy MLP: 784 → 256 → 128 → 10 (e.g. MNIST classifier)
    let layers = [(784, 256), (256, 128), (128, 10)];

    println!("== Glorot / Xavier init ==");
    println!("  variance = 2 / (fan_in + fan_out)");
    for (i, &(fan_in, fan_out)) in layers.iter().enumerate() {
        let layer_seed = seed ^ (i as u64); // different seed per layer
        let w = glorot_weights(&gpu, fan_in, fan_out, layer_seed);
        stats(&format!("layer{i}"), &w);
    }

    println!();
    println!("== He / Kaiming init (preferred for ReLU) ==");
    println!("  variance = 2 / fan_in");
    for (i, &(fan_in, fan_out)) in layers.iter().enumerate() {
        let layer_seed = seed ^ (i as u64);
        let w = he_weights(&gpu, fan_in, fan_out, layer_seed);
        stats(&format!("layer{i}"), &w);
    }

    println!();
    println!("note: the empirical std should match the theoretical formula");
    println!("      for each scheme (sqrt(variance)). Re-running with the");
    println!("      same seed gives bit-identical weights — useful for");
    println!("      reproducible training.");
}

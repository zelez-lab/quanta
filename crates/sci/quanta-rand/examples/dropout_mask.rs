//! Dropout mask for neural-network training.
//!
//! Dropout zeroes out a fraction `p_drop` of activations each
//! forward pass to regularise training. Drawing the mask on the
//! same device as the activations avoids a host roundtrip.
//!
//! Convention: `keep_prob = 1 - p_drop`. Scale surviving activations
//! by `1 / keep_prob` so the expected value of the layer output is
//! unchanged (inverted dropout — the standard).
//!
//! Run: cargo run --example dropout_mask -p quanta-rand --features gpu

use quanta_rand::fill_bernoulli_u32_gpu;

fn dropout(gpu: &quanta::Gpu, activations: &[f32], p_drop: f32, seed: u64) -> Vec<f32> {
    let keep_prob = 1.0 - p_drop;
    let mask = fill_bernoulli_u32_gpu(gpu, activations.len(), seed, keep_prob).expect("dispatch");
    // Inverted dropout: a[i] * mask[i] / keep_prob.
    activations
        .iter()
        .zip(mask.iter())
        .map(|(&a, &m)| if m == 1 { a / keep_prob } else { 0.0 })
        .collect()
}

fn main() {
    let gpu = quanta::init_cpu();

    // Pretend layer activations.
    let acts: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
    let p_drop = 0.5;

    let out = dropout(&gpu, &acts, p_drop, 0xD0_DEAD);

    let kept = out.iter().filter(|&&v| v != 0.0).count();
    let sum_in: f32 = acts.iter().sum();
    let sum_out: f32 = out.iter().sum();

    println!("Dropout demo");
    println!("  n            : {}", acts.len());
    println!("  p_drop       : {p_drop}");
    println!(
        "  kept         : {kept} / {} ({:.0}%)",
        acts.len(),
        100.0 * kept as f32 / acts.len() as f32
    );
    println!("  sum input    : {sum_in:.3}");
    println!("  sum output   : {sum_out:.3}  (should ≈ sum input under inverted dropout)");
    println!();
    println!("  acts in  : {:?}", &acts[..8]);
    println!("  acts out : {:?}", &out[..8]);
    println!();
    println!("note: re-running with the same seed gives the same mask,");
    println!("      so training is reproducible. Bump the seed each");
    println!("      step (e.g. seed ^ step_idx) for real training.");
}

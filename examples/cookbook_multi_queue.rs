//! Multi-queue cookbook example.
//!
//! Inspects available queues, acquires a Compute queue if one exists, and
//! prints the topology. Falls back gracefully when only Graphics is
//! exposed (e.g. WebGPU).
//!
//! Run: cargo run --example cookbook_multi_queue

use quanta::*;

fn main() -> Result<(), QuantaError> {
    let gpu = quanta::init()?;
    println!("GPU: {}", gpu.name());

    let families = gpu.queue_families();
    println!("available queue families:");
    for fam in &families {
        println!("  {:?}: {} queue(s)", fam.queue_type, fam.count);
    }

    // Always-available: graphics queue.
    let _graphics = gpu.queue(QueueType::Graphics)?;
    println!("acquired Graphics queue");

    // Optional: dedicated compute.
    match gpu.queue(QueueType::Compute) {
        Ok(_) => println!("acquired Compute queue"),
        Err(e) => println!("Compute queue not available: {e:?}"),
    }

    // Optional: dedicated transfer.
    match gpu.queue(QueueType::Transfer) {
        Ok(_) => println!("acquired Transfer queue"),
        Err(e) => println!("Transfer queue not available: {e:?}"),
    }
    Ok(())
}

//! Demo: pure-CPU reference module.
//!
//! Shows every reference function in `quanta_prims::reference`
//! on the same small input. No GPU required — this example
//! works whether or not the `gpu` feature is enabled.
//!
//! Useful when writing differential tests: import these
//! references as the oracle and compare GPU output against
//! them.
//!
//! Run: cargo run -p quanta-prims --example cpu_oracle

use quanta_prims::reference;

fn main() {
    let xs: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];

    println!("== quanta-prims CPU reference tour ==");
    println!("input: {xs:?}");
    println!();

    // ── reduce family (u32) ──────────────────────────────────────
    println!("reduce_add_u32: {}", reference::reduce_add_u32(&xs));
    println!("reduce_min_u32: {}", reference::reduce_min_u32(&xs));
    println!("reduce_max_u32: {}", reference::reduce_max_u32(&xs));
    println!();

    // ── reduce family (i32) ──────────────────────────────────────
    let ys: Vec<i32> = xs
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { v as i32 } else { -(v as i32) })
        .collect();
    println!("input (signed): {ys:?}");
    println!("reduce_add_i32: {}", reference::reduce_add_i32(&ys));
    println!("reduce_min_i32: {}", reference::reduce_min_i32(&ys));
    println!("reduce_max_i32: {}", reference::reduce_max_i32(&ys));
    println!();

    // ── reduce family (f32) ──────────────────────────────────────
    let zs: Vec<f32> = xs.iter().map(|&v| v as f32 / 4.0).collect();
    println!("input (float): {zs:?}");
    println!("reduce_add_f32: {:.4}", reference::reduce_add_f32(&zs));
    println!("reduce_min_f32: {:.4}", reference::reduce_min_f32(&zs));
    println!("reduce_max_f32: {:.4}", reference::reduce_max_f32(&zs));
    println!();

    // ── scan + sort ──────────────────────────────────────────────
    println!("scan_add_u32:   {:?}", reference::scan_add_u32(&xs));
    println!("radix_sort_u32: {:?}", reference::radix_sort_u32(&xs));
}

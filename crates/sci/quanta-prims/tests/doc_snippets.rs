//! Validates the pure-Rust snippets in `COOKBOOK.md` and
//! `GETTING_STARTED.md`. The GPU-kernel snippets are exercised
//! by the corresponding `tests/block_*.rs` integration tests
//! (which `cargo test` runs alongside this one). What's checked
//! here is the reference-module surface — the snippets that
//! readers can run on the CPU without a GPU.

use quanta_prims::reference;

#[test]
fn cookbook_cpu_oracle_reduce_add() {
    let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(reference::reduce_add_u32(&xs), 31);
}

#[test]
fn cookbook_cpu_oracle_scan_add() {
    let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(
        reference::scan_add_u32(&xs),
        vec![3, 4, 8, 9, 14, 23, 25, 31]
    );
}

#[test]
fn cookbook_cpu_oracle_reduce_min() {
    let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(reference::reduce_min_u32(&xs), 1);
}

#[test]
fn cookbook_cpu_oracle_reduce_max() {
    let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(reference::reduce_max_u32(&xs), 9);
}

#[test]
fn cookbook_cpu_oracle_radix_sort() {
    let xs = vec![3u32, 1, 4, 1, 5, 9, 2, 6];
    assert_eq!(reference::radix_sort_u32(&xs), vec![1, 1, 2, 3, 4, 5, 6, 9]);
}

#[test]
fn getting_started_reduce_smoke() {
    // The GETTING_STARTED first-reduce example computes the
    // sum of 1..=256 = 32896. Verify the reference matches.
    let n = 256usize;
    let data: Vec<u32> = (1..=n as u32).collect();
    let expected = (1..=n as u32).sum::<u32>();
    assert_eq!(reference::reduce_add_u32(&data), expected);
}

//! Benchmark: GPU scan over host-imported memory vs a GPU-owned field.
//!
//! The point of zero-copy import: on unified topology a scan over an
//! imported region should sit within a few percent of the same scan
//! over a field the driver allocated (the acceptance bar is ≥95%).
//! On a staged-copy backend the import path pays its copy at creation,
//! not per dispatch — the per-dispatch times should still match.
//!
//! Run: cargo run --example bench_host_import --release

use std::time::Instant;

#[quanta::kernel]
fn scan_scale(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i] * 2.0f32 + 1.0f32;
}

const REPS: usize = 20;

fn time_scan(gpu: &quanta::Gpu, wave: &quanta::Wave, count: usize) -> f64 {
    // Warm up
    gpu.dispatch(wave, count as u32).unwrap().wait().unwrap();
    let start = Instant::now();
    for _ in 0..REPS {
        gpu.dispatch(wave, count as u32).unwrap().wait().unwrap();
    }
    start.elapsed().as_secs_f64() / REPS as f64
}

fn main() {
    let gpu = quanta::init().expect("no GPU found");
    let align = gpu.host_import_alignment().unwrap_or(4096).max(4);
    println!(
        "GPU: {} — topology {:?}, host import: {} (granularity {:?})",
        gpu.name(),
        gpu.memory_topology(),
        gpu.supports_host_import(),
        gpu.host_import_alignment(),
    );

    // 64 MiB of f32s — page-multiple for every real granularity.
    let count = 16 * 1024 * 1024;
    let layout = std::alloc::Layout::from_size_align(count * 4, align).unwrap();
    let host = unsafe { std::alloc::alloc_zeroed(layout) } as *mut f32;
    assert!(!host.is_null());
    let host_slice = unsafe { std::slice::from_raw_parts_mut(host, count) };
    for (i, x) in host_slice.iter_mut().enumerate() {
        *x = i as f32 * 0.25;
    }

    let out = gpu.field::<f32>(count).unwrap();

    // Owned path: driver-allocated field, one upload.
    let owned = gpu.field::<f32>(count).unwrap();
    owned.write(host_slice).unwrap();
    let mut wave = scan_scale(&gpu).expect("create wave");
    wave.bind(0, &owned);
    wave.bind(1, &out);
    let owned_time = time_scan(&gpu, &wave, count);

    // Imported path: the device reads the caller's pages.
    let imported = gpu.field_from_host(&host_slice[..]).unwrap();
    let mut wave = scan_scale(&gpu).expect("create wave");
    wave.bind_host(0, &imported);
    wave.bind(1, &out);
    let import_time = time_scan(&gpu, &wave, count);

    let ratio = owned_time / import_time;
    println!(
        "{count} elements × {REPS} reps:  owned {:.3} ms  imported {:.3} ms  → imported at {:.1}% of owned",
        owned_time * 1e3,
        import_time * 1e3,
        ratio * 100.0,
    );
    println!(
        "zero-copy: {}  ({} path)",
        imported.is_imported(),
        if imported.is_imported() {
            "native import"
        } else {
            "staged copy"
        },
    );

    drop(imported);
    unsafe { std::alloc::dealloc(host as *mut u8, layout) };
}

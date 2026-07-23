//! Memory topology as a first-class queryable property.
//!
//! `gpu.memory_topology()` reports the transfer-cost regime of the
//! (backend, device) pair — `Unified` when host and device share one
//! physical pool, `Discrete` when transfers cross a real interconnect.
//! Consumers with a CPU-vs-GPU routing decision branch on this instead
//! of hardcoding one topology's crossover.
//!
//! Run: cargo test --test memory_topology --features software

// Only the cfg-gated tests name the enum — a plain `use` would be
// unused in e.g. the vulkan-only CI combo.
#[cfg(any(
    feature = "software",
    all(target_os = "macos", target_arch = "aarch64")
))]
use quanta::MemoryTopology;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// The software driver executes in host memory: trivially unified.
#[test]
#[cfg(feature = "software")]
fn cpu_backend_is_unified() {
    let gpu = quanta::init_cpu();
    assert_eq!(gpu.memory_topology(), MemoryTopology::Unified);
}

/// The convenience accessor is a read of the caps field, never a
/// separate code path.
#[test]
fn accessor_matches_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    assert_eq!(gpu.memory_topology(), gpu.caps().memory_topology);
}

/// Every Apple-silicon Mac has unified memory; Metal's
/// `hasUnifiedMemory` must agree. (Gated to aarch64 so an Intel Mac
/// with a discrete card doesn't fail spuriously.)
#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn apple_silicon_reports_unified() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if gpu.caps().vendor != quanta::Vendor::Apple {
        eprintln!("skipping: active device is not the Metal GPU");
        return;
    }
    assert_eq!(gpu.memory_topology(), MemoryTopology::Unified);
}

//! Tier 2 -- Device capabilities.
//!
//! Verifies that the GPU reports sane capability values.
//! Requires a GPU; skips gracefully if none available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn device_has_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.caps();
    assert!(caps.nuclei > 0, "nuclei must be > 0, got {}", caps.nuclei);
    assert!(
        caps.protons_per_nucleus > 0,
        "protons_per_nucleus must be > 0, got {}",
        caps.protons_per_nucleus
    );
    assert!(
        caps.quarks_per_proton > 0,
        "quarks_per_proton must be > 0, got {}",
        caps.quarks_per_proton
    );
    assert!(
        caps.memory_bytes > 0,
        "memory_bytes must be > 0, got {}",
        caps.memory_bytes
    );
    assert!(!caps.name.is_empty(), "device name must not be empty");
}

#[test]
fn total_quarks_consistent() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.caps();
    let expected = caps.nuclei * caps.protons_per_nucleus * caps.quarks_per_proton;
    assert_eq!(
        caps.total_quarks(),
        expected,
        "total_quarks() should be nuclei * protons * quarks"
    );
    assert_eq!(gpu.total_quarks(), expected);
}

#[test]
fn convenience_accessors_match_caps() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.caps();
    assert_eq!(gpu.nuclei(), caps.nuclei);
    assert_eq!(gpu.protons_per_nucleus(), caps.protons_per_nucleus);
    assert_eq!(gpu.quarks_per_proton(), caps.quarks_per_proton);
    assert_eq!(gpu.name(), caps.name.as_str());
}

#[test]
fn max_groups_are_nonzero() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.caps();
    for (dim, &val) in caps.max_groups.iter().enumerate() {
        assert!(val > 0, "max_groups[{}] must be > 0, got {}", dim, val);
    }
}

#[test]
fn max_quarks_per_dispatch_nonzero() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let caps = gpu.caps();
    assert!(
        caps.max_quarks_per_dispatch > 0,
        "max_quarks_per_dispatch must be > 0"
    );
}

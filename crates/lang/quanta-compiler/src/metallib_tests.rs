//! Unit tests for the metallib platform-variant emission.
//!
//! Like the other metallib tests, these gate on `xcrun` presence and the
//! relevant SDK, so they `SKIP` on non-Apple hosts / a Command-Line-Tools
//! -only mac rather than fail. On a full-Xcode host they prove all three
//! variants are produced, each begins with the `MTLB` magic, and the iOS
//! variants differ byte-wise from the macOS one.

use super::{
    MetalPlatform, MetallibVariants, compile_msl_to_metallib, compile_msl_to_metallib_variants,
};

/// Serializes every test that reads or writes the `QUANTA_METAL_PLATFORMS`
/// / `QUANTA_SKIP_METALLIB` env vars. Env is process-global, so a test that
/// mutates it races any concurrent test that reads it (cargo runs tests in
/// parallel). Both the mutators AND the default-state readers hold this
/// lock. Poison is ignored — a panicking test still asserted correctly;
/// the lock only orders access.
fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// Every metallib begins with the four-byte `MTLB` magic.
const MTLB_MAGIC: &[u8; 4] = b"MTLB";

/// A trivial, portable compute kernel — no platform-specific features, so
/// it compiles for macOS and both iOS targets.
const TRIVIAL_MSL: &str = "\
#include <metal_stdlib>
using namespace metal;

kernel void trivial(device float* out [[buffer(0)]],
                    uint gid [[thread_position_in_grid]]) {
    out[gid] = float(gid) * 2.0;
}
";

fn xcrun_present() -> bool {
    std::process::Command::new("xcrun")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Whether an iOS SDK is installed (full Xcode). Mirrors the probe in the
/// module under test so the assertions only fire where the variant can be
/// produced.
fn ios_sdk_present(sdk: &str) -> bool {
    std::process::Command::new("xcrun")
        .args(["-sdk", sdk, "--show-sdk-path"])
        .output()
        .map(|o| o.status.success() && !o.stdout.is_empty())
        .unwrap_or(false)
}

fn begins_with_mtlb(bytes: &[u8], label: &str) {
    assert!(
        bytes.len() >= 4 && &bytes[..4] == MTLB_MAGIC,
        "{label}: expected MTLB magic, got {:02x?}",
        &bytes[..bytes.len().min(4)]
    );
}

#[test]
fn produces_all_three_variants_on_full_xcode() {
    // Reads the DEFAULT env state (no QUANTA_METAL_PLATFORMS override); hold
    // the lock so a concurrent env-mutating test can't exclude iOS mid-run.
    let _guard = env_lock();
    if !xcrun_present() {
        eprintln!("SKIP produces_all_three_variants: xcrun absent");
        return;
    }
    let has_ios = ios_sdk_present("iphoneos");
    let has_ios_sim = ios_sdk_present("iphonesimulator");

    let variants = compile_msl_to_metallib_variants(TRIVIAL_MSL)
        .expect("trivial MSL must compile on a mac with the Metal toolchain");

    // macOS is always attempted and must succeed where xcrun exists.
    let macos = variants
        .macos
        .as_ref()
        .expect("macOS metallib must be produced");
    begins_with_mtlb(macos, "macos");

    // iOS variants track SDK availability: present iff the SDK is.
    assert_eq!(
        variants.ios.is_some(),
        has_ios,
        "ios variant presence must match SDK availability"
    );
    assert_eq!(
        variants.ios_sim.is_some(),
        has_ios_sim,
        "ios-sim variant presence must match SDK availability"
    );

    if let Some(ios) = variants.ios.as_ref() {
        begins_with_mtlb(ios, "ios");
        // A platform-correct iOS metallib is not byte-identical to the
        // macOS one (different target triple / platform record) — this is
        // the whole point of the feature.
        assert_ne!(ios, macos, "ios metallib must differ from macos bytes");
    }
    if let Some(ios_sim) = variants.ios_sim.as_ref() {
        begins_with_mtlb(ios_sim, "ios-sim");
        assert_ne!(
            ios_sim, macos,
            "ios-sim metallib must differ from macos bytes"
        );
    }
    if let (Some(ios), Some(ios_sim)) = (variants.ios.as_ref(), variants.ios_sim.as_ref()) {
        assert_ne!(
            ios, ios_sim,
            "ios device and simulator metallibs must differ"
        );
    }
}

#[test]
fn platforms_env_restricts_to_macos_only() {
    let _guard = env_lock();
    if !xcrun_present() {
        eprintln!("SKIP platforms_env_restricts_to_macos_only: xcrun absent");
        return;
    }
    // Forcing QUANTA_METAL_PLATFORMS=macos exercises the SDK-absent code
    // path deterministically: only the macOS variant is attempted, the iOS
    // fields stay None even on a full-Xcode host.
    //
    // SAFETY: set_var/remove_var are unsafe on 2024 edition (they mutate
    // process-global env). `env_lock` serializes every env-touching test, so
    // no concurrent test observes the mutation; it is restored before
    // returning regardless.
    let prev = std::env::var("QUANTA_METAL_PLATFORMS").ok();
    unsafe {
        std::env::set_var("QUANTA_METAL_PLATFORMS", "macos");
    }
    let variants = compile_msl_to_metallib_variants(TRIVIAL_MSL);
    // Restore before asserting so a failure can't leak the override.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("QUANTA_METAL_PLATFORMS", v),
            None => std::env::remove_var("QUANTA_METAL_PLATFORMS"),
        }
    }
    let variants = variants.expect("macOS-only compile must succeed");
    begins_with_mtlb(
        variants.macos.as_ref().expect("macOS metallib present"),
        "macos",
    );
    assert!(
        variants.ios.is_none(),
        "ios excluded by QUANTA_METAL_PLATFORMS=macos"
    );
    assert!(
        variants.ios_sim.is_none(),
        "ios-sim excluded by QUANTA_METAL_PLATFORMS=macos"
    );
}

#[test]
fn skip_metallib_env_yields_no_variants() {
    let _guard = env_lock();
    // QUANTA_SKIP_METALLIB=1 opts out entirely — no xcrun invocation, all
    // fields None. This holds on every host (no toolchain needed).
    let prev_skip = std::env::var("QUANTA_SKIP_METALLIB").ok();
    unsafe {
        std::env::set_var("QUANTA_SKIP_METALLIB", "1");
    }
    let result = compile_msl_to_metallib_variants(TRIVIAL_MSL);
    let scalar = compile_msl_to_metallib(TRIVIAL_MSL);
    unsafe {
        match prev_skip {
            Some(v) => std::env::set_var("QUANTA_SKIP_METALLIB", v),
            None => std::env::remove_var("QUANTA_SKIP_METALLIB"),
        }
    }
    let variants = result.expect("skip path is not an error");
    assert!(variants.macos.is_none());
    assert!(variants.ios.is_none());
    assert!(variants.ios_sim.is_none());
    assert!(
        scalar.expect("scalar skip is not an error").is_none(),
        "scalar shim also returns None under QUANTA_SKIP_METALLIB"
    );
}

#[test]
fn scalar_shim_matches_variant_macos() {
    // Reads default env state — serialize with the env-mutating tests.
    let _guard = env_lock();
    if !xcrun_present() {
        eprintln!("SKIP scalar_shim_matches_variant_macos: xcrun absent");
        return;
    }
    // The back-compat scalar entry point returns exactly the macOS variant.
    let scalar = compile_msl_to_metallib(TRIVIAL_MSL).expect("scalar compile");
    let variants = compile_msl_to_metallib_variants(TRIVIAL_MSL).expect("variant compile");
    assert_eq!(
        scalar.is_some(),
        variants.macos.is_some(),
        "scalar shim presence tracks the macOS variant"
    );
    if let (Some(s), Some(m)) = (scalar.as_ref(), variants.macos.as_ref()) {
        begins_with_mtlb(s, "scalar");
        begins_with_mtlb(m, "variant-macos");
    }
}

#[test]
fn all_constant_covers_three_platforms() {
    // Guard the enum/probe shape the spec calls out as leaving room for
    // watchOS/tvOS/visionOS: ALL currently enumerates exactly the three.
    assert_eq!(MetalPlatform::ALL.len(), 3);
    assert!(MetalPlatform::ALL.contains(&MetalPlatform::MacOs));
    assert!(MetalPlatform::ALL.contains(&MetalPlatform::Ios));
    assert!(MetalPlatform::ALL.contains(&MetalPlatform::IosSim));
    // Default is all-None.
    let d = MetallibVariants::default();
    assert!(d.macos.is_none() && d.ios.is_none() && d.ios_sim.is_none());
}

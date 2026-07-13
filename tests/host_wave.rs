//! Tier 1 (host, no GPU) conformance tests — Wave struct layout and sizing.
//!
//! Tests the Wave struct's inline binding model without requiring a GPU.
//! Since Wave fields are pub(crate), we test through layout verification
//! and the set_value/set_bytes public API where accessible.
//!
//! Run: cargo test --test host_wave

use quanta::*;

// ===========================================================================
// Wave struct layout — no hidden allocations
// ===========================================================================

#[test]
fn wave_struct_size_is_bounded() {
    // Wave contains:
    // - handle: u64 (8)
    // - bindings: [u64; 16] (128)
    // - binding_count: u8 (1)
    // - texture_bindings: [u64; 16] (128)
    // - texture_count: u8 (1)
    // - push_data: [u8; 256] (256)
    // - push_len: u16 (2)
    // - push_mask: u16 (2)
    // - workgroup_size: [u32; 3] (12)
    // - drop_fn: Option<Box<dyn FnOnce(u64)>> (16 on 64-bit)
    // Total: ~554 + padding
    //
    // The key invariant: Wave is stack-allocated with inline arrays,
    // no heap allocation on the hot path.
    let size = core::mem::size_of::<Wave>();

    // Must be reasonable — inline arrays mean it's a few hundred bytes,
    // not a pointer to a heap allocation.
    assert!(
        size >= 500,
        "Wave must be large enough for inline arrays: got {} bytes",
        size
    );
    assert!(
        size <= 1024,
        "Wave must not be unreasonably large: got {} bytes",
        size
    );
}

#[test]
fn wave_alignment() {
    // Wave should be aligned to at least 8 bytes (contains u64 fields)
    let align = core::mem::align_of::<Wave>();
    assert!(align >= 8, "Wave alignment must be >= 8: got {}", align);
}

// ===========================================================================
// KernelBinary — for_vendor logic (binary-only)
// ===========================================================================

#[test]
fn kernel_binary_for_vendor_amd_prefers_amd_then_spirv() {
    let binary = KernelBinary {
        amd: Some(b"amd_binary"),
        nvidia: Some(b"nvidia_binary"),
        spirv: Some(b"spirv_binary"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Amd);
    assert_eq!(result, Some(b"amd_binary" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_amd_falls_back_to_spirv() {
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: Some(b"spirv_binary"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Amd);
    assert_eq!(result, Some(b"spirv_binary" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_nvidia_prefers_nvidia_then_spirv() {
    let binary = KernelBinary {
        amd: None,
        nvidia: Some(b"ptx_binary"),
        spirv: Some(b"spirv_binary"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Nvidia);
    assert_eq!(result, Some(b"ptx_binary" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_nvidia_falls_back_to_spirv() {
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: Some(b"spirv_binary"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Nvidia);
    assert_eq!(result, Some(b"spirv_binary" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_apple_returns_metallib_only() {
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: Some(b"metallib_binary"),
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Apple);
    assert_eq!(result, Some(b"metallib_binary" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_apple_returns_none_without_metallib() {
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: Some(b"spirv_binary"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Apple);
    assert!(
        result.is_none(),
        "Apple without metallib should return None"
    );
}

#[test]
fn kernel_binary_for_vendor_intel_prefers_spirv() {
    let binary = KernelBinary {
        amd: Some(b"amd"),
        nvidia: None,
        spirv: Some(b"spirv"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Intel);
    assert_eq!(result, Some(b"spirv" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_intel_falls_back_to_amd() {
    let binary = KernelBinary {
        amd: Some(b"amd"),
        nvidia: None,
        spirv: None,
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    // Intel prefers spirv, then amd
    let result = binary.for_vendor(Vendor::Intel);
    assert_eq!(result, Some(b"amd" as &[u8]));
}

#[test]
fn kernel_binary_for_vendor_unknown_returns_spirv_only() {
    // Unknown Vulkan vendors get SPIR-V only.
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    let result = binary.for_vendor(Vendor::Unknown);
    assert!(
        result.is_none(),
        "unknown vendor without SPIR-V should return None"
    );

    // With SPIR-V available, it should work
    let binary2 = KernelBinary {
        spirv: Some(&[0x03, 0x02, 0x23, 0x07]),
        ..binary
    };
    let result2 = binary2.for_vendor(Vendor::Unknown);
    assert!(result2.is_some());
}

#[test]
fn kernel_binary_for_vendor_none_returns_none() {
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
    };

    assert_eq!(binary.for_vendor(Vendor::Amd), None);
    assert_eq!(binary.for_vendor(Vendor::Nvidia), None);
    assert_eq!(binary.for_vendor(Vendor::Apple), None);
    assert_eq!(binary.for_vendor(Vendor::Unknown), None);
}

// ===========================================================================
// ShaderBinary �� for_vendor logic (binary-only)
// ===========================================================================

#[test]
fn shader_binary_for_vendor_apple_returns_metallib() {
    let shader = ShaderBinary {
        spirv: Some(b"spirv_bytes"),
        metallib: Some(b"MTLBmetallib_bytes"),
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
        entry_point: "main",
        stage: ShaderStage::Vertex,
    };

    let result = shader.for_vendor(Vendor::Apple);
    assert_eq!(result, Some(b"MTLBmetallib_bytes" as &[u8]));
}

#[test]
fn shader_binary_for_vendor_nvidia_returns_spirv() {
    let shader = ShaderBinary {
        spirv: Some(b"spirv_bytes"),
        metallib: Some(b"metallib_bytes"),
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
        entry_point: "main",
        stage: ShaderStage::Vertex,
    };

    let result = shader.for_vendor(Vendor::Nvidia);
    assert_eq!(result, Some(b"spirv_bytes" as &[u8]));
}

#[test]
fn shader_binary_for_vendor_apple_falls_back_to_spirv() {
    let shader = ShaderBinary {
        spirv: Some(b"spirv_bytes"),
        metallib: None,
        metallib_ios: None,
        metallib_ios_sim: None,
        wgsl: None,
        entry_point: "main",
        stage: ShaderStage::Fragment,
    };

    let result = shader.for_vendor(Vendor::Apple);
    assert_eq!(result, Some(b"spirv_bytes" as &[u8]));
}

// ===========================================================================
// Platform-targeted metallib selection (cfg-gated)
//
// `for_vendor` picks the metallib variant matching the *compile target*.
// The chain is iOS-sim → iOS-device → macOS, falling to SPIR-V for shaders.
// Each target compiles exactly one arm, so the assertions are split by cfg:
// the host suite (macOS/desktop) sees the macOS-only behavior; the iOS and
// iOS-simulator arms are proven to compile by the cross-target `cargo check`
// and run when the suite is executed on those targets.
// ===========================================================================

#[cfg(not(target_os = "ios"))]
#[test]
fn kernel_apple_selection_is_macos_on_desktop() {
    // On a non-iOS build, the iOS fields are ignored — Apple resolves to the
    // macOS metallib only, exactly as before the iOS variants existed.
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: Some(b"macos_metallib"),
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
    };
    assert_eq!(
        binary.for_vendor(Vendor::Apple),
        Some(b"macos_metallib" as &[u8]),
        "desktop build must select the macOS metallib"
    );

    // With only iOS variants present, a desktop build finds no macOS
    // metallib and returns None (kernels then JIT) — the iOS bytes are
    // never picked off-target.
    let ios_only = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
    };
    assert!(
        ios_only.for_vendor(Vendor::Apple).is_none(),
        "desktop build must not select an iOS metallib"
    );
}

#[cfg(not(target_os = "ios"))]
#[test]
fn shader_apple_selection_is_macos_then_spirv_on_desktop() {
    // Shader Apple chain on desktop: macOS metallib, else SPIR-V. iOS
    // variants are inert off-target.
    let shader = ShaderBinary {
        spirv: Some(b"spirv_bytes"),
        metallib: Some(b"macos_metallib"),
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
        entry_point: "main",
        stage: ShaderStage::Fragment,
    };
    assert_eq!(
        shader.for_vendor(Vendor::Apple),
        Some(b"macos_metallib" as &[u8])
    );

    // No macOS metallib on desktop → SPIR-V fallback, never the iOS bytes.
    let ios_only = ShaderBinary {
        spirv: Some(b"spirv_bytes"),
        metallib: None,
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
        entry_point: "main",
        stage: ShaderStage::Fragment,
    };
    assert_eq!(
        ios_only.for_vendor(Vendor::Apple),
        Some(b"spirv_bytes" as &[u8]),
        "desktop build falls back to SPIR-V, not an iOS metallib"
    );
}

#[cfg(all(target_os = "ios", not(target_abi = "sim")))]
#[test]
fn kernel_apple_selection_prefers_ios_device() {
    // iOS device build: iOS metallib preferred, macOS as fallback.
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: Some(b"macos_metallib"),
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
    };
    assert_eq!(
        binary.for_vendor(Vendor::Apple),
        Some(b"ios_metallib" as &[u8])
    );
    // Falls back to macOS when the device variant wasn't produced.
    let no_ios = KernelBinary {
        metallib_ios: None,
        ..binary
    };
    assert_eq!(
        no_ios.for_vendor(Vendor::Apple),
        Some(b"macos_metallib" as &[u8])
    );
}

#[cfg(all(target_os = "ios", target_abi = "sim"))]
#[test]
fn kernel_apple_selection_prefers_ios_sim() {
    // iOS simulator build: sim metallib preferred, then device, then macOS.
    let binary = KernelBinary {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: Some(b"macos_metallib"),
        metallib_ios: Some(b"ios_metallib"),
        metallib_ios_sim: Some(b"ios_sim_metallib"),
        wgsl: None,
    };
    assert_eq!(
        binary.for_vendor(Vendor::Apple),
        Some(b"ios_sim_metallib" as &[u8])
    );
    let no_sim = KernelBinary {
        metallib_ios_sim: None,
        ..binary
    };
    assert_eq!(
        no_sim.for_vendor(Vendor::Apple),
        Some(b"ios_metallib" as &[u8])
    );
}

// ===========================================================================
// KernelBinary static fields
// ===========================================================================

#[quanta::kernel]
fn wave_test_kernel(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 1.0;
}

#[test]
fn kernel_static_has_correct_fields() {
    // Verify the static is truly static (no runtime allocation)
    let _ref: &'static KernelBinary = &WAVE_TEST_KERNEL_BINARY;
}

// ===========================================================================
// Wave inline push constant buffer
// ===========================================================================

// We cannot directly call set_value on a Wave without a GPU to create one,
// but we can verify the design constraints through type system checks.

#[test]
fn wave_constants() {
    // Verify the constants are accessible and reasonable
    // MAX_BINDINGS = 16, MAX_TEXTURES = 16, PUSH_DATA_CAP = 256
    // These are pub(crate) so we verify indirectly through the Wave size
    let wave_size = core::mem::size_of::<Wave>();
    // bindings: 16 * 8 = 128
    // texture_bindings: 16 * 8 = 128
    // push_data: 256
    // Together these make up at least 512 bytes of the Wave struct
    assert!(wave_size >= 512);
}

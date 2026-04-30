//! Vulkan-only end-to-end test for the sparse VkImage path —
//! step 063 slice 21.
//!
//! Validates that `gpu.sparse_texture(...)` on a Vulkan backend
//! actually creates a real sparse VkImage with
//! `VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
//! VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT` (no memory bound) on a
//! device that supports `sparseBinding`. On capable hardware the
//! call succeeds; on incapable hardware (e.g. RPi 5 V3D 7.1) it
//! must surface NotSupported with the slice-16 message.
//!
//! Run on a Vulkan host (Linux + lavapipe, RPi 5 lavapipe forced
//! via `VK_DRIVER_FILES`, or real GPU):
//!
//!     VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json \
//!     cargo test --test sparse_texture_vulkan \
//!         --no-default-features --features vulkan -- --nocapture

#![cfg(feature = "vulkan")]
#![cfg(not(feature = "metal"))]

use quanta::{QuantaErrorKind, TextureDesc};

fn try_vulkan() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn sparse_create_either_succeeds_or_surfaces_capability_error() {
    let Some(gpu) = try_vulkan() else {
        eprintln!("skipping: no Vulkan GPU available");
        return;
    };

    eprintln!(
        "vulkan device: name={:?} vendor={:?} supports_sparse={} supports_tess={} supports_mesh={} supports_rt={}",
        gpu.caps().name,
        gpu.caps().vendor,
        gpu.supports_sparse_residency(),
        gpu.supports_tessellation(),
        gpu.supports_mesh_shaders(),
        gpu.supports_ray_tracing(),
    );

    let r = gpu.sparse_texture(&TextureDesc {
        width: 256,
        height: 256,
        ..TextureDesc::default()
    });

    match r {
        Ok(st) => {
            // Slice 21 success path — VkImage created with sparse
            // flags, no memory bound. The handle is a live texture.
            assert_eq!(st.width(), 256);
            assert_eq!(st.height(), 256);
            assert!(
                gpu.supports_sparse_residency(),
                "sparse_texture succeeded but supports_sparse_residency reports false — caches out of sync"
            );
            eprintln!("sparse VkImage created — handle = {:#x}", st.handle());
        }
        Err(e) => {
            // Slice 16 gate fired — feature absent on this device.
            assert!(
                matches!(e.kind, QuantaErrorKind::NotSupported(_)),
                "expected NotSupported, got {:?}",
                e.kind
            );
            assert!(
                !gpu.supports_sparse_residency(),
                "sparse_texture failed with NotSupported but supports_sparse_residency reports true — caches out of sync"
            );
            eprintln!("sparse not supported on this device: {}", e);
        }
    }
}

#[test]
fn sparse_map_tile_still_returns_not_supported() {
    // Slice 22 (vkQueueBindSparse wiring) hasn't landed yet. The
    // typed wrapper's map_tile must continue to surface
    // NotSupported with the slice-7 message even after slice 21
    // creates a real sparse VkImage. This test will flip to
    // expecting Ok once slice 22 lands.
    let Some(gpu) = try_vulkan() else {
        eprintln!("skipping: no Vulkan GPU available");
        return;
    };

    let st = match gpu.sparse_texture(&TextureDesc {
        width: 256,
        height: 256,
        ..TextureDesc::default()
    }) {
        Ok(st) => st,
        Err(_) => {
            eprintln!("skipping: sparse not supported on this device");
            return;
        }
    };

    let backing = gpu.field::<u32>(64).expect("backing alloc");
    let r = st.map_tile(0, 0, 0, backing.handle());
    match r {
        Ok(()) => panic!("slice 22 not yet shipped — map_tile should NotSupported"),
        Err(e) => assert!(
            matches!(e.kind, QuantaErrorKind::NotSupported(_)),
            "expected NotSupported, got {:?}",
            e.kind
        ),
    }
}

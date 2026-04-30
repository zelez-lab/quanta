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
fn sparse_map_unmap_tile_native() {
    // Slice 22 — vkQueueBindSparse wiring shipped. Allocate a
    // backing field, map a tile, unmap it, then map two
    // different tiles. Each step must return Ok on a backend
    // with sparseBinding (lavapipe, real GPU) or skip when the
    // device gates fail.
    let Some(gpu) = try_vulkan() else {
        eprintln!("skipping: no Vulkan GPU available");
        return;
    };
    if !gpu.supports_sparse_residency() {
        eprintln!("skipping: sparse residency not supported on this device");
        return;
    }

    let st = gpu
        .sparse_texture(&TextureDesc {
            width: 1024,
            height: 1024,
            ..TextureDesc::default()
        })
        .expect("sparse_texture create");

    let backing = gpu.field::<u32>(64).expect("backing alloc");

    // T7602 refinement: map_tile on a live texture succeeds.
    st.map_tile(0, 0, 0, backing.handle())
        .expect("map_tile (0,0,0) → vkQueueBindSparse should succeed");
    st.map_tile(0, 1, 0, backing.handle())
        .expect("map_tile (0,1,0) → second tile should succeed");

    // T7602 boundary: re-mapping the same tile overwrites.
    st.map_tile(0, 0, 0, backing.handle())
        .expect("map_tile re-bind should succeed");

    // T7604 refinement: unmap_tile after map succeeds.
    st.unmap_tile(0, 0, 0).expect("unmap_tile (0,0,0)");

    // Idempotent unmap (filter semantics): unmapping an already-
    // unmapped tile is allowed.
    st.unmap_tile(0, 0, 0)
        .expect("unmap_tile of already-unmapped tile should be Ok");

    // Drop frees remaining bindings via the registry walker.
    drop(st);
}

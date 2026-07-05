//! End-to-end Metal sparse-residency test.
//!
//! Exercises `gpu.sparse_texture(...)` + `map_tile` + `unmap_tile`
//! on the Metal backend (Apple Silicon family 7+). On unsupported
//! hardware the test asserts the expected NotSupported category;
//! on supported hardware it must succeed end-to-end.
//!
//! Run: cargo test --features metal --test sparse_texture_metal -- --nocapture

#![cfg(feature = "metal")]

use quanta::{Format, QuantaErrorKind, TextureDesc};

fn try_metal() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

#[test]
fn sparse_create_either_succeeds_or_surfaces_capability_error() {
    let Some(gpu) = try_metal() else {
        eprintln!("skipping: no Metal GPU available");
        return;
    };
    eprintln!(
        "metal device: name={:?} supports_sparse={}",
        gpu.caps().name,
        gpu.supports_sparse_residency()
    );

    let r = gpu.sparse_texture(&TextureDesc::new(256, 256, Format::RGBA8));

    match r {
        Ok(st) => {
            assert!(
                gpu.supports_sparse_residency(),
                "sparse_texture succeeded but supports_sparse_residency() = false — caches out of sync"
            );
            assert_eq!(st.width(), 256);
            assert_eq!(st.height(), 256);
            eprintln!("Metal sparse texture created — handle = {:#x}", st.handle());
        }
        Err(e) => {
            assert!(
                matches!(e.kind, QuantaErrorKind::NotSupported(_)),
                "expected NotSupported, got {:?}",
                e.kind
            );
            eprintln!("Metal sparse not supported on this device: {}", e);
        }
    }
}

#[test]
fn sparse_map_unmap_tile_native() {
    let Some(gpu) = try_metal() else {
        eprintln!("skipping: no Metal GPU available");
        return;
    };
    if !gpu.supports_sparse_residency() {
        eprintln!("skipping: sparse residency not supported on this device");
        return;
    }

    let st = gpu
        .sparse_texture(&TextureDesc::new(1024, 1024, Format::RGBA8))
        .expect("Metal sparse_texture create");

    // Backing field is unused on Metal (placement heap supplies
    // pages) but the typed wrapper requires it for the contract.
    let backing = gpu.field::<u32>(64).expect("backing alloc");

    st.map_tile(0, 0, 0, backing.handle())
        .expect("map_tile (0,0,0) → updateTextureMapping(.map) should succeed");
    st.map_tile(0, 1, 0, backing.handle())
        .expect("map_tile (0,1,0) → second tile should succeed");

    st.map_tile(0, 0, 0, backing.handle())
        .expect("map_tile re-bind should succeed");

    st.unmap_tile(0, 0, 0).expect("unmap_tile (0,0,0)");

    // Idempotent unmap of an already-unmapped tile is allowed
    // per T7604; Metal's updateTextureMapping(.unmap) on an
    // unbound region is a no-op so this should pass cleanly.
    st.unmap_tile(0, 0, 0)
        .expect("unmap_tile of already-unmapped tile should be Ok");

    drop(st);
}

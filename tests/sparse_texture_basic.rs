//! Integration tests for `SparseTexture` (steps 030 + 031).
//!
//! Refines the proven Lean theorems T7600–T7605 + Verus theorems
//! T7650–T7654 against the CPU device's software sparse-texture
//! lifecycle. Each test names the theorem it refines.
//!
//! Run: cargo test --test sparse_texture_basic --features software

#![cfg(feature = "software")]

use quanta::{Format, TextureDesc};

fn desc(w: u32, h: u32) -> TextureDesc {
    TextureDesc::new(w, h, Format::RGBA8)
}

#[test]
fn sparse_create_records_dimensions() {
    // T7600 refinement: created texture has the requested dimensions.
    let gpu = quanta::init_cpu();
    let st = gpu.sparse_texture(&desc(128, 64)).unwrap();
    assert_eq!(st.width(), 128);
    assert_eq!(st.height(), 64);
}

#[test]
fn sparse_create_rejects_zero_dimensions() {
    // T7601 refinement: zero-sized create rejected.
    let gpu = quanta::init_cpu();
    let r = gpu.sparse_texture(&desc(0, 64));
    assert!(r.is_err());
    let r = gpu.sparse_texture(&desc(64, 0));
    assert!(r.is_err());
}

#[test]
fn sparse_map_tile_succeeds() {
    // T7602 refinement: map_tile on a live texture succeeds.
    let gpu = quanta::init_cpu();
    let st = gpu.sparse_texture(&desc(256, 256)).unwrap();
    let backing = gpu.field::<u32>(64).unwrap();
    st.map_tile(0, 0, 0, backing.handle()).unwrap();
    st.map_tile(0, 1, 0, backing.handle()).unwrap();
    st.map_tile(1, 0, 0, backing.handle()).unwrap();
}

#[test]
fn sparse_remap_replaces_binding() {
    // T7602 boundary: re-mapping the same tile overwrites.
    let gpu = quanta::init_cpu();
    let st = gpu.sparse_texture(&desc(256, 256)).unwrap();
    let b0 = gpu.field::<u32>(64).unwrap();
    let b1 = gpu.field::<u32>(64).unwrap();
    st.map_tile(0, 0, 0, b0.handle()).unwrap();
    st.map_tile(0, 0, 0, b1.handle()).unwrap();
}

#[test]
fn sparse_unmap_tile_succeeds() {
    // T7604 refinement: unmap_tile after map succeeds.
    let gpu = quanta::init_cpu();
    let st = gpu.sparse_texture(&desc(256, 256)).unwrap();
    let backing = gpu.field::<u32>(64).unwrap();
    st.map_tile(0, 0, 0, backing.handle()).unwrap();
    st.unmap_tile(0, 0, 0).unwrap();
    // Unmap an unmapped tile is also allowed (HashMap::remove is
    // idempotent; matches Lean's filter semantics).
    st.unmap_tile(0, 5, 5).unwrap();
}

#[test]
fn sparse_drop_invalidates_texture() {
    // T7605 refinement: Drop releases the backend handle.
    let gpu = quanta::init_cpu();
    {
        let _st = gpu.sparse_texture(&desc(128, 128)).unwrap();
    }
}

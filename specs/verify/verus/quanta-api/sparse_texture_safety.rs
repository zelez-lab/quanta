//! Verus mirror — sparse texture invariants (steps 030 + 031).
//!
//! Mirrors `Quanta.SparseTexture.Texture` from Lean. Every backend
//! that implements sparse textures (Vulkan VK_EXT_sparse_binding,
//! Metal MTLHeap-backed sparse textures) refines this contract:
//!
//! - `create(w, h)` returns a fresh texture with an empty tile map,
//!   live = true, iff w >= 1 and h >= 1.
//! - `map_tile(tex, k, b)` succeeds iff the texture is live; the
//!   target tile is rebound to backing handle b.
//! - `unmap_tile(tex, k)` succeeds iff live; the target tile is
//!   removed from the map.
//! - `destroy(tex)` flips live to false.
//!
//! Theorems mirror Lean T7600-T7605:
//!   T7650 — fresh texture matches Lean shape.
//!   T7651 — map then lookup returns the bound handle.
//!   T7652 — map preserves dimensions + live.
//!   T7653 — unmap then lookup returns None.
//!   T7654 — destroy invalidates + blocks map/unmap.

use vstd::prelude::*;

verus! {

pub struct TileKey {
    pub mip: nat,
    pub x: nat,
    pub y: nat,
}

pub struct SparseTexture {
    pub handle: u64,
    pub width: nat,
    pub height: nat,
    pub tiles: Map<TileKey, u64>,
    pub live: bool,
}

pub open spec fn create(handle: u64, w: nat, h: nat) -> Option<SparseTexture> {
    if 1nat <= w && 1nat <= h {
        Option::Some(SparseTexture {
            handle,
            width: w,
            height: h,
            tiles: Map::empty(),
            live: true,
        })
    } else {
        Option::None
    }
}

pub open spec fn map_tile(t: SparseTexture, k: TileKey, b: u64) -> Option<SparseTexture> {
    if t.live {
        Option::Some(SparseTexture {
            tiles: t.tiles.insert(k, b),
            ..t
        })
    } else {
        Option::None
    }
}

pub open spec fn unmap_tile(t: SparseTexture, k: TileKey) -> Option<SparseTexture> {
    if t.live {
        Option::Some(SparseTexture {
            tiles: t.tiles.remove(k),
            ..t
        })
    } else {
        Option::None
    }
}

pub open spec fn destroy(t: SparseTexture) -> SparseTexture {
    SparseTexture { live: false, ..t }
}

// ── T7650: fresh texture matches Lean shape ───────────────────────────────

proof fn t7650_create_fresh(handle: u64, w: nat, h: nat)
    requires
        1nat <= w,
        1nat <= h,
    ensures
        create(handle, w, h) matches Option::Some(t) ==>
            t.handle == handle
            && t.width == w
            && t.height == h
            && t.tiles == Map::<TileKey, u64>::empty()
            && t.live == true,
{}

// ── T7651: map then lookup returns bound handle ──────────────────────────

proof fn t7651_map_then_contains(t: SparseTexture, k: TileKey, b: u64, t2: SparseTexture)
    requires
        t.live,
        map_tile(t, k, b) == Option::<SparseTexture>::Some(t2),
    ensures
        t2.tiles.contains_key(k),
        t2.tiles[k] == b,
{}

// ── T7652: map preserves dimensions + live ───────────────────────────────

proof fn t7652_map_preserves(t: SparseTexture, k: TileKey, b: u64, t2: SparseTexture)
    requires
        map_tile(t, k, b) == Option::<SparseTexture>::Some(t2),
    ensures
        t2.handle == t.handle,
        t2.width == t.width,
        t2.height == t.height,
        t2.live == t.live,
{}

// ── T7653: unmap then lookup returns None ────────────────────────────────

proof fn t7653_unmap_then_no_contains(t: SparseTexture, k: TileKey, t2: SparseTexture)
    requires
        t.live,
        unmap_tile(t, k) == Option::<SparseTexture>::Some(t2),
    ensures
        !t2.tiles.contains_key(k),
{}

// ── T7654: destroy invalidates + blocks map/unmap + idempotent ───────────

proof fn t7654_destroy_invalidates(t: SparseTexture)
    ensures
        destroy(t).live == false,
        destroy(t).handle == t.handle,
        destroy(t).width == t.width,
        destroy(t).height == t.height,
{}

proof fn t7654b_destroy_blocks_map(t: SparseTexture, k: TileKey, b: u64)
    ensures
        map_tile(destroy(t), k, b) == Option::<SparseTexture>::None,
{}

proof fn t7654c_destroy_blocks_unmap(t: SparseTexture, k: TileKey)
    ensures
        unmap_tile(destroy(t), k) == Option::<SparseTexture>::None,
{}

}  // verus!

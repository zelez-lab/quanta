//! Device-side pool of builder-managed MSAA intermediates.
//!
//! `RenderBuilder::msaa(n)` (in `quanta-render`) redirects a pass to an
//! n-sample intermediate texture matching the pass's final target, so
//! callers never hand-manage the MSAA lifecycle (allocate the
//! intermediate, keep samples alive across sub-passes with `Store`,
//! resolve at frame end). The intermediates live here, on the shared
//! [`Gpu`](crate::Gpu) wrapper, beside the driver's own registries —
//! one pool per device, shared by every `Gpu` clone.
//!
//! ## Keying
//!
//! The pool has two lanes, split by [`MsaaPoolKey::for_target`] on
//! whether the pass's final target OWNS its driver texture:
//!
//! - **Owned targets** (`render_target` and friends) key by the
//!   target's driver handle. Driver handles are never reused, and a
//!   texture's dimensions/format are immutable, so a key can never
//!   silently alias a different target. An entry is created on first
//!   use and reused by every later `.msaa(n)` pass over the same
//!   target; a sample-count change (`.msaa(2)` after `.msaa(4)`) — or,
//!   defensively, a dimension/format mismatch (impossible for a live
//!   handle) — evicts and recreates it.
//! - **Surface frames** (swapchain aliases from `surface_acquire` —
//!   the wrapper does not own the driver texture, and a FRESH handle
//!   is minted every acquire) key by shape: `(width, height, format,
//!   samples)`. Keying these by handle would make every entry
//!   unreachable after present and grow the pool by one full-size MSAA
//!   texture per frame — the windowed-session leak. Per-shape keying
//!   gives one intermediate per surface configuration, reused by every
//!   frame of that configuration.
//!
//! ## Why frames of one shape may share one intermediate
//!
//! All render submissions on a device go down its single in-order
//! queue, and a `.msaa(n)` pass writes the intermediate and (when
//! resolving) reads it back within that one submission. Two frames in
//! flight are two queued submissions: the second's accesses to the
//! shared intermediate are ordered after the first's by exactly the
//! edges that already order consecutive `.msaa(n)` passes over one
//! owned target — the pool's designed reuse (Vulkan: the pre-pass
//! attachment barriers in `driver/vulkan/render`; Metal: the in-order
//! queue plus default hazard tracking). Per-shape sharing therefore
//! introduces no hazard the per-target lane doesn't already carry.
//!
//! The one semantic consequence: cross-pass sample PERSISTENCE
//! (`Store` without resolve, `.load()` in a later pass) on surface
//! frames is per shape, not per acquired frame. Interleaving
//! unresolved `.msaa(n)` passes between two live frames of the same
//! shape clobbers the earlier frame's samples. The frame loop the
//! `Surface` docs describe — acquire, render (any number of passes),
//! present — is unaffected; a caller that really needs two
//! concurrently persistent intermediates owns them through the manual
//! path (`msaa_target` + explicit `ColorTarget` + `resolve_texture`).
//!
//! ## Eviction and destroy safety
//!
//! - **Owned lane**: a mismatch evicts and recreates (above).
//!   Dropping the final target does NOT evict its entry — the pool
//!   cannot observe `Texture::drop`, so the (unreachable) entry holds
//!   its GPU memory until the device is torn down. Long-lived frame
//!   targets (the intended use) never notice; an app churning through
//!   short-lived `.msaa()`-rendered targets should prefer the manual
//!   path, which gives it ownership of the intermediate.
//! - **Frame lane**: entries unused for the last `FRAME_SHAPE_KEEP`
//!   frame-lane lookups are trimmed on the next lookup, so stale
//!   shapes (a resize storm mints one per step; a closed window
//!   leaves one behind) are reclaimed instead of accumulating. The
//!   lane's size is bounded by the shapes in active use plus at most
//!   `FRAME_SHAPE_KEEP` stragglers awaiting the next trim.
//! - **Destroying an evicted intermediate is safe at any point**,
//!   because destruction defers to GPU completion in the drivers, not
//!   here: Vulkan `texture_destroy` unregisters the handle (no later
//!   submission can bind it) and parks the image in the fence-gated
//!   retire bin (`driver/vulkan/retire.rs`); Metal command buffers
//!   hold their own references to every resource they encode (the
//!   driver never uses unretained command buffers), so releasing the
//!   pool's reference leaves in-flight passes intact; WebGPU defers
//!   deallocation of a destroyed-in-use texture by spec; the CPU
//!   driver has no surface path and executes synchronously.
//!
//! Whatever survives eviction drops with the last `Gpu` clone, which
//! drops the pool and with it every pooled `Texture`.

use alloc::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::render_pass::ColorTarget;
use crate::{Format, GpuDevice, QuantaError, Texture, TextureDesc, TextureUsage};

/// Frame-lane retention horizon: a per-shape intermediate unused for
/// this many frame-lane lookups is destroyed on the next lookup. One
/// `.msaa(n)` pass over a surface frame = one lookup, so a shape's
/// lookup gap is the number of windowed shapes alternating on the
/// device — anything in active per-frame use survives comfortably,
/// while a resize storm's single-lookup shapes die within the next
/// eight windowed msaa passes.
const FRAME_SHAPE_KEEP: u64 = 8;

/// How the pool keys a pass's final target. Computed by
/// [`MsaaPoolKey::for_target`] while the `&Texture` is in hand
/// (`RenderBuilder::new`) and carried to `pulse()`. Internal machinery
/// of `RenderBuilder::msaa` — not part of the stable public surface.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum MsaaPoolKey {
    /// The target owns its driver texture (`render_target` & co):
    /// entry per driver handle — handles are never reused.
    Target(u64),
    /// The target is a swapchain frame alias (`surface_acquire` mints
    /// a fresh handle every acquire): entry per `(width, height,
    /// format, samples)` shape, shared by every frame of that
    /// configuration. The shape arrives with the lookup, so the
    /// variant carries no data.
    Frame,
}

impl MsaaPoolKey {
    /// Classify `target`. Surface frames are the only user-visible
    /// `Texture`s that do not own their driver resource (the swapchain
    /// does), so ownership is the discriminator.
    pub fn for_target(target: &Texture) -> Self {
        if target.live {
            Self::Target(target.handle)
        } else {
            Self::Frame
        }
    }
}

/// One pooled intermediate in the owned-target lane: the n-sample
/// texture plus the sample count it was created with (the texture also
/// carries it; kept explicit for the eviction check).
struct MsaaEntry {
    samples: u32,
    texture: Texture,
}

impl MsaaEntry {
    fn create(
        device: &Arc<dyn GpuDevice>,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<Self, QuantaError> {
        Ok(Self {
            samples,
            texture: create_intermediate(device, width, height, format, samples)?,
        })
    }

    fn matches(&self, width: u32, height: u32, format: Format, samples: u32) -> bool {
        self.samples == samples
            && self.texture.width() == width
            && self.texture.height() == height
            && self.texture.format() == format
    }
}

/// One pooled intermediate in the frame lane. Its shape lives in the
/// map key; `last_used` is the frame-lane lookup counter at the
/// entry's most recent use, driving the staleness trim.
struct FrameEntry {
    last_used: u64,
    texture: Texture,
}

/// Create an n-sample intermediate. Same shape as
/// `RenderGpu::msaa_target`: render-target usage only — the
/// intermediate is drawn into and resolved from, never sampled.
fn create_intermediate(
    device: &Arc<dyn GpuDevice>,
    width: u32,
    height: u32,
    format: Format,
    samples: u32,
) -> Result<Texture, QuantaError> {
    let desc = TextureDesc::new(width, height, format)
        .with_sample_count(samples)
        .with_usage(TextureUsage::RENDER_TARGET);
    let mut texture = device.texture_create(&desc)?;
    // Attach the device so dropping the entry (eviction, pool
    // teardown) releases the driver resource.
    texture.device = Some(device.clone());
    Ok(texture)
}

/// The per-device pool of builder-managed MSAA intermediates. See the
/// [module docs](self) for keying and lifetime. Internal machinery of
/// `RenderBuilder::msaa` — not part of the stable public surface.
#[doc(hidden)]
#[derive(Default)]
pub struct MsaaPool {
    state: Mutex<PoolState>,
}

#[derive(Default)]
struct PoolState {
    /// Owned-target lane: driver handle → intermediate.
    targets: HashMap<u64, MsaaEntry>,
    /// Frame lane: `(width, height, format, samples)` → intermediate.
    frames: HashMap<(u32, u32, Format, u32), FrameEntry>,
    /// Frame-lane lookup counter; stamps [`FrameEntry::last_used`].
    frame_lookups: u64,
}

impl MsaaPool {
    /// Get-or-create the pooled `samples`-sample intermediate for
    /// `key` and return a [`ColorTarget`] over it (default ops — the
    /// caller overrides load/store). Evictions (owned-lane mismatch,
    /// frame-lane staleness trim) drop the old intermediate here; the
    /// drivers defer the actual destroy past in-flight work (see the
    /// [module docs](self)).
    pub fn intermediate_color_target(
        &self,
        device: &Arc<dyn GpuDevice>,
        key: MsaaPoolKey,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<ColorTarget, QuantaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| QuantaError::internal("MSAA intermediate pool lock poisoned"))?;
        match key {
            MsaaPoolKey::Target(handle) => match state.targets.entry(handle) {
                std::collections::hash_map::Entry::Occupied(mut slot) => {
                    if !slot.get().matches(width, height, format, samples) {
                        // Evict + recreate: replacing the entry drops
                        // the old intermediate, releasing its driver
                        // texture.
                        *slot.get_mut() =
                            MsaaEntry::create(device, width, height, format, samples)?;
                    }
                    Ok(ColorTarget::new(&slot.get().texture))
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let entry = MsaaEntry::create(device, width, height, format, samples)?;
                    Ok(ColorTarget::new(&slot.insert(entry).texture))
                }
            },
            MsaaPoolKey::Frame => {
                state.frame_lookups += 1;
                let now = state.frame_lookups;
                let target = match state.frames.entry((width, height, format, samples)) {
                    std::collections::hash_map::Entry::Occupied(slot) => {
                        let entry = slot.into_mut();
                        entry.last_used = now;
                        ColorTarget::new(&entry.texture)
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        let entry = FrameEntry {
                            last_used: now,
                            texture: create_intermediate(device, width, height, format, samples)?,
                        };
                        ColorTarget::new(&slot.insert(entry).texture)
                    }
                };
                // Staleness trim. The entry just touched has
                // `last_used == now` and always survives.
                state
                    .frames
                    .retain(|_, e| now - e.last_used <= FRAME_SHAPE_KEEP);
                Ok(target)
            }
        }
    }
}

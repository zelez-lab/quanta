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
//! ## Keying and lifetime
//!
//! Entries are keyed by the **final target's driver handle**. Driver
//! handles are never reused, and a texture's dimensions/format are
//! immutable, so a key can never silently alias a different target.
//! An entry is created on first use and reused by every later
//! `.msaa(n)` pass over the same target:
//!
//! - **Sample-count change** (`.msaa(2)` after `.msaa(4)` on the same
//!   target) evicts and recreates the entry. The old intermediate is
//!   destroyed on eviction — callers must not change `n` for a target
//!   while a pass using the old intermediate is still in flight.
//! - **Dimension/format mismatch** (impossible for a live handle, but
//!   checked defensively) also evicts and recreates.
//! - **Dropping the final target does NOT evict its entry.** The pool
//!   cannot observe `Texture::drop`, so the (unreachable) entry holds
//!   its GPU memory until the device is torn down — i.e. until the
//!   last `Gpu` clone drops, which drops the pool and with it every
//!   pooled `Texture`. Long-lived frame targets (the intended use)
//!   never notice; an app that churns through short-lived
//!   `.msaa()`-rendered targets should prefer the manual path
//!   (`msaa_target` + explicit `ColorTarget` + `resolve_texture`),
//!   which gives it ownership of the intermediate.

use alloc::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::render_pass::ColorTarget;
use crate::{Format, GpuDevice, QuantaError, Texture, TextureDesc, TextureUsage};

/// One pooled intermediate: the n-sample texture plus the sample count
/// it was created with (the texture also carries it; kept explicit for
/// the eviction check).
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
        // Same shape as `RenderGpu::msaa_target`: render-target usage
        // only — the intermediate is drawn into and resolved from,
        // never sampled.
        let desc = TextureDesc::new(width, height, format)
            .with_sample_count(samples)
            .with_usage(TextureUsage::RENDER_TARGET);
        let mut texture = device.texture_create(&desc)?;
        // Attach the device so dropping the entry (eviction, pool
        // teardown) releases the driver resource.
        texture.device = Some(device.clone());
        Ok(Self { samples, texture })
    }

    fn matches(&self, width: u32, height: u32, format: Format, samples: u32) -> bool {
        self.samples == samples
            && self.texture.width() == width
            && self.texture.height() == height
            && self.texture.format() == format
    }
}

/// The per-device pool of builder-managed MSAA intermediates. See the
/// [module docs](self) for keying and lifetime. Internal machinery of
/// `RenderBuilder::msaa` — not part of the stable public surface.
#[doc(hidden)]
#[derive(Default)]
pub struct MsaaPool {
    entries: Mutex<HashMap<u64, MsaaEntry>>,
}

impl MsaaPool {
    /// Get-or-create the pooled `samples`-sample intermediate for
    /// `target_handle` and return a [`ColorTarget`] over it (default
    /// ops — the caller overrides load/store). Evicts and recreates on
    /// a sample-count (or, defensively, dimension/format) mismatch.
    pub fn intermediate_color_target(
        &self,
        device: &Arc<dyn GpuDevice>,
        target_handle: u64,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<ColorTarget, QuantaError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| QuantaError::internal("MSAA intermediate pool lock poisoned"))?;
        match entries.entry(target_handle) {
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                if !slot.get().matches(width, height, format, samples) {
                    // Evict + recreate: replacing the entry drops the
                    // old intermediate, destroying its driver texture.
                    *slot.get_mut() = MsaaEntry::create(device, width, height, format, samples)?;
                }
                Ok(ColorTarget::new(&slot.get().texture))
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                let entry = MsaaEntry::create(device, width, height, format, samples)?;
                Ok(ColorTarget::new(&slot.insert(entry).texture))
            }
        }
    }
}

//! Pooled intermediates for render GROUPS — offscreen layers rendered
//! by one pass and sampled by a later pass in the same frame.
//!
//! Differs from [`MsaaPool`](super::msaa_pool) in one structural way:
//! group textures are CHECKED OUT. Nested groups mean several
//! same-shaped intermediates live at once, so the pool is a free-list
//! per shape, not a single slot per key: `checkout` pops a matching
//! returned texture (or creates one), and the handle returns it on
//! Drop. Returned entries carry a use-stamp; a staleness trim on
//! checkout evicts shapes a consumer stopped using (the windowed-MSAA
//! lesson — pools keyed by dead shapes otherwise grow forever).
//! Eviction drops the `Texture`, whose attached device routes the
//! driver destroy through the deferred/retire path — never mid-flight.
//!
//! Internal machinery of `Gpu::render_group` — not part of the stable
//! public surface.

use alloc::sync::Arc;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::{Format, GpuDevice, QuantaError, Texture, TextureDesc, TextureUsage};

/// Checkouts between trims after which an unreused returned entry is
/// evicted. Mirrors the MSAA frame-lane staleness discipline.
const STALE_AFTER: u64 = 64;

/// One returned (idle) intermediate awaiting reuse.
struct Idle {
    last_used: u64,
    texture: Texture,
}

#[derive(Default)]
struct PoolState {
    /// Free lists: `(width, height, format, samples)` → returned
    /// textures of that shape.
    idle: HashMap<(u32, u32, Format, u32), Vec<Idle>>,
    /// Checkout counter; stamps `Idle::last_used` on return.
    checkouts: u64,
}

/// The per-device pool of group intermediates. See the module docs.
#[doc(hidden)]
#[derive(Default)]
pub struct GroupPool {
    state: Mutex<PoolState>,
}

impl GroupPool {
    /// Pop a matching idle intermediate or create one. The texture is
    /// renderable AND sampleable — a group exists to be drawn into and
    /// then bound.
    pub fn checkout(
        &self,
        device: &Arc<dyn GpuDevice>,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<Texture, QuantaError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| QuantaError::internal("group intermediate pool lock poisoned"))?;
        state.checkouts += 1;
        let now = state.checkouts;
        // Staleness trim: drop idle entries no checkout has touched in
        // STALE_AFTER checkouts (destroys defer past in-flight work).
        state.idle.retain(|_, list| {
            list.retain(|e| now.saturating_sub(e.last_used) <= STALE_AFTER);
            !list.is_empty()
        });
        if let Some(list) = state.idle.get_mut(&(width, height, format, samples))
            && let Some(entry) = list.pop()
        {
            return Ok(entry.texture);
        }
        drop(state);
        let desc = TextureDesc::new(width, height, format)
            .with_sample_count(samples)
            .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ));
        let mut texture = device.texture_create(&desc)?;
        // Attach the device so an evicted/unreturned entry releases its
        // driver resource on drop.
        texture.device = Some(device.clone());
        Ok(texture)
    }

    /// Return a checked-out intermediate to its shape's free list.
    pub fn give_back(&self, texture: Texture) {
        let Ok(mut state) = self.state.lock() else {
            // Poisoned pool: let the texture drop — the deferred destroy
            // path still runs.
            return;
        };
        let key = (
            texture.width(),
            texture.height(),
            texture.format(),
            texture.sample_count(),
        );
        let last_used = state.checkouts;
        state
            .idle
            .entry(key)
            .or_default()
            .push(Idle { last_used, texture });
    }
}

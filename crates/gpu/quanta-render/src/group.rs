//! Render groups — pooled offscreen layers.
//!
//! A GROUP is the structural form of the offscreen-compositing idiom:
//! draw a set of primitives into an intermediate texture, then bind
//! that texture in a later pass of the same frame (UI layer trees,
//! post-processing chains, cached backdrops). The intermediate comes
//! from the device's [`GroupPool`] — callers never size, create, or
//! free layer textures, and nested groups just work (each group is its
//! own pass; attachments never switch mid-pass, which Metal and Vulkan
//! forbid anyway).
//!
//! Ordering: the group's pass is submitted when its closure pulses, so
//! any LATER pass on the same `Gpu` that samples the group sees the
//! finished contents (submission order + the render-then-sample layout
//! transition the drivers already guarantee). Waiting is only needed
//! to read the layer back on the host.

use alloc::sync::Arc;

use quanta_core::{GroupPool, Texture};

/// A pooled offscreen layer returned by
/// [`RenderGpu::render_group`](crate::RenderGpu::render_group).
///
/// Dereferences to [`Texture`], so it binds anywhere a texture does
/// (`builder.texture(slot, &layer)`). On drop the texture returns to
/// the device's group pool for reuse; an in-flight pass keeps the
/// driver resource alive through the deferred-destroy machinery, so
/// dropping the handle is always safe.
pub struct GroupTexture {
    texture: Option<Texture>,
    pool: Arc<GroupPool>,
}

impl GroupTexture {
    pub(crate) fn new(texture: Texture, pool: Arc<GroupPool>) -> Self {
        Self {
            texture: Some(texture),
            pool,
        }
    }
}

impl core::ops::Deref for GroupTexture {
    type Target = Texture;
    fn deref(&self) -> &Texture {
        self.texture
            .as_ref()
            .expect("GroupTexture texture present until drop")
    }
}

impl Drop for GroupTexture {
    fn drop(&mut self) {
        if let Some(texture) = self.texture.take() {
            self.pool.give_back(texture);
        }
    }
}

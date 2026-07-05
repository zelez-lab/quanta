//! Presentation surfaces — Quanta-owned present.
//!
//! A [`Surface`] is a swapchain over a platform presentation target.
//! The frame loop is:
//!
//! ```ignore
//! let mut surface = gpu.create_surface(&target, &config)?;
//! loop {
//!     let frame = match surface.acquire() {
//!         Ok(frame) => frame,
//!         Err(e) if matches!(e.kind, QuantaErrorKind::SurfaceOutdated(_)) => {
//!             surface.configure(new_config)?; // window resized — reconfigure
//!             continue;
//!         }
//!         Err(e) => return Err(e),
//!     };
//!     // Render into the frame through the ordinary render-pass API.
//!     let mut pulse = gpu.render(frame.texture())?.clear(color).pulse()?;
//!     // Present after the pass is SUBMITTED (pulse() has run). No
//!     // CPU wait is required: presentation is ordered after the
//!     // submitted GPU work by the driver, asynchronously.
//!     frame.present()?;
//! }
//! ```
//!
//! The configuration types the drivers speak ([`SurfaceConfig`],
//! [`SurfaceTarget`](quanta_core::SurfaceTarget), `PresentMode`) live in `quanta-core`.
//!
//! This is one of Quanta's two presentation models. The other —
//! [`Texture::native_handle`](quanta_core::Texture::native_handle) —
//! exports the rendered texture so an external compositor owns present
//! instead.

use alloc::sync::Arc;

use quanta_core::{GpuDevice, QuantaError, SurfaceConfig, Texture};

/// A swapchain over a platform presentation target. Created with
/// [`RenderGpu::create_surface`](crate::RenderGpu::create_surface);
/// Quanta owns present. See the [module docs](self) for the frame loop.
///
/// Dropping the `Surface` releases the swapchain (and, for
/// [`SurfaceTarget::Headless`](quanta_core::SurfaceTarget::Headless), the backend-created target).
pub struct Surface {
    pub(crate) handle: u64,
    pub(crate) config: SurfaceConfig,
    pub(crate) device: Arc<dyn GpuDevice>,
}

impl Surface {
    /// The active configuration.
    pub fn config(&self) -> &SurfaceConfig {
        &self.config
    }

    /// Current frame width in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Current frame height in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Reconfigure the surface — resize, format or present-mode
    /// change. Call after the presentation target was resized
    /// (typically on a `SurfaceOutdated` error
    /// ([`QuantaErrorKind::SurfaceOutdated`](quanta_core::QuantaErrorKind))
    /// from [`acquire`](Surface::acquire)). Frames acquired before
    /// the reconfigure must be presented or dropped first.
    pub fn configure(&mut self, config: SurfaceConfig) -> Result<(), QuantaError> {
        self.device.surface_configure(self.handle, &config)?;
        self.config = config;
        Ok(())
    }

    /// Acquire the next presentable frame.
    ///
    /// Blocks briefly if no frame is available yet (all in flight);
    /// returns `Timeout` if none became available within the
    /// backend's deadline (retry next loop iteration), or
    /// `SurfaceOutdated` when the target no longer matches the
    /// configuration (reconfigure with the new extent, then retry).
    pub fn acquire(&mut self) -> Result<SurfaceFrame, QuantaError> {
        let (frame, mut texture) = self.device.surface_acquire(self.handle)?;
        texture.__attach_device(self.device.clone());
        Ok(SurfaceFrame {
            surface: self.handle,
            frame,
            texture,
            device: self.device.clone(),
            presented: false,
        })
    }
}

impl core::fmt::Debug for Surface {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Surface")
            .field("handle", &self.handle)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        // Best-effort: the driver default no-ops.
        let _ = self.device.surface_destroy(self.handle);
    }
}

/// One acquired, presentable frame of a [`Surface`].
///
/// **Lifetime contract (freeze-critical):**
/// [`texture`](SurfaceFrame::texture) aliases the swapchain's backing image — it
/// is a borrow owned by the swapchain, valid only until the frame is
/// presented or dropped. Do not store the texture or its
/// [`native_handle`](Texture::native_handle) beyond the frame; acquire
/// a fresh frame each iteration. Dropping an unpresented frame
/// discards it (the image returns to the swapchain unshown).
pub struct SurfaceFrame {
    pub(crate) surface: u64,
    pub(crate) frame: u64,
    pub(crate) texture: Texture,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) presented: bool,
}

impl SurfaceFrame {
    /// The frame's target texture. Render into it through the
    /// ordinary render-pass API (`gpu.render(frame.texture())`).
    /// Valid only until present/drop — see the type docs.
    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    /// Present this frame, consuming it.
    ///
    /// Call after the render pass targeting
    /// [`texture`](SurfaceFrame::texture) has been **submitted**
    /// (`.pulse()` returned). Presentation is ordered after that
    /// submitted GPU work by the driver; the call returns without
    /// waiting for the GPU or the display — no `Pulse::wait` is
    /// needed between submit and present.
    pub fn present(mut self) -> Result<(), QuantaError> {
        self.presented = true;
        self.device.surface_present(self.surface, self.frame)
    }
}

impl core::fmt::Debug for SurfaceFrame {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SurfaceFrame")
            .field("surface", &self.surface)
            .field("frame", &self.frame)
            .field("presented", &self.presented)
            .finish_non_exhaustive()
    }
}

impl Drop for SurfaceFrame {
    fn drop(&mut self) {
        if !self.presented {
            // Discard: return the image to the swapchain unshown.
            let _ = self.device.surface_discard(self.surface, self.frame);
        }
    }
}

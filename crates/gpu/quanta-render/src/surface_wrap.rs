//! Presentation surfaces — Quanta-owned present.
//!
//! A [`Surface`] is a swapchain over a platform presentation target.
//! The frame loop is one closure per frame
//! ([`render_frame`](Surface::render_frame) — acquire, render,
//! present, with resize self-healing built in):
//!
//! ```ignore
//! // `window` is anything winit-style (rwh 0.6 HasWindowHandle +
//! // HasDisplayHandle); feature `raw-window-handle`.
//! let target = SurfaceTarget::from_window(&window)?;
//! let mut surface = gpu.create_surface(&target, &config)?;
//! loop {
//!     surface.render_frame(|frame| {
//!         // Render into the frame through the ordinary render-pass
//!         // API. Present after the pass is SUBMITTED (pulse() has
//!         // run) — render_frame presents on Ok, right after the
//!         // closure; no CPU wait is required in between, because the
//!         // driver orders presentation after the submitted GPU work,
//!         // asynchronously.
//!         gpu.render(frame.texture())?.clear(color).pulse()?;
//!         Ok(())
//!     })?;
//! }
//! ```
//!
//! or, spelled out over the primitives ([`acquire`](Surface::acquire) /
//! [`SurfaceFrame::present`]) when the loop needs custom
//! resize/timeout policy:
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
//!     let mut pulse = gpu.render(frame.texture())?.clear(color).pulse()?;
//!     frame.present()?;
//! }
//! ```
//!
//! ## Pacing: fully demand-driven
//!
//! Quanta never renders or presents on its own — a frame happens only
//! when the caller runs acquire → render → present. There is no
//! internal timer, display link, or frame scheduler, so the loop above
//! may run at ANY cadence: seconds between frames (an idle UI waiting
//! on a dirty flag), a burst at input rate, or a steady animation
//! clock. An idle surface holds no acquired frame and costs zero GPU
//! or CPU work; nothing leaks or stalls across idle gaps. The only
//! back-pressure is [`acquire`](Surface::acquire) itself: when every
//! swapchain image is still in flight (a burst faster than the display
//! consumes), it blocks briefly — which throttles the loop to the
//! present rate. Pinned by `tests/gpu_surface.rs`
//! (`surface_sparse_then_burst_cadence`,
//! `metal_layer_demand_driven_cadence`).
//!
//! The configuration types the drivers speak ([`SurfaceConfig`],
//! [`SurfaceTarget`](quanta_core::SurfaceTarget), `PresentMode`) live in `quanta-core`.
//!
//! This is one of Quanta's two presentation models. The other —
//! [`Texture::native_handle`](quanta_core::Texture::native_handle) —
//! exports the rendered texture so an external compositor owns present
//! instead.

use quanta_core::{Gpu, QuantaError, QuantaErrorKind, SurfaceConfig, Texture};

/// A swapchain over a platform presentation target. Created with
/// [`RenderGpu::create_surface`](crate::RenderGpu::create_surface);
/// Quanta owns present. See [`render_frame`](Surface::render_frame)
/// for the frame loop.
///
/// Dropping the `Surface` releases the swapchain (and, for
/// [`SurfaceTarget::Headless`](quanta_core::SurfaceTarget::Headless), the backend-created target).
pub struct Surface {
    pub(crate) handle: u64,
    pub(crate) config: SurfaceConfig,
    /// The owning device handle — the driver for the swapchain calls,
    /// the pending lane for the present (which submits what the
    /// frame's passes encoded before showing the frame).
    pub(crate) gpu: Gpu,
}

impl Surface {
    /// The active configuration.
    pub fn config(&self) -> &SurfaceConfig {
        &self.config
    }

    /// The pixel [`Format`](quanta_core::Format) the surface's frames
    /// actually use.
    ///
    /// This is the **negotiated** format, which on Vulkan may differ from
    /// the [`SurfaceConfig::format`] you requested:
    /// [`SurfaceConfig::format`] is a *preference*, and the surface may
    /// only offer a different one (an Android surface offering `RGBA8`
    /// where you asked for `BGRA8`, say). On Metal it always equals the
    /// configured format. Every [`SurfaceFrame::texture`]'s
    /// [`format()`](quanta_core::Texture::format) reports exactly this.
    ///
    /// Call it after `create_surface` and **before building pipelines**,
    /// and pass the result to
    /// [`PipelineDesc::with_color_formats`](quanta_core::PipelineDesc) —
    /// otherwise a pipeline typed for the requested format is rejected at
    /// encode time against a frame carrying the negotiated one. The
    /// preference chain is fixed at `[requested, BGRA8, RGBA8]`; a
    /// consumer needing to order the fallback differently should build the
    /// pipeline per acquired frame from `frame.texture().format()`
    /// instead.
    pub fn format(&self) -> Result<quanta_core::Format, QuantaError> {
        self.gpu.device_handle().surface_format(self.handle)
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
        self.gpu
            .device_handle()
            .surface_configure(self.handle, &config)?;
        self.config = config;
        Ok(())
    }

    /// One frame, one closure: acquire → render (the closure) →
    /// present, with resize self-healing built in.
    ///
    /// The closure receives the acquired [`SurfaceFrame`] and renders
    /// into `frame.texture()` through the ordinary render-pass API,
    /// submitting with `.pulse()`. When it returns `Ok`, the frame is
    /// presented and the closure's value returned; on `Err` the frame
    /// drops **unpresented** (the image returns to the swapchain
    /// unshown) and the error propagates — the loop keeps working on
    /// the next call.
    ///
    /// ```ignore
    /// loop {
    ///     surface.render_frame(|frame| {
    ///         gpu.render(frame.texture())?.clear(color).pulse()?;
    ///         Ok(())
    ///     })?;
    /// }
    /// ```
    ///
    /// # Resize self-healing
    ///
    /// When [`acquire`](Surface::acquire) reports `SurfaceOutdated`
    /// (the window was resized) and the driver can read the target's
    /// **current** extent (Metal: the layer's `drawableSize`; Vulkan:
    /// `VkSurfaceCapabilitiesKHR::currentExtent`), the surface
    /// reconfigures itself to that extent — same format preference and
    /// present mode — and retries the acquire **once**. The healed
    /// extent is visible through [`config`](Surface::config) /
    /// [`width`](Surface::width) / [`height`](Surface::height) after
    /// the call. When the driver cannot tell the new extent, or the
    /// retry fails, `SurfaceOutdated` propagates and the caller
    /// reconfigures manually ([`configure`](Surface::configure)), as
    /// in the primitive loop.
    ///
    /// `Timeout` propagates untouched — no frame was free within the
    /// backend's deadline; call again next loop iteration.
    pub fn render_frame<R>(
        &mut self,
        f: impl FnOnce(&SurfaceFrame) -> Result<R, QuantaError>,
    ) -> Result<R, QuantaError> {
        let frame = match self.acquire() {
            Ok(frame) => frame,
            Err(e) if matches!(e.kind, QuantaErrorKind::SurfaceOutdated(_)) => {
                // Self-heal: adopt the target's current extent if the
                // driver can read it; otherwise the caller reconfigures.
                let Some((width, height)) =
                    self.gpu.device_handle().surface_current_extent(self.handle)
                else {
                    return Err(e);
                };
                let mut config = self.config;
                config.width = width;
                config.height = height;
                self.configure(config)?;
                self.acquire()?
            }
            Err(e) => return Err(e),
        };
        let value = f(&frame)?; // Err → frame drops unpresented.
        frame.present()?;
        Ok(value)
    }

    /// Acquire the next presentable frame.
    ///
    /// Blocks briefly if no frame is available yet (all in flight);
    /// returns `Timeout` if none became available within the
    /// backend's deadline (retry next loop iteration), or
    /// `SurfaceOutdated` when the target no longer matches the
    /// configuration (reconfigure with the new extent, then retry).
    /// For the standard loop shape, prefer
    /// [`render_frame`](Surface::render_frame) — acquire/present plus
    /// resize self-healing in one call.
    pub fn acquire(&mut self) -> Result<SurfaceFrame, QuantaError> {
        let (frame, mut texture) = self.gpu.device_handle().surface_acquire(self.handle)?;
        self.gpu.__attach_texture(&mut texture);
        Ok(SurfaceFrame {
            surface: self.handle,
            frame,
            texture,
            gpu: self.gpu.clone(),
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
        let _ = self.gpu.device_handle().surface_destroy(self.handle);
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
    pub(crate) gpu: Gpu,
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
        // The frame's passes sit in the pending lane: submit them
        // (no wait) so the present lands behind them in queue order.
        // This is the frame loop's natural submit point — one command
        // buffer per frame.
        self.gpu.__submit_pending()?;
        self.gpu
            .device_handle()
            .surface_present(self.surface, self.frame)
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
            let _ = self
                .gpu
                .device_handle()
                .surface_discard(self.surface, self.frame);
        }
    }
}

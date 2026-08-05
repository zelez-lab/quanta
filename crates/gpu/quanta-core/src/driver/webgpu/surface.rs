//! Canvas presentation for the WebGPU driver (step 096).
//!
//! The browser's swapchain is the canvas: `getContext("webgpu")` +
//! `configure()` own the frame pool, `getCurrentTexture()` is the
//! per-frame target, and presentation is implicit — the compositor
//! shows the current texture when the task returns to the event loop.
//! This module folds that shape into the backend-agnostic surface
//! contract, deliberately mirroring the **Metal** driver (the closest
//! native analogue) point for point:
//!
//! - Out-of-date is an **extent poll at acquire** (canvas backing size
//!   vs. configured extent), exactly like Metal's `drawableSize`
//!   check — not a driver error code.
//! - The acquired frame's texture is **aliased into the ordinary
//!   texture registry** under a fresh handle, so `render_begin` /
//!   `render_end` stay surface-blind.
//! - `surface_present` is pure bookkeeping: unregister the alias and
//!   release the JS handles. Queue submission order guarantees the
//!   frame's draws precede compositing, so the wrapper contract
//!   ("present after `.pulse()` returned, non-blocking") holds with
//!   no work at all.
//!
//! Divergences from the native backends, all inherent to the platform
//! and documented on the public types:
//!
//! - `acquire` never blocks and `Timeout` never fires — the host's
//!   requestAnimationFrame cadence is the back-pressure. An idle
//!   caller that doesn't acquire causes zero canvas traffic, so the
//!   demand-driven pacing contract is preserved.
//! - One presentable texture per browser task: a second acquire while
//!   a frame is outstanding is refused loudly (`InvalidParam`) instead
//!   of handing out a second alias of the same image.
//! - `surface_discard` cannot un-show: the compositor still presents
//!   the (cleared) current texture at end of task. A skipped frame
//!   must simply not acquire.

use alloc::format;

use crate::{Format, QuantaError, SurfaceConfig, SurfaceTarget, Texture, TextureUsage};

use super::WebgpuDevice;
use super::ffi;
use super::state::{TextureEntry, WebgpuSurface, WebgpuSurfaceFrame};

/// Resolve the configured format preference against the browser.
///
/// Vulkan-style negotiation with the browser's own preference signal:
/// the two 8-bit color formats are interchangeable on every browser,
/// but only `getPreferredCanvasFormat()` avoids a per-present swizzle
/// — so an 8-bit request resolves to the preferred one (the same
/// reality as Android surfaces offering `RGBA8` to a `BGRA8` request
/// on Vulkan). `RGBA16Float` is honored exactly. Read the result back
/// with `Surface::format()` before building pipelines.
fn negotiate_format(requested: Format) -> Result<Format, QuantaError> {
    match requested {
        Format::BGRA8 | Format::RGBA8 => match unsafe { ffi::quanta_canvas_preferred_format() } {
            ffi::format::RGBA8UNORM => Ok(Format::RGBA8),
            ffi::format::BGRA8UNORM => Ok(Format::BGRA8),
            other => Err(QuantaError::internal(format!(
                "glue returned unknown preferred canvas format code {other}"
            ))),
        },
        Format::RGBA16Float => Ok(Format::RGBA16Float),
        other => Err(QuantaError::not_supported(format!(
            "WebGPU canvases support BGRA8, RGBA8 and RGBA16Float frames; \
             {other:?} was requested"
        ))),
    }
}

/// Validate the parts of a config every (re)configure must honor and
/// map the usage onto GPUCanvasConfiguration usage bits.
fn validate_config(config: &SurfaceConfig) -> Result<u32, QuantaError> {
    match config.present_mode {
        crate::PresentMode::Fifo => {}
        other => {
            return Err(QuantaError::not_supported(format!(
                "the browser compositor always presents at its own cadence \
                 (vsync) — PresentMode::Fifo is the only mode on WebGPU, \
                 {other:?} was requested"
            )));
        }
    }
    if config.usage.has(TextureUsage::STORAGE) {
        return Err(QuantaError::not_supported(
            "canvas frame textures cannot be storage images",
        ));
    }
    // RENDER_ATTACHMENT is the point; COPY_SRC always rides along (the
    // `framebufferOnly = NO` analogue — keeps frames readable).
    let mut usage = ffi::texture_usage::RENDER_ATTACHMENT | ffi::texture_usage::COPY_SRC;
    if config.usage.has(TextureUsage::SHADER_READ) {
        usage |= ffi::texture_usage::TEXTURE_BINDING;
    }
    Ok(usage)
}

impl WebgpuDevice {
    /// Set the canvas backing size and (re)configure its context.
    fn apply_config(
        &self,
        canvas: u32,
        context: u32,
        config: &SurfaceConfig,
    ) -> Result<Format, QuantaError> {
        let usage = validate_config(config)?;
        let negotiated = negotiate_format(config.format)?;
        let format_code = match negotiated {
            Format::BGRA8 => ffi::format::BGRA8UNORM,
            Format::RGBA8 => ffi::format::RGBA8UNORM,
            Format::RGBA16Float => ffi::format::RGBA16FLOAT,
            // negotiate_format only returns the three canvas formats.
            _ => unreachable!(),
        };
        let device = self.dev()?;
        unsafe {
            ffi::quanta_canvas_context_configure(
                context,
                canvas,
                device,
                format_code,
                usage,
                config.width,
                config.height,
            );
        }
        Ok(negotiated)
    }

    pub(super) fn surface_create_impl(
        &self,
        target: &SurfaceTarget,
        config: &SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        let (canvas, owns_canvas) = match target {
            SurfaceTarget::Canvas { canvas } => {
                if *canvas == ffi::NULL_HANDLE {
                    return Err(QuantaError::invalid_param(
                        "SurfaceTarget::Canvas carries the null handle — pass \
                         the id returned by the glue's registerCanvas",
                    ));
                }
                (*canvas, false)
            }
            SurfaceTarget::Headless => {
                let canvas =
                    unsafe { ffi::quanta_canvas_create_offscreen(config.width, config.height) };
                if canvas == ffi::NULL_HANDLE {
                    return Err(QuantaError::not_supported(
                        "this browser cannot create an OffscreenCanvas for a \
                         headless surface",
                    ));
                }
                (canvas, true)
            }
            other => {
                return Err(QuantaError::not_supported(format!(
                    "the WebGPU backend presents to browser canvases — use \
                     SurfaceTarget::Canvas or SurfaceTarget::Headless, got \
                     {other:?}"
                )));
            }
        };

        let context = unsafe { ffi::quanta_canvas_context_create(canvas) };
        if context == ffi::NULL_HANDLE {
            if owns_canvas {
                unsafe { ffi::quanta_release(canvas) };
            }
            return Err(QuantaError::not_supported(
                "the canvas could not provide a webgpu context — WebGPU is \
                 absent, or the canvas already handed out a 2d/webgl context",
            ));
        }

        let format = match self.apply_config(canvas, context, config) {
            Ok(f) => f,
            Err(e) => {
                unsafe {
                    ffi::quanta_release(context);
                    if owns_canvas {
                        ffi::quanta_release(canvas);
                    }
                }
                return Err(e);
            }
        };

        let handle = self.state.alloc_handle();
        self.state.surfaces.0.borrow_mut().insert(
            handle,
            WebgpuSurface {
                canvas,
                context,
                owns_canvas,
                width: config.width,
                height: config.height,
                format,
            },
        );
        Ok(handle)
    }

    pub(super) fn surface_configure_impl(
        &self,
        surface: u64,
        config: &SurfaceConfig,
    ) -> Result<(), QuantaError> {
        let (canvas, context) = {
            let surfaces = self.state.surfaces.0.borrow();
            let entry = surfaces
                .get(&surface)
                .ok_or_else(|| QuantaError::not_found("unknown surface handle"))?;
            (entry.canvas, entry.context)
        };
        let format = self.apply_config(canvas, context, config)?;
        let mut surfaces = self.state.surfaces.0.borrow_mut();
        if let Some(entry) = surfaces.get_mut(&surface) {
            entry.width = config.width;
            entry.height = config.height;
            entry.format = format;
        }
        Ok(())
    }

    pub(super) fn surface_format_impl(&self, surface: u64) -> Result<Format, QuantaError> {
        let surfaces = self.state.surfaces.0.borrow();
        let entry = surfaces
            .get(&surface)
            .ok_or_else(|| QuantaError::not_found("unknown surface handle"))?;
        Ok(entry.format)
    }

    pub(super) fn surface_current_extent_impl(&self, surface: u64) -> Option<(u32, u32)> {
        let surfaces = self.state.surfaces.0.borrow();
        let entry = surfaces.get(&surface)?;
        let w = unsafe { ffi::quanta_canvas_width(entry.canvas) };
        let h = unsafe { ffi::quanta_canvas_height(entry.canvas) };
        if w == 0 || h == 0 {
            return None;
        }
        Some((w, h))
    }

    pub(super) fn surface_acquire_impl(&self, surface: u64) -> Result<(u64, Texture), QuantaError> {
        let (canvas, context, width, height, format) = {
            let surfaces = self.state.surfaces.0.borrow();
            let entry = surfaces
                .get(&surface)
                .ok_or_else(|| QuantaError::not_found("unknown surface handle"))?;
            (
                entry.canvas,
                entry.context,
                entry.width,
                entry.height,
                entry.format,
            )
        };

        // One presentable texture per browser task: a second
        // getCurrentTexture would alias the same image under a second
        // handle, so refuse loudly instead.
        if self
            .state
            .surface_frames
            .0
            .borrow()
            .values()
            .any(|f| f.surface == surface)
        {
            return Err(QuantaError::invalid_param(
                "a frame is already acquired for this surface — present or \
                 discard it before acquiring again (WebGPU exposes one canvas \
                 texture per frame)",
            ));
        }

        // Out-of-date poll — the Metal drawableSize check, verbatim:
        // the embedder resized the canvas backing store out from under
        // the configuration.
        let cur_w = unsafe { ffi::quanta_canvas_width(canvas) };
        let cur_h = unsafe { ffi::quanta_canvas_height(canvas) };
        if cur_w != width || cur_h != height {
            return Err(QuantaError::surface_outdated(
                "canvas backing size no longer matches the surface configuration",
            ));
        }

        let tex = unsafe { ffi::quanta_canvas_get_current_texture(context) };
        if tex == ffi::NULL_HANDLE {
            return Err(QuantaError::internal(
                "getCurrentTexture returned no texture",
            ));
        }
        let view = unsafe { ffi::quanta_texture_create_view(tex) };

        // Alias the canvas texture into the ordinary registry so the
        // render path targets it like any texture. 256-align the row
        // stride so `texture_read_async` works on frames too.
        let bytes_per_row = (width * format.bytes_per_pixel() as u32).div_ceil(256) * 256;
        let texture_handle = self.state.alloc_handle();
        self.state.textures.0.borrow_mut().insert(
            texture_handle,
            TextureEntry {
                texture: tex,
                view,
                width,
                height,
                format,
                // Canvas frames are always single-sample; an MSAA pass
                // reaches them as the resolve destination.
                samples: 1,
                bytes_per_row,
            },
        );

        let frame = self.state.alloc_handle();
        self.state.surface_frames.0.borrow_mut().insert(
            frame,
            WebgpuSurfaceFrame {
                surface,
                texture_handle,
            },
        );

        Ok((
            frame,
            Texture {
                handle: texture_handle,
                width,
                height,
                format,
                // Canvas frames are always single-sample.
                sample_count: 1,
                device: None,
                // The canvas owns the image — the wrapper must not
                // destroy it.
                live: false,
            },
        ))
    }

    /// Unregister the frame's texture alias and release the JS
    /// handles. `quanta_release`, never destroy: the canvas owns its
    /// texture.
    fn retire_frame(&self, frame: WebgpuSurfaceFrame) {
        if let Some(entry) = self
            .state
            .textures
            .0
            .borrow_mut()
            .remove(&frame.texture_handle)
        {
            unsafe {
                ffi::quanta_release(entry.view);
                ffi::quanta_release(entry.texture);
            }
        }
    }

    pub(super) fn surface_present_impl(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        let entry = self
            .state
            .surface_frames
            .0
            .borrow_mut()
            .remove(&frame)
            .ok_or_else(|| QuantaError::not_found("unknown surface frame handle"))?;
        if entry.surface != surface {
            // Re-insert so the rightful surface can still retire it.
            self.state
                .surface_frames
                .0
                .borrow_mut()
                .insert(frame, entry);
            return Err(QuantaError::invalid_param(
                "frame does not belong to this surface",
            ));
        }
        // No JS present call exists: the compositor shows the current
        // texture when this task returns to the event loop, and queue
        // submission order places the frame's draws before that.
        self.retire_frame(entry);
        Ok(())
    }

    pub(super) fn surface_discard_impl(&self, surface: u64, frame: u64) -> Result<(), QuantaError> {
        // Same bookkeeping as present. The compositor still shows the
        // (cleared) current texture at end of task — a discard cannot
        // un-show on this backend; callers that skip a frame skip the
        // acquire instead.
        self.surface_present_impl(surface, frame)
    }

    pub(super) fn surface_destroy_impl(&self, surface: u64) -> Result<(), QuantaError> {
        let Some(entry) = self.state.surfaces.0.borrow_mut().remove(&surface) else {
            // Unknown handle: already destroyed — fine, mirrors the
            // native drivers' Drop-friendly behavior.
            return Ok(());
        };
        // Sweep any frame still outstanding against this surface.
        let stale: alloc::vec::Vec<u64> = self
            .state
            .surface_frames
            .0
            .borrow()
            .iter()
            .filter(|(_, f)| f.surface == surface)
            .map(|(&k, _)| k)
            .collect();
        for key in stale {
            if let Some(f) = self.state.surface_frames.0.borrow_mut().remove(&key) {
                self.retire_frame(f);
            }
        }
        unsafe {
            // Deliberately NO unconfigure(): it destroys the pending
            // current texture, blanking the canvas before the
            // compositor ever shows a frame presented in this same
            // task. Releasing only Quanta's handles matches the native
            // backends — the target (layer/canvas) keeps its last
            // presented image, and a later surface_create over the
            // same canvas re-obtains the context (getContext is
            // idempotent) and configure() overrides in place.
            ffi::quanta_release(entry.context);
            if entry.owns_canvas {
                ffi::quanta_release(entry.canvas);
            }
            // Embedder-registered canvases stay registered — the
            // embedder owns the registration and may create another
            // surface over the same canvas later.
        }
        Ok(())
    }
}

//! Presentation surfaces for Metal — CAMetalLayer drawables.
//!
//! `surface_create` attaches to (or, for the headless target, creates)
//! a `CAMetalLayer`; `surface_acquire` pulls the next
//! `CAMetalDrawable` and registers its texture in the device's texture
//! registry so the ordinary render-pass path targets it unchanged;
//! `surface_present` schedules `presentDrawable:` on a fresh command
//! buffer — queue order places it after the already-submitted render
//! pass, so presentation never stalls the CPU.

use alloc::format;

use crate::surface::{PresentMode, SurfaceConfig, SurfaceTarget};
use crate::{Format, QuantaError, Texture, TextureUsage};

use super::MetalDevice;
use super::device::{MetalSurface, MetalSurfaceFrame};
use super::ffi;

// CAMetalLayer lives in QuartzCore; link it alongside Metal. The
// block is intentionally empty — every call goes through objc_msgSend.
#[link(name = "QuartzCore", kind = "framework")]
unsafe extern "C" {}

// Autorelease-pool management. `nextDrawable` (and `commandBuffer`)
// return autoreleased objects; without a pool bracket each acquired
// drawable would keep an extra never-released reference and the
// layer's small drawable pool would run dry after a few frames.
#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_autoreleasePoolPush() -> *mut core::ffi::c_void;
    fn objc_autoreleasePoolPop(pool: *mut core::ffi::c_void);
}

/// CoreGraphics size — `CAMetalLayer.drawableSize`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CGSize {
    width: f64,
    height: f64,
}

unsafe fn msg_void_cgsize(obj: ffi::Id, name: &[u8], size: CGSize) {
    unsafe {
        let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, CGSize) =
            core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
        f(obj, ffi::sel(name), size);
    }
}

unsafe fn msg_cgsize(obj: ffi::Id, name: &[u8]) -> CGSize {
    unsafe {
        let f: unsafe extern "C" fn(ffi::Id, ffi::Sel) -> CGSize =
            core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
        f(obj, ffi::sel(name))
    }
}

/// The CAMetalLayer-legal subset of Quanta formats.
fn layer_format_supported(format: Format) -> bool {
    matches!(format, Format::BGRA8 | Format::RGBA16Float)
}

/// Apply a `SurfaceConfig` to a CAMetalLayer. Shared by create and
/// configure.
unsafe fn apply_config(
    device: ffi::Id,
    layer: ffi::Id,
    config: &SurfaceConfig,
) -> Result<(), QuantaError> {
    if !layer_format_supported(config.format) {
        return Err(QuantaError::not_supported(
            "Metal surfaces support BGRA8 and RGBA16Float frame formats",
        ));
    }
    if config.usage.has(TextureUsage::SHADER_WRITE) {
        return Err(QuantaError::not_supported(
            "Metal surface frames cannot be shader-writable (drawable textures)",
        ));
    }
    let display_sync = match config.present_mode {
        PresentMode::Fifo => true,
        PresentMode::Immediate => false,
        _ => {
            return Err(QuantaError::not_supported(
                "Metal surfaces support Fifo and Immediate present modes",
            ));
        }
    };
    unsafe {
        ffi::msg_void_id(layer, b"setDevice:\0", device);
        ffi::msg_void_u64(
            layer,
            b"setPixelFormat:\0",
            super::format_to_metal(config.format),
        );
        // framebufferOnly = NO keeps the drawable textures sampleable
        // (SHADER_READ) and blit-able.
        ffi::msg_void_bool(layer, b"setFramebufferOnly:\0", false);
        ffi::msg_void_bool(layer, b"setDisplaySyncEnabled:\0", display_sync);
        msg_void_cgsize(
            layer,
            b"setDrawableSize:\0",
            CGSize {
                width: config.width as f64,
                height: config.height as f64,
            },
        );
    }
    Ok(())
}

impl MetalDevice {
    pub(crate) fn surface_create_impl(
        &self,
        target: &SurfaceTarget,
        config: &SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        let layer: ffi::Id = unsafe {
            match target {
                SurfaceTarget::MetalLayer { layer } => {
                    if layer.is_null() {
                        return Err(QuantaError::invalid_param(
                            "SurfaceTarget::MetalLayer pointer is null",
                        ));
                    }
                    // Borrowed from the caller — retain for the
                    // surface's lifetime, released on destroy.
                    ffi::msg_id(*layer as ffi::Id, b"retain\0")
                }
                SurfaceTarget::Headless => {
                    let cls = ffi::cls(b"CAMetalLayer\0");
                    if cls.is_null() {
                        return Err(QuantaError::internal(
                            "CAMetalLayer class unavailable (QuartzCore not loaded)",
                        ));
                    }
                    // Owned (+1) — released on destroy.
                    ffi::msg_id(cls as ffi::Id, b"new\0")
                }
                _ => {
                    return Err(QuantaError::not_supported(
                        "this surface target is not available on the Metal backend",
                    ));
                }
            }
        };
        if layer.is_null() {
            return Err(QuantaError::internal("failed to obtain CAMetalLayer"));
        }

        if let Err(e) = unsafe { apply_config(self.device, layer, config) } {
            unsafe { ffi::msg_void(layer, b"release\0") };
            return Err(e);
        }

        let handle = self.alloc_handle();
        self.surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                MetalSurface {
                    layer,
                    width: config.width,
                    height: config.height,
                    format: config.format,
                },
            );
        Ok(handle)
    }

    pub(crate) fn surface_configure_impl(
        &self,
        surface: u64,
        config: &SurfaceConfig,
    ) -> Result<(), QuantaError> {
        let mut surfaces = self
            .surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let s = surfaces
            .get_mut(&surface)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;
        unsafe { apply_config(self.device, s.layer, config)? };
        s.width = config.width;
        s.height = config.height;
        s.format = config.format;
        Ok(())
    }

    pub(crate) fn surface_acquire_impl(&self, surface: u64) -> Result<(u64, Texture), QuantaError> {
        let surfaces = self
            .surfaces
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let s = surfaces
            .get(&surface)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))?;

        unsafe {
            // Out-of-date check: the layer's drawable size diverged
            // from the configured extent (the windowing environment
            // resized the layer). The caller must reconfigure.
            let ds = msg_cgsize(s.layer, b"drawableSize\0");
            if ds.width as u32 != s.width || ds.height as u32 != s.height {
                return Err(QuantaError::surface_outdated(
                    "layer drawable size no longer matches the surface configuration",
                ));
            }

            // Pool bracket: nextDrawable is autoreleased; retain ours
            // and drain the pool so the layer's drawable pool recycles.
            let pool = objc_autoreleasePoolPush();
            let drawable = ffi::msg_id(s.layer, b"nextDrawable\0");
            let drawable = if drawable.is_null() {
                core::ptr::null_mut()
            } else {
                ffi::msg_id(drawable, b"retain\0")
            };
            objc_autoreleasePoolPop(pool);

            if drawable.is_null() {
                return Err(
                    QuantaError::timeout().with_context("surface_acquire: no drawable available")
                );
            }

            let tex = ffi::msg_id(drawable, b"texture\0");
            if tex.is_null() {
                ffi::msg_void(drawable, b"release\0");
                return Err(QuantaError::internal("drawable has no texture"));
            }

            let texture_handle = self.alloc_handle();
            self.textures
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(texture_handle, tex);

            let frame = self.alloc_handle();
            self.surface_frames
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(
                    frame,
                    MetalSurfaceFrame {
                        drawable,
                        texture_handle,
                    },
                );

            Ok((
                frame,
                Texture {
                    handle: texture_handle,
                    width: s.width,
                    height: s.height,
                    format: s.format,
                    device: None,
                    // The swapchain owns the drawable; it is recycled on
                    // present, so this wrapper must not destroy it.
                    live: false,
                },
            ))
        }
    }

    /// Remove and return an in-flight frame's state.
    fn take_frame(&self, frame: u64) -> Result<MetalSurfaceFrame, QuantaError> {
        self.surface_frames
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&frame)
            .ok_or_else(|| {
                QuantaError::not_found("surface frame not found")
                    .with_context(&format!("frame handle {frame}"))
            })
    }

    /// Drop the frame's texture-registry alias.
    fn unregister_frame_texture(&self, f: &MetalSurfaceFrame) -> Result<(), QuantaError> {
        self.textures
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&f.texture_handle);
        Ok(())
    }

    pub(crate) fn surface_format_impl(&self, surface: u64) -> Result<Format, QuantaError> {
        // Quanta sets the CAMetalLayer's pixel format itself
        // (`apply_config` → `setPixelFormat:`), so the actual frame
        // format always equals the configured one — no negotiation.
        let surfaces = self
            .surfaces
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        surfaces
            .get(&surface)
            .map(|s| s.format)
            .ok_or_else(|| QuantaError::not_found("surface handle not found"))
    }

    pub(crate) fn surface_present_impl(
        &self,
        _surface: u64,
        frame: u64,
    ) -> Result<(), QuantaError> {
        let f = self.take_frame(frame)?;
        self.unregister_frame_texture(&f)?;
        unsafe {
            // A fresh command buffer whose only job is the present.
            // The queue executes buffers in commit order, so it runs
            // after the render pass the caller already submitted; the
            // drawable additionally waits for all work targeting its
            // texture. Commit and return — no waitUntilCompleted.
            let pool = objc_autoreleasePoolPush();
            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            ffi::msg_void_id(cmd, b"presentDrawable:\0", f.drawable);
            ffi::msg_void(cmd, b"commit\0");
            objc_autoreleasePoolPop(pool);
            ffi::msg_void(f.drawable, b"release\0");
        }
        Ok(())
    }

    pub(crate) fn surface_discard_impl(
        &self,
        _surface: u64,
        frame: u64,
    ) -> Result<(), QuantaError> {
        let f = self.take_frame(frame)?;
        self.unregister_frame_texture(&f)?;
        unsafe {
            // Never presented: releasing our reference returns the
            // drawable to the layer's pool unshown.
            ffi::msg_void(f.drawable, b"release\0");
        }
        Ok(())
    }

    pub(crate) fn surface_destroy_impl(&self, surface: u64) -> Result<(), QuantaError> {
        let s = self
            .surfaces
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .remove(&surface);
        if let Some(s) = s {
            unsafe { ffi::msg_void(s.layer, b"release\0") };
        }
        Ok(())
    }
}

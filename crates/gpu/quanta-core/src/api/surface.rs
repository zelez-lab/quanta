//! Presentation-surface data model — Quanta-owned present.
//!
//! A surface is a swapchain over a platform presentation target. The
//! configuration types here ([`PresentMode`], [`SurfaceConfig`],
//! [`SurfaceTarget`]) are what the [`GpuDevice`](crate::GpuDevice)
//! trait and the drivers speak (`surface_create` /
//! `surface_configure` / `surface_acquire` / `surface_present`). The
//! typed `Surface` / `SurfaceFrame` wrappers that run the
//! acquire → render → present frame loop live in the `quanta-render`
//! crate.
//!
//! This is one of Quanta's two presentation models. The other —
//! [`Texture::native_handle`](crate::Texture::native_handle) — exports the
//! rendered texture so an external compositor owns present instead.

use crate::{Format, TextureUsage};

/// When a presented frame becomes visible.
///
/// Marked `#[non_exhaustive]`: modes can be added (e.g. adaptive/tearing
/// hybrids) without a breaking change — match with a wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PresentMode {
    /// Present at the display's vertical refresh (vsync). Never
    /// tears; every backend that presents at all supports it. The
    /// default.
    #[default]
    Fifo,
    /// Present as soon as possible, possibly mid-scanout (may tear).
    /// Lowest latency. Backends without it reject at
    /// create/configure time with `NotSupported`.
    Immediate,
    /// Triple-buffered: new frames replace the queued one instead of
    /// waiting behind it. Low latency without tearing. Backends
    /// without it reject at create/configure time with
    /// `NotSupported`.
    Mailbox,
}

/// Configuration for a presentation surface.
///
/// Marked `#[non_exhaustive]`: fields will grow (color space / HDR,
/// buffer count, alpha compositing mode, …). Construct through
/// [`SurfaceConfig::new`] and adjust fields by assignment:
///
/// ```ignore
/// let mut config = SurfaceConfig::new(1280, 720);
/// config.present_mode = PresentMode::Fifo;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct SurfaceConfig {
    /// Frame width in pixels. Must match the current size of the
    /// presentation target; when the target is resized, reconfigure
    /// with the new extent.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Preferred pixel format of the frames. `BGRA8` is the portable
    /// default.
    ///
    /// This is a single-format **preference**, not a guarantee. On
    /// Vulkan the swapchain negotiates against what the surface actually
    /// offers — it tries this format first, then `BGRA8`, then `RGBA8`,
    /// then any other offered format Quanta can express (so a surface
    /// that only offers `RGBA8`, as Android's conventionally do, still
    /// works with the default `BGRA8` request). Only a surface offering
    /// nothing expressible is rejected. On Metal the format is exact —
    /// Quanta sets the layer format, so the frames use exactly this.
    ///
    /// Read what was actually chosen with `Surface::format()` (the typed
    /// wrapper in `quanta-render`); the acquired frames' textures report
    /// the negotiated format. There
    /// is deliberately no preference-*list* here — if you need the
    /// fallback ordered differently than `[requested, BGRA8, RGBA8]`,
    /// type your pipeline per frame from the acquired texture's format.
    pub format: Format,
    /// When a presented frame becomes visible. Default [`PresentMode::Fifo`].
    pub present_mode: PresentMode,
    /// How the frame textures may be used besides presentation.
    /// Default `RENDER_TARGET`; add `SHADER_READ` to sample the frame
    /// from a later pass.
    pub usage: TextureUsage,
}

impl SurfaceConfig {
    /// A configuration with the given extent and portable defaults:
    /// `BGRA8`, [`PresentMode::Fifo`], `RENDER_TARGET` usage.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: Format::BGRA8,
            present_mode: PresentMode::Fifo,
            usage: TextureUsage::RENDER_TARGET,
        }
    }
}

/// The platform target a presentation surface presents to.
///
/// Marked `#[non_exhaustive]`: window-system variants land with their
/// backend implementations (Vulkan Wayland/X11/Win32 handles, a WebGPU
/// canvas selector, an OS-provided buffer target) — match with a
/// wildcard arm.
///
/// With the `raw-window-handle` feature on, [`SurfaceTarget::from_window`]
/// maps a winit-style window (anything implementing rwh 0.6's
/// `HasWindowHandle + HasDisplayHandle`) to the right variant with zero
/// per-OS matching — the one-value window handoff.
#[non_exhaustive]
#[derive(Debug)]
pub enum SurfaceTarget {
    /// An existing `CAMetalLayer`, provided by the windowing
    /// environment (or by a compositor handing the app a layer to
    /// draw into). This is the OS window handle on Apple — the
    /// consumer hands it over; which driver presents to it is
    /// Quanta's business (Metal here). The pointer must be a valid
    /// `CAMetalLayer` and must outlive the surface. Quanta configures
    /// the layer (device, pixel format, drawable size) and presents
    /// drawables to it; the caller keeps ownership of the layer itself.
    MetalLayer {
        /// `CAMetalLayer*` as a raw pointer.
        layer: *mut core::ffi::c_void,
    },
    /// An AppKit `NSView`, the window handle a macOS windowing library
    /// (winit, or `raw-window-handle` producers generally) actually
    /// exposes. The Metal driver makes the view layer-backed
    /// (`setWantsLayer:YES`) and attaches a `CAMetalLayer` — reusing
    /// the view's existing layer when it already is one — then
    /// proceeds exactly as [`SurfaceTarget::MetalLayer`]. Same safety
    /// contract as `MetalLayer`: the pointer must be a valid `NSView`
    /// that outlives the surface, and the caller keeps ownership of
    /// the view. AppKit views belong to the main thread — create the
    /// surface there. Non-Metal backends return `NotSupported`
    /// (MoltenVK-on-macOS wiring is a documented deferral).
    AppKitView {
        /// `NSView*` as a raw pointer.
        ns_view: *mut core::ffi::c_void,
    },
    /// An X11 window, named by its Xlib display connection and window
    /// id. The consumer hands over the OS window handle; which driver
    /// presents to it is Quanta's business (Vulkan `VK_KHR_xlib_surface`
    /// here). The display connection and window must outlive the
    /// surface.
    Xlib {
        /// `Display*` as a raw pointer.
        display: *mut core::ffi::c_void,
        /// The X11 `Window` id.
        window: u64,
    },
    /// An Android window (`ANativeWindow`), obtained from the embedder
    /// (e.g. `ANativeWindow_fromSurface` on a Java `Surface`). The
    /// consumer hands over the OS window handle; which driver presents
    /// to it is Quanta's business (Vulkan `VK_KHR_android_surface`
    /// here). The window must outlive the surface.
    AndroidWindow {
        /// `ANativeWindow*` as a raw pointer.
        a_native_window: *mut core::ffi::c_void,
    },
    /// A Win32 window, named by its `HWND` and the `HINSTANCE` of the
    /// module that owns it. The consumer hands over the OS window
    /// handle; which driver presents to it is Quanta's business (Vulkan
    /// `VK_KHR_win32_surface` here). The `HWND` and its `HINSTANCE` come
    /// from the embedder's window; both must outlive the surface.
    Win32 {
        /// `HINSTANCE` of the owning module, as a raw pointer.
        hinstance: *mut core::ffi::c_void,
        /// `HWND` of the target window, as a raw pointer.
        hwnd: *mut core::ffi::c_void,
    },
    /// A presentation target with no window attached: the backend
    /// creates and owns its native target (Metal: an off-screen
    /// `CAMetalLayer`; Vulkan: `VK_EXT_headless_surface`). The full
    /// acquire/present machinery runs — frames just aren't composited
    /// anywhere. For tests, warm-up, and consumers that composite
    /// through another channel.
    Headless,
}

// ─── raw-window-handle interop (feature `raw-window-handle`) ───────────
//
// The one-value window handoff: a winit-style consumer goes from window
// to surface target with zero per-OS matching. Pure mapping — no OS
// calls happen here; pointer validation and platform wiring stay in the
// drivers, exactly as when the target is constructed by hand.

#[cfg(feature = "raw-window-handle")]
mod rwh_interop {
    use alloc::format;

    use raw_window_handle as rwh;

    use super::SurfaceTarget;
    use crate::QuantaError;

    /// A `HandleError` from the windowing library, mapped into Quanta's
    /// error model: `NotSupported` when the window system cannot expose
    /// a representable handle, `InvalidParam` otherwise (e.g. the
    /// handle is temporarily unavailable — retry when the window is
    /// back).
    fn map_handle_error(e: rwh::HandleError) -> QuantaError {
        match e {
            rwh::HandleError::NotSupported => QuantaError::not_supported(
                "the windowing system does not expose a raw handle Quanta can use",
            ),
            other => QuantaError::invalid_param(format!(
                "window-handle source could not produce a handle: {other}"
            )),
        }
    }

    impl SurfaceTarget {
        /// The surface target for a window, with zero per-OS matching —
        /// the one-value window handoff:
        ///
        /// ```ignore
        /// // `window` is anything implementing raw-window-handle 0.6's
        /// // HasWindowHandle + HasDisplayHandle (a winit Window, say).
        /// let target = SurfaceTarget::from_window(&window)?;
        /// let mut surface = gpu.create_surface(&target, &SurfaceConfig::new(w, h))?;
        /// ```
        ///
        /// Fetches the window and display handles from `source` and
        /// delegates to [`SurfaceTarget::from_raw`] — see there for the
        /// exact mapping and the unsupported window systems. A refusal
        /// from the windowing library (`HandleError`) maps to
        /// `NotSupported` / `InvalidParam`.
        ///
        /// # Safety contract (inherited, not checked here)
        ///
        /// The mapping itself is pure, but the pointers inside the
        /// resulting target are only as valid as `source`: the window
        /// (and its display connection) must outlive the surface
        /// created from the target, exactly as documented per variant.
        pub fn from_window(
            source: &(impl rwh::HasWindowHandle + rwh::HasDisplayHandle),
        ) -> Result<SurfaceTarget, QuantaError> {
            let window = source.window_handle().map_err(map_handle_error)?.as_raw();
            let display = source.display_handle().map_err(map_handle_error)?.as_raw();
            SurfaceTarget::from_raw(window, display)
        }

        /// Map raw `raw-window-handle` 0.6 handles to a surface target —
        /// the escape hatch under [`SurfaceTarget::from_window`] for
        /// callers that already hold the raw handles. A **pure**
        /// mapping: no OS calls, no pointer dereferences.
        ///
        /// | Window handle | Target |
        /// |---------------|--------|
        /// | `AppKit` | [`SurfaceTarget::AppKitView`] (the Metal driver attaches the `CAMetalLayer`) |
        /// | `Xlib` (+ `Xlib` display) | [`SurfaceTarget::Xlib`] |
        /// | `Win32` | [`SurfaceTarget::Win32`] |
        /// | `AndroidNdk` | [`SurfaceTarget::AndroidWindow`] |
        /// | `Wayland` | `Err(NotSupported)` — run under XWayland for now |
        /// | anything else | `Err(NotSupported)`, naming the variant |
        ///
        /// Mapping notes:
        ///
        /// - **Xlib**: the display handle must also be `Xlib` and carry
        ///   a `Display*` — an EGL default-display handle (`display:
        ///   None`) is rejected with `InvalidParam`, because
        ///   `VK_KHR_xlib_surface` needs the connection. The handle's
        ///   `visual_id` is not carried; the swapchain negotiates its
        ///   own format.
        /// - **Win32**: an absent `hinstance` (legal in rwh 0.6) maps to
        ///   a **null** pointer, passed through as-is. The Vulkan
        ///   backend requires a non-null `HINSTANCE` and rejects null
        ///   with `InvalidParam` at create time — if your handle
        ///   producer omits it (winit supplies it), fetch the module
        ///   handle yourself (`GetModuleHandleW(NULL)`) and construct
        ///   [`SurfaceTarget::Win32`] directly.
        /// - **Wayland**: `VK_KHR_wayland_surface` wiring is a
        ///   documented deferral. Until it lands, run the app under
        ///   XWayland (force your windowing library's X11 backend) so
        ///   the window arrives as `Xlib`.
        pub fn from_raw(
            window: rwh::RawWindowHandle,
            display: rwh::RawDisplayHandle,
        ) -> Result<SurfaceTarget, QuantaError> {
            match (window, display) {
                (rwh::RawWindowHandle::AppKit(w), _) => Ok(SurfaceTarget::AppKitView {
                    ns_view: w.ns_view.as_ptr(),
                }),
                (rwh::RawWindowHandle::Xlib(w), rwh::RawDisplayHandle::Xlib(d)) => {
                    let Some(display) = d.display else {
                        return Err(QuantaError::invalid_param(
                            "the Xlib display handle carries no Display* (an EGL \
                             default-display handle) — VK_KHR_xlib_surface needs \
                             the connection; open the Display and use \
                             SurfaceTarget::Xlib directly",
                        ));
                    };
                    Ok(SurfaceTarget::Xlib {
                        display: display.as_ptr(),
                        window: w.window,
                    })
                }
                (rwh::RawWindowHandle::Xlib(_), other) => Err(QuantaError::invalid_param(format!(
                    "an Xlib window handle needs an Xlib display handle, \
                     got {other:?}"
                ))),
                (rwh::RawWindowHandle::Win32(w), _) => Ok(SurfaceTarget::Win32 {
                    // None → null, passed through (see the mapping notes
                    // above): the Vulkan backend rejects a null
                    // HINSTANCE at create time.
                    hinstance: w
                        .hinstance
                        .map(|h| h.get() as *mut core::ffi::c_void)
                        .unwrap_or(core::ptr::null_mut()),
                    hwnd: w.hwnd.get() as *mut core::ffi::c_void,
                }),
                (rwh::RawWindowHandle::AndroidNdk(w), _) => Ok(SurfaceTarget::AndroidWindow {
                    a_native_window: w.a_native_window.as_ptr(),
                }),
                (rwh::RawWindowHandle::Wayland(_), _) => Err(QuantaError::not_supported(
                    "Wayland window handles are not wired yet (VK_KHR_wayland_surface \
                     is a documented deferral) — run under XWayland for now: force \
                     your windowing library's X11 backend so the window arrives as \
                     an Xlib handle",
                )),
                (other, _) => Err(QuantaError::not_supported(format!(
                    "no Quanta surface target for this window-handle kind: {other:?} \
                     (supported: AppKit, Xlib, Win32, AndroidNdk; Wayland via \
                     XWayland for now)"
                ))),
            }
        }
    }
}

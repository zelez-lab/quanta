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
#[non_exhaustive]
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

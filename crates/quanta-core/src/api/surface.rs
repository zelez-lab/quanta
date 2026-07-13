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
    /// Pixel format of the frames. Presentation targets accept a
    /// restricted set — `BGRA8` is the portable default; backends
    /// reject unsupported formats at create/configure time.
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
    /// Metal: an existing `CAMetalLayer`, provided by the windowing
    /// environment (or by a compositor handing the app a layer to
    /// draw into). The pointer must be a valid `CAMetalLayer` and
    /// must outlive the surface. Quanta configures the layer
    /// (device, pixel format, drawable size) and presents drawables
    /// to it; the caller keeps ownership of the layer itself.
    MetalLayer {
        /// `CAMetalLayer*` as a raw pointer.
        layer: *mut core::ffi::c_void,
    },
    /// Vulkan on X11: create the surface from an Xlib display +
    /// window (`VK_KHR_xlib_surface`). The display connection and
    /// window must outlive the surface.
    VulkanXlib {
        /// `Display*` as a raw pointer.
        display: *mut core::ffi::c_void,
        /// The X11 `Window` id.
        window: u64,
    },
    /// Vulkan on Android: create the surface from an `ANativeWindow`
    /// (`VK_KHR_android_surface`). The window is obtained from the
    /// embedder (e.g. `ANativeWindow_fromSurface` on a Java `Surface`)
    /// and must outlive the surface.
    VulkanAndroid {
        /// `ANativeWindow*` as a raw pointer.
        a_native_window: *mut core::ffi::c_void,
    },
    /// A presentation target with no window attached: the backend
    /// creates and owns its native target (Metal: an off-screen
    /// `CAMetalLayer`; Vulkan: `VK_EXT_headless_surface`). The full
    /// acquire/present machinery runs — frames just aren't composited
    /// anywhere. For tests, warm-up, and consumers that composite
    /// through another channel.
    Headless,
}

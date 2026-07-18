#![cfg(all(feature = "render", feature = "raw-window-handle"))]
//! `SurfaceTarget::from_raw` / `from_window` — the raw-window-handle
//! interop (feature `raw-window-handle`).
//!
//! Pure-mapping tests: `from_raw` performs no OS calls, so every case
//! runs on every platform with dummy (dangling, non-null) pointers.
//! The platform legs behind the mapped targets are exercised by
//! `tests/gpu_surface.rs`.

use core::ffi::c_void;
use core::num::NonZeroIsize;
use core::ptr::NonNull;

use quanta::rwh;
use quanta::{QuantaErrorKind, SurfaceTarget};

/// A dangling-but-non-null pointer standing in for an OS object.
fn dummy_ptr() -> NonNull<c_void> {
    NonNull::<c_void>::dangling()
}

/// A display handle of a kind the mapping ignores for AppKit / Win32 /
/// Android windows.
fn dont_care_display() -> rwh::RawDisplayHandle {
    rwh::RawDisplayHandle::AppKit(rwh::AppKitDisplayHandle::new())
}

// --- The four supported mappings ---

#[test]
fn appkit_maps_to_appkit_view() {
    let ns_view = dummy_ptr();
    let window = rwh::RawWindowHandle::AppKit(rwh::AppKitWindowHandle::new(ns_view));
    let target = SurfaceTarget::from_raw(window, dont_care_display()).unwrap();
    match target {
        SurfaceTarget::AppKitView { ns_view: mapped } => assert_eq!(mapped, ns_view.as_ptr()),
        other => panic!("expected AppKitView, got {other:?}"),
    }
}

#[test]
fn xlib_maps_to_xlib() {
    let dpy = dummy_ptr();
    let window = rwh::RawWindowHandle::Xlib(rwh::XlibWindowHandle::new(0x0042_1234));
    let display = rwh::RawDisplayHandle::Xlib(rwh::XlibDisplayHandle::new(Some(dpy), 0));
    let target = SurfaceTarget::from_raw(window, display).unwrap();
    match target {
        SurfaceTarget::Xlib { display, window } => {
            assert_eq!(display, dpy.as_ptr());
            assert_eq!(window, 0x0042_1234);
        }
        other => panic!("expected Xlib, got {other:?}"),
    }
}

#[test]
fn win32_maps_to_win32() {
    let hwnd = NonZeroIsize::new(0x1000).unwrap();
    let hinstance = NonZeroIsize::new(0x2000).unwrap();
    let mut handle = rwh::Win32WindowHandle::new(hwnd);
    handle.hinstance = Some(hinstance);
    let target =
        SurfaceTarget::from_raw(rwh::RawWindowHandle::Win32(handle), dont_care_display()).unwrap();
    match target {
        SurfaceTarget::Win32 {
            hinstance: hi,
            hwnd: hw,
        } => {
            assert_eq!(hi, 0x2000 as *mut c_void);
            assert_eq!(hw, 0x1000 as *mut c_void);
        }
        other => panic!("expected Win32, got {other:?}"),
    }
}

#[test]
fn win32_absent_hinstance_maps_to_null() {
    // rwh 0.6 makes hinstance optional; the mapping passes null through
    // (documented: the Vulkan backend then rejects at create time).
    let handle = rwh::Win32WindowHandle::new(NonZeroIsize::new(0x1000).unwrap());
    assert!(handle.hinstance.is_none());
    let target =
        SurfaceTarget::from_raw(rwh::RawWindowHandle::Win32(handle), dont_care_display()).unwrap();
    match target {
        SurfaceTarget::Win32 { hinstance, hwnd } => {
            assert!(hinstance.is_null());
            assert_eq!(hwnd, 0x1000 as *mut c_void);
        }
        other => panic!("expected Win32, got {other:?}"),
    }
}

#[test]
fn android_maps_to_android_window() {
    let anw = dummy_ptr();
    let window = rwh::RawWindowHandle::AndroidNdk(rwh::AndroidNdkWindowHandle::new(anw));
    let display = rwh::RawDisplayHandle::Android(rwh::AndroidDisplayHandle::new());
    let target = SurfaceTarget::from_raw(window, display).unwrap();
    match target {
        SurfaceTarget::AndroidWindow { a_native_window } => {
            assert_eq!(a_native_window, anw.as_ptr());
        }
        other => panic!("expected AndroidWindow, got {other:?}"),
    }
}

// --- Rejections ---

#[test]
fn wayland_rejected_with_xwayland_pointer() {
    let window = rwh::RawWindowHandle::Wayland(rwh::WaylandWindowHandle::new(dummy_ptr()));
    let display = rwh::RawDisplayHandle::Wayland(rwh::WaylandDisplayHandle::new(dummy_ptr()));
    let err = SurfaceTarget::from_raw(window, display).unwrap_err();
    match err.kind {
        QuantaErrorKind::NotSupported(ref msg) => {
            assert!(msg.contains("Wayland"), "message must name Wayland: {msg}");
            assert!(
                msg.contains("XWayland"),
                "message must point at the XWayland workaround: {msg}"
            );
        }
        other => panic!("expected NotSupported, got {other:?}"),
    }
}

#[test]
fn unsupported_kind_rejected_naming_the_variant() {
    // UiKit: representable in rwh, no Quanta target yet.
    let window = rwh::RawWindowHandle::UiKit(rwh::UiKitWindowHandle::new(dummy_ptr()));
    let display = rwh::RawDisplayHandle::UiKit(rwh::UiKitDisplayHandle::new());
    let err = SurfaceTarget::from_raw(window, display).unwrap_err();
    match err.kind {
        QuantaErrorKind::NotSupported(ref msg) => {
            assert!(
                msg.contains("UiKit"),
                "message must name the variant: {msg}"
            );
        }
        other => panic!("expected NotSupported, got {other:?}"),
    }
}

#[test]
fn xlib_window_without_display_pointer_rejected() {
    // An EGL default-display handle (display: None) cannot create a
    // VK_KHR_xlib_surface.
    let window = rwh::RawWindowHandle::Xlib(rwh::XlibWindowHandle::new(7));
    let display = rwh::RawDisplayHandle::Xlib(rwh::XlibDisplayHandle::new(None, 0));
    let err = SurfaceTarget::from_raw(window, display).unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
}

#[test]
fn xlib_window_with_mismatched_display_rejected() {
    let window = rwh::RawWindowHandle::Xlib(rwh::XlibWindowHandle::new(7));
    let err = SurfaceTarget::from_raw(window, dont_care_display()).unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
}

// --- from_window: the zero-glue form over a handle source ---

/// A winit-window stand-in: hands out the raw handles it was built
/// with, through the rwh 0.6 borrowed-handle traits.
struct FakeWindow {
    window: rwh::RawWindowHandle,
    display: rwh::RawDisplayHandle,
}

impl rwh::HasWindowHandle for FakeWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        // SAFETY: test-only dangling handles; from_raw never dereferences.
        Ok(unsafe { rwh::WindowHandle::borrow_raw(self.window) })
    }
}

impl rwh::HasDisplayHandle for FakeWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        // SAFETY: test-only dangling handles; from_raw never dereferences.
        Ok(unsafe { rwh::DisplayHandle::borrow_raw(self.display) })
    }
}

/// A window whose handle is temporarily unavailable (rwh's Android /
/// layer-shell case).
struct UnavailableWindow;

impl rwh::HasWindowHandle for UnavailableWindow {
    fn window_handle(&self) -> Result<rwh::WindowHandle<'_>, rwh::HandleError> {
        Err(rwh::HandleError::Unavailable)
    }
}

impl rwh::HasDisplayHandle for UnavailableWindow {
    fn display_handle(&self) -> Result<rwh::DisplayHandle<'_>, rwh::HandleError> {
        Err(rwh::HandleError::Unavailable)
    }
}

#[test]
fn from_window_delegates_to_from_raw() {
    let dpy = dummy_ptr();
    let fake = FakeWindow {
        window: rwh::RawWindowHandle::Xlib(rwh::XlibWindowHandle::new(99)),
        display: rwh::RawDisplayHandle::Xlib(rwh::XlibDisplayHandle::new(Some(dpy), 0)),
    };
    let target = SurfaceTarget::from_window(&fake).unwrap();
    match target {
        SurfaceTarget::Xlib { display, window } => {
            assert_eq!(display, dpy.as_ptr());
            assert_eq!(window, 99);
        }
        other => panic!("expected Xlib, got {other:?}"),
    }
}

#[test]
fn from_window_maps_handle_errors() {
    let err = SurfaceTarget::from_window(&UnavailableWindow).unwrap_err();
    // Unavailable is a transient refusal — InvalidParam, carrying the
    // library's own description.
    match err.kind {
        QuantaErrorKind::InvalidParam(ref msg) => {
            assert!(
                msg.contains("not available"),
                "message should carry rwh's description: {msg}"
            );
        }
        other => panic!("expected InvalidParam, got {other:?}"),
    }
}

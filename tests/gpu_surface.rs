#![cfg(feature = "render")]
//! Presentation & interop — native-handle export + Surface frame loop.
//!
//! Part A: `Texture::native_handle()` exports the backend-native
//! object (borrow, valid for the Texture's lifetime).
//! Part B: `Gpu::create_surface` → acquire → render → present.
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::{Color, Format, NativeTextureHandle, QuantaErrorKind, SurfaceConfig, SurfaceTarget};

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- Part A: native-handle export ---

#[test]
fn native_handle_from_rendered_texture() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_native_handle_export() {
        eprintln!("skipping: backend does not export native handles");
        return;
    }

    let target = gpu.render_target(32, 32, Format::RGBA8).unwrap();
    let mut pulse = gpu.render(&target).unwrap().pulse().unwrap();
    pulse.wait().unwrap();

    let handle = target.native_handle().unwrap();
    match handle {
        NativeTextureHandle::Metal { texture } => {
            assert!(!texture.is_null(), "MTLTexture pointer must be non-null");
        }
        NativeTextureHandle::Vulkan { image, memory, .. } => {
            assert!(!image.is_null(), "VkImage must be non-null");
            assert!(!memory.is_null(), "VkDeviceMemory must be non-null");
        }
        other => panic!("unexpected native handle variant: {other:?}"),
    }
}

#[test]
fn native_handle_is_stable_across_calls() {
    // Borrow semantics: the same texture exports the same native
    // object every time while it lives.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_native_handle_export() {
        eprintln!("skipping: backend does not export native handles");
        return;
    }

    let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
    let (a, b) = (
        target.native_handle().unwrap(),
        target.native_handle().unwrap(),
    );
    match (a, b) {
        (
            NativeTextureHandle::Metal { texture: ta },
            NativeTextureHandle::Metal { texture: tb },
        ) => assert_eq!(ta, tb),
        (
            NativeTextureHandle::Vulkan { image: ia, .. },
            NativeTextureHandle::Vulkan { image: ib, .. },
        ) => assert_eq!(ia, ib),
        _ => {} // cross-variant impossible on one device
    }
}

// --- Part B: Surface frame loop (headless target) ---

#[test]
fn surface_acquire_render_present_loop() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let config = SurfaceConfig::new(64, 64);
    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &config)
        .unwrap();
    assert_eq!(surface.width(), 64);
    assert_eq!(surface.height(), 64);

    // More frames than the swapchain depth (3) proves drawables
    // recycle after present.
    for _ in 0..5 {
        let frame = surface.acquire().unwrap();
        assert_eq!(frame.texture().width(), 64);
        assert_eq!(frame.texture().height(), 64);
        assert_eq!(frame.texture().format(), Format::BGRA8);

        let mut pulse = gpu.render(frame.texture()).unwrap().pulse().unwrap();
        pulse.wait().unwrap();

        frame.present().unwrap();
    }
}

#[test]
fn surface_present_without_cpu_wait() {
    // The async contract: present immediately after submit, no
    // Pulse::wait in between.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(32, 32))
        .unwrap();
    for _ in 0..4 {
        let frame = surface.acquire().unwrap();
        let _pulse = gpu.render(frame.texture()).unwrap().pulse().unwrap();
        frame.present().unwrap(); // no wait — driver orders the present
    }
}

#[test]
fn surface_drop_unpresented_frame_recycles() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(16, 16))
        .unwrap();
    // Acquire and drop (discard) more frames than the swapchain
    // depth — recycling must keep working without presents.
    for _ in 0..5 {
        let frame = surface.acquire().unwrap();
        drop(frame);
    }
    // And a normal present still works afterwards.
    let frame = surface.acquire().unwrap();
    frame.present().unwrap();
}

#[test]
fn surface_reconfigure_resizes_frames() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(32, 32))
        .unwrap();
    let frame = surface.acquire().unwrap();
    frame.present().unwrap();

    surface.configure(SurfaceConfig::new(128, 64)).unwrap();
    assert_eq!(surface.width(), 128);
    let frame = surface.acquire().unwrap();
    assert_eq!(frame.texture().width(), 128);
    assert_eq!(frame.texture().height(), 64);
    frame.present().unwrap();
}

#[test]
fn surface_rejects_zero_extent() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let err = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(0, 32))
        .unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
}

#[test]
fn surface_rejects_null_layer() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }
    let err = gpu
        .create_surface(
            &SurfaceTarget::MetalLayer {
                layer: core::ptr::null_mut(),
            },
            &SurfaceConfig::new(32, 32),
        )
        .unwrap_err();
    // Metal rejects the null pointer (InvalidParam); backends without
    // a MetalLayer target reject the target itself (NotSupported).
    assert!(matches!(
        err.kind,
        QuantaErrorKind::InvalidParam(_) | QuantaErrorKind::NotSupported(_)
    ));
}

#[test]
fn surface_android_target_rejected_off_android() {
    // The Android surface leg: an `AndroidWindow` target only creates a
    // surface on an Android Vulkan that offers `VK_KHR_android_surface`.
    // Everywhere the suite actually runs — the Metal backend here, the
    // lavapipe Vulkan lane in CI — that extension is absent, so creation
    // must fail `NotSupported`. On Vulkan the failure names the missing
    // extension; on Metal the target is simply unavailable.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }
    // A non-null dummy so the Vulkan path reaches the extension check
    // rather than the null-pointer guard; Metal rejects the target
    // before inspecting the pointer.
    let a_native_window = core::ptr::NonNull::<core::ffi::c_void>::dangling().as_ptr();
    let err = gpu
        .create_surface(
            &SurfaceTarget::AndroidWindow { a_native_window },
            &SurfaceConfig::new(32, 32),
        )
        .unwrap_err();
    match err.kind {
        QuantaErrorKind::NotSupported(ref msg) => assert!(
            msg.contains("VK_KHR_android_surface")
                || msg.contains("not available on the Metal backend"),
            "unexpected NotSupported reason: {msg}"
        ),
        other => panic!("expected NotSupported, got {other:?}"),
    }
}

#[test]
fn surface_win32_target_rejected_off_windows() {
    // The Win32 surface leg: a `Win32` target only creates a surface on a
    // Windows Vulkan that offers `VK_KHR_win32_surface`. Everywhere the
    // suite actually runs — the Metal backend here, the lavapipe Vulkan
    // lane in CI — that extension is absent, so creation must fail
    // `NotSupported`. On Vulkan the failure names the missing extension;
    // on Metal the target is simply unavailable.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }
    // Non-null dummies so the Vulkan path reaches the extension check
    // rather than a null-pointer guard; Metal rejects the target before
    // inspecting the pointers.
    let hinstance = core::ptr::NonNull::<core::ffi::c_void>::dangling().as_ptr();
    let hwnd = core::ptr::NonNull::<core::ffi::c_void>::dangling().as_ptr();
    let err = gpu
        .create_surface(
            &SurfaceTarget::Win32 { hinstance, hwnd },
            &SurfaceConfig::new(32, 32),
        )
        .unwrap_err();
    match err.kind {
        QuantaErrorKind::NotSupported(ref msg) => assert!(
            msg.contains("VK_KHR_win32_surface")
                || msg.contains("not available on the Metal backend"),
            "unexpected NotSupported reason: {msg}"
        ),
        other => panic!("expected NotSupported, got {other:?}"),
    }
}

#[test]
fn surface_format_is_defined_after_create() {
    // The negotiated format is queryable right after create and is one
    // of the formats a presentation surface can actually carry.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(48, 48))
        .unwrap();
    let format = surface.format().unwrap();
    // Whatever the backend negotiated, it is a real presentable format;
    // the default request is BGRA8, and every backend the suite runs on
    // (Metal, lavapipe Vulkan) offers it, so the negotiation keeps it.
    assert_eq!(
        format,
        Format::BGRA8,
        "a BGRA8-requesting surface should negotiate BGRA8 where it is offered"
    );
    // And an acquired frame's texture reports the negotiated format.
    let mut surface = surface;
    let frame = surface.acquire().unwrap();
    assert_eq!(frame.texture().format(), format);
    frame.present().unwrap();
}

#[test]
fn surface_negotiates_requested_rgba8_or_falls_back() {
    // A surface requesting RGBA8. The negotiated format must be a member
    // of the preference chain [RGBA8, BGRA8, RGBA8], and an acquired
    // frame's texture must agree with `surface.format()` — the property
    // the encode-time color-format check depends on. Conditional on what
    // the surface offers so it can't fail falsely on any backend: Metal
    // rejects an RGBA8 surface outright (only BGRA8/RGBA16Float layers),
    // which is a legitimate outcome we skip on.
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut config = SurfaceConfig::new(40, 40);
    config.format = Format::RGBA8;
    let mut surface = match gpu.create_surface(&SurfaceTarget::Headless, &config) {
        Ok(s) => s,
        Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
            // Backend can't present an RGBA8 surface at all (Metal). The
            // negotiation still applies wherever RGBA8 or the fallbacks
            // are offered — nothing to assert here.
            eprintln!("skipping: RGBA8 surface not supported on this backend: {e}");
            return;
        }
        Err(e) => panic!("create_surface(RGBA8) failed unexpectedly: {e}"),
    };

    let negotiated = surface.format().unwrap();
    assert!(
        matches!(negotiated, Format::RGBA8 | Format::BGRA8),
        "negotiated format must be a chain member [requested=RGBA8, BGRA8, RGBA8], got {negotiated:?}"
    );

    let frame = surface.acquire().unwrap();
    assert_eq!(
        frame.texture().format(),
        negotiated,
        "an acquired frame's texture must report the negotiated format"
    );
    frame.present().unwrap();
}

// --- Reserved-but-NotSupported backends (CPU software driver) ---

#[cfg(feature = "software")]
mod cpu_not_supported {
    use super::*;

    #[test]
    fn cpu_native_handle_not_supported() {
        let gpu = quanta::init_cpu();
        assert!(!gpu.supports_native_handle_export());
        let target = gpu.render_target(8, 8, Format::RGBA8).unwrap();
        let err = target.native_handle().unwrap_err();
        assert!(matches!(err.kind, QuantaErrorKind::NotSupported(_)));
    }

    #[test]
    fn cpu_surface_not_supported() {
        let gpu = quanta::init_cpu();
        assert!(!gpu.supports_surface_present());
        let err = gpu
            .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(32, 32))
            .unwrap_err();
        assert!(matches!(err.kind, QuantaErrorKind::NotSupported(_)));
    }
}

// --- Part C: demand-driven pacing ---

/// One acquire→render→present frame.
fn drive_frame(gpu: &quanta::Gpu, surface: &mut quanta::Surface) {
    let frame = surface.acquire().unwrap();
    let mut pulse = gpu.render(frame.texture()).unwrap().pulse().unwrap();
    pulse.wait().unwrap();
    frame.present().unwrap();
}

/// The pacing contract from the `Surface` docs: the frame loop may run
/// at ANY cadence — long idle gaps, bursts deeper than the swapchain,
/// then idle again — without leaking frame textures or stalling.
fn assert_demand_driven_cadence(gpu: &quanta::Gpu, surface: &mut quanta::Surface) {
    let baseline = gpu.debug_registry_counts();

    // Sparse: frames separated by real idle gaps (an idle UI waking
    // on a dirty flag).
    for _ in 0..3 {
        std::thread::sleep(std::time::Duration::from_millis(120));
        drive_frame(gpu, surface);
    }

    // Burst: back-to-back frames, deeper than the drawable pool (3),
    // throttled only by acquire's back-pressure.
    for _ in 0..8 {
        drive_frame(gpu, surface);
    }

    // Idle again, then one more frame — the cadence changes both ways.
    std::thread::sleep(std::time::Duration::from_millis(150));
    drive_frame(gpu, surface);

    assert_eq!(
        gpu.debug_registry_counts(),
        baseline,
        "cadence changes must not leak registry entries"
    );
}

#[test]
fn surface_sparse_then_burst_cadence() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let config = SurfaceConfig::new(32, 32);
    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &config)
        .unwrap();
    assert_demand_driven_cadence(&gpu, &mut surface);
}

// --- Part D: `render_frame` — the closure over acquire/present ---

#[test]
fn render_frame_runs_closure_and_presents() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(64, 64))
        .unwrap();
    let baseline = gpu.debug_registry_counts();

    // More frames than the swapchain depth (3): each Ok closure must
    // end in a present, or the drawable pool runs dry.
    for i in 0..5u32 {
        let value = surface
            .render_frame(|frame| {
                assert_eq!(frame.texture().width(), 64);
                let mut pulse = gpu.render(frame.texture())?.pulse()?;
                pulse.wait()?;
                Ok(i) // the closure's value comes back out
            })
            .unwrap();
        assert_eq!(value, i);
    }

    assert_eq!(
        gpu.debug_registry_counts(),
        baseline,
        "render_frame must not leak registry entries"
    );
}

#[test]
fn render_frame_closure_error_discards_and_recovers() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(32, 32))
        .unwrap();

    // A failing closure: the error propagates untouched and the frame
    // is discarded, not presented.
    let err = surface
        .render_frame(|_frame| -> Result<(), quanta::QuantaError> {
            Err(quanta::QuantaError::invalid_param("scene not ready"))
        })
        .unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));

    // The loop keeps working after the failure — deeper than the
    // swapchain, so the discarded frame demonstrably recycled.
    for _ in 0..4 {
        surface
            .render_frame(|frame| {
                let mut pulse = gpu.render(frame.texture())?.pulse()?;
                pulse.wait()?;
                Ok(())
            })
            .unwrap();
    }
}

/// Minimal ObjC FFI to create a standalone `CAMetalLayer` — no window
/// needed; the driver configures the layer (device/format/size) itself.
#[cfg(target_os = "macos")]
mod layer_ffi {
    use core::ffi::{c_char, c_void};

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }
    // Linking QuartzCore registers the CAMetalLayer class with the
    // ObjC runtime.
    #[link(name = "QuartzCore", kind = "framework")]
    unsafe extern "C" {}

    /// `[CAMetalLayer new]`
    pub fn new_metal_layer() -> *mut c_void {
        unsafe {
            let cls = objc_getClass(c"CAMetalLayer".as_ptr());
            assert!(!cls.is_null(), "CAMetalLayer class not found");
            let sel = sel_registerName(c"new".as_ptr());
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                core::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            send(cls, sel)
        }
    }

    /// `[layer release]` — balance the +1 from `new`.
    pub fn release(layer: *mut c_void) {
        unsafe {
            let sel = sel_registerName(c"release".as_ptr());
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                core::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            send(layer, sel);
        }
    }
}

#[test]
fn surface_rejects_null_appkit_view() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }
    let err = gpu
        .create_surface(
            &SurfaceTarget::AppKitView {
                ns_view: core::ptr::null_mut(),
            },
            &SurfaceConfig::new(32, 32),
        )
        .unwrap_err();
    // Metal rejects the null pointer (InvalidParam); backends without
    // an AppKitView target reject the target itself (NotSupported).
    assert!(matches!(
        err.kind,
        QuantaErrorKind::InvalidParam(_) | QuantaErrorKind::NotSupported(_)
    ));
}

/// Minimal ObjC FFI to create a standalone `NSView` — the shape a
/// raw-window-handle producer hands over as the macOS window handle.
/// AppKit view classes instantiate fine without an NSApplication run
/// loop; the view simply never appears on screen.
#[cfg(target_os = "macos")]
mod view_ffi {
    use core::ffi::{c_char, c_void};

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> *mut c_void;
        fn sel_registerName(name: *const c_char) -> *mut c_void;
        fn objc_msgSend();
    }
    // Linking AppKit registers the NSView class with the ObjC runtime.
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}

    /// `[NSView new]` — zero frame; the surface config sizes the layer.
    pub fn new_ns_view() -> *mut c_void {
        unsafe {
            let cls = objc_getClass(c"NSView".as_ptr());
            assert!(!cls.is_null(), "NSView class not found");
            let sel = sel_registerName(c"new".as_ptr());
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                core::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            send(cls, sel)
        }
    }

    /// `[obj release]` — balance the +1 from `new`.
    pub fn release(obj: *mut c_void) {
        unsafe {
            let sel = sel_registerName(c"release".as_ptr());
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                core::mem::transmute(objc_msgSend as unsafe extern "C" fn());
            send(obj, sel);
        }
    }
}

/// The raw-window-handle handoff, end to end on this Mac: a bare
/// `NSView` (what rwh's AppKit handle carries) goes through
/// `SurfaceTarget::AppKitView`; the Metal driver attaches the
/// `CAMetalLayer` itself. A second surface over the same view must
/// REUSE that layer (the driver's isKindOfClass arm) rather than
/// replace it.
#[cfg(target_os = "macos")]
#[test]
fn appkit_view_surface_end_to_end() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let ns_view = view_ffi::new_ns_view();
    assert!(!ns_view.is_null());
    let target = SurfaceTarget::AppKitView { ns_view };

    match gpu.create_surface(&target, &SurfaceConfig::new(64, 64)) {
        Ok(mut surface) => {
            // Deeper than the drawable pool: presents must recycle.
            for _ in 0..4 {
                surface
                    .render_frame(|frame| {
                        assert_eq!(frame.texture().width(), 64);
                        assert_eq!(frame.texture().format(), Format::BGRA8);
                        let mut pulse = gpu.render(frame.texture())?.pulse()?;
                        pulse.wait()?;
                        Ok(())
                    })
                    .unwrap();
            }
            drop(surface);

            // Round 2 over the same view: the layer attached in round
            // 1 is still on the view (the view retains it) and gets
            // reused; the frame loop must work identically.
            let mut surface = gpu
                .create_surface(&target, &SurfaceConfig::new(32, 32))
                .unwrap();
            for _ in 0..4 {
                surface
                    .render_frame(|frame| {
                        assert_eq!(frame.texture().width(), 32);
                        let mut pulse = gpu.render(frame.texture())?.pulse()?;
                        pulse.wait()?;
                        Ok(())
                    })
                    .unwrap();
            }
            drop(surface);
            view_ffi::release(ns_view);
        }
        Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
            // A non-Metal backend on macOS (forced Vulkan) has no
            // AppKitView path yet (MoltenVK deferral).
            eprintln!("skipping: AppKitView target not supported: {e}");
            view_ffi::release(ns_view);
        }
        Err(e) => panic!("create_surface(AppKitView) failed: {e}"),
    }
}

/// Same contract over a caller-provided `CAMetalLayer` — the real
/// windowing-integration path (`SurfaceTarget::MetalLayer`).
#[cfg(target_os = "macos")]
#[test]
fn metal_layer_demand_driven_cadence() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let layer = layer_ffi::new_metal_layer();
    assert!(!layer.is_null());

    let config = SurfaceConfig::new(64, 64);
    match gpu.create_surface(&SurfaceTarget::MetalLayer { layer }, &config) {
        Ok(mut surface) => {
            assert_demand_driven_cadence(&gpu, &mut surface);
            drop(surface);
            layer_ffi::release(layer);
        }
        Err(e) if matches!(e.kind, QuantaErrorKind::NotSupported(_)) => {
            // A non-Metal backend on macOS (forced Vulkan) has no
            // CAMetalLayer path — the Headless variant covers it.
            eprintln!("skipping: MetalLayer target not supported: {e}");
            layer_ffi::release(layer);
        }
        Err(e) => panic!("create_surface(MetalLayer) failed: {e}"),
    }
}

/// The windowed effect-frame shape, headless: MSAA render into the
/// acquired frame (pooled intermediate + subpass resolve), a per-frame
/// scratch texture dropped while the frame is in flight, present with
/// no CPU wait, and a pulse held across the frame boundary — the
/// closest CI walk of the swapchain path that only the Iris Xe rig used
/// to see. Guards the fence-deferred destroy + checked present-fence
/// recycling as one piece.
#[test]
fn surface_msaa_effect_frame_loop_with_midflight_drops() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    if !gpu.supports_surface_present() {
        eprintln!("skipping: backend has no present path");
        return;
    }

    let mut surface = gpu
        .create_surface(&SurfaceTarget::Headless, &SurfaceConfig::new(64, 64))
        .unwrap();
    let mut prev_pulse: Option<quanta::Pulse> = None;
    for _ in 0..6 {
        let frame = surface.acquire().unwrap();
        // A per-frame "effect intermediate", dropped mid-flight below.
        let scratch = gpu.render_target(64, 64, Format::RGBA8).unwrap();
        let pulse = gpu
            .render(frame.texture())
            .unwrap()
            .msaa(4)
            .clear(Color::rgba(0.2, 0.4, 0.6, 1.0))
            .pulse()
            .unwrap();
        drop(scratch); // destroyed while this frame is still executing
        frame.present().unwrap(); // async — no CPU wait before present
        prev_pulse = Some(pulse); // previous frame's pulse released here
    }
    drop(prev_pulse);
}

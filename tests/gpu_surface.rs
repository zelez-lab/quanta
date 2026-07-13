#![cfg(feature = "render")]
//! Presentation & interop — native-handle export + Surface frame loop.
//!
//! Part A: `Texture::native_handle()` exports the backend-native
//! object (borrow, valid for the Texture's lifetime).
//! Part B: `Gpu::create_surface` → acquire → render → present.
//! Requires a GPU; skips gracefully if none available.

use quanta::RenderGpu;

use quanta::{Format, NativeTextureHandle, QuantaErrorKind, SurfaceConfig, SurfaceTarget};

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
    // The Android surface leg: a `VulkanAndroid` target only creates a
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
            &SurfaceTarget::VulkanAndroid { a_native_window },
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

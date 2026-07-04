#![cfg(feature = "render")]
//! Presentation & interop — native-handle export + Surface frame loop.
//!
//! Part A: `Texture::native_handle()` exports the backend-native
//! object (borrow, valid for the Texture's lifetime).
//! Part B: `Gpu::create_surface` → acquire → render → present.
//! Requires a GPU; skips gracefully if none available.

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
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
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

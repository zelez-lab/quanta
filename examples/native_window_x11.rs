//! A real X11 window running Quanta's demand-driven present loop on the
//! Vulkan swapchain (`SurfaceTarget::Xlib`).
//!
//! Run: `cargo run --example native_window_x11 --no-default-features \
//!       --features vulkan,render`
//!
//! The Linux twin of `native_window.rs` (macOS): everything windowing-
//! related is bare Xlib FFI — no windowing crates, per the
//! no-transitive-deps policy. The demo is clear-only (an animated clear
//! color, no shaders), so it runs on a machine with no `quanta-compiler`
//! or LLVM toolchain — exactly what a fresh validation rig has. Three
//! pacing phases share one acquire → render → present loop:
//!
//! 1. **sparse**   — 2 fps, long idle gaps (an idle UI redrawing on demand)
//! 2. **burst**    — uncapped; `acquire` back-pressure throttles to the display
//! 3. **animated** — a steady clock with a color sweep
//!
//! The windowed lane never calls `pulse.wait()` — presentation is
//! ordered after the submitted GPU work by the present semaphore.
//!
//! Rig gotcha: with FIFO, a DPMS-blanked monitor consumes no vblanks,
//! so `vkQueuePresentKHR` blocks forever once the swapchain queue
//! fills. Wake the display first (`xset s off -dpms; xset dpms force
//! on`) — or use `SurfaceTarget::Headless`, which has no such
//! dependency (that's what the CI lane does).

#[cfg(not(all(target_os = "linux", feature = "render")))]
fn main() {
    eprintln!(
        "native_window_x11 is a Linux example (Xlib + VkSwapchainKHR); \
         build with --features vulkan,render"
    );
}

/// Minimal Xlib bindings: open a display, make a window, pump events.
#[cfg(all(target_os = "linux", feature = "render"))]
mod xlib {
    use core::ffi::{c_char, c_int, c_uint, c_ulong, c_void};

    pub type Display = c_void;

    /// XEvent is a C union sized as 24 longs; the first long is the
    /// event type. This is the standard opaque-pad representation.
    #[repr(C)]
    pub struct XEvent {
        pub pad: [c_ulong; 24],
    }

    pub const STRUCTURE_NOTIFY_MASK: i64 = 1 << 17;
    pub const EXPOSURE_MASK: i64 = 1 << 15;

    #[link(name = "X11")]
    unsafe extern "C" {
        pub fn XInitThreads() -> c_int;
        pub fn XOpenDisplay(name: *const c_char) -> *mut Display;
        pub fn XDefaultScreen(d: *mut Display) -> c_int;
        pub fn XRootWindow(d: *mut Display, screen: c_int) -> c_ulong;
        pub fn XCreateSimpleWindow(
            d: *mut Display,
            parent: c_ulong,
            x: c_int,
            y: c_int,
            w: c_uint,
            h: c_uint,
            border_w: c_uint,
            border: c_ulong,
            background: c_ulong,
        ) -> c_ulong;
        pub fn XStoreName(d: *mut Display, w: c_ulong, name: *const c_char) -> c_int;
        pub fn XSelectInput(d: *mut Display, w: c_ulong, mask: i64) -> c_int;
        pub fn XMapWindow(d: *mut Display, w: c_ulong) -> c_int;
        pub fn XFlush(d: *mut Display) -> c_int;
        pub fn XPending(d: *mut Display) -> c_int;
        pub fn XNextEvent(d: *mut Display, ev: *mut XEvent) -> c_int;
        pub fn XDestroyWindow(d: *mut Display, w: c_ulong) -> c_int;
        pub fn XCloseDisplay(d: *mut Display) -> c_int;
    }
}

#[cfg(all(target_os = "linux", feature = "render"))]
fn main() {
    use quanta::RenderGpu;
    use quanta::render_pass::ColorTarget;
    use quanta::{Color, LoadOp, StoreOp};

    let stay = std::env::args().any(|a| a == "--stay");

    // --- Xlib: display + window ---
    let (display, window) = unsafe {
        // Mesa's Vulkan WSI runs its own threads over this Display
        // connection — Xlib is only thread-safe after XInitThreads(),
        // and it must be the FIRST Xlib call.
        xlib::XInitThreads();
        let display = xlib::XOpenDisplay(core::ptr::null());
        assert!(
            !display.is_null(),
            "cannot open the X display — is DISPLAY set?"
        );
        let screen = xlib::XDefaultScreen(display);
        let root = xlib::XRootWindow(display, screen);
        let window = xlib::XCreateSimpleWindow(display, root, 100, 100, 640, 480, 0, 0, 0);
        xlib::XStoreName(
            display,
            window,
            c"Quanta \u{2014} Vulkan swapchain".as_ptr(),
        );
        xlib::XSelectInput(
            display,
            window,
            xlib::STRUCTURE_NOTIFY_MASK | xlib::EXPOSURE_MASK,
        );
        xlib::XMapWindow(display, window);
        xlib::XFlush(display);
        (display, window)
    };

    // --- Quanta: surface over the X window ---
    let gpu = quanta::init().expect("no GPU");
    println!("backend: {}", gpu.name());
    assert!(
        gpu.supports_surface_present(),
        "this Vulkan environment offers no WSI extensions"
    );
    let config = quanta::SurfaceConfig::new(640, 480);
    let mut surface = gpu
        .create_surface(
            &quanta::SurfaceTarget::Xlib {
                display: display as *mut core::ffi::c_void,
                window,
            },
            &config,
        )
        .expect("surface over the X11 window");

    let pump_events = || unsafe {
        while xlib::XPending(display) > 0 {
            let mut ev = xlib::XEvent { pad: [0; 24] };
            xlib::XNextEvent(display, &mut ev);
        }
    };

    let mut frame_no = 0u32;
    let mut draw_frame = |surface: &mut quanta::Surface| {
        pump_events();
        let t = frame_no as f32 * 0.05;
        let clear = Color::rgba(
            0.12 + 0.10 * t.sin().abs(),
            0.30 + 0.20 * (t * 1.3).sin().abs(),
            0.25 + 0.15 * (t * 0.7).cos().abs(),
            1.0,
        );
        // A window manager may have resized the window — reconfigure
        // adopts the surface's real extent and the loop continues.
        let frame = match surface.acquire() {
            Ok(frame) => frame,
            Err(e) if matches!(e.kind, quanta::QuantaErrorKind::SurfaceOutdated(_)) => {
                surface
                    .configure(quanta::SurfaceConfig::new(640, 480))
                    .expect("reconfigure");
                surface.acquire().expect("acquire after reconfigure")
            }
            Err(e) => panic!("acquire failed: {e}"),
        };
        // No pulse.wait(): present is ordered after the submitted work
        // by the present semaphore — the windowed lane never blocks.
        let _pulse = gpu
            .render(frame.texture())
            .expect("pass")
            .color_targets(vec![
                ColorTarget::new(frame.texture())
                    .with_load_op(LoadOp::Clear(clear))
                    .with_store_op(StoreOp::Store),
            ])
            .pulse()
            .expect("submit");
        frame.present().expect("present");
        frame_no += 1;
    };

    // --- Phase 1: sparse (2 fps — an idle UI redrawing on demand) ---
    println!("phase 1: sparse — 4 frames, 500 ms apart");
    for _ in 0..4 {
        draw_frame(&mut surface);
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    // --- Phase 2: burst (uncapped; acquire throttles to the display) ---
    println!("phase 2: burst — 120 frames, no sleep");
    let t0 = std::time::Instant::now();
    for _ in 0..120 {
        draw_frame(&mut surface);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "  120 frames in {dt:.2}s = {:.0} fps (acquire back-pressure)",
        120.0 / dt
    );

    // --- Phase 3: animated (steady clock) ---
    println!(
        "phase 3: animated — {}",
        if stay {
            "until Ctrl-C"
        } else {
            "3 s (pass --stay to keep it open)"
        }
    );
    let t0 = std::time::Instant::now();
    while stay || t0.elapsed().as_secs_f64() < 3.0 {
        draw_frame(&mut surface);
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    println!("done: {frame_no} frames total, zero CPU waits on the GPU");
    drop(surface);
    unsafe {
        xlib::XDestroyWindow(display, window);
        xlib::XCloseDisplay(display);
    }
}

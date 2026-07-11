//! A real macOS window running Quanta's demand-driven present loop.
//!
//! Run: `cargo run --example native_window`            (auto-exits after the demo)
//!      `cargo run --example native_window -- --stay`  (keeps animating until closed)
//!
//! Everything windowing-related is bare Cocoa FFI (`objc_msgSend`) — no
//! windowing crates, matching the no-transitive-deps policy. One
//! acquire → render → present loop drives three pacing phases to show
//! the surface imposes no frame scheduler of its own:
//!
//! 1. **sparse**   — 2 fps, long idle gaps (an idle UI redrawing on demand)
//! 2. **burst**    — uncapped; `acquire` back-pressure throttles to the display
//! 3. **animated** — a steady clock with a color sweep
//!
//! The window lane never calls `pulse.wait()`: presentation is ordered
//! after the submitted GPU work by the driver (the async-pulse contract).

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("native_window is a macOS example (CAMetalLayer + NSWindow).");
}

#[cfg(target_os = "macos")]
#[quanta::vertex]
fn window_vertex(pos: Vec3, color: Vec3) -> Vec4 {
    Vec4::new(pos.x, pos.y, pos.z, 1.0)
}

#[cfg(target_os = "macos")]
#[quanta::fragment]
fn window_fragment() -> Vec4 {
    Vec4::new(1.0, 0.42, 0.21, 1.0)
}

/// Minimal Cocoa bindings: just enough `objc_msgSend` casts to open a
/// window, host a `CAMetalLayer`, and pump events.
#[cfg(target_os = "macos")]
mod cocoa {
    use core::ffi::{c_char, c_void};

    pub type Id = *mut c_void;
    pub type Sel = *mut c_void;

    #[link(name = "objc")]
    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
    }
    #[link(name = "AppKit", kind = "framework")]
    unsafe extern "C" {}
    #[link(name = "QuartzCore", kind = "framework")]
    unsafe extern "C" {}

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct CGRect {
        pub x: f64,
        pub y: f64,
        pub w: f64,
        pub h: f64,
    }

    pub fn cls(name: &'static core::ffi::CStr) -> Id {
        let c = unsafe { objc_getClass(name.as_ptr()) };
        assert!(!c.is_null(), "class not found: {name:?}");
        c
    }

    pub fn sel(name: &'static core::ffi::CStr) -> Sel {
        unsafe { sel_registerName(name.as_ptr()) }
    }

    macro_rules! msg {
        ($name:ident, ($($arg:ident: $ty:ty),*) -> $ret:ty) => {
            pub fn $name(obj: Id, s: Sel, $($arg: $ty),*) -> $ret {
                unsafe {
                    let f: unsafe extern "C" fn(Id, Sel, $($ty),*) -> $ret =
                        core::mem::transmute(objc_msgSend as unsafe extern "C" fn());
                    f(obj, s, $($arg),*)
                }
            }
        };
    }

    msg!(msg0, () -> Id);
    msg!(msg1, (a: Id) -> Id);
    msg!(msg_u64, (a: u64) -> Id);
    msg!(msg_bool, (a: i8) -> Id);
    msg!(msg_str, (a: *const c_char) -> Id);
    msg!(msg_is, () -> i8);
    msg!(msg_window_init, (rect: CGRect, style: u64, backing: u64, defer: i8) -> Id);
    msg!(msg_next_event, (mask: u64, until: Id, mode: Id, dequeue: i8) -> Id);

    pub fn nsstring(s: &core::ffi::CStr) -> Id {
        msg_str(cls(c"NSString"), sel(c"stringWithUTF8String:"), s.as_ptr())
    }
}

#[cfg(target_os = "macos")]
fn main() {
    use cocoa::*;
    use quanta::render_pass::ColorTarget;
    use quanta::{Color, FieldUsage, Format, LoadOp, RenderGpu, StoreOp};

    let stay = std::env::args().any(|a| a == "--stay");

    // --- Cocoa: app, window, layer-hosting content view ---
    let app = msg0(cls(c"NSApplication"), sel(c"sharedApplication"));
    msg_u64(app, sel(c"setActivationPolicy:"), 0); // Regular: dock icon + key window

    let window = msg_window_init(
        msg0(cls(c"NSWindow"), sel(c"alloc")),
        sel(c"initWithContentRect:styleMask:backing:defer:"),
        CGRect {
            x: 200.0,
            y: 200.0,
            w: 640.0,
            h: 480.0,
        },
        15, // titled | closable | miniaturizable | resizable
        2,  // buffered backing store
        0,
    );
    msg1(
        window,
        sel(c"setTitle:"),
        nsstring(c"Quanta \u{2014} demand-driven present"),
    );

    let layer = msg0(cls(c"CAMetalLayer"), sel(c"new"));
    let content_view = msg0(window, sel(c"contentView"));
    msg1(content_view, sel(c"setLayer:"), layer); // layer-hosting: set layer first,
    msg_bool(content_view, sel(c"setWantsLayer:"), 1); // then wantsLayer

    msg1(window, sel(c"makeKeyAndOrderFront:"), core::ptr::null_mut());
    msg_bool(app, sel(c"activateIgnoringOtherApps:"), 1);

    // --- Quanta: surface over the layer + a triangle pipeline ---
    let gpu = quanta::init().expect("no GPU");
    let mut surface = gpu
        .create_surface(
            &quanta::SurfaceTarget::MetalLayer { layer },
            &quanta::SurfaceConfig::new(1280, 960), // 2x for Retina
        )
        .expect("surface over the window's CAMetalLayer");

    let pipeline = {
        let layouts = vec![quanta::VertexLayout {
            stride: 24,
            step: quanta::StepMode::Vertex,
            attributes: vec![
                quanta::VertexAttribute {
                    location: 0,
                    offset: 0,
                    format: quanta::AttributeFormat::Float3,
                },
                quanta::VertexAttribute {
                    location: 1,
                    offset: 12,
                    format: quanta::AttributeFormat::Float3,
                },
            ],
        }];
        let desc = quanta::PipelineDesc::new(quanta::ShaderSource::Binaries {
            vertex: &WINDOW_VERTEX_SHADER,
            fragment: &WINDOW_FRAGMENT_SHADER,
        })
        .with_entries(
            WINDOW_VERTEX_SHADER.entry_point,
            WINDOW_FRAGMENT_SHADER.entry_point,
        )
        .with_color_formats(vec![Format::BGRA8]) // surface frames are BGRA8
        .with_vertex_layouts(&layouts)
        .with_blend(quanta::BlendState::NONE);
        gpu.pipeline(&desc).expect("pipeline")
    };

    #[rustfmt::skip]
    let verts: [f32; 18] = [
         0.0, -0.6, 0.5,   0.0, 0.0, 0.0,
        -0.6,  0.6, 0.5,   0.0, 0.0, 0.0,
         0.6,  0.6, 0.5,   0.0, 0.0, 0.0,
    ];
    let vb: quanta::Field<f32> = gpu
        .field_with_usage(verts.len(), FieldUsage::default_render())
        .unwrap();
    vb.write(&verts).unwrap();

    // --- The one frame function every phase shares ---
    let run_loop_mode = nsstring(c"kCFRunLoopDefaultMode");
    let distant_past = msg0(cls(c"NSDate"), sel(c"distantPast"));
    let pump_events = || loop {
        let event = msg_next_event(
            app,
            sel(c"nextEventMatchingMask:untilDate:inMode:dequeue:"),
            u64::MAX,
            distant_past,
            run_loop_mode,
            1,
        );
        if event.is_null() {
            break;
        }
        msg1(app, sel(c"sendEvent:"), event);
    };
    let window_open = || msg_is(window, sel(c"isVisible")) != 0;

    let mut frame_no = 0u32;
    let mut draw_frame = |surface: &mut quanta::Surface| {
        pump_events();
        let t = frame_no as f32 * 0.05;
        let clear = Color::rgba(
            0.12 + 0.10 * t.sin().abs(),
            0.12,
            0.25 + 0.15 * (t * 0.7).cos().abs(),
            1.0,
        );
        let frame = surface.acquire().unwrap();
        // No pulse.wait(): presentation is ordered after the submitted
        // work by the driver — the windowed lane never blocks the CPU.
        let _pulse = gpu
            .render(frame.texture())
            .unwrap()
            .color_targets(vec![
                ColorTarget::new(frame.texture())
                    .with_load_op(LoadOp::Clear(clear))
                    .with_store_op(StoreOp::Store),
            ])
            .viewport(0.0, 0.0, 1280.0, 960.0)
            .pipeline(&pipeline)
            .vertices(0, &vb)
            .draw(3)
            .pulse()
            .unwrap();
        frame.present().unwrap();
        frame_no += 1;
    };

    // --- Phase 1: sparse (2 fps — an idle UI redrawing on demand) ---
    println!("phase 1: sparse — 4 frames, 500 ms apart");
    for _ in 0..4 {
        if !window_open() {
            return;
        }
        draw_frame(&mut surface);
        std::thread::sleep(std::time::Duration::from_millis(500));
    }

    // --- Phase 2: burst (uncapped; acquire throttles to the display) ---
    println!("phase 2: burst — 120 frames, no sleep");
    let t0 = std::time::Instant::now();
    for _ in 0..120 {
        if !window_open() {
            return;
        }
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
            "until the window closes"
        } else {
            "3 s (pass --stay to keep it open)"
        }
    );
    let t0 = std::time::Instant::now();
    while window_open() && (stay || t0.elapsed().as_secs_f64() < 3.0) {
        draw_frame(&mut surface);
        std::thread::sleep(std::time::Duration::from_millis(16));
    }

    println!("done: {frame_no} frames total, zero CPU waits on the GPU");
}

# Presenting to the screen

> **You'll learn:** how rendered pixels leave Quanta — either Quanta presents
> them to a platform surface itself, or you export the texture and let a
> compositor own present.

Everything so far rendered into an offscreen texture. Quanta offers two ways
to get that image on screen:

1. **Quanta owns present** — create a `Surface` (a swapchain over a
   platform target) and run the acquire → render → present frame loop.
2. **Compositor owns present** — export the texture's backend-native object
   with `Texture::native_handle()` and hand it to an external compositor,
   zero-copy.

Both are capability-gated; query before use:

```rust
use quanta::*; // brings the RenderGpu extension trait into scope

if gpu.supports_surface_present() { /* model 1 available */ }
if gpu.supports_native_handle_export() { /* model 2 available */ }
```

## Quanta owns present: `Surface`

Create a surface over a platform target with `gpu.create_surface` (a
`RenderGpu` extension method). `SurfaceConfig::new(width, height)` defaults
to `BGRA8`, `PresentMode::Fifo` (vsync), and `RENDER_TARGET` usage.

### From a window: the one-value handoff

With the `raw-window-handle` feature on, `SurfaceTarget::from_window` takes
any winit-style window — anything implementing raw-window-handle 0.6's
`HasWindowHandle + HasDisplayHandle` — straight to a target, with no per-OS
matching:

```rust,ignore
// `window` is a winit Window (or any rwh 0.6 handle source).
let target = SurfaceTarget::from_window(&window)?;
let mut surface = gpu.create_surface(&target, &SurfaceConfig::new(1280, 720))?;
```

`from_window` maps `AppKit → AppKitView` (the Metal driver attaches the
`CAMetalLayer`), `Xlib → Xlib`, `Win32 → Win32`, and `AndroidNdk →
AndroidWindow`; Wayland is a documented deferral (`NotSupported` — run
under XWayland for now, forcing your windowing library's X11 backend so
the window arrives as an `Xlib` handle). The mapping is **pure** — no OS
calls happen — and the window and its display connection must outlive the
surface. Callers already holding raw handles can use
`SurfaceTarget::from_raw(window, display)` directly. The `raw-window-handle`
crate is re-exported as `quanta::rwh`, so you need no dependency line of
your own.

### By hand

When you hold a platform handle directly, name the variant:

```rust
// A CAMetalLayer handed to you by the windowing environment:
let target = SurfaceTarget::MetalLayer { layer };
// or SurfaceTarget::Headless — full acquire/present machinery, no window
// (tests, warm-up, composition through another channel).

let mut surface = gpu.create_surface(&target, &SurfaceConfig::new(1280, 720))?;
```

### The frame loop

The frame loop is one closure per frame — `render_frame` folds acquire →
render → present into a single call and self-heals on resize:

```rust
loop {
    surface.render_frame(|frame| {
        // Render into the frame through the ordinary render-pass API.
        gpu.render(frame.texture())?
            .clear(Color::BLACK)
            .pipeline(&pipeline)
            .vertices(0, &vb)
            .draw(3)
            .pulse()?;
        Ok(())
    })?;
}
```

The closure renders into `frame.texture()` and submits with `.pulse()`;
`render_frame` presents when it returns `Ok` (no CPU wait needed — the
driver orders presentation after the submitted GPU work). On a closure
`Err` the frame drops **unpresented** and the error propagates. When
`acquire` reports `SurfaceOutdated` (the window resized) and the driver can
read the target's current extent (Metal `drawableSize`, Vulkan
`currentExtent`), `render_frame` reconfigures to it and retries the acquire
**once** — the healed extent shows through `surface.config()` / `width()` /
`height()`. `Timeout` propagates for you to retry next iteration.

#### The manual loop

When the loop needs custom resize or timeout policy, spell it out over the
primitives (`acquire` / `SurfaceFrame::present`):

```rust
loop {
    let frame = match surface.acquire() {
        Ok(frame) => frame,
        Err(e) if matches!(e.kind, QuantaErrorKind::SurfaceOutdated(_)) => {
            // Window resized — reconfigure with the new extent, retry.
            surface.configure(SurfaceConfig::new(new_w, new_h))?;
            continue;
        }
        Err(e) if matches!(e.kind, QuantaErrorKind::Timeout) => continue,
        Err(e) => return Err(e),
    };

    // Render into the frame through the ordinary render-pass API.
    let mut pulse = gpu.render(frame.texture())?
        .clear(Color::BLACK)
        .pipeline(&pipeline)
        .vertices(0, &vb)
        .draw(3)
        .pulse()?;

    // Present after the pass is SUBMITTED (`.pulse()` returned). No CPU
    // wait is needed: the driver orders presentation after the submitted
    // GPU work, asynchronously.
    frame.present()?;
}
```

Rules of the loop:

- **The frame texture is a borrow.** `frame.texture()` aliases the
  swapchain's backing image — valid only until the frame is presented or
  dropped. Don't store it (or its `native_handle()`) across iterations;
  acquire a fresh frame each time.
- **Dropping an unpresented frame discards it** — the image returns to the
  swapchain unshown. That's the correct way to skip a frame.
- **Reconfigure on `SurfaceOutdated`.** Present or drop any acquired frames
  first, then `surface.configure(...)` with the new extent.
- Dropping the `Surface` releases the swapchain (and, for
  `SurfaceTarget::Headless`, the backend-created target).

### Format negotiation

`SurfaceConfig::format` is a **preference**, not a guarantee. A
presentation surface only offers a restricted set of formats, and on
Vulkan the set is platform-dependent — Android surfaces conventionally
offer `RGBA8`, not the `BGRA8` desktop habit assumes. So the swapchain
negotiates: it picks the first format the surface offers, all with an
SRGB-nonlinear colorspace, from the chain

1. the format you requested (`config.format`),
2. `BGRA8`,
3. `RGBA8`,
4. otherwise the first offered format Quanta can express.

Only a surface offering nothing Quanta can name fails, and the error
lists what it offered. On Metal there is no negotiation — Quanta sets the
layer's format, so the frames always use exactly what you configured.

Read the negotiated result with `surface.format()` and build your
pipelines against it — the frame texture carries the negotiated format,
and a pipeline typed for a different one is rejected when you draw:

```rust
let surface = gpu.create_surface(&target, &SurfaceConfig::new(w, h))?;
let color = surface.format()?; // may not be the BGRA8 you asked for
let pipeline = gpu.pipeline(&PipelineDesc::new(&shader).with_color_formats(vec![color]))?;
```

The chain order is fixed. If you need the fallback to prefer a different
format, build the pipeline per frame from `frame.texture().format()`
instead.

### Pacing: fully demand-driven

Quanta never renders or presents on its own — a frame happens only when you
run the loop body. There is no internal timer, display link, or frame
scheduler, so the loop may run at **any cadence**: seconds between frames (an
idle UI waiting on a dirty flag), a burst at input rate, or a steady animation
clock. An idle surface holds no acquired frame and costs zero GPU or CPU work;
nothing leaks or stalls across idle gaps. The only back-pressure is
`acquire()` itself: when every swapchain image is still in flight, it blocks
briefly, throttling a burst to the present rate.

`examples/native_window.rs` demonstrates all three cadences through one loop
on a real window (bare Cocoa FFI, no windowing crates):

```text
cargo run --example native_window
phase 1: sparse — 4 frames, 500 ms apart
phase 2: burst — 120 frames, no sleep
  120 frames in 1.96s = 61 fps (acquire back-pressure)
phase 3: animated — 3 s (pass --stay to keep it open)
```

`examples/native_window_x11.rs` is the Linux twin — the same three-phase
loop on the Vulkan swapchain (`SurfaceTarget::Xlib`) over a bare
Xlib window, clear-only so it needs no compiler or LLVM toolchain:

```text
cargo run --example native_window_x11 --no-default-features --features vulkan,render
```

### Present modes

| Mode                     | Behavior                                              |
|--------------------------|-------------------------------------------------------|
| `PresentMode::Fifo`      | Vsync; never tears; always supported. The default.   |
| `PresentMode::Immediate` | Present ASAP, may tear; lowest latency.              |
| `PresentMode::Mailbox`   | Triple-buffered: low latency without tearing.        |

Backends without `Immediate`/`Mailbox` reject at create/configure time with
`NotSupported`.

On Vulkan, a swapchain the driver reports as *suboptimal* (a resize the
window system tolerated) self-heals: the frame completes normally and the
swapchain is rebuilt on the next `acquire`, adopting the surface's real
extent — no error surfaces and no platform resize event is required. A
hard `VK_ERROR_OUT_OF_DATE_KHR` still reports `SurfaceOutdated`.

## Compositor owns present: `native_handle`

When another process or runtime composites the final image, render to an
ordinary texture and export the backend-native object behind it:

```rust
let target = gpu.render_target(1920, 1080, Format::BGRA8)?;
// ... render, then wait for the GPU work to finish:
gpu.render(&target)?.clear(Color::BLACK).draw(3).pulse()?.wait()?;

match target.native_handle()? {
    NativeTextureHandle::Metal { texture } => {
        // raw id<MTLTexture> — bind, blit, or retain it natively
    }
    NativeTextureHandle::Vulkan { image, memory, vk_format, layout } => {
        // raw VkImage + backing memory; transition from exactly `layout`
    }
    _ => { /* new variants may be added — always keep a wildcard arm */ }
}
```

The exported handle is a **borrow**: it stays valid exactly as long as the
`Texture` (and the `Gpu` it came from) are alive, and ownership is not
transferred. An importer that needs the native object to outlive the
`Texture` must take its own reference through the native API (e.g. ObjC
`retain` on the `MTLTexture`) before the `Texture` drops. The GPU work that
produced the contents must be complete (`Pulse::wait`) — or ordered against
the importer's reads by native means — before the importer samples it.

## Backend matrix

| Backend | Surface present (`Surface`)        | Native-handle export               |
|---------|------------------------------------|------------------------------------|
| Metal   | ✅ `CAMetalLayer` drawables         | ✅ `id<MTLTexture>`                 |
| Vulkan  | ✅ `VkSwapchainKHR` (Headless via `VK_EXT_headless_surface`, X11 via `SurfaceTarget::Xlib`, Android via `SurfaceTarget::AndroidWindow` over an `ANativeWindow` that must outlive the surface, Windows via `SurfaceTarget::Win32` over an `HWND` and its `HINSTANCE` that must both outlive the surface; needs loader WSI support — query `supports_surface_present`) | ✅ `VkImage` + memory/format/layout |
| WebGPU  | `NotSupported` (reserved variant)  | `NotSupported` (reserved variant)  |
| CPU     | `NotSupported`                     | `NotSupported` (no native object)  |

## Next

- [Tessellation](tessellation.md) — back to the pipeline: subdivide patches on the GPU
- [Textures](textures.md) — the render-target substrate both models share

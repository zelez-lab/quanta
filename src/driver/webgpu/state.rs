//! Handle table for the WebGPU driver.
//!
//! Two layers of handles meet in this module:
//!
//! - **Quanta-API u64 handles.** What `Wave`, `Texture`, `Pipeline`,
//!   `Sampler`, etc. carry on the public API. Stable across drivers,
//!   never escapes into the FFI.
//! - **Quanta WebGPU ABI u32 handles.** What `ffi.rs` exchanges with
//!   `web/src/glue.ts`. Each `u32` is an index into the JS-side
//!   `HandleTable`. Owned and released through the FFI imports
//!   (`quanta_destroy_buffer`, `quanta_release`, …).
//!
//! Each `State` map below maps the first to the second (plus any
//! Rust-side metadata we want to keep cheap-to-fetch without crossing
//! the FFI). Releases must go through both maps to free the JS-side
//! object, otherwise the JS handle table grows unbounded.

use alloc::collections::BTreeMap;
use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};

pub(super) struct WaveEntry {
    pub pipeline: u32,
    pub _shader: u32,
    /// Workgroup size from the kernel — informational; passed back via
    /// the public `Wave` shape.
    #[allow(dead_code)]
    pub workgroup_size: [u32; 3],
    /// Bind-group-layout(0) captured at pipeline creation time so
    /// subsequent dispatches can build bind groups against it without
    /// recreating.
    pub layout: u32,
    /// Slot-indexed bindings written by Wave::bind. None = unbound.
    pub bindings: BTreeMap<u32, u64>,
}

pub(super) struct TextureEntry {
    pub texture: u32,
    pub view: u32,
    pub width: u32,
    pub height: u32,
    pub format: crate::Format,
    pub bytes_per_row: u32,
}

pub(super) struct PipelineEntry {
    pub pipeline: u32,
    /// Layout(0): vertex/fragment uniforms + textures.
    pub layout: u32,
}

/// Thin newtype that promises `Send + Sync` for handle tables.
///
/// SAFETY: this module is `cfg(target_arch = "wasm32")`-gated. wasm32
/// has no real threads — `Send + Sync` is purely a type-level
/// requirement imposed by the `GpuDevice: Send + Sync` trait bound.
pub(super) struct SendCell<T>(pub RefCell<T>);

unsafe impl<T> Send for SendCell<T> {}
unsafe impl<T> Sync for SendCell<T> {}

impl<T> SendCell<T> {
    pub fn new(value: T) -> Self {
        Self(RefCell::new(value))
    }
}

pub(super) struct State {
    next: AtomicU64,
    /// Quanta u64 → JS GPUBuffer handle.
    pub buffers: SendCell<BTreeMap<u64, u32>>,
    pub waves: SendCell<BTreeMap<u64, WaveEntry>>,
    pub textures: SendCell<BTreeMap<u64, TextureEntry>>,
    /// Quanta u64 → JS GPUSampler handle.
    pub samplers: SendCell<BTreeMap<u64, u32>>,
    pub pipelines: SendCell<BTreeMap<u64, PipelineEntry>>,
}

impl State {
    pub fn new() -> Self {
        Self {
            // Start at 1 — handle 0 is reserved as "unbound" by the
            // Wave bookkeeping in `src/api/wave.rs`.
            next: AtomicU64::new(1),
            buffers: SendCell::new(BTreeMap::new()),
            waves: SendCell::new(BTreeMap::new()),
            textures: SendCell::new(BTreeMap::new()),
            samplers: SendCell::new(BTreeMap::new()),
            pipelines: SendCell::new(BTreeMap::new()),
        }
    }

    pub fn alloc_handle(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

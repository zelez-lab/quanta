//! Handle table for the WebGPU driver.
//!
//! The trait surface (`GpuDevice`) uses `u64` opaque handles for buffers,
//! pipelines, and waves. The browser side hands us `JsValue`-shaped objects
//! (`GPUBuffer`, `GPUComputePipeline`, …) which can't fit into a `u64`. We
//! keep one map per resource kind and mint monotonically-increasing handles
//! from a single counter.

use alloc::collections::BTreeMap;
use core::cell::RefCell;
use core::sync::atomic::{AtomicU64, Ordering};

use super::ffi::{
    GpuBuffer, GpuComputePipeline, GpuRenderPipeline, GpuSampler, GpuShaderModule, GpuTexture,
};

pub(super) struct WaveEntry {
    pub pipeline: GpuComputePipeline,
    pub _shader: GpuShaderModule,
    /// Workgroup size from the kernel — kept here so the dispatch path can
    /// validate the requested grid size against the kernel's declared
    /// workgroup if needed. Currently informational.
    #[allow(dead_code)]
    pub workgroup_size: [u32; 3],
    /// Bind-group-layout(0) captured at pipeline creation time so subsequent
    /// dispatches can build bind groups against it without recreating.
    pub layout: wasm_bindgen::JsValue,
    /// Slot-indexed bindings written by Wave::bind. None = unbound.
    pub bindings: BTreeMap<u32, u64>,
}

pub(super) struct TextureEntry {
    pub texture: GpuTexture,
    pub view: super::ffi::GpuTextureView,
    pub width: u32,
    pub height: u32,
    pub format: crate::Format,
    pub bytes_per_row: u32,
}

pub(super) struct PipelineEntry {
    pub pipeline: GpuRenderPipeline,
    /// Layout(0): vertex/fragment uniforms + textures.
    pub layout: wasm_bindgen::JsValue,
}

/// Thin newtype that promises `Send + Sync` for `JsValue`-shaped resources.
///
/// SAFETY: this module is `cfg(target_arch = "wasm32")`-gated. wasm32 has
/// no real threads — `Send + Sync` is purely a type-level requirement
/// imposed by the `GpuDevice: Send + Sync` trait bound.
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
    pub buffers: SendCell<BTreeMap<u64, GpuBuffer>>,
    pub waves: SendCell<BTreeMap<u64, WaveEntry>>,
    pub textures: SendCell<BTreeMap<u64, TextureEntry>>,
    pub samplers: SendCell<BTreeMap<u64, GpuSampler>>,
    pub pipelines: SendCell<BTreeMap<u64, PipelineEntry>>,
}

impl State {
    pub fn new() -> Self {
        Self {
            // Start at 1 — handle 0 is reserved as "unbound" by the Wave
            // bookkeeping in `src/api/wave.rs`.
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

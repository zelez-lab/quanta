//! Handle table for the WebGPU driver.
//!
//! Two layers of handles meet in this module:
//!
//! - **Quanta-API u64 handles.** What `Wave`, `Texture`, `Pipeline`,
//!   `Sampler`, etc. carry on the public API. Stable across drivers,
//!   never escapes into the FFI.
//! - **Quanta WebGPU ABI u32 handles.** What `ffi.rs` exchanges with
//!   `web/src/quanta.ts`. Each `u32` is an index into the JS-side
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
    /// Default view — used by the render-gated attachment path (step 085).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub view: u32,
    pub width: u32,
    pub height: u32,
    pub format: crate::Format,
    pub bytes_per_row: u32,
}

// Render-only: populated by the render-gated pipeline path (step 085).
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(super) struct PipelineEntry {
    pub pipeline: u32,
    /// Layout(0): vertex/fragment uniforms + textures.
    pub layout: u32,
}

/// One recorded ICB command. Compute = Dispatch; render = Draw.
/// Mirrors the Lean `Quanta.Icb.Command` sum type.
pub(super) enum WebgpuIcbCommand {
    Dispatch {
        wave_handle: u64,
        bindings: [u64; crate::api::wave::MAX_BINDINGS],
        binding_count: u8,
        workgroup_size: [u32; 3],
        groups: [u32; 3],
    },
    /// Render-path draw — recording shape only. WebGPU's native
    /// lowering is `GPURenderBundle`, which records into a render
    /// pass context; that wiring is a future commit.
    Draw {
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    },
}

/// One Indirect Command Buffer for the WebGPU driver.
///
/// W3C WebGPU has GPURenderBundle for the render path but does not
/// expose compute bundles, so a "true" native ICB lowering is not
/// available. This refinement records dispatches as snapshots and
/// replays them via the existing wave_dispatch path at execute
/// time — directly satisfying the Lean T7000 equivalence theorem
/// that the typed API contract is parametric in.
pub(super) struct WebgpuIcb {
    pub cap: u32,
    pub commands: alloc::vec::Vec<WebgpuIcbCommand>,
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
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub pipelines: SendCell<BTreeMap<u64, PipelineEntry>>,
    /// Indirect Command Buffers (steps 032 + 033). See `WebgpuIcb`.
    pub icbs: SendCell<BTreeMap<u64, WebgpuIcb>>,
    /// Render bundles (steps 032 + 033, render path). Each holds a
    /// JS-side `GPURenderBundleEncoder` while recording, then a
    /// `GPURenderBundle` after `finish()`.
    pub render_bundles: SendCell<BTreeMap<u64, WebgpuRenderBundle>>,
    /// Bindless texture arrays (steps 034 + 035). WebGPU has no
    /// native bindless; this maintains a software table of texture
    /// handles. Shaders that want to index into the array must do
    /// so via host-side rebinding before each dispatch.
    pub bindless_textures: SendCell<BTreeMap<u64, WebgpuBindlessArray>>,
    pub bindless_buffers: SendCell<BTreeMap<u64, WebgpuBindlessArray>>,
    /// Occlusion query sets (post-step-063 closure).
    /// Quanta u64 handle → (JS GPUQuerySet handle, slot count).
    pub query_sets: SendCell<BTreeMap<u64, (u32, u32)>>,
}

pub(super) struct WebgpuBindlessArray {
    pub cap: u32,
    pub entries: alloc::vec::Vec<u64>,
}

/// State for one WebGPU render bundle.
///
/// W3C `GPURenderBundleEncoder` requires a matching color /
/// depth format at create time, but our typed API records before
/// any render pass is active. Resolution: store snapshots and
/// build the JS-side encoder + bundle lazily inside
/// `RenderPass::execute_bundle` translation, when the render
/// target's format is known. This matches the proof contract
/// (record-then-execute = direct execution; the encoder shape
/// observable to the GPU is identical).
pub(super) struct WebgpuRenderBundle {
    pub cap: u32,
    pub draws: alloc::vec::Vec<RenderBundleDraw>,
}

/// One recorded draw inside a render bundle.
/// Render-only: read by the render-gated render-bundle path (step 085).
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(super) struct RenderBundleDraw {
    pub pipeline_handle: u64,
    pub vertex_count: u32,
    pub instance_count: u32,
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
            icbs: SendCell::new(BTreeMap::new()),
            render_bundles: SendCell::new(BTreeMap::new()),
            bindless_textures: SendCell::new(BTreeMap::new()),
            bindless_buffers: SendCell::new(BTreeMap::new()),
            query_sets: SendCell::new(BTreeMap::new()),
        }
    }

    pub fn alloc_handle(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed)
    }
}

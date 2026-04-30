//! MetalDevice struct definition and device discovery.

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{Caps, GpuDevice, Vendor};
use std::collections::HashMap;
use std::sync::RwLock;

use super::ffi;

/// State for one Metal MTLIndirectCommandBuffer.
pub(crate) struct MetalIcb {
    /// The MTLIndirectCommandBuffer object.
    pub(crate) icb: ffi::Id,
    /// Capacity (max command count) supplied at create time.
    pub(crate) cap: u32,
    /// Resource buffer handles touched by recorded commands. Used at
    /// execute time to call `useResource:usage:` so the GPU resource
    /// hazard tracker sees the dependencies (Metal does not infer
    /// these from the recorded ICB itself).
    pub(crate) used_buffers: Vec<u64>,
    /// Number of commands recorded so far. Always ≤ cap.
    pub(crate) recorded: u32,
}

/// State for one tessellation pipeline. Steps 022 + 023.
///
/// Metal has no fixed-function tessellator: factors are written into
/// an `MTLBuffer` by a compute kernel (or by the host, as we do
/// here), then bound to the render pipeline via
/// `setTessellationFactorBuffer:offset:instanceStride:` and consumed
/// by `drawIndexedPatches:`. The factor buffer here is the real
/// MTLBuffer a future render-pipeline integration will bind directly.
///
/// The buffer is laid out as `cap` × u32 outer slots followed by
/// `cap` × u32 inner slots. We keep u32 storage (matching the typed
/// API); a future commit converts to half-precision per Metal's
/// `MTL{Triangle,Quad}TessellationFactorsHalf` layout when the
/// drawIndexedPatches call site is wired.
pub(crate) struct MetalTessPipeline {
    /// Factor buffer (host-visible MTLBuffer, storageModeShared).
    pub(crate) factor_buf: ffi::Id,
    /// Number of outer factors (3 for triangle, 4 for quad).
    pub(crate) outer_count: u32,
    /// Number of inner factors (1 for triangle, 2 for quad).
    pub(crate) inner_count: u32,
}

/// State for one Metal mesh-shader pipeline. Steps 024 + 025.
///
/// Metal 3+ supports mesh shaders via
/// `MTLMeshRenderPipelineDescriptor` (object + mesh + fragment
/// functions) and `drawMeshThreadgroups:` on the render encoder.
/// MVP here is a software state container — limits + recorded
/// dispatch sequence — that satisfies the
/// `Quanta.MeshShader.Pipeline` contract today. The render-pipeline
/// integration (replacing the classical vertex stage with the
/// object/mesh path) lands when the render path is rebuilt to
/// support meshlets.
/// State for one Metal VRS handle. Steps 028 + 029.
///
/// Native lowering uses `MTLRasterizationRateMap` per render pass on
/// Apple Silicon. MVP here is a software state container — current
/// rate code, immutable on destroy. The native rate-map integration
/// lands when the render path is rebuilt to support per-tile rates.
#[allow(dead_code)]
pub(crate) struct MetalVrsState {
    pub(crate) rate_code: u8,
}

#[allow(dead_code)]
pub(crate) struct MetalMeshPipeline {
    pub(crate) max_vertices: u32,
    pub(crate) max_primitives: u32,
    pub(crate) task_threads: u32,
    pub(crate) dispatched: Vec<[u32; 3]>,
}

/// State for one Metal MTLIndirectCommandBuffer used as a *render*
/// bundle (DRAW command type instead of ConcurrentDispatch). Steps
/// 032 + 033, render path. Replayed from inside an active render
/// pass via `executeCommandsInBuffer:withRange:` on the render
/// encoder; see `RenderOp::ExecuteRenderBundle`.
pub(crate) struct MetalRenderBundle {
    /// The MTLIndirectCommandBuffer object (DRAW-typed).
    pub(crate) icb: ffi::Id,
    /// Capacity (max command count) supplied at create time.
    pub(crate) cap: u32,
    /// Number of draws recorded so far. Always ≤ cap.
    pub(crate) recorded: u32,
    /// Resource buffer handles touched by recorded draws — declared
    /// via `useResource:usage:` on the render encoder before
    /// execution.
    pub(crate) used_buffers: Vec<u64>,
}

/// Metal-backed GPU device.
pub struct MetalDevice {
    pub(crate) device: ffi::Id,
    pub(crate) queue: ffi::Id,
    pub(crate) caps: Caps,
    // Resource storage — keyed by handle.
    // RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    pub(crate) buffers: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) textures: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) compute_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) render_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) depth_stencil_states: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) samplers: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) queues: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) icbs: RwLock<HashMap<u64, MetalIcb>>,
    pub(crate) render_bundles: RwLock<HashMap<u64, MetalRenderBundle>>,
    pub(crate) tess_pipelines: RwLock<HashMap<u64, MetalTessPipeline>>,
    pub(crate) mesh_pipelines: RwLock<HashMap<u64, MetalMeshPipeline>>,
    pub(crate) vrs_states: RwLock<HashMap<u64, MetalVrsState>>,
    pub(crate) next_handle: AtomicU64,
}

// Safety: Metal objects (MTLDevice, MTLCommandQueue, etc.) are thread-safe.
// All mutable state is protected by RwLock.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

impl MetalDevice {
    pub(crate) fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Discover Metal devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let device = unsafe { ffi::MTLCreateSystemDefaultDevice() };
    if device.is_null() {
        return Vec::new();
    }

    let name = unsafe {
        let ns_name = ffi::msg_id(device, b"name\0");
        let cstr = ffi::msg_utf8_string(ns_name);
        std::ffi::CStr::from_ptr(cstr as *const _)
            .to_string_lossy()
            .into_owned()
    };

    let max_threads = unsafe { ffi::msg_mtlsize(device, b"maxThreadsPerThreadgroup\0") };
    let memory_bytes = unsafe { ffi::msg_u64(device, b"recommendedMaxWorkingSetSize\0") };

    let caps = Caps {
        nuclei: (max_threads.width as u32 / 32).max(1),
        protons_per_nucleus: 32,
        quarks_per_proton: 32,
        memory_bytes,
        max_quarks_per_dispatch: u32::MAX,
        max_groups: [u32::MAX, u32::MAX, u32::MAX],
        vendor: Vendor::Apple,
        name,
    };

    let queue = unsafe { ffi::msg_id(device, b"newCommandQueue\0") };

    vec![Box::new(MetalDevice {
        device,
        queue,
        caps,
        buffers: RwLock::new(HashMap::new()),
        textures: RwLock::new(HashMap::new()),
        compute_pipelines: RwLock::new(HashMap::new()),
        render_pipelines: RwLock::new(HashMap::new()),
        depth_stencil_states: RwLock::new(HashMap::new()),
        samplers: RwLock::new(HashMap::new()),
        queues: RwLock::new(HashMap::new()),
        icbs: RwLock::new(HashMap::new()),
        render_bundles: RwLock::new(HashMap::new()),
        tess_pipelines: RwLock::new(HashMap::new()),
        mesh_pipelines: RwLock::new(HashMap::new()),
        vrs_states: RwLock::new(HashMap::new()),
        next_handle: AtomicU64::new(0),
    })]
}

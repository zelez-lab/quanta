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
#[cfg_attr(not(feature = "compute"), allow(dead_code))]
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
// Render-path state; some fields are only read by the render-gated
// render_bundle paths (step 085).
#[cfg_attr(not(feature = "render"), allow(dead_code))]
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

/// Native Metal sparse texture state. Each entry binds a sparse
/// texture handle to its backing `MTLHeap` (placement type) and
/// the per-tile granularity Metal returns from
/// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:`.
/// `sparse_map_tile` / `sparse_unmap_tile` issue
/// `updateTextureMapping:mode:region:mipLevel:slice:` calls on a
/// resource state encoder against this heap. Step 063 follow-up.
pub(crate) struct MetalSparseTexture {
    /// Backing `MTLHeap` (kept alive — Metal heap-allocated
    /// textures borrow pages from this heap; releasing it would
    /// invalidate the texture). Unread today because Drop teardown
    /// for sparse textures isn't a separate path yet.
    #[allow(dead_code)]
    pub(crate) heap: ffi::Id,
    pub(crate) tile_w: u64,
    pub(crate) tile_h: u64,
}

/// State for one Metal presentation surface: a `CAMetalLayer` the
/// device presents drawables to. The layer is either caller-provided
/// (windowed — we retain it for the surface's lifetime) or created by
/// the driver (headless target). Present path: `nextDrawable` →
/// render into the drawable's texture via the ordinary render pass →
/// `presentDrawable:` on a fresh command buffer (queue order places
/// it after the submitted pass; no CPU stall).
// Only read by the render-gated surface path.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) struct MetalSurface {
    /// The `CAMetalLayer` (retained; released on surface destroy).
    pub(crate) layer: ffi::Id,
    /// Configured extent — checked against the layer's drawable size
    /// at acquire to surface `SurfaceOutdated`.
    pub(crate) width: u32,
    pub(crate) height: u32,
    /// Configured format (the layer's pixelFormat mirrors it).
    pub(crate) format: crate::Format,
}

/// One acquired, not-yet-presented drawable of a `MetalSurface`.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) struct MetalSurfaceFrame {
    /// The `CAMetalDrawable` (retained; released on present/discard).
    pub(crate) drawable: ffi::Id,
    /// Registry handle under which the drawable's texture was
    /// inserted into `MetalDevice::textures` (removed on
    /// present/discard).
    pub(crate) texture_handle: u64,
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
    // Only populated/read by the compute-gated dispatch path.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(crate) compute_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) render_pipelines: RwLock<HashMap<u64, ffi::Id>>,
    // Only populated/read by the render-gated pipeline path (step 085).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) depth_stencil_states: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) samplers: RwLock<HashMap<u64, ffi::Id>>,
    pub(crate) queues: RwLock<HashMap<u64, ffi::Id>>,
    // Compute ICBs — populated/read by the compute-gated ICB path.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(crate) icbs: RwLock<HashMap<u64, MetalIcb>>,
    pub(crate) render_bundles: RwLock<HashMap<u64, MetalRenderBundle>>,
    pub(crate) tess_pipelines: RwLock<HashMap<u64, MetalTessPipeline>>,
    pub(crate) mesh_pipelines: RwLock<HashMap<u64, MetalMeshPipeline>>,
    pub(crate) vrs_states: RwLock<HashMap<u64, MetalVrsState>>,
    /// Sparse-texture native state (handle → heap + tile size).
    pub(crate) sparse_textures: RwLock<HashMap<u64, MetalSparseTexture>>,
    /// Presentation surfaces (handle → CAMetalLayer state).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) surfaces: RwLock<HashMap<u64, MetalSurface>>,
    /// In-flight acquired surface frames (frame handle → drawable).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(crate) surface_frames: RwLock<HashMap<u64, MetalSurfaceFrame>>,
    pub(crate) next_handle: AtomicU64,
    /// Whether the device supports MTLSparseTexture
    /// (`supportsFamily:MTLGPUFamilyApple7` = 1007). Cached at
    /// discovery so `sparse_texture_create` doesn't dynamically
    /// query per request, and so a future native sparse-tile
    /// updateMappings path can gate uniformly.
    /// Step 063 slice 17 — symmetric to the Vulkan slice-16 cache.
    pub(crate) sparse_supported: bool,
    /// Whether the device supports tessellation
    /// (`supportsFamily:MTLGPUFamilyApple4` = 1004). Cached at
    /// discovery so `tessellation_pipeline_create` doesn't query
    /// per request — symmetric to the Vulkan tessellation_feature
    /// cache (slice 6).
    pub(crate) tessellation_supported: bool,
    /// Whether the device supports MTLMeshRenderPipelineDescriptor
    /// (`supportsFamily:MTLGPUFamilyMetal3` = 5001). Cached at
    /// discovery — symmetric to slice 9's per-call check.
    pub(crate) mesh_shader_supported: bool,
    /// Whether the device supports ray tracing
    /// (`supportsFamily:MTLGPUFamilyApple6` = 1006). Cached at
    /// discovery — symmetric to slice 10's per-call check.
    pub(crate) ray_tracing_supported: bool,
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

/// Create a Pulse backed by a dispatch_semaphore + addCompletedHandler.
/// The GPU signals the semaphore when the command buffer completes.
/// Pulse.wait() waits on the semaphore — no busy-polling, no thread parking.
///
/// Shared compute/render plumbing: compute dispatch, batch submit and
/// the render-pass end all wrap their command buffer in one of these
/// (which is why it lives here and not in the compute-gated module).
#[cfg(any(feature = "compute", feature = "render"))]
pub(crate) fn make_async_pulse(device: &MetalDevice, cmd: ffi::Id) -> crate::Pulse {
    unsafe {
        let sem = ffi::dispatch_semaphore_create(0);
        let block = ffi::make_completion_block(sem);
        ffi::msg_add_completed_handler(cmd, block);
        ffi::msg_void(cmd, b"commit\0");

        // libdispatch semaphores may be waited/released from any thread,
        // and the heap block is freed only after its handler ran — safe
        // to move the deferred wait onto Pulse::on_complete's waiter
        // thread.
        struct Waiter {
            sem: *mut core::ffi::c_void,
            block: *mut ffi::CompletionBlock,
        }
        unsafe impl Send for Waiter {}
        impl Waiter {
            // By-value method: the closure must capture the whole
            // (Send-asserted) struct, not its raw-pointer fields.
            fn take(self) -> (*mut core::ffi::c_void, *mut ffi::CompletionBlock) {
                (self.sem, self.block)
            }
        }
        let waiter = Waiter { sem, block };

        let handle = device.alloc_handle();
        crate::Pulse {
            handle,
            completed: false,
            wait_fn: Some(Box::new(move || {
                let (sem, block) = waiter.take();
                ffi::dispatch_semaphore_wait(sem, ffi::DISPATCH_TIME_FOREVER);
                ffi::dispatch_release(sem);
                // Free the heap-allocated block
                drop(Box::from_raw(block));
            })),
        }
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

    // Slice 17 — query Apple GPU family + Metal 3 support once at
    // discovery so per-call gates on sparse / tessellation / mesh /
    // ray-tracing don't re-issue Objective-C messages on every
    // create_*. Each `supports_family` value is a Metal-stable
    // GPU-family enum from Apple's MTLGPUFamily.
    let supports_family = |fam: u64| -> bool {
        unsafe {
            let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, u64) -> ffi::BOOL =
                core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
            f(device, ffi::sel(b"supportsFamily:\0"), fam) != 0
        }
    };
    let sparse_supported = supports_family(1007); // MTLGPUFamilyApple7
    let tessellation_supported = supports_family(1004); // MTLGPUFamilyApple4
    let mesh_shader_supported = supports_family(5001); // MTLGPUFamilyMetal3
    let ray_tracing_supported = supports_family(1006); // MTLGPUFamilyApple6

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
        sparse_textures: RwLock::new(HashMap::new()),
        surfaces: RwLock::new(HashMap::new()),
        surface_frames: RwLock::new(HashMap::new()),
        next_handle: AtomicU64::new(0),
        sparse_supported,
        tessellation_supported,
        mesh_shader_supported,
        ray_tracing_supported,
    })]
}

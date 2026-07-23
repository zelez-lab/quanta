//! CpuDevice — software GPU device implementation.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::{
    Caps, FieldUsage, GpuDevice, MemoryTopology, Pulse, QuantaError, Texture, TextureDesc, Vendor,
    Wave,
};
// Render types used only by the render-gated GpuDevice impl methods (085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};
use quanta_ir::{KernelDef, KernelOp};

use super::exec::{
    CoopGroup, ExecCtx, SUBGROUP_SIZE, SubgroupKind, SubgroupMode, execute_ops, resolve_warp,
    segment_has_barrier_loop, segment_has_subgroup,
};
use super::value::Value;

// ── CPU Device ───────────────────────────────────────────────────────────────

/// Internal buffer allocation. `Owned` is a driver allocation;
/// `Borrowed` is caller-owned memory imported zero-copy through
/// `field_import_host` — read-only by contract and never freed here
/// (dropping the entry releases the view, not the pages).
enum CpuBuffer {
    Owned { data: Vec<u8> },
    Borrowed { ptr: *const u8, len: usize },
}

// Safety: the `Borrowed` pointer is kept alive and immutable by the
// importing `HostField`'s borrow, and this backend is fully
// synchronous (a pulse completes inside dispatch), so every access
// happens while that borrow is live.
unsafe impl Send for CpuBuffer {}

impl CpuBuffer {
    fn bytes(&self) -> &[u8] {
        match self {
            CpuBuffer::Owned { data } => data,
            CpuBuffer::Borrowed { ptr, len } => unsafe { core::slice::from_raw_parts(*ptr, *len) },
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            CpuBuffer::Owned { data } => data,
            CpuBuffer::Borrowed { .. } => panic!(
                "write to a read-only host-imported field — bind \
                 HostField only to &[T] kernel parameters"
            ),
        }
    }
}

/// Texel metadata for a compute-bound texture. The pixel bytes live in
/// `buffers` under the same handle (textures are byte buffers); this records
/// the geometry/format the executor needs to index and decode a texel.
#[derive(Clone, Copy)]
struct CpuTextureMeta {
    width: u32,
    height: u32,
    format: crate::api::types::Format,
}

/// Stored kernel ready for dispatch.
struct CpuKernel {
    def: KernelDef,
    /// Pre-computed barrier segment ranges into def.body (start, end).
    /// Computed once at wave_jit time, reused every dispatch.
    segments: Vec<(usize, usize)>,
}

/// One recorded ICB command. Mirrors the Lean
/// `Quanta.Icb.Command` sum type (Dispatch | Draw); the CPU
/// device's `indirect_buffer_execute` folds over this list, which
/// is the direct refinement of T7000.
#[allow(clippy::large_enum_variant)]
enum RecordedCommand {
    Dispatch {
        wave_handle: u64,
        bindings: [u64; crate::api::types::MAX_BINDINGS],
        binding_count: u8,
        texture_bindings: [u64; crate::api::types::MAX_TEXTURES],
        texture_count: u8,
        push_data: [u8; crate::api::types::PUSH_DATA_CAP],
        push_len: u16,
        push_mask: u16,
        workgroup_size: [u32; 3],
        groups: [u32; 3],
    },
    /// Render-path draw command. The CPU device has no rasterizer —
    /// recording snapshots the parameters and `execute` replays
    /// them as no-ops. The proof contract (T7006: record extends
    /// the recorded sequence by exactly that command) is satisfied;
    /// observable rendering side-effects belong to GPU backends.
    Draw {
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    },
}

/// CPU-side ICB state — sized capacity + recorded commands.
struct CpuIcb {
    cap: u32,
    commands: Vec<RecordedCommand>,
}

/// CPU bindless texture/buffer array. Mirrors `Quanta.Bindless.Array`:
/// fixed capacity at create, slot updates via `set`, destroy
/// invalidates the handle. Texture and buffer arrays share the
/// same shape; backend differentiates by which trait method is called.
struct CpuBindlessArray {
    cap: u32,
    entries: Vec<u64>,
}

/// CPU tessellation pipeline state. Mirrors
/// `Quanta.Tessellation.Pipeline`: fixed topology and control-point
/// count at create, mutable inner / outer factor lists.
struct CpuTessPipeline {
    outer: Vec<u32>,
    inner: Vec<u32>,
}

/// CPU mesh-shader pipeline state. Mirrors
/// `Quanta.MeshShader.Pipeline`: bounded limits set at create + an
/// in-order dispatch history, both immutable on `destroy`.
struct CpuMeshPipeline {
    max_vertices: u32,
    max_primitives: u32,
    task_threads: u32,
    dispatched: Vec<[u32; 3]>,
}

/// CPU printf buffer state. Mirrors `Quanta.Printf.Buffer`:
/// capacity-bounded FIFO of message ids; record appends, drain
/// empties.
#[allow(dead_code)]
struct CpuPrintfBuffer {
    cap: u32,
    messages: Vec<u64>,
}

/// CPU async-copy queue state. Mirrors `Quanta.AsyncCopy.Queue`:
/// in-order list of (dst, src, size) triples.
#[allow(dead_code)]
struct CpuAsyncCopyQueue {
    submitted: Vec<(u64, u64, usize)>,
}

/// CPU queue state. Mirrors `Quanta.MultiQueue.Queue`: kind set at
/// create + an in-order submitted command counter + last_signal
/// pair. Software FIFO satisfies the contract; cross-queue ordering
/// is trivially serial on the CPU.
#[allow(dead_code)]
struct CpuQueue {
    kind: u8, // 0 = graphics, 1 = compute, 2 = transfer
    submit_count: u32,
    last_signal: Option<(u64, u64)>,
}

/// CPU sparse-texture state. Mirrors `Quanta.SparseTexture.Texture`:
/// dimensions captured at create + a tile-association map keyed by
/// (mip, x, y).
#[allow(dead_code)]
struct CpuSparseTexture {
    width: u32,
    height: u32,
    tiles: HashMap<(u32, u32, u32), u64>,
}

/// CPU VRS state. Mirrors `Quanta.Vrs.State`: a single rate code
/// (0 = 1×1, …, 6 = 4×4) writable until destroy.
#[allow(dead_code)]
struct CpuVrsState {
    rate_code: u8,
}

/// CPU acceleration-structure state. Mirrors
/// `Quanta.RayTracing.AccelerationStructure`: kind + geometry count
/// captured at build, immutable until `destroy`.
#[allow(dead_code)]
struct CpuAccelStructure {
    kind: u8, // 0 = bottom, 1 = top
    geom_count: u32,
}

/// CPU ray-tracing pipeline state. Mirrors
/// `Quanta.RayTracing.Pipeline`: max recursion depth set at create +
/// an in-order dispatch history.
#[allow(dead_code)]
struct CpuRtPipeline {
    max_recursion: u32,
    dispatched: Vec<(u32, u32)>,
}

/// CPU software device — executes GPU kernel IR without hardware.
pub struct CpuDevice {
    caps: Caps,
    next_handle: Mutex<u64>,
    buffers: Mutex<HashMap<u64, CpuBuffer>>,
    /// Geometry/format for each texture handle (pixel bytes stay in `buffers`).
    texture_meta: Mutex<HashMap<u64, CpuTextureMeta>>,
    kernels: Mutex<HashMap<u64, CpuKernel>>,
    /// Indirect command buffers indexed by handle.
    icbs: Mutex<HashMap<u64, CpuIcb>>,
    /// Bindless texture arrays indexed by handle.
    bindless_textures: Mutex<HashMap<u64, CpuBindlessArray>>,
    /// Bindless buffer arrays indexed by handle.
    bindless_buffers: Mutex<HashMap<u64, CpuBindlessArray>>,
    /// Tessellation pipelines indexed by handle.
    tess_pipelines: Mutex<HashMap<u64, CpuTessPipeline>>,
    /// Mesh-shader pipelines indexed by handle.
    mesh_pipelines: Mutex<HashMap<u64, CpuMeshPipeline>>,
    /// Ray-tracing acceleration structures indexed by handle.
    accel_structures: Mutex<HashMap<u64, CpuAccelStructure>>,
    /// Ray-tracing pipelines indexed by handle.
    rt_pipelines: Mutex<HashMap<u64, CpuRtPipeline>>,
    /// VRS states indexed by handle.
    vrs_states: Mutex<HashMap<u64, CpuVrsState>>,
    /// Sparse textures indexed by handle.
    sparse_textures: Mutex<HashMap<u64, CpuSparseTexture>>,
    /// Queues indexed by handle.
    queues: Mutex<HashMap<u64, CpuQueue>>,
    /// Async-copy queues indexed by handle.
    async_copy_queues: Mutex<HashMap<u64, CpuAsyncCopyQueue>>,
    /// Printf buffers indexed by handle.
    printf_buffers: Mutex<HashMap<u64, CpuPrintfBuffer>>,
}

impl CpuDevice {
    /// Create a new CPU software device.
    pub fn new() -> Self {
        Self {
            caps: Caps {
                nuclei: 1,
                protons_per_nucleus: 1,
                quarks_per_proton: 1,
                memory_bytes: 1024 * 1024 * 1024, // 1 GB virtual
                max_quarks_per_dispatch: u32::MAX,
                max_groups: [u32::MAX; 3],
                vendor: Vendor::Software,
                name: String::from("Quanta CPU (software)"),
                memory_topology: MemoryTopology::Unified,
            },
            next_handle: Mutex::new(1),
            buffers: Mutex::new(HashMap::new()),
            texture_meta: Mutex::new(HashMap::new()),
            kernels: Mutex::new(HashMap::new()),
            icbs: Mutex::new(HashMap::new()),
            bindless_textures: Mutex::new(HashMap::new()),
            bindless_buffers: Mutex::new(HashMap::new()),
            tess_pipelines: Mutex::new(HashMap::new()),
            mesh_pipelines: Mutex::new(HashMap::new()),
            accel_structures: Mutex::new(HashMap::new()),
            rt_pipelines: Mutex::new(HashMap::new()),
            vrs_states: Mutex::new(HashMap::new()),
            sparse_textures: Mutex::new(HashMap::new()),
            queues: Mutex::new(HashMap::new()),
            async_copy_queues: Mutex::new(HashMap::new()),
            printf_buffers: Mutex::new(HashMap::new()),
        }
    }

    fn alloc_handle(&self) -> u64 {
        let mut h = self.next_handle.lock().unwrap();
        let handle = *h;
        *h += 1;
        handle
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::api::device::sealed::Sealed for CpuDevice {}

impl GpuDevice for CpuDevice {
    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn supports_f64(&self) -> bool {
        // The software interpreter computes f64 ops natively.
        true
    }

    fn supports_i64(&self) -> bool {
        // The software interpreter computes i64/u64 ops natively.
        true
    }

    fn supports_subgroups(&self) -> bool {
        // The software interpreter resolves subgroup reduce/scan ops
        // warp-cooperatively (SubgroupMode Collect + Resolve passes).
        true
    }

    // === Fields ===

    fn field_alloc(&self, size: usize, _usage: FieldUsage) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer::Owned {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        Ok(handle)
    }

    fn field_import_host(&self, ptr: *const u8, len: usize) -> Result<u64, QuantaError> {
        if ptr.is_null() || len == 0 {
            return Err(QuantaError::invalid_param(
                "host import requires a non-null pointer and non-zero length",
            ));
        }
        let handle = self.alloc_handle();
        self.buffers
            .lock()
            .unwrap()
            .insert(handle, CpuBuffer::Borrowed { ptr, len });
        Ok(handle)
    }

    fn supports_host_import(&self) -> bool {
        true
    }

    fn host_import_alignment(&self) -> Option<usize> {
        // Pointer passthrough has no hardware granularity.
        Some(1)
    }

    fn field_free(&self, handle: u64) {
        self.buffers.lock().unwrap().remove(&handle);
    }

    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("field handle not found"))?;
        let dst = buf.bytes_mut();
        let len = data.len().min(dst.len());
        dst[..len].copy_from_slice(&data[..len]);
        Ok(())
    }

    fn field_write_bytes_at(
        &self,
        handle: u64,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("field handle not found"))?;
        let dst = buf.bytes_mut();
        if byte_offset >= dst.len() {
            return Ok(());
        }
        let len = data.len().min(dst.len() - byte_offset);
        dst[byte_offset..byte_offset + len].copy_from_slice(&data[..len]);
        Ok(())
    }

    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError> {
        let bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get(&handle)
            .ok_or_else(|| QuantaError::not_found("field handle not found"))?;
        let src = buf.bytes();
        let len = size.min(src.len());
        Ok(src[..len].to_vec())
    }

    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        // Copy src data first to avoid borrow conflict
        let src_data = {
            let src_buf = bufs
                .get(&src)
                .ok_or_else(|| QuantaError::not_found("src field not found"))?;
            let src = src_buf.bytes();
            let len = size.min(src.len());
            src[..len].to_vec()
        };
        let dst_buf = bufs
            .get_mut(&dst)
            .ok_or_else(|| QuantaError::not_found("dst field not found"))?;
        let dst_bytes = dst_buf.bytes_mut();
        let len = src_data.len().min(dst_bytes.len());
        dst_bytes[..len].copy_from_slice(&src_data[..len]);
        Ok(())
    }

    fn field_map(&self, handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        let mut bufs = self.buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("field handle not found"))?;
        Ok(buf.bytes_mut().as_mut_ptr())
    }

    fn field_unmap(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(()) // CPU memory is always mapped
    }

    fn field_create_mapped(
        &self,
        size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        let handle = self.alloc_handle();
        let buf = CpuBuffer::Owned {
            data: vec![0u8; size],
        };
        self.buffers.lock().unwrap().insert(handle, buf);
        let ptr = self
            .buffers
            .lock()
            .unwrap()
            .get_mut(&handle)
            .unwrap()
            .bytes_mut()
            .as_mut_ptr();
        Ok((handle, ptr))
    }

    // === Textures (minimal stubs) ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let handle = self.alloc_handle();
        let size = (desc.width * desc.height) as usize * desc.format.bytes_per_pixel();
        self.buffers.lock().unwrap().insert(
            handle,
            CpuBuffer::Owned {
                data: vec![0u8; size],
            },
        );
        self.texture_meta.lock().unwrap().insert(
            handle,
            CpuTextureMeta {
                width: desc.width,
                height: desc.height,
                format: desc.format,
            },
        );
        Ok(Texture {
            handle,
            width: desc.width,
            height: desc.height,
            format: desc.format,
            sample_count: desc.sample_count,
            device: None,
            live: true,
        })
    }

    fn texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        // CPU textures are plain byte buffers in the buffer registry.
        self.buffers.lock().unwrap().remove(&handle);
        self.texture_meta.lock().unwrap().remove(&handle);
        Ok(())
    }

    fn debug_registry_counts(&self) -> crate::RegistryCounts {
        crate::RegistryCounts {
            buffers: self.buffers.lock().unwrap().len(),
            waves: self.kernels.lock().unwrap().len(),
            ..Default::default()
        }
    }

    // === Compute-resource lifecycle ===

    /// Destroy a wave: drop its deserialized kernel definition.
    #[cfg(feature = "compute")]
    fn wave_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.kernels.lock().unwrap().remove(&handle);
        Ok(())
    }

    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError> {
        self.field_write_bytes(texture.handle(), data)
    }

    fn supports_texture_write_region(&self) -> bool {
        true
    }

    fn supports_compute_textures(&self) -> bool {
        true
    }

    fn texture_write_region(
        &self,
        texture: &Texture,
        origin: (u32, u32),
        size: (u32, u32),
        data: &[u8],
    ) -> Result<(), QuantaError> {
        // The texture is field-backed: copy the region row by row into
        // the tightly packed backing store.
        let bpp = texture.format().bytes_per_pixel();
        let row_bytes = size.0 as usize * bpp;
        for row in 0..size.1 as usize {
            let src = &data[row * row_bytes..(row + 1) * row_bytes];
            let dst =
                ((origin.1 as usize + row) * texture.width() as usize + origin.0 as usize) * bpp;
            self.field_write_bytes_at(texture.handle(), dst, src)?;
        }
        Ok(())
    }

    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        let size =
            (texture.width() * texture.height()) as usize * texture.format().bytes_per_pixel();
        self.field_read_bytes(texture.handle(), size)
    }

    fn sampler_create(
        &self,
        _desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        // No CPU-side sampler state — the handle is a pure token, so
        // there is nothing to destroy (sampler_destroy default no-op).
        Ok(crate::Sampler {
            handle: self.alloc_handle(),
            device: None,
            live: true,
        })
    }

    fn generate_mipmaps(&self, _texture: &Texture) -> Result<(), QuantaError> {
        Ok(()) // no-op on CPU
    }

    // === Compute ===

    fn wave(&self, _kernel: &[u8]) -> Result<Wave, QuantaError> {
        Err(QuantaError::invalid_param(
            "CPU device only supports JIT path (wave_jit). \
             Pre-compiled binaries cannot be executed on CPU.",
        ))
    }

    fn wave_jit(&self, kernel_def_bytes: &[u8]) -> Result<Wave, QuantaError> {
        let mut def = quanta_ir::deserialize_kernel(kernel_def_bytes)
            .map_err(|e| QuantaError::compilation_failed(e.to_string()))?;
        // The texture-access rejections the GPU emitters run. The CPU
        // interpreter would happily treat a sample as a texel load (they
        // coincide under the fixed NEAREST/CLAMP contract) and write through
        // a read-only slot, so reject here to keep all backends agreeing.
        for check in [
            quanta_ir::types::reject_sample_on_storage,
            quanta_ir::types::reject_write_on_read_only,
            quanta_ir::types::reject_sampled_u32_texture,
        ] {
            check(&def).map_err(QuantaError::compilation_failed)?;
        }
        // Hoist barriers nested in (uniform) control flow up to the top
        // level so the cooperative segmenter sees them. A barrier under a
        // divergent branch is UB on real GPUs, so the only barriers we
        // encounter inside branches are uniform — the inliner's structural
        // wrappers — and lifting them preserves semantics (the guard is
        // duplicated onto both halves). See `hoist_barriers`.
        def.body = hoist_barriers(core::mem::take(&mut def.body));
        let handle = self.alloc_handle();
        let workgroup_size = def.workgroup_size;
        let segments = barrier_segment_ranges(&def.body);
        self.kernels
            .lock()
            .unwrap()
            .insert(handle, CpuKernel { def, segments });
        Ok(Wave {
            handle,
            bindings: [0u64; 16],
            binding_count: 0,
            texture_bindings: [0u64; 16],
            texture_count: 0,
            storage_texture_kinds: [0; 16],
            push_data: [0u8; 256],
            push_len: 0,
            push_mask: 0,
            workgroup_size,
            device: None,
            live: true,
        })
    }

    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError> {
        let kernels = self.kernels.lock().unwrap();
        let kernel = kernels
            .get(&wave.handle)
            .ok_or_else(|| QuantaError::not_found("wave handle not found"))?;

        let wg = kernel.def.workgroup_size;
        let total_groups = groups[0] as u64 * groups[1] as u64 * groups[2] as u64;
        let threads_per_group = wg[0] as u64 * wg[1] as u64 * wg[2] as u64;
        let total_threads = total_groups * threads_per_group;

        // Snapshot bound buffer data into per-slot Mutex<Vec<u8>>. The
        // Mutex gives each `Load`/`Store`/`AtomicOp` a per-field
        // critical section; the parallel-group dispatch below shares
        // `&field_data` across worker threads so concurrent groups
        // serialise only at field touches (compute parallelises freely).
        let mut field_data: [Option<Mutex<Vec<u8>>>; 16] = Default::default();
        {
            let bufs = self.buffers.lock().unwrap();
            for (i, slot) in field_data
                .iter_mut()
                .enumerate()
                .take(wave.binding_count as usize)
            {
                let handle = wave.bindings[i];
                if handle != 0
                    && let Some(buf) = bufs.get(&handle)
                {
                    *slot = Some(Mutex::new(buf.bytes().to_vec()));
                }
            }
        }

        // Snapshot bound textures the same way: pixel bytes into a per-slot
        // Mutex, plus the geometry/format the executor needs to index a texel.
        // The scalar-driven format contract is validated up front per slot kind
        // (1 = R32Float, 2 = RGBA8-unorm packed-u32); a texture whose format
        // doesn't match its storage-slot kind is InvalidParam, matching
        // Metal/Vulkan.
        use crate::api::types::Format;
        let mut tex_data: [Option<super::exec::CpuTexSlot>; 16] = Default::default();
        {
            let bufs = self.buffers.lock().unwrap();
            let metas = self.texture_meta.lock().unwrap();
            for (i, slot) in tex_data
                .iter_mut()
                .enumerate()
                .take(wave.texture_count as usize)
            {
                let handle = wave.texture_bindings[i];
                if handle == 0 {
                    continue;
                }
                let (Some(buf), Some(meta)) = (bufs.get(&handle), metas.get(&handle)) else {
                    continue;
                };
                let (expected, ok) = match wave.storage_texture_kinds[i] {
                    1 | 3 => ("R32Float", meta.format == Format::R32Float),
                    2 | 4 => ("RGBA8", meta.format == Format::RGBA8),
                    _ => ("", true), // not a texel slot — no format contract
                };
                if !ok {
                    return Err(QuantaError::invalid_param(
                        "compute storage texture format mismatch",
                    )
                    .with_context(&format!(
                        "slot {i}: expected {expected}, got {:?}",
                        meta.format
                    )));
                }
                *slot = Some(super::exec::CpuTexSlot {
                    data: Mutex::new(buf.bytes().to_vec()),
                    width: meta.width,
                    height: meta.height,
                    format: meta.format,
                });
            }
        }

        let group_size_x = wg[0];

        // Use pre-computed barrier segment ranges (zero-copy slices)
        let segments = &kernel.segments;
        let body = &kernel.def.body;
        let push_data = &wave.push_data;

        // Parallel-group dispatch. Split `0..total_groups` across
        // `available_parallelism()` worker threads via `thread::scope`
        // so that compute-heavy kernels (PTRD-style) scale with cores.
        // Each worker owns its own per-group `shared` HashMap and
        // per-quark `thread_regs`; field reads/writes serialise
        // through the per-slot `Mutex` in `field_data`. Atomics get
        // cross-group atomicity for free (the lock spans the
        // read-modify-write).
        //
        // For tiny dispatches (< THREAD_THRESHOLD groups) the
        // thread-spawn cost outweighs the parallelism gain, so we
        // fall back to a single worker.
        const THREAD_THRESHOLD: u64 = 4;
        let worker_count = if total_groups < THREAD_THRESHOLD {
            1
        } else {
            std::thread::available_parallelism()
                .map(|n| n.get() as u64)
                .unwrap_or(1)
                .min(total_groups)
                .max(1)
        };

        let first_err: Mutex<Option<String>> = Mutex::new(None);

        std::thread::scope(|scope| {
            for worker_idx in 0..worker_count {
                // Chunked: worker `w` handles groups
                // `[w*total/N .. (w+1)*total/N)` (last worker mops up
                // the remainder).
                let start = worker_idx * total_groups / worker_count;
                let end = if worker_idx + 1 == worker_count {
                    total_groups
                } else {
                    (worker_idx + 1) * total_groups / worker_count
                };
                let field_data = &field_data;
                let tex_data = &tex_data;
                let first_err = &first_err;
                scope.spawn(move || {
                    let mut shared: HashMap<u32, Vec<u8>> = HashMap::new();
                    let mut thread_regs: Vec<HashMap<u32, Value>> =
                        (0..threads_per_group).map(|_| HashMap::new()).collect();
                    for gid in start..end {
                        // Early-out if another worker already failed.
                        if first_err.lock().unwrap().is_some() {
                            return;
                        }
                        shared.clear();
                        for reg_map in thread_regs.iter_mut() {
                            reg_map.clear();
                        }
                        for &(seg_start, seg_end) in segments {
                            let segment = &body[seg_start..seg_end];
                            let tpg = threads_per_group as u32;

                            // Subgroup reduce/scan ops need all warp lanes'
                            // inputs at once, but lanes run sequentially. For
                            // segments that use them, resolve cooperatively
                            // per warp via a side-effect-free Collect dry run
                            // + a real Resolve pass (see SubgroupMode). Other
                            // segments take the plain single-pass loop.
                            let run_lane = |lid: u32,
                                            regs: HashMap<u32, Value>,
                                            shared: &mut HashMap<u32, Vec<u8>>,
                                            mode: SubgroupMode|
                             -> Result<HashMap<u32, Value>, String> {
                                let mut ctx = ExecCtx {
                                    quark_id: (gid * threads_per_group + lid as u64) as u32,
                                    local_id: lid,
                                    group_id: gid as u32,
                                    group_size: group_size_x,
                                    quark_count: total_threads as u32,
                                    regs,
                                    fields: field_data,
                                    textures: tex_data,
                                    shared,
                                    push_data,
                                    subgroup: mode,
                                };
                                execute_ops(&mut ctx, segment)?;
                                Ok(ctx.regs)
                            };

                            let report = |e: String, quark_id: u32| {
                                let mut slot = first_err.lock().unwrap();
                                if slot.is_none() {
                                    *slot = Some(format!(
                                        "CPU execution error (quark {quark_id}): {e}"
                                    ));
                                }
                            };

                            if segment_has_barrier_loop(segment) && !segment_has_subgroup(segment) {
                                // A barrier inside a loop is a cross-lane sync
                                // point; run the loop iteration-synchronized
                                // across all lanes so in-loop shared writes are
                                // visible (the top-level segmenter only splits
                                // at top-level barriers). Segments that ALSO use
                                // subgroup ops keep the warp-cooperative path
                                // below (the prims block kernels put their
                                // barriers at the top level, not inside loops,
                                // so they never need both).
                                let coop = CoopGroup {
                                    gid,
                                    threads_per_group,
                                    group_size: group_size_x,
                                    quark_count: total_threads as u32,
                                    fields: field_data,
                                    textures: tex_data,
                                    push_data,
                                };
                                if let Err(e) =
                                    coop.run_segment(segment, &mut thread_regs, &mut shared)
                                {
                                    report(e, (gid * threads_per_group) as u32);
                                    return;
                                }
                            } else if segment_has_subgroup(segment) {
                                // Warp-cooperative path.
                                let mut lid = 0u32;
                                while lid < tpg {
                                    let warp_end = (lid + SUBGROUP_SIZE).min(tpg);
                                    let warp: Vec<u32> = (lid..warp_end).collect();

                                    // Pass 1 — Collect (dry run; writes off).
                                    let mut cohort: Vec<Vec<(SubgroupKind, Value)>> =
                                        Vec::with_capacity(warp.len());
                                    for &wl in &warp {
                                        let regs = thread_regs[wl as usize].clone();
                                        let mut inputs = Vec::new();
                                        let r = {
                                            let mut sh = shared.clone();
                                            run_lane(
                                                wl,
                                                regs,
                                                &mut sh,
                                                SubgroupMode::Collect {
                                                    inputs: &mut inputs,
                                                },
                                            )
                                        };
                                        if let Err(e) = r {
                                            report(e, (gid * threads_per_group + wl as u64) as u32);
                                            return;
                                        }
                                        cohort.push(inputs);
                                    }

                                    // Reduce per site across the warp.
                                    let resolved = resolve_warp(&cohort);

                                    // Pass 2 — Resolve (real writes).
                                    for (slot_i, &wl) in warp.iter().enumerate() {
                                        let regs = core::mem::take(&mut thread_regs[wl as usize]);
                                        let res = run_lane(
                                            wl,
                                            regs,
                                            &mut shared,
                                            SubgroupMode::Resolve {
                                                resolved: &resolved[slot_i],
                                                cursor: 0,
                                            },
                                        );
                                        match res {
                                            Ok(regs) => thread_regs[wl as usize] = regs,
                                            Err(e) => {
                                                report(
                                                    e,
                                                    (gid * threads_per_group + wl as u64) as u32,
                                                );
                                                return;
                                            }
                                        }
                                    }
                                    lid = warp_end;
                                }
                            } else {
                                // Plain single-pass path (no subgroup ops).
                                for lid in 0..tpg {
                                    let regs = core::mem::take(&mut thread_regs[lid as usize]);
                                    match run_lane(lid, regs, &mut shared, SubgroupMode::None) {
                                        Ok(regs) => thread_regs[lid as usize] = regs,
                                        Err(e) => {
                                            report(
                                                e,
                                                (gid * threads_per_group + lid as u64) as u32,
                                            );
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                });
            }
        });

        if let Some(msg) = first_err.into_inner().unwrap() {
            return Err(QuantaError::compilation_failed(msg));
        }

        // Write back modified buffer data
        {
            let mut bufs = self.buffers.lock().unwrap();
            for (i, slot) in field_data
                .iter_mut()
                .enumerate()
                .take(wave.binding_count as usize)
            {
                let handle = wave.bindings[i];
                if handle != 0
                    && let Some(modified) = slot.take()
                    && let Some(buf) = bufs.get_mut(&handle)
                {
                    let modified = modified.into_inner().unwrap();
                    match buf {
                        CpuBuffer::Owned { data } => *data = modified,
                        // Host-imported fields are read-only by
                        // contract; the snapshot lets us detect a
                        // kernel that violated it instead of
                        // scribbling the caller's memory.
                        CpuBuffer::Borrowed { ptr, len } => {
                            let orig = unsafe { core::slice::from_raw_parts(*ptr, *len) };
                            if orig != modified.as_slice() {
                                return Err(QuantaError::invalid_param(
                                    "kernel wrote a read-only host-imported field",
                                )
                                .with_context(&format!(
                                    "binding slot {i}: bind HostField only to \
                                     &[T] kernel parameters"
                                )));
                            }
                        }
                    }
                }
            }
        }

        // Write back modified texture pixels (a TextureWrite2D kernel mutates
        // its snapshot; persist so texture.read() sees the result).
        {
            let mut bufs = self.buffers.lock().unwrap();
            for (i, slot) in tex_data
                .iter_mut()
                .enumerate()
                .take(wave.texture_count as usize)
            {
                let handle = wave.texture_bindings[i];
                if handle != 0
                    && let Some(tex) = slot.take()
                    && let Some(CpuBuffer::Owned { data }) = bufs.get_mut(&handle)
                {
                    *data = tex.data.into_inner().unwrap();
                }
            }
        }

        Ok(Pulse {
            handle: 0,
            completed: true,
            wait_fn: None,
            // Synchronous backend: the pulse carries no deferred device
            // work, so there is nothing a keep-alive would protect.
            keep_alive: None,
        })
    }

    fn wave_dispatch_indirect(
        &self,
        _wave: &Wave,
        _buffer: u64,
        _offset: u64,
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param(
            "indirect dispatch not supported on CPU device",
        ))
    }

    // === Render (stubs) === (render-gated, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError> {
        // Step 063 slice 12 — close the silent-drop on CPU for
        // symmetry with the Metal/Vulkan/WebGPU gates (slices 5
        // and 11). The CPU device has no rasterizer at all so any
        // PipelineDesc field that names a render-only feature
        // surfaces NotSupported here; render_begin returns
        // NotSupported regardless, but a user who constructs a
        // tessellation-using PipelineDesc deserves a feature-named
        // error rather than a successful create that fails later.
        if desc.tessellation.is_some() {
            return Err(QuantaError::not_supported(
                "CPU pipelines: tessellation is not supported (CPU has no rasterizer)",
            ));
        }
        if desc.mesh_shader.is_some() {
            return Err(QuantaError::not_supported(
                "CPU pipelines: mesh shaders are not supported (CPU has no rasterizer)",
            ));
        }
        if desc.conservative_rasterization {
            return Err(QuantaError::not_supported(
                "CPU pipelines: conservative rasterization is not supported (CPU has no rasterizer)",
            ));
        }
        // No CPU-side pipeline state — the handle is a pure token, so
        // there is nothing to destroy (pipeline_destroy default no-op).
        Ok(Pipeline::from_desc(self.alloc_handle(), desc))
    }

    #[cfg(feature = "render")]
    fn render_begin(&self, _target: &Texture) -> Result<RenderPass, QuantaError> {
        Err(QuantaError::not_supported(
            "render passes not supported on CPU device",
        ))
    }

    #[cfg(feature = "render")]
    fn render_end(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        Err(QuantaError::not_supported(
            "render passes not supported on CPU device",
        ))
    }

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError> {
        pulse.completed = true;
        Ok(())
    }

    fn pulse_poll(&self, _pulse: &Pulse) -> bool {
        true // CPU execution is synchronous
    }

    fn wait_idle(&self) -> Result<(), QuantaError> {
        Ok(()) // CPU execution is synchronous
    }

    // === M4.2: Mesh shaders ===

    fn dispatch_mesh(&self, _pipeline: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "mesh shaders not supported on CPU device",
        ))
    }

    // === M4.3: Ray tracing (steps 026 + 027) ===
    //
    // CPU implementation refines `Quanta.RayTracing` lifecycle:
    // build_acceleration_structure stores kind + geom_count; create_
    // ray_tracing_pipeline stores max_recursion + empty dispatch list;
    // dispatch_rays appends (width, height); destroy_* removes the
    // handle. Rasterization / intersection are not modeled — the
    // proven contract is lifecycle + dispatch ordering only.

    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError> {
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }
        let handle = self.alloc_handle();
        self.accel_structures.lock().unwrap().insert(
            handle,
            CpuAccelStructure {
                kind: 0,
                geom_count: geometry.len() as u32,
            },
        );
        Ok(handle)
    }

    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.rt_pipelines.lock().unwrap().insert(
            handle,
            CpuRtPipeline {
                max_recursion: desc.max_recursion,
                dispatched: Vec::new(),
            },
        );
        Ok(handle)
    }

    fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError> {
        let mut pipes = self.rt_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&pipeline)
            .ok_or_else(|| QuantaError::not_found("ray tracing pipeline not found"))?;
        pipe.dispatched.push((width, height));
        Ok(())
    }

    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        self.accel_structures.lock().unwrap().remove(&handle);
        Ok(())
    }

    fn destroy_ray_tracing_pipeline(&self, handle: u64) -> Result<(), QuantaError> {
        self.rt_pipelines.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === M5.1: Sparse textures (steps 030 + 031) ===
    //
    // CPU implementation refines `Quanta.SparseTexture.Texture`:
    // - sparse_texture_create stores width/height with an empty
    //   tile map.
    // - sparse_map_tile inserts (mip, x, y) -> backing into the map.
    // - sparse_unmap_tile removes (mip, x, y).
    // - sparse_texture_destroy removes the handle.

    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError> {
        if desc.width == 0 || desc.height == 0 {
            return Err(QuantaError::invalid_param(
                "sparse texture requires non-zero dimensions",
            ));
        }
        let handle = self.alloc_handle();
        self.sparse_textures.lock().unwrap().insert(
            handle,
            CpuSparseTexture {
                width: desc.width,
                height: desc.height,
                tiles: HashMap::new(),
            },
        );
        Ok(handle)
    }

    fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        backing: u64,
    ) -> Result<(), QuantaError> {
        let mut texs = self.sparse_textures.lock().unwrap();
        let tex = texs
            .get_mut(&texture)
            .ok_or_else(|| QuantaError::not_found("sparse texture not found"))?;
        tex.tiles.insert((mip, x, y), backing);
        Ok(())
    }

    fn sparse_unmap_tile(&self, texture: u64, mip: u32, x: u32, y: u32) -> Result<(), QuantaError> {
        let mut texs = self.sparse_textures.lock().unwrap();
        let tex = texs
            .get_mut(&texture)
            .ok_or_else(|| QuantaError::not_found("sparse texture not found"))?;
        tex.tiles.remove(&(mip, x, y));
        Ok(())
    }

    fn sparse_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.sparse_textures.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Multi-queue (steps 018 + 019) ===
    //
    // CPU implementation refines `Quanta.MultiQueue.Queue` as a
    // software FIFO. The CPU device executes everything serially so
    // queue ordering is trivially preserved; the contract bars
    // re-ordering, which is unconditionally satisfied here.

    fn queue_families(&self) -> Vec<crate::QueueFamily> {
        vec![
            crate::QueueFamily {
                queue_type: crate::QueueType::Graphics,
                count: 1,
            },
            crate::QueueFamily {
                queue_type: crate::QueueType::Compute,
                count: 1,
            },
            crate::QueueFamily {
                queue_type: crate::QueueType::Transfer,
                count: 1,
            },
        ]
    }

    fn create_queue(&self, queue_type: crate::QueueType) -> Result<u64, QuantaError> {
        let kind: u8 = match queue_type {
            crate::QueueType::Graphics => 0,
            crate::QueueType::Compute => 1,
            crate::QueueType::Transfer => 2,
        };
        let handle = self.alloc_handle();
        self.queues.lock().unwrap().insert(
            handle,
            CpuQueue {
                kind,
                submit_count: 0,
                last_signal: None,
            },
        );
        Ok(handle)
    }

    fn queue_dispatch(&self, queue: u64, wave: &Wave, groups: [u32; 3]) -> Result<(), QuantaError> {
        {
            let mut qs = self.queues.lock().unwrap();
            let q = qs
                .get_mut(&queue)
                .ok_or_else(|| QuantaError::not_found("queue not found"))?;
            q.submit_count += 1;
        }
        // Execute serially against the existing wave_dispatch path.
        let _ = self.wave_dispatch(wave, groups)?;
        Ok(())
    }

    fn queue_signal(&self, queue: u64, semaphore: u64) -> Result<(), QuantaError> {
        let mut qs = self.queues.lock().unwrap();
        let q = qs
            .get_mut(&queue)
            .ok_or_else(|| QuantaError::not_found("queue not found"))?;
        q.last_signal = Some((semaphore, q.submit_count as u64));
        Ok(())
    }

    fn queue_wait(&self, queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        let qs = self.queues.lock().unwrap();
        if !qs.contains_key(&queue) {
            return Err(QuantaError::not_found("queue not found"));
        }
        Ok(())
    }

    fn queue_destroy(&self, queue: u64) -> Result<(), QuantaError> {
        self.queues.lock().unwrap().remove(&queue);
        Ok(())
    }

    // === Async memory copy (step 044) ===
    //
    // CPU implementation refines `Quanta.AsyncCopy.Queue` as a
    // software FIFO. Submitted copies execute serially via the
    // existing field_copy_bytes path; the recorded sequence
    // satisfies T7801 directly.

    fn async_copy_create(&self) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.async_copy_queues.lock().unwrap().insert(
            handle,
            CpuAsyncCopyQueue {
                submitted: Vec::new(),
            },
        );
        Ok(handle)
    }

    fn async_copy_submit(
        &self,
        queue: u64,
        dst: u64,
        src: u64,
        size: usize,
    ) -> Result<(), QuantaError> {
        {
            let mut qs = self.async_copy_queues.lock().unwrap();
            let q = qs
                .get_mut(&queue)
                .ok_or_else(|| QuantaError::not_found("async copy queue not found"))?;
            q.submitted.push((dst, src, size));
        }
        // Execute the copy synchronously through the existing path.
        self.field_copy_bytes(dst, src, size)?;
        Ok(())
    }

    fn async_copy_destroy(&self, queue: u64) -> Result<(), QuantaError> {
        self.async_copy_queues.lock().unwrap().remove(&queue);
        Ok(())
    }

    // === GPU printf (step 049) ===
    //
    // CPU implementation refines `Quanta.Printf.Buffer`:
    // - printf_create allocates a Vec<u64> sized to cap.
    // - printf_record appends if not full.
    // - printf_drain returns the messages and empties the buffer.
    // - printf_destroy removes the handle.

    fn printf_create(&self, cap: u32) -> Result<u64, QuantaError> {
        if cap == 0 {
            return Err(QuantaError::invalid_param(
                "printf buffer capacity must be >= 1",
            ));
        }
        let handle = self.alloc_handle();
        self.printf_buffers.lock().unwrap().insert(
            handle,
            CpuPrintfBuffer {
                cap,
                messages: Vec::new(),
            },
        );
        Ok(handle)
    }

    fn printf_record(&self, handle: u64, msg_id: u64) -> Result<(), QuantaError> {
        let mut bufs = self.printf_buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("printf buffer not found"))?;
        if buf.messages.len() as u32 >= buf.cap {
            return Err(QuantaError::invalid_param("printf buffer is full"));
        }
        buf.messages.push(msg_id);
        Ok(())
    }

    fn printf_drain(&self, handle: u64) -> Result<Vec<u64>, QuantaError> {
        let mut bufs = self.printf_buffers.lock().unwrap();
        let buf = bufs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("printf buffer not found"))?;
        Ok(core::mem::take(&mut buf.messages))
    }

    fn printf_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.printf_buffers.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === M5.2: Indirect command buffers (steps 032 + 033) ===
    //
    // CPU implementation refines the abstract `Quanta.Icb` model:
    // - create allocates a fresh handle + empty Vec sized to capacity.
    // - icb_record_dispatch snapshots the wave + group counts.
    // - indirect_buffer_execute replays the first `count` recorded
    //   dispatches sequentially through the existing `wave_dispatch`
    //   path, satisfying the Lean T7000 equivalence theorem.

    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.icbs.lock().unwrap().insert(
            handle,
            CpuIcb {
                cap: max_commands,
                commands: Vec::with_capacity(max_commands as usize),
            },
        );
        Ok(handle)
    }

    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        let mut icbs = self.icbs.lock().unwrap();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        icb.commands.push(RecordedCommand::Dispatch {
            wave_handle: wave.handle,
            bindings: wave.bindings,
            binding_count: wave.binding_count,
            texture_bindings: wave.texture_bindings,
            texture_count: wave.texture_count,
            push_data: wave.push_data,
            push_len: wave.push_len,
            push_mask: wave.push_mask,
            workgroup_size: wave.workgroup_size,
            groups,
        });
        Ok(())
    }

    fn icb_record_draw(
        &self,
        handle: u64,
        index: u32,
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<(), QuantaError> {
        let mut icbs = self.icbs.lock().unwrap();
        let icb = icbs
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
        if index != icb.commands.len() as u32 {
            return Err(QuantaError::invalid_param(
                "ICB record index must equal current length",
            ));
        }
        if index >= icb.cap {
            return Err(QuantaError::invalid_param("ICB index >= capacity"));
        }
        icb.commands.push(RecordedCommand::Draw {
            pipeline,
            vertex_count,
            instance_count,
        });
        Ok(())
    }

    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError> {
        // Snapshot the recorded commands while holding the ICB lock,
        // then drop the lock before re-entering wave_dispatch (which
        // takes its own locks on buffers/kernels).
        let snapshot: Vec<RecordedCommand> = {
            let icbs = self.icbs.lock().unwrap();
            let icb = icbs
                .get(&handle)
                .ok_or_else(|| QuantaError::not_found("ICB handle not found"))?;
            if count as usize > icb.commands.len() {
                return Err(QuantaError::invalid_param(
                    "ICB execute count exceeds recorded length",
                ));
            }
            icb.commands[..count as usize]
                .iter()
                .map(|r| match r {
                    RecordedCommand::Dispatch {
                        wave_handle,
                        bindings,
                        binding_count,
                        texture_bindings,
                        texture_count,
                        push_data,
                        push_len,
                        push_mask,
                        workgroup_size,
                        groups,
                    } => RecordedCommand::Dispatch {
                        wave_handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: *texture_bindings,
                        texture_count: *texture_count,
                        push_data: *push_data,
                        push_len: *push_len,
                        push_mask: *push_mask,
                        workgroup_size: *workgroup_size,
                        groups: *groups,
                    },
                    RecordedCommand::Draw {
                        pipeline,
                        vertex_count,
                        instance_count,
                    } => RecordedCommand::Draw {
                        pipeline: *pipeline,
                        vertex_count: *vertex_count,
                        instance_count: *instance_count,
                    },
                })
                .collect()
        };
        for rec in &snapshot {
            match rec {
                RecordedCommand::Dispatch {
                    wave_handle,
                    bindings,
                    binding_count,
                    texture_bindings,
                    texture_count,
                    push_data,
                    push_len,
                    push_mask,
                    workgroup_size,
                    groups,
                } => {
                    // Reconstruct a transient Wave from the snapshot.
                    // `live: false` + `device: None` disarm its Drop,
                    // so replay does not destroy the real wave's
                    // registry entry.
                    let wave = Wave {
                        handle: *wave_handle,
                        bindings: *bindings,
                        binding_count: *binding_count,
                        texture_bindings: *texture_bindings,
                        texture_count: *texture_count,
                        // ICB replay does not carry the storage-texture mask;
                        // format validation runs on the direct dispatch path.
                        storage_texture_kinds: [0; 16],
                        push_data: *push_data,
                        push_len: *push_len,
                        push_mask: *push_mask,
                        workgroup_size: *workgroup_size,
                        device: None,
                        live: false,
                    };
                    let mut pulse = self.wave_dispatch(&wave, *groups)?;
                    pulse.wait()?;
                }
                RecordedCommand::Draw {
                    pipeline: _,
                    vertex_count: _,
                    instance_count: _,
                } => {
                    // CPU device has no rasterizer — recorded draws
                    // replay as no-ops. T7006 (record-draw appends
                    // to the sequence) is still satisfied; backends
                    // with a real raster path provide the visible
                    // side-effect.
                }
            }
        }
        Ok(())
    }

    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.icbs.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Bindless typed wrappers (steps 034 + 035) ===
    //
    // CPU implementation refines the abstract `Quanta.Bindless.Array`:
    // - create allocates a Vec<u64> sized to cap, all zeroed.
    // - set updates one slot, bounds-checked.
    // - destroy removes the handle.

    fn bindless_texture_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_textures.lock().unwrap().insert(
            handle,
            CpuBindlessArray {
                cap,
                entries: vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_texture_set(
        &self,
        handle: u64,
        index: u32,
        texture: u64,
    ) -> Result<(), QuantaError> {
        let mut arrays = self.bindless_textures.lock().unwrap();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless texture array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless texture index >= capacity",
            ));
        }
        arr.entries[index as usize] = texture;
        Ok(())
    }

    fn bindless_texture_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_textures.lock().unwrap().remove(&handle);
        Ok(())
    }

    fn bindless_buffer_create(&self, cap: u32) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.bindless_buffers.lock().unwrap().insert(
            handle,
            CpuBindlessArray {
                cap,
                entries: vec![0u64; cap as usize],
            },
        );
        Ok(handle)
    }

    fn bindless_buffer_set(&self, handle: u64, index: u32, buffer: u64) -> Result<(), QuantaError> {
        let mut arrays = self.bindless_buffers.lock().unwrap();
        let arr = arrays
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("bindless buffer array not found"))?;
        if index >= arr.cap {
            return Err(QuantaError::invalid_param(
                "bindless buffer index >= capacity",
            ));
        }
        arr.entries[index as usize] = buffer;
        Ok(())
    }

    fn bindless_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.bindless_buffers.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Tessellation pipelines (steps 022 + 023) ===
    //
    // CPU implementation refines `Quanta.Tessellation.Pipeline`:
    // - create allocates outer/inner Vecs sized to topology, all 1s.
    // - set_{outer,inner} updates one slot, bounds-checked. The typed
    //   wrapper has already clamped the factor into [1, MAX_TESS_LEVEL].
    // - destroy removes the handle.

    fn tessellation_pipeline_create(
        &self,
        topology: u8,
        _control_points: u32,
    ) -> Result<u64, QuantaError> {
        let (outer_count, inner_count) = match topology {
            0 => (3usize, 1usize),
            1 => (4usize, 2usize),
            _ => {
                return Err(QuantaError::invalid_param(
                    "tessellation topology must be 0 (triangle) or 1 (quad)",
                ));
            }
        };
        let _ = topology;
        let handle = self.alloc_handle();
        self.tess_pipelines.lock().unwrap().insert(
            handle,
            CpuTessPipeline {
                outer: vec![1u32; outer_count],
                inner: vec![1u32; inner_count],
            },
        );
        Ok(handle)
    }

    fn tessellation_set_outer(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self.tess_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.outer.len() {
            return Err(QuantaError::invalid_param(
                "tessellation outer index out of range",
            ));
        }
        pipe.outer[index as usize] = factor;
        Ok(())
    }

    fn tessellation_set_inner(
        &self,
        handle: u64,
        index: u32,
        factor: u32,
    ) -> Result<(), QuantaError> {
        let mut pipes = self.tess_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("tessellation pipeline not found"))?;
        if (index as usize) >= pipe.inner.len() {
            return Err(QuantaError::invalid_param(
                "tessellation inner index out of range",
            ));
        }
        pipe.inner[index as usize] = factor;
        Ok(())
    }

    fn tessellation_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.tess_pipelines.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Mesh shader pipelines (steps 024 + 025) ===
    //
    // CPU implementation refines `Quanta.MeshShader.Pipeline`:
    // - create stores the requested limits with an empty dispatch list.
    // - mesh_dispatch appends one [gx, gy, gz] entry to dispatched.
    // - destroy removes the handle.
    //
    // Rasterization is not modeled — the CPU device is a software
    // device with no display path, and the proven contract concerns
    // only lifecycle + dispatch ordering (T7302).

    fn mesh_pipeline_create(
        &self,
        max_vertices: u32,
        max_primitives: u32,
        task_threads: u32,
    ) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.mesh_pipelines.lock().unwrap().insert(
            handle,
            CpuMeshPipeline {
                max_vertices,
                max_primitives,
                task_threads,
                dispatched: Vec::new(),
            },
        );
        Ok(handle)
    }

    fn mesh_dispatch(&self, handle: u64, groups: [u32; 3]) -> Result<(), QuantaError> {
        let mut pipes = self.mesh_pipelines.lock().unwrap();
        let pipe = pipes
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("mesh pipeline not found"))?;
        pipe.dispatched.push(groups);
        let _ = pipe.max_vertices;
        let _ = pipe.max_primitives;
        let _ = pipe.task_threads;
        Ok(())
    }

    fn mesh_pipeline_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.mesh_pipelines.lock().unwrap().remove(&handle);
        Ok(())
    }

    // === Variable rate shading (steps 028 + 029) ===
    //
    // CPU implementation refines `Quanta.Vrs.State`:
    // - vrs_create allocates a state at default rate code 0 (1×1).
    // - vrs_set_rate writes the rate code (the typed wrapper has
    //   already encoded it from the ShadingRate enum).
    // - vrs_destroy removes the handle.

    fn vrs_create(&self) -> Result<u64, QuantaError> {
        let handle = self.alloc_handle();
        self.vrs_states
            .lock()
            .unwrap()
            .insert(handle, CpuVrsState { rate_code: 0 });
        Ok(handle)
    }

    fn vrs_set_rate(&self, handle: u64, rate_code: u8) -> Result<(), QuantaError> {
        let mut states = self.vrs_states.lock().unwrap();
        let st = states
            .get_mut(&handle)
            .ok_or_else(|| QuantaError::not_found("VRS state not found"))?;
        if rate_code > 6 {
            return Err(QuantaError::invalid_param("VRS rate code out of range"));
        }
        st.rate_code = rate_code;
        Ok(())
    }

    fn vrs_destroy(&self, handle: u64) -> Result<(), QuantaError> {
        self.vrs_states.lock().unwrap().remove(&handle);
        Ok(())
    }
}

/// Compute barrier segment ranges as (start, end) indices into the ops slice.
///
/// Each range covers the ops between consecutive barriers.
/// The barrier ops themselves are excluded. Zero allocation at dispatch
/// time — segments are sliced from the original body via `&ops[start..end]`.
/// Rewrite an op list so every barrier sits at the top level, lifting
/// barriers out of (uniform) control flow.
///
/// The CPU dispatcher segments a workgroup body at top-level barriers and
/// runs each segment for all lanes cooperatively (so a `shared_store`
/// before a barrier is visible to a `shared_load` after it). But inlining a
/// `#[quanta::device]` fn wraps the callee body in structural blocks, so a
/// barrier written between a store and a load can end up nested inside a
/// `Branch`. The segmenter then misses it and the cooperative store→load
/// collapses into one segment, reading stale shared memory.
///
/// A barrier inside a *divergent* branch is undefined behaviour on real
/// GPUs, so any branch we find containing a barrier is workgroup-uniform.
/// Lifting it is therefore sound: a branch whose body is `pre; barrier;
/// post` is equivalent to `if c {pre}; barrier; if c {post}` when `c` is
/// uniform. We split such a branch at each barrier, duplicating the
/// condition onto each half, and emit the barrier at the top level.
fn hoist_barriers(ops: Vec<KernelOp>) -> Vec<KernelOp> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                let then_h = hoist_barriers(then_ops);
                let else_h = hoist_barriers(else_ops);
                let then_contains = then_h.iter().any(|o| matches!(o, KernelOp::Barrier));
                let else_contains = else_h.iter().any(|o| matches!(o, KernelOp::Barrier));
                if !then_contains && !else_contains {
                    // No barrier inside — keep as a single branch.
                    out.push(KernelOp::Branch {
                        cond,
                        then_ops: then_h,
                        else_ops: else_h,
                    });
                } else {
                    // Split both arms into barrier-separated chunks and
                    // interleave: chunk0(then/else); Barrier; chunk1; ...
                    let then_chunks = split_chunks(then_h);
                    let else_chunks = split_chunks(else_h);
                    let n = then_chunks.len().max(else_chunks.len());
                    for i in 0..n {
                        let tc = then_chunks.get(i).cloned().unwrap_or_default();
                        let ec = else_chunks.get(i).cloned().unwrap_or_default();
                        if !tc.is_empty() || !ec.is_empty() {
                            out.push(KernelOp::Branch {
                                cond,
                                then_ops: tc,
                                else_ops: ec,
                            });
                        }
                        if i + 1 < n {
                            out.push(KernelOp::Barrier);
                        }
                    }
                }
            }
            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                // Recurse for nested correctness; barriers crossing a loop
                // boundary aren't hoisted (and prims kernels don't put
                // barriers in loops).
                out.push(KernelOp::Loop {
                    count,
                    iter_reg,
                    body: hoist_barriers(body),
                });
            }
            other => out.push(other),
        }
    }
    out
}

/// Split a (barrier-hoisted) op list into the maximal chunks between
/// top-level barriers. Barriers themselves are dropped. A list with no
/// barrier yields a single chunk.
fn split_chunks(ops: Vec<KernelOp>) -> Vec<Vec<KernelOp>> {
    let mut chunks = vec![Vec::new()];
    for op in ops {
        if matches!(op, KernelOp::Barrier) {
            chunks.push(Vec::new());
        } else {
            chunks.last_mut().unwrap().push(op);
        }
    }
    chunks
}

fn barrier_segment_ranges(ops: &[KernelOp]) -> Vec<(usize, usize)> {
    let mut segments = Vec::new();
    let mut start = 0;

    for (i, op) in ops.iter().enumerate() {
        if matches!(op, KernelOp::Barrier) {
            segments.push((start, i));
            start = i + 1;
        }
    }
    segments.push((start, ops.len()));
    segments
}

/// Discover CPU devices. Always returns exactly one.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    vec![Box::new(CpuDevice::new())]
}

#[cfg(test)]
mod tests {
    use super::super::eval::{eval_binop, eval_cmp};
    use super::super::value::{Value, f16_to_f32, f32_to_f16, read_scalar, write_scalar};
    use super::*;
    use quanta_ir::{BinOp, CmpOp, ScalarType};

    #[test]
    fn value_conversions() {
        assert_eq!(Value::U32(42).as_u32(), 42);
        assert_eq!(Value::U32(42).as_f32(), 42.0);
        assert_eq!(Value::F32(3.25).as_u32(), 3);
        assert!(Value::U32(1).as_bool());
        assert!(!Value::U32(0).as_bool());
        assert!(Value::Bool(true).as_bool());
    }

    #[test]
    fn scalar_read_write_roundtrip() {
        let mut buf = vec![0u8; 16];
        write_scalar(&mut buf, 0, Value::F32(3.25), &ScalarType::F32);
        let v = read_scalar(&buf, 0, &ScalarType::F32);
        assert!((v.as_f32() - 3.25).abs() < 1e-6);

        write_scalar(&mut buf, 1, Value::U32(42), &ScalarType::U32);
        let v = read_scalar(&buf, 1, &ScalarType::U32);
        assert_eq!(v.as_u32(), 42);
    }

    #[test]
    fn binop_add() {
        let r = eval_binop(Value::U32(3), Value::U32(4), &BinOp::Add, &ScalarType::U32);
        assert_eq!(r.as_u32(), 7);

        let r = eval_binop(
            Value::F32(1.5),
            Value::F32(2.5),
            &BinOp::Add,
            &ScalarType::F32,
        );
        assert!((r.as_f32() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn binop_div_by_zero() {
        let r = eval_binop(Value::U32(10), Value::U32(0), &BinOp::Div, &ScalarType::U32);
        assert_eq!(r.as_u32(), 0);
    }

    #[test]
    fn cmp_ops() {
        let r = eval_cmp(Value::U32(3), Value::U32(5), &CmpOp::Lt, &ScalarType::U32);
        assert!(r.as_bool());

        let r = eval_cmp(Value::U32(5), Value::U32(3), &CmpOp::Lt, &ScalarType::U32);
        assert!(!r.as_bool());
    }

    #[test]
    fn f16_roundtrip() {
        let original = 1.5f32;
        let bits = f32_to_f16(original);
        let back = f16_to_f32(bits);
        assert!((back - original).abs() < 1e-3);
    }

    #[test]
    fn cpu_device_field_alloc_write_read() {
        let dev = CpuDevice::new();
        let handle = dev.field_alloc(16, FieldUsage::default_compute()).unwrap();
        dev.field_write_bytes(handle, &[1, 2, 3, 4]).unwrap();
        let data = dev.field_read_bytes(handle, 16).unwrap();
        assert_eq!(&data[..4], &[1, 2, 3, 4]);
        assert_eq!(&data[4..], &[0; 12]);
        dev.field_free(handle);
    }

    #[test]
    fn cpu_device_caps() {
        let dev = CpuDevice::new();
        assert_eq!(dev.caps().vendor, Vendor::Software);
        assert_eq!(dev.caps().name, "Quanta CPU (software)");
    }

    #[test]
    fn cpu_device_wave_rejects_binary() {
        let dev = CpuDevice::new();
        let result = dev.wave(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn cpu_device_pulse_is_synchronous() {
        let dev = CpuDevice::new();
        let pulse = Pulse {
            handle: 0,
            completed: false,
            wait_fn: None,
            keep_alive: None,
        };
        assert!(dev.pulse_poll(&pulse));
    }
}

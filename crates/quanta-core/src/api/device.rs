use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Caps, FieldUsage, Format, FormatCaps, NativeTextureHandle, Pulse, QuantaError, QueueFamily,
    QueueType, ResourceState, Texture, TextureDesc, TextureViewDesc, Timeline,
};
// `Wave` is a compute type; only the compute-gated trait methods
// (wave / dispatch / batch / queue-dispatch / compute ICB) reference it.
#[cfg(feature = "compute")]
use crate::Wave;
// Render types used only by the render-gated trait methods (step 085).
#[cfg(feature = "render")]
use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
#[cfg(feature = "render")]
use crate::surface::{SurfaceConfig, SurfaceTarget};
#[cfg(feature = "render")]
use crate::{Pipeline, RenderPass};

/// Sealed-trait guard for [`GpuDevice`]. Private module, private
/// trait: only this crate can name it, so only this crate's drivers
/// can implement `GpuDevice`.
pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Core trait — every GPU driver implements this.
///
/// Methods use raw bytes and handles to keep the trait dyn-compatible.
/// Users interact with the `Gpu` wrapper which provides typed, ergonomic methods.
///
/// **Sealed**: this trait can only be implemented inside the `quanta`
/// crate (drivers are in-tree). Consumers hold it as `Arc<dyn
/// GpuDevice>` through [`Gpu`](crate::Gpu); because no external impls
/// exist, new trait methods with default bodies can be added after the
/// API freeze without a breaking change.
pub trait GpuDevice: sealed::Sealed + Send + Sync {
    // === Device info ===

    fn caps(&self) -> &Caps;

    // === Feature support queries (step 063 slice 20) ===
    //
    // Public surface for the per-backend capability caches added by
    // slices 6 / 14 / 16 / 17. Default returns `false` so a caller
    // doing `if gpu.supports_vrs() { … }` correctly skips the path
    // on backends that don't override (CPU, WebGPU). Backends that
    // cache the answer at device discovery override these with a
    // simple `self.<cached_field>` read.

    /// Whether the backend can lower `RenderOp::SetShadingRate` to
    /// a native VRS path (Vulkan vkCmdSetFragmentShadingRateKHR or
    /// Metal MTLRasterizationRateMap). Returns `false` when the
    /// extension / device-family is absent.
    fn supports_variable_rate_shading(&self) -> bool {
        false
    }

    /// Whether the backend can build acceleration structures and
    /// dispatch ray tracing pipelines. Returns `true` only when
    /// every prerequisite (extensions, family-7+ on Apple,
    /// proc-addr resolution) is in place.
    fn supports_ray_tracing(&self) -> bool {
        false
    }

    /// Whether the backend can create mesh-shader pipelines
    /// (Vulkan VK_EXT_mesh_shader / Metal 3
    /// MTLMeshRenderPipelineDescriptor).
    fn supports_mesh_shaders(&self) -> bool {
        false
    }

    /// Whether the backend can create tessellation pipelines
    /// (Vulkan tessellationShader feature / Metal Apple GPU
    /// family 4+).
    fn supports_tessellation(&self) -> bool {
        false
    }

    /// Whether the backend can create sparse textures with real
    /// residency control (Vulkan sparseBinding feature + queue
    /// support / Metal Apple GPU family 7+).
    fn supports_sparse_residency(&self) -> bool {
        false
    }

    /// Whether the backend can lower the cooperative-matrix IR ops
    /// (`CooperativeMatrixLoad` / `CooperativeMMA` / `CooperativeMatrixStore`)
    /// to native tensor-core / SIMD-group-matrix instructions. Metal: Apple
    /// GPU family 7+ (`simdgroup_matrix`). Vulkan: `VK_KHR_cooperative_matrix`
    /// (not yet wired). The CPU reference interpreter reports `false`, so
    /// callers (e.g. `quanta-blas`'s tensor-core GEMM) fall back to a scalar
    /// kernel there.
    fn supports_cooperative_matrix(&self) -> bool {
        false
    }

    /// Whether the backend can run kernels that use 64-bit floats.
    /// Vulkan: `VkPhysicalDeviceFeatures.shaderFloat64` enabled at
    /// device creation (true on llvmpipe, false on Broadcom V3D).
    /// The CPU reference interpreter always supports f64; Metal has
    /// no `double` type and reports false.
    fn supports_f64(&self) -> bool {
        false
    }

    /// Whether the backend can run kernels that use 64-bit integers.
    /// Vulkan: `VkPhysicalDeviceFeatures.shaderInt64` enabled at device
    /// creation (true on llvmpipe, false on Broadcom V3D). The CPU
    /// reference interpreter always supports i64/u64.
    fn supports_i64(&self) -> bool {
        false
    }

    /// Whether the backend can run kernels that use subgroup (warp/
    /// SIMD-group) *arithmetic* operations — `reduce_add_*` /
    /// `reduce_min_*` / `reduce_max_*` / `scan_add_*` and friends.
    /// Vulkan: `VkPhysicalDeviceSubgroupProperties.supportedOperations`
    /// contains `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT` for the compute
    /// stage (true on llvmpipe, false on Broadcom V3D — Mesa's V3D NIR
    /// backend cannot lower `OpGroupNonUniformFAdd`, the driver aborts
    /// at pipeline creation). Metal always has SIMD-group reductions;
    /// the CPU reference interpreter resolves subgroup ops
    /// warp-cooperatively. Callers with a subgroup-free fallback (e.g.
    /// quanta-prims' shared-memory tree reduction) should select on
    /// this at dispatch-build time.
    fn supports_subgroups(&self) -> bool {
        false
    }

    /// Whether narrow-float buffers (bf16 / fp8) on this backend use the
    /// portable u32-slot layout — one element per 32-bit word — instead of
    /// native stride (16-/8-bit elements, the contract shared by the host
    /// upload, the CPU executor and the MSL/SPIR-V emitters). WGSL has no
    /// 16-/8-bit storage types, so only the WebGPU backend returns `true`;
    /// hosts feeding it tight bf16/fp8 data must expand it
    /// one-element-per-word before binding (see `quanta-blas`'s mixed GEMM
    /// dispatch for the reference repack).
    fn narrow_storage_u32_slot(&self) -> bool {
        false
    }

    /// Hardware-supported shading rates as `(width, height)` pairs.
    /// Empty when VRS isn't supported. The render encoder validates
    /// requested rates against this list before submission.
    fn supported_shading_rates(&self) -> Vec<(u32, u32)> {
        Vec::new()
    }

    // === Fields (GPU memory) ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError>;
    fn field_free(&self, handle: u64);
    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError>;
    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError>;
    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError>;

    /// Write `data` into a field starting at byte offset `byte_offset`,
    /// leaving the rest of the buffer untouched. Native partial-upload on
    /// CPU / Metal / Vulkan (host-visible mapping at the offset); WebGPU uses
    /// the default read-modify-write fallback below. Writing past the end of
    /// the buffer is clamped.
    fn field_write_bytes_at(
        &self,
        handle: u64,
        byte_offset: usize,
        data: &[u8],
    ) -> Result<(), QuantaError> {
        if byte_offset == 0 {
            return self.field_write_bytes(handle, data);
        }
        // Read the prefix we must preserve, splice in `data`, write it back.
        let end = byte_offset + data.len();
        let mut buf = self.field_read_bytes(handle, end)?;
        if buf.len() < end {
            buf.resize(end, 0);
        }
        buf[byte_offset..end].copy_from_slice(data);
        self.field_write_bytes(handle, &buf)
    }

    /// Map a GPU buffer into CPU address space for direct read/write access.
    fn field_map(&self, _handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        Err(QuantaError::not_supported("mapped buffers not supported"))
    }

    /// Unmap a previously mapped GPU buffer.
    fn field_unmap(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("mapped buffers not supported"))
    }

    /// Create a buffer that is permanently mapped into CPU address space.
    /// Returns (handle, pointer) — the pointer remains valid until the buffer is freed.
    fn field_create_mapped(
        &self,
        _size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        Err(QuantaError::not_supported("mapped buffers not supported"))
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError>;
    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError>;
    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError>;
    fn sampler_create(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError>;
    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError>;

    // === Compute === (compute-typed; gated with the `compute` feature,
    // mirroring the render-gated methods below)

    #[cfg(feature = "compute")]
    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError>;

    /// JIT-compile a kernel from its serialized KernelDef at runtime.
    ///
    /// Deserializes the IR, emits the appropriate shader format (MSL text for
    /// Metal, SPIR-V binary for Vulkan), and compiles it. Requires the `jit`
    /// feature on quanta-ir.
    #[cfg(feature = "compute")]
    fn wave_jit(&self, _kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        Err(QuantaError::compilation_failed(
            "JIT compilation not supported by this driver",
        ))
    }

    /// Dispatch by threadgroup count (e.g., [4, 1, 1] = 4 groups of workgroup_size threads).
    #[cfg(feature = "compute")]
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError>;
    /// Dispatch by total thread count (Metal clips, Vulkan computes groups).
    #[cfg(feature = "compute")]
    fn wave_dispatch_threads(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        let groups = quarks.div_ceil(wave.workgroup_size[0]);
        self.wave_dispatch(wave, [groups, 1, 1])
    }
    /// Dispatch with group counts from a GPU buffer (GPU decides grid size).
    #[cfg(feature = "compute")]
    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError>;

    // === Batch ===

    #[cfg(feature = "compute")]
    fn batch_begin(&self) -> Result<crate::Batch, QuantaError> {
        Err(QuantaError::not_supported("batch dispatch not supported"))
    }

    // === Render === (render-typed; gated with the `render` feature, step 085)

    #[cfg(feature = "render")]
    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError>;
    #[cfg(feature = "render")]
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError>;
    #[cfg(feature = "render")]
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError>;

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError>;
    fn pulse_poll(&self, pulse: &Pulse) -> bool;

    // === Timestamps ===

    /// Create a timestamp query set with `count` slots.
    fn timestamp_query_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported("timestamps not supported"))
    }

    /// Write a timestamp at the given index in the query set.
    fn timestamp_write(&self, _query_handle: u64, _index: u32) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("timestamps not supported"))
    }

    /// Read timestamp values from a query set.
    fn timestamp_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::not_supported("timestamps not supported"))
    }

    /// GPU timestamp counter frequency in Hz. Default: 1 GHz (timestamps in nanoseconds).
    fn timestamp_frequency(&self) -> u64 {
        1_000_000_000
    }

    // === Async compute ===

    /// Whether this device supports a dedicated async compute queue.
    fn supports_async_compute(&self) -> bool {
        false
    }

    /// Dispatch a compute wave on the async compute queue.
    /// Returns immediately; the returned Pulse signals completion.
    #[cfg(feature = "compute")]
    fn async_compute_dispatch(
        &self,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::not_supported("async compute not supported"))
    }

    // === Timeline semaphores ===

    /// Create a timeline semaphore (monotonic u64 counter for multi-frame sync).
    fn timeline_create(&self) -> Result<Timeline, QuantaError> {
        Err(QuantaError::not_supported(
            "timeline semaphores not supported",
        ))
    }

    /// Signal a timeline to the given value.
    fn timeline_signal(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "timeline semaphores not supported",
        ))
    }

    /// Block until a timeline reaches at least the given value.
    fn timeline_wait(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "timeline semaphores not supported",
        ))
    }

    // === Barriers ===

    /// Full pipeline barrier — wait for all prior GPU work to complete.
    ///
    /// This is a heavyweight synchronization point. Prefer `barrier_buffer`
    /// or `barrier_texture` for fine-grained transitions when possible.
    fn barrier(&self) -> Result<(), QuantaError> {
        Ok(()) // default no-op for drivers that don't need explicit barriers
    }

    /// Transition a buffer between resource states.
    ///
    /// On Vulkan, this inserts a `VkBufferMemoryBarrier2` with the appropriate
    /// stage and access masks. On Metal, this is a no-op (hazard tracking).
    fn barrier_buffer(
        &self,
        _handle: u64,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Transition a texture (image) between resource states.
    ///
    /// On Vulkan, this inserts a `VkImageMemoryBarrier2` with the appropriate
    /// layout transition. On Metal, this is a no-op (hazard tracking).
    fn barrier_texture(
        &self,
        _texture: &Texture,
        _from: ResourceState,
        _to: ResourceState,
    ) -> Result<(), QuantaError> {
        Ok(())
    }

    // === MSAA Resolve ===

    /// Resolve an MSAA texture to a single-sample texture.
    #[cfg(feature = "render")]
    fn resolve_texture(&self, _src_handle: u64, _dst_handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("MSAA resolve not supported"))
    }

    // === M2.2: Format capability queries ===

    /// Query what a given format can do on this device.
    fn format_caps(&self, _format: Format) -> FormatCaps {
        FormatCaps {
            filterable: true,
            renderable: true,
            storage: true,
            blendable: true,
            msaa: true,
            depth: false,
        }
    }

    // === M2.3: Texture views ===

    /// Create a view into an existing texture (sub-range of mips/layers).
    fn texture_view_create(
        &self,
        _texture: u64,
        _desc: &TextureViewDesc,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported("texture views not supported"))
    }

    /// Destroy a texture view.
    fn texture_view_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("texture views not supported"))
    }

    // === M2.6: Stencil read-back ===

    /// Read stencil buffer contents from a depth/stencil texture.
    #[cfg(feature = "render")]
    fn stencil_read(&self, _texture: u64) -> Result<Vec<u8>, QuantaError> {
        Err(QuantaError::not_supported(
            "stencil read-back not supported",
        ))
    }

    // === M3.1: Multi-queue ===

    /// List available queue families on this device.
    fn queue_families(&self) -> Vec<QueueFamily> {
        vec![QueueFamily {
            queue_type: QueueType::Graphics,
            count: 1,
        }]
    }

    /// Create a queue of the given type.
    fn create_queue(&self, _queue_type: QueueType) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported("multi-queue not supported"))
    }

    /// Submit a compute dispatch to a specific queue.
    #[cfg(feature = "compute")]
    fn queue_dispatch(
        &self,
        _queue: u64,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("multi-queue not supported"))
    }

    /// Signal a semaphore from a queue.
    fn queue_signal(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("multi-queue not supported"))
    }

    /// Wait on a semaphore before executing more work on a queue.
    fn queue_wait(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported("multi-queue not supported"))
    }

    /// Destroy a queue handle. Default no-ops so backends without
    /// an explicit registry don't error on Drop.
    fn queue_destroy(&self, _queue: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Async memory copy (step 044) ===
    //
    // Async-copy typed wrapper (`AsyncCopyQueue`) refines
    // `Quanta.AsyncCopy.Queue` from the Lean equivalence theorems
    // (T7800–T7804). Backends opt in by overriding these methods;
    // defaults return NotSupported so the typed wrapper surfaces a
    // clear error on platforms without a dedicated DMA engine.

    /// Allocate a fresh async-copy queue. Default returns
    /// "not yet implemented".
    fn async_copy_create(&self) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "async memory copy not yet implemented on this backend",
        ))
    }

    /// Submit a buffer-to-buffer copy on the async-copy queue.
    fn async_copy_submit(
        &self,
        _queue: u64,
        _dst: u64,
        _src: u64,
        _size: usize,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "async memory copy not yet implemented on this backend",
        ))
    }

    /// Destroy an async-copy queue. Default no-ops so backends
    /// without an explicit registry don't error on Drop.
    fn async_copy_destroy(&self, _queue: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === GPU printf (step 049) ===
    //
    // Printf typed wrapper (`PrintfBuffer`) refines
    // `Quanta.Printf.Buffer` from the Lean equivalence theorems
    // (T7900–T7905). Backends opt in by overriding these methods;
    // defaults return NotSupported.

    /// Allocate a printf buffer with capacity for `cap` messages.
    fn printf_create(&self, _cap: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "GPU printf not yet implemented on this backend",
        ))
    }

    /// Record a message id into the printf buffer.
    fn printf_record(&self, _handle: u64, _msg_id: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "GPU printf not yet implemented on this backend",
        ))
    }

    /// Drain recorded messages, leaving the buffer empty.
    fn printf_drain(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::not_supported(
            "GPU printf not yet implemented on this backend",
        ))
    }

    /// Destroy a printf buffer. Default no-ops so backends without
    /// an explicit registry don't error on Drop.
    fn printf_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M3.3: Occlusion queries ===

    /// Create an occlusion query set with `count` slots.
    fn occlusion_query_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "occlusion queries not supported",
        ))
    }

    /// Read results from an occlusion query set (fragment counts per slot).
    fn occlusion_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::not_supported(
            "occlusion queries not supported",
        ))
    }

    // === M4.2: Mesh shaders ===

    /// Dispatch a mesh shader pipeline.
    fn dispatch_mesh(&self, pipeline: u64, groups: [u32; 3]) -> Result<(), QuantaError>;

    // === Mesh shader pipelines (steps 024 + 025) ===
    //
    // Mesh-shader typed wrapper (`MeshPipeline`) refines
    // `Quanta.MeshShader.Pipeline` from the Lean equivalence theorems
    // (T7300–T7305). Backends opt in by overriding these methods;
    // defaults return `NotSupported` so the typed wrapper surfaces
    // a clear error on platforms without mesh shaders (WebGPU,
    // older Metal / pre-Vulkan-1.3 software-only fallbacks).

    /// Create a mesh pipeline state with the given vertex / primitive
    /// / task-thread limits. Default returns "not yet implemented".
    fn mesh_pipeline_create(
        &self,
        _max_vertices: u32,
        _max_primitives: u32,
        _task_threads: u32,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "mesh shaders not yet implemented on this backend",
        ))
    }

    /// Dispatch `[gx, gy, gz]` mesh workgroups on the typed pipeline.
    /// The typed wrapper has already bounds-checked groups against
    /// `MAX_GROUP_COUNT`.
    fn mesh_dispatch(&self, _handle: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "mesh shaders not yet implemented on this backend",
        ))
    }

    /// Destroy a mesh pipeline. Default no-ops so backends without an
    /// implementation don't error on `Drop`.
    fn mesh_pipeline_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M4.3: Ray tracing ===

    /// Build a bottom-level acceleration structure from geometry.
    /// Render-typed (`GeometryDesc`); gated with the `render` feature.
    #[cfg(feature = "render")]
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError>;

    /// Create a ray tracing pipeline from shader stages.
    /// Render-typed (`RayTracingPipelineDesc`); gated with `render`.
    #[cfg(feature = "render")]
    fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError>;

    /// Dispatch rays through a ray tracing pipeline.
    fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError>;

    /// Destroy an acceleration structure.
    fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError>;

    /// Destroy a ray tracing pipeline. Default no-ops so backends
    /// without a registry don't error on `Drop`.
    fn destroy_ray_tracing_pipeline(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M5.1: Sparse textures ===

    /// Create a sparse (virtual) texture — memory is not committed until tiles are mapped.
    fn sparse_texture_create(&self, desc: &TextureDesc) -> Result<u64, QuantaError>;

    /// Map a physical backing page to a sparse texture tile.
    fn sparse_map_tile(
        &self,
        texture: u64,
        mip: u32,
        x: u32,
        y: u32,
        backing: u64,
    ) -> Result<(), QuantaError>;

    /// Unmap a sparse texture tile (release backing memory).
    fn sparse_unmap_tile(&self, texture: u64, mip: u32, x: u32, y: u32) -> Result<(), QuantaError>;

    /// Destroy a sparse texture handle. Default no-ops so backends
    /// without an explicit registry don't error on Drop.
    fn sparse_texture_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === M5.2: Indirect command buffers (steps 032 + 033) ===
    //
    // The compute-ICB family (create / record / execute / destroy) is
    // gated with the `compute` feature: the only caller is the
    // compute-gated `IndirectCommandBuffer` wrapper, and per-backend
    // record/execute paths lean on the compute dispatch machinery.
    // The render bundle (`render_bundle_*`) family below stays
    // independent — it serves the render-gated `IndirectRenderBundle`.

    /// Create an indirect command buffer (GPU-driven draw/dispatch).
    #[cfg(feature = "compute")]
    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError>;

    /// Record a single dispatch command at `index` in the ICB.
    ///
    /// Snapshots the wave's pipeline + current bindings + group counts.
    /// `index` is the command position assigned by
    /// [`IndirectCommandBuffer::record_dispatch`](crate::IndirectCommandBuffer::record_dispatch);
    /// the typed wrapper enforces `index < max_commands`.
    #[cfg(feature = "compute")]
    fn icb_record_dispatch(
        &self,
        handle: u64,
        index: u32,
        wave: &Wave,
        groups: [u32; 3],
    ) -> Result<(), QuantaError>;

    /// Record a single render-path draw command at `index` in the
    /// ICB. Default returns "not yet implemented" — backends opt in
    /// by overriding when they wire their render bundle / indirect
    /// render command lowering.
    ///
    /// Refines the `Quanta.Icb.Command.draw` constructor; the proof
    /// contract (T7000 / T7006) holds for any backend that respects
    /// the recorded order on execute.
    #[cfg(feature = "compute")]
    fn icb_record_draw(
        &self,
        _handle: u64,
        _index: u32,
        _pipeline: u64,
        _vertex_count: u32,
        _instance_count: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "render-path ICB record_draw not yet implemented on this backend",
        ))
    }

    // === Indirect render bundles (steps 032 + 033, render path) ===
    //
    // A render bundle is a separate ICB type from the compute
    // `indirect_buffer_*` family — it's recorded with DRAW command
    // types and must be replayed inside an active render pass via
    // `RenderPass::execute_bundle`. Backends with no native render
    // bundle support fall through to the default error.

    /// Allocate a render-path Indirect Command Buffer with capacity
    /// `max_commands`. Returns a fresh handle that the typed
    /// [`IndirectRenderBundle`](crate::IndirectRenderBundle)
    /// wraps. Default returns "not yet implemented".
    fn render_bundle_create(&self, _max_commands: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "render-path indirect command buffers not yet implemented on this backend",
        ))
    }

    /// Record one draw into a render bundle at `index`.
    fn render_bundle_record_draw(
        &self,
        _handle: u64,
        _index: u32,
        _pipeline: u64,
        _vertex_count: u32,
        _instance_count: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "render-path indirect command buffers not yet implemented on this backend",
        ))
    }

    /// Destroy a render bundle handle. Default no-ops so backends
    /// without an implementation don't error on `Drop`.
    fn render_bundle_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Execute commands from an indirect command buffer.
    #[cfg(feature = "compute")]
    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError>;

    /// Destroy an indirect command buffer.
    #[cfg(feature = "compute")]
    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError>;

    // === M5.3: Bindless resources (steps 034 + 035) ===
    //
    // Bindless typed wrappers (`BindlessTextureArray`,
    // `BindlessBufferArray`) refine `Quanta.Bindless.Array` from the
    // Lean equivalence theorems (T7100-T7106).

    /// Allocate a bindless texture array with the given capacity.
    /// Default returns "not yet implemented"; backends override.
    fn bindless_texture_create(&self, _cap: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "bindless texture arrays not yet implemented on this backend",
        ))
    }

    /// Update slot `index` of a bindless texture array.
    fn bindless_texture_set(
        &self,
        _handle: u64,
        _index: u32,
        _texture: u64,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "bindless texture arrays not yet implemented on this backend",
        ))
    }

    /// Destroy a bindless texture array. Default no-ops so backends
    /// without an implementation don't error on `Drop`.
    fn bindless_texture_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Allocate a bindless buffer array with the given capacity.
    fn bindless_buffer_create(&self, _cap: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "bindless buffer arrays not yet implemented on this backend",
        ))
    }

    /// Update slot `index` of a bindless buffer array.
    fn bindless_buffer_set(
        &self,
        _handle: u64,
        _index: u32,
        _buffer: u64,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "bindless buffer arrays not yet implemented on this backend",
        ))
    }

    /// Destroy a bindless buffer array.
    fn bindless_buffer_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Tessellation pipelines (steps 022 + 023) ===
    //
    // Tessellation typed wrapper (`TessellationPipeline`) refines
    // `Quanta.Tessellation.Pipeline` from the Lean equivalence
    // theorems (T7200–T7206). Backends opt in by overriding these
    // methods; defaults return `NotSupported` so the typed wrapper
    // surfaces a clear error on platforms without tessellation
    // (WebGPU, software-only fallbacks).

    /// Create a tessellation pipeline state with the given topology
    /// and control-point count. `topology` is `0` for triangle, `1`
    /// for quad. Default returns "not yet implemented".
    fn tessellation_pipeline_create(
        &self,
        _topology: u8,
        _control_points: u32,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "tessellation not yet implemented on this backend",
        ))
    }

    /// Update the outer (edge) tessellation factor at `index`. The
    /// factor is already clamped into `[1, MAX_TESS_LEVEL]` by the
    /// typed wrapper.
    fn tessellation_set_outer(
        &self,
        _handle: u64,
        _index: u32,
        _factor: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "tessellation not yet implemented on this backend",
        ))
    }

    /// Update the inner (interior) tessellation factor at `index`.
    fn tessellation_set_inner(
        &self,
        _handle: u64,
        _index: u32,
        _factor: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "tessellation not yet implemented on this backend",
        ))
    }

    /// Destroy a tessellation pipeline. Default no-ops so backends
    /// without an implementation don't error on `Drop`.
    fn tessellation_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Variable rate shading (steps 028 + 029) ===
    //
    // VRS typed wrapper (`VrsState`) refines `Quanta.Vrs.State` from
    // the Lean equivalence theorems (T7500–T7505). Backends opt in
    // by overriding these methods; defaults return NotSupported so
    // the typed wrapper surfaces a clear error on platforms without
    // VRS (WebGPU, devices without VK_KHR_fragment_shading_rate /
    // pre-Apple-Silicon Metal).

    /// Create a fresh VRS state. Default rate is 1×1 (no reduction).
    fn vrs_create(&self) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "variable rate shading not yet implemented on this backend",
        ))
    }

    /// Set the current shading rate. `rate_code` is the Verus
    /// encoding (0 = 1×1, 1 = 1×2, … 6 = 4×4). The typed wrapper
    /// has already validated the code.
    fn vrs_set_rate(&self, _handle: u64, _rate_code: u8) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "variable rate shading not yet implemented on this backend",
        ))
    }

    /// Destroy a VRS state. Default no-ops so backends without an
    /// implementation don't error on `Drop`.
    fn vrs_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // === Debug ===

    /// Push a debug group label (shows in GPU profilers).
    fn debug_push(&self, _label: &str) {}

    /// Pop a debug group label.
    fn debug_pop(&self) {}

    // ╭──────────────────────────────────────────────────────────────╮
    // │ Presentation & interop block (native-handle export + Surface)│
    // ╰──────────────────────────────────────────────────────────────╯
    //
    // Two presentation models, two consumers:
    //  1. native-handle export — an external compositor (the OS)
    //     imports Quanta's rendered texture directly; the importer
    //     owns present.
    //  2. Surface/swapchain — Quanta owns present; apps run the
    //     acquire → render → present frame loop (`crate::Surface`).
    //
    // Defaults return `NotSupported` so the typed wrappers surface a
    // clear error on backends that haven't wired these paths yet.

    /// Whether `texture_native_handle` returns a real backend object
    /// on this device. `false` on the CPU software driver (no native
    /// object exists) and on backends that haven't wired export yet.
    fn supports_native_handle_export(&self) -> bool {
        false
    }

    /// Export the backend-native handle behind `texture`. The handle
    /// is a borrow valid for the `Texture`'s lifetime — see
    /// [`Texture::native_handle`] for the full contract.
    fn texture_native_handle(
        &self,
        _texture: &Texture,
    ) -> Result<NativeTextureHandle, QuantaError> {
        Err(QuantaError::not_supported(
            "native texture handle export not supported on this backend",
        ))
    }

    /// Whether this device can create presentation surfaces
    /// (`surface_create` and the acquire/present family below).
    fn supports_surface_present(&self) -> bool {
        false
    }

    /// Create a presentation surface for `target`, configured with
    /// `config`. Returns the surface handle.
    #[cfg(feature = "render")]
    fn surface_create(
        &self,
        _target: &SurfaceTarget,
        _config: &SurfaceConfig,
    ) -> Result<u64, QuantaError> {
        Err(QuantaError::not_supported(
            "surface presentation not yet implemented on this backend",
        ))
    }

    /// Reconfigure a surface (resize, format or present-mode change).
    /// Frames acquired before the reconfigure must be presented or
    /// dropped first.
    #[cfg(feature = "render")]
    fn surface_configure(&self, _surface: u64, _config: &SurfaceConfig) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "surface presentation not yet implemented on this backend",
        ))
    }

    /// Acquire the next presentable frame from a surface. Returns the
    /// frame id and a `Texture` aliasing the frame's target image
    /// (registered in the device's texture registry until the frame
    /// is presented or discarded). Errors: `Timeout` when no frame
    /// became available, `SurfaceOutdated` when the surface must be
    /// reconfigured before acquiring.
    #[cfg(feature = "render")]
    fn surface_acquire(&self, _surface: u64) -> Result<(u64, Texture), QuantaError> {
        Err(QuantaError::not_supported(
            "surface presentation not yet implemented on this backend",
        ))
    }

    /// Present an acquired frame. Presentation is ordered after all
    /// GPU work already submitted against the frame's texture; the
    /// call schedules the present and returns without blocking on
    /// the GPU.
    #[cfg(feature = "render")]
    fn surface_present(&self, _surface: u64, _frame: u64) -> Result<(), QuantaError> {
        Err(QuantaError::not_supported(
            "surface presentation not yet implemented on this backend",
        ))
    }

    /// Discard an acquired frame without presenting (releases the
    /// backing image back to the swapchain). Default no-ops so
    /// backends without an implementation don't error on `Drop`.
    #[cfg(feature = "render")]
    fn surface_discard(&self, _surface: u64, _frame: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Destroy a surface. Default no-ops so backends without an
    /// implementation don't error on `Drop`.
    #[cfg(feature = "render")]
    fn surface_destroy(&self, _surface: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // ╭──────────────────────────────────────────────────────────────╮
    // │ End presentation & interop block                             │
    // ╰──────────────────────────────────────────────────────────────╯
    // ─────────────────────────────────────────────────────────────────
    // Render-resource lifecycle (destroy methods).
    //
    // Each destroy removes the handle from the driver registry and
    // frees the native object. The typed wrappers (`Texture`,
    // `Sampler`, `Pipeline`, `TextureView`, `OcclusionQuery`) call
    // these from `Drop`, guarded by their `live` flag, so each handle
    // is destroyed exactly once. Defaults are `Ok(())` no-ops so
    // registry-less backends stay silent on Drop.
    // ─────────────────────────────────────────────────────────────────

    /// Destroy a texture: remove it from the registry and free the
    /// native object. Destroying an unknown handle is a no-op.
    fn texture_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Destroy a sampler.
    fn sampler_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Destroy a render pipeline (and its associated per-pipeline
    /// state — depth/stencil objects, layouts, render passes).
    #[cfg(feature = "render")]
    fn pipeline_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Destroy an occlusion query set.
    fn occlusion_query_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────
    // Compute-resource lifecycle (destroy methods).
    //
    // Same shape as the render-resource destroys above: remove the
    // handle from the driver registry and free the native object
    // (compute pipeline state / compiled kernel). `Wave::drop` calls
    // this guarded by its `live` flag, so each handle is destroyed
    // exactly once. Default is an `Ok(())` no-op so registry-less
    // backends stay silent on Drop.
    // ─────────────────────────────────────────────────────────────────

    /// Destroy a wave: remove it from the registry and free the
    /// native object. Destroying an unknown handle is a no-op.
    #[cfg(feature = "compute")]
    fn wave_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Test-support hook: current sizes of the driver's resource
    /// registries. Lifecycle tests assert entries are freed on Drop.
    #[doc(hidden)]
    fn debug_registry_counts(&self) -> RegistryCounts {
        RegistryCounts::default()
    }
}

/// Snapshot of a driver's resource-registry sizes (test support).
///
/// Backends map their internal registries onto these fields as
/// applicable; unused fields stay 0. Compare full snapshots taken
/// before/after a create+drop cycle to prove the entry was freed.
#[doc(hidden)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RegistryCounts {
    pub buffers: usize,
    pub textures: usize,
    pub samplers: usize,
    pub render_pipelines: usize,
    pub query_sets: usize,
    /// Compute waves (compiled kernel pipelines / kernel defs).
    pub waves: usize,
}

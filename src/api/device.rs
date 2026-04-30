use alloc::vec;
use alloc::vec::Vec;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{
    Caps, FieldUsage, Format, FormatCaps, Pipeline, Pulse, QuantaError, QueueFamily, QueueType,
    RenderPass, ResourceState, Texture, TextureDesc, TextureViewDesc, Timeline, Wave,
};

/// Core trait — every GPU driver implements this.
///
/// Methods use raw bytes and handles to keep the trait dyn-compatible.
/// Users interact with the `Gpu` wrapper which provides typed, ergonomic methods.
pub trait GpuDevice: Send + Sync {
    // === Device info ===

    fn caps(&self) -> &Caps;

    // === Fields (GPU memory) ===

    fn field_alloc(&self, size: usize, usage: FieldUsage) -> Result<u64, QuantaError>;
    fn field_free(&self, handle: u64);
    fn field_write_bytes(&self, handle: u64, data: &[u8]) -> Result<(), QuantaError>;
    fn field_read_bytes(&self, handle: u64, size: usize) -> Result<Vec<u8>, QuantaError>;
    fn field_copy_bytes(&self, dst: u64, src: u64, size: usize) -> Result<(), QuantaError>;

    /// Map a GPU buffer into CPU address space for direct read/write access.
    fn field_map(&self, _handle: u64, _size: usize) -> Result<*mut u8, QuantaError> {
        Err(QuantaError::invalid_param("mapped buffers not supported"))
    }

    /// Unmap a previously mapped GPU buffer.
    fn field_unmap(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("mapped buffers not supported"))
    }

    /// Create a buffer that is permanently mapped into CPU address space.
    /// Returns (handle, pointer) — the pointer remains valid until the buffer is freed.
    fn field_create_mapped(
        &self,
        _size: usize,
        _usage: FieldUsage,
    ) -> Result<(u64, *mut u8), QuantaError> {
        Err(QuantaError::invalid_param("mapped buffers not supported"))
    }

    // === Textures ===

    fn texture_create(&self, desc: &TextureDesc) -> Result<Texture, QuantaError>;
    fn texture_write(&self, texture: &Texture, data: &[u8]) -> Result<(), QuantaError>;
    fn texture_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError>;
    fn sampler_create(
        &self,
        desc: &crate::render_pass::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError>;
    fn generate_mipmaps(&self, texture: &Texture) -> Result<(), QuantaError>;

    // === Compute ===

    fn wave(&self, kernel: &[u8]) -> Result<Wave, QuantaError>;

    /// JIT-compile a kernel from its serialized KernelDef at runtime.
    ///
    /// Deserializes the IR, emits the appropriate shader format (MSL text for
    /// Metal, SPIR-V binary for Vulkan), and compiles it. Requires the `jit`
    /// feature on quanta-ir.
    fn wave_jit(&self, _kernel_def: &[u8]) -> Result<Wave, QuantaError> {
        Err(QuantaError::compilation_failed(
            "JIT compilation not supported by this driver",
        ))
    }

    /// Dispatch by threadgroup count (e.g., [4, 1, 1] = 4 groups of workgroup_size threads).
    fn wave_dispatch(&self, wave: &Wave, groups: [u32; 3]) -> Result<Pulse, QuantaError>;
    /// Dispatch by total thread count (Metal clips, Vulkan computes groups).
    fn wave_dispatch_threads(&self, wave: &Wave, quarks: u32) -> Result<Pulse, QuantaError> {
        let groups = quarks.div_ceil(wave.workgroup_size[0]);
        self.wave_dispatch(wave, [groups, 1, 1])
    }
    /// Dispatch with group counts from a GPU buffer (GPU decides grid size).
    fn wave_dispatch_indirect(
        &self,
        wave: &Wave,
        buffer: u64,
        offset: u64,
    ) -> Result<Pulse, QuantaError>;

    // === Batch ===

    fn batch_begin(&self) -> Result<crate::Batch, QuantaError> {
        Err(QuantaError::invalid_param("batch dispatch not supported"))
    }

    // === Render ===

    fn pipeline_create(&self, desc: &crate::PipelineDesc) -> Result<Pipeline, QuantaError>;
    fn render_begin(&self, target: &Texture) -> Result<RenderPass, QuantaError>;
    fn render_end(&self, pass: RenderPass) -> Result<Pulse, QuantaError>;

    // === Sync ===

    fn pulse_wait(&self, pulse: &mut Pulse) -> Result<(), QuantaError>;
    fn pulse_poll(&self, pulse: &Pulse) -> bool;

    // === Queries ===

    /// Create a timestamp query set.
    fn query_set_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param("queries not supported"))
    }

    /// Read query results.
    fn query_set_read(
        &self,
        _handle: u64,
        _first: u32,
        _count: u32,
    ) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::invalid_param("queries not supported"))
    }

    // === Timestamps ===

    /// Create a timestamp query set with `count` slots.
    fn timestamp_query_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param("timestamps not supported"))
    }

    /// Write a timestamp at the given index in the query set.
    fn timestamp_write(&self, _query_handle: u64, _index: u32) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("timestamps not supported"))
    }

    /// Read timestamp values from a query set.
    fn timestamp_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::invalid_param("timestamps not supported"))
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
    fn async_compute_dispatch(
        &self,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<Pulse, QuantaError> {
        Err(QuantaError::invalid_param("async compute not supported"))
    }

    // === Timeline semaphores ===

    /// Create a timeline semaphore (monotonic u64 counter for multi-frame sync).
    fn timeline_create(&self) -> Result<Timeline, QuantaError> {
        Err(QuantaError::invalid_param(
            "timeline semaphores not supported",
        ))
    }

    /// Signal a timeline to the given value.
    fn timeline_signal(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "timeline semaphores not supported",
        ))
    }

    /// Block until a timeline reaches at least the given value.
    fn timeline_wait(&self, _timeline: &Timeline, _value: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
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
    fn resolve_texture(&self, _src_handle: u64, _dst_handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("MSAA resolve not supported"))
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
        Err(QuantaError::invalid_param("texture views not supported"))
    }

    /// Destroy a texture view.
    fn texture_view_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("texture views not supported"))
    }

    // === M2.6: Stencil read-back ===

    /// Read stencil buffer contents from a depth/stencil texture.
    fn stencil_read(&self, _texture: u64) -> Result<Vec<u8>, QuantaError> {
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param("multi-queue not supported"))
    }

    /// Submit a compute dispatch to a specific queue.
    fn queue_dispatch(
        &self,
        _queue: u64,
        _wave: &Wave,
        _groups: [u32; 3],
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("multi-queue not supported"))
    }

    /// Signal a semaphore from a queue.
    fn queue_signal(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("multi-queue not supported"))
    }

    /// Wait on a semaphore before executing more work on a queue.
    fn queue_wait(&self, _queue: u64, _semaphore: u64) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param("multi-queue not supported"))
    }

    // === M3.3: Occlusion queries ===

    /// Create an occlusion query set with `count` slots.
    fn occlusion_query_create(&self, _count: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
            "occlusion queries not supported",
        ))
    }

    /// Read results from an occlusion query set (fragment counts per slot).
    fn occlusion_query_read(&self, _handle: u64) -> Result<Vec<u64>, QuantaError> {
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
            "mesh shaders not yet implemented on this backend",
        ))
    }

    /// Dispatch `[gx, gy, gz]` mesh workgroups on the typed pipeline.
    /// The typed wrapper has already bounds-checked groups against
    /// `MAX_GROUP_COUNT`.
    fn mesh_dispatch(&self, _handle: u64, _groups: [u32; 3]) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
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
    fn build_acceleration_structure(&self, geometry: &[GeometryDesc]) -> Result<u64, QuantaError>;

    /// Create a ray tracing pipeline from shader stages.
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

    /// Create an indirect command buffer (GPU-driven draw/dispatch).
    fn indirect_buffer_create(&self, max_commands: u32) -> Result<u64, QuantaError>;

    /// Record a single dispatch command at `index` in the ICB.
    ///
    /// Snapshots the wave's pipeline + current bindings + group counts.
    /// `index` is the command position assigned by
    /// [`IndirectCommandBuffer::record_dispatch`](crate::IndirectCommandBuffer::record_dispatch);
    /// the typed wrapper enforces `index < max_commands`.
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
    fn icb_record_draw(
        &self,
        _handle: u64,
        _index: u32,
        _pipeline: u64,
        _vertex_count: u32,
        _instance_count: u32,
    ) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
            "render-path indirect command buffers not yet implemented on this backend",
        ))
    }

    /// Destroy a render bundle handle. Default no-ops so backends
    /// without an implementation don't error on `Drop`.
    fn render_bundle_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    /// Execute commands from an indirect command buffer.
    fn indirect_buffer_execute(&self, handle: u64, count: u32) -> Result<(), QuantaError>;

    /// Destroy an indirect command buffer.
    fn indirect_buffer_destroy(&self, handle: u64) -> Result<(), QuantaError>;

    // === M5.3: Bindless resources (steps 034 + 035) ===
    //
    // Bindless typed wrappers (`BindlessTextureArray`,
    // `BindlessBufferArray`) refine `Quanta.Bindless.Array` from the
    // Lean equivalence theorems (T7100-T7106).

    /// Allocate a bindless texture array with the given capacity.
    /// Default returns "not yet implemented"; backends override.
    fn bindless_texture_create(&self, _cap: u32) -> Result<u64, QuantaError> {
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
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
        Err(QuantaError::invalid_param(
            "variable rate shading not yet implemented on this backend",
        ))
    }

    /// Set the current shading rate. `rate_code` is the Verus
    /// encoding (0 = 1×1, 1 = 1×2, … 6 = 4×4). The typed wrapper
    /// has already validated the code.
    fn vrs_set_rate(&self, _handle: u64, _rate_code: u8) -> Result<(), QuantaError> {
        Err(QuantaError::invalid_param(
            "variable rate shading not yet implemented on this backend",
        ))
    }

    /// Destroy a VRS state. Default no-ops so backends without an
    /// implementation don't error on `Drop`.
    fn vrs_destroy(&self, _handle: u64) -> Result<(), QuantaError> {
        Ok(())
    }

    // ── Legacy bind-array creation (one-shot, no update path) ──────

    /// Create a bindless texture array (all textures accessible by index in shaders).
    fn bind_texture_array(&self, textures: &[u64]) -> Result<u64, QuantaError>;

    /// Create a bindless buffer array (all buffers accessible by index in shaders).
    fn bind_buffer_array(&self, buffers: &[u64]) -> Result<u64, QuantaError>;

    // === Debug ===

    /// Push a debug group label (shows in GPU profilers).
    fn debug_push(&self, _label: &str) {}

    /// Pop a debug group label.
    fn debug_pop(&self) {}
}

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::{
    Caps, Field, FieldUsage, Format, FormatCaps, GpuDevice, HostField, MappedField, MemoryTopology,
    QuantaError, ResourceState, Texture, TextureDesc, TextureView, TextureViewDesc, TimestampQuery,
};
#[cfg(feature = "compute")]
mod compute;

/// A GPU device handle. The main entry point for Quanta.
///
/// All GPU operations go through this type. No trait imports needed.
///
/// ```ignore
/// let gpu = quanta::init()?;
/// let field = gpu.field::<f32>(1_000_000)?;
/// let wave = vector_add(&gpu)?;
/// gpu.dispatch(&wave, 1_000_000)?;
/// ```
/// Cheap to clone — the handle is an `Arc` over the one device, so clones
/// share it (no second device, no re-init). Lets consumers (e.g.
/// `quanta-array`) store a `Gpu` alongside their data.
#[derive(Clone)]
pub struct Gpu {
    pub(crate) ctx: Arc<DeviceContext>,
}

/// Everything that exists exactly once per live device — the anchor
/// the per-device singletons were always documented against ("one
/// lane per device", "one pool per device, living beside the device
/// like the driver registries do"). `Gpu` is a thin cloneable handle
/// to this; the device registry hands out `Weak`s of it, so any two
/// `Gpu`s over the same physical device share one context and the
/// singleton contracts hold across independent `init()` calls.
///
/// The lane and pool keep their own `Arc`s because two seams extend
/// their lifetimes independently of the context: every lazy pulse
/// clones the lane, and `RenderBuilder` clones the pool.
///
/// Field order: children before `device`, as defense-in-depth (the
/// batch UAF is closed structurally — a parked `Batch` owns its
/// device Arc — but future fields holding device children without
/// their own keep-alive get the safe order for free).
pub(crate) struct DeviceContext {
    /// The deferred-dispatch pending lane — one per device (a single
    /// lane = a single submission order; see [`crate::api::deferred`]).
    /// Deferral is THE dispatch model, not a mode: every `dispatch`
    /// encodes here and the lane submits at the sync points.
    #[cfg(all(feature = "compute", feature = "std"))]
    pub(crate) pending: Arc<crate::api::deferred::PendingLane>,
    /// Compiled-wave cache — one per device, so `Gpu::wave` /
    /// `Gpu::wave_jit` dedup pipeline construction by kernel bytes
    /// (see [`crate::api::wave_cache`]). Isolated devices get their
    /// own by construction. No external seam clones it, so unlike the
    /// lane and the MSAA pool it needs no `Arc` of its own.
    #[cfg(all(feature = "compute", feature = "std"))]
    pub(crate) wave_cache: crate::api::wave_cache::WaveCache,
    /// Pool of builder-managed MSAA intermediates (`RenderBuilder::
    /// msaa`) — one per device, dropped (destroying every pooled
    /// texture) with the last `Gpu` clone. See [`crate::msaa_pool`]
    /// for the keying/lifetime story.
    #[cfg(all(feature = "render", feature = "std"))]
    pub(crate) msaa_pool: Arc<crate::MsaaPool>,
    /// The render-group intermediate pool — same lifetime story as
    /// `msaa_pool` (dies with the last `Gpu` clone; see
    /// [`crate::group_pool`]).
    #[cfg(all(feature = "render", feature = "std"))]
    pub(crate) group_pool: Arc<crate::GroupPool>,
    pub(crate) device: Arc<dyn GpuDevice>,
}

impl Gpu {
    #[allow(dead_code)]
    pub(crate) fn new(inner: Arc<dyn GpuDevice>) -> Self {
        Self {
            ctx: Arc::new(DeviceContext {
                #[cfg(all(feature = "compute", feature = "std"))]
                pending: Arc::new(crate::api::deferred::PendingLane::default()),
                #[cfg(all(feature = "compute", feature = "std"))]
                wave_cache: crate::api::wave_cache::WaveCache::default(),
                #[cfg(all(feature = "render", feature = "std"))]
                msaa_pool: Arc::new(crate::MsaaPool::default()),
                #[cfg(all(feature = "render", feature = "std"))]
                group_pool: Arc::new(crate::GroupPool::default()),
                device: inner,
            }),
        }
    }

    /// Internal device-handle accessor for the `quanta-render` extension
    /// crate. The render methods live on `RenderGpu` (a sibling-crate
    /// extension trait) and reach the shared `GpuDevice` through this.
    /// Not part of the stable public surface — hidden from docs so the
    /// compute crate's documented surface stays render-free.
    #[doc(hidden)]
    pub fn device_handle(&self) -> &Arc<dyn GpuDevice> {
        &self.ctx.device
    }

    /// Internal accessor for the builder-managed MSAA intermediate
    /// pool, for the `quanta-render` extension crate (same role as
    /// [`Gpu::device_handle`]). Not part of the stable public surface.
    #[cfg(all(feature = "render", feature = "std"))]
    #[doc(hidden)]
    pub fn __msaa_pool(&self) -> &Arc<crate::MsaaPool> {
        &self.ctx.msaa_pool
    }

    /// Internal accessor for the render-group intermediate pool, for
    /// the `quanta-render` extension crate. Not part of the stable
    /// public surface.
    #[cfg(all(feature = "render", feature = "std"))]
    #[doc(hidden)]
    pub fn __group_pool(&self) -> &Arc<crate::GroupPool> {
        &self.ctx.group_pool
    }

    /// Test-support: snapshot of the driver's resource-registry sizes.
    /// Lifecycle tests compare snapshots around create+drop cycles to
    /// prove destroy paths free their registry entries.
    #[doc(hidden)]
    pub fn debug_registry_counts(&self) -> crate::RegistryCounts {
        self.ctx.device.debug_registry_counts()
    }

    /// Test-support: drop every cache-held compiled pipeline,
    /// returning how many entries the cache released. Cached
    /// pipelines are real registry entries that linger by design, so
    /// absolute-count tests drain before asserting. Entries still
    /// referenced by outstanding `Wave`s free when those drop.
    #[doc(hidden)]
    pub fn __wave_cache_drain(&self) -> usize {
        #[cfg(all(feature = "compute", feature = "std"))]
        {
            self.ctx.wave_cache.drain()
        }
        #[cfg(not(all(feature = "compute", feature = "std")))]
        {
            0
        }
    }

    // === Device info ===

    pub fn caps(&self) -> &Caps {
        self.ctx.device.caps()
    }

    pub fn nuclei(&self) -> u32 {
        self.caps().nuclei
    }

    pub fn protons_per_nucleus(&self) -> u32 {
        self.caps().protons_per_nucleus
    }

    pub fn quarks_per_proton(&self) -> u32 {
        self.caps().quarks_per_proton
    }

    pub fn total_quarks(&self) -> u32 {
        self.caps().total_quarks()
    }

    /// Memory topology of the active device: whether host and device
    /// share one physical memory pool ([`MemoryTopology::Unified`]) or
    /// host↔device transfers cross a real interconnect
    /// ([`MemoryTopology::Discrete`]). Consumers with a CPU-vs-GPU
    /// routing decision (query planners, schedulers) branch on this
    /// instead of hardcoding one topology's crossover point.
    pub fn memory_topology(&self) -> MemoryTopology {
        self.caps().memory_topology
    }

    // === Feature support queries (step 063 slice 20) ===
    //
    // Lets callers check whether a feature has a native lowering
    // on the active backend before submitting work. The
    // alternative is trial-and-error: submit and inspect the
    // returned `QuantaError::NotSupported`. These queries return
    // exactly the bool the device's render path will gate on.

    /// Whether the active backend can lower
    /// `RenderOp::SetShadingRate` to a native VRS path. `false`
    /// when the extension / device-family is missing — the typed
    /// API still accepts VRS state, but the render encoder will
    /// surface NotSupported at submit time.
    pub fn supports_vrs(&self) -> bool {
        self.ctx.device.supports_variable_rate_shading()
    }

    /// Whether the active backend can build acceleration structures
    /// and dispatch ray tracing.
    pub fn supports_ray_tracing(&self) -> bool {
        self.ctx.device.supports_ray_tracing()
    }

    /// Whether the active backend can create mesh-shader pipelines.
    pub fn supports_mesh_shaders(&self) -> bool {
        self.ctx.device.supports_mesh_shaders()
    }

    /// Whether the active backend can create tessellation pipelines
    /// (Vulkan tessellationShader feature / Metal Apple GPU
    /// family 4+).
    pub fn supports_tessellation(&self) -> bool {
        self.ctx.device.supports_tessellation()
    }

    /// Whether the active backend can create sparse textures with
    /// residency control.
    pub fn supports_sparse_residency(&self) -> bool {
        self.ctx.device.supports_sparse_residency()
    }

    /// Whether the active backend can lower the cooperative-matrix ops to
    /// native tensor-core / SIMD-group-matrix instructions (Metal Apple GPU
    /// family 7+; Vulkan `VK_KHR_cooperative_matrix`, not yet wired). The
    /// software lane reports `false`.
    pub fn supports_cooperative_matrix(&self) -> bool {
        self.ctx.device.supports_cooperative_matrix()
    }

    /// Whether the active backend can run kernels that use 64-bit
    /// floats. The software lane and llvmpipe support f64; Metal and
    /// the Broadcom V3D GPU do not.
    pub fn supports_f64(&self) -> bool {
        self.ctx.device.supports_f64()
    }

    /// Whether the active backend can run kernels that use 64-bit
    /// integers. The software lane and llvmpipe support i64/u64; the
    /// Broadcom V3D GPU does not.
    pub fn supports_i64(&self) -> bool {
        self.ctx.device.supports_i64()
    }

    /// Whether the active backend can run kernels over narrow-int
    /// (`u8` / `i8` / `u16` / `i16`) fields at native 1-/2-byte stride.
    /// One query for all four — they gate together everywhere this
    /// stack runs. The software lane and Metal always support them;
    /// Vulkan requires the 8-bit AND 16-bit storage-buffer device
    /// features (universal on 1.2+ desktop drivers and lavapipe);
    /// WebGPU does not — WGSL has no narrow storage types, and such
    /// kernels fail `wave_jit` with `NotSupported` carrying the
    /// capability-table reason. The query is the polite pre-check;
    /// validation is the enforcement.
    pub fn supports_narrow_int(&self) -> bool {
        self.ctx.device.supports_narrow_int()
    }

    /// Whether the active backend can run kernels that use subgroup
    /// (warp / SIMD-group) arithmetic operations (subgroup reduce /
    /// scan). The software lane, Metal, and llvmpipe support them;
    /// the Broadcom V3D GPU does not (its Mesa backend cannot lower
    /// `OpGroupNonUniform*` arithmetic and aborts at pipeline
    /// creation). Kernels with a subgroup-free fallback should select
    /// on this before building the wave.
    pub fn supports_subgroups(&self) -> bool {
        self.ctx.device.supports_subgroups()
    }

    /// Whether narrow-float buffers (bf16 / fp8) on the active backend
    /// use the portable u32-slot layout (one element per 32-bit word)
    /// instead of native stride. Only WebGPU: WGSL storage buffers
    /// cannot hold 16-/8-bit array elements. Hosts holding tight
    /// bf16/fp8 data (one element per 2/1 bytes — the layout Metal,
    /// Vulkan and the CPU consume directly) must expand it
    /// one-element-per-word before binding on such a backend.
    pub fn narrow_storage_u32_slot(&self) -> bool {
        self.ctx.device.narrow_storage_u32_slot()
    }

    /// Hardware-supported VRS shading rates as `(width, height)`
    /// pairs. Empty when VRS isn't supported.
    pub fn supported_shading_rates(&self) -> Vec<(u32, u32)> {
        self.ctx.device.supported_shading_rates()
    }

    /// Whether [`Texture::native_handle`](crate::Texture::native_handle)
    /// and [`Field::native_handle`](crate::Field::native_handle)
    /// export a real backend object on the active backend. `false`
    /// on the CPU software driver (no native object exists) and on
    /// backends whose export path isn't wired yet; those return
    /// `NotSupported` from `native_handle`.
    pub fn supports_native_handle_export(&self) -> bool {
        self.ctx.device.supports_native_handle_export()
    }

    /// Whether the active backend can create presentation surfaces
    /// (`Gpu::create_surface` under the `render` feature). Backends
    /// without a present path return `NotSupported` at
    /// `create_surface` time.
    pub fn supports_surface_present(&self) -> bool {
        self.ctx.device.supports_surface_present()
    }

    /// Whether compute kernels can bind textures on the active backend
    /// (`&Texture2D` sampled reads / `&mut Texture2D<f32>` storage writes).
    /// True on Metal, CPU, and native Vulkan; false on WebGPU. Tests skip the
    /// live-dispatch path when this is false.
    pub fn supports_compute_textures(&self) -> bool {
        self.ctx.device.supports_compute_textures()
    }

    pub fn name(&self) -> &str {
        &self.caps().name
    }

    // === Fields (typed GPU memory) ===

    /// Allocate a GPU field with default compute usage (storage + transfer).
    ///
    /// This is the most common allocation — use it for kernel inputs and outputs.
    /// For specific usage flags, see [`field_with_usage`](Gpu::field_with_usage).
    pub fn field<T: Copy>(&self, count: usize) -> Result<Field<T>, QuantaError> {
        self.field_with_usage(count, FieldUsage::default_compute())
    }

    /// Allocate a GPU field with explicit usage flags.
    pub fn field_with_usage<T: Copy>(
        &self,
        count: usize,
        usage: FieldUsage,
    ) -> Result<Field<T>, QuantaError> {
        let size = count * size_of::<T>();
        let handle = self.ctx.device.field_alloc(size, usage)?;
        Ok(Field {
            handle,
            count,
            device: self.ctx.device.clone(),
            #[cfg(all(feature = "compute", feature = "std"))]
            lane: Arc::clone(&self.ctx.pending),
            _marker: PhantomData,
        })
    }

    /// Create a GPU buffer permanently mapped into CPU address space.
    ///
    /// Enables zero-copy writes: data written to the returned `MappedField`
    /// is immediately visible to the GPU (on unified memory architectures)
    /// or automatically synchronized on the next dispatch.
    pub fn field_mapped<T: Copy>(&self, count: usize) -> Result<MappedField<T>, QuantaError> {
        let size = count * size_of::<T>();
        let usage = FieldUsage::default_compute();
        let (handle, ptr) = self.ctx.device.field_create_mapped(size, usage)?;
        Ok(MappedField {
            handle,
            ptr,
            count,
            device: self.ctx.device.clone(),
            _marker: PhantomData,
        })
    }

    /// Import caller-owned host memory as a read-only field — the
    /// inverse of [`field_mapped`](Self::field_mapped): the host
    /// already owns the memory (typically an mmap'd file region) and
    /// the device gets a view of it, with zero copies where the
    /// backend supports import
    /// ([`supports_host_import`](Self::supports_host_import)).
    ///
    /// Contract: the base pointer AND the byte length must be
    /// multiples of [`host_import_alignment`](Self::host_import_alignment)
    /// (violations return `InvalidParam` — never a silent copy).
    /// mmap'd regions satisfy the base by construction; pass the
    /// page-padded slice and keep your logical element count in your
    /// own metadata. On backends without an import path the same call
    /// succeeds through one staged copy and
    /// [`HostField::is_imported`] reports `false`.
    pub fn field_from_host<'a, T: Copy>(
        &self,
        data: &'a [T],
    ) -> Result<HostField<'a, T>, QuantaError> {
        // Safety of the raw call: `data` is a live shared borrow for
        // 'a, and HostField carries 'a, so the region outlives the
        // field and stays immutable while it exists.
        unsafe { self.field_from_host_raw(data.as_ptr(), data.len()) }
    }

    /// Raw-pointer variant of [`field_from_host`](Self::field_from_host)
    /// for owners whose lifetime the borrow checker can't see (e.g. a
    /// long-lived process owning an mmap).
    ///
    /// # Safety
    ///
    /// The region `[ptr, ptr + count * size_of::<T>())` must be valid,
    /// initialized, and unmutated for the whole life of the returned
    /// field AND of every in-flight wave that binds it. The alignment
    /// contract of `field_from_host` applies unchanged.
    pub unsafe fn field_from_host_ptr<T: Copy>(
        &self,
        ptr: *const T,
        count: usize,
    ) -> Result<HostField<'static, T>, QuantaError> {
        unsafe { self.field_from_host_raw(ptr, count) }
    }

    unsafe fn field_from_host_raw<'a, T: Copy>(
        &self,
        ptr: *const T,
        count: usize,
    ) -> Result<HostField<'a, T>, QuantaError> {
        if count == 0 {
            return Err(QuantaError::invalid_param(
                "cannot import an empty host region",
            ));
        }
        let bytes_ptr = ptr as *const u8;
        let len = count * size_of::<T>();
        let (handle, imported) = if self.ctx.device.supports_host_import() {
            (self.ctx.device.field_import_host(bytes_ptr, len)?, true)
        } else {
            // Staged-copy fallback: same API, exactly one copy, and
            // the difference stays queryable via `is_imported`.
            let usage = FieldUsage::READ
                .union(FieldUsage::COMPUTE)
                .union(FieldUsage::TRANSFER);
            let handle = self.ctx.device.field_alloc(len, usage)?;
            let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, len) };
            self.ctx.device.field_write_bytes(handle, bytes)?;
            (handle, false)
        };
        Ok(HostField {
            handle,
            count,
            imported,
            device: self.ctx.device.clone(),
            _borrow: PhantomData,
        })
    }

    /// Whether [`field_from_host`](Self::field_from_host) has a
    /// native zero-copy path on the active backend. `false` means the
    /// call still succeeds but stages one copy.
    pub fn supports_host_import(&self) -> bool {
        self.ctx.device.supports_host_import()
    }

    /// The granularity the host-import contract checks pointers and
    /// lengths against (Metal: the VM page size; Vulkan:
    /// `minImportedHostPointerAlignment`). `None` when the backend
    /// has no import path.
    pub fn host_import_alignment(&self) -> Option<usize> {
        self.ctx.device.host_import_alignment()
    }

    // === Textures ===

    /// Create a texture from a descriptor (full control).
    pub fn create_texture(&self, desc: &TextureDesc) -> Result<Texture, QuantaError> {
        let mut tex = self.ctx.device.texture_create(desc)?;
        tex.device = Some(self.ctx.device.clone());
        Ok(tex)
    }

    /// Create a simple RGBA8 texture (convenience).
    pub fn texture(&self, width: u32, height: u32) -> Result<Texture, QuantaError> {
        self.create_texture(&TextureDesc {
            width,
            height,
            format: Format::RGBA8,
            ..TextureDesc::default()
        })
    }

    /// Create a reusable sampler.
    pub fn sampler(
        &self,
        desc: &crate::texture::SamplerDesc,
    ) -> Result<crate::Sampler, QuantaError> {
        let mut sampler = self.ctx.device.sampler_create(desc)?;
        sampler.device = Some(self.ctx.device.clone());
        Ok(sampler)
    }

    // === Sync ===

    // === Timeline semaphores ===

    /// Create a timeline semaphore for multi-frame synchronization.
    pub fn timeline_create(&self) -> Result<crate::Timeline, QuantaError> {
        self.ctx.device.timeline_create()
    }

    /// Signal a timeline to the given value.
    pub fn timeline_signal(
        &self,
        timeline: &crate::Timeline,
        value: u64,
    ) -> Result<(), QuantaError> {
        self.ctx.device.timeline_signal(timeline, value)
    }

    /// Block until a timeline reaches at least the given value.
    pub fn timeline_wait(&self, timeline: &crate::Timeline, value: u64) -> Result<(), QuantaError> {
        self.ctx.device.timeline_wait(timeline, value)
    }

    // === Barriers ===

    /// Full pipeline barrier — wait for all prior GPU work to complete.
    ///
    /// This is a heavyweight synchronization point. Prefer `barrier_field`
    /// or `barrier_texture` for fine-grained resource transitions.
    pub fn barrier(&self) -> Result<(), QuantaError> {
        self.ctx.device.barrier()
    }

    /// Transition a field between resource states.
    ///
    /// On Vulkan, this inserts pipeline barriers with correct stage/access masks.
    /// On Metal, this is a no-op (automatic hazard tracking).
    pub fn barrier_field<T: Copy>(
        &self,
        field: &Field<T>,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.ctx.device.barrier_buffer(field.handle(), from, to)
    }

    /// Transition a texture between resource states.
    ///
    /// On Vulkan, this inserts an image layout transition.
    /// On Metal, this is a no-op (automatic hazard tracking).
    pub fn barrier_texture(
        &self,
        texture: &Texture,
        from: ResourceState,
        to: ResourceState,
    ) -> Result<(), QuantaError> {
        self.ctx.device.barrier_texture(texture, from, to)
    }

    // === Host synchronization ===

    /// Block until every operation submitted to this device has completed
    /// on the GPU.
    ///
    /// Dispatch and render submissions are asynchronous: the returned
    /// [`Pulse`](crate::Pulse) is still in flight when the call returns.
    /// Waiting on the pulse syncs that one submission; `wait_idle` drains
    /// everything submitted so far. Use one of the two before any CPU-side
    /// read of a GPU-written target — `Texture::read` and `Field::read` do
    /// no implicit sync. Presenting an acquired surface frame needs no
    /// wait: same-queue ordering covers it.
    pub fn wait_idle(&self) -> Result<(), QuantaError> {
        // Deferred work that was never submitted is invisible to the
        // driver's drain — flush the pending lane first so "everything
        // submitted so far" includes everything *dispatched* so far.
        #[cfg(all(feature = "compute", feature = "std"))]
        self.ctx.pending.flush_and_wait()?;
        self.ctx.device.wait_idle()
    }

    /// Submit all deferred dispatches and block until they complete.
    /// The explicit sync point for consumers that bypass [`Pulse`]s —
    /// e.g. an external reader of a
    /// [`Field::native_handle`](crate::Field::native_handle) export.
    /// A no-op when nothing is pending.
    #[cfg(all(feature = "compute", feature = "std"))]
    pub fn flush(&self) -> Result<(), QuantaError> {
        self.ctx.pending.flush_and_wait()
    }

    /// Complete all deferred compute work, for the sibling extension
    /// crates (`quanta-render`) whose submissions bypass the pending
    /// lane: a render pass sampling a compute-written field must not
    /// overtake the encoded producer. Exists on every feature combo
    /// (no-op without `compute` + `std`) so callers need no cfg. Not
    /// part of the stable public surface.
    #[doc(hidden)]
    pub fn __flush_pending(&self) -> Result<(), QuantaError> {
        #[cfg(all(feature = "compute", feature = "std"))]
        self.ctx.pending.flush_and_wait()?;
        Ok(())
    }

    // === Timestamps ===

    /// Create a `TimestampQuery` object wrapping a query set handle.
    pub fn timestamp_query(&self, count: u32) -> Result<TimestampQuery, QuantaError> {
        let handle = self.ctx.device.timestamp_query_create(count)?;
        Ok(TimestampQuery { handle, count })
    }

    /// Write a timestamp at the given index in the query set.
    pub fn write_timestamp(&self, query: &TimestampQuery, index: u32) -> Result<(), QuantaError> {
        self.ctx.device.timestamp_write(query.handle, index)
    }

    /// Read all timestamps from a query set.
    pub fn read_timestamps(&self, query: &TimestampQuery) -> Result<Vec<u64>, QuantaError> {
        self.ctx.device.timestamp_query_read(query.handle)
    }

    /// Convert raw timestamp ticks to nanoseconds using the device frequency.
    pub fn timestamp_to_ns(&self, ticks: u64) -> u64 {
        let freq = self.ctx.device.timestamp_frequency();
        if freq == 0 || freq == 1_000_000_000 {
            ticks
        } else {
            // ticks * 1_000_000_000 / freq, but avoid overflow with u128
            ((ticks as u128 * 1_000_000_000) / freq as u128) as u64
        }
    }

    // === M2.2: Format capability queries ===

    /// Query what a given format can do on this device.
    pub fn format_caps(&self, format: Format) -> FormatCaps {
        self.ctx.device.format_caps(format)
    }

    // === M2.3: Texture views ===

    /// Create a view into an existing texture (sub-range of mips/layers, optional format reinterpret).
    pub fn texture_view_create(
        &self,
        texture: &Texture,
        desc: &TextureViewDesc,
    ) -> Result<TextureView, QuantaError> {
        let handle = self
            .ctx
            .device
            .texture_view_create(texture.handle(), desc)?;
        Ok(TextureView {
            handle,
            device: self.ctx.device.clone(),
            live: true,
        })
    }

    /// Destroy a texture view.
    ///
    /// Equivalent to dropping the view; kept for explicit error
    /// propagation. Disarms the view's Drop so the handle is
    /// destroyed exactly once.
    pub fn texture_view_destroy(&self, mut view: TextureView) -> Result<(), QuantaError> {
        view.live = false;
        self.ctx.device.texture_view_destroy(view.handle())
    }

    // === M5.1: Sparse textures ===

    /// Allocate a typed [`SparseTexture`](crate::SparseTexture) (virtual
    /// texture whose physical memory is allocated lazily per tile).
    /// Steps 030 + 031.
    ///
    /// Shared compute/render: compute kernels read sparse textures
    /// (`texture_load_2d`), so this stays on the compute surface.
    ///
    /// Backends without sparse-binding support (`VK_EXT_sparse_binding`,
    /// Apple Silicon) return `NotSupported` here so user code can branch.
    pub fn sparse_texture(&self, desc: &TextureDesc) -> Result<crate::SparseTexture, QuantaError> {
        let handle = self.ctx.device.sparse_texture_create(desc)?;
        Ok(crate::SparseTexture {
            handle,
            width: desc.width,
            height: desc.height,
            device: self.ctx.device.clone(),
            live: true,
        })
    }

    // === Debug ===

    /// Push a debug group label (visible in GPU profilers like Xcode GPU Capture).
    pub fn debug_push(&self, label: &str) {
        self.ctx.device.debug_push(label);
    }

    /// Pop a debug group label.
    pub fn debug_pop(&self) {
        self.ctx.device.debug_pop();
    }
}

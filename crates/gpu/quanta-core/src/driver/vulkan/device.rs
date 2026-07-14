//! VulkanDevice struct definition, discovery, and internal helpers.

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::{Caps, GpuDevice, Pulse, QuantaError, Vendor};
use std::collections::HashMap;
use std::sync::{Mutex, RwLock};

use super::ffi;

/// Pack a per-binding descriptor-kind list into a cache key: 2 bits per
/// binding (0 = storage buffer, 1 = storage image, 2 = sampled image). Two
/// layouts with the same length but different kinds get distinct keys, so a
/// buffer-only layout never aliases a mixed one.
///
/// Compute-only: `spirv_meta` (the descriptor-kind reflection this keys
/// on) is gated on `compute`, and only the compute pipeline path builds
/// mixed layouts. A `vulkan,render`-without-`compute` build carries no
/// descriptor reflection, so this and the pieces below are gated to match.
#[cfg(feature = "compute")]
pub(super) fn layout_signature(kinds: &[crate::driver::spirv_meta::DescriptorKind]) -> u64 {
    use crate::driver::spirv_meta::DescriptorKind;
    // 32 bindings × 2 bits fit in a u64; our layouts use ≤ 16.
    let mut sig: u64 = 0;
    for (i, kind) in kinds.iter().take(32).enumerate() {
        let bits: u64 = match kind {
            DescriptorKind::StorageBuffer => 0,
            DescriptorKind::StorageImage => 1,
            DescriptorKind::SampledImage => 2,
        };
        sig |= bits << (i * 2);
    }
    // Fold in the length so a trailing run of storage buffers (bits 0) is not
    // confused with a shorter list.
    sig | ((kinds.len() as u64) << 40)
}

/// Vulkan-backed GPU device.
pub struct VulkanDevice {
    pub(super) instance: ffi::VkInstance,
    pub(super) physical_device: ffi::VkPhysicalDevice,
    pub(super) device: ffi::VkDevice,
    pub(super) queue: ffi::VkQueue,
    #[allow(dead_code)]
    pub(super) queue_family: u32,
    pub(super) command_pool: ffi::VkCommandPool,
    pub(super) pipeline_cache: ffi::VkPipelineCache,
    pub(super) caps: Caps,
    // Read by the compute-gated dispatch path only.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(super) max_push_constants_size: u32,
    /// `VkPhysicalDeviceLimits.maxVertexInputAttributes` — the device's
    /// hard cap on vertex attribute locations. Cached at discovery so
    /// `pipeline_create` can reject an over-limit descriptor with a NAMED
    /// error BEFORE calling `vkCreateGraphicsPipelines`. Desktop drivers
    /// report 32 and mask the problem; Broadcom V3D reports 16, where a
    /// failing pipeline build has been observed to corrupt the process
    /// heap — so the cheap pre-check is the real defense (step 085).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(super) max_vertex_input_attributes: u32,
    // Resource storage — RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    pub(super) buffers: RwLock<HashMap<u64, VkBuffer>>,
    pub(super) textures: RwLock<HashMap<u64, VkTexture>>,
    pub(super) compute_pipelines: RwLock<HashMap<u64, VkComputePipeline>>,
    pub(super) render_pipelines: RwLock<HashMap<u64, VkRenderPipeline>>,
    pub(super) samplers: RwLock<HashMap<u64, ffi::VkSampler>>,
    /// Render-path sampler cache keyed by the FULL `SamplerDesc`.
    ///
    /// The render encoder used to create a fresh VkSampler for every
    /// `SetSampler` op on every `render_end` — one per textured draw per
    /// frame, never destroyed — which exhausts the device's
    /// `maxSamplerAllocationCount` pool within minutes (65,536 on v3dv)
    /// and then every glyph/textured draw fails. Sampler state is a pure
    /// function of the descriptor, so a distinct-desc-keyed cache creates
    /// each sampler exactly once and reuses it across draws and frames;
    /// the cache is bounded by the number of DISTINCT descriptors, not by
    /// draw count. Populated lazily under a read-then-upgrade path and
    /// drained at device teardown.
    #[cfg(feature = "render")]
    pub(super) render_sampler_cache: RwLock<HashMap<crate::texture::SamplerDesc, ffi::VkSampler>>,
    /// Standalone image views created via texture_view_create (not tied to a full VkTexture).
    pub(super) image_views: RwLock<HashMap<u64, ffi::VkImageView>>,
    pub(super) query_pools: RwLock<HashMap<u64, VkQueryPool>>,
    pub(super) queues: RwLock<HashMap<u64, ffi::VkQueue>>,
    /// WSI extension procs; `None` means no present support here.
    #[cfg(feature = "render")]
    pub(super) surface_procs: Option<super::surface::SurfaceProcs>,
    #[cfg(feature = "render")]
    pub(super) vk_surfaces: RwLock<HashMap<u64, super::surface::VkSurfaceEntry>>,
    #[cfg(feature = "render")]
    pub(super) vk_surface_frames: RwLock<HashMap<u64, super::surface::VkSurfaceFrame>>,
    pub(super) next_handle: AtomicU64,
    /// Pool of reusable command buffers — Arc<Mutex> for sharing with Pulse closures.
    pub(super) cmd_buffer_pool: std::sync::Arc<Mutex<Vec<ffi::VkCommandBuffer>>>,
    /// Pool of reusable descriptor pools — avoids create/destroy per dispatch.
    pub(super) descriptor_pool_cache: Mutex<Vec<ffi::VkDescriptorPool>>,
    /// Pool of reusable staging buffers — avoids alloc/free per texture upload.
    pub(super) staging_pool: Mutex<Vec<(ffi::VkBuffer, ffi::VkDeviceMemory, usize)>>,
    /// Cache of descriptor set layouts keyed by a per-binding descriptor-kind
    /// signature (2 bits per binding) — a buffer-only layout and a mixed
    /// buffer+image layout of the same length must not collide.
    pub(super) layout_cache: Mutex<HashMap<u64, ffi::VkDescriptorSetLayout>>,
    /// The one compute sampler this device binds for every `&Texture2D` sampled
    /// read (`texture_sample_2d`). Contract: NEAREST min/mag/mip,
    /// CLAMP_TO_EDGE, no anisotropy/compare, UNNORMALIZED coordinates — chosen
    /// so a GPU `sample()` matches the CPU executor's nearest+clamp texel fetch
    /// exactly, and to satisfy Vulkan's unnormalized-sampler rules. Lazily
    /// created on first sampled-read dispatch and destroyed at teardown. This is
    /// deliberately NOT the render sampler cache (which is keyed by a full
    /// `SamplerDesc` and has no unnormalized field); the compute contract is
    /// fixed, not descriptor-driven. `null_handle()` until first use.
    #[cfg(feature = "compute")]
    pub(super) compute_sampler: Mutex<ffi::VkSampler>,
    /// Indirect command buffers (steps 032 + 033). Stores recorded
    /// dispatches that `indirect_buffer_execute` replays sequentially
    /// on the same compute path used by `wave_dispatch`. The Lean
    /// `T7000` equivalence theorem is parametric in the per-command
    /// transformer, so this list-of-dispatches refinement satisfies
    /// the proof contract on every Vulkan implementation.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(super) icbs: RwLock<HashMap<u64, VkIcb>>,
    /// Render-path Indirect Command Buffers (steps 032 + 033). One
    /// pre-allocated secondary VkCommandBuffer per command slot,
    /// recorded with VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT
    /// against the pipeline's compatible render pass; replayed via
    /// vkCmdExecuteCommands inside an active render pass.
    pub(super) render_bundles: RwLock<HashMap<u64, VulkanRenderBundle>>,
    /// Bindless texture arrays (steps 034 + 035). MVP: software
    /// table of texture handles; perf upgrade via
    /// VK_EXT_descriptor_indexing is a follow-up.
    pub(super) bindless_textures: RwLock<HashMap<u64, VulkanBindlessArray>>,
    /// Bindless buffer arrays (steps 034 + 035). MVP: software
    /// table of buffer handles.
    pub(super) bindless_buffers: RwLock<HashMap<u64, VulkanBindlessArray>>,
    /// Tessellation pipeline state (steps 022 + 023). MVP: software
    /// table of (topology, outer factors, inner factors). Vulkan
    /// hardware tessellation requires enabling the `tessellationShader`
    /// device feature at create time and rebuilding pipeline-create
    /// info to include `VkPipelineTessellationStateCreateInfo` plus
    /// TCS+TES SPIR-V modules — that's a future commit. The proof
    /// contract from `Quanta.Tessellation` holds today.
    pub(super) tess_pipelines: RwLock<HashMap<u64, VulkanTessPipeline>>,
    /// Mesh-shader pipeline state (steps 024 + 025). MVP: software
    /// lifecycle table. Native `vkCmdDrawMeshTasksEXT` integration
    /// is deferred to the render-pipeline rebuild that lands with
    /// 062/063.
    pub(super) mesh_pipelines: RwLock<HashMap<u64, VulkanMeshPipeline>>,
    /// VRS states (steps 028 + 029). MVP: software lifecycle.
    pub(super) vrs_states: RwLock<HashMap<u64, VulkanVrsState>>,
    /// Function pointer for `vkCmdSetFragmentShadingRateKHR`,
    /// resolved via `vkGetDeviceProcAddr` at device creation when
    /// `VK_KHR_fragment_shading_rate` was enabled. `None` means the
    /// extension is unavailable; the render encoder surfaces this as
    /// `NotSupported` from the `RenderOp::SetShadingRate` arm
    /// (step 063 native VRS lowering).
    pub(super) vrs_set_rate_fn: Option<ffi::PfnVkCmdSetFragmentShadingRateKHR>,
    /// Function pointer for `vkCmdDrawMeshTasksEXT`. Resolved when
    /// `VK_EXT_mesh_shader` is enabled; `None` otherwise. Mesh
    /// pipelines surface NotSupported when this is None
    /// (step 063 native mesh-shader scaffolding).
    pub(super) mesh_draw_fn: Option<ffi::PfnVkCmdDrawMeshTasksEXT>,
    /// Function pointer for `vkCmdTraceRaysKHR`. Resolved when
    /// both `VK_KHR_ray_tracing_pipeline` and
    /// `VK_KHR_acceleration_structure` are enabled; `None`
    /// otherwise. Ray-tracing dispatch surfaces NotSupported when
    /// None (step 063 native ray-tracing scaffolding).
    pub(super) trace_rays_fn: Option<ffi::PfnVkCmdTraceRaysKHR>,
    /// Acceleration-structure build proc addresses. All four are
    /// resolved when `VK_KHR_acceleration_structure` is enabled;
    /// `None` otherwise. Stored together because the build path
    /// always needs the whole set. Step 063 slice 15.
    pub(super) accel_create_fn: Option<ffi::PfnVkCreateAccelerationStructureKHR>,
    pub(super) accel_destroy_fn: Option<ffi::PfnVkDestroyAccelerationStructureKHR>,
    // Read only by the render-gated ray-tracing build path (step 085).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(super) accel_build_sizes_fn: Option<ffi::PfnVkGetAccelerationStructureBuildSizesKHR>,
    pub(super) accel_build_fn: Option<ffi::PfnVkCmdBuildAccelerationStructuresKHR>,
    /// Whether `VkPhysicalDeviceFeatures.tessellationShader` is
    /// available on this physical device. Cached at discovery so
    /// `tessellation_pipeline_create` can surface a clean
    /// NotSupported without re-querying. Step 063 slice 6.
    pub(super) tessellation_feature: bool,
    /// Supported shading-rate fragment sizes returned by
    /// `vkGetPhysicalDeviceFragmentShadingRatesKHR`. Empty when the
    /// VRS extension isn't enabled. The render encoder validates the
    /// requested rate against this list before calling
    /// `vkCmdSetFragmentShadingRateKHR`, surfacing a clear
    /// "rate unsupported on this device" error at submit time
    /// instead of a generic Vulkan validation message.
    /// Step 063 slice 14.
    pub(super) supported_shading_rates: Vec<(u32, u32)>,
    /// Whether `VkPhysicalDeviceFeatures.sparseBinding` is
    /// available on this physical device, AND the chosen queue
    /// family supports `VK_QUEUE_SPARSE_BINDING_BIT`. Caching
    /// both at discovery means `sparse_texture_create` no longer
    /// calls `vkGetPhysicalDeviceFeatures` per request, and
    /// future native bind-sparse work can gate on a single bool.
    /// Step 063 slice 16.
    pub(super) sparse_binding_supported: bool,
    /// Whether `VkPhysicalDeviceFeatures.shaderFloat64` is available
    /// on this physical device AND was enabled at `vkCreateDevice`.
    /// A kernel that uses `f64` emits the `Float64` SPIR-V capability,
    /// which is only valid when this feature is enabled — otherwise
    /// `vkCreateComputePipelines` fails with `VK_ERROR_UNKNOWN`. The
    /// Broadcom V3D GPU reports `false`; llvmpipe reports `true`.
    pub(super) shader_float64_supported: bool,
    /// Whether `VkPhysicalDeviceFeatures.shaderInt64` is available and
    /// was enabled at `vkCreateDevice`. Kernels using `i64`/`u64` emit
    /// the `Int64` capability, valid only when this feature is enabled.
    pub(super) shader_int64_supported: bool,
    /// Whether `VkPhysicalDeviceSubgroupProperties.supportedOperations`
    /// advertises `VK_SUBGROUP_FEATURE_ARITHMETIC_BIT` for the compute
    /// stage. Kernels using subgroup reduce/scan emit
    /// `OpGroupNonUniform*` arithmetic, which Broadcom V3D cannot lower
    /// (the driver aborts at pipeline creation); llvmpipe supports it.
    /// Queried at discovery via `vkGetPhysicalDeviceProperties2`.
    pub(super) subgroup_arithmetic_supported: bool,
    /// `vkCmdDispatchBase` (core Vulkan 1.1), resolved at device
    /// creation; `None` on 1.0-only implementations. Used by the
    /// folded 1D-dispatch path (`wave_dispatch_threads_impl`) to issue
    /// the remainder row at a non-zero base workgroup so oversized
    /// thread-count dispatches can exceed
    /// `maxComputeWorkGroupCount[0]` without waste threads.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(super) dispatch_base_fn: Option<ffi::PfnVkCmdDispatchBase>,
    /// Per-tile memory bindings for sparse textures. Key is
    /// `(texture_handle, mip, tile_x, tile_y)`; value is the
    /// `VkDeviceMemory` allocation that backs that tile after
    /// `vkQueueBindSparse`. `sparse_unmap_tile` uses the entry to
    /// unbind + free; `Drop` walks the table to release leaked
    /// allocations. Step 063 slice 22.
    pub(super) sparse_tile_bindings: RwLock<HashMap<(u64, u32, u32, u32), ffi::VkDeviceMemory>>,
    /// Whether `bufferDeviceAddress` was enabled at vkCreateDevice
    /// — true iff `has_accel_ext` was true at discovery. Drives
    /// whether `field_alloc_impl` adds the SHADER_DEVICE_ADDRESS
    /// usage bit (free when feature is on, illegal when off).
    /// Step 063 slice 23.
    pub(super) buffer_device_address_enabled: bool,
    /// Acceleration structure registry. Each entry holds the AS
    /// handle plus the storage buffer + memory it lives in;
    /// destroy_acceleration_structure (and Drop) walks the map to
    /// release everything in the right order. Step 063 slice 23.
    pub(super) acceleration_structures: RwLock<HashMap<u64, VulkanAccelerationStructure>>,
}

/// Native Vulkan acceleration structure — the AS handle plus the
/// storage VkBuffer it lives in. Step 063 slice 23.
pub(super) struct VulkanAccelerationStructure {
    pub(super) as_handle: *mut core::ffi::c_void,
    pub(super) storage_buffer: ffi::VkBuffer,
    pub(super) storage_memory: ffi::VkDeviceMemory,
}

unsafe impl Send for VulkanAccelerationStructure {}
unsafe impl Sync for VulkanAccelerationStructure {}

/// Software tessellation pipeline state — refines
/// `Quanta.Tessellation.Pipeline`.
pub(super) struct VulkanTessPipeline {
    pub(super) outer: Vec<u32>,
    pub(super) inner: Vec<u32>,
}

/// Software VRS state — refines `Quanta.Vrs.State`. Native lowering
/// goes through `vkCmdSetFragmentShadingRateKHR(rate, combiner_op)`
/// on render pipelines that enable the
/// `VK_KHR_fragment_shading_rate` extension; that wiring lands with
/// the render-encoder rebuild.
#[allow(dead_code)]
pub(super) struct VulkanVrsState {
    pub(super) rate_code: u8,
}

/// Software mesh-shader pipeline state — refines
/// `Quanta.MeshShader.Pipeline`. Native lowering goes through
/// `vkCmdDrawMeshTasksEXT` once the render-pipeline rebuild lands;
/// the proof contract holds for the MVP today.
#[allow(dead_code)]
pub(super) struct VulkanMeshPipeline {
    pub(super) max_vertices: u32,
    pub(super) max_primitives: u32,
    pub(super) task_threads: u32,
    pub(super) dispatched: Vec<[u32; 3]>,
}

/// Software bindless table — refines `Quanta.Bindless.Array`.
pub(super) struct VulkanBindlessArray {
    pub(super) cap: u32,
    pub(super) entries: Vec<u64>,
}

/// State for one Vulkan render bundle.
pub(super) struct VulkanRenderBundle {
    pub(super) cap: u32,
    pub(super) recorded: u32,
    pub(super) secondaries: Vec<ffi::VkCommandBuffer>,
}

/// State for one Vulkan ICB.
///
/// Native lowering: each `record_dispatch` writes one secondary
/// VkCommandBuffer (allocated lazily) bound to a dedicated
/// descriptor pool that lives as long as the ICB. `execute(count)`
/// runs `vkCmdExecuteCommands(primary, count, &secondaries[..count])`
/// and submits once. The replay path (commands fold) is no longer
/// used for execute; we keep `commands` only as a Vec<VkIcbCommand>
/// counter / discriminator for record-time state.
#[cfg_attr(not(feature = "compute"), allow(dead_code))]
pub(super) struct VkIcb {
    pub(super) cap: u32,
    pub(super) commands: Vec<VkIcbCommand>,
    /// Pre-allocated secondary command buffers, one per slot.
    /// `secondaries[i]` is recorded by `icb_record_dispatch(handle, i, ...)`.
    pub(super) secondaries: Vec<ffi::VkCommandBuffer>,
    /// Dedicated descriptor pool — outlives any single record.
    /// Reset on `indirect_buffer_destroy` only.
    pub(super) descriptor_pool: ffi::VkDescriptorPool,
}

/// One recorded ICB command. Compute = Dispatch; render = Draw.
/// Mirrors the Lean `Quanta.Icb.Command` sum type.
/// Fields are written at record time and consumed by the replay
/// path — suppressed until native vkCmdExecuteCommands lands.
#[allow(dead_code)]
// Dispatch carries the full binding + push-data payload inline (boxing the
// common dispatch path would add an allocation per recorded command); Draw is
// deliberately small. The size spread is intrinsic to the command payloads.
#[allow(clippy::large_enum_variant)]
pub(super) enum VkIcbCommand {
    Dispatch {
        wave_handle: u64,
        bindings: [u64; crate::api::types::MAX_BINDINGS],
        binding_count: u8,
        push_data: [u8; crate::api::types::PUSH_DATA_CAP],
        push_len: u16,
        push_mask: u16,
        workgroup_size: [u32; 3],
        groups: [u32; 3],
    },
    /// Render-path draw. The replay refinement records the
    /// parameters; live execution requires a real render-pass-
    /// continued secondary command buffer + vkCmdExecuteCommands,
    /// which is a future commit. T7006 is satisfied by the
    /// recording shape alone.
    Draw {
        pipeline: u64,
        vertex_count: u32,
        instance_count: u32,
    },
}

pub(super) struct VkQueryPool {
    pub(super) pool: ffi::VkQueryPool,
    pub(super) count: u32,
}

#[allow(dead_code)]
pub(super) struct VkBuffer {
    pub(super) buffer: ffi::VkBuffer,
    pub(super) memory: ffi::VkDeviceMemory,
    pub(super) size: u64,
    /// Persistently mapped pointer for HOST_VISIBLE buffers (avoids map/unmap per write).
    pub(super) mapped_ptr: Option<*mut u8>,
}

// Safety: The raw pointer in mapped_ptr points to Vulkan host-visible memory that
// outlives the VkBuffer. Access is synchronized by the RwLock in VulkanDevice.
unsafe impl Send for VkBuffer {}
unsafe impl Sync for VkBuffer {}

// Safety: Vulkan handles are thread-safe when externally synchronized.
// All mutable state is protected by RwLock/Mutex.
unsafe impl Send for VulkanDevice {}
unsafe impl Sync for VulkanDevice {}

#[allow(dead_code)]
pub(super) struct VkTexture {
    pub(super) image: ffi::VkImage,
    pub(super) view: ffi::VkImageView,
    pub(super) memory: ffi::VkDeviceMemory,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: u32,
    pub(super) mip_levels: u32,
    pub(super) current_layout: std::sync::atomic::AtomicU32,
}

// Fields are read by the compute-gated dispatch path only.
#[cfg_attr(not(feature = "compute"), allow(dead_code))]
pub(super) struct VkComputePipeline {
    pub(super) pipeline: ffi::VkPipeline,
    pub(super) layout: ffi::VkPipelineLayout,
    pub(super) descriptor_set_layout: ffi::VkDescriptorSetLayout,
    /// Per-binding descriptor kind reflected from the SPIR-V. Dispatch uses
    /// this to write STORAGE_IMAGE / COMBINED_IMAGE_SAMPLER descriptors for
    /// texture slots; buffer slots stay STORAGE_BUFFER. Ordered by binding.
    ///
    /// Compute-gated: its type (`spirv_meta::DescriptorKind`) lives behind
    /// the same `compute` gate, and only the compute dispatch path reads
    /// it — a `vulkan,render`-without-`compute` build keeps the pipeline
    /// registry (for its lifecycle drain and `waves` count) but carries no
    /// descriptor reflection.
    #[cfg(feature = "compute")]
    pub(super) descriptor_kinds: Vec<crate::driver::spirv_meta::DescriptorKind>,
}

pub(super) struct VkRenderPipeline {
    pub(super) pipeline: ffi::VkPipeline,
    pub(super) layout: ffi::VkPipelineLayout,
    pub(super) render_pass: ffi::VkRenderPass,
    pub(super) descriptor_set_layout: ffi::VkDescriptorSetLayout,
}

impl VulkanDevice {
    pub(super) fn alloc_handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Check if a device extension is available on the physical device.
    /// Used only by the render-gated ray-tracing path (step 085).
    #[cfg_attr(not(feature = "render"), allow(dead_code))]
    pub(super) fn has_device_extension(&self, ext_name: &[u8]) -> bool {
        let mut count = 0u32;
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                core::ptr::null_mut(),
            )
        };
        if result != ffi::VK_SUCCESS || count == 0 {
            return false;
        }
        let mut props = vec![ffi::VkExtensionProperties::default(); count as usize];
        let result = unsafe {
            ffi::vkEnumerateDeviceExtensionProperties(
                self.physical_device,
                core::ptr::null(),
                &mut count,
                props.as_mut_ptr(),
            )
        };
        if result != ffi::VK_SUCCESS {
            return false;
        }
        // ext_name is null-terminated; compare up to the null byte.
        let target = &ext_name[..ext_name.len() - 1]; // strip trailing \0
        props.iter().any(|p| {
            let name_bytes = &p.extension_name;
            let len = name_bytes
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_bytes.len());
            &name_bytes[..len] == target
        })
    }

    pub(super) fn alloc_command_buffer(&self) -> Result<ffi::VkCommandBuffer, QuantaError> {
        // Try to reuse a previously returned command buffer from the pool.
        if let Some(cmd) = self
            .cmd_buffer_pool
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .pop()
        {
            let result = unsafe { ffi::vkResetCommandBuffer(cmd, 0) };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            return Ok(cmd);
        }
        // Pool empty -- allocate a fresh one.
        let alloc_info = ffi::VkCommandBufferAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            command_pool: self.command_pool,
            level: ffi::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            command_buffer_count: 1,
        };
        let mut cmd = ffi::null_handle();
        let result = unsafe { ffi::vkAllocateCommandBuffers(self.device, &alloc_info, &mut cmd) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        Ok(cmd)
    }

    /// Acquire a descriptor pool — pop from cache or create new.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(super) fn acquire_descriptor_pool(&self) -> Result<ffi::VkDescriptorPool, QuantaError> {
        if let Some(pool) = self
            .descriptor_pool_cache
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .pop()
        {
            let result = unsafe { ffi::vkResetDescriptorPool(self.device, pool, 0) };
            if result != ffi::VK_SUCCESS {
                // Reset failed — destroy this pool and fall through to create a fresh one.
                unsafe {
                    ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
                }
            } else {
                return Ok(pool);
            }
        }
        // Sized for the worst case a single set can need: up to 16 storage
        // buffers plus up to 16 storage/sampled images. A compute dispatch
        // allocates one set from this pool, so over-provisioning is cheap.
        let pool_sizes = [
            ffi::VkDescriptorPoolSize {
                ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 16,
            },
            ffi::VkDescriptorPoolSize {
                ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptor_count: 16,
            },
            ffi::VkDescriptorPoolSize {
                ty: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptor_count: 16,
            },
        ];
        let pool_info = ffi::VkDescriptorPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
        };
        let mut pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorPool(self.device, &pool_info, core::ptr::null(), &mut pool)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }
        Ok(pool)
    }

    /// Acquire a compute descriptor-set layout for a per-binding kind list,
    /// cached by a 2-bits-per-binding signature. The buffer-only case (the
    /// 99% path) maps every binding to `STORAGE_BUFFER`, preserving prior
    /// behavior; image bindings emit `STORAGE_IMAGE` /
    /// `COMBINED_IMAGE_SAMPLER` so the layout matches the reflected SPIR-V.
    ///
    /// Compute-only, for the same reason as `layout_signature`: it keys on
    /// `spirv_meta::DescriptorKind` (compute-gated) and only the compute
    /// dispatch path builds these layouts.
    #[cfg(feature = "compute")]
    pub(super) fn acquire_descriptor_set_layout(
        &self,
        kinds: &[crate::driver::spirv_meta::DescriptorKind],
    ) -> Result<ffi::VkDescriptorSetLayout, QuantaError> {
        use crate::driver::spirv_meta::DescriptorKind;
        let signature = layout_signature(kinds);
        {
            let cache = self
                .layout_cache
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            if let Some(&layout) = cache.get(&signature) {
                return Ok(layout);
            }
        }
        // Cache miss — create a new layout.
        let mut bindings = alloc::vec::Vec::new();
        for (i, kind) in kinds.iter().enumerate() {
            let descriptor_type = match kind {
                DescriptorKind::StorageBuffer => ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                DescriptorKind::StorageImage => ffi::VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                DescriptorKind::SampledImage => ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            };
            bindings.push(ffi::VkDescriptorSetLayoutBinding {
                binding: i as u32,
                descriptor_type,
                descriptor_count: 1,
                stage_flags: ffi::VK_SHADER_STAGE_COMPUTE_BIT,
                p_immutable_samplers: core::ptr::null(),
            });
        }
        let ds_layout_info = ffi::VkDescriptorSetLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
        };
        let mut layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorSetLayout(
                self.device,
                &ds_layout_info,
                core::ptr::null(),
                &mut layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::internal(
                "descriptor set layout creation failed",
            ));
        }
        self.layout_cache
            .lock()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(signature, layout);
        Ok(layout)
    }

    /// Return a descriptor pool to the cache for reuse.
    #[cfg_attr(not(feature = "compute"), allow(dead_code))]
    pub(super) fn return_descriptor_pool(&self, pool: ffi::VkDescriptorPool) {
        if let Ok(mut cache) = self.descriptor_pool_cache.lock() {
            cache.push(pool);
        } else {
            // Lock poisoned — destroy to avoid leak.
            unsafe {
                ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
            }
        }
    }

    /// Acquire a staging buffer of at least `min_size` bytes from the pool, or create a new one.
    pub(super) fn acquire_staging_buffer(
        &self,
        min_size: usize,
    ) -> Result<(ffi::VkBuffer, ffi::VkDeviceMemory, usize), QuantaError> {
        // Try to find a suitable buffer in the pool.
        if let Ok(mut pool) = self.staging_pool.lock()
            && let Some(idx) = pool.iter().position(|&(_, _, cap)| cap >= min_size)
        {
            return Ok(pool.swap_remove(idx));
        }
        // Pool miss — allocate a new staging buffer (both SRC and DST for read-back reuse).
        let staging_info = ffi::VkBufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            size: min_size as u64,
            usage: ffi::VK_BUFFER_USAGE_TRANSFER_SRC_BIT | ffi::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharing_mode: ffi::VK_SHARING_MODE_EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: core::ptr::null(),
        };
        let mut buf = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateBuffer(self.device, &staging_info, core::ptr::null(), &mut buf) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        let mut mem_reqs = unsafe { core::mem::zeroed::<ffi::VkMemoryRequirements>() };
        unsafe { ffi::vkGetBufferMemoryRequirements(self.device, buf, &mut mem_reqs) };
        let mem_type = self.find_memory_type(
            mem_reqs.memory_type_bits,
            ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )?;
        let alloc_info = ffi::VkMemoryAllocateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            p_next: core::ptr::null(),
            allocation_size: mem_reqs.size,
            memory_type_index: mem_type,
        };
        let mut mem = ffi::null_handle();
        let result =
            unsafe { ffi::vkAllocateMemory(self.device, &alloc_info, core::ptr::null(), &mut mem) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        let result = unsafe { ffi::vkBindBufferMemory(self.device, buf, mem, 0) };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::out_of_memory());
        }
        Ok((buf, mem, min_size))
    }

    /// Return a staging buffer to the pool for reuse.
    pub(super) fn return_staging_buffer(
        &self,
        buf: ffi::VkBuffer,
        mem: ffi::VkDeviceMemory,
        cap: usize,
    ) {
        if let Ok(mut pool) = self.staging_pool.lock() {
            // Cap pool size to avoid unbounded growth.
            if pool.len() < 8 {
                pool.push((buf, mem, cap));
                return;
            }
        }
        // Pool full or lock poisoned — destroy immediately.
        unsafe {
            ffi::vkDestroyBuffer(self.device, buf, core::ptr::null());
            ffi::vkFreeMemory(self.device, mem, core::ptr::null());
        }
    }

    /// Submit a command buffer with a fence. Returns a Pulse that waits on the
    /// fence when wait() is called. The GPU executes asynchronously — the CPU
    /// can do other work before calling pulse.wait().
    pub(super) fn submit_and_wait(&self, cmd: ffi::VkCommandBuffer) -> Result<Pulse, QuantaError> {
        let fence_info = ffi::VkFenceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
        };
        let mut fence = ffi::null_handle();
        unsafe {
            let r = ffi::vkCreateFence(self.device, &fence_info, core::ptr::null(), &mut fence);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let submit = ffi::VkSubmitInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            p_next: core::ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: core::ptr::null(),
            p_wait_dst_stage_mask: core::ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: &cmd,
            signal_semaphore_count: 0,
            p_signal_semaphores: core::ptr::null(),
        };
        unsafe {
            let r = ffi::vkQueueSubmit(self.queue, 1, &submit, fence);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        // vkWaitForFences/vkDestroyFence are legal from any thread (this
        // pulse is the fence's sole owner), and the pooled command buffer
        // is only touched under the pool mutex — safe to move the wait
        // onto Pulse::on_complete's waiter thread.
        struct FenceWaiter {
            device: ffi::VkDevice,
            fence: ffi::VkFence,
            cmd: ffi::VkCommandBuffer,
            pool: std::sync::Arc<Mutex<Vec<ffi::VkCommandBuffer>>>,
        }
        unsafe impl Send for FenceWaiter {}
        type FenceParts = (
            ffi::VkDevice,
            ffi::VkFence,
            ffi::VkCommandBuffer,
            std::sync::Arc<Mutex<Vec<ffi::VkCommandBuffer>>>,
        );
        impl FenceWaiter {
            // By-value method: the closure must capture the whole
            // (Send-asserted) struct, not its raw-pointer fields.
            fn take(self) -> FenceParts {
                (self.device, self.fence, self.cmd, self.pool)
            }
        }
        let waiter = FenceWaiter {
            device: self.device,
            fence,
            cmd,
            pool: self.cmd_buffer_pool.clone(),
        };

        let handle = self.alloc_handle();
        Ok(Pulse {
            handle,
            completed: false,
            wait_fn: Some(Box::new(move || unsafe {
                let (device, fence, cmd, pool) = waiter.take();
                ffi::vkWaitForFences(device, 1, &fence, 1, u64::MAX);
                ffi::vkDestroyFence(device, fence, core::ptr::null());
                if let Ok(mut p) = pool.lock() {
                    p.push(cmd);
                }
            })),
        })
    }
}

/// Discover Vulkan devices on this system.
pub fn discover() -> Vec<Box<dyn GpuDevice>> {
    let app_info = ffi::VkApplicationInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: core::ptr::null(),
        p_application_name: core::ptr::null(),
        application_version: 0,
        p_engine_name: core::ptr::null(),
        engine_version: 0,
        api_version: ffi::make_api_version(0, 1, 3, 0),
    };

    // WSI instance extensions — enabled only when the loader offers
    // them; their absence just means supports_surface_present() stays
    // false on the resulting device.
    let has_surface_ext = instance_has_extension(b"VK_KHR_surface\0");
    let has_headless_ext = has_surface_ext && instance_has_extension(b"VK_EXT_headless_surface\0");
    let has_xlib_ext = cfg!(target_os = "linux")
        && has_surface_ext
        && instance_has_extension(b"VK_KHR_xlib_surface\0");
    let has_android_ext = cfg!(target_os = "android")
        && has_surface_ext
        && instance_has_extension(b"VK_KHR_android_surface\0");
    let has_win32_ext = cfg!(target_os = "windows")
        && has_surface_ext
        && instance_has_extension(b"VK_KHR_win32_surface\0");
    let mut instance_exts: Vec<*const core::ffi::c_char> = Vec::new();
    if has_surface_ext {
        instance_exts.push(c"VK_KHR_surface".as_ptr());
    }
    if has_headless_ext {
        instance_exts.push(c"VK_EXT_headless_surface".as_ptr());
    }
    if has_xlib_ext {
        instance_exts.push(c"VK_KHR_xlib_surface".as_ptr());
    }
    if has_android_ext {
        instance_exts.push(c"VK_KHR_android_surface".as_ptr());
    }
    if has_win32_ext {
        instance_exts.push(c"VK_KHR_win32_surface".as_ptr());
    }
    let (instance_ext_count, instance_ext_ptr) = if instance_exts.is_empty() {
        (0u32, core::ptr::null())
    } else {
        (instance_exts.len() as u32, instance_exts.as_ptr())
    };

    let create_info = ffi::VkInstanceCreateInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: instance_ext_count,
        pp_enabled_extension_names: instance_ext_ptr,
    };

    let mut instance = ffi::null_handle();
    let result = unsafe { ffi::vkCreateInstance(&create_info, core::ptr::null(), &mut instance) };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    // Step 063 slice 14 — resolve the instance-level proc once.
    // `None` is fine on builds without the VRS extension; the
    // per-physical-device query just yields an empty supported-rate
    // list, and the render encoder gate falls through to its own
    // NotSupported message.
    let get_shading_rates_fn: Option<ffi::PfnVkGetPhysicalDeviceFragmentShadingRatesKHR> = {
        let name = b"vkGetPhysicalDeviceFragmentShadingRatesKHR\0";
        let p = unsafe {
            ffi::vkGetInstanceProcAddr(instance, name.as_ptr() as *const core::ffi::c_char)
        };
        if p.is_null() {
            None
        } else {
            // SAFETY: vkGetInstanceProcAddr returns a valid function
            // pointer of the documented signature when non-null.
            Some(unsafe {
                core::mem::transmute::<
                    *const core::ffi::c_void,
                    ffi::PfnVkGetPhysicalDeviceFragmentShadingRatesKHR,
                >(p)
            })
        }
    };

    // TASK 37 — resolve `vkGetPhysicalDeviceProperties2` (core 1.1)
    // once. Used below to chain the subgroup-properties query onto
    // each physical device. Null on 1.0-only loaders — the subgroup
    // capability then conservatively reports false.
    let get_props2_fn: Option<ffi::PfnVkGetPhysicalDeviceProperties2> = {
        let name = b"vkGetPhysicalDeviceProperties2\0";
        let p = unsafe {
            ffi::vkGetInstanceProcAddr(instance, name.as_ptr() as *const core::ffi::c_char)
        };
        if p.is_null() {
            None
        } else {
            // SAFETY: vkGetInstanceProcAddr returns a valid function
            // pointer of the documented signature when non-null.
            Some(unsafe {
                core::mem::transmute::<
                    *const core::ffi::c_void,
                    ffi::PfnVkGetPhysicalDeviceProperties2,
                >(p)
            })
        }
    };

    // Resolve `vkGetPhysicalDeviceFeatures2` (core 1.1) once. Used
    // below to chain the 16-/8-bit storage feature queries onto each
    // physical device — those features gate the native-stride bf16 /
    // fp8 buffer contract. Null on 1.0-only loaders — narrow storage
    // then conservatively stays disabled.
    let get_features2_fn: Option<ffi::PfnVkGetPhysicalDeviceFeatures2> = {
        let name = b"vkGetPhysicalDeviceFeatures2\0";
        let p = unsafe {
            ffi::vkGetInstanceProcAddr(instance, name.as_ptr() as *const core::ffi::c_char)
        };
        if p.is_null() {
            None
        } else {
            // SAFETY: vkGetInstanceProcAddr returns a valid function
            // pointer of the documented signature when non-null.
            Some(unsafe {
                core::mem::transmute::<*const core::ffi::c_void, ffi::PfnVkGetPhysicalDeviceFeatures2>(
                    p,
                )
            })
        }
    };

    let mut count = 0u32;
    let result =
        unsafe { ffi::vkEnumeratePhysicalDevices(instance, &mut count, core::ptr::null_mut()) };
    if result != ffi::VK_SUCCESS || count == 0 {
        return Vec::new();
    }

    let mut physical_devices = vec![ffi::null_handle(); count as usize];
    let result = unsafe {
        ffi::vkEnumeratePhysicalDevices(instance, &mut count, physical_devices.as_mut_ptr())
    };
    if result != ffi::VK_SUCCESS {
        return Vec::new();
    }

    let mut devices: Vec<Box<dyn GpuDevice>> = Vec::new();

    for pd in physical_devices {
        let mut props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceProperties(pd, &mut props) };

        let mut qf_count = 0u32;
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(pd, &mut qf_count, core::ptr::null_mut())
        };
        let mut queue_families = vec![ffi::VkQueueFamilyProperties::default(); qf_count as usize];
        unsafe {
            ffi::vkGetPhysicalDeviceQueueFamilyProperties(
                pd,
                &mut qf_count,
                queue_families.as_mut_ptr(),
            )
        };

        // Slice 6 + 16 — query device features once up-front so
        // both queue-family selection (sparse_binding) and later
        // gating (tessellationShader) can read the cached value.
        let mut device_features = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceFeatures>() };
        unsafe { ffi::vkGetPhysicalDeviceFeatures(pd, &mut device_features) };
        let tessellation_feature = device_features.tessellation_shader != 0;
        let shader_float64_supported = device_features.shader_float64 != 0;
        let shader_int64_supported = device_features.shader_int64 != 0;

        // 16-/8-bit storage-buffer access — gates the native-stride bf16
        // (16-bit elements) / fp8 (8-bit elements) buffer contract shared
        // with the host upload and the CPU executor. Both feature structs
        // are core-defined from Vulkan 1.2 (16-bit storage is 1.1 core,
        // 8-bit was promoted in 1.2), so the chained query is only issued
        // on 1.2+ devices; older devices conservatively report false and
        // bf16/fp8 pipelines fail creation with the capability named.
        const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);
        let api_12 = props.api_version >= VK_API_VERSION_1_2;
        let (storage16_supported, storage8_supported) = match get_features2_fn {
            Some(get_features2) if api_12 => {
                let mut storage8_query = ffi::VkPhysicalDevice8BitStorageFeatures {
                    s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
                    p_next: core::ptr::null_mut(),
                    storage_buffer_8bit_access: 0,
                    uniform_and_storage_buffer_8bit_access: 0,
                    storage_push_constant8: 0,
                };
                let mut storage16_query = ffi::VkPhysicalDevice16BitStorageFeatures {
                    s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
                    p_next: &mut storage8_query as *mut _ as *mut core::ffi::c_void,
                    storage_buffer_16bit_access: 0,
                    uniform_and_storage_buffer_16bit_access: 0,
                    storage_push_constant16: 0,
                    storage_input_output16: 0,
                };
                let mut features2 = ffi::VkPhysicalDeviceFeatures2 {
                    s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                    p_next: &mut storage16_query as *mut _ as *mut core::ffi::c_void,
                    features: unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceFeatures>() },
                };
                unsafe { get_features2(pd, &mut features2) };
                (
                    storage16_query.storage_buffer_16bit_access != 0,
                    storage8_query.storage_buffer_8bit_access != 0,
                )
            }
            _ => (false, false),
        };

        // TASK 37 — subgroup arithmetic capability. Chain
        // VkPhysicalDeviceSubgroupProperties onto a properties2 query;
        // the prims subgroup-reduce path is only sound when the
        // compute stage supports the ARITHMETIC class. V3D: false
        // (BASIC only); llvmpipe: true.
        let subgroup_arithmetic_supported = match get_props2_fn {
            Some(get_props2) => {
                let mut subgroup_props = ffi::VkPhysicalDeviceSubgroupProperties {
                    s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
                    p_next: core::ptr::null_mut(),
                    subgroup_size: 0,
                    supported_stages: 0,
                    supported_operations: 0,
                    quad_operations_in_all_stages: 0,
                };
                let mut props2 = ffi::VkPhysicalDeviceProperties2 {
                    s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                    p_next: &mut subgroup_props as *mut _ as *mut core::ffi::c_void,
                    properties: unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceProperties>() },
                };
                unsafe { get_props2(pd, &mut props2) };
                (subgroup_props.supported_operations & ffi::VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0
                    && (subgroup_props.supported_stages & ffi::VK_SHADER_STAGE_COMPUTE_BIT) != 0
            }
            None => false,
        };

        // Find a queue family that supports compute + graphics
        let queue_family = queue_families.iter().enumerate().find(|(_, qf)| {
            (qf.queue_flags & ffi::VK_QUEUE_GRAPHICS_BIT) != 0
                && (qf.queue_flags & ffi::VK_QUEUE_COMPUTE_BIT) != 0
        });

        let Some((qf_index, qf_props)) = queue_family else {
            continue;
        };
        // Slice 16 — cache whether the chosen queue family also
        // advertises VK_QUEUE_SPARSE_BINDING_BIT. Combined with
        // VkPhysicalDeviceFeatures.sparseBinding from the cached
        // features above, determines whether the future bind-sparse
        // path can run without picking a different queue family.
        let queue_has_sparse = (qf_props.queue_flags & ffi::VK_QUEUE_SPARSE_BINDING_BIT) != 0;
        let sparse_binding_supported = queue_has_sparse && device_features.sparse_binding != 0;

        let queue_priorities = [1.0f32];
        let queue_create = ffi::VkDeviceQueueCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            queue_family_index: qf_index as u32,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
        };

        // Step 063 — enable advanced render extensions when present
        // on the physical device. Each block:
        //   * Detects the extension.
        //   * Adds its null-terminated name to enabled_extensions.
        //   * Resolves its entry-point proc address after vkCreateDevice.
        //   * Stores the resolved Pfn in VulkanDevice; absence keeps
        //     the existing NotSupported behavior on the matching
        //     render-encoder / dispatch arm.
        let has_vrs_ext = physical_device_has_extension(pd, b"VK_KHR_fragment_shading_rate\0");
        let has_mesh_ext = physical_device_has_extension(pd, b"VK_EXT_mesh_shader\0");
        let has_accel_ext = physical_device_has_extension(pd, b"VK_KHR_acceleration_structure\0");
        let has_rt_pipeline_ext =
            physical_device_has_extension(pd, b"VK_KHR_ray_tracing_pipeline\0");
        let has_rt = has_accel_ext && has_rt_pipeline_ext;

        // Slice 23 — chain buffer-device-address + acceleration-
        // structure features after sync2 when the device advertises
        // them. Buffer device addresses are needed by AS builds
        // (geometry inputs, scratch, AS-storage all reference each
        // other by device address). The feature is core in Vulkan
        // 1.2; chaining a 1.2 features struct works on 1.3 devices
        // that don't already promote it.
        let bda_features = ffi::VkPhysicalDeviceBufferDeviceAddressFeatures {
            s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            p_next: core::ptr::null_mut(),
            buffer_device_address: if has_accel_ext { 1 } else { 0 },
            buffer_device_address_capture_replay: 0,
            buffer_device_address_multi_device: 0,
        };
        let accel_features = ffi::VkPhysicalDeviceAccelerationStructureFeaturesKHR {
            s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            p_next: &bda_features as *const _ as *mut core::ffi::c_void,
            acceleration_structure: if has_accel_ext { 1 } else { 0 },
            acceleration_structure_capture_replay: 0,
            acceleration_structure_indirect_build: 0,
            acceleration_structure_host_commands: 0,
            descriptor_binding_acceleration_structure_update_after_bind: 0,
        };

        // Enable synchronization2 (Vulkan 1.3 core) for vkCmdPipelineBarrier2
        #[repr(C)]
        struct VkPhysicalDeviceSynchronization2Features {
            s_type: u32,
            p_next: *const core::ffi::c_void,
            synchronization2: u32,
        }
        let sync2_p_next: *const core::ffi::c_void = if has_accel_ext {
            &accel_features as *const _ as *const core::ffi::c_void
        } else {
            core::ptr::null()
        };
        let sync2_features = VkPhysicalDeviceSynchronization2Features {
            s_type: 1000314007, // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES
            p_next: sync2_p_next,
            synchronization2: 1, // VK_TRUE
        };

        // Enable 16-/8-bit storage-buffer access when the device
        // advertises it (queried above), so bf16 / fp8 kernels — whose
        // SPIR-V declares StorageBuffer16BitAccess / StorageBuffer8BitAccess
        // for native-stride narrow buffers — can create pipelines. The
        // structs are only chained on 1.2+ devices (where both are
        // core-defined); each bit is set exactly to the queried support,
        // since enabling an unadvertised feature fails vkCreateDevice.
        let storage8_enable = ffi::VkPhysicalDevice8BitStorageFeatures {
            s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
            p_next: &sync2_features as *const _ as *mut core::ffi::c_void,
            storage_buffer_8bit_access: if storage8_supported { 1 } else { 0 },
            uniform_and_storage_buffer_8bit_access: 0,
            storage_push_constant8: 0,
        };
        let storage16_enable = ffi::VkPhysicalDevice16BitStorageFeatures {
            s_type: ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            p_next: &storage8_enable as *const _ as *mut core::ffi::c_void,
            storage_buffer_16bit_access: if storage16_supported { 1 } else { 0 },
            uniform_and_storage_buffer_16bit_access: 0,
            storage_push_constant16: 0,
            storage_input_output16: 0,
        };
        let device_p_next: *const core::ffi::c_void = if api_12 {
            &storage16_enable as *const _ as *const core::ffi::c_void
        } else {
            &sync2_features as *const _ as *const core::ffi::c_void
        };

        let has_swapchain_ext =
            has_surface_ext && physical_device_has_extension(pd, b"VK_KHR_swapchain\0");

        let mut enabled_extensions: Vec<*const core::ffi::c_char> = Vec::new();
        if has_swapchain_ext {
            enabled_extensions.push(c"VK_KHR_swapchain".as_ptr());
        }
        if has_vrs_ext {
            enabled_extensions.push(c"VK_KHR_fragment_shading_rate".as_ptr());
        }
        if has_mesh_ext {
            enabled_extensions.push(c"VK_EXT_mesh_shader".as_ptr());
        }
        if has_rt {
            enabled_extensions.push(c"VK_KHR_acceleration_structure".as_ptr());
            enabled_extensions.push(c"VK_KHR_ray_tracing_pipeline".as_ptr());
            // Both ray-tracing extensions require deferred-host-ops.
            enabled_extensions.push(c"VK_KHR_deferred_host_operations".as_ptr());
            // VK_KHR_acceleration_structure requires VK_KHR_buffer_device_address.
            enabled_extensions.push(c"VK_KHR_buffer_device_address".as_ptr());
        }
        let (enabled_ext_count, enabled_ext_ptr) = if enabled_extensions.is_empty() {
            (0u32, core::ptr::null())
        } else {
            (enabled_extensions.len() as u32, enabled_extensions.as_ptr())
        };

        // Slice 18 — enable the device features we depend on.
        // Each field is set only when the physical device already
        // advertises support; enabling an unsupported feature would
        // make vkCreateDevice fail. Today: sparseBinding (used by
        // the future bind-sparse path that builds on slice 16).
        // tessellationShader can join here when the TCS+TES
        // pipeline-create path lands.
        let mut enabled_feats = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceFeatures>() };
        if device_features.sparse_binding != 0 {
            enabled_feats.sparse_binding = 1;
        }
        // Enable 64-bit floats when the device advertises them, so
        // kernels that use `f64` (and emit the Float64 SPIR-V
        // capability) can create a compute pipeline. Left disabled,
        // such a pipeline is invalid and the driver rejects it.
        if shader_float64_supported {
            enabled_feats.shader_float64 = 1;
        }
        // Likewise for 64-bit integers (the Int64 SPIR-V capability).
        if shader_int64_supported {
            enabled_feats.shader_int64 = 1;
        }

        let device_create = ffi::VkDeviceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: device_p_next,
            flags: 0,
            queue_create_info_count: 1,
            p_queue_create_infos: &queue_create,
            enabled_layer_count: 0,
            pp_enabled_layer_names: core::ptr::null(),
            enabled_extension_count: enabled_ext_count,
            pp_enabled_extension_names: enabled_ext_ptr,
            p_enabled_features: &enabled_feats as *const _ as *const core::ffi::c_void,
        };

        let mut device = ffi::null_handle();
        let result =
            unsafe { ffi::vkCreateDevice(pd, &device_create, core::ptr::null(), &mut device) };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        let mut queue = ffi::null_handle();
        unsafe { ffi::vkGetDeviceQueue(device, qf_index as u32, 0, &mut queue) };

        // WSI proc resolution — None (→ NotSupported) when the loader
        // or driver lacks the surface/swapchain extensions.
        #[cfg(not(feature = "render"))]
        let _ = (
            has_headless_ext,
            has_xlib_ext,
            has_android_ext,
            has_win32_ext,
        );
        #[cfg(feature = "render")]
        let surface_procs = if has_swapchain_ext {
            super::surface::SurfaceProcs::resolve(
                instance,
                device,
                has_headless_ext,
                has_xlib_ext,
                has_android_ext,
                has_win32_ext,
            )
        } else {
            None
        };

        // Resolve extension proc addresses. Even with an extension
        // enabled at vkCreateDevice, the driver can return null
        // (lavapipe without the optional feature, partial
        // implementations, etc.). Treat null the same as "extension
        // absent" so the matching arm falls through to NotSupported.
        //
        // SAFETY for each transmute below: vkGetDeviceProcAddr
        // returns a valid function pointer of the documented
        // signature when non-null, per the Vulkan spec.
        let vrs_set_rate_fn: Option<ffi::PfnVkCmdSetFragmentShadingRateKHR> = if has_vrs_ext {
            let name = b"vkCmdSetFragmentShadingRateKHR\0";
            let p = unsafe {
                ffi::vkGetDeviceProcAddr(device, name.as_ptr() as *const core::ffi::c_char)
            };
            if p.is_null() {
                None
            } else {
                Some(unsafe {
                    core::mem::transmute::<
                        *const core::ffi::c_void,
                        ffi::PfnVkCmdSetFragmentShadingRateKHR,
                    >(p)
                })
            }
        } else {
            None
        };
        let mesh_draw_fn: Option<ffi::PfnVkCmdDrawMeshTasksEXT> = if has_mesh_ext {
            let name = b"vkCmdDrawMeshTasksEXT\0";
            let p = unsafe {
                ffi::vkGetDeviceProcAddr(device, name.as_ptr() as *const core::ffi::c_char)
            };
            if p.is_null() {
                None
            } else {
                Some(unsafe {
                    core::mem::transmute::<*const core::ffi::c_void, ffi::PfnVkCmdDrawMeshTasksEXT>(
                        p,
                    )
                })
            }
        } else {
            None
        };
        // Slice 14 — enumerate supported shading rates. The query
        // operates on the physical device, so it uses the
        // instance-level proc resolved before this loop.
        let supported_shading_rates: Vec<(u32, u32)> = match (has_vrs_ext, get_shading_rates_fn) {
            (true, Some(query)) => {
                let mut count = 0u32;
                let r = unsafe { query(pd, &mut count, core::ptr::null_mut()) };
                if r == ffi::VK_SUCCESS && count > 0 {
                    let mut rates = vec![
                        ffi::VkPhysicalDeviceFragmentShadingRateKHR {
                            s_type:
                                ffi::VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR,
                            ..Default::default()
                        };
                        count as usize
                    ];
                    let r = unsafe { query(pd, &mut count, rates.as_mut_ptr()) };
                    if r == ffi::VK_SUCCESS {
                        rates
                            .iter()
                            .filter(|e| e.sample_counts != 0)
                            .map(|e| (e.fragment_size.width, e.fragment_size.height))
                            .collect()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        };

        let trace_rays_fn: Option<ffi::PfnVkCmdTraceRaysKHR> = if has_rt {
            let name = b"vkCmdTraceRaysKHR\0";
            let p = unsafe {
                ffi::vkGetDeviceProcAddr(device, name.as_ptr() as *const core::ffi::c_char)
            };
            if p.is_null() {
                None
            } else {
                Some(unsafe {
                    core::mem::transmute::<*const core::ffi::c_void, ffi::PfnVkCmdTraceRaysKHR>(p)
                })
            }
        } else {
            None
        };
        // Slice 15 — acceleration-structure build procs. Loaded
        // off `has_accel_ext` (not `has_rt`) so AS builds can be
        // available even on devices that lack the ray-tracing
        // pipeline extension. The four procs travel together
        // because a build path needs the whole set.
        let resolve_pfn = |has: bool, name: &[u8]| -> Option<*const core::ffi::c_void> {
            if !has {
                return None;
            }
            let p = unsafe {
                ffi::vkGetDeviceProcAddr(device, name.as_ptr() as *const core::ffi::c_char)
            };
            if p.is_null() { None } else { Some(p) }
        };
        let accel_create_fn = resolve_pfn(has_accel_ext, b"vkCreateAccelerationStructureKHR\0")
            .map(|p| unsafe {
                core::mem::transmute::<
                    *const core::ffi::c_void,
                    ffi::PfnVkCreateAccelerationStructureKHR,
                >(p)
            });
        let accel_destroy_fn = resolve_pfn(has_accel_ext, b"vkDestroyAccelerationStructureKHR\0")
            .map(|p| unsafe {
                core::mem::transmute::<
                    *const core::ffi::c_void,
                    ffi::PfnVkDestroyAccelerationStructureKHR,
                >(p)
            });
        let accel_build_sizes_fn =
            resolve_pfn(has_accel_ext, b"vkGetAccelerationStructureBuildSizesKHR\0").map(
                |p| unsafe {
                    core::mem::transmute::<
                        *const core::ffi::c_void,
                        ffi::PfnVkGetAccelerationStructureBuildSizesKHR,
                    >(p)
                },
            );
        let accel_build_fn = resolve_pfn(has_accel_ext, b"vkCmdBuildAccelerationStructuresKHR\0")
            .map(|p| unsafe {
                core::mem::transmute::<
                    *const core::ffi::c_void,
                    ffi::PfnVkCmdBuildAccelerationStructuresKHR,
                >(p)
            });

        // vkCmdDispatchBase is core Vulkan 1.1 (no extension gate) —
        // resolve unconditionally; null only on 1.0-only drivers.
        // Enables the folded 1D-dispatch path for group counts above
        // maxComputeWorkGroupCount[0].
        let dispatch_base_fn = resolve_pfn(true, b"vkCmdDispatchBase\0").map(|p| unsafe {
            core::mem::transmute::<*const core::ffi::c_void, ffi::PfnVkCmdDispatchBase>(p)
        });

        // Command pool
        let pool_info = ffi::VkCommandPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queue_family_index: qf_index as u32,
        };
        let mut command_pool = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateCommandPool(device, &pool_info, core::ptr::null(), &mut command_pool)
        };
        if result != ffi::VK_SUCCESS {
            continue;
        }

        // Create pipeline cache for faster pipeline creation
        let cache_info = ffi::VkPipelineCacheCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            initial_data_size: 0,
            p_initial_data: core::ptr::null(),
        };
        let mut pipeline_cache = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreatePipelineCache(device, &cache_info, core::ptr::null(), &mut pipeline_cache)
        };
        if result != ffi::VK_SUCCESS {
            // Non-fatal — proceed with null cache (Vulkan allows it)
            pipeline_cache = ffi::null_handle();
        }

        let name = unsafe {
            let cstr =
                std::ffi::CStr::from_ptr(props.device_name.as_ptr() as *const core::ffi::c_char);
            cstr.to_string_lossy().to_string()
        };

        let vendor = match props.vendor_id {
            0x1002 => Vendor::Amd,
            0x10DE => Vendor::Nvidia,
            0x8086 => Vendor::Intel,
            0x13B5 | 0x14E4 => Vendor::Broadcom,
            _ => Vendor::Unknown,
        };

        // Query total device memory from the largest heap
        let mut mem_props = unsafe { core::mem::zeroed::<ffi::VkPhysicalDeviceMemoryProperties>() };
        unsafe { ffi::vkGetPhysicalDeviceMemoryProperties(pd, &mut mem_props) };
        let total_memory = (0..mem_props.memory_heap_count as usize)
            .map(|i| mem_props.memory_heaps[i].size)
            .max()
            .unwrap_or(0);

        let caps = Caps {
            nuclei: props.limits.max_compute_work_group_count[0].min(1024),
            protons_per_nucleus: 1,
            quarks_per_proton: props.limits.max_compute_work_group_size[0],
            memory_bytes: total_memory,
            max_quarks_per_dispatch: props.limits.max_compute_work_group_invocations,
            max_groups: props.limits.max_compute_work_group_count,
            vendor,
            name,
        };

        devices.push(Box::new(VulkanDevice {
            instance,
            physical_device: pd,
            device,
            queue,
            queue_family: qf_index as u32,
            command_pool,
            pipeline_cache,
            caps,
            max_push_constants_size: props.limits.max_push_constants_size,
            max_vertex_input_attributes: props.limits.max_vertex_input_attributes,
            buffers: RwLock::new(HashMap::new()),
            textures: RwLock::new(HashMap::new()),
            compute_pipelines: RwLock::new(HashMap::new()),
            render_pipelines: RwLock::new(HashMap::new()),
            samplers: RwLock::new(HashMap::new()),
            #[cfg(feature = "render")]
            render_sampler_cache: RwLock::new(HashMap::new()),
            image_views: RwLock::new(HashMap::new()),
            query_pools: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            #[cfg(feature = "render")]
            surface_procs,
            #[cfg(feature = "render")]
            vk_surfaces: RwLock::new(HashMap::new()),
            #[cfg(feature = "render")]
            vk_surface_frames: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(0),
            // The pool handle is genuinely shared (cloned out at dispatch time),
            // so Arc is intended; VkCommandBuffer is a raw FFI pointer that can't
            // be Send+Sync, which is inherent to the Vulkan handle model.
            #[allow(clippy::arc_with_non_send_sync)]
            cmd_buffer_pool: std::sync::Arc::new(Mutex::new(Vec::new())),
            descriptor_pool_cache: Mutex::new(Vec::new()),
            staging_pool: Mutex::new(Vec::new()),
            layout_cache: Mutex::new(HashMap::new()),
            #[cfg(feature = "compute")]
            compute_sampler: Mutex::new(ffi::null_handle()),
            icbs: RwLock::new(HashMap::new()),
            render_bundles: RwLock::new(HashMap::new()),
            bindless_textures: RwLock::new(HashMap::new()),
            bindless_buffers: RwLock::new(HashMap::new()),
            tess_pipelines: RwLock::new(HashMap::new()),
            mesh_pipelines: RwLock::new(HashMap::new()),
            vrs_states: RwLock::new(HashMap::new()),
            vrs_set_rate_fn,
            mesh_draw_fn,
            trace_rays_fn,
            accel_create_fn,
            accel_destroy_fn,
            accel_build_sizes_fn,
            accel_build_fn,
            tessellation_feature,
            supported_shading_rates,
            sparse_binding_supported,
            shader_float64_supported,
            shader_int64_supported,
            subgroup_arithmetic_supported,
            dispatch_base_fn,
            sparse_tile_bindings: RwLock::new(HashMap::new()),
            buffer_device_address_enabled: has_accel_ext,
            acceleration_structures: RwLock::new(HashMap::new()),
        }));

        break; // Use first suitable device
    }

    devices
}

/// Whether the Vulkan loader offers an instance-level extension.
fn instance_has_extension(ext_name: &[u8]) -> bool {
    let mut count = 0u32;
    let result = unsafe {
        ffi::vkEnumerateInstanceExtensionProperties(
            core::ptr::null(),
            &mut count,
            core::ptr::null_mut(),
        )
    };
    if result != ffi::VK_SUCCESS || count == 0 {
        return false;
    }
    let mut props = vec![ffi::VkExtensionProperties::default(); count as usize];
    let result = unsafe {
        ffi::vkEnumerateInstanceExtensionProperties(
            core::ptr::null(),
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if result != ffi::VK_SUCCESS {
        return false;
    }
    let target = &ext_name[..ext_name.len() - 1];
    props.iter().take(count as usize).any(|p| {
        let name_bytes = &p.extension_name;
        let len = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        &name_bytes[..len] == target
    })
}

/// Pre-create-time variant of `VulkanDevice::has_device_extension`.
/// Used during device discovery to decide which extensions to enable
/// at `vkCreateDevice`, before the `VulkanDevice` exists. `ext_name`
/// must be a null-terminated byte string.
fn physical_device_has_extension(pd: ffi::VkPhysicalDevice, ext_name: &[u8]) -> bool {
    let mut count = 0u32;
    let result = unsafe {
        ffi::vkEnumerateDeviceExtensionProperties(
            pd,
            core::ptr::null(),
            &mut count,
            core::ptr::null_mut(),
        )
    };
    if result != ffi::VK_SUCCESS || count == 0 {
        return false;
    }
    let mut props = vec![ffi::VkExtensionProperties::default(); count as usize];
    let result = unsafe {
        ffi::vkEnumerateDeviceExtensionProperties(
            pd,
            core::ptr::null(),
            &mut count,
            props.as_mut_ptr(),
        )
    };
    if result != ffi::VK_SUCCESS {
        return false;
    }
    let target = &ext_name[..ext_name.len() - 1];
    props.iter().any(|p| {
        let name_bytes = &p.extension_name;
        let len = name_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_bytes.len());
        &name_bytes[..len] == target
    })
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            ffi::vkDeviceWaitIdle(self.device);

            // Surfaces first: swapchains and VkSurfaceKHR must go
            // before the device/instance they were created from.
            #[cfg(feature = "render")]
            if let Some(procs) = self.surface_procs.as_ref()
                && let Ok(mut surfaces) = self.vk_surfaces.write()
            {
                for (_, entry) in surfaces.drain() {
                    for &view in &entry.views {
                        ffi::vkDestroyImageView(self.device, view, core::ptr::null());
                    }
                    (procs.destroy_swapchain)(self.device, entry.swapchain, core::ptr::null());
                    (procs.destroy_surface)(self.instance, entry.surface, core::ptr::null());
                    ffi::vkDestroyFence(self.device, entry.acquire_fence, core::ptr::null());
                    ffi::vkDestroyFence(self.device, entry.present_fence, core::ptr::null());
                    ffi::vkDestroySemaphore(self.device, entry.present_sem, core::ptr::null());
                }
            }

            // Clean up resources — write locks since we're draining.
            if let Ok(mut buffers) = self.buffers.write() {
                for (_, buf) in buffers.drain() {
                    if buf.mapped_ptr.is_some() {
                        ffi::vkUnmapMemory(self.device, buf.memory);
                    }
                    ffi::vkDestroyBuffer(self.device, buf.buffer, core::ptr::null());
                    ffi::vkFreeMemory(self.device, buf.memory, core::ptr::null());
                }
            }
            // Slice 22 — free per-tile sparse memory before
            // destroying images. The bindings registry holds
            // VkDeviceMemory allocated by sparse_map_tile. Order
            // matters: free memory before image, since the image
            // logically references the memory.
            if let Ok(mut bindings) = self.sparse_tile_bindings.write() {
                for (_, mem) in bindings.drain() {
                    ffi::vkFreeMemory(self.device, mem, core::ptr::null());
                }
            }
            // Slice 23 — destroy AS handles (must precede their
            // storage buffer free; AS objects have an implicit
            // backref to the storage buffer they were created on).
            if let Ok(mut as_map) = self.acceleration_structures.write()
                && let Some(destroy) = self.accel_destroy_fn
            {
                for (_, ax) in as_map.drain() {
                    destroy(self.device, ax.as_handle, core::ptr::null());
                    ffi::vkDestroyBuffer(self.device, ax.storage_buffer, core::ptr::null());
                    ffi::vkFreeMemory(self.device, ax.storage_memory, core::ptr::null());
                }
            }
            if let Ok(mut textures) = self.textures.write() {
                for (_, tex) in textures.drain() {
                    ffi::vkDestroyImageView(self.device, tex.view, core::ptr::null());
                    ffi::vkDestroyImage(self.device, tex.image, core::ptr::null());
                    ffi::vkFreeMemory(self.device, tex.memory, core::ptr::null());
                }
            }
            if let Ok(mut pipelines) = self.compute_pipelines.write() {
                for (_, cp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, cp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, cp.layout, core::ptr::null());
                    // descriptor_set_layout is owned by layout_cache — destroyed separately.
                }
            }
            if let Ok(mut pipelines) = self.render_pipelines.write() {
                for (_, rp) in pipelines.drain() {
                    ffi::vkDestroyPipeline(self.device, rp.pipeline, core::ptr::null());
                    ffi::vkDestroyPipelineLayout(self.device, rp.layout, core::ptr::null());
                    ffi::vkDestroyRenderPass(self.device, rp.render_pass, core::ptr::null());
                    ffi::vkDestroyDescriptorSetLayout(
                        self.device,
                        rp.descriptor_set_layout,
                        core::ptr::null(),
                    );
                }
            }
            if let Ok(mut samplers) = self.samplers.write() {
                for (_, sampler) in samplers.drain() {
                    ffi::vkDestroySampler(self.device, sampler, core::ptr::null());
                }
            }
            // Render-path sampler cache: one VkSampler per distinct desc,
            // shared across every draw/frame. Destroyed here at teardown —
            // the only place these are released (they intentionally
            // outlive individual passes).
            #[cfg(feature = "render")]
            if let Ok(mut cache) = self.render_sampler_cache.write() {
                for (_, sampler) in cache.drain() {
                    ffi::vkDestroySampler(self.device, sampler, core::ptr::null());
                }
            }
            // The compute sampler (the one sampled-read sampler, F3). Lazily
            // created, so null until the first sampled-read dispatch — destroy
            // only if one was ever built.
            #[cfg(feature = "compute")]
            if let Ok(mut sampler) = self.compute_sampler.lock()
                && !sampler.is_null()
            {
                ffi::vkDestroySampler(self.device, *sampler, core::ptr::null());
                *sampler = ffi::null_handle();
            }
            if let Ok(mut views) = self.image_views.write() {
                for (_, view) in views.drain() {
                    ffi::vkDestroyImageView(self.device, view, core::ptr::null());
                }
            }
            if let Ok(mut pools) = self.query_pools.write() {
                for (_, qp) in pools.drain() {
                    ffi::vkDestroyQueryPool(self.device, qp.pool, core::ptr::null());
                }
            }

            // Destroy cached descriptor pools.
            if let Ok(mut pools) = self.descriptor_pool_cache.lock() {
                for pool in pools.drain(..) {
                    ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
                }
            }

            // Destroy cached descriptor set layouts.
            if let Ok(mut cache) = self.layout_cache.lock() {
                for (_, layout) in cache.drain() {
                    ffi::vkDestroyDescriptorSetLayout(self.device, layout, core::ptr::null());
                }
            }

            // Drain and destroy pooled staging buffers.
            if let Ok(mut pool) = self.staging_pool.lock() {
                for (buf, mem, _) in pool.drain(..) {
                    ffi::vkDestroyBuffer(self.device, buf, core::ptr::null());
                    ffi::vkFreeMemory(self.device, mem, core::ptr::null());
                }
            }

            // Destroy pipeline cache.
            if !self.pipeline_cache.is_null() {
                ffi::vkDestroyPipelineCache(self.device, self.pipeline_cache, core::ptr::null());
            }

            // Free pooled command buffers before destroying the pool.
            let pooled: Vec<_> = self
                .cmd_buffer_pool
                .lock()
                .map(|mut pool| pool.drain(..).collect())
                .unwrap_or_default();
            if !pooled.is_empty() {
                ffi::vkFreeCommandBuffers(
                    self.device,
                    self.command_pool,
                    pooled.len() as u32,
                    pooled.as_ptr(),
                );
            }

            ffi::vkDestroyCommandPool(self.device, self.command_pool, core::ptr::null());
            ffi::vkDestroyDevice(self.device, core::ptr::null());
            ffi::vkDestroyInstance(self.instance, core::ptr::null());
        }
    }
}

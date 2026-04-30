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
    pub(super) max_push_constants_size: u32,
    // Resource storage — RwLock: dispatch/render paths take read locks; alloc/free take write locks.
    pub(super) buffers: RwLock<HashMap<u64, VkBuffer>>,
    pub(super) textures: RwLock<HashMap<u64, VkTexture>>,
    pub(super) compute_pipelines: RwLock<HashMap<u64, VkComputePipeline>>,
    pub(super) render_pipelines: RwLock<HashMap<u64, VkRenderPipeline>>,
    pub(super) samplers: RwLock<HashMap<u64, ffi::VkSampler>>,
    /// Standalone image views created via texture_view_create (not tied to a full VkTexture).
    pub(super) image_views: RwLock<HashMap<u64, ffi::VkImageView>>,
    pub(super) query_pools: RwLock<HashMap<u64, VkQueryPool>>,
    pub(super) queues: RwLock<HashMap<u64, ffi::VkQueue>>,
    pub(super) next_handle: AtomicU64,
    /// Pool of reusable command buffers — Arc<Mutex> for sharing with Pulse closures.
    pub(super) cmd_buffer_pool: std::sync::Arc<Mutex<Vec<ffi::VkCommandBuffer>>>,
    /// Pool of reusable descriptor pools — avoids create/destroy per dispatch.
    pub(super) descriptor_pool_cache: Mutex<Vec<ffi::VkDescriptorPool>>,
    /// Pool of reusable staging buffers — avoids alloc/free per texture upload.
    pub(super) staging_pool: Mutex<Vec<(ffi::VkBuffer, ffi::VkDeviceMemory, usize)>>,
    /// Cache of descriptor set layouts keyed by binding count — avoids re-creation.
    pub(super) layout_cache: Mutex<HashMap<u32, ffi::VkDescriptorSetLayout>>,
    /// Indirect command buffers (steps 032 + 033). Stores recorded
    /// dispatches that `indirect_buffer_execute` replays sequentially
    /// on the same compute path used by `wave_dispatch`. The Lean
    /// `T7000` equivalence theorem is parametric in the per-command
    /// transformer, so this list-of-dispatches refinement satisfies
    /// the proof contract on every Vulkan implementation.
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
pub(super) enum VkIcbCommand {
    Dispatch {
        wave_handle: u64,
        bindings: [u64; crate::api::wave::MAX_BINDINGS],
        binding_count: u8,
        push_data: [u8; crate::api::wave::PUSH_DATA_CAP],
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

pub(super) struct VkComputePipeline {
    pub(super) pipeline: ffi::VkPipeline,
    pub(super) layout: ffi::VkPipelineLayout,
    pub(super) descriptor_set_layout: ffi::VkDescriptorSetLayout,
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
        let pool_size = ffi::VkDescriptorPoolSize {
            ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptor_count: 16,
        };
        let pool_info = ffi::VkDescriptorPoolCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            max_sets: 1,
            pool_size_count: 1,
            p_pool_sizes: &pool_size,
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

    /// Acquire a descriptor set layout for compute (storage buffers only), cached by binding count.
    pub(super) fn acquire_descriptor_set_layout(
        &self,
        binding_count: u32,
    ) -> Result<ffi::VkDescriptorSetLayout, QuantaError> {
        {
            let cache = self
                .layout_cache
                .lock()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            if let Some(&layout) = cache.get(&binding_count) {
                return Ok(layout);
            }
        }
        // Cache miss — create a new layout.
        let mut bindings = alloc::vec::Vec::new();
        for i in 0..binding_count {
            bindings.push(ffi::VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
            .insert(binding_count, layout);
        Ok(layout)
    }

    /// Return a descriptor pool to the cache for reuse.
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
        if let Ok(mut pool) = self.staging_pool.lock() {
            if let Some(idx) = pool.iter().position(|&(_, _, cap)| cap >= min_size) {
                return Ok(pool.swap_remove(idx));
            }
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

        let device = self.device;
        let pool = self.cmd_buffer_pool.clone();
        let handle = self.alloc_handle();
        Ok(Pulse {
            handle,
            completed: false,
            wait_fn: Some(Box::new(move || unsafe {
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

    let create_info = ffi::VkInstanceCreateInfo {
        s_type: ffi::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: core::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: core::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: core::ptr::null(),
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

        let mut enabled_extensions: Vec<*const core::ffi::c_char> = Vec::new();
        if has_vrs_ext {
            enabled_extensions
                .push(b"VK_KHR_fragment_shading_rate\0".as_ptr() as *const core::ffi::c_char);
        }
        if has_mesh_ext {
            enabled_extensions.push(b"VK_EXT_mesh_shader\0".as_ptr() as *const core::ffi::c_char);
        }
        if has_rt {
            enabled_extensions
                .push(b"VK_KHR_acceleration_structure\0".as_ptr() as *const core::ffi::c_char);
            enabled_extensions
                .push(b"VK_KHR_ray_tracing_pipeline\0".as_ptr() as *const core::ffi::c_char);
            // Both ray-tracing extensions require deferred-host-ops.
            enabled_extensions
                .push(b"VK_KHR_deferred_host_operations\0".as_ptr() as *const core::ffi::c_char);
            // VK_KHR_acceleration_structure requires VK_KHR_buffer_device_address.
            enabled_extensions
                .push(b"VK_KHR_buffer_device_address\0".as_ptr() as *const core::ffi::c_char);
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

        let device_create = ffi::VkDeviceCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            p_next: &sync2_features as *const _ as *const core::ffi::c_void,
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
            buffers: RwLock::new(HashMap::new()),
            textures: RwLock::new(HashMap::new()),
            compute_pipelines: RwLock::new(HashMap::new()),
            render_pipelines: RwLock::new(HashMap::new()),
            samplers: RwLock::new(HashMap::new()),
            image_views: RwLock::new(HashMap::new()),
            query_pools: RwLock::new(HashMap::new()),
            queues: RwLock::new(HashMap::new()),
            next_handle: AtomicU64::new(0),
            cmd_buffer_pool: std::sync::Arc::new(Mutex::new(Vec::new())),
            descriptor_pool_cache: Mutex::new(Vec::new()),
            staging_pool: Mutex::new(Vec::new()),
            layout_cache: Mutex::new(HashMap::new()),
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
            sparse_tile_bindings: RwLock::new(HashMap::new()),
            buffer_device_address_enabled: has_accel_ext,
            acceleration_structures: RwLock::new(HashMap::new()),
        }));

        break; // Use first suitable device
    }

    devices
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

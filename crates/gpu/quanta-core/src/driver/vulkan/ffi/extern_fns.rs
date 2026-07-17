//! Platform-gated Vulkan entry-point bindings.
//!
//! [`vk_fns!`] holds the ~100 signatures once and hands them to a
//! per-platform emitter macro:
//!
//! - Linux / Android / macOS (MoltenVK under `vulkan-portability`):
//!   `vk_emit_link!` — one link-time `unsafe extern "C"` block against
//!   `libvulkan`.
//! - Windows: `vk_emit_runtime!` — a function-pointer table resolved at
//!   runtime from `vulkan-1.dll` (`LoadLibraryA` + `GetProcAddress`)
//!   behind identically-named unsafe-fn shims, so call sites read the
//!   same on every platform. Link-time binding is wrong on Windows
//!   twice over: the import library `vulkan-1.lib` ships only with the
//!   Vulkan SDK (making the SDK a build dependency), and a link-bound
//!   app fails at *process load* on a machine without `vulkan-1.dll` —
//!   foreclosing the software fallback before discovery can run.
//!   Runtime loading converts both failures into [`ensure_loaded`],
//!   which `discover()` gates on with a loud init line.

use core::ffi::c_void;

use super::constants::*;
use super::device::*;
use super::structs::*;
use super::structs_render::*;

/// The Vulkan entry points quanta binds, listed once. `$emit` receives
/// the full list and chooses the binding form.
macro_rules! vk_fns {
    ($emit:ident) => {
        $emit! {
            pub fn vkCreateInstance(
                create_info: *const VkInstanceCreateInfo,
                allocator: *const c_void,
                instance: *mut VkInstance,
            ) -> VkResult;
            pub fn vkDestroyInstance(instance: VkInstance, allocator: *const c_void);
            pub fn vkEnumeratePhysicalDevices(
                instance: VkInstance,
                count: *mut u32,
                devices: *mut VkPhysicalDevice,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceProperties(
                device: VkPhysicalDevice,
                properties: *mut VkPhysicalDeviceProperties,
            );
            pub fn vkGetPhysicalDeviceMemoryProperties(
                device: VkPhysicalDevice,
                props: *mut VkPhysicalDeviceMemoryProperties,
            );
            pub fn vkGetPhysicalDeviceQueueFamilyProperties(
                device: VkPhysicalDevice,
                count: *mut u32,
                properties: *mut VkQueueFamilyProperties,
            );
            pub fn vkCreateDevice(
                physical_device: VkPhysicalDevice,
                create_info: *const VkDeviceCreateInfo,
                allocator: *const c_void,
                device: *mut VkDevice,
            ) -> VkResult;
            pub fn vkDestroyDevice(device: VkDevice, allocator: *const c_void);
            pub fn vkDeviceWaitIdle(device: VkDevice) -> VkResult;
            /// Resolve an extension function pointer at runtime.
            /// Returns null when the extension wasn't enabled at
            /// device creation. Required because extension symbols
            /// (`vkCmdSetFragmentShadingRateKHR`, `vkCmdDrawMeshTasksEXT`,
            /// `vkCmdTraceRaysKHR`, …) are not part of the core
            /// Vulkan ABI and cannot be link-time-resolved.
            pub fn vkGetDeviceProcAddr(
                device: VkDevice,
                name: *const core::ffi::c_char,
            ) -> *const c_void;
            /// Resolve an instance-level extension function pointer
            /// (commands taking a `VkPhysicalDevice` such as
            /// `vkGetPhysicalDeviceFragmentShadingRatesKHR`). The
            /// loader provides this as a standard export.
            pub fn vkGetInstanceProcAddr(
                instance: VkInstance,
                name: *const core::ffi::c_char,
            ) -> *const c_void;
            pub fn vkGetDeviceQueue(device: VkDevice, family: u32, index: u32, queue: *mut VkQueue);
            pub fn vkCreateCommandPool(
                device: VkDevice,
                create_info: *const VkCommandPoolCreateInfo,
                allocator: *const c_void,
                pool: *mut VkCommandPool,
            ) -> VkResult;
            pub fn vkDestroyCommandPool(device: VkDevice, pool: VkCommandPool, allocator: *const c_void);
            pub fn vkAllocateCommandBuffers(
                device: VkDevice,
                alloc_info: *const VkCommandBufferAllocateInfo,
                cmd_bufs: *mut VkCommandBuffer,
            ) -> VkResult;
            pub fn vkFreeCommandBuffers(
                device: VkDevice,
                pool: VkCommandPool,
                count: u32,
                cmd_bufs: *const VkCommandBuffer,
            );
            pub fn vkBeginCommandBuffer(
                cmd_buf: VkCommandBuffer,
                begin_info: *const VkCommandBufferBeginInfo,
            ) -> VkResult;
            pub fn vkEndCommandBuffer(cmd_buf: VkCommandBuffer) -> VkResult;
            pub fn vkResetCommandBuffer(cmd_buf: VkCommandBuffer, flags: u32) -> VkResult;
            pub fn vkCreateBuffer(
                device: VkDevice,
                create_info: *const VkBufferCreateInfo,
                allocator: *const c_void,
                buffer: *mut VkBuffer,
            ) -> VkResult;
            pub fn vkDestroyBuffer(device: VkDevice, buffer: VkBuffer, allocator: *const c_void);
            pub fn vkAllocateMemory(
                device: VkDevice,
                alloc_info: *const VkMemoryAllocateInfo,
                allocator: *const c_void,
                memory: *mut VkDeviceMemory,
            ) -> VkResult;
            pub fn vkFreeMemory(device: VkDevice, memory: VkDeviceMemory, allocator: *const c_void);
            pub fn vkBindBufferMemory(
                device: VkDevice,
                buffer: VkBuffer,
                memory: VkDeviceMemory,
                offset: VkDeviceSize,
            ) -> VkResult;
            pub fn vkMapMemory(
                device: VkDevice,
                memory: VkDeviceMemory,
                offset: VkDeviceSize,
                size: VkDeviceSize,
                flags: u32,
                data: *mut *mut c_void,
            ) -> VkResult;
            pub fn vkUnmapMemory(device: VkDevice, memory: VkDeviceMemory);
            pub fn vkGetBufferMemoryRequirements(
                device: VkDevice,
                buffer: VkBuffer,
                reqs: *mut VkMemoryRequirements,
            );
            pub fn vkCreateImage(
                device: VkDevice,
                create_info: *const VkImageCreateInfo,
                allocator: *const c_void,
                image: *mut VkImage,
            ) -> VkResult;
            pub fn vkDestroyImage(device: VkDevice, image: VkImage, allocator: *const c_void);
            pub fn vkGetImageMemoryRequirements(
                device: VkDevice,
                image: VkImage,
                reqs: *mut VkMemoryRequirements,
            );
            pub fn vkBindImageMemory(
                device: VkDevice,
                image: VkImage,
                memory: VkDeviceMemory,
                offset: VkDeviceSize,
            ) -> VkResult;
            pub fn vkCreateImageView(
                device: VkDevice,
                create_info: *const VkImageViewCreateInfo,
                allocator: *const c_void,
                view: *mut VkImageView,
            ) -> VkResult;
            pub fn vkDestroyImageView(device: VkDevice, view: VkImageView, allocator: *const c_void);
            pub fn vkCreateShaderModule(
                device: VkDevice,
                create_info: *const VkShaderModuleCreateInfo,
                allocator: *const c_void,
                module: *mut VkShaderModule,
            ) -> VkResult;
            pub fn vkDestroyShaderModule(
                device: VkDevice,
                module: VkShaderModule,
                allocator: *const c_void,
            );
            pub fn vkCreatePipelineLayout(
                device: VkDevice,
                create_info: *const VkPipelineLayoutCreateInfo,
                allocator: *const c_void,
                layout: *mut VkPipelineLayout,
            ) -> VkResult;
            pub fn vkDestroyPipelineLayout(
                device: VkDevice,
                layout: VkPipelineLayout,
                allocator: *const c_void,
            );
            pub fn vkCreatePipelineCache(
                device: VkDevice,
                create_info: *const VkPipelineCacheCreateInfo,
                allocator: *const c_void,
                pipeline_cache: *mut VkPipelineCache,
            ) -> VkResult;
            pub fn vkDestroyPipelineCache(
                device: VkDevice,
                pipeline_cache: VkPipelineCache,
                allocator: *const c_void,
            );
            pub fn vkCreateComputePipelines(
                device: VkDevice,
                cache: VkPipelineCache,
                count: u32,
                create_infos: *const VkComputePipelineCreateInfo,
                allocator: *const c_void,
                pipelines: *mut VkPipeline,
            ) -> VkResult;
            pub fn vkCreateGraphicsPipelines(
                device: VkDevice,
                cache: VkPipelineCache,
                count: u32,
                create_infos: *const VkGraphicsPipelineCreateInfo,
                allocator: *const c_void,
                pipelines: *mut VkPipeline,
            ) -> VkResult;
            pub fn vkDestroyPipeline(device: VkDevice, pipeline: VkPipeline, allocator: *const c_void);
            pub fn vkCreateRenderPass(
                device: VkDevice,
                create_info: *const VkRenderPassCreateInfo,
                allocator: *const c_void,
                render_pass: *mut VkRenderPass,
            ) -> VkResult;
            pub fn vkDestroyRenderPass(
                device: VkDevice,
                render_pass: VkRenderPass,
                allocator: *const c_void,
            );
            pub fn vkCreateFramebuffer(
                device: VkDevice,
                create_info: *const VkFramebufferCreateInfo,
                allocator: *const c_void,
                framebuffer: *mut VkFramebuffer,
            ) -> VkResult;
            pub fn vkDestroyFramebuffer(
                device: VkDevice,
                framebuffer: VkFramebuffer,
                allocator: *const c_void,
            );
            pub fn vkCreateDescriptorSetLayout(
                device: VkDevice,
                create_info: *const VkDescriptorSetLayoutCreateInfo,
                allocator: *const c_void,
                set_layout: *mut VkDescriptorSetLayout,
            ) -> VkResult;
            pub fn vkDestroyDescriptorSetLayout(
                device: VkDevice,
                layout: VkDescriptorSetLayout,
                allocator: *const c_void,
            );
            pub fn vkCreateDescriptorPool(
                device: VkDevice,
                create_info: *const VkDescriptorPoolCreateInfo,
                allocator: *const c_void,
                pool: *mut VkDescriptorPool,
            ) -> VkResult;
            pub fn vkDestroyDescriptorPool(
                device: VkDevice,
                pool: VkDescriptorPool,
                allocator: *const c_void,
            );
            pub fn vkResetDescriptorPool(
                device: VkDevice,
                pool: VkDescriptorPool,
                flags: u32,
            ) -> VkResult;
            pub fn vkAllocateDescriptorSets(
                device: VkDevice,
                alloc_info: *const VkDescriptorSetAllocateInfo,
                sets: *mut VkDescriptorSet,
            ) -> VkResult;
            pub fn vkUpdateDescriptorSets(
                device: VkDevice,
                write_count: u32,
                writes: *const VkWriteDescriptorSet,
                copy_count: u32,
                copies: *const c_void,
            );
            pub fn vkCreateSampler(
                device: VkDevice,
                create_info: *const VkSamplerCreateInfo,
                allocator: *const c_void,
                sampler: *mut VkSampler,
            ) -> VkResult;
            pub fn vkDestroySampler(device: VkDevice, sampler: VkSampler, allocator: *const c_void);
            pub fn vkCreateFence(
                device: VkDevice,
                create_info: *const VkFenceCreateInfo,
                allocator: *const c_void,
                fence: *mut VkFence,
            ) -> VkResult;
            pub fn vkDestroyFence(device: VkDevice, fence: VkFence, allocator: *const c_void);
            pub fn vkWaitForFences(
                device: VkDevice,
                count: u32,
                fences: *const VkFence,
                wait_all: u32,
                timeout: u64,
            ) -> VkResult;
            pub fn vkResetFences(device: VkDevice, count: u32, fences: *const VkFence) -> VkResult;
            pub fn vkCreateSemaphore(
                device: VkDevice,
                create_info: *const VkSemaphoreCreateInfo,
                allocator: *const c_void,
                semaphore: *mut VkSemaphore,
            ) -> VkResult;
            pub fn vkDestroySemaphore(
                device: VkDevice,
                semaphore: VkSemaphore,
                allocator: *const c_void,
            );
            pub fn vkEnumerateInstanceExtensionProperties(
                layer_name: *const u8,
                count: *mut u32,
                properties: *mut VkExtensionProperties,
            ) -> VkResult;
            pub fn vkQueueSubmit(
                queue: VkQueue,
                count: u32,
                submits: *const VkSubmitInfo,
                fence: VkFence,
            ) -> VkResult;
            pub fn vkQueueWaitIdle(queue: VkQueue) -> VkResult;
            pub fn vkCmdBindPipeline(cmd_buf: VkCommandBuffer, bind_point: u32, pipeline: VkPipeline);
            pub fn vkCmdBindDescriptorSets(
                cmd_buf: VkCommandBuffer,
                bind_point: u32,
                layout: VkPipelineLayout,
                first_set: u32,
                count: u32,
                sets: *const VkDescriptorSet,
                dyn_offset_count: u32,
                dyn_offsets: *const u32,
            );
            pub fn vkCmdPushConstants(
                cmd_buf: VkCommandBuffer,
                layout: VkPipelineLayout,
                stage_flags: u32,
                offset: u32,
                size: u32,
                p_values: *const c_void,
            );
            pub fn vkCmdDispatch(cmd_buf: VkCommandBuffer, x: u32, y: u32, z: u32);
            pub fn vkCmdDispatchIndirect(cmd_buf: VkCommandBuffer, buffer: VkBuffer, offset: VkDeviceSize);
            pub fn vkCmdExecuteCommands(
                cmd_buf: VkCommandBuffer,
                count: u32,
                cmd_buffers: *const VkCommandBuffer,
            );
            pub fn vkCmdBeginRenderPass(
                cmd_buf: VkCommandBuffer,
                begin_info: *const VkRenderPassBeginInfo,
                contents: u32,
            );
            pub fn vkCmdEndRenderPass(cmd_buf: VkCommandBuffer);
            pub fn vkCmdSetViewport(
                cmd_buf: VkCommandBuffer,
                first: u32,
                count: u32,
                viewports: *const VkViewport,
            );
            pub fn vkCmdSetScissor(
                cmd_buf: VkCommandBuffer,
                first: u32,
                count: u32,
                scissors: *const VkRect2D,
            );
            pub fn vkCmdSetStencilReference(cmd_buf: VkCommandBuffer, face_mask: u32, reference: u32);
            pub fn vkCmdBindVertexBuffers(
                cmd_buf: VkCommandBuffer,
                first: u32,
                count: u32,
                buffers: *const VkBuffer,
                offsets: *const VkDeviceSize,
            );
            pub fn vkCmdBindIndexBuffer(
                cmd_buf: VkCommandBuffer,
                buffer: VkBuffer,
                offset: VkDeviceSize,
                index_type: u32,
            );
            pub fn vkCmdDraw(
                cmd_buf: VkCommandBuffer,
                vertex_count: u32,
                instance_count: u32,
                first_vertex: u32,
                first_instance: u32,
            );
            pub fn vkCmdDrawIndexed(
                cmd_buf: VkCommandBuffer,
                index_count: u32,
                instance_count: u32,
                first_index: u32,
                vertex_offset: i32,
                first_instance: u32,
            );
            pub fn vkCmdDrawIndirect(
                cmd_buf: VkCommandBuffer,
                buffer: VkBuffer,
                offset: VkDeviceSize,
                draw_count: u32,
                stride: u32,
            );
            pub fn vkCmdDrawIndexedIndirect(
                cmd_buf: VkCommandBuffer,
                buffer: VkBuffer,
                offset: VkDeviceSize,
                draw_count: u32,
                stride: u32,
            );
            pub fn vkCmdCopyBuffer(
                cmd_buf: VkCommandBuffer,
                src: VkBuffer,
                dst: VkBuffer,
                region_count: u32,
                regions: *const VkBufferCopy,
            );
            pub fn vkCmdCopyBufferToImage(
                cmd_buf: VkCommandBuffer,
                src_buffer: VkBuffer,
                dst_image: VkImage,
                dst_image_layout: u32,
                region_count: u32,
                regions: *const VkBufferImageCopy,
            );
            pub fn vkCmdCopyImageToBuffer(
                cmd_buf: VkCommandBuffer,
                src_image: VkImage,
                src_image_layout: u32,
                dst_buffer: VkBuffer,
                region_count: u32,
                regions: *const VkBufferImageCopy,
            );
            pub fn vkCmdBlitImage(
                cmd_buf: VkCommandBuffer,
                src_image: VkImage,
                src_image_layout: u32,
                dst_image: VkImage,
                dst_image_layout: u32,
                region_count: u32,
                regions: *const VkImageBlit,
                filter: u32,
            );
            pub fn vkCmdPipelineBarrier(
                cmd_buf: VkCommandBuffer,
                src_stage_mask: u32,
                dst_stage_mask: u32,
                dependency_flags: u32,
                memory_barrier_count: u32,
                p_memory_barriers: *const c_void,
                buffer_memory_barrier_count: u32,
                p_buffer_memory_barriers: *const c_void,
                image_memory_barrier_count: u32,
                p_image_memory_barriers: *const VkImageMemoryBarrier,
            );
            pub fn vkCmdPipelineBarrier2(cmd_buf: VkCommandBuffer, dep_info: *const VkDependencyInfo);
            pub fn vkCreateQueryPool(
                device: VkDevice,
                create_info: *const VkQueryPoolCreateInfo,
                allocator: *const c_void,
                query_pool: *mut VkQueryPool,
            ) -> VkResult;
            pub fn vkDestroyQueryPool(device: VkDevice, query_pool: VkQueryPool, allocator: *const c_void);
            pub fn vkCmdResetQueryPool(
                cmd_buf: VkCommandBuffer,
                query_pool: VkQueryPool,
                first_query: u32,
                query_count: u32,
            );
            pub fn vkCmdWriteTimestamp(
                cmd_buf: VkCommandBuffer,
                pipeline_stage: u32,
                query_pool: VkQueryPool,
                query: u32,
            );
            pub fn vkGetQueryPoolResults(
                device: VkDevice,
                query_pool: VkQueryPool,
                first_query: u32,
                query_count: u32,
                data_size: usize,
                p_data: *mut c_void,
                stride: VkDeviceSize,
                flags: u32,
            ) -> VkResult;
            pub fn vkCmdResolveImage(
                cmd_buf: VkCommandBuffer,
                src_image: VkImage,
                src_image_layout: u32,
                dst_image: VkImage,
                dst_image_layout: u32,
                region_count: u32,
                p_regions: *const VkImageResolve,
            );
            pub fn vkCmdBeginQuery(
                cmd_buf: VkCommandBuffer,
                query_pool: VkQueryPool,
                query: u32,
                flags: u32,
            );
            pub fn vkCmdEndQuery(cmd_buf: VkCommandBuffer, query_pool: VkQueryPool, query: u32);
            pub fn vkEnumerateDeviceExtensionProperties(
                physical_device: VkPhysicalDevice,
                p_layer_name: *const u8,
                p_property_count: *mut u32,
                p_properties: *mut VkExtensionProperties,
            ) -> VkResult;
            pub fn vkGetPhysicalDeviceFeatures(
                physical_device: VkPhysicalDevice,
                p_features: *mut VkPhysicalDeviceFeatures,
            );
            /// Per-aspect sparse memory requirements (granularity in
            /// pixels, mip-tail layout). Step 063 slice 22.
            pub fn vkGetImageSparseMemoryRequirements(
                device: VkDevice,
                image: VkImage,
                p_count: *mut u32,
                p_requirements: *mut VkSparseImageMemoryRequirements,
            );
            /// Resolve a buffer to its GPU device address.
            /// Vulkan 1.2 core; the underlying buffer must have been
            /// created with VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            /// and the device must have bufferDeviceAddress enabled.
            /// Required by acceleration-structure builds (vertex /
            /// index / scratch / AS-storage buffers all reference
            /// each other by device address). Step 063 slice 23.
            pub fn vkGetBufferDeviceAddress(
                device: VkDevice,
                p_info: *const VkBufferDeviceAddressInfo,
            ) -> u64;
            /// Submit sparse-bind operations to a queue.
            /// Step 063 slice 16. Only the image-bind path is
            /// exercised; buffer/opaque arrays stay zeroed in the
            /// argument struct.
            pub fn vkQueueBindSparse(
                queue: VkQueue,
                bind_info_count: u32,
                p_bind_info: *const VkBindSparseInfo,
                fence: VkFence,
            ) -> VkResult;
        }
    };
}

/// Link-time form: the whole list as one extern block against
/// `libvulkan`.
///
/// macOS links against MoltenVK's `libvulkan`, but only under
/// `vulkan-portability`. This whole file compiles only when the Vulkan
/// module does — and on macOS that requires the feature — so this
/// stanza never reaches a plain Apple build (the MoltenVK link trap):
/// the explicit feature gate keeps that contract legible at the link
/// site.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(feature = "vulkan-portability", target_os = "macos"),
))]
macro_rules! vk_emit_link {
    ( $( $(#[$meta:meta])* pub fn $name:ident( $($arg:ident: $argty:ty),* $(,)? ) $(-> $ret:ty)?; )* ) => {
        #[link(name = "vulkan")]
        unsafe extern "C" {
            $( $(#[$meta])* pub fn $name( $($arg: $argty),* ) $(-> $ret)?; )*
        }
    };
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    all(feature = "vulkan-portability", target_os = "macos"),
))]
vk_fns! { vk_emit_link }

/// Runtime form (Windows): a table of function pointers resolved from
/// the loader DLL, behind shims named exactly like the link-time
/// externs so call sites never see the difference.
#[cfg(target_os = "windows")]
macro_rules! vk_emit_runtime {
    ( $( $(#[$meta:meta])* pub fn $name:ident( $($arg:ident: $argty:ty),* $(,)? ) $(-> $ret:ty)?; )* ) => {
        /// Every entry point quanta binds, resolved once by [`load`].
        /// All of them are loader exports (core Vulkan ≤ 1.3), so
        /// resolution is all-or-nothing: a missing export fails the
        /// whole load with that symbol's name in the error.
        #[allow(non_snake_case)]
        struct VkFns {
            $( $name: unsafe extern "C" fn( $($argty),* ) $(-> $ret)?, )*
        }

        /// Resolve the full table from a loaded loader module.
        ///
        /// # Safety
        /// `module` must be a live module handle for a Vulkan loader.
        unsafe fn resolve_table(
            module: *mut c_void,
            dll: &str,
        ) -> Result<VkFns, alloc::string::String> {
            Ok(VkFns {
                $( $name: {
                    let sym = concat!(stringify!($name), "\0");
                    // SAFETY: `module` is live per the contract above;
                    // `sym` is NUL-terminated.
                    let p = unsafe {
                        GetProcAddress(module, sym.as_ptr() as *const core::ffi::c_char)
                    };
                    if p.is_null() {
                        return Err(alloc::format!(
                            concat!(
                                "{} has no export `",
                                stringify!($name),
                                "` (a Vulkan 1.3 loader is required)"
                            ),
                            dll,
                        ));
                    }
                    // SAFETY: the loader exports this symbol with the
                    // declared core-Vulkan ABI.
                    unsafe {
                        core::mem::transmute::<
                            *mut c_void,
                            unsafe extern "C" fn( $($argty),* ) $(-> $ret)?,
                        >(p)
                    }
                }, )*
            })
        }

        $(
            $(#[$meta])*
            #[allow(non_snake_case, clippy::missing_safety_doc, clippy::too_many_arguments)]
            pub unsafe fn $name( $($arg: $argty),* ) $(-> $ret)? {
                // SAFETY: same contract as the link-time extern — the
                // caller upholds the Vulkan API requirements.
                unsafe { (fns().$name)( $($arg),* ) }
            }
        )*
    };
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryA(file_name: *const core::ffi::c_char) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const core::ffi::c_char) -> *mut c_void;
}

#[cfg(target_os = "windows")]
vk_fns! { vk_emit_runtime }

/// The one loaded table. `Err` caches the failure so every
/// [`ensure_loaded`] call reports the same message without retrying
/// `LoadLibrary`.
#[cfg(target_os = "windows")]
static VK_TABLE: std::sync::OnceLock<Result<VkFns, alloc::string::String>> =
    std::sync::OnceLock::new();

/// Load the loader DLL and resolve the whole entry-point table.
#[cfg(target_os = "windows")]
fn load() -> Result<VkFns, alloc::string::String> {
    // Diagnostic lever: point at an alternate loader DLL — or at a
    // nonexistent name, to exercise the missing-loader fallback on a
    // machine that has Vulkan. Same role QUANTA_BACKEND plays for
    // discovery: making the failure path deterministically reachable.
    let dll = std::env::var("QUANTA_VULKAN_LOADER")
        .unwrap_or_else(|_| alloc::string::String::from("vulkan-1.dll"));
    let mut name_z = alloc::vec::Vec::with_capacity(dll.len() + 1);
    name_z.extend_from_slice(dll.as_bytes());
    name_z.push(0);
    // SAFETY: `name_z` is NUL-terminated.
    let module = unsafe { LoadLibraryA(name_z.as_ptr() as *const core::ffi::c_char) };
    if module.is_null() {
        return Err(alloc::format!(
            "{dll} not found (no Vulkan loader on this machine)"
        ));
    }
    // SAFETY: `module` is the handle LoadLibraryA just returned; it is
    // never freed — the table borrows from it for the process lifetime.
    unsafe { resolve_table(module, &dll) }
}

/// Gate for discovery: load the loader and resolve the table. `Err`
/// carries the human-readable missing piece (`vulkan-1.dll not found…`,
/// `…has no export…`) for the loud init line in `discover()`.
#[cfg(target_os = "windows")]
pub fn ensure_loaded() -> Result<(), alloc::string::String> {
    match VK_TABLE.get_or_init(load) {
        Ok(_) => Ok(()),
        Err(e) => Err(e.clone()),
    }
}

/// Link-time platforms: binding cannot fail at runtime (a missing
/// loader fails at process load instead), so the gate is a no-op.
#[cfg(not(target_os = "windows"))]
pub fn ensure_loaded() -> Result<(), alloc::string::String> {
    Ok(())
}

/// The resolved table, for the shims. Discovery gates on
/// [`ensure_loaded`], so an unresolvable table here is a driver bug —
/// panic with the load error rather than limp on.
#[cfg(target_os = "windows")]
fn fns() -> &'static VkFns {
    match VK_TABLE.get_or_init(load) {
        Ok(t) => t,
        Err(e) => panic!(
            "quanta vulkan: {e} — an entry point was called although the \
             loader is unavailable (discovery gates on ensure_loaded)"
        ),
    }
}

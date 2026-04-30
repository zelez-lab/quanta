//! Platform-gated extern function declarations.
//!
//! Uses a macro to avoid triplicating the ~88 function signatures
//! across Linux, macOS, and Windows.

use core::ffi::c_void;

use super::constants::*;
use super::device::*;
use super::structs::*;
use super::structs_render::*;

macro_rules! vk_extern_fns {
    ($($link_attr:tt)*) => {
        $($link_attr)*
        unsafe extern "C" {
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
        }
    };
}

#[cfg(any(target_os = "linux", target_os = "android"))]
vk_extern_fns! { #[link(name = "vulkan")] }

#[cfg(target_os = "macos")]
vk_extern_fns! { #[link(name = "vulkan")] }

#[cfg(target_os = "windows")]
vk_extern_fns! { #[link(name = "vulkan-1")] }

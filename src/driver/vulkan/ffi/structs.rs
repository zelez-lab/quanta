//! Vulkan #[repr(C)] structures — create infos, memory, images, descriptors.

use core::ffi::{c_char, c_void};

use super::constants::*;

// ============================================================================
// Structures
// ============================================================================

#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub p_application_name: *const c_char,
    pub application_version: u32,
    pub p_engine_name: *const c_char,
    pub engine_version: u32,
    pub api_version: u32,
}

#[repr(C)]
pub struct VkInstanceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_application_info: *const VkApplicationInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
pub struct VkDeviceQueueCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
    pub queue_count: u32,
    pub p_queue_priorities: *const f32,
}

#[repr(C)]
pub struct VkDeviceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_create_info_count: u32,
    pub p_queue_create_infos: *const VkDeviceQueueCreateInfo,
    pub enabled_layer_count: u32,
    pub pp_enabled_layer_names: *const *const c_char,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const c_char,
    pub p_enabled_features: *const c_void,
}

#[repr(C)]
pub struct VkCommandPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub queue_family_index: u32,
}

#[repr(C)]
pub struct VkCommandBufferAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub command_pool: VkCommandPool,
    pub level: u32,
    pub command_buffer_count: u32,
}

#[repr(C)]
pub struct VkCommandBufferBeginInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub p_inheritance_info: *const c_void,
}

/// Inheritance info passed to a secondary command buffer's begin
/// info. For compute-only secondaries, all handle fields are null
/// and `subpass` / `framebuffer` are unused. For render-pass-
/// continued secondaries (steps 032 + 033 render path),
/// `render_pass` and `subpass` must match the primary's pass.
#[repr(C)]
pub struct VkCommandBufferInheritanceInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub render_pass: super::constants::VkRenderPass,
    pub subpass: u32,
    pub framebuffer: super::constants::VkFramebuffer,
    pub occlusion_query_enable: u32,
    pub query_flags: u32,
    pub pipeline_statistics: u32,
}

#[repr(C)]
pub struct VkBufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub size: VkDeviceSize,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
}

#[repr(C)]
pub struct VkMemoryAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub allocation_size: VkDeviceSize,
    pub memory_type_index: u32,
}

#[repr(C)]
pub struct VkMemoryRequirements {
    pub size: VkDeviceSize,
    pub alignment: VkDeviceSize,
    pub memory_type_bits: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkMemoryType {
    pub property_flags: u32,
    pub heap_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkMemoryHeap {
    pub size: VkDeviceSize,
    pub flags: u32,
}

#[repr(C)]
pub struct VkPhysicalDeviceMemoryProperties {
    pub memory_type_count: u32,
    pub memory_types: [VkMemoryType; 32],
    pub memory_heap_count: u32,
    pub memory_heaps: [VkMemoryHeap; 16],
}

#[repr(C)]
pub struct VkImageCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub image_type: u32,
    pub format: u32,
    pub extent: VkExtent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,
    pub tiling: u32,
    pub usage: u32,
    pub sharing_mode: u32,
    pub queue_family_index_count: u32,
    pub p_queue_family_indices: *const u32,
    pub initial_layout: u32,
}

#[repr(C)]
pub struct VkImageViewCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub image: VkImage,
    pub view_type: u32,
    pub format: u32,
    pub components: VkComponentMapping,
    pub subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkComponentMapping {
    pub r: u32,
    pub g: u32,
    pub b: u32,
    pub a: u32,
}

impl Default for VkComponentMapping {
    fn default() -> Self {
        // VK_COMPONENT_SWIZZLE_IDENTITY = 0
        Self {
            r: 0,
            g: 0,
            b: 0,
            a: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkImageSubresourceRange {
    pub aspect_mask: u32,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkImageSubresourceLayers {
    pub aspect_mask: u32,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkExtent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkExtent2D {
    pub width: u32,
    pub height: u32,
}

/// VK_KHR_fragment_shading_rate (step 063 slice 14) — one entry in
/// the supported-rate list returned by
/// `vkGetPhysicalDeviceFragmentShadingRatesKHR`. `sample_counts` is
/// a `VkSampleCountFlags` bitmask for which sample counts each rate
/// supports; we only check that it's non-zero today.
#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct VkPhysicalDeviceFragmentShadingRateKHR {
    pub s_type: u32,
    pub p_next: *mut c_void,
    pub sample_counts: u32,
    pub fragment_size: VkExtent2D,
}

/// `VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR`.
pub const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_KHR: u32 = 1000226001;

impl Default for VkExtent2D {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkOffset2D {
    pub x: i32,
    pub y: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkOffset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkRect2D {
    pub offset: VkOffset2D,
    pub extent: VkExtent2D,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

#[repr(C)]
pub struct VkShaderModuleCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub code_size: usize,
    pub p_code: *const u32,
}

#[repr(C)]
pub struct VkPipelineShaderStageCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: u32,
    pub module: VkShaderModule,
    pub p_name: *const c_char,
    pub p_specialization_info: *const c_void,
}

// ─── Specialization constants ──────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkSpecializationMapEntry {
    pub constant_id: u32,
    pub offset: u32,
    pub size: usize,
}

#[repr(C)]
pub struct VkSpecializationInfo {
    pub map_entry_count: u32,
    pub p_map_entries: *const VkSpecializationMapEntry,
    pub data_size: usize,
    pub p_data: *const c_void,
}

#[repr(C)]
pub struct VkComputePipelineCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage: VkPipelineShaderStageCreateInfo,
    pub layout: VkPipelineLayout,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

#[repr(C)]
pub struct VkPipelineLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub set_layout_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
    pub push_constant_range_count: u32,
    pub p_push_constant_ranges: *const VkPushConstantRange,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkPushConstantRange {
    pub stage_flags: u32,
    pub offset: u32,
    pub size: u32,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutBinding {
    pub binding: u32,
    pub descriptor_type: u32,
    pub descriptor_count: u32,
    pub stage_flags: u32,
    pub p_immutable_samplers: *const VkSampler,
}

#[repr(C)]
pub struct VkDescriptorSetLayoutCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub binding_count: u32,
    pub p_bindings: *const VkDescriptorSetLayoutBinding,
}

#[repr(C)]
pub struct VkDescriptorPoolSize {
    pub ty: u32,
    pub descriptor_count: u32,
}

#[repr(C)]
pub struct VkDescriptorPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub max_sets: u32,
    pub pool_size_count: u32,
    pub p_pool_sizes: *const VkDescriptorPoolSize,
}

#[repr(C)]
pub struct VkDescriptorSetAllocateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub descriptor_pool: VkDescriptorPool,
    pub descriptor_set_count: u32,
    pub p_set_layouts: *const VkDescriptorSetLayout,
}

#[repr(C)]
pub struct VkDescriptorBufferInfo {
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub range: VkDeviceSize,
}

#[repr(C)]
pub struct VkDescriptorImageInfo {
    pub sampler: VkSampler,
    pub image_view: VkImageView,
    pub image_layout: u32,
}

#[repr(C)]
pub struct VkWriteDescriptorSet {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub dst_set: VkDescriptorSet,
    pub dst_binding: u32,
    pub dst_array_element: u32,
    pub descriptor_count: u32,
    pub descriptor_type: u32,
    pub p_image_info: *const VkDescriptorImageInfo,
    pub p_buffer_info: *const VkDescriptorBufferInfo,
    pub p_texel_buffer_view: *const c_void,
}

#[repr(C)]
pub struct VkSubmitInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub wait_semaphore_count: u32,
    pub p_wait_semaphores: *const VkSemaphore,
    pub p_wait_dst_stage_mask: *const u32,
    pub command_buffer_count: u32,
    pub p_command_buffers: *const VkCommandBuffer,
    pub signal_semaphore_count: u32,
    pub p_signal_semaphores: *const VkSemaphore,
}

#[repr(C)]
pub struct VkFenceCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
}

#[repr(C)]
pub struct VkSamplerCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub mag_filter: u32,
    pub min_filter: u32,
    pub mipmap_mode: u32,
    pub address_mode_u: u32,
    pub address_mode_v: u32,
    pub address_mode_w: u32,
    pub mip_lod_bias: f32,
    pub anisotropy_enable: u32,
    pub max_anisotropy: f32,
    pub compare_enable: u32,
    pub compare_op: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: u32,
    pub unnormalized_coordinates: u32,
}

#[repr(C)]
pub struct VkAttachmentDescription {
    pub flags: u32,
    pub format: u32,
    pub samples: u32,
    pub load_op: u32,
    pub store_op: u32,
    pub stencil_load_op: u32,
    pub stencil_store_op: u32,
    pub initial_layout: u32,
    pub final_layout: u32,
}

#[repr(C)]
pub struct VkAttachmentReference {
    pub attachment: u32,
    pub layout: u32,
}

#[repr(C)]
pub struct VkSubpassDescription {
    pub flags: u32,
    pub pipeline_bind_point: u32,
    pub input_attachment_count: u32,
    pub p_input_attachments: *const VkAttachmentReference,
    pub color_attachment_count: u32,
    pub p_color_attachments: *const VkAttachmentReference,
    pub p_resolve_attachments: *const VkAttachmentReference,
    pub p_depth_stencil_attachment: *const VkAttachmentReference,
    pub preserve_attachment_count: u32,
    pub p_preserve_attachments: *const u32,
}

#[repr(C)]
pub struct VkRenderPassCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub attachment_count: u32,
    pub p_attachments: *const VkAttachmentDescription,
    pub subpass_count: u32,
    pub p_subpasses: *const VkSubpassDescription,
    pub dependency_count: u32,
    pub p_dependencies: *const c_void,
}

#[repr(C)]
pub struct VkFramebufferCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub render_pass: VkRenderPass,
    pub attachment_count: u32,
    pub p_attachments: *const VkImageView,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union VkClearColorValue {
    pub float32: [f32; 4],
    pub int32: [i32; 4],
    pub uint32: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkClearDepthStencilValue {
    pub depth: f32,
    pub stencil: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union VkClearValue {
    pub color: VkClearColorValue,
    pub depth_stencil: VkClearDepthStencilValue,
}

#[repr(C)]
pub struct VkRenderPassBeginInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub render_pass: VkRenderPass,
    pub framebuffer: VkFramebuffer,
    pub render_area: VkRect2D,
    pub clear_value_count: u32,
    pub p_clear_values: *const VkClearValue,
}

#[repr(C)]
pub struct VkImageMemoryBarrier {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_access_mask: u32,
    pub dst_access_mask: u32,
    pub old_layout: u32,
    pub new_layout: u32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: VkImage,
    pub subresource_range: VkImageSubresourceRange,
}

pub const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER: u32 = 45;

#[repr(C)]
pub struct VkBufferCopy {
    pub src_offset: VkDeviceSize,
    pub dst_offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[repr(C)]
pub struct VkBufferImageCopy {
    pub buffer_offset: VkDeviceSize,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: VkImageSubresourceLayers,
    pub image_offset: VkOffset3D,
    pub image_extent: VkExtent3D,
}

#[repr(C)]
pub struct VkImageBlit {
    pub src_subresource: VkImageSubresourceLayers,
    pub src_offsets: [VkOffset3D; 2],
    pub dst_subresource: VkImageSubresourceLayers,
    pub dst_offsets: [VkOffset3D; 2],
}

#[repr(C)]
pub struct VkImageResolve {
    pub src_subresource: VkImageSubresourceLayers,
    pub src_offset: VkOffset3D,
    pub dst_subresource: VkImageSubresourceLayers,
    pub dst_offset: VkOffset3D,
    pub extent: VkExtent3D,
}

// ─── Pipeline cache ───────────────────────────────────────────────────────

pub const VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO: u32 = 17;

#[repr(C)]
pub struct VkPipelineCacheCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub initial_data_size: usize,
    pub p_initial_data: *const c_void,
}

// ─── Query pool ────────────────────────────────────────────────────────────

pub const VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO: u32 = 11;

pub type VkQueryPool = *mut c_void;

#[repr(C)]
pub struct VkQueryPoolCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub query_type: u32,
    pub query_count: u32,
    pub pipeline_statistics: u32,
}

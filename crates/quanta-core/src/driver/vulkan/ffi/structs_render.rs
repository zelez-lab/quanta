//! Vulkan graphics pipeline and synchronization structures.

use core::ffi::c_void;

use super::constants::*;
use super::structs::*;

// ─── Vulkan vertex format constants ──────────────────────────────────────────

pub const VK_FORMAT_R32G32B32_SFLOAT: u32 = 106;
pub const VK_FORMAT_R32_SINT: u32 = 99;
pub const VK_FORMAT_R32G32_SINT: u32 = 102;
pub const VK_FORMAT_R32G32B32_SINT: u32 = 105;
pub const VK_FORMAT_R32G32B32A32_SINT: u32 = 108;
pub const VK_FORMAT_R32_UINT: u32 = 98;
pub const VK_FORMAT_R32G32_UINT: u32 = 101;
pub const VK_FORMAT_R32G32B32_UINT: u32 = 104;
pub const VK_FORMAT_R32G32B32A32_UINT: u32 = 107;

pub const VK_VERTEX_INPUT_RATE_VERTEX: u32 = 0;
pub const VK_VERTEX_INPUT_RATE_INSTANCE: u32 = 1;

// ─── Graphics pipeline create info and sub-structures ───────────────────────

#[repr(C)]
pub struct VkVertexInputBindingDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: u32,
}

#[repr(C)]
pub struct VkVertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: u32,
    pub offset: u32,
}

#[repr(C)]
pub struct VkPipelineVertexInputStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub vertex_binding_description_count: u32,
    pub p_vertex_binding_descriptions: *const VkVertexInputBindingDescription,
    pub vertex_attribute_description_count: u32,
    pub p_vertex_attribute_descriptions: *const VkVertexInputAttributeDescription,
}

#[repr(C)]
pub struct VkPipelineInputAssemblyStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub topology: u32,
    pub primitive_restart_enable: u32,
}

#[repr(C)]
pub struct VkPipelineViewportStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub viewport_count: u32,
    pub p_viewports: *const VkViewport,
    pub scissor_count: u32,
    pub p_scissors: *const VkRect2D,
}

#[repr(C)]
pub struct VkPipelineRasterizationStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub depth_clamp_enable: u32,
    pub rasterizer_discard_enable: u32,
    pub polygon_mode: u32,
    pub cull_mode: u32,
    pub front_face: u32,
    pub depth_bias_enable: u32,
    pub depth_bias_constant_factor: f32,
    pub depth_bias_clamp: f32,
    pub depth_bias_slope_factor: f32,
    pub line_width: f32,
}

#[repr(C)]
pub struct VkPipelineMultisampleStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub rasterization_samples: u32,
    pub sample_shading_enable: u32,
    pub min_sample_shading: f32,
    pub p_sample_mask: *const u32,
    pub alpha_to_coverage_enable: u32,
    pub alpha_to_one_enable: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkPipelineColorBlendAttachmentState {
    pub blend_enable: u32,
    pub src_color_blend_factor: u32,
    pub dst_color_blend_factor: u32,
    pub color_blend_op: u32,
    pub src_alpha_blend_factor: u32,
    pub dst_alpha_blend_factor: u32,
    pub alpha_blend_op: u32,
    pub color_write_mask: u32,
}

#[repr(C)]
pub struct VkPipelineColorBlendStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub logic_op_enable: u32,
    pub logic_op: u32,
    pub attachment_count: u32,
    pub p_attachments: *const VkPipelineColorBlendAttachmentState,
    pub blend_constants: [f32; 4],
}

#[repr(C)]
pub struct VkPipelineDynamicStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub dynamic_state_count: u32,
    pub p_dynamic_states: *const u32,
}

#[repr(C)]
pub struct VkGraphicsPipelineCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub stage_count: u32,
    pub p_stages: *const VkPipelineShaderStageCreateInfo,
    pub p_vertex_input_state: *const VkPipelineVertexInputStateCreateInfo,
    pub p_input_assembly_state: *const VkPipelineInputAssemblyStateCreateInfo,
    pub p_tessellation_state: *const c_void,
    pub p_viewport_state: *const VkPipelineViewportStateCreateInfo,
    pub p_rasterization_state: *const VkPipelineRasterizationStateCreateInfo,
    pub p_multisample_state: *const VkPipelineMultisampleStateCreateInfo,
    pub p_depth_stencil_state: *const c_void,
    pub p_color_blend_state: *const VkPipelineColorBlendStateCreateInfo,
    pub p_dynamic_state: *const VkPipelineDynamicStateCreateInfo,
    pub layout: VkPipelineLayout,
    pub render_pass: VkRenderPass,
    pub subpass: u32,
    pub base_pipeline_handle: VkPipeline,
    pub base_pipeline_index: i32,
}

// ─── Synchronization2 structures (Vulkan 1.3) ──────────────────────────────

#[repr(C)]
pub struct VkMemoryBarrier2 {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_stage_mask: u64,
    pub src_access_mask: u64,
    pub dst_stage_mask: u64,
    pub dst_access_mask: u64,
}

#[repr(C)]
pub struct VkBufferMemoryBarrier2 {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_stage_mask: u64,
    pub src_access_mask: u64,
    pub dst_stage_mask: u64,
    pub dst_access_mask: u64,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub buffer: VkBuffer,
    pub offset: VkDeviceSize,
    pub size: VkDeviceSize,
}

#[repr(C)]
pub struct VkImageMemoryBarrier2 {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub src_stage_mask: u64,
    pub src_access_mask: u64,
    pub dst_stage_mask: u64,
    pub dst_access_mask: u64,
    pub old_layout: u32,
    pub new_layout: u32,
    pub src_queue_family_index: u32,
    pub dst_queue_family_index: u32,
    pub image: VkImage,
    pub subresource_range: VkImageSubresourceRange,
}

#[repr(C)]
pub struct VkDependencyInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub dependency_flags: u32,
    pub memory_barrier_count: u32,
    pub p_memory_barriers: *const VkMemoryBarrier2,
    pub buffer_memory_barrier_count: u32,
    pub p_buffer_memory_barriers: *const VkBufferMemoryBarrier2,
    pub image_memory_barrier_count: u32,
    pub p_image_memory_barriers: *const VkImageMemoryBarrier2,
}

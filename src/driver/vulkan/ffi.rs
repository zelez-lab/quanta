//! Raw Vulkan FFI bindings — minimal subset for GPU compute and rendering.
//!
//! Follows Dija's pattern: opaque handles, `#[repr(C)]` structs, platform-gated
//! extern blocks. No `ash` dependency.

#![allow(non_camel_case_types, dead_code)]

use core::ffi::c_void;

// ─── Handle types ───────────────────────────────────────────────────────────
// Dispatchable handles are `*mut c_void` on 64-bit platforms.
// Non-dispatchable handles are also `*mut c_void` here (ash uses u64 internally
// but *mut c_void is correct on 64-bit and simpler to work with).

pub type VkInstance = *mut c_void;
pub type VkPhysicalDevice = *mut c_void;
pub type VkDevice = *mut c_void;
pub type VkQueue = *mut c_void;
pub type VkCommandBuffer = *mut c_void;
pub type VkCommandPool = *mut c_void;
pub type VkBuffer = *mut c_void;
pub type VkDeviceMemory = *mut c_void;
pub type VkImage = *mut c_void;
pub type VkImageView = *mut c_void;
pub type VkPipeline = *mut c_void;
pub type VkPipelineLayout = *mut c_void;
pub type VkPipelineCache = *mut c_void;
pub type VkShaderModule = *mut c_void;
pub type VkRenderPass = *mut c_void;
pub type VkFramebuffer = *mut c_void;
pub type VkDescriptorSetLayout = *mut c_void;
pub type VkDescriptorPool = *mut c_void;
pub type VkDescriptorSet = *mut c_void;
pub type VkSampler = *mut c_void;
pub type VkFence = *mut c_void;
pub type VkSemaphore = *mut c_void;

pub type VkDeviceSize = u64;

/// Null handle constant — used where ash uses `::null()`.
#[inline]
pub const fn null_handle() -> *mut c_void {
    core::ptr::null_mut()
}

// ─── Result codes ───────────────────────────────────────────────────────────

pub type VkResult = i32;
pub const VK_SUCCESS: VkResult = 0;

// ─── VkStructureType constants ──────────────────────────────────────────────

pub const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
pub const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
pub const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
pub const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
pub const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
pub const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
pub const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
pub const VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO: u32 = 14;
pub const VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO: u32 = 15;
pub const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
pub const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
pub const VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO: u32 = 19;
pub const VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO: u32 = 20;
pub const VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO: u32 = 22;
pub const VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO: u32 = 23;
pub const VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO: u32 = 24;
pub const VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO: u32 = 26;
pub const VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO: u32 = 27;
pub const VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO: u32 = 28;
pub const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
pub const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
pub const VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO: u32 = 31;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
pub const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
pub const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
pub const VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO: u32 = 37;
pub const VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO: u32 = 38;
pub const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
pub const VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO: u32 = 43;
pub const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;
pub const VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO: u32 = 9;

// ─── Format constants ───────────────────────────────────────────────────────

pub const VK_FORMAT_R8_UNORM: u32 = 9;
pub const VK_FORMAT_R16_SFLOAT: u32 = 76;
pub const VK_FORMAT_R32_SFLOAT: u32 = 100;
pub const VK_FORMAT_R32G32_SFLOAT: u32 = 103;
pub const VK_FORMAT_R8G8B8A8_UNORM: u32 = 37;
pub const VK_FORMAT_B8G8R8A8_UNORM: u32 = 44;
pub const VK_FORMAT_R16G16B16A16_SFLOAT: u32 = 97;
pub const VK_FORMAT_R32G32B32A32_SFLOAT: u32 = 109;
pub const VK_FORMAT_D32_SFLOAT: u32 = 126;
// Compressed formats
pub const VK_FORMAT_BC1_RGBA_UNORM_BLOCK: u32 = 132;
pub const VK_FORMAT_BC3_UNORM_BLOCK: u32 = 137;
pub const VK_FORMAT_BC5_SNORM_BLOCK: u32 = 142;
pub const VK_FORMAT_BC7_UNORM_BLOCK: u32 = 145;
pub const VK_FORMAT_ASTC_4X4_UNORM_BLOCK: u32 = 157;
pub const VK_FORMAT_ASTC_6X6_UNORM_BLOCK: u32 = 163;
pub const VK_FORMAT_ASTC_8X8_UNORM_BLOCK: u32 = 169;
pub const VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK: u32 = 147;
pub const VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK: u32 = 151;

// ─── Image layout constants ─────────────────────────────────────────────────

pub const VK_IMAGE_LAYOUT_UNDEFINED: u32 = 0;
pub const VK_IMAGE_LAYOUT_GENERAL: u32 = 1;
pub const VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: u32 = 2;
pub const VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: u32 = 3;
pub const VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL: u32 = 5;
pub const VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: u32 = 6;
pub const VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: u32 = 7;
pub const VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: u32 = 1000001002;

// ─── Image type / tiling ────────────────────────────────────────────────────

pub const VK_IMAGE_TYPE_2D: u32 = 1;
pub const VK_IMAGE_TILING_OPTIMAL: u32 = 0;
pub const VK_IMAGE_VIEW_TYPE_2D: u32 = 1;

// ─── Image aspect ───────────────────────────────────────────────────────────

pub const VK_IMAGE_ASPECT_COLOR_BIT: u32 = 0x00000001;
pub const VK_IMAGE_ASPECT_DEPTH_BIT: u32 = 0x00000002;

// ─── Image usage ────────────────────────────────────────────────────────────

pub const VK_IMAGE_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_IMAGE_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_IMAGE_USAGE_SAMPLED_BIT: u32 = 0x00000004;
pub const VK_IMAGE_USAGE_STORAGE_BIT: u32 = 0x00000008;
pub const VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT: u32 = 0x00000010;
pub const VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT: u32 = 0x00000020;

// ─── Buffer usage ───────────────────────────────────────────────────────────

pub const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x00000001;
pub const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x00000002;
pub const VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT: u32 = 0x00000010;
pub const VK_BUFFER_USAGE_INDEX_BUFFER_BIT: u32 = 0x00000040;
pub const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: u32 = 0x00000080;
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000100;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: u32 = 0x00000200;

// ─── Memory property ────────────────────────────────────────────────────────

pub const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
pub const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
pub const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x00000004;

// ─── Sharing mode ───────────────────────────────────────────────────────────

pub const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;

// ─── Pipeline stage flags ───────────────────────────────────────────────────

pub const VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT: u32 = 0x00000001;
pub const VK_PIPELINE_STAGE_VERTEX_SHADER_BIT: u32 = 0x00000008;
pub const VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT: u32 = 0x00000080;
pub const VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT: u32 = 0x00000400;
pub const VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT: u32 = 0x00000800;
pub const VK_PIPELINE_STAGE_TRANSFER_BIT: u32 = 0x00001000;
pub const VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT: u32 = 0x00002000;
pub const VK_PIPELINE_STAGE_ALL_COMMANDS_BIT: u32 = 0x00010000;

// ─── Pipeline stage flags 2 (Vulkan 1.3) ────────────────────────────────────

pub const VK_PIPELINE_STAGE_2_NONE: u64 = 0;
pub const VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT: u64 = 0x00010000;
pub const VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT: u64 = 0x00000800;
pub const VK_PIPELINE_STAGE_2_TRANSFER_BIT: u64 = 0x00001000;
pub const VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT: u64 = 0x00000080;
pub const VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT: u64 = 0x00000400;
pub const VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT: u64 = 0x00000100;
pub const VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT: u64 = 0x00000200;

// ─── Access flags ───────────────────────────────────────────────────────────

pub const VK_ACCESS_SHADER_READ_BIT: u32 = 0x00000020;
pub const VK_ACCESS_SHADER_WRITE_BIT: u32 = 0x00000040;
pub const VK_ACCESS_COLOR_ATTACHMENT_READ_BIT: u32 = 0x00000080;
pub const VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT: u32 = 0x00000100;
pub const VK_ACCESS_TRANSFER_READ_BIT: u32 = 0x00000800;
pub const VK_ACCESS_TRANSFER_WRITE_BIT: u32 = 0x00001000;
pub const VK_ACCESS_MEMORY_READ_BIT: u32 = 0x00008000;
pub const VK_ACCESS_MEMORY_WRITE_BIT: u32 = 0x00010000;

// ─── Access flags 2 (Vulkan 1.3) ────────────────────────────────────────────

pub const VK_ACCESS_2_NONE: u64 = 0;
pub const VK_ACCESS_2_SHADER_READ_BIT: u64 = 0x00000020;
pub const VK_ACCESS_2_SHADER_WRITE_BIT: u64 = 0x00000040;
pub const VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT: u64 = 0x00000080;
pub const VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT: u64 = 0x00000100;
pub const VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT: u64 = 0x00000200;
pub const VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT: u64 = 0x00000400;
pub const VK_ACCESS_2_TRANSFER_READ_BIT: u64 = 0x00000800;
pub const VK_ACCESS_2_TRANSFER_WRITE_BIT: u64 = 0x00001000;
pub const VK_ACCESS_2_MEMORY_READ_BIT: u64 = 0x00008000;
pub const VK_ACCESS_2_MEMORY_WRITE_BIT: u64 = 0x00010000;

// ─── Descriptor type ────────────────────────────────────────────────────────

pub const VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: u32 = 1;
pub const VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: u32 = 3;
pub const VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: u32 = 6;
pub const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;

// ─── Shader stage ───────────────────────────────────────────────────────────

pub const VK_SHADER_STAGE_VERTEX_BIT: u32 = 0x00000001;
pub const VK_SHADER_STAGE_FRAGMENT_BIT: u32 = 0x00000010;
pub const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x00000020;

// ─── Command buffer level / usage ───────────────────────────────────────────

pub const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;

// ─── Command pool create flags ──────────────────────────────────────────────

pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;

// ─── Queue flags ────────────────────────────────────────────────────────────

pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;

// ─── Pipeline bind point ────────────────────────────────────────────────────

pub const VK_PIPELINE_BIND_POINT_GRAPHICS: u32 = 0;
pub const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;

// ─── Blend factor ───────────────────────────────────────────────────────────

pub const VK_BLEND_FACTOR_ZERO: u32 = 0;
pub const VK_BLEND_FACTOR_ONE: u32 = 1;
pub const VK_BLEND_FACTOR_SRC_COLOR: u32 = 2;
pub const VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: u32 = 3;
pub const VK_BLEND_FACTOR_DST_COLOR: u32 = 4;
pub const VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: u32 = 5;
pub const VK_BLEND_FACTOR_SRC_ALPHA: u32 = 6;
pub const VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: u32 = 7;
pub const VK_BLEND_FACTOR_DST_ALPHA: u32 = 8;
pub const VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: u32 = 9;

// ─── Blend op ───────────────────────────────────────────────────────────────

pub const VK_BLEND_OP_ADD: u32 = 0;
pub const VK_BLEND_OP_SUBTRACT: u32 = 1;
pub const VK_BLEND_OP_REVERSE_SUBTRACT: u32 = 2;
pub const VK_BLEND_OP_MIN: u32 = 3;
pub const VK_BLEND_OP_MAX: u32 = 4;

// ─── Color component flags ──────────────────────────────────────────────────

pub const VK_COLOR_COMPONENT_R_BIT: u32 = 0x01;
pub const VK_COLOR_COMPONENT_G_BIT: u32 = 0x02;
pub const VK_COLOR_COMPONENT_B_BIT: u32 = 0x04;
pub const VK_COLOR_COMPONENT_A_BIT: u32 = 0x08;
pub const VK_COLOR_COMPONENT_RGBA: u32 = 0x0F;

// ─── Primitive topology ─────────────────────────────────────────────────────

pub const VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: u32 = 3;

// ─── Polygon mode / cull / front face ───────────────────────────────────────

pub const VK_POLYGON_MODE_FILL: u32 = 0;
pub const VK_CULL_MODE_NONE: u32 = 0;
pub const VK_CULL_MODE_FRONT_BIT: u32 = 1;
pub const VK_CULL_MODE_BACK_BIT: u32 = 2;
pub const VK_FRONT_FACE_COUNTER_CLOCKWISE: u32 = 0;

// ─── Sample count ───────────────────────────────────────────────────────────

pub const VK_SAMPLE_COUNT_1_BIT: u32 = 0x01;
pub const VK_SAMPLE_COUNT_2_BIT: u32 = 0x02;
pub const VK_SAMPLE_COUNT_4_BIT: u32 = 0x04;
pub const VK_SAMPLE_COUNT_8_BIT: u32 = 0x08;
pub const VK_SAMPLE_COUNT_16_BIT: u32 = 0x10;

// ─── Dynamic state ──────────────────────────────────────────────────────────

pub const VK_DYNAMIC_STATE_VIEWPORT: u32 = 0;
pub const VK_DYNAMIC_STATE_SCISSOR: u32 = 1;
pub const VK_DYNAMIC_STATE_STENCIL_REFERENCE: u32 = 8;

// ─── Attachment load/store ops ──────────────────────────────────────────────

pub const VK_ATTACHMENT_LOAD_OP_LOAD: u32 = 0;
pub const VK_ATTACHMENT_LOAD_OP_CLEAR: u32 = 1;
pub const VK_ATTACHMENT_LOAD_OP_DONT_CARE: u32 = 2;
pub const VK_ATTACHMENT_STORE_OP_STORE: u32 = 0;
pub const VK_ATTACHMENT_STORE_OP_DONT_CARE: u32 = 1;

// ─── Subpass contents ───────────────────────────────────────────────────────

pub const VK_SUBPASS_CONTENTS_INLINE: u32 = 0;

// ─── Filter / address mode / mipmap mode ────────────────────────────────────

pub const VK_FILTER_NEAREST: u32 = 0;
pub const VK_FILTER_LINEAR: u32 = 1;
pub const VK_SAMPLER_ADDRESS_MODE_REPEAT: u32 = 0;
pub const VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: u32 = 1;
pub const VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE: u32 = 2;
pub const VK_SAMPLER_MIPMAP_MODE_NEAREST: u32 = 0;
pub const VK_SAMPLER_MIPMAP_MODE_LINEAR: u32 = 1;

// ─── Index type ─────────────────────────────────────────────────────────────

pub const VK_INDEX_TYPE_UINT16: u32 = 0;
pub const VK_INDEX_TYPE_UINT32: u32 = 1;

// ─── Fence flags ────────────────────────────────────────────────────────────

pub const VK_FENCE_CREATE_SIGNALED_BIT: u32 = 0x00000001;

// ─── Special values ─────────────────────────────────────────────────────────

pub const VK_WHOLE_SIZE: u64 = !0u64;
pub const VK_QUEUE_FAMILY_IGNORED: u32 = !0u32;
pub const VK_REMAINING_MIP_LEVELS: u32 = !0u32;
pub const VK_REMAINING_ARRAY_LAYERS: u32 = !0u32;
pub const VK_LOD_CLAMP_NONE: f32 = 1000.0;

// ─── Stencil face flags ─────────────────────────────────────────────────────

pub const VK_STENCIL_FACE_FRONT_AND_BACK: u32 = 0x00000003;

// ─── Dependency info structure type (Vulkan 1.3) ────────────────────────────

pub const VK_STRUCTURE_TYPE_DEPENDENCY_INFO: u32 = 1000314003;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER_2: u32 = 1000314000;
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2: u32 = 1000314001;
pub const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2: u32 = 1000314002;

// ─── Vulkan API version helper ──────────────────────────────────────────────

#[inline]
pub const fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

// ============================================================================
// Structures
// ============================================================================

#[repr(C)]
pub struct VkApplicationInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub p_application_name: *const i8,
    pub application_version: u32,
    pub p_engine_name: *const i8,
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
    pub pp_enabled_layer_names: *const *const i8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const i8,
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
    pub pp_enabled_layer_names: *const *const i8,
    pub enabled_extension_count: u32,
    pub pp_enabled_extension_names: *const *const i8,
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
    pub p_name: *const i8,
    pub p_specialization_info: *const c_void,
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

// ─── Graphics pipeline create info and sub-structures ───────────────────────

#[repr(C)]
pub struct VkPipelineVertexInputStateCreateInfo {
    pub s_type: u32,
    pub p_next: *const c_void,
    pub flags: u32,
    pub vertex_binding_description_count: u32,
    pub p_vertex_binding_descriptions: *const c_void,
    pub vertex_attribute_description_count: u32,
    pub p_vertex_attribute_descriptions: *const c_void,
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

// ─── Physical device properties ─────────────────────────────────────────────

#[repr(C)]
pub struct VkPhysicalDeviceLimits {
    // Only the fields we actually use — padded to correct layout.
    // Full struct is 504 bytes. We only access a few fields.
    pub _pad: [u8; 504],
}

/// Minimal subset: we read vendor_id, device_name, and limits directly via
/// `vkGetPhysicalDeviceProperties` into a raw buffer.
#[repr(C)]
pub struct VkPhysicalDeviceProperties {
    pub api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: u32,
    pub device_name: [u8; 256],
    pub pipeline_cache_uuid: [u8; 16],
    pub limits: VkPhysicalDeviceLimitsRaw,
    pub sparse_properties: VkPhysicalDeviceSparseProperties,
}

/// Raw limits — only the offsets we care about.
/// The Vulkan spec mandates a fixed layout, so field positions are stable.
#[repr(C)]
pub struct VkPhysicalDeviceLimitsRaw {
    pub max_image_dimension_1d: u32,
    pub max_image_dimension_2d: u32,
    pub max_image_dimension_3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    // The rest of the struct (we don't use, but must pad for layout correctness).
    pub _tail: [u8; 172],
}

#[repr(C)]
pub struct VkPhysicalDeviceSparseProperties {
    pub residency_standard_2d_block_shape: u32,
    pub residency_standard_2d_multisample_block_shape: u32,
    pub residency_standard_3d_block_shape: u32,
    pub residency_aligned_mip_size: u32,
    pub residency_non_resident_strict: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

impl Default for VkExtent3D {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            depth: 0,
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find a suitable memory type index given requirements and desired property flags.
pub fn find_memory_type(
    props: &VkPhysicalDeviceMemoryProperties,
    type_bits: u32,
    required_flags: u32,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && (props.memory_types[i as usize].property_flags & required_flags) == required_flags
    })
}

// ============================================================================
// Extern function declarations — platform-gated
// ============================================================================

#[cfg(any(target_os = "linux", target_os = "android"))]
#[link(name = "vulkan")]
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
}

// ─── macOS (MoltenVK or Vulkan loader) ──────────────────────────────────────

#[cfg(target_os = "macos")]
#[link(name = "vulkan")]
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
}

// ─── Windows (vulkan-1.dll) ─────────────────────────────────────────────────

#[cfg(target_os = "windows")]
#[link(name = "vulkan-1")]
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
}

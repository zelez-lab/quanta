//! Vulkan constants — handle types, structure types, format codes, flags.

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
pub const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x00000020;
pub const VK_BUFFER_USAGE_INDEX_BUFFER_BIT: u32 = 0x00000040;
pub const VK_BUFFER_USAGE_VERTEX_BUFFER_BIT: u32 = 0x00000080;
pub const VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT: u32 = 0x00000100;

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
pub const VK_COMMAND_BUFFER_LEVEL_SECONDARY: u32 = 1;
pub const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x00000001;
pub const VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT: u32 = 0x00000002;
pub const VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT: u32 = 0x00000004;
pub const VK_SUBPASS_CONTENTS_INLINE: u32 = 0;
pub const VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS: u32 = 1;
pub const VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO: u32 = 41;

// ─── Command pool create flags ──────────────────────────────────────────────

pub const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x00000002;

// ─── Queue flags ────────────────────────────────────────────────────────────

pub const VK_QUEUE_GRAPHICS_BIT: u32 = 0x00000001;
pub const VK_QUEUE_COMPUTE_BIT: u32 = 0x00000002;
pub const VK_QUEUE_TRANSFER_BIT: u32 = 0x00000004;

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

// ─── Query type ────────────────────────────────────────────────────────────

pub const VK_QUERY_TYPE_OCCLUSION: u32 = 0;
pub const VK_QUERY_TYPE_TIMESTAMP: u32 = 2;
pub const VK_QUERY_RESULT_64_BIT: u32 = 0x00000001;
pub const VK_QUERY_RESULT_WAIT_BIT: u32 = 0x00000002;

// ─── Stencil face flags ─────────────────────────────────────────────────────

pub const VK_STENCIL_FACE_FRONT_AND_BACK: u32 = 0x00000003;

// ─── Dependency info structure type (Vulkan 1.3) ────────────────────────────

pub const VK_STRUCTURE_TYPE_DEPENDENCY_INFO: u32 = 1000314003;
pub const VK_STRUCTURE_TYPE_MEMORY_BARRIER_2: u32 = 1000314000;
pub const VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2: u32 = 1000314001;
pub const VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2: u32 = 1000314002;

// ─── VK_KHR_fragment_shading_rate (step 063) ────────────────────────────────

/// Combiner ops for `vkCmdSetFragmentShadingRateKHR`. KEEP combines
/// per-draw rates by passing the pipeline rate through unchanged —
/// matches the per-draw semantics of `RenderOp::SetShadingRate`.
pub const VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR: u32 = 0;

/// Function-pointer type for `vkCmdSetFragmentShadingRateKHR`. Loaded
/// at device init via `vkGetDeviceProcAddr` when the
/// `VK_KHR_fragment_shading_rate` extension is enabled; null otherwise.
pub type PfnVkCmdSetFragmentShadingRateKHR = unsafe extern "C" fn(
    cmd_buf: VkCommandBuffer,
    rate: *const super::structs::VkExtent2D,
    combiner_ops: *const u32,
);

/// Function-pointer type for `vkGetPhysicalDeviceFragmentShadingRatesKHR`.
/// Loaded once via `vkGetInstanceProcAddr` after the instance is
/// created; per-physical-device rate enumeration runs through this
/// proc before `vkCreateDevice`.
pub type PfnVkGetPhysicalDeviceFragmentShadingRatesKHR = unsafe extern "C" fn(
    physical_device: VkPhysicalDevice,
    p_count: *mut u32,
    p_rates: *mut super::structs::VkPhysicalDeviceFragmentShadingRateKHR,
) -> VkResult;

// ─── VK_EXT_mesh_shader (step 063) ──────────────────────────────────────────

/// Function-pointer type for `vkCmdDrawMeshTasksEXT`. Issued from
/// inside a render pass with a mesh-shader pipeline bound. Loaded at
/// device init when the `VK_EXT_mesh_shader` extension is enabled.
pub type PfnVkCmdDrawMeshTasksEXT = unsafe extern "C" fn(
    cmd_buf: VkCommandBuffer,
    group_count_x: u32,
    group_count_y: u32,
    group_count_z: u32,
);

// ─── VK_KHR_ray_tracing_pipeline (step 063) ─────────────────────────────────

/// Function-pointer type for `vkCmdTraceRaysKHR`. Issued from a
/// command buffer outside a render pass; consumes shader binding
/// tables (raygen / miss / hit / callable). Loaded at device init
/// when both `VK_KHR_ray_tracing_pipeline` and
/// `VK_KHR_acceleration_structure` are enabled.
///
/// The shader-binding-table arguments are `VkStridedDeviceAddressRegionKHR`
/// pointers; we type them as opaque `*const c_void` here because the
/// SBT layout is owned by callers and the layered structs are not
/// otherwise needed in the FFI surface.
pub type PfnVkCmdTraceRaysKHR = unsafe extern "C" fn(
    cmd_buf: VkCommandBuffer,
    raygen_sbt: *const c_void,
    miss_sbt: *const c_void,
    hit_sbt: *const c_void,
    callable_sbt: *const c_void,
    width: u32,
    height: u32,
    depth: u32,
);

// ─── Sparse residency (step 063 slice 16) ───────────────────────────────────

pub const VK_IMAGE_CREATE_SPARSE_BINDING_BIT: u32 = 0x00000001;
pub const VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT: u32 = 0x00000002;
pub const VK_QUEUE_SPARSE_BINDING_BIT: u32 = 0x00000010;
pub const VK_STRUCTURE_TYPE_BIND_SPARSE_INFO: u32 = 7;

// ─── VK_KHR_acceleration_structure (step 063 slice 15) ──────────────────────
//
// Acceleration structure builds: typed handle is opaque; the actual
// `VkAccelerationStructureKHR` is `*mut c_void`. The four helper
// procs below are the entry points needed for a real BLAS/TLAS
// build-out — load them at device discovery so the future
// build-out doesn't need to re-touch device.rs / extern_fns.rs.
//
// The full call signatures take VkAccelerationStructureCreateInfoKHR,
// VkAccelerationStructureBuildGeometryInfoKHR and friends; we type
// the create-info / build-info pointers as opaque `*const c_void`
// here so the FFI surface stays minimal until the build-path
// commit lands.

pub type PfnVkCreateAccelerationStructureKHR = unsafe extern "C" fn(
    device: VkDevice,
    create_info: *const c_void,
    allocator: *const c_void,
    p_acceleration_structure: *mut *mut c_void,
) -> VkResult;

pub type PfnVkDestroyAccelerationStructureKHR = unsafe extern "C" fn(
    device: VkDevice,
    acceleration_structure: *mut c_void,
    allocator: *const c_void,
);

pub type PfnVkGetAccelerationStructureBuildSizesKHR = unsafe extern "C" fn(
    device: VkDevice,
    build_type: u32,
    build_info: *const c_void,
    max_primitive_counts: *const u32,
    p_size_info: *mut c_void,
);

pub type PfnVkCmdBuildAccelerationStructuresKHR = unsafe extern "C" fn(
    cmd_buf: VkCommandBuffer,
    info_count: u32,
    p_infos: *const c_void,
    pp_build_range_infos: *const *const c_void,
);

// ─── Vulkan API version helper ──────────────────────────────────────────────

#[inline]
pub const fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

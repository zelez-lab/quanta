//! Physical device properties, limits, features, and memory helpers.

use super::structs::{VkExtent3D, VkPhysicalDeviceMemoryProperties};

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

/// The FULL `VkPhysicalDeviceLimits`, in spec field order.
///
/// This struct is embedded BY VALUE in [`VkPhysicalDeviceProperties`],
/// so it is not a "read the fields we care about" convenience: the
/// driver writes `sizeof(VkPhysicalDeviceProperties)` bytes into our
/// buffer, and any missing tail is a stack overflow on every
/// `vkGetPhysicalDeviceProperties`. It previously stopped at 56 of the
/// 106 fields (424 bytes vs 504), overflowing by 80 bytes on every
/// Vulkan discovery — harmless in dev builds by stack-layout luck, a
/// segfault in release ones. Keep every field, in order, forever.
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
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias: f32,
    pub max_sampler_anisotropy: f32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2],
    pub viewport_bounds_range: [f32; 2],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset: f32,
    pub max_interpolation_offset: f32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: u32,
    pub framebuffer_depth_sample_counts: u32,
    pub framebuffer_stencil_sample_counts: u32,
    pub framebuffer_no_attachments_sample_counts: u32,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: u32,
    pub sampled_image_integer_sample_counts: u32,
    pub sampled_image_depth_sample_counts: u32,
    pub sampled_image_stencil_sample_counts: u32,
    pub storage_image_sample_counts: u32,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: u32,
    pub timestamp_period: f32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub discrete_queue_priorities: u32,
    pub point_size_range: [f32; 2],
    pub line_width_range: [f32; 2],
    pub point_size_granularity: f32,
    pub line_width_granularity: f32,
    pub strict_lines: u32,
    pub standard_sample_locations: u32,
    pub optimal_buffer_copy_offset_alignment: u64,
    pub optimal_buffer_copy_row_pitch_alignment: u64,
    pub non_coherent_atom_size: u64,
}

#[repr(C)]
pub struct VkPhysicalDeviceSparseProperties {
    pub residency_standard_2d_block_shape: u32,
    pub residency_standard_2d_multisample_block_shape: u32,
    pub residency_standard_3d_block_shape: u32,
    pub residency_aligned_mip_size: u32,
    pub residency_non_resident_strict: u32,
}

/// `VkPhysicalDeviceSubgroupProperties` (core Vulkan 1.1) — chained
/// onto [`VkPhysicalDeviceProperties2`] via `p_next` to learn which
/// subgroup operation classes the device supports. Broadcom V3D
/// reports BASIC only (no ARITHMETIC — its NIR backend cannot lower
/// `OpGroupNonUniform*` reduce/scan); llvmpipe reports the full set.
#[repr(C)]
pub struct VkPhysicalDeviceSubgroupProperties {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub subgroup_size: u32,
    pub supported_stages: u32,
    pub supported_operations: u32,
    pub quad_operations_in_all_stages: u32,
}

/// `VkPhysicalDeviceProperties2` (core Vulkan 1.1) — the base
/// properties plus an extensible `p_next` chain. Only used at device
/// discovery to hang [`VkPhysicalDeviceSubgroupProperties`] off it;
/// the base `properties` are read through the existing v1.0 call.
#[repr(C)]
pub struct VkPhysicalDeviceProperties2 {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub properties: VkPhysicalDeviceProperties,
}

/// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT`
/// (`VK_EXT_external_memory_host`) — chained onto
/// [`VkPhysicalDeviceProperties2`] at discovery to learn the
/// granularity host pointers must satisfy for import
/// (`minImportedHostPointerAlignment`, typically 4096).
#[repr(C)]
pub struct VkPhysicalDeviceExternalMemoryHostPropertiesEXT {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub min_imported_host_pointer_alignment: u64,
}

/// `VkImportMemoryHostPointerInfoEXT` — chained onto
/// `VkMemoryAllocateInfo.p_next` to make `vkAllocateMemory` wrap the
/// given host pointer instead of allocating. `vkFreeMemory` on the
/// resulting memory releases the import, never the host pages.
#[repr(C)]
pub struct VkImportMemoryHostPointerInfoEXT {
    pub s_type: u32,
    pub p_next: *const core::ffi::c_void,
    pub handle_type: u32,
    pub p_host_pointer: *const core::ffi::c_void,
}

/// `VkMemoryHostPointerPropertiesEXT` — out-struct of
/// `vkGetMemoryHostPointerPropertiesEXT`: the memory types that can
/// import the queried host pointer.
#[repr(C)]
pub struct VkMemoryHostPointerPropertiesEXT {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub memory_type_bits: u32,
}

/// `VkPhysicalDeviceFeatures2` (core Vulkan 1.1) — the base features
/// plus an extensible `p_next` chain. Used at device discovery to hang
/// the 16-/8-bit storage feature queries off it; the base `features`
/// are read through the existing v1.0 call.
#[repr(C)]
pub struct VkPhysicalDeviceFeatures2 {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub features: VkPhysicalDeviceFeatures,
}

/// `VkPhysicalDevice16BitStorageFeatures` (core Vulkan 1.1). The
/// `storage_buffer_16bit_access` bit gates the native bf16 storage
/// contract: kernels touching bf16 buffers declare the SPIR-V
/// `StorageBuffer16BitAccess` capability and load/store 16-bit elements.
#[repr(C)]
pub struct VkPhysicalDevice16BitStorageFeatures {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub storage_buffer_16bit_access: u32,
    pub uniform_and_storage_buffer_16bit_access: u32,
    pub storage_push_constant16: u32,
    pub storage_input_output16: u32,
}

/// `VkPhysicalDevice8BitStorageFeatures` (core Vulkan 1.2). The
/// `storage_buffer_8bit_access` bit gates the native fp8 storage
/// contract (SPIR-V `StorageBuffer8BitAccess`, 8-bit buffer elements).
#[repr(C)]
pub struct VkPhysicalDevice8BitStorageFeatures {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub storage_buffer_8bit_access: u32,
    pub uniform_and_storage_buffer_8bit_access: u32,
    pub storage_push_constant8: u32,
}

/// `VkPhysicalDeviceCooperativeMatrixFeaturesKHR`. Chained into
/// `vkCreateDevice` when the extension is advertised; only
/// `cooperative_matrix` is enabled (robust buffer access is not needed
/// by Quanta's bounds-checked kernels).
#[repr(C)]
pub struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub cooperative_matrix: u32,
    pub cooperative_matrix_robust_buffer_access: u32,
}

/// `VkCooperativeMatrixPropertiesKHR` — one natively supported
/// `D = A·B + C` shape. Component types are `VkComponentTypeKHR`,
/// `scope` is `VkScopeKHR`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct VkCooperativeMatrixPropertiesKHR {
    pub s_type: u32,
    pub p_next: *mut core::ffi::c_void,
    pub m_size: u32,
    pub n_size: u32,
    pub k_size: u32,
    pub a_type: u32,
    pub b_type: u32,
    pub c_type: u32,
    pub result_type: u32,
    pub saturating_accumulation: u32,
    pub scope: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct VkQueueFamilyProperties {
    pub queue_flags: u32,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub min_image_transfer_granularity: VkExtent3D,
}

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

// ─── Extension property and device feature structs ──────────────────────────

#[repr(C)]
#[derive(Clone)]
pub struct VkExtensionProperties {
    pub extension_name: [u8; 256],
    pub spec_version: u32,
}

impl Default for VkExtensionProperties {
    fn default() -> Self {
        Self {
            extension_name: [0u8; 256],
            spec_version: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone)]
pub struct VkPhysicalDeviceFeatures {
    pub robust_buffer_access: u32,
    pub full_draw_index_uint32: u32,
    pub image_cube_array: u32,
    pub independent_blend: u32,
    pub geometry_shader: u32,
    pub tessellation_shader: u32,
    pub sample_rate_shading: u32,
    pub dual_src_blend: u32,
    pub logic_op: u32,
    pub multi_draw_indirect: u32,
    pub draw_indirect_first_instance: u32,
    pub depth_clamp: u32,
    pub depth_bias_clamp: u32,
    pub fill_mode_non_solid: u32,
    pub depth_bounds: u32,
    pub wide_lines: u32,
    pub large_points: u32,
    pub alpha_to_one: u32,
    pub multi_viewport: u32,
    pub sampler_anisotropy: u32,
    pub texture_compression_etc2: u32,
    pub texture_compression_astc_ldr: u32,
    pub texture_compression_bc: u32,
    pub occlusion_query_precise: u32,
    pub pipeline_statistics_query: u32,
    pub vertex_pipeline_stores_and_atomics: u32,
    pub fragment_stores_and_atomics: u32,
    pub shader_tessellation_and_geometry_point_size: u32,
    pub shader_image_gather_extended: u32,
    pub shader_storage_image_extended_formats: u32,
    pub shader_storage_image_multisample: u32,
    pub shader_storage_image_read_without_format: u32,
    pub shader_storage_image_write_without_format: u32,
    pub shader_uniform_buffer_array_dynamic_indexing: u32,
    pub shader_sampled_image_array_dynamic_indexing: u32,
    pub shader_storage_buffer_array_dynamic_indexing: u32,
    pub shader_storage_image_array_dynamic_indexing: u32,
    pub shader_clip_distance: u32,
    pub shader_cull_distance: u32,
    pub shader_float64: u32,
    pub shader_int64: u32,
    pub shader_int16: u32,
    pub shader_resource_residency: u32,
    pub shader_resource_min_lod: u32,
    pub sparse_binding: u32,
    pub sparse_residency_buffer: u32,
    pub sparse_residency_image2d: u32,
    pub sparse_residency_image3d: u32,
    pub sparse_residency_2_samples: u32,
    pub sparse_residency_4_samples: u32,
    pub sparse_residency_8_samples: u32,
    pub sparse_residency_16_samples: u32,
    pub sparse_residency_aliased: u32,
    pub variable_multisample_rate: u32,
    pub inherited_queries: u32,
}

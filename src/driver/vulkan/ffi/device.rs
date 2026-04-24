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

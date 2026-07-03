//! Vulkan graphics pipeline creation.

use alloc::format;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::{Pipeline, QuantaError, SpecValue};
use std::ffi::CString;

use super::super::ffi;
use super::super::{
    VkRenderPipeline, VulkanDevice, blend_factor_to_vk, blend_op_to_vk, format_to_vulkan,
    sample_count_to_vk,
};
use super::queries::attr_format_to_vulkan;

impl VulkanDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        // Vulkan requires SPIR-V binaries — MSL/WGSL text source is not supported
        if desc.source.is_some() {
            return Err(QuantaError::compilation_failed(
                "Vulkan backend requires SPIR-V binaries (vertex/fragment), not text source",
            ));
        }
        if desc.vertex.is_empty() || desc.fragment.is_empty() {
            return Err(QuantaError::compilation_failed(
                "vertex and fragment SPIR-V binaries must be non-empty",
            ));
        }
        if !desc.vertex.len().is_multiple_of(4) {
            return Err(QuantaError::compilation_failed(
                "vertex SPIR-V binary length must be a multiple of 4",
            ));
        }
        if !desc.fragment.len().is_multiple_of(4) {
            return Err(QuantaError::compilation_failed(
                "fragment SPIR-V binary length must be a multiple of 4",
            ));
        }
        // Step 063 slice 5 — gate the deferred render-pipeline
        // features. The render-pipeline rebuild that wires
        // TCS+TES SPIR-V (tessellation), object/mesh shader
        // stages, and conservative-rasterization extension state
        // is a separate track. Until then, surface NotSupported
        // when the user asks for them on the descriptor — better
        // than silently dropping the request (matches Kani T419's
        // no-silent-drops contract).
        if desc.tessellation.is_some() {
            return Err(QuantaError::not_supported(
                "Vulkan render pipelines: tessellation (TCS+TES) integration pending — set PipelineDesc.tessellation = None or use the typed TessellationPipeline wrapper",
            ));
        }
        if desc.mesh_shader.is_some() {
            return Err(QuantaError::not_supported(
                "Vulkan render pipelines: object/mesh shader stages pending — use dispatch_mesh on the typed MeshPipeline wrapper",
            ));
        }
        if desc.conservative_rasterization {
            return Err(QuantaError::not_supported(
                "Vulkan render pipelines: conservative rasterization (VK_EXT_conservative_rasterization) pending",
            ));
        }
        let vert_spirv: Vec<u32> = desc
            .vertex
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let frag_spirv: Vec<u32> = desc
            .fragment
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let vert_module_info = ffi::VkShaderModuleCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            code_size: desc.vertex.len(),
            p_code: vert_spirv.as_ptr(),
        };
        let frag_module_info = ffi::VkShaderModuleCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            code_size: desc.fragment.len(),
            p_code: frag_spirv.as_ptr(),
        };

        let mut vert_module = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateShaderModule(
                self.device,
                &vert_module_info,
                core::ptr::null(),
                &mut vert_module,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "vert module: VkResult {}",
                result
            )));
        }

        let mut frag_module = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateShaderModule(
                self.device,
                &frag_module_info,
                core::ptr::null(),
                &mut frag_module,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "frag module: VkResult {}",
                result
            )));
        }

        // Create VkRenderPass
        let color_format = desc
            .color_formats
            .first()
            .copied()
            .unwrap_or(crate::Format::BGRA8);
        let color_attachment = ffi::VkAttachmentDescription {
            flags: 0,
            format: format_to_vulkan(color_format),
            samples: sample_count_to_vk(desc.sample_count),
            load_op: ffi::VK_ATTACHMENT_LOAD_OP_CLEAR,
            store_op: ffi::VK_ATTACHMENT_STORE_OP_STORE,
            stencil_load_op: ffi::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencil_store_op: ffi::VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initial_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
            final_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        let color_ref = ffi::VkAttachmentReference {
            attachment: 0,
            layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass = ffi::VkSubpassDescription {
            flags: 0,
            pipeline_bind_point: ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: core::ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_ref,
            p_resolve_attachments: core::ptr::null(),
            p_depth_stencil_attachment: core::ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: core::ptr::null(),
        };

        let render_pass_info = ffi::VkRenderPassCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            attachment_count: 1,
            p_attachments: &color_attachment,
            subpass_count: 1,
            p_subpasses: &subpass,
            dependency_count: 0,
            p_dependencies: core::ptr::null(),
        };

        let mut render_pass = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateRenderPass(
                self.device,
                &render_pass_info,
                core::ptr::null(),
                &mut render_pass,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "render pass: VkResult {}",
                result
            )));
        }

        // Descriptor set layout: 8 storage buffers (0-7) + 8 combined image samplers (8-15)
        let mut ds_bindings = Vec::new();
        for i in 0..8u32 {
            ds_bindings.push(ffi::VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptor_count: 1,
                stage_flags: ffi::VK_SHADER_STAGE_VERTEX_BIT | ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                p_immutable_samplers: core::ptr::null(),
            });
        }
        for i in 8..16u32 {
            ds_bindings.push(ffi::VkDescriptorSetLayoutBinding {
                binding: i,
                descriptor_type: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                p_immutable_samplers: core::ptr::null(),
            });
        }
        let ds_layout_info = ffi::VkDescriptorSetLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            binding_count: ds_bindings.len() as u32,
            p_bindings: ds_bindings.as_ptr(),
        };
        let mut descriptor_set_layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateDescriptorSetLayout(
                self.device,
                &ds_layout_info,
                core::ptr::null(),
                &mut descriptor_set_layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "ds layout: VkResult {}",
                result
            )));
        }

        let pipeline_layout_info = ffi::VkPipelineLayoutCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            set_layout_count: 1,
            p_set_layouts: &descriptor_set_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges: core::ptr::null(),
        };
        let mut pipeline_layout = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreatePipelineLayout(
                self.device,
                &pipeline_layout_info,
                core::ptr::null(),
                &mut pipeline_layout,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "layout: VkResult {}",
                result
            )));
        }

        // Build specialization info if constants are present.
        // Pack values into a contiguous byte buffer with map entries describing layout.
        let mut spec_data: Vec<u8> = Vec::new();
        let mut spec_entries: Vec<ffi::VkSpecializationMapEntry> = Vec::new();
        for (i, sc) in desc.specialization.iter().enumerate() {
            let offset = spec_data.len() as u32;
            match sc.value {
                SpecValue::F32(v) => {
                    spec_data.extend_from_slice(&v.to_ne_bytes());
                    spec_entries.push(ffi::VkSpecializationMapEntry {
                        constant_id: i as u32,
                        offset,
                        size: 4,
                    });
                }
                SpecValue::I32(v) => {
                    spec_data.extend_from_slice(&v.to_ne_bytes());
                    spec_entries.push(ffi::VkSpecializationMapEntry {
                        constant_id: i as u32,
                        offset,
                        size: 4,
                    });
                }
                SpecValue::U32(v) => {
                    spec_data.extend_from_slice(&v.to_ne_bytes());
                    spec_entries.push(ffi::VkSpecializationMapEntry {
                        constant_id: i as u32,
                        offset,
                        size: 4,
                    });
                }
                SpecValue::Bool(v) => {
                    // Vulkan VK_BOOL32 is a u32
                    let b: u32 = if v { 1 } else { 0 };
                    spec_data.extend_from_slice(&b.to_ne_bytes());
                    spec_entries.push(ffi::VkSpecializationMapEntry {
                        constant_id: i as u32,
                        offset,
                        size: 4,
                    });
                }
            }
        }
        let spec_info = if !spec_entries.is_empty() {
            Some(ffi::VkSpecializationInfo {
                map_entry_count: spec_entries.len() as u32,
                p_map_entries: spec_entries.as_ptr(),
                data_size: spec_data.len(),
                p_data: spec_data.as_ptr() as *const c_void,
            })
        } else {
            None
        };
        let spec_ptr = match spec_info {
            Some(ref info) => info as *const ffi::VkSpecializationInfo as *const c_void,
            None => core::ptr::null(),
        };

        let entry_name = CString::new("main").unwrap();
        let stages = [
            ffi::VkPipelineShaderStageCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                stage: ffi::VK_SHADER_STAGE_VERTEX_BIT,
                module: vert_module,
                p_name: entry_name.as_ptr(),
                p_specialization_info: spec_ptr,
            },
            ffi::VkPipelineShaderStageCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                stage: ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                module: frag_module,
                p_name: entry_name.as_ptr(),
                p_specialization_info: spec_ptr,
            },
        ];

        // Build vertex input bindings and attributes from desc.vertex_layouts
        let mut vk_bindings: Vec<ffi::VkVertexInputBindingDescription> = Vec::new();
        let mut vk_attributes: Vec<ffi::VkVertexInputAttributeDescription> = Vec::new();
        for (buf_idx, layout) in desc.vertex_layouts.iter().enumerate() {
            vk_bindings.push(ffi::VkVertexInputBindingDescription {
                binding: buf_idx as u32,
                stride: layout.stride,
                input_rate: match layout.step {
                    crate::StepMode::Vertex => ffi::VK_VERTEX_INPUT_RATE_VERTEX,
                    crate::StepMode::Instance => ffi::VK_VERTEX_INPUT_RATE_INSTANCE,
                },
            });
            for attr in &layout.attributes {
                vk_attributes.push(ffi::VkVertexInputAttributeDescription {
                    location: attr.location,
                    binding: buf_idx as u32,
                    format: attr_format_to_vulkan(attr.format),
                    offset: attr.offset,
                });
            }
        }

        let vertex_input = ffi::VkPipelineVertexInputStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            vertex_binding_description_count: vk_bindings.len() as u32,
            p_vertex_binding_descriptions: if vk_bindings.is_empty() {
                core::ptr::null()
            } else {
                vk_bindings.as_ptr()
            },
            vertex_attribute_description_count: vk_attributes.len() as u32,
            p_vertex_attribute_descriptions: if vk_attributes.is_empty() {
                core::ptr::null()
            } else {
                vk_attributes.as_ptr()
            },
        };

        let input_assembly = ffi::VkPipelineInputAssemblyStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            topology: ffi::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            primitive_restart_enable: 0,
        };

        let viewport_state = ffi::VkPipelineViewportStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            viewport_count: 1,
            p_viewports: core::ptr::null(),
            scissor_count: 1,
            p_scissors: core::ptr::null(),
        };

        let cull_mode = match desc.cull_mode {
            crate::CullMode::None => ffi::VK_CULL_MODE_NONE,
            crate::CullMode::Front => ffi::VK_CULL_MODE_FRONT_BIT,
            crate::CullMode::Back => ffi::VK_CULL_MODE_BACK_BIT,
        };

        let rasterization = ffi::VkPipelineRasterizationStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: ffi::VK_POLYGON_MODE_FILL,
            cull_mode,
            front_face: ffi::VK_FRONT_FACE_COUNTER_CLOCKWISE,
            depth_bias_enable: 0,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
        };

        let multisample = ffi::VkPipelineMultisampleStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            rasterization_samples: sample_count_to_vk(desc.sample_count),
            sample_shading_enable: 0,
            min_sample_shading: 0.0,
            p_sample_mask: core::ptr::null(),
            alpha_to_coverage_enable: 0,
            alpha_to_one_enable: 0,
        };

        let blend_attachment = ffi::VkPipelineColorBlendAttachmentState {
            blend_enable: if desc.blend.enabled { 1 } else { 0 },
            src_color_blend_factor: blend_factor_to_vk(desc.blend.src_rgb),
            dst_color_blend_factor: blend_factor_to_vk(desc.blend.dst_rgb),
            color_blend_op: blend_op_to_vk(desc.blend.op_rgb),
            src_alpha_blend_factor: blend_factor_to_vk(desc.blend.src_alpha),
            dst_alpha_blend_factor: blend_factor_to_vk(desc.blend.dst_alpha),
            alpha_blend_op: blend_op_to_vk(desc.blend.op_alpha),
            color_write_mask: ffi::VK_COLOR_COMPONENT_RGBA,
        };

        let color_blend = ffi::VkPipelineColorBlendStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            logic_op_enable: 0,
            logic_op: 0,
            attachment_count: 1,
            p_attachments: &blend_attachment,
            blend_constants: [0.0; 4],
        };

        let dynamic_states = [
            ffi::VK_DYNAMIC_STATE_VIEWPORT,
            ffi::VK_DYNAMIC_STATE_SCISSOR,
        ];
        let dynamic_state = ffi::VkPipelineDynamicStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
        };

        let pipeline_info = ffi::VkGraphicsPipelineCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            stage_count: 2,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_tessellation_state: core::ptr::null(),
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: core::ptr::null(),
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: ffi::null_handle(),
            base_pipeline_index: -1,
        };

        let mut pipeline = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateGraphicsPipelines(
                self.device,
                self.pipeline_cache,
                1,
                &pipeline_info,
                core::ptr::null(),
                &mut pipeline,
            )
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::compilation_failed(format!(
                "graphics pipeline: VkResult {}",
                result
            )));
        }

        unsafe {
            ffi::vkDestroyShaderModule(self.device, vert_module, core::ptr::null());
            ffi::vkDestroyShaderModule(self.device, frag_module, core::ptr::null());
        }

        let handle = self.alloc_handle();
        self.render_pipelines
            .write()
            .map_err(|_| QuantaError::internal("lock poisoned"))?
            .insert(
                handle,
                VkRenderPipeline {
                    pipeline,
                    layout: pipeline_layout,
                    render_pass,
                    descriptor_set_layout,
                },
            );
        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }
}

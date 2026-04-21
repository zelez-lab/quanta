//! Render pipeline and render pass operations for Vulkan.

use alloc::format;
use alloc::vec::Vec;
use core::ffi::c_void;

use crate::render_pass::RenderOp;
use crate::{Pipeline, Pulse, QuantaError, RenderPass, Texture};
use std::ffi::CString;

use super::ffi;
use super::{
    VkRenderPipeline, VulkanDevice, blend_factor_to_vk, blend_op_to_vk, format_to_vulkan,
    sample_count_to_vk,
};

impl VulkanDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        if desc.vertex.len() % 4 != 0 {
            return Err(QuantaError::compilation_failed(
                "vertex SPIR-V binary length must be a multiple of 4",
            ));
        }
        if desc.fragment.len() % 4 != 0 {
            return Err(QuantaError::compilation_failed(
                "fragment SPIR-V binary length must be a multiple of 4",
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

        let entry_name = CString::new("main").unwrap();
        let stages = [
            ffi::VkPipelineShaderStageCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                stage: ffi::VK_SHADER_STAGE_VERTEX_BIT,
                module: vert_module,
                p_name: entry_name.as_ptr(),
                p_specialization_info: core::ptr::null(),
            },
            ffi::VkPipelineShaderStageCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                stage: ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                module: frag_module,
                p_name: entry_name.as_ptr(),
                p_specialization_info: core::ptr::null(),
            },
        ];

        let vertex_input = ffi::VkPipelineVertexInputStateCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            vertex_binding_description_count: 0,
            p_vertex_binding_descriptions: core::ptr::null(),
            vertex_attribute_description_count: 0,
            p_vertex_attribute_descriptions: core::ptr::null(),
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
                ffi::null_handle(),
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

    pub(crate) fn render_begin_impl(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle(),
            ops: Vec::new(),
            color_targets: Vec::new(),
            depth_target: None,
        })
    }

    pub(crate) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let pipeline_handle = pass.ops.iter().find_map(|op| {
            if let RenderOp::SetPipeline(h) = op {
                Some(*h)
            } else {
                None
            }
        });

        let render_pipelines = self
            .render_pipelines
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let buffers = self
            .buffers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let samplers = self
            .samplers
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;

        let target_tex = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::invalid_param("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        let (vk_render_pass, pipeline_ref) = if let Some(ph) = pipeline_handle {
            let rp = render_pipelines.get(&ph).ok_or_else(|| {
                QuantaError::invalid_param("pipeline not found")
                    .with_context(&format!("render_end: pipeline handle {}", ph))
            })?;
            (rp.render_pass, Some(rp))
        } else {
            // Create a transient render pass for clear-only usage.
            let color_attachment = ffi::VkAttachmentDescription {
                flags: 0,
                format: target_tex.format,
                samples: ffi::VK_SAMPLE_COUNT_1_BIT,
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
            let rp_info = ffi::VkRenderPassCreateInfo {
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
            let mut transient_rp = ffi::null_handle();
            let result = unsafe {
                ffi::vkCreateRenderPass(self.device, &rp_info, core::ptr::null(), &mut transient_rp)
            };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            (transient_rp, None)
        };

        // Create framebuffer
        let attachments = [target_tex.view];
        let fb_info = ffi::VkFramebufferCreateInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            p_next: core::ptr::null(),
            flags: 0,
            render_pass: vk_render_pass,
            attachment_count: 1,
            p_attachments: attachments.as_ptr(),
            width: target_tex.width,
            height: target_tex.height,
            layers: 1,
        };
        let mut framebuffer = ffi::null_handle();
        let result = unsafe {
            ffi::vkCreateFramebuffer(self.device, &fb_info, core::ptr::null(), &mut framebuffer)
        };
        if result != ffi::VK_SUCCESS {
            return Err(QuantaError::submit_failed());
        }

        // --- Descriptor set allocation and update ---
        let descriptor_pool;
        let descriptor_set;

        if let Some(rp) = pipeline_ref {
            let pool_sizes = [
                ffi::VkDescriptorPoolSize {
                    ty: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    descriptor_count: 8,
                },
                ffi::VkDescriptorPoolSize {
                    ty: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 8,
                },
            ];
            let pool_info = ffi::VkDescriptorPoolCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                max_sets: 1,
                pool_size_count: 2,
                p_pool_sizes: pool_sizes.as_ptr(),
            };
            let mut pool = ffi::null_handle();
            let result = unsafe {
                ffi::vkCreateDescriptorPool(self.device, &pool_info, core::ptr::null(), &mut pool)
            };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            descriptor_pool = Some(pool);

            let alloc_info = ffi::VkDescriptorSetAllocateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: core::ptr::null(),
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &rp.descriptor_set_layout,
            };
            let mut ds = ffi::null_handle();
            let result =
                unsafe { ffi::vkAllocateDescriptorSets(self.device, &alloc_info, &mut ds) };
            if result != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
            descriptor_set = Some(ds);

            // Collect buffer/image info for descriptor writes.
            let mut buffer_infos: Vec<(u32, ffi::VkDescriptorBufferInfo)> = Vec::new();
            let mut image_infos: Vec<(u32, ffi::VkDescriptorImageInfo)> = Vec::new();
            let mut sampler_for_slot: [Option<ffi::VkSampler>; 8] = [None; 8];

            // Default sampler
            let default_sampler_info = ffi::VkSamplerCreateInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                p_next: core::ptr::null(),
                flags: 0,
                mag_filter: ffi::VK_FILTER_LINEAR,
                min_filter: ffi::VK_FILTER_LINEAR,
                mipmap_mode: ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR,
                address_mode_u: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                address_mode_v: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                mip_lod_bias: 0.0,
                anisotropy_enable: 0,
                max_anisotropy: 1.0,
                compare_enable: 0,
                compare_op: 0,
                min_lod: 0.0,
                max_lod: ffi::VK_LOD_CLAMP_NONE,
                border_color: 0,
                unnormalized_coordinates: 0,
            };
            let mut default_sampler = ffi::null_handle();
            unsafe {
                ffi::vkCreateSampler(
                    self.device,
                    &default_sampler_info,
                    core::ptr::null(),
                    &mut default_sampler,
                );
            }

            // First pass: collect sampler assignments
            for op in &pass.ops {
                if let RenderOp::SetSampler { slot, sampler } = op {
                    let idx = *slot as usize;
                    if idx < 8 {
                        let info = ffi::VkSamplerCreateInfo {
                            s_type: ffi::VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                            p_next: core::ptr::null(),
                            flags: 0,
                            mag_filter: super::filter_to_vk(sampler.mag_filter),
                            min_filter: super::filter_to_vk(sampler.min_filter),
                            mipmap_mode: match sampler.mip_filter {
                                crate::render_pass::Filter::Nearest => {
                                    ffi::VK_SAMPLER_MIPMAP_MODE_NEAREST
                                }
                                crate::render_pass::Filter::Linear => {
                                    ffi::VK_SAMPLER_MIPMAP_MODE_LINEAR
                                }
                            },
                            address_mode_u: super::address_to_vk(sampler.address_u),
                            address_mode_v: super::address_to_vk(sampler.address_v),
                            address_mode_w: ffi::VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                            mip_lod_bias: 0.0,
                            anisotropy_enable: if sampler.max_anisotropy > 1 { 1 } else { 0 },
                            max_anisotropy: sampler.max_anisotropy as f32,
                            compare_enable: 0,
                            compare_op: 0,
                            min_lod: 0.0,
                            max_lod: ffi::VK_LOD_CLAMP_NONE,
                            border_color: 0,
                            unnormalized_coordinates: 0,
                        };
                        let mut s = ffi::null_handle();
                        let r = unsafe {
                            ffi::vkCreateSampler(self.device, &info, core::ptr::null(), &mut s)
                        };
                        if r == ffi::VK_SUCCESS {
                            sampler_for_slot[idx] = Some(s);
                        }
                    }
                }
            }

            // Second pass: buffer and image bindings
            for op in &pass.ops {
                match op {
                    RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                        if let Some(buf) = buffers.get(handle) {
                            buffer_infos.push((
                                *slot,
                                ffi::VkDescriptorBufferInfo {
                                    buffer: buf.buffer,
                                    offset: 0,
                                    range: ffi::VK_WHOLE_SIZE,
                                },
                            ));
                        }
                    }
                    RenderOp::SetTexture { slot, handle } => {
                        if let Some(tex) = textures.get(handle) {
                            let idx = *slot as usize;
                            let sampler = if idx < 8 {
                                sampler_for_slot[idx].unwrap_or(default_sampler)
                            } else {
                                default_sampler
                            };
                            image_infos.push((
                                *slot,
                                ffi::VkDescriptorImageInfo {
                                    sampler,
                                    image_view: tex.view,
                                    image_layout: ffi::VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                },
                            ));
                        }
                    }
                    _ => {}
                }
            }

            // Build descriptor writes
            let mut writes: Vec<ffi::VkWriteDescriptorSet> = Vec::new();
            for (slot, info) in &buffer_infos {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: *slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    p_image_info: core::ptr::null(),
                    p_buffer_info: info,
                    p_texel_buffer_view: core::ptr::null(),
                });
            }
            for (slot, info) in &image_infos {
                writes.push(ffi::VkWriteDescriptorSet {
                    s_type: ffi::VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    p_next: core::ptr::null(),
                    dst_set: ds,
                    dst_binding: 8 + *slot,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type: ffi::VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    p_image_info: info,
                    p_buffer_info: core::ptr::null(),
                    p_texel_buffer_view: core::ptr::null(),
                });
            }

            if !writes.is_empty() {
                unsafe {
                    ffi::vkUpdateDescriptorSets(
                        self.device,
                        writes.len() as u32,
                        writes.as_ptr(),
                        0,
                        core::ptr::null(),
                    );
                }
            }
        } else {
            descriptor_pool = None;
            descriptor_set = None;
        }

        // Clear color
        let clear_color = pass
            .ops
            .iter()
            .find_map(|op| {
                if let RenderOp::Clear(c) = op {
                    Some(ffi::VkClearValue {
                        color: ffi::VkClearColorValue {
                            float32: [c.r, c.g, c.b, c.a],
                        },
                    })
                } else {
                    None
                }
            })
            .unwrap_or(ffi::VkClearValue {
                color: ffi::VkClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let clear_values = [clear_color];

        // Allocate command buffer and begin recording.
        let cmd = self.alloc_command_buffer()?;
        let begin_info = ffi::VkCommandBufferBeginInfo {
            s_type: ffi::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            p_next: core::ptr::null(),
            flags: ffi::VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            p_inheritance_info: core::ptr::null(),
        };

        unsafe {
            let r = ffi::vkBeginCommandBuffer(cmd, &begin_info);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }

            // Transition target image to COLOR_ATTACHMENT_OPTIMAL.
            let barrier = ffi::VkImageMemoryBarrier {
                s_type: ffi::VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                p_next: core::ptr::null(),
                src_access_mask: 0,
                dst_access_mask: ffi::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                old_layout: ffi::VK_IMAGE_LAYOUT_UNDEFINED,
                new_layout: ffi::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                src_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: ffi::VK_QUEUE_FAMILY_IGNORED,
                image: target_tex.image,
                subresource_range: ffi::VkImageSubresourceRange {
                    aspect_mask: ffi::VK_IMAGE_ASPECT_COLOR_BIT,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };
            ffi::vkCmdPipelineBarrier(
                cmd,
                ffi::VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                ffi::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                0,
                core::ptr::null(),
                0,
                core::ptr::null(),
                1,
                &barrier,
            );

            // Begin render pass.
            let rp_begin = ffi::VkRenderPassBeginInfo {
                s_type: ffi::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                p_next: core::ptr::null(),
                render_pass: vk_render_pass,
                framebuffer,
                render_area: ffi::VkRect2D {
                    offset: ffi::VkOffset2D { x: 0, y: 0 },
                    extent: ffi::VkExtent2D {
                        width: target_tex.width,
                        height: target_tex.height,
                    },
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
            };
            ffi::vkCmdBeginRenderPass(cmd, &rp_begin, ffi::VK_SUBPASS_CONTENTS_INLINE);

            let mut current_index_buffer: Option<ffi::VkBuffer> = None;

            // Encode each RenderOp.
            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(handle) => {
                        if let Some(rp) = render_pipelines.get(handle) {
                            ffi::vkCmdBindPipeline(
                                cmd,
                                ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                                rp.pipeline,
                            );
                            if let Some(ds) = descriptor_set {
                                ffi::vkCmdBindDescriptorSets(
                                    cmd,
                                    ffi::VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    rp.layout,
                                    0,
                                    1,
                                    &ds,
                                    0,
                                    core::ptr::null(),
                                );
                            }
                        }
                    }

                    RenderOp::BindVertices {
                        slot,
                        handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(handle) {
                            let offsets = [*offset];
                            ffi::vkCmdBindVertexBuffers(
                                cmd,
                                *slot,
                                1,
                                &buf.buffer,
                                offsets.as_ptr(),
                            );
                        }
                    }

                    RenderOp::BindIndices { handle, offset } => {
                        if let Some(buf) = buffers.get(handle) {
                            ffi::vkCmdBindIndexBuffer(
                                cmd,
                                buf.buffer,
                                *offset,
                                ffi::VK_INDEX_TYPE_UINT32,
                            );
                            current_index_buffer = Some(buf.buffer);
                        }
                    }

                    RenderOp::SetField { .. }
                    | RenderOp::SetUniform { .. }
                    | RenderOp::SetTexture { .. }
                    | RenderOp::SetSampler { .. } => {
                        // Already handled via descriptor set update above.
                    }

                    RenderOp::SetValue { slot, data } => {
                        if let Some(rp) = pipeline_ref {
                            ffi::vkCmdPushConstants(
                                cmd,
                                rp.layout,
                                ffi::VK_SHADER_STAGE_VERTEX_BIT | ffi::VK_SHADER_STAGE_FRAGMENT_BIT,
                                (*slot * 16) as u32,
                                data.len() as u32,
                                data.as_ptr() as *const c_void,
                            );
                        }
                    }

                    RenderOp::Draw {
                        vertex_count,
                        instance_count,
                    } => {
                        ffi::vkCmdDraw(cmd, *vertex_count, *instance_count, 0, 0);
                    }

                    RenderOp::DrawIndexed {
                        index_count,
                        instance_count,
                    } => {
                        ffi::vkCmdDrawIndexed(cmd, *index_count, *instance_count, 0, 0, 0);
                    }

                    RenderOp::SetViewport {
                        x,
                        y,
                        width,
                        height,
                        min_depth,
                        max_depth,
                    } => {
                        let viewport = ffi::VkViewport {
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            min_depth: *min_depth,
                            max_depth: *max_depth,
                        };
                        ffi::vkCmdSetViewport(cmd, 0, 1, &viewport);
                    }

                    RenderOp::SetScissor {
                        x,
                        y,
                        width,
                        height,
                    } => {
                        let scissor = ffi::VkRect2D {
                            offset: ffi::VkOffset2D {
                                x: *x as i32,
                                y: *y as i32,
                            },
                            extent: ffi::VkExtent2D {
                                width: *width,
                                height: *height,
                            },
                        };
                        ffi::vkCmdSetScissor(cmd, 0, 1, &scissor);
                    }

                    RenderOp::SetStencilRef(value) => {
                        ffi::vkCmdSetStencilReference(
                            cmd,
                            ffi::VK_STENCIL_FACE_FRONT_AND_BACK,
                            *value,
                        );
                    }

                    RenderOp::DrawIndirect {
                        buffer_handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(buffer_handle) {
                            ffi::vkCmdDrawIndirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::DrawIndexedIndirect {
                        buffer_handle,
                        offset,
                        index_handle,
                    } => {
                        if let Some(idx_buf) = buffers.get(index_handle) {
                            let needs_rebind = current_index_buffer
                                .map(|b| b != idx_buf.buffer)
                                .unwrap_or(true);
                            if needs_rebind {
                                ffi::vkCmdBindIndexBuffer(
                                    cmd,
                                    idx_buf.buffer,
                                    0,
                                    ffi::VK_INDEX_TYPE_UINT32,
                                );
                                current_index_buffer = Some(idx_buf.buffer);
                            }
                        }
                        if let Some(buf) = buffers.get(buffer_handle) {
                            ffi::vkCmdDrawIndexedIndirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::Clear(_) | RenderOp::ClearDepth(_) | RenderOp::ClearStencil(_) => {}
                    RenderOp::DebugPush(_) | RenderOp::DebugPop => {}
                    RenderOp::BeginOcclusionQuery { .. }
                    | RenderOp::EndOcclusionQuery { .. }
                    | RenderOp::SetShadingRate(_)
                    | RenderOp::SetShadingRateImage { .. } => {}
                }
            }

            ffi::vkCmdEndRenderPass(cmd);
            let r = ffi::vkEndCommandBuffer(cmd);
            if r != ffi::VK_SUCCESS {
                return Err(QuantaError::submit_failed());
            }
        }

        let transient_rp = if pipeline_handle.is_none() {
            Some(vk_render_pass)
        } else {
            None
        };
        drop(samplers);
        drop(buffers);
        drop(textures);
        drop(render_pipelines);

        self.submit_and_wait(cmd)?;

        unsafe {
            ffi::vkDestroyFramebuffer(self.device, framebuffer, core::ptr::null());
            if let Some(rp) = transient_rp {
                ffi::vkDestroyRenderPass(self.device, rp, core::ptr::null());
            }
            if let Some(pool) = descriptor_pool {
                ffi::vkDestroyDescriptorPool(self.device, pool, core::ptr::null());
            }
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
            completed: true,
        })
    }
}

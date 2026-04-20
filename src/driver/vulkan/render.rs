//! Render pipeline and render pass operations for Vulkan.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use crate::render_pass::RenderOp;
use crate::{Pipeline, Pulse, QuantaError, RenderPass, Texture};
use ash::vk;
use std::ffi::CString;

use super::{
    VkRenderPipeline, VulkanDevice, blend_factor_to_vk, blend_op_to_vk, format_to_vulkan,
    sample_count_to_vk,
};

impl VulkanDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        // Shader bytes are pre-compiled SPIR-V (same as compute path).
        // Interpret raw bytes as u32 words.
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

        let vert_module_info = vk::ShaderModuleCreateInfo::default().code(&vert_spirv);
        let frag_module_info = vk::ShaderModuleCreateInfo::default().code(&frag_spirv);
        let vert_module = unsafe {
            self.device
                .create_shader_module(&vert_module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("vert module: {:?}", e)))?
        };
        let frag_module = unsafe {
            self.device
                .create_shader_module(&frag_module_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("frag module: {:?}", e)))?
        };

        // Create VkRenderPass
        let color_format = desc
            .color_formats
            .first()
            .copied()
            .unwrap_or(crate::Format::BGRA8);
        let color_attachment = vk::AttachmentDescription::default()
            .format(format_to_vulkan(color_format))
            .samples(sample_count_to_vk(desc.sample_count))
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass));

        let render_pass = unsafe {
            self.device
                .create_render_pass(&render_pass_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("render pass: {:?}", e)))?
        };

        // Descriptor set layout: 8 storage buffers (0-7) + 8 combined image samplers (8-15)
        let mut ds_bindings = Vec::new();
        for i in 0..8u32 {
            ds_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            );
        }
        for i in 8..16u32 {
            ds_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
        }
        let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&ds_bindings);
        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&ds_layout_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("ds layout: {:?}", e)))?
        };

        let ds_layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&ds_layouts);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .map_err(|e| QuantaError::compilation_failed(format!("layout: {:?}", e)))?
        };

        let entry_name = CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_name),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match desc.cull_mode {
                crate::CullMode::None => vk::CullModeFlags::NONE,
                crate::CullMode::Front => vk::CullModeFlags::FRONT,
                crate::CullMode::Back => vk::CullModeFlags::BACK,
            })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(sample_count_to_vk(desc.sample_count));

        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(desc.blend.enabled)
            .src_color_blend_factor(blend_factor_to_vk(desc.blend.src_rgb))
            .dst_color_blend_factor(blend_factor_to_vk(desc.blend.dst_rgb))
            .color_blend_op(blend_op_to_vk(desc.blend.op_rgb))
            .src_alpha_blend_factor(blend_factor_to_vk(desc.blend.src_alpha))
            .dst_alpha_blend_factor(blend_factor_to_vk(desc.blend.dst_alpha))
            .alpha_blend_op(blend_op_to_vk(desc.blend.op_alpha))
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| {
                    QuantaError::compilation_failed(format!("graphics pipeline: {:?}", e.1))
                })?[0]
        };

        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            self.device.destroy_shader_module(frag_module, None);
        }

        let handle = self.alloc_handle();
        self.render_pipelines.lock().unwrap().insert(
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
        // Find the pipeline handle from ops to get the VkRenderPass object.
        let pipeline_handle = pass.ops.iter().find_map(|op| {
            if let RenderOp::SetPipeline(h) = op {
                Some(*h)
            } else {
                None
            }
        });

        let render_pipelines = self.render_pipelines.lock().unwrap();
        let textures = self.textures.lock().unwrap();
        let buffers = self.buffers.lock().unwrap();
        let samplers = self.samplers.lock().unwrap();

        let target_tex = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::invalid_param("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        // Get the render pass object from the pipeline. If no pipeline was set,
        // create a transient render pass for the target format.
        let (vk_render_pass, pipeline_ref) = if let Some(ph) = pipeline_handle {
            let rp = render_pipelines.get(&ph).ok_or_else(|| {
                QuantaError::invalid_param("pipeline not found")
                    .with_context(&format!("render_end: pipeline handle {}", ph))
            })?;
            (rp.render_pass, Some(rp))
        } else {
            // No pipeline bound -- create a transient render pass for clear-only usage.
            let color_attachment = vk::AttachmentDescription::default()
                .format(target_tex.format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let color_ref = vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

            let subpass = vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(std::slice::from_ref(&color_ref));

            let rp_info = vk::RenderPassCreateInfo::default()
                .attachments(std::slice::from_ref(&color_attachment))
                .subpasses(std::slice::from_ref(&subpass));

            let transient_rp = unsafe {
                self.device
                    .create_render_pass(&rp_info, None)
                    .map_err(|_| QuantaError::submit_failed())?
            };
            (transient_rp, None)
        };

        // Create framebuffer from the target texture's image view.
        let attachments = [target_tex.view];
        let fb_info = vk::FramebufferCreateInfo::default()
            .render_pass(vk_render_pass)
            .attachments(&attachments)
            .width(target_tex.width)
            .height(target_tex.height)
            .layers(1);

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(&fb_info, None)
                .map_err(|_| QuantaError::submit_failed())?
        };

        // --- Descriptor set allocation and update ---
        // If a pipeline is bound, allocate a descriptor set and pre-populate it
        // with all SetField/SetTexture/SetSampler ops before recording commands.
        let descriptor_pool;
        let descriptor_set;

        if let Some(rp) = pipeline_ref {
            // Create descriptor pool: 8 storage buffers + 8 combined image samplers
            let pool_sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(8),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(8),
            ];
            let pool_info = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&pool_sizes);
            descriptor_pool = Some(unsafe {
                self.device
                    .create_descriptor_pool(&pool_info, None)
                    .map_err(|_| QuantaError::submit_failed())?
            });

            let layouts = [rp.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool.unwrap())
                .set_layouts(&layouts);
            let sets = unsafe {
                self.device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(|_| QuantaError::submit_failed())?
            };
            descriptor_set = Some(sets[0]);

            // Walk ops to collect descriptor writes.
            // We need stable references for the Vulkan write structs, so collect
            // buffer/image info into Vecs first.
            let mut buffer_infos: Vec<(u32, vk::DescriptorBufferInfo)> = Vec::new();
            let mut image_infos: Vec<(u32, vk::DescriptorImageInfo)> = Vec::new();

            // Track per-slot sampler overrides. SetSampler ops before a SetTexture
            // apply to that texture slot.
            let mut sampler_for_slot: [Option<vk::Sampler>; 8] = [None; 8];

            // Create a default sampler for textures that don't have an explicit one.
            let default_sampler_info = vk::SamplerCreateInfo::default()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .min_lod(0.0)
                .max_lod(vk::LOD_CLAMP_NONE);
            let default_sampler = unsafe {
                self.device
                    .create_sampler(&default_sampler_info, None)
                    .map_err(|_| QuantaError::submit_failed())?
            };

            // First pass: collect sampler assignments
            for op in &pass.ops {
                if let RenderOp::SetSampler { slot, sampler } = op {
                    let idx = *slot as usize;
                    if idx < 8 {
                        // Create an inline sampler from the desc
                        let info = vk::SamplerCreateInfo::default()
                            .min_filter(super::filter_to_vk(sampler.min_filter))
                            .mag_filter(super::filter_to_vk(sampler.mag_filter))
                            .mipmap_mode(match sampler.mip_filter {
                                crate::render_pass::Filter::Nearest => {
                                    vk::SamplerMipmapMode::NEAREST
                                }
                                crate::render_pass::Filter::Linear => vk::SamplerMipmapMode::LINEAR,
                            })
                            .address_mode_u(super::address_to_vk(sampler.address_u))
                            .address_mode_v(super::address_to_vk(sampler.address_v))
                            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                            .max_anisotropy(sampler.max_anisotropy as f32)
                            .anisotropy_enable(sampler.max_anisotropy > 1)
                            .min_lod(0.0)
                            .max_lod(vk::LOD_CLAMP_NONE);
                        if let Ok(s) = unsafe { self.device.create_sampler(&info, None) } {
                            sampler_for_slot[idx] = Some(s);
                        }
                    }
                }
            }

            // Second pass: collect buffer and image bindings
            for op in &pass.ops {
                match op {
                    RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                        if let Some(buf) = buffers.get(handle) {
                            buffer_infos.push((
                                *slot,
                                vk::DescriptorBufferInfo::default()
                                    .buffer(buf.buffer)
                                    .offset(0)
                                    .range(vk::WHOLE_SIZE),
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
                                vk::DescriptorImageInfo::default()
                                    .image_view(tex.view)
                                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                    .sampler(sampler),
                            ));
                        }
                    }
                    _ => {}
                }
            }

            // Build write descriptor sets
            let mut writes: Vec<vk::WriteDescriptorSet> = Vec::new();
            for (slot, info) in &buffer_infos {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set.unwrap())
                        .dst_binding(*slot)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info)),
                );
            }
            for (slot, info) in &image_infos {
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(descriptor_set.unwrap())
                        .dst_binding(8 + *slot)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(std::slice::from_ref(info)),
                );
            }

            if !writes.is_empty() {
                unsafe {
                    self.device.update_descriptor_sets(&writes, &[]);
                }
            }
        } else {
            descriptor_pool = None;
            descriptor_set = None;
        }

        // Determine clear color from ops (first Clear op or default black).
        let clear_color = pass
            .ops
            .iter()
            .find_map(|op| {
                if let RenderOp::Clear(c) = op {
                    Some(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [c.r, c.g, c.b, c.a],
                        },
                    })
                } else {
                    None
                }
            })
            .unwrap_or(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let clear_values = [clear_color];

        // Allocate command buffer and begin recording.
        let cmd = self.alloc_command_buffer()?;
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin_info)
                .map_err(|_| QuantaError::submit_failed())?;

            // Transition target image to COLOR_ATTACHMENT_OPTIMAL.
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(target_tex.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Begin render pass.
            let rp_begin = vk::RenderPassBeginInfo::default()
                .render_pass(vk_render_pass)
                .framebuffer(framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: target_tex.width,
                        height: target_tex.height,
                    },
                })
                .clear_values(&clear_values);
            self.device
                .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            // Track last bound index buffer for DrawIndexedIndirect rebind detection.
            let mut current_index_buffer: Option<vk::Buffer> = None;

            // Encode each RenderOp.
            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(handle) => {
                        if let Some(rp) = render_pipelines.get(handle) {
                            self.device.cmd_bind_pipeline(
                                cmd,
                                vk::PipelineBindPoint::GRAPHICS,
                                rp.pipeline,
                            );
                            // Bind the descriptor set immediately after the pipeline.
                            if let Some(ds) = descriptor_set {
                                self.device.cmd_bind_descriptor_sets(
                                    cmd,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    rp.layout,
                                    0,
                                    &[ds],
                                    &[],
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
                            self.device.cmd_bind_vertex_buffers(
                                cmd,
                                *slot,
                                &[buf.buffer],
                                &offsets,
                            );
                        }
                    }

                    RenderOp::BindIndices { handle, offset } => {
                        if let Some(buf) = buffers.get(handle) {
                            self.device.cmd_bind_index_buffer(
                                cmd,
                                buf.buffer,
                                *offset,
                                vk::IndexType::UINT32,
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
                        // Push constants -- use the pipeline layout's push constant range.
                        if let Some(rp) = pipeline_ref {
                            self.device.cmd_push_constants(
                                cmd,
                                rp.layout,
                                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                (*slot * 16) as u32, // 16-byte aligned offset per slot
                                data,
                            );
                        }
                    }

                    RenderOp::Draw {
                        vertex_count,
                        instance_count,
                    } => {
                        self.device
                            .cmd_draw(cmd, *vertex_count, *instance_count, 0, 0);
                    }

                    RenderOp::DrawIndexed {
                        index_count,
                        instance_count,
                    } => {
                        self.device
                            .cmd_draw_indexed(cmd, *index_count, *instance_count, 0, 0, 0);
                    }

                    RenderOp::SetViewport {
                        x,
                        y,
                        width,
                        height,
                        min_depth,
                        max_depth,
                    } => {
                        let viewport = vk::Viewport {
                            x: *x,
                            y: *y,
                            width: *width,
                            height: *height,
                            min_depth: *min_depth,
                            max_depth: *max_depth,
                        };
                        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
                    }

                    RenderOp::SetScissor {
                        x,
                        y,
                        width,
                        height,
                    } => {
                        let scissor = vk::Rect2D {
                            offset: vk::Offset2D {
                                x: *x as i32,
                                y: *y as i32,
                            },
                            extent: vk::Extent2D {
                                width: *width,
                                height: *height,
                            },
                        };
                        self.device.cmd_set_scissor(cmd, 0, &[scissor]);
                    }

                    RenderOp::SetStencilRef(value) => {
                        self.device.cmd_set_stencil_reference(
                            cmd,
                            vk::StencilFaceFlags::FRONT_AND_BACK,
                            *value,
                        );
                    }

                    RenderOp::DrawIndirect {
                        buffer_handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(buffer_handle) {
                            self.device
                                .cmd_draw_indirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::DrawIndexedIndirect {
                        buffer_handle,
                        offset,
                        index_handle,
                    } => {
                        // Bind index buffer if different from current.
                        if let Some(idx_buf) = buffers.get(index_handle) {
                            let needs_rebind = current_index_buffer
                                .map(|b| b != idx_buf.buffer)
                                .unwrap_or(true);
                            if needs_rebind {
                                self.device.cmd_bind_index_buffer(
                                    cmd,
                                    idx_buf.buffer,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                current_index_buffer = Some(idx_buf.buffer);
                            }
                        }
                        if let Some(buf) = buffers.get(buffer_handle) {
                            self.device
                                .cmd_draw_indexed_indirect(cmd, buf.buffer, *offset, 1, 0);
                        }
                    }

                    RenderOp::Clear(_) => {
                        // Handled by render pass load action (clear values above).
                    }

                    RenderOp::ClearDepth(_) => {
                        // Handled by render pass load action for depth attachment.
                    }

                    RenderOp::ClearStencil(_) => {
                        // Handled by render pass load action for stencil attachment.
                    }

                    RenderOp::DebugPush(_label) => {
                        // VK_EXT_debug_utils: vkCmdBeginDebugUtilsLabelEXT
                        // Requires extension — skip for now.
                    }

                    RenderOp::DebugPop => {
                        // VK_EXT_debug_utils: vkCmdEndDebugUtilsLabelEXT
                    }

                    // M2+ render ops — not yet implemented in the Vulkan driver.
                    RenderOp::BeginOcclusionQuery { .. }
                    | RenderOp::EndOcclusionQuery { .. }
                    | RenderOp::SetShadingRate(_)
                    | RenderOp::SetShadingRateImage { .. } => {}
                }
            }

            // End render pass.
            self.device.cmd_end_render_pass(cmd);

            // End command buffer.
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }

        // Release locks before submit (submit_and_wait acquires none).
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

        // Clean up framebuffer, transient render pass, and descriptor pool.
        unsafe {
            self.device.destroy_framebuffer(framebuffer, None);
            if let Some(rp) = transient_rp {
                self.device.destroy_render_pass(rp, None);
            }
            if let Some(pool) = descriptor_pool {
                // Destroying the pool frees all descriptor sets allocated from it.
                self.device.destroy_descriptor_pool(pool, None);
            }
        }

        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
        })
    }
}

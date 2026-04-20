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
        let vert_wgsl = std::str::from_utf8(desc.vertex)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in vertex shader"))?;
        let frag_wgsl = std::str::from_utf8(desc.fragment)
            .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in fragment shader"))?;

        // NOTE: super::super resolves to crate::driver -- same as the original
        // super::spirv path from the flat vulkan.rs file. The spirv module does
        // not exist yet; this will fail to compile until it is added.
        let vert_spirv = super::super::spirv::wgsl_to_spirv(vert_wgsl)
            .map_err(QuantaError::compilation_failed)?;
        let frag_spirv = super::super::spirv::wgsl_to_spirv(frag_wgsl)
            .map_err(QuantaError::compilation_failed)?;

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

        // Pipeline layout (empty for now -- no descriptors for render)
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
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

                    RenderOp::SetField { slot: _, handle: _ }
                    | RenderOp::SetUniform { slot: _, handle: _ } => {
                        // Descriptor set binding would go here. For now, storage/uniform
                        // buffers require descriptor sets which need layout integration.
                        // TODO: implement descriptor set updates for render resources.
                    }

                    RenderOp::SetTexture { slot: _, handle: _ } => {
                        // TODO: bind texture via descriptor set update.
                    }

                    RenderOp::SetSampler { .. } => {
                        // TODO: bind sampler via descriptor set update.
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
        drop(buffers);
        drop(textures);
        drop(render_pipelines);

        self.submit_and_wait(cmd)?;

        // Clean up framebuffer and transient render pass.
        unsafe {
            self.device.destroy_framebuffer(framebuffer, None);
            if let Some(rp) = transient_rp {
                self.device.destroy_render_pass(rp, None);
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

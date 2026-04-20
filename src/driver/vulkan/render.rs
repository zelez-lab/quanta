//! Render pipeline and render pass operations for Vulkan.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

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
            // Note: render_pass is needed for framebuffer creation in render_end
            // Store it alongside the pipeline -- for now, leak it (TODO: proper storage)
        }

        let handle = self.alloc_handle();
        self.render_pipelines.lock().unwrap().insert(
            handle,
            VkRenderPipeline {
                pipeline,
                layout: pipeline_layout,
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

    pub(crate) fn render_end_impl(&self, _pass: RenderPass) -> Result<Pulse, QuantaError> {
        // For now, render_end submits an empty command buffer.
        // Full RenderOp encoding requires VkFramebuffer creation from the target texture,
        // which needs the VkRenderPass stored from pipeline_create.
        // TODO: store render_pass handle and create framebuffer here.
        let cmd = self.alloc_command_buffer()?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device
                .begin_command_buffer(cmd, &begin)
                .map_err(|_| QuantaError::submit_failed())?;
            // TODO: begin render pass, encode ops, end render pass
            self.device
                .end_command_buffer(cmd)
                .map_err(|_| QuantaError::submit_failed())?;
        }
        self.submit_and_wait(cmd)?;

        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(|_| Ok(()))),
            poll_fn: None,
            completed: false,
        })
    }
}

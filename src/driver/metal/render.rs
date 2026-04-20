//! Render pipeline and pass execution for Metal.

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::format;

use crate::{Pipeline, Pulse, QuantaError, RenderPass, render_pass::RenderOp};
use metal as mtl;

use super::{
    MetalDevice, blend_factor_to_metal, blend_op_to_metal, compare_to_metal, format_to_metal,
    stencil_op_to_metal,
};

impl MetalDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        let opts = mtl::CompileOptions::new();

        // Support combined source or separate vertex/fragment sources
        let (vert_fn, frag_fn) = if let Some(combined) = desc.source {
            let src = std::str::from_utf8(combined)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in shader source"))?;
            let lib = self
                .device
                .new_library_with_source(src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("shader: {}", e)))?;
            let vf = lib.get_function(desc.vertex_entry, None).map_err(|e| {
                QuantaError::compilation_failed(format!("vertex fn '{}': {}", desc.vertex_entry, e))
            })?;
            let ff = lib.get_function(desc.fragment_entry, None).map_err(|e| {
                QuantaError::compilation_failed(format!(
                    "fragment fn '{}': {}",
                    desc.fragment_entry, e
                ))
            })?;
            (vf, ff)
        } else {
            let vert_src = std::str::from_utf8(desc.vertex)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in vertex shader"))?;
            let frag_src = std::str::from_utf8(desc.fragment)
                .map_err(|_| QuantaError::compilation_failed("invalid UTF-8 in fragment shader"))?;
            let vert_lib = self
                .device
                .new_library_with_source(vert_src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("vertex: {}", e)))?;
            let frag_lib = self
                .device
                .new_library_with_source(frag_src, &opts)
                .map_err(|e| QuantaError::compilation_failed(format!("fragment: {}", e)))?;
            let vf = vert_lib
                .get_function(desc.vertex_entry, None)
                .map_err(|e| QuantaError::compilation_failed(format!("vertex fn: {}", e)))?;
            let ff = frag_lib
                .get_function(desc.fragment_entry, None)
                .map_err(|e| QuantaError::compilation_failed(format!("fragment fn: {}", e)))?;
            (vf, ff)
        };

        let pipe_desc = mtl::RenderPipelineDescriptor::new();
        pipe_desc.set_vertex_function(Some(&vert_fn));
        pipe_desc.set_fragment_function(Some(&frag_fn));

        for (i, fmt) in desc.color_formats.iter().enumerate() {
            let ca = pipe_desc.color_attachments().object_at(i as u64).unwrap();
            ca.set_pixel_format(format_to_metal(*fmt));
            if desc.blend.enabled {
                ca.set_blending_enabled(true);
                ca.set_source_rgb_blend_factor(blend_factor_to_metal(desc.blend.src_rgb));
                ca.set_destination_rgb_blend_factor(blend_factor_to_metal(desc.blend.dst_rgb));
                ca.set_source_alpha_blend_factor(blend_factor_to_metal(desc.blend.src_alpha));
                ca.set_destination_alpha_blend_factor(blend_factor_to_metal(desc.blend.dst_alpha));
                ca.set_rgb_blend_operation(blend_op_to_metal(desc.blend.op_rgb));
                ca.set_alpha_blend_operation(blend_op_to_metal(desc.blend.op_alpha));
            }
        }

        if let Some(depth_fmt) = desc.depth_format {
            pipe_desc.set_depth_attachment_pixel_format(format_to_metal(depth_fmt));
        }

        pipe_desc.set_sample_count(desc.sample_count as u64);

        let pipeline_state = self
            .device
            .new_render_pipeline_state(&pipe_desc)
            .map_err(|e| QuantaError::compilation_failed(format!("render pipeline: {}", e)))?;

        // Create depth/stencil state
        let ds_desc = mtl::DepthStencilDescriptor::new();
        if desc.depth_stencil.depth_test {
            ds_desc.set_depth_compare_function(compare_to_metal(desc.depth_stencil.depth_compare));
            ds_desc.set_depth_write_enabled(desc.depth_stencil.depth_write);
        }
        if let Some(ref front) = desc.depth_stencil.stencil_front {
            let s = mtl::StencilDescriptor::new();
            s.set_stencil_failure_operation(stencil_op_to_metal(front.fail));
            s.set_depth_failure_operation(stencil_op_to_metal(front.depth_fail));
            s.set_depth_stencil_pass_operation(stencil_op_to_metal(front.pass));
            s.set_stencil_compare_function(compare_to_metal(front.compare));
            s.set_read_mask(front.read_mask);
            s.set_write_mask(front.write_mask);
            ds_desc.set_front_face_stencil(Some(&s));
        }
        if let Some(ref back) = desc.depth_stencil.stencil_back {
            let s = mtl::StencilDescriptor::new();
            s.set_stencil_failure_operation(stencil_op_to_metal(back.fail));
            s.set_depth_failure_operation(stencil_op_to_metal(back.depth_fail));
            s.set_depth_stencil_pass_operation(stencil_op_to_metal(back.pass));
            s.set_stencil_compare_function(compare_to_metal(back.compare));
            s.set_read_mask(back.read_mask);
            s.set_write_mask(back.write_mask);
            ds_desc.set_back_face_stencil(Some(&s));
        }
        let ds_state = self.device.new_depth_stencil_state(&ds_desc);

        let handle = self.alloc_handle();
        self.render_pipelines
            .lock()
            .unwrap()
            .insert(handle, pipeline_state);
        self.depth_stencil_states
            .lock()
            .unwrap()
            .insert(handle, ds_state);
        Ok(Pipeline {
            handle,
            drop_fn: None,
        })
    }

    pub(crate) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let textures = self.textures.lock().unwrap();
        let target = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::invalid_param("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        // Create render pass descriptor
        let rpd = mtl::RenderPassDescriptor::new();
        let color_attach = rpd.color_attachments().object_at(0).unwrap();
        color_attach.set_texture(Some(target));
        color_attach.set_load_action(mtl::MTLLoadAction::Clear);
        color_attach.set_store_action(mtl::MTLStoreAction::Store);
        color_attach.set_clear_color(mtl::MTLClearColor::new(0.0, 0.0, 0.0, 1.0));

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_render_command_encoder(rpd);

        let buffers = self.buffers.lock().unwrap();
        let render_pipelines = self.render_pipelines.lock().unwrap();

        for op in &pass.ops {
            match op {
                RenderOp::SetPipeline(handle) => {
                    if let Some(ps) = render_pipelines.get(handle) {
                        encoder.set_render_pipeline_state(ps);
                    }
                    let ds_states = self.depth_stencil_states.lock().unwrap();
                    if let Some(ds) = ds_states.get(handle) {
                        encoder.set_depth_stencil_state(ds);
                    }
                }
                RenderOp::BindVertices {
                    slot,
                    handle,
                    offset,
                } => {
                    if let Some(buf) = buffers.get(handle) {
                        encoder.set_vertex_buffer(*slot as u64, Some(buf), *offset);
                    }
                }
                RenderOp::BindIndices { .. } => {
                    // Index buffer is bound at draw_indexed time in Metal
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    if let Some(buf) = buffers.get(handle) {
                        encoder.set_vertex_buffer(*slot as u64, Some(buf), 0);
                    }
                }
                RenderOp::SetTexture { slot, handle } => {
                    if let Some(tex) = textures.get(handle) {
                        encoder.set_fragment_texture(*slot as u64, Some(tex));
                    }
                }
                RenderOp::SetValue { slot, data } => {
                    encoder.set_vertex_bytes(
                        *slot as u64,
                        data.len() as u64,
                        data.as_ptr() as *const _,
                    );
                }
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    if *instance_count <= 1 {
                        encoder.draw_primitives(
                            mtl::MTLPrimitiveType::Triangle,
                            0,
                            *vertex_count as u64,
                        );
                    } else {
                        encoder.draw_primitives_instanced(
                            mtl::MTLPrimitiveType::Triangle,
                            0,
                            *vertex_count as u64,
                            *instance_count as u64,
                        );
                    }
                }
                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    // Find the last BindIndices op to get the index buffer
                    let idx_handle = pass.ops.iter().rev().find_map(|op| {
                        if let RenderOp::BindIndices { handle, .. } = op {
                            Some(*handle)
                        } else {
                            None
                        }
                    });
                    if let Some(ih) = idx_handle
                        && let Some(idx_buf) = buffers.get(&ih)
                    {
                        if *instance_count <= 1 {
                            encoder.draw_indexed_primitives(
                                mtl::MTLPrimitiveType::Triangle,
                                *index_count as u64,
                                mtl::MTLIndexType::UInt32,
                                idx_buf,
                                0,
                            );
                        } else {
                            encoder.draw_indexed_primitives_instanced(
                                mtl::MTLPrimitiveType::Triangle,
                                *index_count as u64,
                                mtl::MTLIndexType::UInt32,
                                idx_buf,
                                0,
                                *instance_count as u64,
                            );
                        }
                    }
                }
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => {
                    encoder.set_scissor_rect(mtl::MTLScissorRect {
                        x: *x as u64,
                        y: *y as u64,
                        width: *width as u64,
                        height: *height as u64,
                    });
                }
                RenderOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    encoder.set_viewport(mtl::MTLViewport {
                        originX: *x as f64,
                        originY: *y as f64,
                        width: *width as f64,
                        height: *height as f64,
                        znear: *min_depth as f64,
                        zfar: *max_depth as f64,
                    });
                }
                RenderOp::Clear(_color) => {
                    // Clear is handled by load action on the render pass descriptor.
                    // Dynamic clear within a pass would need a new encoder — skip for now.
                }
                RenderOp::ClearDepth(_depth) => {
                    // Same — handled by render pass descriptor load action.
                }
                RenderOp::SetStencilRef(value) => {
                    encoder.set_stencil_reference_value(*value);
                }
                RenderOp::ClearStencil(_) => {
                    // Handled by render pass descriptor load action
                }
                RenderOp::DrawIndirect {
                    buffer_handle,
                    offset,
                } => {
                    if let Some(buf) = buffers.get(buffer_handle) {
                        encoder.draw_primitives_indirect(
                            mtl::MTLPrimitiveType::Triangle,
                            buf,
                            *offset,
                        );
                    }
                }
                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    offset,
                    index_handle,
                } => {
                    if let Some(buf) = buffers.get(buffer_handle)
                        && let Some(idx_buf) = buffers.get(index_handle)
                    {
                        encoder.draw_indexed_primitives_indirect(
                            mtl::MTLPrimitiveType::Triangle,
                            mtl::MTLIndexType::UInt32,
                            idx_buf,
                            0,
                            buf,
                            *offset,
                        );
                    }
                }
                RenderOp::DebugPush(label) => {
                    encoder.push_debug_group(label);
                }
                RenderOp::DebugPop => {
                    encoder.pop_debug_group();
                }
                RenderOp::SetSampler {
                    slot,
                    sampler: desc,
                } => {
                    let sd = mtl::SamplerDescriptor::new();
                    sd.set_min_filter(super::filter_to_metal(desc.min_filter));
                    sd.set_mag_filter(super::filter_to_metal(desc.mag_filter));
                    sd.set_address_mode_s(super::address_to_metal(desc.address_u));
                    sd.set_address_mode_t(super::address_to_metal(desc.address_v));
                    sd.set_max_anisotropy(desc.max_anisotropy as u64);
                    let samp = self.device.new_sampler(&sd);
                    encoder.set_fragment_sampler_state(*slot as u64, Some(&samp));
                }
                // M2+ render ops — not yet implemented in the Metal driver.
                // Fall through silently so the API surface exists without crashing.
                RenderOp::BeginOcclusionQuery { .. }
                | RenderOp::EndOcclusionQuery { .. }
                | RenderOp::SetShadingRate(_)
                | RenderOp::SetShadingRateImage { .. } => {}
            }
        }

        encoder.end_encoding();
        cmd.commit();

        let cmd_clone = cmd.to_owned();
        Ok(Pulse {
            handle: self.alloc_handle(),
            wait_fn: Some(Box::new(move |_| {
                cmd_clone.wait_until_completed();
                Ok(())
            })),
            poll_fn: None,
            completed: false,
        })
    }
}

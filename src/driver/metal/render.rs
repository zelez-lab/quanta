//! Render pipeline and pass execution for Metal.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    LoadOp, Pipeline, Pulse, QuantaError, RenderPass, SpecValue, StoreOp, render_pass::RenderOp,
};

use super::ffi;
use super::{
    MetalDevice, blend_factor_to_metal, blend_op_to_metal, compare_to_metal, format_to_metal,
    stencil_op_to_metal,
};

impl MetalDevice {
    pub(crate) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        // Build MTLFunctionConstantValues if specialization constants are present.
        let fcv = if !desc.specialization.is_empty() {
            unsafe {
                let fcv = ffi::msg_id(
                    ffi::cls(b"MTLFunctionConstantValues\0") as ffi::Id,
                    b"new\0",
                );
                for (index, sc) in desc.specialization.iter().enumerate() {
                    match sc.value {
                        SpecValue::F32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const f32 as *const _,
                                ffi::MTL_DATA_TYPE_FLOAT,
                                index as u64,
                            );
                        }
                        SpecValue::I32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const i32 as *const _,
                                ffi::MTL_DATA_TYPE_INT,
                                index as u64,
                            );
                        }
                        SpecValue::U32(v) => {
                            ffi::msg_set_constant_value(
                                fcv,
                                &v as *const u32 as *const _,
                                ffi::MTL_DATA_TYPE_UINT,
                                index as u64,
                            );
                        }
                        SpecValue::Bool(v) => {
                            let b: u8 = if v { 1 } else { 0 };
                            ffi::msg_set_constant_value(
                                fcv,
                                &b as *const u8 as *const _,
                                ffi::MTL_DATA_TYPE_BOOL,
                                index as u64,
                            );
                        }
                    }
                }
                Some(fcv)
            }
        } else {
            None
        };

        // Compile shader source(s) into Metal library/libraries.
        let (vert_fn, frag_fn) = unsafe {
            if let Some(combined) = desc.source {
                let src = std::str::from_utf8(combined).map_err(|_| {
                    QuantaError::compilation_failed("invalid UTF-8 in shader source")
                })?;
                let mut src_bytes: Vec<u8> = src.bytes().collect();
                src_bytes.push(0);
                let ns_src = ffi::nsstring(&src_bytes);
                let (lib, error) = ffi::msg_new_library_with_source(self.device, ns_src, ffi::NIL);
                if lib.is_null() {
                    let msg = error_string(error);
                    return Err(QuantaError::compilation_failed(format!("shader: {msg}")));
                }
                let vf = get_function_maybe_specialized(lib, desc.vertex_entry, fcv)?;
                let ff = get_function_maybe_specialized(lib, desc.fragment_entry, fcv)?;
                (vf, ff)
            } else {
                let vert_src = std::str::from_utf8(desc.vertex).map_err(|_| {
                    QuantaError::compilation_failed("invalid UTF-8 in vertex shader")
                })?;
                let frag_src = std::str::from_utf8(desc.fragment).map_err(|_| {
                    QuantaError::compilation_failed("invalid UTF-8 in fragment shader")
                })?;

                let mut vb: Vec<u8> = vert_src.bytes().collect();
                vb.push(0);
                let ns_vert = ffi::nsstring(&vb);
                let (vert_lib, err) =
                    ffi::msg_new_library_with_source(self.device, ns_vert, ffi::NIL);
                if vert_lib.is_null() {
                    let msg = error_string(err);
                    return Err(QuantaError::compilation_failed(format!("vertex: {msg}")));
                }

                let mut fb: Vec<u8> = frag_src.bytes().collect();
                fb.push(0);
                let ns_frag = ffi::nsstring(&fb);
                let (frag_lib, err) =
                    ffi::msg_new_library_with_source(self.device, ns_frag, ffi::NIL);
                if frag_lib.is_null() {
                    let msg = error_string(err);
                    return Err(QuantaError::compilation_failed(format!("fragment: {msg}")));
                }

                let vf = get_function_maybe_specialized(vert_lib, desc.vertex_entry, fcv)?;
                let ff = get_function_maybe_specialized(frag_lib, desc.fragment_entry, fcv)?;
                (vf, ff)
            }
        };

        unsafe {
            let pipe_desc = ffi::msg_id(
                ffi::cls(b"MTLRenderPipelineDescriptor\0") as ffi::Id,
                b"new\0",
            );
            ffi::msg_void_id(pipe_desc, b"setVertexFunction:\0", vert_fn);
            ffi::msg_void_id(pipe_desc, b"setFragmentFunction:\0", frag_fn);

            // Color attachments
            let attachments = ffi::msg_id(pipe_desc, b"colorAttachments\0");
            for (i, fmt) in desc.color_formats.iter().enumerate() {
                let ca = ffi::msg_id_u64(attachments, b"objectAtIndexedSubscript:\0", i as u64);
                ffi::msg_void_u64(ca, b"setPixelFormat:\0", format_to_metal(*fmt));
                if desc.blend.enabled {
                    ffi::msg_void_bool(ca, b"setBlendingEnabled:\0", true);
                    ffi::msg_void_u64(
                        ca,
                        b"setSourceRGBBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.src_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setDestinationRGBBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.dst_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setSourceAlphaBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.src_alpha),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setDestinationAlphaBlendFactor:\0",
                        blend_factor_to_metal(desc.blend.dst_alpha),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setRgbBlendOperation:\0",
                        blend_op_to_metal(desc.blend.op_rgb),
                    );
                    ffi::msg_void_u64(
                        ca,
                        b"setAlphaBlendOperation:\0",
                        blend_op_to_metal(desc.blend.op_alpha),
                    );
                }
            }

            if let Some(depth_fmt) = desc.depth_format {
                ffi::msg_void_u64(
                    pipe_desc,
                    b"setDepthAttachmentPixelFormat:\0",
                    format_to_metal(depth_fmt),
                );
            }

            ffi::msg_void_u64(pipe_desc, b"setSampleCount:\0", desc.sample_count as u64);

            let (pipeline_state, error) = ffi::msg_new_render_pipeline(self.device, pipe_desc);
            if pipeline_state.is_null() {
                let msg = error_string(error);
                return Err(QuantaError::compilation_failed(format!(
                    "render pipeline: {msg}"
                )));
            }

            // Depth/stencil state
            let ds_desc = ffi::msg_id(
                ffi::cls(b"MTLDepthStencilDescriptor\0") as ffi::Id,
                b"new\0",
            );
            if desc.depth_stencil.depth_test {
                ffi::msg_void_u64(
                    ds_desc,
                    b"setDepthCompareFunction:\0",
                    compare_to_metal(desc.depth_stencil.depth_compare),
                );
                ffi::msg_void_bool(
                    ds_desc,
                    b"setDepthWriteEnabled:\0",
                    desc.depth_stencil.depth_write,
                );
            }
            if let Some(ref front) = desc.depth_stencil.stencil_front {
                let s = ffi::msg_id(ffi::cls(b"MTLStencilDescriptor\0") as ffi::Id, b"new\0");
                ffi::msg_void_u64(
                    s,
                    b"setStencilFailureOperation:\0",
                    stencil_op_to_metal(front.fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthFailureOperation:\0",
                    stencil_op_to_metal(front.depth_fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthStencilPassOperation:\0",
                    stencil_op_to_metal(front.pass),
                );
                ffi::msg_void_u64(
                    s,
                    b"setStencilCompareFunction:\0",
                    compare_to_metal(front.compare),
                );
                ffi::msg_void_u32(s, b"setReadMask:\0", front.read_mask);
                ffi::msg_void_u32(s, b"setWriteMask:\0", front.write_mask);
                ffi::msg_void_id(ds_desc, b"setFrontFaceStencil:\0", s);
            }
            if let Some(ref back) = desc.depth_stencil.stencil_back {
                let s = ffi::msg_id(ffi::cls(b"MTLStencilDescriptor\0") as ffi::Id, b"new\0");
                ffi::msg_void_u64(
                    s,
                    b"setStencilFailureOperation:\0",
                    stencil_op_to_metal(back.fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthFailureOperation:\0",
                    stencil_op_to_metal(back.depth_fail),
                );
                ffi::msg_void_u64(
                    s,
                    b"setDepthStencilPassOperation:\0",
                    stencil_op_to_metal(back.pass),
                );
                ffi::msg_void_u64(
                    s,
                    b"setStencilCompareFunction:\0",
                    compare_to_metal(back.compare),
                );
                ffi::msg_void_u32(s, b"setReadMask:\0", back.read_mask);
                ffi::msg_void_u32(s, b"setWriteMask:\0", back.write_mask);
                ffi::msg_void_id(ds_desc, b"setBackFaceStencil:\0", s);
            }
            let ds_state = ffi::msg_id_id(
                self.device,
                b"newDepthStencilStateWithDescriptor:\0",
                ds_desc,
            );

            let handle = self.alloc_handle();
            self.render_pipelines
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, pipeline_state);
            self.depth_stencil_states
                .write()
                .map_err(|_| QuantaError::internal("lock poisoned"))?
                .insert(handle, ds_state);
            Ok(Pipeline {
                handle,
                drop_fn: None,
            })
        }
    }

    pub(crate) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let target = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::invalid_param("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        unsafe {
            // Create render pass descriptor
            let rpd = ffi::msg_id(ffi::cls(b"MTLRenderPassDescriptor\0") as ffi::Id, b"new\0");
            let color_attachments = ffi::msg_id(rpd, b"colorAttachments\0");

            if pass.color_targets.is_empty() {
                // Legacy single-target path: use pass.handle as the target
                let color_attach =
                    ffi::msg_id_u64(color_attachments, b"objectAtIndexedSubscript:\0", 0);
                ffi::msg_void_id(color_attach, b"setTexture:\0", *target);
                ffi::msg_void_u64(
                    color_attach,
                    b"setLoadAction:\0",
                    ffi::MTL_LOAD_ACTION_CLEAR,
                );
                ffi::msg_void_u64(
                    color_attach,
                    b"setStoreAction:\0",
                    ffi::MTL_STORE_ACTION_STORE,
                );
                ffi::msg_set_clear_color(color_attach, ffi::MTLClearColor::new(0.0, 0.0, 0.0, 1.0));
            } else {
                // MRT path: configure each color target with load/store ops
                for (i, ct) in pass.color_targets.iter().enumerate() {
                    let ca = ffi::msg_id_u64(
                        color_attachments,
                        b"objectAtIndexedSubscript:\0",
                        i as u64,
                    );
                    let ct_tex = textures.get(&ct.texture).ok_or_else(|| {
                        QuantaError::invalid_param("color target texture not found")
                    })?;
                    ffi::msg_void_id(ca, b"setTexture:\0", *ct_tex);

                    // Load action
                    match ct.load_op {
                        LoadOp::Clear(color) => {
                            ffi::msg_void_u64(ca, b"setLoadAction:\0", ffi::MTL_LOAD_ACTION_CLEAR);
                            ffi::msg_set_clear_color(
                                ca,
                                ffi::MTLClearColor::new(
                                    color.r as f64,
                                    color.g as f64,
                                    color.b as f64,
                                    color.a as f64,
                                ),
                            );
                        }
                        LoadOp::Load => {
                            ffi::msg_void_u64(ca, b"setLoadAction:\0", ffi::MTL_LOAD_ACTION_LOAD);
                        }
                        LoadOp::DontCare => {
                            ffi::msg_void_u64(
                                ca,
                                b"setLoadAction:\0",
                                ffi::MTL_LOAD_ACTION_DONT_CARE,
                            );
                        }
                    }

                    // Store action
                    match ct.store_op {
                        StoreOp::Store => {
                            ffi::msg_void_u64(
                                ca,
                                b"setStoreAction:\0",
                                ffi::MTL_STORE_ACTION_STORE,
                            );
                        }
                        StoreOp::DontCare => {
                            ffi::msg_void_u64(
                                ca,
                                b"setStoreAction:\0",
                                ffi::MTL_STORE_ACTION_DONT_CARE,
                            );
                        }
                        StoreOp::Resolve(resolve_handle) => {
                            ffi::msg_void_u64(
                                ca,
                                b"setStoreAction:\0",
                                ffi::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
                            );
                            if let Some(resolve_tex) = textures.get(&resolve_handle) {
                                ffi::msg_void_id(ca, b"setResolveTexture:\0", *resolve_tex);
                            }
                        }
                    }
                }
            }

            // Depth/stencil target
            if let Some(ref dt) = pass.depth_target {
                let depth_attach = ffi::msg_id(rpd, b"depthAttachment\0");
                let dt_tex = textures
                    .get(&dt.texture)
                    .ok_or_else(|| QuantaError::invalid_param("depth target texture not found"))?;
                ffi::msg_void_id(depth_attach, b"setTexture:\0", *dt_tex);

                // Depth load action
                match dt.load_op {
                    LoadOp::Clear(color) => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_CLEAR,
                        );
                        ffi::msg_void_f64(depth_attach, b"setClearDepth:\0", color.r as f64);
                    }
                    LoadOp::Load => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_LOAD,
                        );
                    }
                    LoadOp::DontCare => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_DONT_CARE,
                        );
                    }
                }

                // Depth store action
                match dt.store_op {
                    StoreOp::Store => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_STORE,
                        );
                    }
                    StoreOp::DontCare => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_DONT_CARE,
                        );
                    }
                    StoreOp::Resolve(resolve_handle) => {
                        ffi::msg_void_u64(
                            depth_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
                        );
                        if let Some(resolve_tex) = textures.get(&resolve_handle) {
                            ffi::msg_void_id(depth_attach, b"setResolveTexture:\0", *resolve_tex);
                        }
                    }
                }

                // Stencil attachment (shares the same texture for depth/stencil formats)
                let stencil_attach = ffi::msg_id(rpd, b"stencilAttachment\0");
                ffi::msg_void_id(stencil_attach, b"setTexture:\0", *dt_tex);

                match dt.stencil_load_op {
                    LoadOp::Clear(_) => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_CLEAR,
                        );
                    }
                    LoadOp::Load => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_LOAD,
                        );
                    }
                    LoadOp::DontCare => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setLoadAction:\0",
                            ffi::MTL_LOAD_ACTION_DONT_CARE,
                        );
                    }
                }

                match dt.stencil_store_op {
                    StoreOp::Store => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_STORE,
                        );
                    }
                    StoreOp::DontCare => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_DONT_CARE,
                        );
                    }
                    StoreOp::Resolve(_) => {
                        ffi::msg_void_u64(
                            stencil_attach,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_STORE,
                        );
                    }
                }
            }

            // Set visibility result buffer if any occlusion query ops are present.
            let buffers = self
                .buffers
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;
            for op in &pass.ops {
                if let RenderOp::BeginOcclusionQuery { handle, .. } = op
                    && let Some(vis_buf) = buffers.get(handle)
                {
                    ffi::msg_void_id(rpd, b"setVisibilityResultBuffer:\0", *vis_buf);
                    break;
                }
            }

            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let encoder = ffi::msg_new_render_encoder(cmd, rpd);

            let render_pipelines = self
                .render_pipelines
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;

            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(handle) => {
                        if let Some(ps) = render_pipelines.get(handle) {
                            ffi::msg_void_id(encoder, b"setRenderPipelineState:\0", *ps);
                        }
                        let ds_states = self
                            .depth_stencil_states
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        if let Some(ds) = ds_states.get(handle) {
                            ffi::msg_void_id(encoder, b"setDepthStencilState:\0", *ds);
                        }
                    }
                    RenderOp::BindVertices {
                        slot,
                        handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(handle) {
                            ffi::msg_set_buffer(
                                encoder,
                                b"setVertexBuffer:offset:atIndex:\0",
                                *buf,
                                *offset,
                                *slot as u64,
                            );
                        }
                    }
                    RenderOp::BindIndices { .. } => {
                        // Index buffer is bound at draw_indexed time in Metal
                    }
                    RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                        if let Some(buf) = buffers.get(handle) {
                            ffi::msg_set_buffer(
                                encoder,
                                b"setVertexBuffer:offset:atIndex:\0",
                                *buf,
                                0,
                                *slot as u64,
                            );
                        }
                    }
                    RenderOp::SetTexture { slot, handle } => {
                        if let Some(tex) = textures.get(handle) {
                            ffi::msg_set_texture(
                                encoder,
                                b"setFragmentTexture:atIndex:\0",
                                *tex,
                                *slot as u64,
                            );
                        }
                    }
                    RenderOp::SetValue { slot, data } => {
                        ffi::msg_set_bytes(
                            encoder,
                            b"setVertexBytes:length:atIndex:\0",
                            data.as_ptr() as *const _,
                            data.len() as u64,
                            *slot as u64,
                        );
                    }
                    RenderOp::Draw {
                        vertex_count,
                        instance_count,
                    } => {
                        if *instance_count <= 1 {
                            ffi::msg_draw_primitives(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                0,
                                *vertex_count as u64,
                            );
                        } else {
                            ffi::msg_draw_primitives_instanced(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
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
                                ffi::msg_draw_indexed(
                                    encoder,
                                    ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                    *index_count as u64,
                                    ffi::MTL_INDEX_TYPE_UINT32,
                                    *idx_buf,
                                    0,
                                );
                            } else {
                                ffi::msg_draw_indexed_instanced(
                                    encoder,
                                    ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                    *index_count as u64,
                                    ffi::MTL_INDEX_TYPE_UINT32,
                                    *idx_buf,
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
                        ffi::msg_set_scissor_rect(
                            encoder,
                            ffi::MTLScissorRect {
                                x: *x as u64,
                                y: *y as u64,
                                width: *width as u64,
                                height: *height as u64,
                            },
                        );
                    }
                    RenderOp::SetViewport {
                        x,
                        y,
                        width,
                        height,
                        min_depth,
                        max_depth,
                    } => {
                        ffi::msg_set_viewport(
                            encoder,
                            ffi::MTLViewport {
                                origin_x: *x as f64,
                                origin_y: *y as f64,
                                width: *width as f64,
                                height: *height as f64,
                                znear: *min_depth as f64,
                                zfar: *max_depth as f64,
                            },
                        );
                    }
                    RenderOp::Clear(_color) => {
                        // Clear is handled by load action on the render pass descriptor.
                    }
                    RenderOp::ClearDepth(_depth) => {
                        // Handled by render pass descriptor load action.
                    }
                    RenderOp::SetStencilRef(value) => {
                        ffi::msg_set_stencil_ref(encoder, *value);
                    }
                    RenderOp::ClearStencil(_) => {
                        // Handled by render pass descriptor load action
                    }
                    RenderOp::DrawIndirect {
                        buffer_handle,
                        offset,
                    } => {
                        if let Some(buf) = buffers.get(buffer_handle) {
                            ffi::msg_draw_primitives_indirect(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                *buf,
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
                            ffi::msg_draw_indexed_indirect(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                ffi::MTL_INDEX_TYPE_UINT32,
                                *idx_buf,
                                0,
                                *buf,
                                *offset,
                            );
                        }
                    }
                    RenderOp::DebugPush(label) => {
                        ffi::msg_push_debug_group(encoder, label);
                    }
                    RenderOp::DebugPop => {
                        ffi::msg_pop_debug_group(encoder);
                    }
                    RenderOp::SetSampler {
                        slot,
                        sampler: sdesc,
                    } => {
                        let sd =
                            ffi::msg_id(ffi::cls(b"MTLSamplerDescriptor\0") as ffi::Id, b"new\0");
                        ffi::msg_void_u64(
                            sd,
                            b"setMinFilter:\0",
                            super::filter_to_metal(sdesc.min_filter),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setMagFilter:\0",
                            super::filter_to_metal(sdesc.mag_filter),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setSAddressMode:\0",
                            super::address_to_metal(sdesc.address_u),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setTAddressMode:\0",
                            super::address_to_metal(sdesc.address_v),
                        );
                        ffi::msg_void_u64(sd, b"setMaxAnisotropy:\0", sdesc.max_anisotropy as u64);
                        let samp =
                            ffi::msg_id_id(self.device, b"newSamplerStateWithDescriptor:\0", sd);
                        ffi::msg_set_sampler(
                            encoder,
                            b"setFragmentSamplerState:atIndex:\0",
                            samp,
                            *slot as u64,
                        );
                    }
                    // Occlusion queries (M3.3)
                    RenderOp::BeginOcclusionQuery { handle, index } => {
                        if buffers.contains_key(handle) {
                            ffi::msg_set_visibility_result_mode(
                                encoder,
                                ffi::MTL_VISIBILITY_RESULT_MODE_COUNTING,
                                (*index as u64) * 8,
                            );
                        }
                    }
                    RenderOp::EndOcclusionQuery { .. } => {
                        ffi::msg_set_visibility_result_mode(
                            encoder,
                            ffi::MTL_VISIBILITY_RESULT_MODE_DISABLED,
                            0,
                        );
                    }

                    // M4+ render ops — not yet implemented.
                    RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {}
                }
            }

            ffi::msg_void(encoder, b"endEncoding\0");
            ffi::msg_void(cmd, b"commit\0");
            ffi::msg_void(cmd, b"waitUntilCompleted\0");

            Ok(Pulse {
                handle: self.alloc_handle(),
                completed: true,
            })
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

unsafe fn error_string(error: ffi::Id) -> String {
    if !error.is_null() {
        unsafe {
            let desc = ffi::msg_id(error, b"localizedDescription\0");
            let cstr = ffi::msg_utf8_string(desc);
            std::ffi::CStr::from_ptr(cstr as *const _)
                .to_string_lossy()
                .into_owned()
        }
    } else {
        "unknown error".into()
    }
}

unsafe fn get_named_function(library: ffi::Id, name: &str) -> Result<ffi::Id, QuantaError> {
    let mut name_bytes: Vec<u8> = name.bytes().collect();
    name_bytes.push(0);
    let ns_name = ffi::nsstring(&name_bytes);
    let func = unsafe { ffi::msg_get_function(library, ns_name) };
    if func.is_null() {
        return Err(QuantaError::compilation_failed(format!(
            "function '{}' not found",
            name
        )));
    }
    Ok(func)
}

/// Get a function from a library, optionally with specialization constants.
/// When `constants` is `Some`, uses `newFunctionWithName:constantValues:error:`.
/// When `None`, falls back to `newFunctionWithName:`.
unsafe fn get_function_maybe_specialized(
    library: ffi::Id,
    name: &str,
    constants: Option<ffi::Id>,
) -> Result<ffi::Id, QuantaError> {
    match constants {
        Some(fcv) => {
            let mut name_bytes: Vec<u8> = name.bytes().collect();
            name_bytes.push(0);
            let ns_name = ffi::nsstring(&name_bytes);
            let (func, error) =
                unsafe { ffi::msg_new_function_with_constants(library, ns_name, fcv) };
            if func.is_null() {
                let msg = unsafe { error_string(error) };
                return Err(QuantaError::compilation_failed(format!(
                    "function '{}' with constants: {}",
                    name, msg
                )));
            }
            Ok(func)
        }
        None => unsafe { get_named_function(library, name) },
    }
}

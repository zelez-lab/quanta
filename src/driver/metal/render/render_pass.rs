//! Render pass command recording for Metal.

use alloc::format;

use crate::{LoadOp, Pulse, QuantaError, RenderPass, StoreOp, render_pass::RenderOp};

use super::super::MetalDevice;
use super::super::ffi;

impl MetalDevice {
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
                            super::super::filter_to_metal(sdesc.min_filter),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setMagFilter:\0",
                            super::super::filter_to_metal(sdesc.mag_filter),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setSAddressMode:\0",
                            super::super::address_to_metal(sdesc.address_u),
                        );
                        ffi::msg_void_u64(
                            sd,
                            b"setTAddressMode:\0",
                            super::super::address_to_metal(sdesc.address_v),
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

                    // M4+ render ops — not yet implemented. Per Kani
                    // T418 (no silent RenderOp drops on Metal), we
                    // surface this as an explicit error rather than a
                    // no-op so a render pass that requested VRS doesn't
                    // silently fall back to uniform shading.
                    RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {
                        ffi::msg_void(encoder, b"endEncoding\0");
                        return Err(QuantaError::invalid_param(
                            "Metal render: variable-rate shading pending (Tier A 028)",
                        ));
                    }

                    // Indirect render bundle replay (steps 032 + 033).
                    // Metal: executeCommandsInBuffer:withRange: on the
                    // active render encoder. The bundle's recorded
                    // resources must be declared via useResource on
                    // the encoder so the GPU hazard tracker sees
                    // them.
                    RenderOp::ExecuteRenderBundle {
                        bundle_handle,
                        count,
                    } => {
                        let bundles = self
                            .render_bundles
                            .read()
                            .map_err(|_| QuantaError::internal("lock poisoned"))?;
                        let bundle = bundles.get(bundle_handle).ok_or_else(|| {
                            QuantaError::invalid_param("render bundle handle not found")
                        })?;
                        if *count > bundle.recorded {
                            return Err(QuantaError::invalid_param(
                                "execute_bundle count exceeds recorded length",
                            ));
                        }
                        const MTL_RESOURCE_USAGE_READ: ffi::NSUInteger = 1;
                        const MTL_RESOURCE_USAGE_WRITE: ffi::NSUInteger = 2;
                        for buf_handle in &bundle.used_buffers {
                            if let Some(buf) = buffers.get(buf_handle) {
                                ffi::msg_use_resource(
                                    encoder,
                                    *buf,
                                    MTL_RESOURCE_USAGE_READ | MTL_RESOURCE_USAGE_WRITE,
                                );
                            }
                        }
                        let range = ffi::NSRange {
                            location: 0,
                            length: *count as u64,
                        };
                        ffi::msg_execute_commands_in_buffer(encoder, bundle.icb, range);
                    }
                }
            }

            ffi::msg_void(encoder, b"endEncoding\0");

            Ok(super::super::compute::make_async_pulse(self, cmd))
        }
    }
}

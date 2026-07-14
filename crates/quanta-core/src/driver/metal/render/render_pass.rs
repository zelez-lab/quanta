//! Render pass command recording for Metal.

use std::collections::HashMap;

use alloc::format;

use crate::{LoadOp, Pulse, QuantaError, RenderPass, StoreOp, render_pass::RenderOp};

use super::super::MetalDevice;
use super::super::ffi;

/// Map a [`LoadOp`] to its Metal `setLoadAction:` value and apply it to
/// `attachment`. This covers only the genuinely-identical part of the
/// six per-attachment mapping blocks: the CLEAR/LOAD/DONT_CARE action
/// selector. Clear *values* (color / depth) are site-specific and stay
/// at the call sites — the stencil attachment sets no clear value at all.
unsafe fn set_load_action(attachment: ffi::Id, load_op: LoadOp) {
    unsafe {
        let action = match load_op {
            LoadOp::Clear(_) => ffi::MTL_LOAD_ACTION_CLEAR,
            LoadOp::Load => ffi::MTL_LOAD_ACTION_LOAD,
            LoadOp::DontCare => ffi::MTL_LOAD_ACTION_DONT_CARE,
        };
        ffi::msg_void_u64(attachment, b"setLoadAction:\0", action);
    }
}

/// Map the non-resolving [`StoreOp`] variants (`Store` / `DontCare`) to
/// their Metal `setStoreAction:` value and apply it to `attachment`.
///
/// The `Resolve` variant is deliberately excluded: color/depth resolve to
/// `MULTISAMPLE_RESOLVE` and bind a resolve texture, while stencil (which
/// cannot resolve) falls back to `STORE` and ignores the handle. Those
/// site-specific arms stay at the call sites.
unsafe fn set_store_action_non_resolve(attachment: ffi::Id, store_op: &StoreOp) {
    unsafe {
        match store_op {
            StoreOp::Store => {
                ffi::msg_void_u64(attachment, b"setStoreAction:\0", ffi::MTL_STORE_ACTION_STORE);
            }
            StoreOp::DontCare => {
                ffi::msg_void_u64(
                    attachment,
                    b"setStoreAction:\0",
                    ffi::MTL_STORE_ACTION_DONT_CARE,
                );
            }
            StoreOp::Resolve(_) => {}
        }
    }
}

/// Running encoder-side state threaded through the op-walk.
///
/// Metal has no encoder-level index-buffer bind — the buffer is passed
/// per draw call — so the replay tracks the most recent `BindIndices` as
/// the walk advances. Scanning the whole op list instead would make every
/// `DrawIndexed` use the pass's LAST index buffer.
///
/// `metal_vrs_active` records whether the pre-encoder VRS rate-map build
/// path actually ran; the in-encoder `SetShadingRate` op is a no-op only
/// when it did.
struct EncoderState {
    encoder: ffi::Id,
    bound_indices: Option<(u64, u64)>,
    metal_vrs_active: bool,
}

impl MetalDevice {
    /// Configure the color attachment(s) on the render pass descriptor.
    ///
    /// Legacy single-target path (empty `color_targets`) uses `pass.handle`
    /// as the target with a fixed CLEAR/STORE black clear; the MRT path
    /// configures each color target with its own load/store ops.
    unsafe fn configure_color_attachments(
        &self,
        rpd: ffi::Id,
        pass: &RenderPass,
        textures: &HashMap<u64, ffi::Id>,
        target: ffi::Id,
    ) -> Result<(), QuantaError> {
        unsafe {
            let color_attachments = ffi::msg_id(rpd, b"colorAttachments\0");

            if pass.color_targets.is_empty() {
                // Legacy single-target path: use pass.handle as the target
                let color_attach =
                    ffi::msg_id_u64(color_attachments, b"objectAtIndexedSubscript:\0", 0);
                ffi::msg_void_id(color_attach, b"setTexture:\0", target);
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
                    let ct_tex = textures
                        .get(&ct.texture)
                        .ok_or_else(|| QuantaError::not_found("color target texture not found"))?;
                    ffi::msg_void_id(ca, b"setTexture:\0", *ct_tex);

                    // Load action
                    set_load_action(ca, ct.load_op);
                    if let LoadOp::Clear(color) = ct.load_op {
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

                    // Store action
                    set_store_action_non_resolve(ca, &ct.store_op);
                    if let StoreOp::Resolve(resolve_handle) = ct.store_op {
                        ffi::msg_void_u64(
                            ca,
                            b"setStoreAction:\0",
                            ffi::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
                        );
                        if let Some(resolve_tex) = textures.get(&resolve_handle.0) {
                            ffi::msg_void_id(ca, b"setResolveTexture:\0", *resolve_tex);
                        }
                    }
                }
            }
            Ok(())
        }
    }

    /// Configure the depth and stencil attachments on the render pass
    /// descriptor. The stencil attachment shares the depth texture (Metal
    /// combined depth/stencil formats).
    unsafe fn configure_depth_stencil_attachments(
        &self,
        rpd: ffi::Id,
        dt: &crate::render_pass::DepthTarget,
        textures: &HashMap<u64, ffi::Id>,
    ) -> Result<(), QuantaError> {
        unsafe {
            let depth_attach = ffi::msg_id(rpd, b"depthAttachment\0");
            let dt_tex = textures
                .get(&dt.texture)
                .ok_or_else(|| QuantaError::not_found("depth target texture not found"))?;
            ffi::msg_void_id(depth_attach, b"setTexture:\0", *dt_tex);

            // Depth load action
            set_load_action(depth_attach, dt.load_op);
            if let LoadOp::Clear(color) = dt.load_op {
                ffi::msg_void_f64(depth_attach, b"setClearDepth:\0", color.r as f64);
            }

            // Depth store action
            set_store_action_non_resolve(depth_attach, &dt.store_op);
            if let StoreOp::Resolve(resolve_handle) = dt.store_op {
                ffi::msg_void_u64(
                    depth_attach,
                    b"setStoreAction:\0",
                    ffi::MTL_STORE_ACTION_MULTISAMPLE_RESOLVE,
                );
                if let Some(resolve_tex) = textures.get(&resolve_handle.0) {
                    ffi::msg_void_id(depth_attach, b"setResolveTexture:\0", *resolve_tex);
                }
            }

            // Stencil attachment (shares the same texture for depth/stencil formats)
            let stencil_attach = ffi::msg_id(rpd, b"stencilAttachment\0");
            ffi::msg_void_id(stencil_attach, b"setTexture:\0", *dt_tex);

            set_load_action(stencil_attach, dt.stencil_load_op);

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
            Ok(())
        }
    }

    /// Pre-encoder VRS lowering (step 063 slice 3). Metal's VRS path is
    /// pass-descriptor-level (MTLRasterizationRateMap) rather than per-draw,
    /// so the rate must be applied before the encoder begins. Pre-walk
    /// `pass.ops` to find the first `SetShadingRate`; if present and the
    /// device supports it, build a single-layer rate map and attach it to
    /// the descriptor. The in-encoder `SetShadingRate` op then becomes a
    /// no-op. Returns whether a rate map was built (`metal_vrs_active`).
    unsafe fn apply_shading_rate(
        &self,
        rpd: ffi::Id,
        pass: &RenderPass,
        textures: &HashMap<u64, ffi::Id>,
    ) -> Result<bool, QuantaError> {
        unsafe {
            let target_size = textures.get(&pass.handle).and_then(|t| {
                let w: u64 = ffi::msg_u64(*t, b"width\0");
                let h: u64 = ffi::msg_u64(*t, b"height\0");
                if w > 0 && h > 0 { Some((w, h)) } else { None }
            });
            let requested_rate = pass.ops.iter().find_map(|op| {
                if let RenderOp::SetShadingRate(rate) = op {
                    Some(*rate)
                } else {
                    None
                }
            });
            let mut metal_vrs_active = false;
            if let (Some(rate), Some((w, h))) = (requested_rate, target_size) {
                let supports = ffi::msg_bool_u64(
                    self.device,
                    b"supportsRasterizationRateMapWithLayerCount:\0",
                    1,
                );
                if !supports {
                    return Err(QuantaError::not_supported(
                        "Metal render encoder: device does not support MTLRasterizationRateMap",
                    ));
                }
                // Build a 1-sample-per-axis layer descriptor. Each
                // sample value is the screen-space density at that
                // axis position; uniform 1/rate gives a uniform
                // rate across the attachment.
                let layer_cls = ffi::cls(b"MTLRasterizationRateLayerDescriptor\0") as ffi::Id;
                let layer_alloc = ffi::msg_id(layer_cls, b"alloc\0");
                let one_one_one = ffi::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                };
                let layer =
                    ffi::msg_id_mtlsize(layer_alloc, b"initWithSampleCount:\0", one_one_one);
                let h_storage = ffi::msg_ptr_f32(layer, b"horizontalSampleStorage\0");
                let v_storage = ffi::msg_ptr_f32(layer, b"verticalSampleStorage\0");
                if h_storage.is_null() || v_storage.is_null() {
                    ffi::msg_void(layer, b"release\0");
                    return Err(QuantaError::internal(
                        "Metal render encoder: rasterization rate layer storage was null",
                    ));
                }
                *h_storage = 1.0_f32 / rate.x_axis() as f32;
                *v_storage = 1.0_f32 / rate.y_axis() as f32;

                let map_desc_cls = ffi::cls(b"MTLRasterizationRateMapDescriptor\0") as ffi::Id;
                let screen_size = ffi::MTLSize {
                    width: w,
                    height: h,
                    depth: 0,
                };
                // +rasterizationRateMapDescriptorWithScreenSize:layer:
                let f: unsafe extern "C" fn(ffi::Id, ffi::Sel, ffi::MTLSize, ffi::Id) -> ffi::Id =
                    core::mem::transmute(ffi::objc_msgSend as *const core::ffi::c_void);
                let map_desc = f(
                    map_desc_cls,
                    ffi::sel(b"rasterizationRateMapDescriptorWithScreenSize:layer:\0"),
                    screen_size,
                    layer,
                );
                ffi::msg_void(layer, b"release\0");
                if map_desc.is_null() {
                    return Err(QuantaError::internal(
                        "Metal render encoder: failed to build MTLRasterizationRateMapDescriptor",
                    ));
                }
                let map = ffi::msg_id_id(
                    self.device,
                    b"newRasterizationRateMapWithDescriptor:\0",
                    map_desc,
                );
                if map.is_null() {
                    return Err(QuantaError::not_supported(
                        "Metal render encoder: device declined to build rasterization rate map (rate unsupported)",
                    ));
                }
                ffi::msg_void_id(rpd, b"setRasterizationRateMap:\0", map);
                ffi::msg_void(map, b"release\0");
                metal_vrs_active = true;
            }
            Ok(metal_vrs_active)
        }
    }

    /// Bind/draw op family: pipeline, vertex/index/uniform/texture binds,
    /// value bytes, and (indexed / instanced) draws. Threads `state` so
    /// `BindIndices` reaches the following `DrawIndexed`.
    unsafe fn encode_bind_draw_op(
        &self,
        op: &RenderOp,
        state: &mut EncoderState,
        buffers: &HashMap<u64, ffi::Id>,
        textures: &HashMap<u64, ffi::Id>,
        render_pipelines: &HashMap<u64, ffi::Id>,
    ) -> Result<(), QuantaError> {
        let encoder = state.encoder;
        unsafe {
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
                            super::VERTEX_ATTRIBUTE_BUFFER_BASE + *slot as u64,
                        );
                    }
                }
                RenderOp::BindIndices { handle, offset } => {
                    // Consumed by the next DrawIndexed ops (see above).
                    state.bound_indices = Some((*handle, *offset));
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    if let Some(buf) = buffers.get(handle) {
                        // Bind to BOTH stages, matching Vulkan's
                        // descriptor visibility (VERTEX | FRAGMENT):
                        // a fragment shader reading a uniform/field
                        // sees the same slot the vertex stage does.
                        ffi::msg_set_buffer(
                            encoder,
                            b"setVertexBuffer:offset:atIndex:\0",
                            *buf,
                            0,
                            *slot as u64,
                        );
                        ffi::msg_set_buffer(
                            encoder,
                            b"setFragmentBuffer:offset:atIndex:\0",
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
                    // Both stages — see SetField above.
                    ffi::msg_set_bytes(
                        encoder,
                        b"setVertexBytes:length:atIndex:\0",
                        data.as_ptr() as *const _,
                        data.len() as u64,
                        *slot as u64,
                    );
                    ffi::msg_set_bytes(
                        encoder,
                        b"setFragmentBytes:length:atIndex:\0",
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
                    if let Some((ih, ioff)) = state.bound_indices
                        && let Some(idx_buf) = buffers.get(&ih)
                    {
                        if *instance_count <= 1 {
                            ffi::msg_draw_indexed(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                *index_count as u64,
                                ffi::MTL_INDEX_TYPE_UINT32,
                                *idx_buf,
                                ioff,
                            );
                        } else {
                            ffi::msg_draw_indexed_instanced(
                                encoder,
                                ffi::MTL_PRIMITIVE_TYPE_TRIANGLE,
                                *index_count as u64,
                                ffi::MTL_INDEX_TYPE_UINT32,
                                *idx_buf,
                                ioff,
                                *instance_count as u64,
                            );
                        }
                    }
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
                _ => unreachable!("encode_bind_draw_op called with non bind/draw op"),
            }
            Ok(())
        }
    }

    /// Fixed-function / render-state op family: scissor, viewport, clears
    /// (handled by load actions), stencil ref, debug groups, samplers.
    unsafe fn encode_state_op(&self, op: &RenderOp, state: &mut EncoderState) {
        let encoder = state.encoder;
        unsafe {
            match op {
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => {
                    // Scissor-clamp parity: `set_scissor`'s contract is
                    // that offsets are clamped to the render area on every
                    // backend. Metal already delivers that — a rectangle
                    // reaching past the drawable (including a negative
                    // offset arriving as a wrapped-in `u32`, here widened
                    // to a large `u64`) is clamped to the drawable by
                    // `setScissorRect`, no error. The Vulkan backend
                    // clamps explicitly (`clamp_scissor`) to reach the same
                    // result; nothing extra is needed here.
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
                    let sd = ffi::msg_id(ffi::cls(b"MTLSamplerDescriptor\0") as ffi::Id, b"new\0");
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
                    let samp = ffi::msg_id_id(self.device, b"newSamplerStateWithDescriptor:\0", sd);
                    ffi::msg_set_sampler(
                        encoder,
                        b"setFragmentSamplerState:atIndex:\0",
                        samp,
                        *slot as u64,
                    );
                }
                _ => unreachable!("encode_state_op called with non-state op"),
            }
        }
    }

    /// Occlusion-query op family (M3.3): begin/end visibility result mode.
    unsafe fn encode_occlusion_op(
        &self,
        op: &RenderOp,
        state: &mut EncoderState,
        buffers: &HashMap<u64, ffi::Id>,
    ) {
        let encoder = state.encoder;
        unsafe {
            match op {
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
                _ => unreachable!("encode_occlusion_op called with non-occlusion op"),
            }
        }
    }

    /// VRS in-encoder op family (step 063 slice 3). The rate was already
    /// applied to the render pass descriptor as an MTLRasterizationRateMap
    /// before the encoder began (Metal's VRS is pass-level, not per-draw),
    /// so `SetShadingRate` is a no-op — but only when that build path
    /// actually ran. On failure this ends the encoder and returns
    /// NotSupported.
    unsafe fn encode_vrs_op(
        &self,
        op: &RenderOp,
        state: &mut EncoderState,
    ) -> Result<(), QuantaError> {
        let encoder = state.encoder;
        unsafe {
            match op {
                // VRS native lowering (step 063 slice 3). The
                // rate was already applied to the render pass
                // descriptor as an MTLRasterizationRateMap above
                // (Metal's VRS is pass-level, not per-draw), so
                // the in-encoder op is a no-op — but only when
                // the descriptor-build path actually ran. If
                // metal_vrs_active is false, the rate map could
                // not be built; surface that as NotSupported.
                RenderOp::SetShadingRate(_) => {
                    if !state.metal_vrs_active {
                        ffi::msg_void(encoder, b"endEncoding\0");
                        return Err(QuantaError::not_supported(
                            "Metal render encoder: VRS rate not applied (no rate map built)",
                        ));
                    }
                }
                RenderOp::SetShadingRateImage { .. } => {
                    ffi::msg_void(encoder, b"endEncoding\0");
                    return Err(QuantaError::not_supported(
                        "Metal render encoder: shading-rate-image (texel-driven VRS) deferred",
                    ));
                }
                _ => unreachable!("encode_vrs_op called with non-VRS op"),
            }
            Ok(())
        }
    }

    /// Indirect render-bundle replay (steps 032 + 033). Metal:
    /// `executeCommandsInBuffer:withRange:` on the active render encoder.
    /// The bundle's recorded resources must be declared via `useResource`
    /// on the encoder so the GPU hazard tracker sees them.
    unsafe fn encode_bundle_op(
        &self,
        op: &RenderOp,
        state: &mut EncoderState,
        buffers: &HashMap<u64, ffi::Id>,
    ) -> Result<(), QuantaError> {
        let encoder = state.encoder;
        unsafe {
            match op {
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
                    let bundle = bundles
                        .get(bundle_handle)
                        .ok_or_else(|| QuantaError::not_found("render bundle handle not found"))?;
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
                _ => unreachable!("encode_bundle_op called with non-bundle op"),
            }
            Ok(())
        }
    }

    pub(crate) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let textures = self
            .textures
            .read()
            .map_err(|_| QuantaError::internal("lock poisoned"))?;
        let target = textures.get(&pass.handle).ok_or_else(|| {
            QuantaError::not_found("render target not found")
                .with_context(&format!("render_end: target handle {}", pass.handle))
        })?;

        unsafe {
            // Create render pass descriptor
            let rpd = ffi::msg_id(ffi::cls(b"MTLRenderPassDescriptor\0") as ffi::Id, b"new\0");

            self.configure_color_attachments(rpd, &pass, &textures, *target)?;

            // Depth/stencil target
            if let Some(ref dt) = pass.depth_target {
                self.configure_depth_stencil_attachments(rpd, dt, &textures)?;
            }

            // Set visibility result buffer if any occlusion query ops are present.
            let buffers = self
                .buffers
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;

            // Fail loudly on any dead handle BEFORE encoding starts —
            // a silently skipped bind renders wrong (classic cause: a
            // Field dropped before pulse()).
            {
                use crate::render_pass::HandleKind;
                let pipelines = self
                    .render_pipelines
                    .read()
                    .map_err(|_| QuantaError::internal("lock poisoned"))?;
                pass.validate_handles(|kind, h| match kind {
                    // Metal's occlusion queries are visibility buffers.
                    HandleKind::Buffer | HandleKind::OcclusionQuery => buffers.contains_key(&h),
                    HandleKind::Texture => textures.contains_key(&h),
                    HandleKind::Pipeline => pipelines.contains_key(&h),
                })?;
                // Also fail loudly on a pipeline/target shape mismatch —
                // a phantom or mis-typed attachment that Metal would
                // accept silently and then drop draws for.
                pass.validate_pass_shape()?;
            }

            for op in &pass.ops {
                if let RenderOp::BeginOcclusionQuery { handle, .. } = op
                    && let Some(vis_buf) = buffers.get(handle)
                {
                    ffi::msg_void_id(rpd, b"setVisibilityResultBuffer:\0", *vis_buf);
                    break;
                }
            }

            // VRS native lowering (step 063 slice 3). Metal's VRS
            // path is pass-descriptor-level (MTLRasterizationRateMap)
            // rather than per-draw, so the rate must be applied
            // before the encoder begins. Pre-walk pass.ops to find
            // the first SetShadingRate; if present and the device
            // supports it, build a single-layer rate map and attach
            // it to the descriptor. The in-encoder SetShadingRate
            // op then becomes a no-op below.
            let metal_vrs_active = self.apply_shading_rate(rpd, &pass, &textures)?;

            let cmd = ffi::msg_id(self.queue, b"commandBuffer\0");
            let encoder = ffi::msg_new_render_encoder(cmd, rpd);

            let render_pipelines = self
                .render_pipelines
                .read()
                .map_err(|_| QuantaError::internal("lock poisoned"))?;

            // Metal has no encoder-level index-buffer bind — the buffer is
            // passed per draw call — so the replay tracks the most recent
            // BindIndices as the walk advances. Scanning the whole op list
            // instead would make every DrawIndexed use the pass's LAST
            // index buffer.
            let mut state = EncoderState {
                encoder,
                bound_indices: None,
                metal_vrs_active,
            };

            for op in &pass.ops {
                match op {
                    RenderOp::SetPipeline(_)
                    | RenderOp::BindVertices { .. }
                    | RenderOp::BindIndices { .. }
                    | RenderOp::SetField { .. }
                    | RenderOp::SetUniform { .. }
                    | RenderOp::SetTexture { .. }
                    | RenderOp::SetValue { .. }
                    | RenderOp::Draw { .. }
                    | RenderOp::DrawIndexed { .. }
                    | RenderOp::DrawIndirect { .. }
                    | RenderOp::DrawIndexedIndirect { .. } => {
                        self.encode_bind_draw_op(
                            op,
                            &mut state,
                            &buffers,
                            &textures,
                            &render_pipelines,
                        )?;
                    }
                    RenderOp::SetScissor { .. }
                    | RenderOp::SetViewport { .. }
                    | RenderOp::Clear(_)
                    | RenderOp::ClearDepth(_)
                    | RenderOp::SetStencilRef(_)
                    | RenderOp::ClearStencil(_)
                    | RenderOp::DebugPush(_)
                    | RenderOp::DebugPop
                    | RenderOp::SetSampler { .. } => {
                        self.encode_state_op(op, &mut state);
                    }
                    RenderOp::BeginOcclusionQuery { .. } | RenderOp::EndOcclusionQuery { .. } => {
                        self.encode_occlusion_op(op, &mut state, &buffers);
                    }
                    RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {
                        self.encode_vrs_op(op, &mut state)?;
                    }
                    RenderOp::ExecuteRenderBundle { .. } => {
                        self.encode_bundle_op(op, &mut state, &buffers)?;
                    }
                }
            }

            ffi::msg_void(encoder, b"endEncoding\0");

            Ok(super::super::device::make_async_pulse(self, cmd))
        }
    }
}

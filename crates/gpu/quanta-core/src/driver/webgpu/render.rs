//! Render pipeline and render pass execution for the WebGPU driver.
//!
//! Render-gated (step 085): pipeline creation, the render-pass op
//! walk (`render_end`), bind-group assembly, and the enum→ABI-code
//! translations used only by the render path. `WebgpuDevice`'s render
//! trait methods live here; the compute and core plumbing stay in the
//! parent module.

use alloc::vec::Vec;

use crate::{Pipeline, Pulse, QuantaError, RenderPass, Texture};

use super::state;
use super::{
    WebgpuDevice, address_code, compare_op_code, ffi, filter_code, format_code, make_pulse,
};

// ── Enum → code translations (render-only; Rust API → ABI codes) ─────────────

#[cfg(feature = "render")]
fn attribute_format_code(f: crate::pipeline::AttributeFormat) -> u32 {
    use crate::pipeline::AttributeFormat as A;
    match f {
        A::Float => ffi::attribute_format::FLOAT,
        A::Float2 => ffi::attribute_format::FLOAT2,
        A::Float3 => ffi::attribute_format::FLOAT3,
        A::Float4 => ffi::attribute_format::FLOAT4,
        A::Int => ffi::attribute_format::SINT,
        A::Int2 => ffi::attribute_format::SINT2,
        A::Int3 => ffi::attribute_format::SINT3,
        A::Int4 => ffi::attribute_format::SINT4,
        A::UInt => ffi::attribute_format::UINT,
        A::UInt2 => ffi::attribute_format::UINT2,
        A::UInt3 => ffi::attribute_format::UINT3,
        A::UInt4 => ffi::attribute_format::UINT4,
        A::UByte4Norm => ffi::attribute_format::UNORM8X4,
    }
}

#[cfg(feature = "render")]
fn topology_code(p: crate::pipeline::Primitive) -> u32 {
    use crate::pipeline::Primitive as P;
    match p {
        P::Point => ffi::topology::POINT,
        P::Line => ffi::topology::LINE,
        P::LineStrip => ffi::topology::LINE_STRIP,
        P::Triangle => ffi::topology::TRIANGLE,
        P::TriangleStrip => ffi::topology::TRIANGLE_STRIP,
    }
}

#[cfg(feature = "render")]
fn cull_mode_code(c: crate::pipeline::CullMode) -> u32 {
    use crate::pipeline::CullMode as C;
    match c {
        C::None => ffi::cull_mode::NONE,
        C::Front => ffi::cull_mode::FRONT,
        C::Back => ffi::cull_mode::BACK,
    }
}

#[cfg(feature = "render")]
fn blend_factor_code(f: crate::pipeline::BlendFactor) -> u32 {
    use crate::pipeline::BlendFactor as F;
    match f {
        F::Zero => ffi::blend_factor::ZERO,
        F::One => ffi::blend_factor::ONE,
        F::SrcAlpha => ffi::blend_factor::SRC_ALPHA,
        F::OneMinusSrcAlpha => ffi::blend_factor::ONE_MINUS_SRC_ALPHA,
        F::DstAlpha => ffi::blend_factor::DST_ALPHA,
        F::OneMinusDstAlpha => ffi::blend_factor::ONE_MINUS_DST_ALPHA,
        F::SrcColor => ffi::blend_factor::SRC_COLOR,
        F::OneMinusSrcColor => ffi::blend_factor::ONE_MINUS_SRC_COLOR,
        F::DstColor => ffi::blend_factor::DST_COLOR,
        F::OneMinusDstColor => ffi::blend_factor::ONE_MINUS_DST_COLOR,
    }
}

#[cfg(feature = "render")]
fn blend_op_code(o: crate::pipeline::BlendOp) -> u32 {
    use crate::pipeline::BlendOp as O;
    match o {
        O::Add => ffi::blend_op::ADD,
        O::Subtract => ffi::blend_op::SUBTRACT,
        O::ReverseSubtract => ffi::blend_op::REVERSE_SUBTRACT,
        O::Min => ffi::blend_op::MIN,
        O::Max => ffi::blend_op::MAX,
    }
}

#[cfg(feature = "render")]
fn step_mode_code(s: crate::pipeline::StepMode) -> u32 {
    match s {
        crate::pipeline::StepMode::Vertex => ffi::step_mode::VERTEX,
        crate::pipeline::StepMode::Instance => ffi::step_mode::INSTANCE,
    }
}

// ── GpuDevice render impl ────────────────────────────────────────────────────

impl WebgpuDevice {
    pub(super) fn pipeline_create_impl(
        &self,
        desc: &crate::PipelineDesc,
    ) -> Result<Pipeline, QuantaError> {
        // Step 063 slice 11 — WebGPU spec doesn't include
        // tessellation, mesh shaders, or conservative rasterization.
        // Surface NotSupported up-front rather than silently dropping
        // the request when the user sets these on PipelineDesc
        // (matches Kani T418 / T419 no-silent-drops contract,
        // symmetric to slices 5 and 8/9).
        if desc.tessellation.is_some() {
            return Err(Self::not_supported(
                "WebGPU render pipelines: tessellation is not in the WebGPU spec",
            ));
        }
        if desc.mesh_shader.is_some() {
            return Err(Self::not_supported(
                "WebGPU render pipelines: mesh shaders are not in the WebGPU spec",
            ));
        }
        if desc.conservative_rasterization {
            return Err(Self::not_supported(
                "WebGPU render pipelines: conservative rasterization is not in the WebGPU spec",
            ));
        }
        let device = self.dev()?;

        let combined = desc.shader.combined();
        let (vs_src, fs_src) = desc.shader.stage_wgsl_bytes().ok_or_else(|| {
            Self::err("pipeline shader binaries carry no WGSL payload for WebGPU")
        })?;
        let vs_text = core::str::from_utf8(vs_src)
            .map_err(|_| Self::err("vertex shader is not valid UTF-8 WGSL"))?;
        let fs_text = core::str::from_utf8(fs_src)
            .map_err(|_| Self::err("fragment shader is not valid UTF-8 WGSL"))?;

        let vs_module =
            unsafe { ffi::quanta_create_shader_module(device, vs_text.as_ptr(), vs_text.len()) };
        let fs_module = if combined.is_some() || vs_text == fs_text {
            // Reuse the same module handle — JS side doesn't need a
            // distinct copy. Keep the lifecycle simple by allocating
            // a parallel handle on the JS side instead of aliasing in
            // the table; cheap call but clearer ownership.
            unsafe { ffi::quanta_create_shader_module(device, vs_text.as_ptr(), vs_text.len()) }
        } else {
            unsafe { ffi::quanta_create_shader_module(device, fs_text.as_ptr(), fs_text.len()) }
        };

        let rp_desc = unsafe { ffi::quanta_rp_desc_create() };

        unsafe {
            ffi::quanta_rp_desc_set_vertex(
                rp_desc,
                vs_module,
                desc.vertex_entry.as_ptr(),
                desc.vertex_entry.len(),
            );
        }

        for (buf_index, layout) in desc.vertex_layouts.iter().enumerate() {
            unsafe {
                ffi::quanta_rp_desc_add_vertex_buffer(
                    rp_desc,
                    layout.stride,
                    step_mode_code(layout.step),
                );
            }
            for a in &layout.attributes {
                unsafe {
                    ffi::quanta_rp_desc_add_vertex_attribute(
                        rp_desc,
                        buf_index as u32,
                        attribute_format_code(a.format),
                        a.offset,
                        a.location,
                    );
                }
            }
        }

        for (i, fmt) in desc.color_formats.iter().enumerate() {
            let blend_state = desc
                .blend_states
                .get(i)
                .copied()
                .or_else(|| desc.blend_states.last().copied())
                .unwrap_or(desc.blend);
            unsafe {
                ffi::quanta_rp_desc_add_color_target(
                    rp_desc,
                    format_code(*fmt)?,
                    if blend_state.enabled { 1 } else { 0 },
                    blend_factor_code(blend_state.src_rgb),
                    blend_factor_code(blend_state.dst_rgb),
                    blend_op_code(blend_state.op_rgb),
                    blend_factor_code(blend_state.src_alpha),
                    blend_factor_code(blend_state.dst_alpha),
                    blend_op_code(blend_state.op_alpha),
                );
            }
        }

        unsafe {
            ffi::quanta_rp_desc_set_fragment(
                rp_desc,
                fs_module,
                desc.fragment_entry.as_ptr(),
                desc.fragment_entry.len(),
            );
        }
        unsafe {
            ffi::quanta_rp_desc_set_primitive(
                rp_desc,
                topology_code(desc.primitive),
                cull_mode_code(desc.cull_mode),
            );
            ffi::quanta_rp_desc_set_multisample(rp_desc, desc.sample_count.max(1));
        }
        if let Some(depth_fmt) = desc.depth_format {
            unsafe {
                ffi::quanta_rp_desc_set_depth_stencil(
                    rp_desc,
                    format_code(depth_fmt)?,
                    if desc.depth_stencil.depth_write { 1 } else { 0 },
                    compare_op_code(desc.depth_stencil.depth_compare),
                );
            }
        }

        let pipeline = unsafe { ffi::quanta_create_render_pipeline(device, rp_desc) };
        let layout = unsafe { ffi::quanta_render_pipeline_get_bind_group_layout(pipeline, 0) };

        // Modules are referenced by the pipeline; we no longer need
        // the JS-side handles.
        unsafe {
            ffi::quanta_release(vs_module);
            ffi::quanta_release(fs_module);
        }

        let handle = self.state.alloc_handle();
        self.state
            .pipelines
            .0
            .borrow_mut()
            .insert(handle, state::PipelineEntry { pipeline, layout });

        Ok(Pipeline::from_desc(handle, desc))
    }

    pub(super) fn pipeline_destroy_impl(&self, handle: u64) -> Result<(), QuantaError> {
        if let Some(entry) = self.state.pipelines.0.borrow_mut().remove(&handle) {
            // GPURenderPipeline / GPUBindGroupLayout have no destroy();
            // releasing the JS handles lets the GC collect them.
            unsafe {
                ffi::quanta_release(entry.layout);
                ffi::quanta_release(entry.pipeline);
            }
        }
        Ok(())
    }

    pub(super) fn render_begin_impl(&self, target: &Texture) -> Result<RenderPass, QuantaError> {
        Ok(RenderPass {
            handle: target.handle,
            ops: Vec::new(),
            value_data: Vec::new(),
            color_targets: alloc::vec![crate::render_pass::ColorTarget {
                texture: target.handle,
                format: target.format,
                samples: target.sample_count,
                load_op: crate::LoadOp::Clear(crate::Color::CLEAR),
                store_op: crate::StoreOp::Store,
            }],
            depth_target: None,
            primary_format: Some(target.format),
            primary_samples: Some(target.sample_count),
            pipeline_shapes: Vec::new(),
        })
    }

    pub(super) fn render_end_impl(&self, pass: RenderPass) -> Result<Pulse, QuantaError> {
        let device = self.dev()?;

        // Fail loudly on a pipeline/target shape mismatch BEFORE any
        // encoding — same backend-agnostic check Metal and Vulkan run,
        // so a mismatched draw errors identically on every backend
        // instead of being dropped by the browser's own validation
        // (which surfaces only in the JS console, invisible to Rust).
        pass.validate_pass_shape()?;

        // Declared color targets beyond the primary are not wired on
        // this backend: the encoder below binds `pass.handle` as the
        // sole CLEAR/STORE color attachment and never reads
        // `pass.color_targets`. A pass that declares anything else —
        // MRT, a builder-managed MSAA intermediate (`.msaa(n)`), or a
        // subpass resolve (`StoreOp::Resolve`) — would be silently
        // misdrawn into the wrong texture; fail loudly instead.
        // (WebGPU has native `resolveTarget` machinery; wiring it is a
        // documented deferral.)
        if pass.color_targets.len() > 1
            || pass.color_targets.first().is_some_and(|ct| {
                ct.texture != pass.handle || !matches!(ct.store_op, crate::StoreOp::Store)
            })
        {
            return Err(QuantaError::not_supported(
                "WebGPU render passes bind only the primary target: MRT, \
                 builder-managed MSAA (.msaa) and subpass resolve are not \
                 wired on this backend",
            ));
        }

        let textures = self.state.textures.0.borrow();
        let target = textures
            .get(&pass.handle)
            .ok_or_else(|| Self::err("unknown render target"))?;

        // Pre-walk: find the clear color and (if attached) the depth
        // clear. Both end up on the rpass descriptor, not on encoder
        // calls — WebGPU §16 lets the user specify them once at
        // beginRenderPass time.
        let mut clear_rgba = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        let mut clear_depth: Option<f32> = None;
        for op in &pass.ops {
            match op {
                crate::render_pass::RenderOp::Clear(color) => {
                    clear_rgba = (color.r, color.g, color.b, color.a);
                }
                crate::render_pass::RenderOp::ClearDepth(d) => {
                    clear_depth = Some(*d);
                }
                _ => {}
            }
        }

        let rpass_desc = unsafe { ffi::quanta_rpass_desc_create() };
        unsafe {
            ffi::quanta_rpass_desc_add_color_attachment(
                rpass_desc,
                target.view,
                ffi::load_op::CLEAR,
                ffi::store_op::STORE,
                clear_rgba.0,
                clear_rgba.1,
                clear_rgba.2,
                clear_rgba.3,
            );
        }
        // If the API caller attached a depth target, wire its view onto
        // the rpass desc. WebGPU only takes the clear value alongside
        // the attachment; ClearDepth carries the value from the op
        // stream into this attachment, so the depth target itself is
        // the source of truth for "which texture", and ClearDepth is
        // the source of truth for "what value."
        if let Some(depth) = &pass.depth_target {
            let depth_tex = textures
                .get(&depth.texture)
                .ok_or_else(|| Self::err("unknown depth target"))?;
            unsafe {
                ffi::quanta_rpass_desc_set_depth_attachment(
                    rpass_desc,
                    depth_tex.view,
                    if clear_depth.is_some() {
                        ffi::load_op::CLEAR
                    } else {
                        ffi::load_op::LOAD
                    },
                    ffi::store_op::STORE,
                    clear_depth.unwrap_or(1.0),
                );
            }
        }

        // Occlusion-query attachment: pre-walk pass.ops for the
        // first BeginOcclusionQuery, look up its query set in the
        // device registry, and bind it to the render pass desc
        // BEFORE beginRenderPass — WebGPU requires the
        // occlusionQuerySet to be set at descriptor time.
        let occlusion_qs_js: Option<u32> = pass
            .ops
            .iter()
            .find_map(|op| {
                if let crate::render_pass::RenderOp::BeginOcclusionQuery { handle, .. } = op {
                    Some(*handle)
                } else {
                    None
                }
            })
            .and_then(|h| self.state.query_sets.0.borrow().get(&h).map(|(js, _)| *js));
        if let Some(qs_js) = occlusion_qs_js {
            unsafe { ffi::quanta_rpass_desc_set_occlusion_query_set(rpass_desc, qs_js) };
        }

        let encoder = unsafe { ffi::quanta_create_command_encoder(device) };
        let rp = unsafe { ffi::quanta_encoder_begin_render_pass(encoder, rpass_desc) };

        let pipelines = self.state.pipelines.0.borrow();
        let buffers = self.state.buffers.0.borrow();
        let mut current_pipeline: Option<&state::PipelineEntry> = None;

        /// One slot of a pending bind group. The JS-side resource is
        /// either a buffer (long-lived; not owned by `render_end`), a
        /// texture view (long-lived; lookup via `state.textures`), a
        /// sampler (created here from a `SamplerDesc`; owned), or a
        /// freshly-allocated uniform buffer holding push-constant
        /// bytes (WebGPU has no push constants — the SetValue
        /// fallback below allocates a per-call buffer; owned).
        enum BindEntry {
            Buffer(u32),
            TextureView(u32),
            Sampler(u32),
            OwnedBuffer(u32),
        }
        let mut bind_entries: alloc::collections::BTreeMap<u32, BindEntry> =
            alloc::collections::BTreeMap::new();

        // Helper: flush pending bind entries into a real bind group
        // and bind it. Hoisted out of the match for the two draw
        // variants below.
        let flush_bg = |bind_entries: &mut alloc::collections::BTreeMap<u32, BindEntry>,
                        cur: Option<&state::PipelineEntry>|
         -> Option<u32> {
            if bind_entries.is_empty() {
                return None;
            }
            let p = cur?;
            let bg_desc = unsafe { ffi::quanta_bg_desc_create(p.layout) };
            for (slot, entry) in bind_entries.iter() {
                match entry {
                    BindEntry::Buffer(h) | BindEntry::OwnedBuffer(h) => unsafe {
                        ffi::quanta_bg_desc_add_buffer(bg_desc, *slot, *h)
                    },
                    BindEntry::TextureView(h) => unsafe {
                        ffi::quanta_bg_desc_add_texture_view(bg_desc, *slot, *h)
                    },
                    BindEntry::Sampler(h) => unsafe {
                        ffi::quanta_bg_desc_add_sampler(bg_desc, *slot, *h)
                    },
                }
            }
            let bg = unsafe { ffi::quanta_create_bind_group(device, bg_desc) };
            unsafe { ffi::quanta_render_pass_set_bind_group(rp, 0, bg) };
            bind_entries.clear();
            Some(bg)
        };

        let mut owned_bgs: Vec<u32> = Vec::new();
        // Resources allocated within this pass that must be released
        // after submit: samplers minted from `SetSampler` and uniform
        // buffers allocated as push-constant fallbacks for `SetValue`.
        let mut owned_samplers: Vec<u32> = Vec::new();
        let mut owned_buffers: Vec<u32> = Vec::new();

        for op in &pass.ops {
            use crate::render_pass::RenderOp;
            match op {
                RenderOp::SetPipeline(handle) => {
                    let entry = pipelines
                        .get(handle)
                        .ok_or_else(|| Self::err("unknown pipeline"))?;
                    unsafe { ffi::quanta_render_pass_set_pipeline(rp, entry.pipeline) };
                    current_pipeline = Some(entry);
                }
                RenderOp::BindVertices {
                    slot,
                    handle,
                    offset,
                } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("vbuf"))?;
                    unsafe {
                        ffi::quanta_render_pass_set_vertex_buffer(rp, *slot, buf, *offset as f64);
                    }
                }
                RenderOp::BindIndices { handle, offset } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("ibuf"))?;
                    unsafe {
                        ffi::quanta_render_pass_set_index_buffer(
                            rp,
                            buf,
                            ffi::index_format::UINT32,
                            *offset as f64,
                        );
                    }
                }
                RenderOp::SetField { slot, handle } | RenderOp::SetUniform { slot, handle } => {
                    let &buf = buffers.get(handle).ok_or_else(|| Self::err("ubuf"))?;
                    bind_entries.insert(*slot, BindEntry::Buffer(buf));
                }
                RenderOp::Clear(_) | RenderOp::ClearDepth(_) => {
                    // Both clear values are picked up in the pre-walk
                    // above and applied as `clearValue` on the rpass
                    // descriptor; nothing to emit per-op.
                }
                RenderOp::Draw {
                    vertex_count,
                    instance_count,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    unsafe {
                        ffi::quanta_render_pass_draw(rp, *vertex_count, *instance_count);
                    }
                }
                RenderOp::DrawIndexed {
                    index_count,
                    instance_count,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    unsafe {
                        ffi::quanta_render_pass_draw_indexed(rp, *index_count, *instance_count);
                    }
                }
                RenderOp::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => unsafe {
                    ffi::quanta_render_pass_set_viewport(
                        rp, *x, *y, *width, *height, *min_depth, *max_depth,
                    );
                },
                RenderOp::SetScissor {
                    x,
                    y,
                    width,
                    height,
                } => unsafe {
                    ffi::quanta_render_pass_set_scissor(rp, *x, *y, *width, *height);
                },
                // ── Step C wiring ───────────────────────────────────────────
                RenderOp::SetTexture { slot, handle } => {
                    let view = textures
                        .get(handle)
                        .ok_or_else(|| Self::err("unknown texture for SetTexture"))?
                        .view;
                    bind_entries.insert(*slot, BindEntry::TextureView(view));
                }
                RenderOp::SetSampler { slot, sampler } => {
                    let s = unsafe {
                        ffi::quanta_create_sampler(
                            device,
                            filter_code(sampler.mag_filter),
                            filter_code(sampler.min_filter),
                            filter_code(sampler.mip_filter),
                            address_code(sampler.address_u),
                            address_code(sampler.address_v),
                            // WebGPU samplers are 3D-addressable; the
                            // public `SamplerDesc` only carries U/V, so
                            // mirror V into W (same as Vulkan/Metal
                            // drivers do for 2D textures).
                            address_code(sampler.address_v),
                            sampler.max_anisotropy as u32,
                            // `compare::UNSET` is the JS-side sentinel
                            // for "no compare function" — the JS layer
                            // omits the field entirely when it sees
                            // this code.
                            sampler
                                .compare
                                .map(compare_op_code)
                                .unwrap_or(ffi::compare::UNSET),
                        )
                    };
                    bind_entries.insert(*slot, BindEntry::Sampler(s));
                    owned_samplers.push(s);
                }
                RenderOp::SetValue { slot, offset, len } => {
                    // WebGPU has no push constants. Fallback: allocate
                    // a one-shot uniform buffer, write the bytes, bind
                    // it as if it were a `SetUniform`. The caller
                    // pays per-call allocation cost; semantics match
                    // Metal's `setVertexBytes` and Vulkan's
                    // `vkCmdPushConstants`. The buffer is released
                    // after submit, below.
                    let data = pass.value_bytes(*offset, *len);
                    let size = data.len() as f64;
                    let buf = unsafe {
                        ffi::quanta_create_buffer(
                            device,
                            size,
                            ffi::buffer_usage::UNIFORM | ffi::buffer_usage::COPY_DST,
                        )
                    };
                    unsafe {
                        ffi::quanta_write_buffer(device, buf, 0.0, data.as_ptr(), data.len());
                    }
                    bind_entries.insert(*slot, BindEntry::OwnedBuffer(buf));
                    owned_buffers.push(buf);
                }
                RenderOp::SetStencilRef(reference) => unsafe {
                    ffi::quanta_render_pass_set_stencil_reference(rp, *reference);
                },
                // Stencil clear value — like color/depth, the WebGPU
                // pass descriptor takes it once at begin time. We
                // currently always store DISCARD on the stencil aspect
                // (no consumer wires stencil load yet), so absorbing
                // the value here is a no-op until depth-target growth.
                RenderOp::ClearStencil(_) => {}
                // Variants below are not in the 050 baseline. Per Kani
                // theorem T417, the rule is **every RenderOp is either
                // wired or explicitly rejected** — no silent drops.
                RenderOp::DebugPush { .. } | RenderOp::DebugPop => {
                    // Debug labels are advisory; safe to skip on WebGPU.
                }
                RenderOp::DrawIndirect {
                    buffer_handle,
                    offset,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    let &buf = buffers
                        .get(buffer_handle)
                        .ok_or_else(|| Self::err("draw_indirect buffer handle not found"))?;
                    unsafe {
                        ffi::quanta_render_pass_draw_indirect(rp, buf, *offset as f64);
                    }
                }
                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    offset,
                    index_handle,
                } => {
                    if let Some(bg) = flush_bg(&mut bind_entries, current_pipeline) {
                        owned_bgs.push(bg);
                    }
                    let &idx_buf = buffers.get(index_handle).ok_or_else(|| {
                        Self::err("draw_indexed_indirect index buffer handle not found")
                    })?;
                    unsafe {
                        ffi::quanta_render_pass_set_index_buffer(
                            rp,
                            idx_buf,
                            ffi::index_format::UINT32,
                            0.0,
                        );
                    }
                    let &buf = buffers.get(buffer_handle).ok_or_else(|| {
                        Self::err("draw_indexed_indirect indirect buffer handle not found")
                    })?;
                    unsafe {
                        ffi::quanta_render_pass_draw_indexed_indirect(rp, buf, *offset as f64);
                    }
                }
                RenderOp::ExecuteRenderBundle {
                    bundle_handle,
                    count,
                } => {
                    let bundles = self.state.render_bundles.0.borrow();
                    let bundle = bundles
                        .get(bundle_handle)
                        .ok_or_else(|| Self::err("render bundle handle not found in execute"))?;
                    if *count > bundle.draws.len() as u32 {
                        unsafe { ffi::quanta_render_pass_end(rp) };
                        return Err(Self::err("execute_bundle count exceeds recorded length"));
                    }
                    if *count == 0 {
                        continue;
                    }
                    // Build a fresh GPURenderBundleEncoder against the
                    // active render target's format, replay snapshots,
                    // finish, and pass.executeBundles.
                    let target_format = format_code(target.format)?;
                    let depth_format = if let Some(depth) = &pass.depth_target {
                        let depth_tex = textures
                            .get(&depth.texture)
                            .ok_or_else(|| Self::err("unknown depth target in execute_bundle"))?;
                        format_code(depth_tex.format)?
                    } else {
                        0
                    };
                    let bundle_enc = unsafe {
                        ffi::quanta_create_render_bundle_encoder(
                            device,
                            target_format,
                            depth_format,
                            1,
                        )
                    };
                    for draw in bundle.draws.iter().take(*count as usize) {
                        if let Some(pe) = pipelines.get(&draw.pipeline_handle) {
                            unsafe {
                                ffi::quanta_render_bundle_set_pipeline(bundle_enc, pe.pipeline);
                                ffi::quanta_render_bundle_draw(
                                    bundle_enc,
                                    draw.vertex_count,
                                    draw.instance_count.max(1),
                                );
                            }
                        }
                    }
                    let bundle_h = unsafe { ffi::quanta_render_bundle_finish(bundle_enc) };
                    let bundles_arr = [bundle_h];
                    unsafe {
                        ffi::quanta_render_pass_execute_bundles(rp, bundles_arr.as_ptr(), 1);
                        ffi::quanta_release(bundle_h);
                    }
                }
                RenderOp::BeginOcclusionQuery { index, .. } => {
                    unsafe { ffi::quanta_render_pass_begin_occlusion_query(rp, *index) };
                }
                RenderOp::EndOcclusionQuery { .. } => {
                    unsafe { ffi::quanta_render_pass_end_occlusion_query(rp) };
                }
                RenderOp::SetShadingRate(_) | RenderOp::SetShadingRateImage { .. } => {
                    unsafe { ffi::quanta_render_pass_end(rp) };
                    return Err(Self::not_supported(
                        "WebGPU render encoder: variable-rate shading is not in the WebGPU spec",
                    ));
                }
            }
        }

        unsafe { ffi::quanta_render_pass_end(rp) };
        let cmd = unsafe { ffi::quanta_encoder_finish(encoder) };
        unsafe { ffi::quanta_queue_submit(device, cmd) };
        for bg in owned_bgs {
            unsafe { ffi::quanta_release(bg) };
        }
        for s in owned_samplers {
            unsafe { ffi::quanta_release(s) };
        }
        // SetValue's per-call uniform buffers go through
        // `quanta_destroy_buffer` (not `quanta_release`) because the
        // JS side allocates a real `GPUBuffer.destroy()`-bearing
        // resource. The two FFI routes are not interchangeable.
        for b in owned_buffers {
            unsafe { ffi::quanta_destroy_buffer(b) };
        }
        Ok(make_pulse())
    }
}

use alloc::vec::Vec;

use crate::{
    Color, Field, LoadOp, OcclusionQuery, Pipeline, SamplerDesc, ShadingRate, StoreOp, Texture,
};

/// A color attachment target with load/store operations.
///
/// Construct with [`ColorTarget::new`] from a typed [`Texture`] —
/// the attachment handle is derived from the texture, never passed as
/// a raw `u64`.
pub struct ColorTarget {
    /// Driver handle of the attachment texture (derived from the
    /// `Texture` passed to [`ColorTarget::new`]).
    pub(crate) texture: u64,
    /// Pixel format of the attachment texture (captured from the
    /// `Texture` at construction). Retained so the render pass can
    /// check it against the pipeline's `color_formats[i]` at encode
    /// time.
    pub(crate) format: crate::Format,
    /// What to do with existing contents at pass start.
    pub load_op: LoadOp,
    /// What to do with results at pass end.
    pub store_op: StoreOp,
}

impl ColorTarget {
    /// A color attachment over `texture`, defaulting to
    /// `LoadOp::Clear(Color::BLACK)` + `StoreOp::Store`.
    pub fn new(texture: &Texture) -> Self {
        Self {
            texture: texture.handle(),
            format: texture.format(),
            load_op: LoadOp::Clear(Color::BLACK),
            store_op: StoreOp::Store,
        }
    }

    /// Set the load operation.
    pub fn with_load_op(mut self, load_op: LoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Set the store operation.
    pub fn with_store_op(mut self, store_op: StoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// The driver handle of the attachment texture (read-only).
    pub fn texture_handle(&self) -> u64 {
        self.texture
    }
}

/// A depth/stencil attachment target with load/store operations.
///
/// Construct with [`DepthTarget::new`] from a typed [`Texture`] —
/// the attachment handle is derived from the texture, never passed as
/// a raw `u64`.
pub struct DepthTarget {
    /// Driver handle of the depth/stencil texture (derived from the
    /// `Texture` passed to [`DepthTarget::new`]).
    pub(crate) texture: u64,
    /// Depth load operation.
    pub load_op: LoadOp,
    /// Depth store operation.
    pub store_op: StoreOp,
    /// Stencil load operation.
    pub stencil_load_op: LoadOp,
    /// Stencil store operation.
    pub stencil_store_op: StoreOp,
}

impl DepthTarget {
    /// A depth/stencil attachment over `texture`, defaulting to
    /// clear-to-1.0 depth, `StoreOp::DontCare`, and don't-care stencil.
    pub fn new(texture: &Texture) -> Self {
        Self {
            texture: texture.handle(),
            load_op: LoadOp::Clear(Color::WHITE),
            store_op: StoreOp::DontCare,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
        }
    }

    /// Set the depth load operation.
    pub fn with_load_op(mut self, load_op: LoadOp) -> Self {
        self.load_op = load_op;
        self
    }

    /// Set the depth store operation.
    pub fn with_store_op(mut self, store_op: StoreOp) -> Self {
        self.store_op = store_op;
        self
    }

    /// Set the stencil load operation.
    pub fn with_stencil_load_op(mut self, load_op: LoadOp) -> Self {
        self.stencil_load_op = load_op;
        self
    }

    /// Set the stencil store operation.
    pub fn with_stencil_store_op(mut self, store_op: StoreOp) -> Self {
        self.stencil_store_op = store_op;
        self
    }

    /// The driver handle of the depth/stencil texture (read-only).
    pub fn texture_handle(&self) -> u64 {
        self.texture
    }
}

/// An active render pass — record draw commands, then submit.
#[allow(dead_code)]
pub struct RenderPass {
    pub(crate) handle: u64,
    pub(crate) ops: Vec<RenderOp>,
    /// Byte arena for op payloads (`SetValue` data, debug labels). Ops
    /// carry `(offset, len)` into this buffer, so recording touches the
    /// allocator only on arena growth — never per call — and `RenderOp`
    /// stays small.
    pub(crate) value_data: Vec<u8>,
    /// Color attachment targets (MRT support).
    pub(crate) color_targets: Vec<ColorTarget>,
    /// Depth/stencil attachment target.
    pub(crate) depth_target: Option<DepthTarget>,
    /// Pixel format of the primary render target (the `Texture` passed
    /// to `render`). Used for pass-shape validation in the single-target
    /// path, where `color_targets` is empty but the pass still binds one
    /// implicit color attachment. Stamped by the `Gpu` wrapper after
    /// `render_begin`; `None` if it could not be captured.
    pub(crate) primary_format: Option<crate::Format>,
    /// The color/depth formats of the pipeline bound in this pass, if
    /// one was bound. Captured from the [`Pipeline`] at `set_pipeline`
    /// so [`RenderPass::validate_pass_shape`] can compare the pipeline's
    /// declared attachments against the bound targets without a driver
    /// registry lookup. `None` means no pipeline was bound (nothing to
    /// validate).
    pub(crate) pipeline_shape: Option<PipelineShape>,
}

/// The declared attachment shape of the pipeline bound in a render pass,
/// snapshotted at bind time for encode-time validation.
pub(crate) struct PipelineShape {
    pub(crate) color_formats: Vec<crate::Format>,
    pub(crate) depth_format: Option<crate::Format>,
}

#[allow(dead_code)]
pub(crate) enum RenderOp {
    // Pipeline
    SetPipeline(u64),

    // Vertex/index buffers
    BindVertices {
        slot: u32,
        handle: u64,
        offset: u64,
    },
    BindIndices {
        handle: u64,
        offset: u64,
    },

    // Shader resources
    SetField {
        slot: u32,
        handle: u64,
    },
    SetUniform {
        slot: u32,
        handle: u64,
    },
    SetTexture {
        slot: u32,
        handle: u64,
    },
    SetSampler {
        slot: u32,
        sampler: SamplerDesc,
    },
    SetValue {
        slot: u32,
        offset: usize,
        len: usize,
    },

    // Draw
    Draw {
        vertex_count: u32,
        instance_count: u32,
    },
    DrawIndexed {
        index_count: u32,
        instance_count: u32,
    },

    // Render state
    Clear(Color),
    ClearDepth(f32),
    ClearStencil(u32),
    SetStencilRef(u32),

    // Debug
    DebugPush {
        offset: usize,
        len: usize,
    },
    DebugPop,

    // Indirect
    DrawIndirect {
        buffer_handle: u64,
        offset: u64,
    },
    DrawIndexedIndirect {
        buffer_handle: u64,
        offset: u64,
        index_handle: u64,
    },
    SetScissor {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    SetViewport {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    },

    // Occlusion queries (M3.3)
    BeginOcclusionQuery {
        handle: u64,
        index: u32,
    },
    EndOcclusionQuery {
        handle: u64,
        index: u32,
    },

    // Variable-rate shading (M4.4)
    SetShadingRate(ShadingRate),
    SetShadingRateImage {
        texture_handle: u64,
    },

    // Indirect render bundle (steps 032 + 033, render path)
    /// Replay the first `count` recorded draws from an
    /// `IndirectRenderBundle`. Lowered to Metal
    /// `executeCommandsInBuffer:withRange:`, Vulkan
    /// `vkCmdExecuteCommands` (with secondary CBs recorded in
    /// VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT mode), or
    /// WebGPU `pass.executeBundles`.
    ExecuteRenderBundle {
        bundle_handle: u64,
        count: u32,
    },
}

/// Which driver registry a recorded handle points into. Drivers map
/// each kind onto their own registry in [`RenderPass::validate_handles`]
/// (occlusion queries live in different registries per backend).
pub(crate) enum HandleKind {
    Buffer,
    Texture,
    Pipeline,
    OcclusionQuery,
}

impl RenderPass {
    /// Pre-encode validation: walk the recorded ops, check every
    /// registry handle they reference via `lookup`, and enforce the
    /// ordering rules a replay depends on (`draw_indexed` needs a
    /// prior `indices` bind). Drivers call this before encoding so a
    /// dead handle fails the submission loudly instead of silently
    /// skipping the bind — the classic cause is a `Field` dropped
    /// before `pulse()`; bound resources must outlive the submission.
    /// (Handles are never reused, so a dead handle can only be
    /// missing, never rebound to another resource.)
    pub(crate) fn validate_handles(
        &self,
        mut lookup: impl FnMut(HandleKind, u64) -> bool,
    ) -> Result<(), crate::QuantaError> {
        let mut check = |kind: HandleKind, handle: u64, what: &str, i: usize| {
            if lookup(kind, handle) {
                Ok(())
            } else {
                Err(
                    crate::QuantaError::not_found("render pass references a dead handle")
                        .with_context(&alloc::format!(
                            "op {i}: {what} handle {handle} is not registered — was the \
                     resource dropped before pulse()? Everything bound to a \
                     render pass must outlive the submission"
                        )),
                )
            }
        };
        let mut indices_bound = false;
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                RenderOp::SetPipeline(h) => check(HandleKind::Pipeline, *h, "pipeline", i)?,
                RenderOp::BindVertices { handle, .. } => {
                    check(HandleKind::Buffer, *handle, "vertex buffer", i)?
                }
                RenderOp::BindIndices { handle, .. } => {
                    indices_bound = true;
                    check(HandleKind::Buffer, *handle, "index buffer", i)?
                }
                RenderOp::SetField { handle, .. } => {
                    check(HandleKind::Buffer, *handle, "field", i)?
                }
                RenderOp::SetUniform { handle, .. } => {
                    check(HandleKind::Buffer, *handle, "uniform field", i)?
                }
                RenderOp::SetTexture { handle, .. } => {
                    check(HandleKind::Texture, *handle, "texture", i)?
                }
                RenderOp::DrawIndexed { .. } if !indices_bound => {
                    return Err(crate::QuantaError::invalid_param(
                        "draw_indexed with no index buffer bound",
                    )
                    .with_context(&alloc::format!(
                        "op {i}: call .indices(&field) before .draw_indexed(n)"
                    )));
                }
                RenderOp::DrawIndirect { buffer_handle, .. } => check(
                    HandleKind::Buffer,
                    *buffer_handle,
                    "indirect argument buffer",
                    i,
                )?,
                RenderOp::DrawIndexedIndirect {
                    buffer_handle,
                    index_handle,
                    ..
                } => {
                    check(
                        HandleKind::Buffer,
                        *buffer_handle,
                        "indirect argument buffer",
                        i,
                    )?;
                    check(HandleKind::Buffer, *index_handle, "index buffer", i)?
                }
                RenderOp::BeginOcclusionQuery { handle, .. }
                | RenderOp::EndOcclusionQuery { handle, .. } => {
                    check(HandleKind::OcclusionQuery, *handle, "occlusion query", i)?
                }
                RenderOp::SetShadingRateImage { texture_handle } => check(
                    HandleKind::Texture,
                    *texture_handle,
                    "shading-rate image",
                    i,
                )?,
                _ => {}
            }
        }
        Ok(())
    }

    /// Pre-encode pass-shape validation: check that the pipeline bound
    /// in this pass agrees with the color/depth targets the pass binds.
    ///
    /// `PipelineDesc::color_formats` is **per-attachment** — entry `i`
    /// types color attachment `i` of the pass — so the pipeline's
    /// attachment count and per-index formats must match the pass. A
    /// consumer that read `color_formats` as a candidate list (declaring
    /// `[BGRA8, RGBA8]` for a single `RGBA8` target) produced a phantom
    /// second attachment and a mis-typed first one, which some drivers
    /// accept silently and then drop draws for — undebuggable without
    /// this check. It is always-on (never `QUANTA_VALIDATE`-gated,
    /// backend-agnostic) because a mismatch here is never legitimate.
    ///
    /// Runs only when a pipeline was bound (otherwise there is nothing
    /// to validate — a clear-only pass binds targets but no pipeline).
    /// Drivers call this beside [`RenderPass::validate_handles`], before
    /// encoding, so a mismatch fails `pulse()` loudly.
    pub(crate) fn validate_pass_shape(&self) -> Result<(), crate::QuantaError> {
        let Some(shape) = self.pipeline_shape.as_ref() else {
            return Ok(());
        };
        let declared = shape.color_formats.len();

        // How many color attachments the pass actually binds. With
        // explicit `color_targets`, that count is exact. Otherwise the
        // pass uses the legacy single-target path, which binds the
        // primary render target as the sole color attachment (count 1) —
        // *unless* the pipeline is depth-only (declares no color
        // formats), where the primary target is the depth target and
        // there are zero color attachments (count 0).
        let bound_count = if !self.color_targets.is_empty() {
            self.color_targets.len()
        } else if declared == 0 {
            0
        } else {
            1
        };
        if declared != bound_count {
            return Err(crate::QuantaError::invalid_param(alloc::format!(
                "pipeline declares {declared} color attachments but the pass binds \
                 {bound_count} color targets"
            ))
            .with_context(
                "color_formats[i] types color attachment i, so the pipeline's \
                 attachment count must equal the pass's color-target count — a \
                 consumer that read color_formats as a candidate list of usable \
                 formats is the usual cause",
            ));
        }

        // Per-attachment format: the bound target `i` must carry the
        // format the pipeline declared for attachment `i`. In the
        // single-target path the one target's format is `primary_format`.
        // The loop only visits declared attachments, so a depth-only
        // pipeline (empty `color_formats`) checks nothing here.
        let format_of = |i: usize| -> Option<crate::Format> {
            if self.color_targets.is_empty() {
                self.primary_format
            } else {
                Some(self.color_targets[i].format)
            }
        };
        for (i, &expected) in shape.color_formats.iter().enumerate() {
            if let Some(got) = format_of(i)
                && got != expected
            {
                return Err(crate::QuantaError::invalid_param(alloc::format!(
                    "color target {i} format mismatch: pipeline color_formats[{i}] is \
                     {expected:?} but the bound target is {got:?}"
                )));
            }
        }

        // Depth presence: enforced only for passes with explicit color
        // targets, where the attachment roles are unambiguous. The pure
        // legacy single-target path routes the primary handle to color
        // *or* depth implicitly (a depth-only shadow pass binds its depth
        // map as the primary target with no explicit `depth_target`), so
        // its depth shape is not something we can flag without false
        // positives.
        if !self.color_targets.is_empty() {
            match (self.depth_target.is_some(), shape.depth_format) {
                (true, None) => {
                    return Err(crate::QuantaError::invalid_param(
                        "pass binds a depth target but the pipeline declares no depth format",
                    ));
                }
                (false, Some(fmt)) => {
                    return Err(crate::QuantaError::invalid_param(alloc::format!(
                        "pipeline declares depth format {fmt:?} but the pass binds no \
                         depth target"
                    )));
                }
                _ => {}
            }
        }

        Ok(())
    }

    // === Pipeline ===

    /// Bind a render pipeline.
    pub fn set_pipeline(&mut self, pipeline: &Pipeline) {
        self.ops.push(RenderOp::SetPipeline(pipeline.handle()));
        // Snapshot the pipeline's declared attachment shape so the
        // pre-encode `validate_pass_shape` scan can check it against the
        // bound targets. The last pipeline bound wins — drivers encode
        // draws against the most recently set pipeline.
        self.pipeline_shape = Some(PipelineShape {
            color_formats: pipeline.color_formats.clone(),
            depth_format: pipeline.depth_format,
        });
    }

    // === Vertex/Index data ===

    /// Bind a vertex buffer at a slot (0 = vertices, 1 = instances, etc.).
    pub fn bind_vertices<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        self.ops.push(RenderOp::BindVertices {
            slot,
            handle: field.handle(),
            offset: 0,
        });
    }

    /// Bind a vertex buffer at a slot with a byte offset.
    pub fn bind_vertices_offset<T: Copy>(&mut self, slot: u32, field: &Field<T>, offset: u64) {
        self.ops.push(RenderOp::BindVertices {
            slot,
            handle: field.handle(),
            offset,
        });
    }

    /// Bind an index buffer (u32 indices).
    pub fn bind_indices(&mut self, field: &Field<u32>) {
        self.ops.push(RenderOp::BindIndices {
            handle: field.handle(),
            offset: 0,
        });
    }

    // === Shader resources ===

    /// Bind a storage buffer at a shader slot.
    pub fn set_field<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        self.ops.push(RenderOp::SetField {
            slot,
            handle: field.handle(),
        });
    }

    /// Bind a uniform buffer at a shader slot.
    pub fn set_uniform<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        self.ops.push(RenderOp::SetUniform {
            slot,
            handle: field.handle(),
        });
    }

    /// Bind a texture at a shader texture slot.
    pub fn set_texture(&mut self, slot: u32, texture: &Texture) {
        self.ops.push(RenderOp::SetTexture {
            slot,
            handle: texture.handle(),
        });
    }

    /// Set sampler state for a texture slot.
    pub fn set_sampler(&mut self, slot: u32, sampler: SamplerDesc) {
        self.ops.push(RenderOp::SetSampler { slot, sampler });
    }

    /// Set push constant / uniform data at a slot (any size).
    pub fn set_value<V: Copy>(&mut self, slot: u32, value: &V) {
        let size = size_of::<V>();
        let offset = self.value_data.len();
        self.value_data.resize(offset + size, 0);
        unsafe {
            core::ptr::copy_nonoverlapping(
                value as *const V as *const u8,
                self.value_data.as_mut_ptr().add(offset),
                size,
            );
        }
        self.ops.push(RenderOp::SetValue {
            slot,
            offset,
            len: size,
        });
    }

    /// Resolve an `(offset, len)` pair recorded by [`Self::set_value`] /
    /// [`Self::debug_push`] back into its bytes in the pass arena.
    pub(crate) fn value_bytes(&self, offset: usize, len: usize) -> &[u8] {
        &self.value_data[offset..offset + len]
    }

    /// Resolve a recorded debug label. The bytes came from a `&str`, so
    /// the empty-string fallback can only fire on a recording bug.
    #[allow(dead_code)]
    pub(crate) fn debug_label(&self, offset: usize, len: usize) -> &str {
        core::str::from_utf8(self.value_bytes(offset, len)).unwrap_or("")
    }

    // === Draw commands ===

    /// Draw vertices (non-indexed, non-instanced).
    pub fn draw(&mut self, vertex_count: u32) {
        self.ops.push(RenderOp::Draw {
            vertex_count,
            instance_count: 1,
        });
    }

    /// Draw instanced geometry.
    pub fn draw_instanced(&mut self, vertex_count: u32, instance_count: u32) {
        self.ops.push(RenderOp::Draw {
            vertex_count,
            instance_count,
        });
    }

    /// Draw indexed geometry (tessellated paths, 3D meshes).
    pub fn draw_indexed(&mut self, index_count: u32) {
        self.ops.push(RenderOp::DrawIndexed {
            index_count,
            instance_count: 1,
        });
    }

    /// Draw indexed + instanced.
    pub fn draw_indexed_instanced(&mut self, index_count: u32, instance_count: u32) {
        self.ops.push(RenderOp::DrawIndexed {
            index_count,
            instance_count,
        });
    }

    // === Render state ===

    /// Clear the color attachment.
    pub fn clear(&mut self, color: Color) {
        self.ops.push(RenderOp::Clear(color));
    }

    /// Clear the depth attachment.
    pub fn clear_depth(&mut self, depth: f32) {
        self.ops.push(RenderOp::ClearDepth(depth));
    }

    /// Clear the stencil attachment.
    pub fn clear_stencil(&mut self, value: u32) {
        self.ops.push(RenderOp::ClearStencil(value));
    }

    /// Set the stencil reference value for comparison.
    pub fn set_stencil_ref(&mut self, value: u32) {
        self.ops.push(RenderOp::SetStencilRef(value));
    }

    /// Set scissor rectangle (pixel coordinates).
    ///
    /// Offsets are clamped to the render area on every backend: an offset
    /// that would fall outside the target (including a negative offset
    /// passed as a wrapped-in `u32` — the common "clip a child scrolled
    /// past its parent" case) is pulled to the render-area edge and the
    /// extent shrinks to match; a rectangle that clamps entirely away
    /// disables drawing for that pass without raising an error. This gives
    /// identical results across Metal (which tolerates such rectangles
    /// natively) and Vulkan (which would otherwise reject a negative
    /// offset), so the same app code behaves the same everywhere.
    pub fn set_scissor(&mut self, x: u32, y: u32, width: u32, height: u32) {
        self.ops.push(RenderOp::SetScissor {
            x,
            y,
            width,
            height,
        });
    }

    /// Set viewport (normalized device coordinates mapping).
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
        self.ops.push(RenderOp::SetViewport {
            x,
            y,
            width,
            height,
            min_depth: 0.0,
            max_depth: 1.0,
        });
    }

    /// Draw with arguments from a GPU buffer (GPU-driven rendering).
    /// The buffer contains: [vertex_count: u32, instance_count: u32, first_vertex: u32, first_instance: u32]
    pub fn draw_indirect<T: Copy>(&mut self, buffer: &Field<T>, offset: u64) {
        self.ops.push(RenderOp::DrawIndirect {
            buffer_handle: buffer.handle(),
            offset,
        });
    }

    /// Draw indexed with arguments from a GPU buffer.
    /// The buffer contains: [index_count: u32, instance_count: u32, first_index: u32, base_vertex: i32, first_instance: u32]
    pub fn draw_indexed_indirect<T: Copy>(
        &mut self,
        buffer: &Field<T>,
        offset: u64,
        indices: &Field<u32>,
    ) {
        self.ops.push(RenderOp::DrawIndexedIndirect {
            buffer_handle: buffer.handle(),
            offset,
            index_handle: indices.handle(),
        });
    }

    /// Push a debug label for this section of the render pass.
    pub fn debug_push(&mut self, label: &str) {
        let offset = self.value_data.len();
        self.value_data.extend_from_slice(label.as_bytes());
        self.ops.push(RenderOp::DebugPush {
            offset,
            len: label.len(),
        });
    }

    /// Pop a debug label.
    pub fn debug_pop(&mut self) {
        self.ops.push(RenderOp::DebugPop);
    }

    /// Set viewport with depth range.
    pub fn set_viewport_depth(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        self.ops.push(RenderOp::SetViewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        });
    }

    // === Occlusion Queries (M3.3) ===

    /// Begin an occlusion query at the given index.
    ///
    /// All fragments drawn between begin and end are counted. A result
    /// of zero means every fragment was culled by depth/stencil.
    pub fn begin_occlusion_query(&mut self, query: &OcclusionQuery, index: u32) {
        self.ops.push(RenderOp::BeginOcclusionQuery {
            handle: query.handle(),
            index,
        });
    }

    /// End an occlusion query at the given index.
    pub fn end_occlusion_query(&mut self, query: &OcclusionQuery, index: u32) {
        self.ops.push(RenderOp::EndOcclusionQuery {
            handle: query.handle(),
            index,
        });
    }

    // === Variable-Rate Shading (M4.4) ===

    /// Set a uniform shading rate for subsequent draw calls.
    pub fn set_shading_rate(&mut self, rate: ShadingRate) {
        self.ops.push(RenderOp::SetShadingRate(rate));
    }

    /// Set a per-pixel shading rate from a shading rate image.
    ///
    /// The texture must contain shading rate values; each texel
    /// controls the rate for a screen-space tile.
    pub fn set_shading_rate_image(&mut self, texture: &Texture) {
        self.ops.push(RenderOp::SetShadingRateImage {
            texture_handle: texture.handle(),
        });
    }

    // === Indirect render bundles (steps 032 + 033) ===

    /// Replay the first `count` recorded draws from an
    /// [`IndirectRenderBundle`](crate::IndirectRenderBundle).
    ///
    /// Backends translate this to Metal
    /// `executeCommandsInBuffer:withRange:` on the active render
    /// encoder, Vulkan `vkCmdExecuteCommands` inside the parent
    /// render pass, or WebGPU `pass.executeBundles`. The proof
    /// contract — recorded order is preserved on execute (T7000) —
    /// applies regardless of backend.
    pub fn execute_bundle(&mut self, bundle: &crate::IndirectRenderBundle, count: u32) {
        self.ops.push(RenderOp::ExecuteRenderBundle {
            bundle_handle: bundle.handle(),
            count,
        });
    }

    // === Multiple Render Targets (MRT) ===

    /// Set the color attachment targets for this render pass.
    /// Enables multiple render targets (MRT) — drivers read these when executing.
    pub fn set_color_targets(&mut self, targets: Vec<ColorTarget>) {
        self.color_targets = targets;
    }

    /// Set the depth/stencil attachment target for this render pass.
    pub fn set_depth_target(&mut self, target: DepthTarget) {
        self.depth_target = Some(target);
    }
}

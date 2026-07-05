use alloc::string::{String, ToString};
use alloc::vec;
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
    /// Color attachment targets (MRT support).
    pub(crate) color_targets: Vec<ColorTarget>,
    /// Depth/stencil attachment target.
    pub(crate) depth_target: Option<DepthTarget>,
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
        data: Vec<u8>,
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
    DebugPush(String),
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

impl RenderPass {
    // === Pipeline ===

    /// Bind a render pipeline.
    pub fn set_pipeline(&mut self, pipeline: &Pipeline) {
        self.ops.push(RenderOp::SetPipeline(pipeline.handle()));
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
        let mut data = vec![0u8; size];
        unsafe {
            core::ptr::copy_nonoverlapping(value as *const V as *const u8, data.as_mut_ptr(), size);
        }
        self.ops.push(RenderOp::SetValue { slot, data });
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
        self.ops.push(RenderOp::DebugPush(label.to_string()));
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

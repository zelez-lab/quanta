use crate::{Color, Field, Pipeline, Texture};

/// An active render pass — record draw commands, then submit.
#[allow(dead_code)]
pub struct RenderPass {
    pub(crate) handle: u64,
    pub(crate) ops: Vec<RenderOp>,
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
}

/// Texture sampling configuration.
#[derive(Debug, Clone, Copy)]
pub struct SamplerDesc {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub mip_filter: Filter,
    pub address_u: AddressMode,
    pub address_v: AddressMode,
    pub max_anisotropy: u8,
}

impl Default for SamplerDesc {
    fn default() -> Self {
        Self {
            min_filter: Filter::Linear,
            mag_filter: Filter::Linear,
            mip_filter: Filter::Nearest,
            address_u: AddressMode::ClampToEdge,
            address_v: AddressMode::ClampToEdge,
            max_anisotropy: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressMode {
    ClampToEdge,
    Repeat,
    MirrorRepeat,
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

    /// Draw instanced (Dija's primary draw call).
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
}

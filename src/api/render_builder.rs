//! Builder API for render passes.
//!
//! Wraps `RenderPass` in a chainable builder that terminates with `.pulse()`.
//!
//! ```ignore
//! let mut pulse = gpu.render(&target)?
//!     .clear(Color::BLACK)
//!     .pipeline(&pipeline)
//!     .vertices(0, &verts)
//!     .draw(3)
//!     .pulse()?;
//! gpu.wait(&mut pulse)?;
//! ```

use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::render_pass::{ColorTarget, DepthTarget, SamplerDesc};
use crate::{
    Color, Field, GpuDevice, OcclusionQuery, Pipeline, Pulse, QuantaError, RenderPass, ShadingRate,
    Texture,
};

/// A chainable render pass builder.
///
/// Created by [`Gpu::render()`]. Every method consumes and returns `self`,
/// so the entire pass can be expressed as a single expression ending in
/// `.pulse()`.
pub struct RenderBuilder {
    device: Arc<dyn GpuDevice>,
    pass: RenderPass,
}

impl RenderBuilder {
    pub(crate) fn new(device: Arc<dyn GpuDevice>, pass: RenderPass) -> Self {
        Self { device, pass }
    }

    // === Pipeline ===

    /// Bind a render pipeline.
    pub fn pipeline(mut self, p: &Pipeline) -> Self {
        self.pass.set_pipeline(p);
        self
    }

    // === Vertex / Index data ===

    /// Bind a vertex buffer at a slot.
    pub fn vertices<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.bind_vertices(slot, field);
        self
    }

    /// Bind a vertex buffer at a slot with a byte offset.
    pub fn vertices_offset<T: Copy>(mut self, slot: u32, field: &Field<T>, offset: u64) -> Self {
        self.pass.bind_vertices_offset(slot, field, offset);
        self
    }

    /// Bind an index buffer (u32 indices).
    pub fn indices(mut self, field: &Field<u32>) -> Self {
        self.pass.bind_indices(field);
        self
    }

    // === Shader resources ===

    /// Bind a storage buffer at a shader slot.
    pub fn field<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.set_field(slot, field);
        self
    }

    /// Bind a uniform buffer at a shader slot.
    pub fn uniform<T: Copy>(mut self, slot: u32, field: &Field<T>) -> Self {
        self.pass.set_uniform(slot, field);
        self
    }

    /// Bind a texture at a shader texture slot.
    pub fn texture(mut self, slot: u32, tex: &Texture) -> Self {
        self.pass.set_texture(slot, tex);
        self
    }

    /// Set sampler state for a texture slot.
    pub fn sampler(mut self, slot: u32, desc: SamplerDesc) -> Self {
        self.pass.set_sampler(slot, desc);
        self
    }

    /// Set push constant / uniform data at a slot.
    pub fn value<V: Copy>(mut self, slot: u32, val: &V) -> Self {
        self.pass.set_value(slot, val);
        self
    }

    // === Draw commands ===

    /// Draw vertices (non-indexed, non-instanced).
    pub fn draw(mut self, vertex_count: u32) -> Self {
        self.pass.draw(vertex_count);
        self
    }

    /// Draw instanced geometry.
    pub fn draw_instanced(mut self, vertex_count: u32, instance_count: u32) -> Self {
        self.pass.draw_instanced(vertex_count, instance_count);
        self
    }

    /// Draw indexed geometry.
    pub fn draw_indexed(mut self, index_count: u32) -> Self {
        self.pass.draw_indexed(index_count);
        self
    }

    /// Draw indexed + instanced.
    pub fn draw_indexed_instanced(mut self, index_count: u32, instance_count: u32) -> Self {
        self.pass
            .draw_indexed_instanced(index_count, instance_count);
        self
    }

    /// Draw with arguments from a GPU buffer (GPU-driven rendering).
    pub fn draw_indirect<T: Copy>(mut self, buffer: &Field<T>, offset: u64) -> Self {
        self.pass.draw_indirect(buffer, offset);
        self
    }

    /// Draw indexed with arguments from a GPU buffer.
    pub fn draw_indexed_indirect<T: Copy>(
        mut self,
        buffer: &Field<T>,
        offset: u64,
        indices: &Field<u32>,
    ) -> Self {
        self.pass.draw_indexed_indirect(buffer, offset, indices);
        self
    }

    // === Render state ===

    /// Clear the color attachment.
    pub fn clear(mut self, color: Color) -> Self {
        self.pass.clear(color);
        self
    }

    /// Clear the depth attachment.
    pub fn clear_depth(mut self, depth: f32) -> Self {
        self.pass.clear_depth(depth);
        self
    }

    /// Clear the stencil attachment.
    pub fn clear_stencil(mut self, value: u32) -> Self {
        self.pass.clear_stencil(value);
        self
    }

    /// Set the stencil reference value for comparison.
    pub fn stencil_ref(mut self, value: u32) -> Self {
        self.pass.set_stencil_ref(value);
        self
    }

    /// Set scissor rectangle (pixel coordinates).
    pub fn scissor(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.pass.set_scissor(x, y, width, height);
        self
    }

    /// Set viewport (normalized device coordinates mapping).
    pub fn viewport(mut self, x: f32, y: f32, width: f32, height: f32) -> Self {
        self.pass.set_viewport(x, y, width, height);
        self
    }

    /// Set viewport with depth range.
    pub fn viewport_depth(
        mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) -> Self {
        self.pass
            .set_viewport_depth(x, y, width, height, min_depth, max_depth);
        self
    }

    // === Debug ===

    /// Push a debug label for this section of the render pass.
    pub fn debug_push(mut self, label: &str) -> Self {
        self.pass.debug_push(label);
        self
    }

    /// Pop a debug label.
    pub fn debug_pop(mut self) -> Self {
        self.pass.debug_pop();
        self
    }

    // === Occlusion Queries ===

    /// Begin an occlusion query at the given index.
    pub fn begin_occlusion_query(mut self, query: &OcclusionQuery, index: u32) -> Self {
        self.pass.begin_occlusion_query(query, index);
        self
    }

    /// End an occlusion query at the given index.
    pub fn end_occlusion_query(mut self, query: &OcclusionQuery, index: u32) -> Self {
        self.pass.end_occlusion_query(query, index);
        self
    }

    // === Variable-Rate Shading ===

    /// Set a uniform shading rate for subsequent draw calls.
    pub fn shading_rate(mut self, rate: ShadingRate) -> Self {
        self.pass.set_shading_rate(rate);
        self
    }

    /// Set a per-pixel shading rate from a shading rate image.
    pub fn shading_rate_image(mut self, texture: &Texture) -> Self {
        self.pass.set_shading_rate_image(texture);
        self
    }

    // === Multiple Render Targets ===

    /// Set the color attachment targets for this render pass.
    pub fn color_targets(mut self, targets: Vec<ColorTarget>) -> Self {
        self.pass.set_color_targets(targets);
        self
    }

    /// Set the depth/stencil attachment target for this render pass.
    pub fn depth_target(mut self, target: DepthTarget) -> Self {
        self.pass.set_depth_target(target);
        self
    }

    // === Terminal ===

    /// Submit the render pass for execution.
    ///
    /// Consumes the builder and returns a `Pulse` that signals when the
    /// GPU finishes rendering.
    pub fn pulse(self) -> Result<Pulse, QuantaError> {
        self.device.render_end(self.pass)
    }
}

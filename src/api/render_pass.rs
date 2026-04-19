use crate::{Color, Field, Pipeline};

/// An active render pass — draw commands between begin and end.
///
/// Created via [`GpuDevice::render_begin`]. Draw geometry, then end.
#[allow(dead_code)]
pub struct RenderPass {
    pub(crate) handle: u64,
    pub(crate) ops: Vec<RenderOp>,
}

#[allow(dead_code)]
pub(crate) enum RenderOp {
    SetPipeline(u64),
    BindVertices(u64),
    BindIndices(u64),
    SetField { slot: u32, handle: u64 },
    SetValue { slot: u32, data: [u8; 16] },
    Draw { vertex_count: u32 },
    DrawIndexed { index_count: u32 },
    Clear(Color),
}

impl RenderPass {
    /// Bind a render pipeline.
    pub fn set_pipeline(&mut self, pipeline: &Pipeline) {
        self.ops.push(RenderOp::SetPipeline(pipeline.handle()));
    }

    /// Bind vertex data.
    pub fn bind_vertices<T: Copy>(&mut self, field: &Field<T>) {
        self.ops.push(RenderOp::BindVertices(field.handle()));
    }

    /// Bind index data.
    pub fn bind_indices(&mut self, field: &Field<u32>) {
        self.ops.push(RenderOp::BindIndices(field.handle()));
    }

    /// Bind a field at a shader slot.
    pub fn set_field<T: Copy>(&mut self, slot: u32, field: &Field<T>) {
        self.ops.push(RenderOp::SetField {
            slot,
            handle: field.handle(),
        });
    }

    /// Set a push constant value.
    pub fn set_value<V: Copy>(&mut self, slot: u32, value: V) {
        assert!(size_of::<V>() <= 16, "push constant must be ≤ 16 bytes");
        let mut data = [0u8; 16];
        unsafe {
            core::ptr::copy_nonoverlapping(
                &value as *const V as *const u8,
                data.as_mut_ptr(),
                size_of::<V>(),
            );
        }
        self.ops.push(RenderOp::SetValue { slot, data });
    }

    /// Draw vertices.
    pub fn draw(&mut self, vertex_count: u32) {
        self.ops.push(RenderOp::Draw { vertex_count });
    }

    /// Draw indexed geometry.
    pub fn draw_indexed(&mut self, index_count: u32) {
        self.ops.push(RenderOp::DrawIndexed { index_count });
    }

    /// Clear the render target.
    pub fn clear(&mut self, color: Color) {
        self.ops.push(RenderOp::Clear(color));
    }
}

use crate::Format;

/// Compiled render pipeline (vertex + fragment shaders + state).
pub struct Pipeline {
    pub(crate) handle: u64,
    pub(crate) drop_fn: Option<Box<dyn FnOnce(u64)>>,
}

impl Pipeline {
    pub fn handle(&self) -> u64 {
        self.handle
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if let Some(f) = self.drop_fn.take() {
            f(self.handle);
        }
    }
}

/// Describes how to create a render pipeline.
pub struct PipelineDesc<'a> {
    /// Vertex shader source (MSL, WGSL, or compiled binary).
    pub vertex: &'a [u8],
    /// Fragment shader source.
    pub fragment: &'a [u8],
    /// Vertex shader entry point name.
    pub vertex_entry: &'a str,
    /// Fragment shader entry point name.
    pub fragment_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [VertexLayout],
    /// Color attachment format.
    pub color_format: Format,
    /// Depth attachment format (None = no depth buffer).
    pub depth_format: Option<Format>,
    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    pub sample_count: u32,
    /// Blend state for color attachment.
    pub blend: BlendState,
    /// Face culling mode.
    pub cull_mode: CullMode,
    /// Primitive topology.
    pub primitive: Primitive,
}

impl<'a> Default for PipelineDesc<'a> {
    fn default() -> Self {
        Self {
            vertex: &[],
            fragment: &[],
            vertex_entry: "vertex_main",
            fragment_entry: "fragment_main",
            vertex_layouts: &[],
            color_format: Format::BGRA8,
            depth_format: None,
            sample_count: 1,
            blend: BlendState::PREMULTIPLIED_ALPHA,
            cull_mode: CullMode::None,
            primitive: Primitive::Triangle,
        }
    }
}

/// Describes vertex buffer layout — how to read vertex/instance data.
pub struct VertexLayout {
    /// Byte stride between elements.
    pub stride: u32,
    /// Per-vertex or per-instance data.
    pub step: StepMode,
    /// Attributes within this buffer.
    pub attributes: Vec<VertexAttribute>,
}

/// How a vertex buffer advances — per vertex or per instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepMode {
    Vertex,
    Instance,
}

/// A single vertex attribute within a layout.
pub struct VertexAttribute {
    /// Shader location (attribute index).
    pub location: u32,
    /// Byte offset within the stride.
    pub offset: u32,
    /// Data format.
    pub format: AttributeFormat,
}

/// Vertex attribute data formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeFormat {
    Float,
    Float2,
    Float3,
    Float4,
    Int,
    Int2,
    Int3,
    Int4,
    UInt,
    UInt2,
    UInt3,
    UInt4,
    UByte4Norm,
}

/// Blend state for a color attachment.
#[derive(Debug, Clone, Copy)]
pub struct BlendState {
    pub enabled: bool,
    pub src_rgb: BlendFactor,
    pub dst_rgb: BlendFactor,
    pub src_alpha: BlendFactor,
    pub dst_alpha: BlendFactor,
    pub op_rgb: BlendOp,
    pub op_alpha: BlendOp,
}

impl BlendState {
    /// No blending — overwrite destination.
    pub const NONE: Self = Self {
        enabled: false,
        src_rgb: BlendFactor::One,
        dst_rgb: BlendFactor::Zero,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::Zero,
        op_rgb: BlendOp::Add,
        op_alpha: BlendOp::Add,
    };

    /// Premultiplied alpha blending (what Dija uses).
    pub const PREMULTIPLIED_ALPHA: Self = Self {
        enabled: true,
        src_rgb: BlendFactor::One,
        dst_rgb: BlendFactor::OneMinusSrcAlpha,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        op_rgb: BlendOp::Add,
        op_alpha: BlendOp::Add,
    };

    /// Standard alpha blending (non-premultiplied).
    pub const ALPHA: Self = Self {
        enabled: true,
        src_rgb: BlendFactor::SrcAlpha,
        dst_rgb: BlendFactor::OneMinusSrcAlpha,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::OneMinusSrcAlpha,
        op_rgb: BlendOp::Add,
        op_alpha: BlendOp::Add,
    };

    /// Additive blending (particles, glow effects).
    pub const ADDITIVE: Self = Self {
        enabled: true,
        src_rgb: BlendFactor::One,
        dst_rgb: BlendFactor::One,
        src_alpha: BlendFactor::One,
        dst_alpha: BlendFactor::One,
        op_rgb: BlendOp::Add,
        op_alpha: BlendOp::Add,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Face culling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// Primitive topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

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
    /// If `source` is set, this is ignored.
    pub vertex: &'a [u8],
    /// Fragment shader source.
    /// If `source` is set, this is ignored.
    pub fragment: &'a [u8],
    /// Combined shader source containing both vertex and fragment functions.
    /// When set, `vertex` and `fragment` fields are ignored.
    /// The driver finds functions by `vertex_entry` and `fragment_entry` names.
    pub source: Option<&'a [u8]>,
    /// Vertex shader entry point name.
    pub vertex_entry: &'a str,
    /// Fragment shader entry point name.
    pub fragment_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [VertexLayout],
    /// Color attachment formats (MRT — multiple render targets).
    /// First entry is the primary color format.
    pub color_formats: Vec<Format>,
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
    /// Depth/stencil state.
    pub depth_stencil: DepthStencilState,
}

impl<'a> Default for PipelineDesc<'a> {
    fn default() -> Self {
        Self {
            vertex: &[],
            fragment: &[],
            source: None,
            vertex_entry: "vertex_main",
            fragment_entry: "fragment_main",
            vertex_layouts: &[],
            color_formats: vec![Format::BGRA8],
            depth_format: None,
            sample_count: 1,
            blend: BlendState::PREMULTIPLIED_ALPHA,
            cull_mode: CullMode::None,
            primitive: Primitive::Triangle,
            depth_stencil: DepthStencilState::NONE,
        }
    }
}

/// Depth and stencil testing configuration.
#[derive(Debug, Clone, Copy)]
pub struct DepthStencilState {
    /// Enable depth testing.
    pub depth_test: bool,
    /// Enable depth writing.
    pub depth_write: bool,
    /// Depth comparison function.
    pub depth_compare: CompareFunc,
    /// Front face stencil operations.
    pub stencil_front: Option<StencilState>,
    /// Back face stencil operations.
    pub stencil_back: Option<StencilState>,
}

impl DepthStencilState {
    /// No depth or stencil testing (2D rendering).
    pub const NONE: Self = Self {
        depth_test: false,
        depth_write: false,
        depth_compare: CompareFunc::Always,
        stencil_front: None,
        stencil_back: None,
    };

    /// Standard 3D depth testing — closer fragments win.
    pub const DEPTH_LESS: Self = Self {
        depth_test: true,
        depth_write: true,
        depth_compare: CompareFunc::Less,
        stencil_front: None,
        stencil_back: None,
    };

    /// Depth test without writing — for transparent objects after opaques.
    pub const DEPTH_READ_ONLY: Self = Self {
        depth_test: true,
        depth_write: false,
        depth_compare: CompareFunc::Less,
        stencil_front: None,
        stencil_back: None,
    };
}

/// Per-face stencil operations.
#[derive(Debug, Clone, Copy)]
pub struct StencilState {
    /// What to do when stencil test fails.
    pub fail: StencilOp,
    /// What to do when stencil passes but depth fails.
    pub depth_fail: StencilOp,
    /// What to do when both stencil and depth pass.
    pub pass: StencilOp,
    /// Stencil comparison function.
    pub compare: CompareFunc,
    /// Read mask for stencil value.
    pub read_mask: u32,
    /// Write mask for stencil value.
    pub write_mask: u32,
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            fail: StencilOp::Keep,
            depth_fail: StencilOp::Keep,
            pass: StencilOp::Keep,
            compare: CompareFunc::Always,
            read_mask: 0xFF,
            write_mask: 0xFF,
        }
    }
}

/// Comparison function for depth and stencil tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareFunc {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Stencil operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncrementClamp,
    DecrementClamp,
    Invert,
    IncrementWrap,
    DecrementWrap,
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

    /// Premultiplied alpha blending.
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

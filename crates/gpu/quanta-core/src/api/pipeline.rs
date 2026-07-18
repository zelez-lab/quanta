use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use crate::{CompareOp, Format, GpuDevice, ShaderBinary, Vendor};

/// Compiled render pipeline (vertex + fragment shaders + state).
///
/// Dropping a pipeline calls `GpuDevice::pipeline_destroy` exactly once.
pub struct Pipeline {
    pub(crate) handle: u64,
    /// Drivers construct pipelines with `device: None`; the `Gpu`
    /// wrapper attaches the device Arc so Drop can release the handle.
    pub(crate) device: Option<Arc<dyn GpuDevice>>,
    pub(crate) live: bool,
    /// Per-attachment color formats this pipeline was built with —
    /// a copy of `PipelineDesc::color_formats`, retained so the render
    /// pass can validate that the bound targets match at encode time.
    /// Drivers construct pipelines with this empty; the `Gpu` wrapper
    /// stamps it from the descriptor.
    pub(crate) color_formats: Vec<Format>,
    /// Depth format this pipeline was built with (a copy of
    /// `PipelineDesc::depth_format`), retained for the same encode-time
    /// pass-shape validation.
    pub(crate) depth_format: Option<Format>,
    /// Rasterization sample count this pipeline was built with (a copy
    /// of `PipelineDesc::sample_count`), retained for the same
    /// encode-time pass-shape validation: every attachment a pass binds
    /// under this pipeline must carry exactly this sample count.
    pub(crate) sample_count: u32,
}

impl Pipeline {
    /// Construct a live pipeline wrapper around a fresh driver handle,
    /// with no device attached and no formats recorded yet. Drivers use
    /// this from `pipeline_create`; the `Gpu` wrapper then attaches the
    /// device and stamps the formats. Centralising the field list here
    /// keeps future `Pipeline` fields off every backend's construction
    /// site.
    pub(crate) fn from_handle(handle: u64) -> Self {
        Self {
            handle,
            device: None,
            live: true,
            color_formats: Vec::new(),
            depth_format: None,
            sample_count: 1,
        }
    }

    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Attach the owning device Arc so Drop can release the handle.
    /// Internal hook for the `quanta-render` sibling crate (drivers
    /// construct pipelines with `device: None`); not part of the
    /// stable public surface.
    #[doc(hidden)]
    pub fn __attach_device(&mut self, device: Arc<dyn GpuDevice>) {
        self.device = Some(device);
    }

    /// Record the color/depth formats and rasterization sample count
    /// the pipeline was built with so the render pass can validate
    /// bound targets against them at encode time. Internal hook for the
    /// `quanta-render` sibling crate (drivers construct pipelines
    /// shape-less); not part of the stable public surface.
    #[doc(hidden)]
    pub fn __set_shape(
        &mut self,
        color_formats: Vec<Format>,
        depth_format: Option<Format>,
        sample_count: u32,
    ) {
        self.color_formats = color_formats;
        self.depth_format = depth_format;
        self.sample_count = sample_count;
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        if self.live {
            self.live = false;
            if let Some(ref dev) = self.device {
                let _ = dev.pipeline_destroy(self.handle);
            }
        }
    }
}

/// Shader input for a render pipeline.
///
/// Replaces the historical `vertex`/`fragment`/`source` tri-state: the
/// enum makes the three supply modes mutually exclusive, and the
/// [`Binaries`](ShaderSource::Binaries) arm lets the driver pick the
/// right per-vendor format so callers never call
/// [`ShaderBinary::for_vendor`] by hand.
///
/// Marked `#[non_exhaustive]`: supply modes can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub enum ShaderSource<'a> {
    /// Separate per-stage payloads, already in the active backend's
    /// native format: MSL source or metallib on Metal, SPIR-V on
    /// Vulkan, WGSL text on WebGPU.
    Stages {
        /// Vertex shader payload.
        vertex: &'a [u8],
        /// Fragment shader payload.
        fragment: &'a [u8],
    },
    /// One combined payload containing both entry points; the driver
    /// finds them by `vertex_entry` / `fragment_entry`. Metal and
    /// WebGPU accept this; Vulkan (SPIR-V modules are per-stage here)
    /// rejects it at create time.
    Combined(&'a [u8]),
    /// Pre-compiled multi-vendor shader binaries — the output of
    /// `#[quanta::vertex]` / `#[quanta::fragment]`. The driver selects
    /// the right format for its vendor (metallib on Apple, SPIR-V on
    /// Vulkan, WGSL on WebGPU).
    Binaries {
        /// Vertex-stage binary.
        vertex: &'a ShaderBinary,
        /// Fragment-stage binary.
        fragment: &'a ShaderBinary,
    },
}

impl<'a> ShaderSource<'a> {
    /// The combined payload, when this source is [`Combined`](Self::Combined).
    pub fn combined(&self) -> Option<&'a [u8]> {
        match self {
            Self::Combined(src) => Some(src),
            _ => None,
        }
    }

    /// Resolve per-stage `(vertex, fragment)` payloads for `vendor`.
    ///
    /// [`Combined`](Self::Combined) yields the same payload for both
    /// stages. [`Binaries`](Self::Binaries) picks the vendor format via
    /// [`ShaderBinary::for_vendor`]; `None` when a stage has no payload
    /// for that vendor.
    pub fn stage_bytes(&self, vendor: Vendor) -> Option<(&'a [u8], &'a [u8])> {
        match self {
            Self::Stages { vertex, fragment } => Some((vertex, fragment)),
            Self::Combined(src) => Some((src, src)),
            Self::Binaries { vertex, fragment } => {
                Some((vertex.for_vendor(vendor)?, fragment.for_vendor(vendor)?))
            }
        }
    }

    /// Resolve per-stage `(vertex, fragment)` WGSL payloads (WebGPU).
    /// `None` when a [`Binaries`](Self::Binaries) stage carries no WGSL.
    pub fn stage_wgsl_bytes(&self) -> Option<(&'a [u8], &'a [u8])> {
        match self {
            Self::Stages { vertex, fragment } => Some((vertex, fragment)),
            Self::Combined(src) => Some((src, src)),
            Self::Binaries { vertex, fragment } => Some((
                vertex.wgsl.map(str::as_bytes)?,
                fragment.wgsl.map(str::as_bytes)?,
            )),
        }
    }
}

/// Describes how to create a render pipeline.
///
/// Marked `#[non_exhaustive]`: fields will be added without a breaking
/// change. Construct with [`PipelineDesc::new`] (or `Default::default()`
/// plus field assignment) and adjust settings through the `with_*`
/// builder methods:
///
/// ```ignore
/// let desc = PipelineDesc::new(ShaderSource::Combined(MSL.as_bytes()))
///     .with_color_formats(vec![Format::RGBA8])
///     .with_cull_mode(CullMode::Back);
/// ```
#[non_exhaustive]
pub struct PipelineDesc<'a> {
    /// Shader payloads (per-stage, combined, or multi-vendor binaries).
    pub shader: ShaderSource<'a>,
    /// Vertex shader entry point name.
    pub vertex_entry: &'a str,
    /// Fragment shader entry point name.
    pub fragment_entry: &'a str,
    /// Vertex buffer layouts.
    pub vertex_layouts: &'a [VertexLayout],
    /// Color attachment formats, **per attachment** (MRT — multiple
    /// render targets): entry `i` types color attachment `i` of *every*
    /// render pass this pipeline is used in. It is **not** a candidate
    /// list of formats the pipeline may be used against — declaring
    /// `[BGRA8, RGBA8]` gives a pipeline with two color attachments (a
    /// BGRA8 one and an RGBA8 one), not one attachment usable as either.
    ///
    /// The count is a contract: it must equal the pass's color-target
    /// count, and format `i` must match bound target `i`. Both are
    /// enforced at encode time (a mismatch fails `pulse()`), and a
    /// descriptor declaring more attachments than the fragment writes is
    /// rejected at creation when a SPIR-V fragment is available (a
    /// metallib-only shader cannot be pre-reflected, so that creation
    /// check is skipped for it). First entry is the primary color format.
    pub color_formats: Vec<Format>,
    /// Depth attachment format (None = no depth buffer).
    pub depth_format: Option<Format>,
    /// MSAA sample count (1 = no MSAA, 4 = 4x MSAA).
    pub sample_count: u32,
    /// Blend state for color attachment (legacy — single attachment).
    pub blend: BlendState,
    /// Per-attachment blend states (M2.5).
    /// One entry per color attachment. If shorter than `color_formats`,
    /// the last entry is reused for remaining attachments.
    /// Empty means use `blend` field for all attachments.
    pub blend_states: Vec<BlendState>,
    /// Face culling mode.
    pub cull_mode: CullMode,
    /// Primitive topology.
    pub primitive: Primitive,
    /// Depth/stencil state.
    pub depth_stencil: DepthStencilState,
    /// Specialization constants (M2.1) — override shader constants at pipeline creation.
    pub specialization: Vec<SpecConstant>,
    /// Tessellation configuration (M4.1). `None` = no tessellation.
    pub tessellation: Option<TessellationDesc>,
    /// Mesh shader configuration (M4.2). `None` = standard vertex pipeline.
    pub mesh_shader: Option<MeshShaderDesc<'a>>,
    /// Enable conservative rasterization (M4.5).
    /// When true, any pixel touched by a primitive is considered covered.
    pub conservative_rasterization: bool,
}

impl<'a> Default for PipelineDesc<'a> {
    fn default() -> Self {
        Self {
            shader: ShaderSource::Stages {
                vertex: &[],
                fragment: &[],
            },
            vertex_entry: "vertex_main",
            fragment_entry: "fragment_main",
            vertex_layouts: &[],
            color_formats: vec![Format::BGRA8],
            depth_format: None,
            sample_count: 1,
            blend: BlendState::PREMULTIPLIED_ALPHA,
            blend_states: Vec::new(),
            cull_mode: CullMode::None,
            primitive: Primitive::Triangle,
            depth_stencil: DepthStencilState::NONE,
            specialization: Vec::new(),
            tessellation: None,
            mesh_shader: None,
            conservative_rasterization: false,
        }
    }
}

impl<'a> PipelineDesc<'a> {
    /// A descriptor with the given shader source and portable defaults
    /// (single `BGRA8` color target, no depth, no blending surprises —
    /// the same defaults as `Default::default()`).
    pub fn new(shader: ShaderSource<'a>) -> Self {
        Self {
            shader,
            ..Default::default()
        }
    }

    /// Set the vertex / fragment entry point names.
    pub fn with_entries(mut self, vertex_entry: &'a str, fragment_entry: &'a str) -> Self {
        self.vertex_entry = vertex_entry;
        self.fragment_entry = fragment_entry;
        self
    }

    /// Set the vertex buffer layouts.
    pub fn with_vertex_layouts(mut self, layouts: &'a [VertexLayout]) -> Self {
        self.vertex_layouts = layouts;
        self
    }

    /// Set the color attachment formats — **one per attachment** (MRT).
    /// Entry `i` types color attachment `i` of every pass this pipeline
    /// is used in; this is not a candidate list. The length must equal
    /// the pass's color-target count (enforced at encode time), and
    /// declaring more attachments than the fragment writes is rejected
    /// at pipeline creation. See [`PipelineDesc::color_formats`].
    pub fn with_color_formats(mut self, formats: Vec<Format>) -> Self {
        self.color_formats = formats;
        self
    }

    /// Set the depth attachment format.
    pub fn with_depth_format(mut self, format: Format) -> Self {
        self.depth_format = Some(format);
        self
    }

    /// Set the MSAA sample count.
    pub fn with_sample_count(mut self, samples: u32) -> Self {
        self.sample_count = samples;
        self
    }

    /// Set the blend state applied to all color attachments.
    pub fn with_blend(mut self, blend: BlendState) -> Self {
        self.blend = blend;
        self
    }

    /// Set per-attachment blend states.
    pub fn with_blend_states(mut self, states: Vec<BlendState>) -> Self {
        self.blend_states = states;
        self
    }

    /// Set the face culling mode.
    pub fn with_cull_mode(mut self, cull_mode: CullMode) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Set the primitive topology.
    pub fn with_primitive(mut self, primitive: Primitive) -> Self {
        self.primitive = primitive;
        self
    }

    /// Set the depth/stencil state.
    pub fn with_depth_stencil(mut self, state: DepthStencilState) -> Self {
        self.depth_stencil = state;
        self
    }

    /// Set the specialization constants.
    pub fn with_specialization(mut self, constants: Vec<SpecConstant>) -> Self {
        self.specialization = constants;
        self
    }

    /// Enable tessellation with the given configuration.
    pub fn with_tessellation(mut self, tessellation: TessellationDesc) -> Self {
        self.tessellation = Some(tessellation);
        self
    }

    /// Use a mesh-shader pipeline instead of the vertex stage.
    pub fn with_mesh_shader(mut self, mesh_shader: MeshShaderDesc<'a>) -> Self {
        self.mesh_shader = Some(mesh_shader);
        self
    }

    /// Enable or disable conservative rasterization.
    pub fn with_conservative_rasterization(mut self, enabled: bool) -> Self {
        self.conservative_rasterization = enabled;
        self
    }
}

/// Reflect a fragment shader's declared color-output count from its
/// SPIR-V and reject a descriptor that declares **more** color
/// attachments than the fragment writes.
///
/// `color_formats[i]` types color attachment `i`, so a descriptor
/// declaring `N` attachments over a fragment that writes only `M < N`
/// has a phantom attachment `N-1 … M` that no shader output feeds — the
/// exact mistake a consumer makes by reading `color_formats` as a list
/// of formats the pipeline "may be used against". Rejected at creation
/// with a `CompilationFailed` naming both counts.
///
/// Only fires when a SPIR-V fragment payload is present (the
/// `ShaderSource::Binaries` case with a `spirv` binary). metallib-only
/// shaders cannot be pre-reflected, so they skip the check — see the
/// note on [`PipelineDesc::color_formats`].
///
/// `N < M` (writing **fewer** attachments than the fragment declares)
/// is allowed: it is the driver-legal partial-write case.
///
/// Internal hook for the `quanta-render` sibling crate, called at the
/// API layer before the driver `pipeline_create`; not part of the
/// stable public surface.
#[cfg(feature = "render")]
#[doc(hidden)]
pub fn __check_fragment_outputs(
    fragment_spirv: Option<&[u8]>,
    declared_color_count: usize,
) -> Result<(), crate::QuantaError> {
    let Some(bytes) = fragment_spirv else {
        return Ok(()); // metallib-only or no SPIR-V — cannot pre-reflect.
    };
    // SPIR-V is a little-endian u32 stream; a non-multiple-of-4 length
    // is not a module we can read, so skip rather than guess.
    if bytes.len() % 4 != 0 {
        return Ok(());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let Some(writes) = crate::driver::spirv_meta::fragment_output_count(&words) else {
        return Ok(()); // not a readable fragment module — skip.
    };
    if declared_color_count > writes {
        return Err(crate::QuantaError::compilation_failed(alloc::format!(
            "pipeline declares {declared_color_count} color attachments; fragment \
             writes {writes}"
        )));
    }
    Ok(())
}

/// Depth and stencil testing configuration.
///
/// Marked `#[non_exhaustive]`: fields will be added. Start from one of
/// the presets ([`NONE`](Self::NONE), [`DEPTH_LESS`](Self::DEPTH_LESS),
/// [`DEPTH_READ_ONLY`](Self::DEPTH_READ_ONLY)) and adjust fields by
/// assignment.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct DepthStencilState {
    /// Enable depth testing.
    pub depth_test: bool,
    /// Enable depth writing.
    pub depth_write: bool,
    /// Depth comparison function.
    pub depth_compare: CompareOp,
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
        depth_compare: CompareOp::Always,
        stencil_front: None,
        stencil_back: None,
    };

    /// Standard 3D depth testing — closer fragments win.
    pub const DEPTH_LESS: Self = Self {
        depth_test: true,
        depth_write: true,
        depth_compare: CompareOp::Less,
        stencil_front: None,
        stencil_back: None,
    };

    /// Depth test without writing — for transparent objects after opaques.
    pub const DEPTH_READ_ONLY: Self = Self {
        depth_test: true,
        depth_write: false,
        depth_compare: CompareOp::Less,
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
    pub compare: CompareOp,
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
            compare: CompareOp::Always,
            read_mask: 0xFF,
            write_mask: 0xFF,
        }
    }
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
///
/// Marked `#[non_exhaustive]`: formats can be added — match with a
/// wildcard arm.
#[non_exhaustive]
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
///
/// Marked `#[non_exhaustive]`: fields will be added. Start from one of
/// the presets ([`NONE`](Self::NONE), [`PREMULTIPLIED_ALPHA`](Self::PREMULTIPLIED_ALPHA),
/// [`ALPHA`](Self::ALPHA), [`ADDITIVE`](Self::ADDITIVE)) and adjust
/// fields by assignment.
#[non_exhaustive]
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

/// Blend factor — scales source or destination color/alpha.
///
/// Marked `#[non_exhaustive]`: factors can be added — match with a
/// wildcard arm.
#[non_exhaustive]
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

/// Blend operation — combines the scaled source and destination.
///
/// Marked `#[non_exhaustive]`: operations can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

/// Face culling mode.
///
/// Marked `#[non_exhaustive]`: modes can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

/// Primitive topology.
///
/// Marked `#[non_exhaustive]`: topologies can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Point,
    Line,
    LineStrip,
    Triangle,
    TriangleStrip,
}

// === M2.1: Specialization Constants ===

/// A named constant that can be overridden at pipeline creation time.
///
/// Allows a single compiled shader to behave differently depending on
/// compile-time constants (e.g. tile size, feature toggles), avoiding
/// the cost of maintaining multiple shader variants.
pub struct SpecConstant {
    /// Name matching the constant in the shader source.
    pub name: String,
    /// The value to substitute.
    pub value: SpecValue,
}

/// Specialization constant value.
#[derive(Debug, Clone, Copy)]
pub enum SpecValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
}

// === M4.1: Tessellation ===

/// Tessellation stage configuration.
pub struct TessellationDesc {
    /// Number of control points per patch.
    pub patch_size: u32,
    /// How the tessellator subdivides edges.
    pub spacing: TessSpacing,
    /// Triangle winding order for generated primitives.
    pub winding: TessWinding,
}

/// Tessellation edge subdivision mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessSpacing {
    /// Integer subdivision levels.
    Equal,
    /// Smooth transitions using fractional odd levels.
    FractionalOdd,
    /// Smooth transitions using fractional even levels.
    FractionalEven,
}

/// Winding order for tessellation-generated triangles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TessWinding {
    /// Clockwise.
    Cw,
    /// Counter-clockwise.
    Ccw,
}

// === M4.2: Mesh Shaders ===

/// Mesh shader pipeline configuration.
///
/// Replaces the traditional vertex input assembly with a programmable
/// mesh generation stage. The task shader (optional) does coarse culling
/// and launches mesh shader threadgroups; the mesh shader emits vertices
/// and primitives directly.
pub struct MeshShaderDesc<'a> {
    /// Task (amplification) shader binary. `None` = no task shader.
    pub task_shader: Option<&'a [u8]>,
    /// Mesh shader binary (required).
    pub mesh_shader: &'a [u8],
    /// Maximum vertices the mesh shader can emit per threadgroup.
    pub max_vertices: u32,
    /// Maximum primitives the mesh shader can emit per threadgroup.
    pub max_primitives: u32,
}

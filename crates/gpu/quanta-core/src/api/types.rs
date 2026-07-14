use alloc::string::String;

/// Pixel/data format for textures.
///
/// Marked `#[non_exhaustive]`: formats will be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    RGBA8,
    BGRA8,
    R8,
    R16Float,
    R32Float,
    RG32Float,
    RGBA16Float,
    RGBA32Float,
    Depth32Float,

    // Compressed formats (M2.4)
    /// BC1 (DXT1) with alpha — 8:1 compression, 4x4 blocks.
    Bc1Rgba,
    /// BC3 (DXT5) — 4:1 compression with full alpha, 4x4 blocks.
    Bc3Rgba,
    /// BC5 (2-channel) — normal maps, 4x4 blocks.
    Bc5Rg,
    /// BC7 — high quality 4:1, all channel configs, 4x4 blocks.
    Bc7Rgba,
    /// ASTC 4x4 — mobile/desktop, high quality.
    Astc4x4,
    /// ASTC 6x6 — higher compression, lower quality.
    Astc6x6,
    /// ASTC 8x8 — maximum compression.
    Astc8x8,
    /// ETC2 RGB8 — mobile baseline.
    Etc2Rgb8,
    /// ETC2 RGBA8 — mobile baseline with alpha.
    Etc2Rgba8,
}

impl Format {
    /// Bytes per pixel for this format.
    ///
    /// For block-compressed formats this returns the average bytes per pixel
    /// (block size / pixels-per-block), which is useful for estimating
    /// uncompressed buffer sizes. Exact storage requires block math.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::R8 => 1,
            Self::R16Float => 2,
            Self::RGBA8 | Self::BGRA8 | Self::R32Float | Self::Depth32Float => 4,
            Self::RG32Float => 8,
            Self::RGBA16Float => 8,
            Self::RGBA32Float => 16,
            // Block-compressed: report the block size in bytes (not per-pixel).
            // BC1/ETC2-RGB: 8 bytes per 4x4 block = 0.5 bpp, round up to 1.
            Self::Bc1Rgba | Self::Etc2Rgb8 => 1,
            // BC3/BC5/BC7/ETC2-RGBA: 16 bytes per 4x4 block = 1 bpp.
            Self::Bc3Rgba | Self::Bc5Rg | Self::Bc7Rgba | Self::Etc2Rgba8 => 1,
            // ASTC: 16 bytes per block regardless of block size.
            Self::Astc4x4 => 1,
            Self::Astc6x6 => 1,
            Self::Astc8x8 => 1,
        }
    }
}

/// How a field will be used. Drivers optimize placement based on usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldUsage(u8);

impl FieldUsage {
    /// Field will be read by GPU (compute input, vertex data).
    pub const READ: Self = Self(1 << 0);
    /// Field will be written by GPU (compute output, render target).
    pub const WRITE: Self = Self(1 << 1);
    /// Field will be used in compute dispatches.
    pub const COMPUTE: Self = Self(1 << 2);
    /// Field will be used as vertex/index data in render.
    pub const RENDER: Self = Self(1 << 3);
    /// Field will be transferred to/from CPU.
    pub const TRANSFER: Self = Self(1 << 4);
    /// Field will be used as a uniform buffer.
    pub const UNIFORM: Self = Self(1 << 5);

    /// Combine usage flags.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Default: read + write + compute + transfer.
    pub const fn default_compute() -> Self {
        Self(Self::READ.0 | Self::WRITE.0 | Self::COMPUTE.0 | Self::TRANSFER.0)
    }

    /// Default: read + render + transfer.
    pub const fn default_render() -> Self {
        Self(Self::READ.0 | Self::RENDER.0 | Self::TRANSFER.0)
    }

    /// Default: uniform buffer (read + uniform + transfer).
    pub const fn default_uniform() -> Self {
        Self(Self::READ.0 | Self::UNIFORM.0 | Self::TRANSFER.0)
    }

    /// Check if a usage flag is set.
    pub const fn has(self, flag: Self) -> bool {
        self.0 & flag.0 == flag.0
    }

    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Color value (linear, 0.0-1.0).
#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const CLEAR: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }
}

/// GPU device capabilities.
///
/// Marked `#[non_exhaustive]`: capability fields will be added without a
/// breaking change. Constructed by drivers; consumers read the fields.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Caps {
    /// Number of compute units (SM on NVIDIA, CU on AMD).
    pub nuclei: u32,
    /// Cores per compute unit.
    pub protons_per_nucleus: u32,
    /// Threads per core (warp/wave width: 32 or 64).
    pub quarks_per_proton: u32,
    /// Total GPU memory in bytes.
    pub memory_bytes: u64,
    /// Maximum quarks per dispatch.
    pub max_quarks_per_dispatch: u32,
    /// Maximum work groups per dimension.
    pub max_groups: [u32; 3],
    /// GPU vendor.
    pub vendor: Vendor,
    /// Device name (for diagnostics).
    pub name: String,
}

impl Caps {
    /// Total parallel execution units.
    pub fn total_quarks(&self) -> u32 {
        self.nuclei * self.protons_per_nucleus * self.quarks_per_proton
    }
}

/// GPU vendor — determines which kernel binary format to use.
///
/// Marked `#[non_exhaustive]`: vendors can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Amd,
    Nvidia,
    Intel,
    Apple,
    Broadcom,
    Software,
    Unknown,
}

/// Resource state for barrier transitions.
///
/// Tracks how a resource is being used so the driver can insert
/// the correct synchronization between pipeline stages.
///
/// Marked `#[non_exhaustive]`: states can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// General-purpose layout (suboptimal but always valid).
    General,
    /// Written by a compute shader.
    ComputeWrite,
    /// Read by a compute shader.
    ComputeRead,
    /// Used as a render target (color attachment).
    RenderTarget,
    /// Used as a depth/stencil attachment.
    DepthStencil,
    /// Read by a shader (sampled image or storage buffer read).
    ShaderRead,
    /// Source of a transfer/copy operation.
    TransferSrc,
    /// Destination of a transfer/copy operation.
    TransferDst,
    /// Ready for presentation to a swapchain.
    Present,
}

/// Load operation for a render target attachment.
///
/// Marked `#[non_exhaustive]`: operations can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum LoadOp {
    /// Clear the attachment to a specific color at the start.
    Clear(Color),
    /// Preserve existing contents.
    Load,
    /// Contents are undefined — driver may optimize away a load.
    DontCare,
}

/// Store operation for a render target attachment.
///
/// Marked `#[non_exhaustive]`: operations can be added — match with a
/// wildcard arm. Construct the MSAA-resolve variant with
/// [`StoreOp::resolve`], which derives the target from a typed
/// [`Texture`](crate::Texture) instead of a raw handle.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum StoreOp {
    /// Write results back to memory.
    Store,
    /// Results are not needed — driver may discard.
    DontCare,
    /// Resolve MSAA samples to the target texture. Construct with
    /// [`StoreOp::resolve`].
    Resolve(ResolveTarget),
}

impl StoreOp {
    /// Resolve MSAA samples into `target` at the end of the pass.
    ///
    /// `target` must be a single-sample texture with the same
    /// dimensions and format as the MSAA attachment.
    pub fn resolve(target: &crate::Texture) -> Self {
        Self::Resolve(ResolveTarget(target.handle()))
    }
}

/// Opaque MSAA-resolve destination carried by [`StoreOp::Resolve`].
/// Obtained through [`StoreOp::resolve`]; wraps the driver handle so
/// callers never pass raw `u64`s.
#[derive(Debug, Clone, Copy)]
pub struct ResolveTarget(pub(crate) u64);

impl ResolveTarget {
    /// The driver handle of the resolve destination (read-only).
    pub fn handle(&self) -> u64 {
        self.0
    }
}

/// Comparison operation for depth/stencil testing and comparison samplers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

/// Kernel binary format — compiled output from #[quanta::kernel].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFormat {
    /// AMD GCN binary (compiled via LLVM amdgcn backend).
    AmdGcn,
    /// NVIDIA PTX text (compiled via LLVM nvptx64 backend).
    NvidiaPtx,
    /// Metal Shading Language source (generated for Apple GPUs).
    Msl,
    /// WebGPU Shading Language source (generated for browsers).
    Wgsl,
    /// Platform-agnostic IR (fallback — driver compiles at load time).
    LlvmIr,
}

// === M2.2: Format Capability Queries ===

/// What a given format supports on this device.
///
/// Marked `#[non_exhaustive]`: capability fields will be added.
/// Constructed by drivers; consumers read the fields.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct FormatCaps {
    /// Can be used with a filtering sampler (linear/mip).
    pub filterable: bool,
    /// Can be used as a color render target.
    pub renderable: bool,
    /// Can be used as a storage texture (read-write from shaders).
    pub storage: bool,
    /// Supports blending when used as a render target.
    pub blendable: bool,
    /// Supports multisampled rendering.
    pub msaa: bool,
    /// Can be used as a depth/stencil attachment.
    pub depth: bool,
}

// === M3.1: Multi-Queue ===

/// GPU queue family type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    /// Full graphics + compute + transfer.
    Graphics,
    /// Compute + transfer only (async compute).
    Compute,
    /// Transfer/DMA only.
    Transfer,
}

/// Describes one family of queues on the device.
///
/// Marked `#[non_exhaustive]`: fields will be added. Constructed by
/// drivers; consumers read the fields.
#[non_exhaustive]
pub struct QueueFamily {
    /// What kind of work this family can execute.
    pub queue_type: QueueType,
    /// Number of queues available in this family.
    pub count: u32,
}

// ── Binding-model capacity constants ────────────────────────────────────
//
// Shared compute/render plumbing: `Wave` (compute face) sizes its inline
// binding/push-constant state with these, and the drivers' indirect
// command buffers (including the render-path bundles) allocate their
// per-command binding tables from the same caps. They live here — not in
// the compute-gated `wave` module — so a render-only build still sees
// the device binding model.

/// Maximum buffer binding slots per dispatch/draw command.
#[allow(dead_code)] // read by the compute face and the metal/vulkan ICB paths
pub(crate) const MAX_BINDINGS: usize = 16;
/// Maximum texture binding slots per dispatch command.
#[allow(dead_code)] // read by the compute face and the webgpu driver only
pub(crate) const MAX_TEXTURES: usize = 16;
/// Inline push constant buffer size in bytes.
#[allow(dead_code)] // read by the compute face and the cpu/webgpu drivers only
pub(crate) const PUSH_DATA_CAP: usize = 256;

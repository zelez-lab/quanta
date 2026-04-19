/// Pixel/data format for textures.
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

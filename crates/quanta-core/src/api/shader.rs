//! Render-stage shader types.
//!
//! `ShaderStage` and `ShaderBinary` describe programmable *render*
//! pipeline stages (vertex, fragment, tessellation, mesh, ray tracing)
//! — they live on the render surface, not in the compute kernel module.
//! Compute kernels compile to [`KernelBinary`](crate::KernelBinary)
//! instead.

/// Shader stage — which programmable pipeline stage this shader runs in.
///
/// Marked `#[non_exhaustive]`: stages can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    /// Tessellation control (hull) shader.
    TessControl,
    /// Tessellation evaluation (domain) shader.
    TessEval,
    /// Task (amplification) shader — launches mesh shader threadgroups.
    Task,
    /// Mesh shader — generates vertices and primitives.
    Mesh,
    /// Ray generation shader.
    RayGen,
    /// Closest-hit shader.
    ClosestHit,
    /// Miss shader.
    Miss,
}

/// A compiled shader binary — output of `#[quanta::vertex]` or `#[quanta::fragment]`.
///
/// Contains pre-compiled binaries for each supported GPU vendor.
/// The driver selects the appropriate binary at pipeline creation time.
pub struct ShaderBinary {
    /// Pre-compiled SPIR-V binary.
    pub spirv: Option<&'static [u8]>,
    /// Pre-compiled Metal library binary.
    pub metallib: Option<&'static [u8]>,
    /// WGSL source for WebGPU.
    pub wgsl: Option<&'static str>,
    /// Shader entry point name.
    pub entry_point: &'static str,
    /// Shader stage (vertex or fragment).
    pub stage: ShaderStage,
}

impl ShaderBinary {
    /// Select the best shader binary for the given vendor.
    ///
    /// Apple: metallib binary. All others: SPIR-V binary.
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Apple => self.metallib.or(self.spirv),
            _ => self.spirv,
        }
    }
}

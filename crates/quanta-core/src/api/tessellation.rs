//! Typed tessellation pipelines (steps 022 + 023).
//!
//! Tessellation lets the GPU subdivide a coarse "patch" of control
//! points into a finer mesh of triangles via tessellation factors.
//! Backends:
//!
//! - Metal: no hardware tessellator. A compute kernel fills a
//!   per-patch factor `MTLBuffer`, then the post-tessellation
//!   vertex shader runs via `drawIndexedPatches:`.
//! - Vulkan: native TCS + TES stages, gated on the
//!   `tessellationShader` device feature (core in Vulkan 1.0).
//! - WebGPU: not in the W3C spec — `NotSupported` at create time.
//! - CPU: software model — the source of truth for the proof contract.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.Tessellation.Pipeline` (Lean) and
//! `quanta-api/tessellation_safety.rs` (Verus):
//!
//! - `set_outer(i, _)` / `set_inner(i, _)` fail when `i` is out of
//!   range for the topology, or the pipeline is destroyed.
//! - `Drop` calls `tessellation_destroy` exactly once.
//!
//! Stored factors are clamped into `[1, MAX_TESS_LEVEL]` (matches
//! Vulkan `maxTessellationGenerationLevel` / Metal
//! `maxTessellationFactor`).

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError};

/// Maximum tessellation factor any axis can request. Matches Vulkan
/// `maxTessellationGenerationLevel` and Metal `maxTessellationFactor`.
pub const MAX_TESS_LEVEL: u32 = 64;

/// Maximum control-point count per patch. Matches Vulkan / Metal / D3D12.
pub const MAX_PATCH_SIZE: u32 = 32;

/// Patch topology — triangle (3 outer + 1 inner) or quad (4 outer + 2
/// inner). Isolines are not modeled; Metal does not support them and
/// they are vanishingly rare in modern pipelines.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TessTopology {
    Triangle,
    Quad,
}

impl TessTopology {
    /// Number of outer (edge) tessellation factors for this topology.
    pub fn outer_count(self) -> u32 {
        match self {
            TessTopology::Triangle => 3,
            TessTopology::Quad => 4,
        }
    }

    /// Number of inner (interior) tessellation factors.
    pub fn inner_count(self) -> u32 {
        match self {
            TessTopology::Triangle => 1,
            TessTopology::Quad => 2,
        }
    }
}

/// Clamp a candidate factor into `[1, MAX_TESS_LEVEL]`.
///
/// Refines `Quanta.Tessellation.clampFactor`.
pub(crate) fn clamp_factor(f: u32) -> u32 {
    f.clamp(1, MAX_TESS_LEVEL)
}

/// A typed tessellation pipeline state — fixed topology and patch
/// size, mutable inner / outer factors. Drop releases the backend
/// handle.
pub struct TessellationPipeline {
    pub(crate) handle: u64,
    pub(crate) topology: TessTopology,
    pub(crate) control_points: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl TessellationPipeline {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Patch topology this pipeline was created for.
    pub fn topology(&self) -> TessTopology {
        self.topology
    }

    /// Number of control points per patch (immutable after create).
    pub fn control_points(&self) -> u32 {
        self.control_points
    }

    /// Update the outer (edge) tessellation factor at `index`.
    ///
    /// Returns `Err(InvalidParam)` if `index >= topology.outer_count()`
    /// or the pipeline has been destroyed. The factor is clamped into
    /// `[1, MAX_TESS_LEVEL]` before being stored — matches Vulkan and
    /// Metal hardware bounds.
    ///
    /// Refines `Quanta.Tessellation.Pipeline.setOuter` and the Verus
    /// theorem `t7252_set_outer_localizes`.
    pub fn set_outer(&self, index: u32, factor: u32) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param(
                "tessellation pipeline is not live",
            ));
        }
        if index >= self.topology.outer_count() {
            return Err(QuantaError::invalid_param(
                "tessellation outer index >= topology.outer_count()",
            ));
        }
        self.device
            .tessellation_set_outer(self.handle, index, clamp_factor(factor))
    }

    /// Update the inner (interior) tessellation factor at `index`.
    pub fn set_inner(&self, index: u32, factor: u32) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param(
                "tessellation pipeline is not live",
            ));
        }
        if index >= self.topology.inner_count() {
            return Err(QuantaError::invalid_param(
                "tessellation inner index >= topology.inner_count()",
            ));
        }
        self.device
            .tessellation_set_inner(self.handle, index, clamp_factor(factor))
    }
}

impl Drop for TessellationPipeline {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.tessellation_destroy(self.handle);
            self.live = false;
        }
    }
}

//! Ray tracing acceleration structures + pipelines (steps 026 + 027).
//!
//! Backends:
//!
//! - Metal: `MTLAccelerationStructureDescriptor` + intersector tables
//!   invoked from a compute kernel.
//! - Vulkan: `VK_KHR_acceleration_structure` +
//!   `VK_KHR_ray_tracing_pipeline`; `vkCmdTraceRaysKHR(width, height, 1)`.
//! - WebGPU: not in W3C — `NotSupported`.
//! - CPU: software lifecycle only.
//!
//! The wrappers enforce the lifecycle proven in
//! `Quanta.RayTracing.{AccelerationStructure, Pipeline}` (Lean) and
//! `quanta-api/ray_tracing_safety.rs` (Verus):
//!
//! - `dispatch_rays(w, h)` fails when any dimension exceeds
//!   `MAX_DISPATCH_DIM` or the pipeline is destroyed.
//! - `Drop` calls the matching destroy method exactly once.

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError};

/// Maximum ray recursion depth — Vulkan + Metal hardware-minimum.
pub const MAX_RECURSION_DEPTH: u32 = 31;

/// Maximum dispatch dimension (width or height).
pub const MAX_DISPATCH_DIM: u32 = 65535;

/// Acceleration-structure tier.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AsKind {
    /// Bottom-level: BVH over user geometry.
    Bottom,
    /// Top-level: BVH over BLAS instances.
    Top,
}

/// Geometry descriptor for building an acceleration structure.
#[derive(Copy, Clone, Debug)]
pub struct GeometryDesc {
    /// Field handle containing vertex data.
    pub vertices: u64,
    /// Field handle containing index data (`None` = non-indexed).
    pub indices: Option<u64>,
    /// Number of vertices.
    pub vertex_count: u32,
    /// Number of indices (0 if non-indexed).
    pub index_count: u32,
    /// Byte stride between consecutive vertices.
    pub vertex_stride: u32,
}

/// A typed acceleration structure (BLAS or TLAS). Drop releases the
/// backend handle.
///
/// Refines `Quanta.RayTracing.AccelerationStructure`.
pub struct AccelerationStructure {
    pub handle: u64,
    pub(crate) kind: AsKind,
    pub(crate) geom_count: u32,
    pub(crate) device: Option<Arc<dyn GpuDevice>>,
    pub(crate) live: bool,
}

impl AccelerationStructure {
    /// Tier (BLAS or TLAS).
    pub fn kind(&self) -> AsKind {
        self.kind
    }

    /// Number of geometries (BLAS) or instances (TLAS).
    pub fn geom_count(&self) -> u32 {
        self.geom_count
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        if self.live
            && let Some(device) = self.device.as_ref()
        {
            let _ = device.destroy_acceleration_structure(self.handle);
            self.live = false;
        }
    }
}

/// Describes how to create a ray tracing pipeline.
///
/// Ray tracing pipelines contain three required shader stages:
/// - **ray_gen**: launched once per pixel, fires rays
/// - **closest_hit**: invoked when a ray hits the nearest surface
/// - **miss**: invoked when a ray hits nothing
pub struct RayTracingPipelineDesc<'a> {
    /// Ray generation shader binary.
    pub ray_gen: &'a [u8],
    /// Closest-hit shader binary.
    pub closest_hit: &'a [u8],
    /// Miss shader binary.
    pub miss: &'a [u8],
    /// Maximum ray recursion depth (e.g. reflections bouncing).
    pub max_recursion: u32,
}

/// A typed ray tracing pipeline. Drop releases the backend handle.
///
/// Refines `Quanta.RayTracing.Pipeline`.
pub struct RayTracingPipeline {
    pub(crate) handle: u64,
    pub(crate) max_recursion: u32,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl RayTracingPipeline {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Maximum ray recursion depth this pipeline was created with.
    pub fn max_recursion(&self) -> u32 {
        self.max_recursion
    }

    /// Trace `width × height` rays through this pipeline.
    ///
    /// Returns `Err(InvalidParam)` if either dimension exceeds
    /// `MAX_DISPATCH_DIM` or the pipeline has been destroyed.
    /// Refines `Quanta.RayTracing.Pipeline.dispatch` and the Verus
    /// theorem `t7452_dispatch_appends`.
    pub fn dispatch_rays(&self, width: u32, height: u32) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param(
                "ray tracing pipeline is not live",
            ));
        }
        if width > MAX_DISPATCH_DIM || height > MAX_DISPATCH_DIM {
            return Err(QuantaError::invalid_param(
                "dispatch_rays dimension exceeds MAX_DISPATCH_DIM",
            ));
        }
        self.device.dispatch_rays(self.handle, width, height)
    }
}

impl Drop for RayTracingPipeline {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.destroy_ray_tracing_pipeline(self.handle);
            self.live = false;
        }
    }
}

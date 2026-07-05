//! Ray tracing data model (steps 026 + 027).
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
//! The descriptors here are what the [`GpuDevice`](crate::GpuDevice)
//! trait and the drivers speak (`build_acceleration_structure`,
//! `create_ray_tracing_pipeline`). The typed wrappers —
//! `AccelerationStructure` / `RayTracingPipeline`, whose lifecycles are
//! proven in `Quanta.RayTracing.{AccelerationStructure, Pipeline}`
//! (Lean) and `quanta-api/ray_tracing_safety.rs` (Verus) — live in the
//! `quanta-render` crate.

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

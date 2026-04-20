//! Ray tracing acceleration structures and pipeline types (M4.3).

/// GPU-resident acceleration structure (BVH) for ray-scene intersection.
///
/// Built from geometry descriptors; the driver constructs an optimized
/// bounding volume hierarchy. Bind to a ray tracing pipeline before
/// dispatching rays.
pub struct AccelerationStructure {
    pub handle: u64,
}

/// Describes how to create a ray tracing pipeline.
///
/// Ray tracing pipelines contain three shader stages:
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

/// Geometry descriptor for building an acceleration structure.
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

//! Render pipeline and render pass methods on Gpu.

use alloc::vec::Vec;

use crate::ray_tracing::{GeometryDesc, RayTracingPipelineDesc};
use crate::{OcclusionQuery, Pipeline, PipelineDesc, QuantaError, RenderBuilder, Texture};

use super::Gpu;

impl Gpu {
    // === Render ===

    /// Create a render pipeline from a descriptor.
    pub fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError> {
        self.inner.pipeline_create(desc)
    }

    /// Allocate a render-path Indirect Command Buffer
    /// ([`IndirectRenderBundle`](crate::IndirectRenderBundle)) with
    /// the given capacity.
    ///
    /// Backends with no native render-bundle support return an error
    /// at create time. Steps 032 + 033, render path.
    pub fn render_bundle(
        &self,
        max_commands: u32,
    ) -> Result<crate::IndirectRenderBundle, QuantaError> {
        let handle = self.inner.render_bundle_create(max_commands)?;
        Ok(crate::IndirectRenderBundle {
            handle,
            cap: max_commands,
            recorded: 0,
            device: self.inner.clone(),
            live: true,
        })
    }

    /// Allocate a typed
    /// [`MeshPipeline`](crate::MeshPipeline) for the given mesh
    /// pipeline descriptor. Steps 024 + 025.
    ///
    /// Backends without mesh-shader support (WebGPU, older Metal,
    /// pre-1.3 Vulkan without VK_EXT_mesh_shader) return
    /// `NotSupported` here so user code can branch.
    pub fn mesh_pipeline(
        &self,
        desc: crate::MeshPipelineDesc,
    ) -> Result<crate::MeshPipeline, QuantaError> {
        if !(1..=crate::MAX_MESH_VERTICES).contains(&desc.max_vertices_per_meshlet) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline max_vertices_per_meshlet out of range",
            ));
        }
        if !(1..=crate::MAX_MESH_PRIMITIVES).contains(&desc.max_primitives_per_meshlet) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline max_primitives_per_meshlet out of range",
            ));
        }
        if !(1..=crate::MAX_TASK_THREADS).contains(&desc.task_threads_per_group) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline task_threads_per_group out of range",
            ));
        }
        let handle = self.inner.mesh_pipeline_create(
            desc.max_vertices_per_meshlet,
            desc.max_primitives_per_meshlet,
            desc.task_threads_per_group,
        )?;
        Ok(crate::MeshPipeline {
            handle,
            desc,
            device: self.inner.clone(),
            live: true,
        })
    }

    /// Allocate a typed
    /// [`TessellationPipeline`](crate::TessellationPipeline) for the
    /// given patch topology and control-point count. Steps 022 + 023.
    ///
    /// Backends without tessellation (WebGPU, CPU-only fallbacks
    /// missing the feature) return `NotSupported` here so user code
    /// can branch.
    pub fn tessellation_pipeline(
        &self,
        topology: crate::TessTopology,
        control_points: u32,
    ) -> Result<crate::TessellationPipeline, QuantaError> {
        if !(1..=crate::MAX_PATCH_SIZE).contains(&control_points) {
            return Err(QuantaError::invalid_param(
                "tessellation control_points must be in [1, MAX_PATCH_SIZE]",
            ));
        }
        let topo_byte: u8 = match topology {
            crate::TessTopology::Triangle => 0,
            crate::TessTopology::Quad => 1,
        };
        let handle = self
            .inner
            .tessellation_pipeline_create(topo_byte, control_points)?;
        Ok(crate::TessellationPipeline {
            handle,
            topology,
            control_points,
            device: self.inner.clone(),
            live: true,
        })
    }

    /// Begin a chainable render pass targeting a texture.
    ///
    /// Returns a `RenderBuilder` that records draw commands via method
    /// chaining and submits them with `.pulse()`.
    ///
    /// ```ignore
    /// let mut pulse = gpu.render(&target)?
    ///     .clear(Color::BLACK)
    ///     .pipeline(&pipeline)
    ///     .vertices(0, &verts)
    ///     .draw(3)
    ///     .pulse()?;
    /// gpu.wait(&mut pulse)?;
    /// ```
    pub fn render(&self, target: &Texture) -> Result<RenderBuilder, QuantaError> {
        let pass = self.inner.render_begin(target)?;
        Ok(RenderBuilder::new(self.inner.clone(), pass))
    }

    // === M3.3: Occlusion queries ===

    /// Create an occlusion query set with `count` slots.
    pub fn occlusion_query_create(&self, count: u32) -> Result<OcclusionQuery, QuantaError> {
        let handle = self.inner.occlusion_query_create(count)?;
        Ok(OcclusionQuery { handle, count })
    }

    /// Read results from an occlusion query set (fragment counts per slot).
    pub fn occlusion_query_read(&self, query: &OcclusionQuery) -> Result<Vec<u64>, QuantaError> {
        self.inner.occlusion_query_read(query.handle)
    }

    // === M4.3: Ray tracing ===

    /// Build a bottom-level acceleration structure from geometry.
    /// Raw-handle API; prefer `acceleration_structure_blas` for the
    /// Drop-safe typed wrapper.
    pub fn build_acceleration_structure(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<u64, QuantaError> {
        self.inner.build_acceleration_structure(geometry)
    }

    /// Build a typed bottom-level
    /// [`AccelerationStructure`](crate::AccelerationStructure)
    /// (BLAS) from geometry. Steps 026 + 027.
    ///
    /// Returns `Err(NotSupported)` when the backend does not
    /// implement ray tracing (WebGPU, devices without
    /// `VK_KHR_acceleration_structure`).
    pub fn acceleration_structure_blas(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<crate::AccelerationStructure, QuantaError> {
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }
        let handle = self.inner.build_acceleration_structure(geometry)?;
        Ok(crate::AccelerationStructure {
            handle,
            kind: crate::AsKind::Bottom,
            geom_count: geometry.len() as u32,
            device: Some(self.inner.clone()),
            live: true,
        })
    }

    /// Create a ray tracing pipeline from shader stages. Raw-handle
    /// API; prefer `ray_tracing_pipeline` for the Drop-safe typed
    /// wrapper.
    pub fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        self.inner.create_ray_tracing_pipeline(desc)
    }

    /// Allocate a typed
    /// [`RayTracingPipeline`](crate::RayTracingPipeline) from the
    /// given descriptor. Steps 026 + 027.
    ///
    /// Bounds-checks `max_recursion` against the proven
    /// hardware-minimum cap before dispatching to the backend.
    pub fn ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<crate::RayTracingPipeline, QuantaError> {
        if desc.max_recursion > crate::MAX_RECURSION_DEPTH {
            return Err(QuantaError::invalid_param(
                "max_recursion exceeds MAX_RECURSION_DEPTH",
            ));
        }
        let handle = self.inner.create_ray_tracing_pipeline(desc)?;
        Ok(crate::RayTracingPipeline {
            handle,
            max_recursion: desc.max_recursion,
            device: self.inner.clone(),
            live: true,
        })
    }

    /// Dispatch rays through a ray tracing pipeline (raw-handle API).
    pub fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError> {
        self.inner.dispatch_rays(pipeline, width, height)
    }

    /// Destroy an acceleration structure (raw-handle API).
    pub fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.destroy_acceleration_structure(handle)
    }
}

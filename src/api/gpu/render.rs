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
    pub fn build_acceleration_structure(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<u64, QuantaError> {
        self.inner.build_acceleration_structure(geometry)
    }

    /// Create a ray tracing pipeline from shader stages.
    pub fn create_ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<u64, QuantaError> {
        self.inner.create_ray_tracing_pipeline(desc)
    }

    /// Dispatch rays through a ray tracing pipeline.
    pub fn dispatch_rays(&self, pipeline: u64, width: u32, height: u32) -> Result<(), QuantaError> {
        self.inner.dispatch_rays(pipeline, width, height)
    }

    /// Destroy an acceleration structure.
    pub fn destroy_acceleration_structure(&self, handle: u64) -> Result<(), QuantaError> {
        self.inner.destroy_acceleration_structure(handle)
    }
}

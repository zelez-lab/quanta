//! The [`RenderGpu`] extension trait for `quanta_core::Gpu`.
//!
//! The render methods used to be inherent on `Gpu`, gated behind the
//! `render` Cargo feature. With the render face promoted to its own
//! crate they become a **sealed extension trait**: a foreign type
//! cannot grow inherent methods across a crate boundary, so
//! `quanta-render` implements `RenderGpu` for `quanta_core::Gpu`
//! through the `#[doc(hidden)]` device-handle hook.
//!
//! Bring the trait into scope to call the render methods:
//!
//! ```ignore
//! use quanta_render::RenderGpu; // or `use quanta::*;` via the facade
//!
//! let gpu = quanta::init()?;
//! let target = gpu.render_target(640, 480, Format::RGBA8)?;
//! let pipe = gpu.pipeline(&desc)?;
//! let pulse = gpu.render(&target)?.pipeline(&pipe).draw(3).pulse()?;
//! ```
//!
//! **Sealed**: implemented only for `quanta_core::Gpu`. Because no
//! external impls exist, methods can be added after the API freeze
//! without a breaking change — same policy as `GpuDevice`.

use alloc::vec::Vec;

use quanta_core::{
    Format, GeometryDesc, IndirectRenderBundle, OcclusionQuery, Pipeline, PipelineDesc,
    QuantaError, RayTracingPipelineDesc, ShadingRate, SurfaceConfig, SurfaceTarget, Texture,
    TextureDesc, TextureUsage,
};

use crate::mesh_shader::{
    MAX_MESH_PRIMITIVES, MAX_MESH_VERTICES, MAX_TASK_THREADS, MeshPipeline, MeshPipelineDesc,
};
use crate::ray_tracing_wrap::{
    AccelerationStructure, AsKind, MAX_RECURSION_DEPTH, RayTracingPipeline,
};
use crate::render_builder::RenderBuilder;
use crate::surface_wrap::Surface;
use crate::tessellation::{MAX_PATCH_SIZE, TessTopology, TessellationPipeline};
use crate::vrs_wrap::VrsState;

mod sealed {
    pub trait Sealed {}
    impl Sealed for quanta_core::Gpu {}
}

/// Render extension methods on [`Gpu`](quanta_core::Gpu).
///
/// Implemented only for `quanta_core::Gpu` (sealed). See the
/// [module docs](self).
pub trait RenderGpu: sealed::Sealed {
    /// Create a render pipeline from a descriptor.
    fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError>;

    /// Begin a chainable render pass targeting a texture.
    ///
    /// Returns a [`RenderBuilder`] that records draw commands via
    /// method chaining and submits them with `.pulse()`.
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
    fn render(&self, target: &Texture) -> Result<RenderBuilder, QuantaError>;

    /// Create a render target texture (can be drawn to and read from
    /// shaders).
    fn render_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<Texture, QuantaError>;

    /// Create an MSAA render target.
    fn msaa_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<Texture, QuantaError>;

    /// Resolve an MSAA texture to a single-sample texture.
    ///
    /// The source must be a multi-sampled render target, and the
    /// destination must be a single-sample texture of the same
    /// dimensions and format.
    fn resolve_texture(&self, msaa_src: &Texture, resolve_dst: &Texture)
    -> Result<(), QuantaError>;

    /// Read stencil buffer contents from a depth/stencil texture.
    fn stencil_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError>;

    /// Allocate a render-path Indirect Command Buffer
    /// ([`IndirectRenderBundle`]) with the given capacity.
    ///
    /// Backends with no native render-bundle support return an error
    /// at create time. Steps 032 + 033, render path.
    fn render_bundle(&self, max_commands: u32) -> Result<IndirectRenderBundle, QuantaError>;

    /// Allocate a typed [`VrsState`] for variable rate shading.
    /// Default rate is 1×1 (no reduction). Steps 028 + 029.
    ///
    /// Backends without VRS (WebGPU, pre-Apple-Silicon Metal, Vulkan
    /// without `VK_KHR_fragment_shading_rate`) return `NotSupported`
    /// here so user code can branch.
    fn vrs_state(&self) -> Result<VrsState, QuantaError>;

    /// Allocate a typed [`MeshPipeline`] for the given mesh pipeline
    /// descriptor. Steps 024 + 025.
    ///
    /// Backends without mesh-shader support (WebGPU, older Metal,
    /// pre-1.3 Vulkan without VK_EXT_mesh_shader) return
    /// `NotSupported` here so user code can branch.
    fn mesh_pipeline(&self, desc: MeshPipelineDesc) -> Result<MeshPipeline, QuantaError>;

    /// Allocate a typed [`TessellationPipeline`] for the given patch
    /// topology and control-point count. Steps 022 + 023.
    ///
    /// Backends without tessellation (WebGPU, CPU-only fallbacks
    /// missing the feature) return `NotSupported` here so user code
    /// can branch.
    fn tessellation_pipeline(
        &self,
        topology: TessTopology,
        control_points: u32,
    ) -> Result<TessellationPipeline, QuantaError>;

    /// Create a presentation surface ([`Surface`]) over `target`,
    /// configured with `config`.
    ///
    /// This is the "Quanta owns present" model: run the
    /// acquire → render → present frame loop documented on
    /// [`Surface`]. Backends without a present path return
    /// `NotSupported`; query
    /// [`supports_surface_present`](quanta_core::Gpu::supports_surface_present)
    /// to branch ahead of time.
    fn create_surface(
        &self,
        target: &SurfaceTarget,
        config: &SurfaceConfig,
    ) -> Result<Surface, QuantaError>;

    /// Create an occlusion query set with `count` slots.
    fn occlusion_query_create(&self, count: u32) -> Result<OcclusionQuery, QuantaError>;

    /// Read results from an occlusion query set (fragment counts per slot).
    fn occlusion_query_read(&self, query: &OcclusionQuery) -> Result<Vec<u64>, QuantaError>;

    /// Build a typed bottom-level [`AccelerationStructure`] (BLAS)
    /// from geometry. Steps 026 + 027.
    ///
    /// Returns `Err(NotSupported)` when the backend does not
    /// implement ray tracing (WebGPU, devices without
    /// `VK_KHR_acceleration_structure`).
    fn acceleration_structure_blas(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<AccelerationStructure, QuantaError>;

    /// Allocate a typed [`RayTracingPipeline`] from the given
    /// descriptor. Steps 026 + 027.
    ///
    /// Bounds-checks `max_recursion` against the proven
    /// hardware-minimum cap before dispatching to the backend.
    fn ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<RayTracingPipeline, QuantaError>;
}

impl RenderGpu for quanta_core::Gpu {
    fn pipeline(&self, desc: &PipelineDesc) -> Result<Pipeline, QuantaError> {
        let device = self.device_handle();
        let mut pipeline = device.pipeline_create(desc)?;
        pipeline.__attach_device(device.clone());
        Ok(pipeline)
    }

    fn render(&self, target: &Texture) -> Result<RenderBuilder, QuantaError> {
        let device = self.device_handle();
        let pass = device.render_begin(target)?;
        Ok(RenderBuilder::new(device.clone(), pass))
    }

    fn render_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
    ) -> Result<Texture, QuantaError> {
        self.create_texture(
            &TextureDesc::new(width, height, format)
                .with_usage(TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ)),
        )
    }

    fn msaa_target(
        &self,
        width: u32,
        height: u32,
        format: Format,
        samples: u32,
    ) -> Result<Texture, QuantaError> {
        self.create_texture(
            &TextureDesc::new(width, height, format)
                .with_sample_count(samples)
                .with_usage(TextureUsage::RENDER_TARGET),
        )
    }

    fn resolve_texture(
        &self,
        msaa_src: &Texture,
        resolve_dst: &Texture,
    ) -> Result<(), QuantaError> {
        self.device_handle()
            .resolve_texture(msaa_src.handle(), resolve_dst.handle())
    }

    fn stencil_read(&self, texture: &Texture) -> Result<Vec<u8>, QuantaError> {
        self.device_handle().stencil_read(texture.handle())
    }

    fn render_bundle(&self, max_commands: u32) -> Result<IndirectRenderBundle, QuantaError> {
        let device = self.device_handle();
        let handle = device.render_bundle_create(max_commands)?;
        Ok(IndirectRenderBundle::__new(
            handle,
            max_commands,
            device.clone(),
        ))
    }

    fn vrs_state(&self) -> Result<VrsState, QuantaError> {
        let device = self.device_handle();
        let handle = device.vrs_create()?;
        Ok(VrsState {
            handle,
            current: ShadingRate::R1x1,
            device: device.clone(),
            live: true,
        })
    }

    fn mesh_pipeline(&self, desc: MeshPipelineDesc) -> Result<MeshPipeline, QuantaError> {
        if !(1..=MAX_MESH_VERTICES).contains(&desc.max_vertices_per_meshlet) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline max_vertices_per_meshlet out of range",
            ));
        }
        if !(1..=MAX_MESH_PRIMITIVES).contains(&desc.max_primitives_per_meshlet) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline max_primitives_per_meshlet out of range",
            ));
        }
        if !(1..=MAX_TASK_THREADS).contains(&desc.task_threads_per_group) {
            return Err(QuantaError::invalid_param(
                "mesh pipeline task_threads_per_group out of range",
            ));
        }
        let device = self.device_handle();
        let handle = device.mesh_pipeline_create(
            desc.max_vertices_per_meshlet,
            desc.max_primitives_per_meshlet,
            desc.task_threads_per_group,
        )?;
        Ok(MeshPipeline {
            handle,
            desc,
            device: device.clone(),
            live: true,
        })
    }

    fn tessellation_pipeline(
        &self,
        topology: TessTopology,
        control_points: u32,
    ) -> Result<TessellationPipeline, QuantaError> {
        if !(1..=MAX_PATCH_SIZE).contains(&control_points) {
            return Err(QuantaError::invalid_param(
                "tessellation control_points must be in [1, MAX_PATCH_SIZE]",
            ));
        }
        let topo_byte: u8 = match topology {
            TessTopology::Triangle => 0,
            TessTopology::Quad => 1,
        };
        let device = self.device_handle();
        let handle = device.tessellation_pipeline_create(topo_byte, control_points)?;
        Ok(TessellationPipeline {
            handle,
            topology,
            control_points,
            device: device.clone(),
            live: true,
        })
    }

    fn create_surface(
        &self,
        target: &SurfaceTarget,
        config: &SurfaceConfig,
    ) -> Result<Surface, QuantaError> {
        if config.width == 0 || config.height == 0 {
            return Err(QuantaError::invalid_param(
                "surface extent must be non-zero",
            ));
        }
        let device = self.device_handle();
        let handle = device.surface_create(target, config)?;
        Ok(Surface {
            handle,
            config: *config,
            device: device.clone(),
        })
    }

    fn occlusion_query_create(&self, count: u32) -> Result<OcclusionQuery, QuantaError> {
        let device = self.device_handle();
        let handle = device.occlusion_query_create(count)?;
        Ok(OcclusionQuery::__new(handle, count, device.clone()))
    }

    fn occlusion_query_read(&self, query: &OcclusionQuery) -> Result<Vec<u64>, QuantaError> {
        self.device_handle().occlusion_query_read(query.handle())
    }

    fn acceleration_structure_blas(
        &self,
        geometry: &[GeometryDesc],
    ) -> Result<AccelerationStructure, QuantaError> {
        if geometry.is_empty() {
            return Err(QuantaError::invalid_param(
                "acceleration structure requires at least one geometry descriptor",
            ));
        }
        let device = self.device_handle();
        let handle = device.build_acceleration_structure(geometry)?;
        Ok(AccelerationStructure {
            handle,
            kind: AsKind::Bottom,
            geom_count: geometry.len() as u32,
            device: Some(device.clone()),
            live: true,
        })
    }

    fn ray_tracing_pipeline(
        &self,
        desc: &RayTracingPipelineDesc,
    ) -> Result<RayTracingPipeline, QuantaError> {
        if desc.max_recursion > MAX_RECURSION_DEPTH {
            return Err(QuantaError::invalid_param(
                "max_recursion exceeds MAX_RECURSION_DEPTH",
            ));
        }
        let device = self.device_handle();
        let handle = device.create_ray_tracing_pipeline(desc)?;
        Ok(RayTracingPipeline {
            handle,
            max_recursion: desc.max_recursion,
            device: device.clone(),
            live: true,
        })
    }
}

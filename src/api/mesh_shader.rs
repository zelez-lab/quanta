//! Typed mesh shader pipelines (steps 024 + 025).
//!
//! Mesh shaders replace the classical vertex / tessellation / geometry
//! pipeline with a programmable two-stage path that emits geometry
//! directly from compute-style workgroups (meshlets). The user
//! creates a `MeshPipeline` with bounded vertex / primitive limits,
//! optionally a task-stage thread count, and dispatches workgroups
//! via [`MeshPipeline::dispatch`].
//!
//! Backends:
//!
//! - Metal 3+: `MTLMeshRenderPipelineDescriptor` + object/mesh
//!   functions; `drawMeshThreadgroups:` dispatches.
//! - Vulkan: `VK_EXT_mesh_shader` + `vkCmdDrawMeshTasksEXT`.
//! - WebGPU: not in W3C — `NotSupported` at create time.
//! - CPU: software lifecycle only.
//!
//! The wrapper enforces the lifecycle proven in
//! `Quanta.MeshShader.Pipeline` (Lean) and
//! `quanta-api/mesh_shader_safety.rs` (Verus):
//!
//! - `dispatch(gx, gy, gz)` fails when any axis exceeds
//!   `MAX_GROUP_COUNT` or the pipeline is destroyed.
//! - `Drop` calls `mesh_pipeline_destroy` exactly once.
//!
//! Limits are clamped at create time against the proven
//! hardware-minimum bounds (Vulkan EXT guarantees).

use alloc::sync::Arc;

use crate::{GpuDevice, QuantaError};

/// Maximum vertices per meshlet — Vulkan EXT minimum + Metal 3 cap.
pub const MAX_MESH_VERTICES: u32 = 256;

/// Maximum primitives per meshlet — Vulkan EXT minimum + Metal 3 cap.
pub const MAX_MESH_PRIMITIVES: u32 = 256;

/// Maximum threads per task workgroup — Vulkan EXT minimum.
pub const MAX_TASK_THREADS: u32 = 128;

/// Maximum workgroups in one dispatch axis — Vulkan + Metal both
/// guarantee at least this.
pub const MAX_GROUP_COUNT: u32 = 65535;

/// Descriptor for a mesh pipeline.
#[derive(Copy, Clone, Debug)]
pub struct MeshPipelineDesc {
    /// Maximum vertices any meshlet workgroup will emit. Must be in
    /// `1..=MAX_MESH_VERTICES`.
    pub max_vertices_per_meshlet: u32,
    /// Maximum primitives any meshlet workgroup will emit. Must be
    /// in `1..=MAX_MESH_PRIMITIVES`.
    pub max_primitives_per_meshlet: u32,
    /// Number of threads per task-stage workgroup. Must be in
    /// `1..=MAX_TASK_THREADS`. Set to `1` if no task stage is used
    /// (mesh-only pipeline).
    pub task_threads_per_group: u32,
}

impl Default for MeshPipelineDesc {
    fn default() -> Self {
        Self {
            max_vertices_per_meshlet: 64,
            max_primitives_per_meshlet: 124,
            task_threads_per_group: 1,
        }
    }
}

/// A typed mesh-shader pipeline. Drop releases the backend handle.
pub struct MeshPipeline {
    pub(crate) handle: u64,
    pub(crate) desc: MeshPipelineDesc,
    pub(crate) device: Arc<dyn GpuDevice>,
    pub(crate) live: bool,
}

impl MeshPipeline {
    /// Underlying device handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Descriptor the pipeline was created with.
    pub fn desc(&self) -> &MeshPipelineDesc {
        &self.desc
    }

    /// Maximum vertices any meshlet from this pipeline emits.
    pub fn max_vertices_per_meshlet(&self) -> u32 {
        self.desc.max_vertices_per_meshlet
    }

    /// Maximum primitives any meshlet from this pipeline emits.
    pub fn max_primitives_per_meshlet(&self) -> u32 {
        self.desc.max_primitives_per_meshlet
    }

    /// Dispatch `[gx, gy, gz]` mesh workgroups.
    ///
    /// Returns `Err(InvalidParam)` if any axis exceeds
    /// `MAX_GROUP_COUNT` or the pipeline has been destroyed.
    /// Refines `Quanta.MeshShader.Pipeline.dispatch` and the Verus
    /// theorem `t7351_dispatch_appends`.
    pub fn dispatch(&self, groups: [u32; 3]) -> Result<(), QuantaError> {
        if !self.live {
            return Err(QuantaError::invalid_param("mesh pipeline is not live"));
        }
        if groups[0] > MAX_GROUP_COUNT || groups[1] > MAX_GROUP_COUNT || groups[2] > MAX_GROUP_COUNT
        {
            return Err(QuantaError::invalid_param(
                "mesh dispatch group count exceeds MAX_GROUP_COUNT",
            ));
        }
        self.device.mesh_dispatch(self.handle, groups)
    }
}

impl Drop for MeshPipeline {
    fn drop(&mut self) {
        if self.live {
            let _ = self.device.mesh_pipeline_destroy(self.handle);
            self.live = false;
        }
    }
}

pub mod async_copy;
pub mod batch;
pub mod bindless;
pub mod device;
pub mod error;
pub mod field;
pub mod gpu;
pub mod icb;
pub mod multi_queue;
pub mod printf;
pub mod pulse;
pub mod sparse_texture;
pub mod texture;
pub mod types;
pub mod wave;

// Render face — only when the `render` feature is on (step 085).
#[cfg(feature = "render")]
pub mod mesh_shader;
#[cfg(feature = "render")]
pub mod pipeline;
#[cfg(feature = "render")]
pub mod ray_tracing;
#[cfg(feature = "render")]
pub mod render_builder;
#[cfg(feature = "render")]
pub mod render_pass;
#[cfg(feature = "render")]
pub mod shader;
#[cfg(feature = "render")]
pub mod surface;
#[cfg(feature = "render")]
pub mod tessellation;
#[cfg(feature = "render")]
pub mod vrs;

pub use async_copy::AsyncCopyQueue;
pub use batch::Batch;
pub use bindless::{BindlessBufferArray, BindlessTextureArray};
pub use device::{GpuDevice, RegistryCounts};
pub use error::{QuantaError, QuantaErrorKind};
pub use field::{Field, MappedField};
pub use gpu::Gpu;
pub use icb::IndirectCommandBuffer;
pub use multi_queue::Queue;
pub use printf::PrintfBuffer;
pub use pulse::{OcclusionQuery, Pulse, Timeline, TimestampQuery};
pub use sparse_texture::SparseTexture;
pub use texture::*;
pub use types::*;
pub use wave::Wave;

// Render-face re-exports — gated with the `render` feature.
#[cfg(feature = "render")]
pub use icb::IndirectRenderBundle;
#[cfg(feature = "render")]
pub use mesh_shader::{
    MAX_GROUP_COUNT, MAX_MESH_PRIMITIVES, MAX_MESH_VERTICES, MAX_TASK_THREADS, MeshPipeline,
    MeshPipelineDesc,
};
#[cfg(feature = "render")]
pub use pipeline::*;
#[cfg(feature = "render")]
pub use ray_tracing::*;
#[cfg(feature = "render")]
pub use render_builder::RenderBuilder;
#[cfg(feature = "render")]
pub use render_pass::*;
#[cfg(feature = "render")]
pub use shader::{ShaderBinary, ShaderStage};
#[cfg(feature = "render")]
pub use surface::{PresentMode, Surface, SurfaceConfig, SurfaceFrame, SurfaceTarget};
#[cfg(feature = "render")]
pub use tessellation::{MAX_PATCH_SIZE, MAX_TESS_LEVEL, TessTopology, TessellationPipeline};
#[cfg(feature = "render")]
pub use vrs::{ShadingRate, VrsState};

pub mod async_copy;
pub mod bindless;
pub mod device;
pub mod error;
pub mod field;
pub mod gpu;
pub mod printf;
pub mod pulse;
pub mod sparse_texture;
pub mod texture;
pub mod types;

// Compute face — only when the `compute` feature is on (symmetric to
// the render face below). `Wave` is the compute dispatch handle;
// `Batch` records wave dispatches; `Queue` submits wave dispatches on
// explicit queue handles.
#[cfg(feature = "compute")]
pub mod batch;
#[cfg(feature = "compute")]
pub mod multi_queue;
#[cfg(feature = "compute")]
pub mod wave;

// Kernel-language types named by `#[quanta::kernel]`-generated code:
// the `GpuType` marker trait, the `KernelBinary` a compiled kernel
// expands to, and the `ScalarType` tag. They live here (behind
// `compute`) so companion crates that host kernels reach them without
// depending on the `quanta` facade; the facade re-exports all three.
#[cfg(feature = "compute")]
pub mod gpu_type;

// Host-side stubs for every GPU intrinsic, injected by the `_src!()`
// macro `#[quanta::device]` emits (via `use
// <crate>::__device_host_stubs::*`) and by the differential host
// oracle. Hidden from the public API. Lives here so downstream crates
// name-resolve spliced device-fn bodies through `quanta-core` rather
// than the facade.
#[cfg(feature = "compute")]
#[doc(hidden)]
pub mod device_host_stubs;

// Generated device-fn `_src!` macros and host oracles name the stub
// module through `<crate>::__device_host_stubs` (the historical facade
// path). Expose that `__`-prefixed alias here too so a `crate =
// quanta_core` override resolves it directly, not only through the
// facade's own re-export.
#[cfg(feature = "compute")]
#[doc(hidden)]
pub use device_host_stubs as __device_host_stubs;

// Indirect command buffers are a split surface: the compute ICB
// (`IndirectCommandBuffer`, records wave dispatches) needs `compute`;
// the render bundle (`IndirectRenderBundle`) needs `render`. The
// module gates its two halves internally.
#[cfg(any(feature = "compute", feature = "render"))]
pub mod icb;

// Render data model — only when the `render` feature is on. This is
// the render surface the `GpuDevice` trait and the drivers speak
// (pipeline descriptors, the render-pass op stream, shader binaries,
// surface configuration, shading rates, ray-tracing descriptors).
// The typed wrappers and the `RenderGpu` extension trait live in the
// `quanta-render` crate.
// The builder-managed MSAA intermediate pool (`RenderBuilder::msaa`)
// needs `std` for its lock; every backend feature implies `std`, so
// the gate only prunes the pure type-check no_std configuration.
#[cfg(all(feature = "render", feature = "std"))]
pub mod msaa_pool;
#[cfg(feature = "render")]
pub mod pipeline;
#[cfg(feature = "render")]
pub mod ray_tracing;
#[cfg(feature = "render")]
pub mod render_pass;
#[cfg(feature = "render")]
pub mod shader;
#[cfg(feature = "render")]
pub mod surface;
#[cfg(feature = "render")]
pub mod vrs;

pub use async_copy::AsyncCopyQueue;
pub use bindless::{BindlessBufferArray, BindlessTextureArray};
pub use device::{GpuDevice, RegistryCounts};
pub use error::{QuantaError, QuantaErrorKind};
pub use field::{Field, HostField, MappedField, NativeBufferHandle, SharedField};
pub use gpu::Gpu;
pub use printf::PrintfBuffer;
pub use pulse::{OcclusionQuery, Pulse, Timeline, TimestampQuery};
pub use sparse_texture::SparseTexture;
pub use texture::*;
pub use types::*;

// Compute-face re-exports — gated with the `compute` feature.
#[cfg(feature = "compute")]
pub use batch::Batch;
#[cfg(feature = "compute")]
pub use gpu_type::{GpuType, KernelBinary, ScalarType};
#[cfg(feature = "compute")]
pub use icb::IndirectCommandBuffer;
#[cfg(feature = "compute")]
pub use multi_queue::Queue;
#[cfg(feature = "compute")]
pub use wave::Wave;

// Render data-model re-exports — gated with the `render` feature.
#[cfg(feature = "render")]
pub use icb::IndirectRenderBundle;
#[cfg(all(feature = "render", feature = "std"))]
pub use msaa_pool::MsaaPool;
#[cfg(feature = "render")]
pub use pipeline::*;
#[cfg(feature = "render")]
pub use ray_tracing::*;
#[cfg(feature = "render")]
pub use render_pass::*;
#[cfg(feature = "render")]
pub use shader::{ShaderBinary, ShaderStage, Vec2, Vec3, Vec4};
#[cfg(feature = "render")]
pub use surface::{PresentMode, SurfaceConfig, SurfaceTarget};
#[cfg(feature = "render")]
pub use vrs::ShadingRate;

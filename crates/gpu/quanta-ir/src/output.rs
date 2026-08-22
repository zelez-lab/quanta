//! Compiler output types.

/// Compiler output — one slot per artifact a driver loads.
///
/// There is deliberately no PTX or GCN-ELF slot: no driver consumes
/// them (the Vulkan driver runs NVIDIA and AMD cards on SPIR-V), and the
/// LLVM `nvptx`/`amdgpu` targets stay compiler-internal experiments
/// reachable through `--llvm-only`.
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` and
/// `metallib_ios_sim` are the platform-correct variants for an iOS device
/// and the iOS simulator; each is `None` when its SDK was absent at
/// compile time or the platform was excluded via `QUANTA_METAL_PLATFORMS`.
/// The runtime picks among them by compile target (see
/// `KernelBinary::for_artifact`).
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub metallib_ios: Option<Vec<u8>>,
    pub metallib_ios_sim: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}

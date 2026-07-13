//! Compiler output types.

/// Compiler output — compiled kernel for all targets.
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` and
/// `metallib_ios_sim` are the platform-correct variants for an iOS device
/// and the iOS simulator; each is `None` when its SDK was absent at
/// compile time or the platform was excluded via `QUANTA_METAL_PLATFORMS`.
/// The runtime picks among them by compile target (see
/// `KernelBinary::for_vendor`).
#[derive(Debug, Clone)]
pub struct CompilerOutput {
    pub amd: Option<Vec<u8>>,
    pub nvidia: Option<Vec<u8>>,
    pub spirv: Option<Vec<u8>>,
    pub metallib: Option<Vec<u8>>,
    pub metallib_ios: Option<Vec<u8>>,
    pub metallib_ios_sim: Option<Vec<u8>>,
    pub wgsl: Option<String>,
}

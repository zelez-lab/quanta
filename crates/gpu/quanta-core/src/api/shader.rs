//! Render-stage shader types.
//!
//! `ShaderStage` and `ShaderBinary` describe programmable *render*
//! pipeline stages (vertex, fragment, tessellation, mesh, ray tracing)
//! — they live on the render surface, not in the compute kernel module.
//! Compute kernels compile to [`KernelBinary`](crate::KernelBinary)
//! instead.

/// Shader-interface vector types — the field types of a
/// `#[derive(quanta::Varyings)]` struct (and the value types of the shader
/// DSL: `Vec2`/`Vec3`/`Vec4` in a `#[quanta::vertex]` / `#[quanta::fragment]`
/// signature name these).
///
/// The shader FUNCTIONS themselves are erased by the proc macros (their
/// bodies compile to GPU binaries, never to host code), but a Varyings
/// struct is a real Rust item — its field types must resolve. These are
/// plain `#[repr(C)]` PODs so the struct is also usable as ordinary host
/// data if a consumer wants to.
macro_rules! shader_vec {
    ($(#[$doc:meta] $name:ident { $($field:ident),+ })+) => {$(
        #[$doc]
        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct $name {
            $(pub $field: f32,)+
        }

        impl $name {
            /// Component-wise constructor — the same spelling shader bodies
            /// use (`Vec4::new(x, y, z, w)`).
            #[allow(clippy::new_without_default)]
            pub const fn new($($field: f32),+) -> Self {
                Self { $($field),+ }
            }
        }
    )+};
}

shader_vec! {
    /// A 2-component `f32` vector (`float2` / `vec2<f32>`).
    Vec2 { x, y }
    /// A 3-component `f32` vector (`float3` / `vec3<f32>`).
    Vec3 { x, y, z }
    /// A 4-component `f32` vector (`float4` / `vec4<f32>`).
    Vec4 { x, y, z, w }
}

/// Shader stage — which programmable pipeline stage this shader runs in.
///
/// Marked `#[non_exhaustive]`: stages can be added — match with a
/// wildcard arm.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    Fragment,
    /// Tessellation control (hull) shader.
    TessControl,
    /// Tessellation evaluation (domain) shader.
    TessEval,
    /// Task (amplification) shader — launches mesh shader threadgroups.
    Task,
    /// Mesh shader — generates vertices and primitives.
    Mesh,
    /// Ray generation shader.
    RayGen,
    /// Closest-hit shader.
    ClosestHit,
    /// Miss shader.
    Miss,
}

/// A compiled shader binary — output of `#[quanta::vertex]` or `#[quanta::fragment]`.
///
/// Contains pre-compiled binaries for each supported GPU vendor.
/// The driver selects the appropriate binary at pipeline creation time.
///
/// `metallib` is the macOS-platform Metal library. `metallib_ios` /
/// `metallib_ios_sim` are the iOS-device / iOS-simulator variants: iOS
/// rejects a macOS-platform metallib, so a build targeting an iOS device
/// or the simulator embeds and selects its own. The proc macro cannot see
/// the consumer's target, so it embeds every variant the compiler produced
/// and [`ShaderBinary::for_vendor`] picks the platform-correct one by `cfg`.
pub struct ShaderBinary {
    /// Pre-compiled SPIR-V binary.
    pub spirv: Option<&'static [u8]>,
    /// Pre-compiled Metal library binary (macOS platform).
    pub metallib: Option<&'static [u8]>,
    /// Pre-compiled Metal library binary (iOS device).
    pub metallib_ios: Option<&'static [u8]>,
    /// Pre-compiled Metal library binary (iOS simulator).
    pub metallib_ios_sim: Option<&'static [u8]>,
    /// WGSL source for WebGPU.
    pub wgsl: Option<&'static str>,
    /// Shader entry point name.
    pub entry_point: &'static str,
    /// Shader stage (vertex or fragment).
    pub stage: ShaderStage,
}

impl ShaderBinary {
    /// Select the best shader binary for the given vendor.
    ///
    /// Apple: platform-correct metallib (see [`Self::apple_metallib`]),
    /// falling back to SPIR-V. All others: SPIR-V binary.
    pub fn for_vendor(&self, vendor: crate::Vendor) -> Option<&[u8]> {
        match vendor {
            crate::Vendor::Apple => self.apple_metallib().or(self.spirv),
            _ => self.spirv,
        }
    }

    /// Resolve the metallib for the current *compile target*.
    ///
    /// Proc macros cannot learn the build target, so all variants are
    /// embedded and the choice is made here by `cfg`. Each target compiles
    /// exactly one arm — the chain (iOS-sim → iOS-device → macOS) only
    /// degrades to a less-specific variant when the more-specific one
    /// wasn't produced. macOS/desktop builds see only the macOS field, so
    /// behavior there is unchanged (the SPIR-V fallback still lives in
    /// [`Self::for_vendor`]).
    #[cfg(all(target_os = "ios", target_abi = "sim"))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib_ios_sim
            .or(self.metallib_ios)
            .or(self.metallib)
    }

    #[cfg(all(target_os = "ios", not(target_abi = "sim")))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib_ios.or(self.metallib)
    }

    #[cfg(not(target_os = "ios"))]
    fn apple_metallib(&self) -> Option<&[u8]> {
        self.metallib
    }
}

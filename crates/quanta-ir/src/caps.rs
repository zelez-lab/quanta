//! Backend capability tables — step 082 Layer 4.
//!
//! Each emitter target (Metal, Vulkan, WebGPU, CPU) declares which
//! `ScalarType`s it natively supports, which it polyfills, and
//! which it refuses outright. The IR validator (`crate::validate`)
//! consults these tables before invoking an emitter so that a
//! kernel using an unsupported type returns a clean error at IR
//! validation time instead of producing nonsense backend output.
//!
//! The table is declarative data, not behavior — it documents the
//! shape of the backend ecosystem in one place. New rows land here
//! when a real backend constraint surfaces; the existing four cover
//! the cases we already know about:
//!
//! - **Metal** has no `double` type (MSL spec): F64 → NotSupported.
//! - **Vulkan** can run F64 on devices with the `shaderFloat64`
//!   feature enabled, F16 with `shaderFloat16`: those are
//!   RequiresFeature, not Native.
//! - **WebGPU** has no F64 or I64 in the WGSL surface today: both
//!   NotSupported. F16 requires the `f16` extension.
//! - **CPU** software executor handles every type natively.
//!
//! The table covers `ScalarType` only today; per-op feature flags
//! (atomic ordering modes, subgroup uniformity, saturating
//! arithmetic, etc.) are a planned extension.

use crate::types::ScalarType;

/// Which backend target a `BackendCaps` describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Vulkan,
    WebGpu,
    Cpu,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::WebGpu => "webgpu",
            Backend::Cpu => "cpu",
        }
    }
}

/// How a backend handles a given type or feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSupport {
    /// The backend's emitter handles this type directly.
    Native,
    /// The backend can handle this type, but only when a device
    /// feature is enabled at runtime (e.g. Vulkan `shaderFloat64`).
    /// The string is the feature name from the relevant spec.
    RequiresFeature(&'static str),
    /// The backend cannot handle this type at all. The string is a
    /// short human-readable reason suitable for inclusion in error
    /// messages.
    NotSupported(&'static str),
}

impl TypeSupport {
    /// True if the type can be emitted without a feature gate. The
    /// IR validator treats `RequiresFeature` as a soft warning today
    /// (the runtime device-caps check is the source of truth) and
    /// returns `Ok` so existing code that branches on
    /// `Gpu::supports_*` continues to work.
    pub fn is_unconditionally_supported(self) -> bool {
        matches!(self, TypeSupport::Native)
    }

    /// True if emission is hard-rejected. The IR validator uses
    /// this to decide whether to skip an emitter entirely.
    pub fn is_rejected(self) -> bool {
        matches!(self, TypeSupport::NotSupported(_))
    }

    /// Reason string when rejected, otherwise empty.
    pub fn reason(self) -> &'static str {
        match self {
            TypeSupport::Native => "",
            TypeSupport::RequiresFeature(_) => "",
            TypeSupport::NotSupported(r) => r,
        }
    }
}

/// Per-backend support matrix. One row per backend target.
#[derive(Debug, Clone, Copy)]
pub struct BackendCaps {
    pub backend: Backend,
    pub f16: TypeSupport,
    pub bf16: TypeSupport,
    pub fp8_e5m2: TypeSupport,
    pub fp8_e4m3: TypeSupport,
    pub f32: TypeSupport,
    pub f64: TypeSupport,
    pub u8_: TypeSupport,
    pub u16_: TypeSupport,
    pub u32_: TypeSupport,
    pub u64_: TypeSupport,
    pub i8_: TypeSupport,
    pub i16_: TypeSupport,
    pub i32_: TypeSupport,
    pub i64_: TypeSupport,
    pub bool_: TypeSupport,
}

impl BackendCaps {
    /// Look up the support level for a `ScalarType`.
    pub fn scalar(&self, ty: ScalarType) -> TypeSupport {
        match ty {
            ScalarType::F16 => self.f16,
            ScalarType::BF16 => self.bf16,
            ScalarType::FP8E5M2 => self.fp8_e5m2,
            ScalarType::FP8E4M3 => self.fp8_e4m3,
            ScalarType::F32 => self.f32,
            ScalarType::F64 => self.f64,
            ScalarType::U8 => self.u8_,
            ScalarType::U16 => self.u16_,
            ScalarType::U32 => self.u32_,
            ScalarType::U64 => self.u64_,
            ScalarType::I8 => self.i8_,
            ScalarType::I16 => self.i16_,
            ScalarType::I32 => self.i32_,
            ScalarType::I64 => self.i64_,
            ScalarType::Bool => self.bool_,
        }
    }
}

/// Apple Metal Shading Language. F64 is not in the MSL spec at all;
/// the emitter currently maps `ScalarType::F64 -> "double"` which
/// `xcrun metal` rejects with no recovery path. F16 is native on
/// Apple Silicon (`half`). Narrow ints are emitted as MSL `uchar`/
/// `char`/`ushort`/`short`/`uint`/`int`/`ulong`/`long`.
pub const METAL: BackendCaps = BackendCaps {
    backend: Backend::Metal,
    f16: TypeSupport::Native,
    bf16: TypeSupport::Native, // always available via f32-emulated path
    fp8_e5m2: TypeSupport::Native,
    fp8_e4m3: TypeSupport::Native,
    f32: TypeSupport::Native,
    f64: TypeSupport::NotSupported("Metal Shading Language has no double type"),
    u8_: TypeSupport::Native,
    u16_: TypeSupport::Native,
    u32_: TypeSupport::Native,
    u64_: TypeSupport::Native,
    i8_: TypeSupport::Native,
    i16_: TypeSupport::Native,
    i32_: TypeSupport::Native,
    i64_: TypeSupport::Native,
    bool_: TypeSupport::Native,
};

/// SPIR-V via Vulkan. F64 and F16 are spec-optional features that
/// require enabling `shaderFloat64` / `shaderFloat16` on the
/// `VkDevice` at creation. The runtime device-caps check
/// (`Gpu::supports_*`) is the source of truth for whether a given
/// device actually advertises them. I64 atomics likewise require a
/// feature gate; plain I64 storage is core in SPIR-V 1.5+.
pub const VULKAN: BackendCaps = BackendCaps {
    backend: Backend::Vulkan,
    f16: TypeSupport::RequiresFeature("shaderFloat16"),
    bf16: TypeSupport::Native, // always available via f32-emulated path
    fp8_e5m2: TypeSupport::Native,
    fp8_e4m3: TypeSupport::Native,
    f32: TypeSupport::Native,
    f64: TypeSupport::RequiresFeature("shaderFloat64"),
    u8_: TypeSupport::Native,
    u16_: TypeSupport::Native,
    u32_: TypeSupport::Native,
    u64_: TypeSupport::Native,
    i8_: TypeSupport::Native,
    i16_: TypeSupport::Native,
    i32_: TypeSupport::Native,
    i64_: TypeSupport::Native,
    bool_: TypeSupport::Native,
};

/// WebGPU / WGSL. The 1.0 spec has only `i32`, `u32`, `f32`, `bool`
/// in the scalar surface; `f16` is gated behind the `f16` extension
/// (now widely shipped). `f64` and `i64`/`u64` are not in the spec
/// today.
pub const WEBGPU: BackendCaps = BackendCaps {
    backend: Backend::WebGpu,
    f16: TypeSupport::RequiresFeature("f16"),
    bf16: TypeSupport::Native, // always available via f32-emulated path
    fp8_e5m2: TypeSupport::Native,
    fp8_e4m3: TypeSupport::Native,
    f32: TypeSupport::Native,
    f64: TypeSupport::NotSupported("WGSL has no f64 type"),
    u8_: TypeSupport::NotSupported("WGSL has no u8 storage type"),
    u16_: TypeSupport::NotSupported("WGSL has no u16 storage type"),
    u32_: TypeSupport::Native,
    u64_: TypeSupport::NotSupported("WGSL has no u64 type"),
    i8_: TypeSupport::NotSupported("WGSL has no i8 storage type"),
    i16_: TypeSupport::NotSupported("WGSL has no i16 storage type"),
    i32_: TypeSupport::Native,
    i64_: TypeSupport::NotSupported("WGSL has no i64 type"),
    bool_: TypeSupport::Native,
};

/// CPU software executor. Pure Rust evaluation, handles every type
/// natively. Listed for completeness so the validator can be called
/// uniformly across all backends.
pub const CPU: BackendCaps = BackendCaps {
    backend: Backend::Cpu,
    f16: TypeSupport::Native,
    bf16: TypeSupport::Native, // always available via f32-emulated path
    fp8_e5m2: TypeSupport::Native,
    fp8_e4m3: TypeSupport::Native,
    f32: TypeSupport::Native,
    f64: TypeSupport::Native,
    u8_: TypeSupport::Native,
    u16_: TypeSupport::Native,
    u32_: TypeSupport::Native,
    u64_: TypeSupport::Native,
    i8_: TypeSupport::Native,
    i16_: TypeSupport::Native,
    i32_: TypeSupport::Native,
    i64_: TypeSupport::Native,
    bool_: TypeSupport::Native,
};

/// Resolve a `Backend` to its caps row.
pub fn caps_for(backend: Backend) -> &'static BackendCaps {
    match backend {
        Backend::Metal => &METAL,
        Backend::Vulkan => &VULKAN,
        Backend::WebGpu => &WEBGPU,
        Backend::Cpu => &CPU,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_rejects_f64() {
        assert!(METAL.scalar(ScalarType::F64).is_rejected());
        assert_eq!(
            METAL.scalar(ScalarType::F64).reason(),
            "Metal Shading Language has no double type"
        );
    }

    #[test]
    fn vulkan_gates_f64_on_feature() {
        match VULKAN.scalar(ScalarType::F64) {
            TypeSupport::RequiresFeature(name) => assert_eq!(name, "shaderFloat64"),
            other => panic!("expected RequiresFeature, got {:?}", other),
        }
    }

    #[test]
    fn webgpu_rejects_64bit_types() {
        for ty in [ScalarType::F64, ScalarType::U64, ScalarType::I64] {
            assert!(
                WEBGPU.scalar(ty).is_rejected(),
                "WebGPU should reject {:?}",
                ty
            );
        }
    }

    #[test]
    fn cpu_supports_everything_natively() {
        for ty in [
            ScalarType::F16,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::U8,
            ScalarType::U16,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::Bool,
        ] {
            assert!(
                CPU.scalar(ty).is_unconditionally_supported(),
                "CPU should support {:?} natively",
                ty
            );
        }
    }

    #[test]
    fn caps_for_round_trip() {
        for b in [
            Backend::Metal,
            Backend::Vulkan,
            Backend::WebGpu,
            Backend::Cpu,
        ] {
            assert_eq!(caps_for(b).backend, b);
        }
    }
}

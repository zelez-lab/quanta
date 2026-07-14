//! GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate.
//!
//! Source: web/webgpu.idl  (sha256: e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7)
//! Generator: crates/lang/quanta-codegen (B′ track of the FFI TCB shrink).
//!
//! Each `SPEC_*` table holds every value that the W3C `webgpu.idl`
//! lists for the corresponding enum, in source order. Quanta's
//! `src/driver/webgpu/ffi.rs` enum-code modules (`format`,
//! `blend_factor`, …) define a *subset* of these strings — every
//! string they expose is required to appear here. The codegen
//! cross-checks at build time; the Rust unit test
//! `tests::quanta_strings_are_spec_subsets` exercises it again at
//! `cargo test`.

#![allow(dead_code, non_upper_case_globals)]

/// SHA-256 of `web/webgpu.idl` at codegen time.
pub const WEBGPU_IDL_SHA256: &str =
    "e85178050c10e68381efabbc664cf6c3368055242060abb0627d923ded5c69a7";

/// W3C `webgpu.idl` enum `GPUTextureFormat` — 101 values.
pub static SPEC_GPUTextureFormat: &[&str] = &[
    "r8unorm",
    "r8snorm",
    "r8uint",
    "r8sint",
    "r16unorm",
    "r16snorm",
    "r16uint",
    "r16sint",
    "r16float",
    "rg8unorm",
    "rg8snorm",
    "rg8uint",
    "rg8sint",
    "r32uint",
    "r32sint",
    "r32float",
    "rg16unorm",
    "rg16snorm",
    "rg16uint",
    "rg16sint",
    "rg16float",
    "rgba8unorm",
    "rgba8unorm-srgb",
    "rgba8snorm",
    "rgba8uint",
    "rgba8sint",
    "bgra8unorm",
    "bgra8unorm-srgb",
    "rgb9e5ufloat",
    "rgb10a2uint",
    "rgb10a2unorm",
    "rg11b10ufloat",
    "rg32uint",
    "rg32sint",
    "rg32float",
    "rgba16unorm",
    "rgba16snorm",
    "rgba16uint",
    "rgba16sint",
    "rgba16float",
    "rgba32uint",
    "rgba32sint",
    "rgba32float",
    "stencil8",
    "depth16unorm",
    "depth24plus",
    "depth24plus-stencil8",
    "depth32float",
    "depth32float-stencil8",
    "bc1-rgba-unorm",
    "bc1-rgba-unorm-srgb",
    "bc2-rgba-unorm",
    "bc2-rgba-unorm-srgb",
    "bc3-rgba-unorm",
    "bc3-rgba-unorm-srgb",
    "bc4-r-unorm",
    "bc4-r-snorm",
    "bc5-rg-unorm",
    "bc5-rg-snorm",
    "bc6h-rgb-ufloat",
    "bc6h-rgb-float",
    "bc7-rgba-unorm",
    "bc7-rgba-unorm-srgb",
    "etc2-rgb8unorm",
    "etc2-rgb8unorm-srgb",
    "etc2-rgb8a1unorm",
    "etc2-rgb8a1unorm-srgb",
    "etc2-rgba8unorm",
    "etc2-rgba8unorm-srgb",
    "eac-r11unorm",
    "eac-r11snorm",
    "eac-rg11unorm",
    "eac-rg11snorm",
    "astc-4x4-unorm",
    "astc-4x4-unorm-srgb",
    "astc-5x4-unorm",
    "astc-5x4-unorm-srgb",
    "astc-5x5-unorm",
    "astc-5x5-unorm-srgb",
    "astc-6x5-unorm",
    "astc-6x5-unorm-srgb",
    "astc-6x6-unorm",
    "astc-6x6-unorm-srgb",
    "astc-8x5-unorm",
    "astc-8x5-unorm-srgb",
    "astc-8x6-unorm",
    "astc-8x6-unorm-srgb",
    "astc-8x8-unorm",
    "astc-8x8-unorm-srgb",
    "astc-10x5-unorm",
    "astc-10x5-unorm-srgb",
    "astc-10x6-unorm",
    "astc-10x6-unorm-srgb",
    "astc-10x8-unorm",
    "astc-10x8-unorm-srgb",
    "astc-10x10-unorm",
    "astc-10x10-unorm-srgb",
    "astc-12x10-unorm",
    "astc-12x10-unorm-srgb",
    "astc-12x12-unorm",
    "astc-12x12-unorm-srgb",
];

/// W3C `webgpu.idl` enum `GPUAddressMode` — 3 values.
pub static SPEC_GPUAddressMode: &[&str] = &["clamp-to-edge", "repeat", "mirror-repeat"];

/// W3C `webgpu.idl` enum `GPUFilterMode` — 2 values.
pub static SPEC_GPUFilterMode: &[&str] = &["nearest", "linear"];

/// W3C `webgpu.idl` enum `GPUMipmapFilterMode` — 2 values.
pub static SPEC_GPUMipmapFilterMode: &[&str] = &["nearest", "linear"];

/// W3C `webgpu.idl` enum `GPUCompareFunction` — 8 values.
pub static SPEC_GPUCompareFunction: &[&str] = &[
    "never",
    "less",
    "equal",
    "less-equal",
    "greater",
    "not-equal",
    "greater-equal",
    "always",
];

/// W3C `webgpu.idl` enum `GPUPrimitiveTopology` — 5 values.
pub static SPEC_GPUPrimitiveTopology: &[&str] = &[
    "point-list",
    "line-list",
    "line-strip",
    "triangle-list",
    "triangle-strip",
];

/// W3C `webgpu.idl` enum `GPUFrontFace` — 2 values.
pub static SPEC_GPUFrontFace: &[&str] = &["ccw", "cw"];

/// W3C `webgpu.idl` enum `GPUCullMode` — 3 values.
pub static SPEC_GPUCullMode: &[&str] = &["none", "front", "back"];

/// W3C `webgpu.idl` enum `GPUBlendFactor` — 17 values.
pub static SPEC_GPUBlendFactor: &[&str] = &[
    "zero",
    "one",
    "src",
    "one-minus-src",
    "src-alpha",
    "one-minus-src-alpha",
    "dst",
    "one-minus-dst",
    "dst-alpha",
    "one-minus-dst-alpha",
    "src-alpha-saturated",
    "constant",
    "one-minus-constant",
    "src1",
    "one-minus-src1",
    "src1-alpha",
    "one-minus-src1-alpha",
];

/// W3C `webgpu.idl` enum `GPUBlendOperation` — 5 values.
pub static SPEC_GPUBlendOperation: &[&str] = &["add", "subtract", "reverse-subtract", "min", "max"];

/// W3C `webgpu.idl` enum `GPUIndexFormat` — 2 values.
pub static SPEC_GPUIndexFormat: &[&str] = &["uint16", "uint32"];

/// W3C `webgpu.idl` enum `GPUVertexFormat` — 41 values.
pub static SPEC_GPUVertexFormat: &[&str] = &[
    "uint8",
    "uint8x2",
    "uint8x4",
    "sint8",
    "sint8x2",
    "sint8x4",
    "unorm8",
    "unorm8x2",
    "unorm8x4",
    "snorm8",
    "snorm8x2",
    "snorm8x4",
    "uint16",
    "uint16x2",
    "uint16x4",
    "sint16",
    "sint16x2",
    "sint16x4",
    "unorm16",
    "unorm16x2",
    "unorm16x4",
    "snorm16",
    "snorm16x2",
    "snorm16x4",
    "float16",
    "float16x2",
    "float16x4",
    "float32",
    "float32x2",
    "float32x3",
    "float32x4",
    "uint32",
    "uint32x2",
    "uint32x3",
    "uint32x4",
    "sint32",
    "sint32x2",
    "sint32x3",
    "sint32x4",
    "unorm10-10-10-2",
    "unorm8x4-bgra",
];

/// W3C `webgpu.idl` enum `GPUVertexStepMode` — 2 values.
pub static SPEC_GPUVertexStepMode: &[&str] = &["vertex", "instance"];

/// W3C `webgpu.idl` enum `GPULoadOp` — 2 values.
pub static SPEC_GPULoadOp: &[&str] = &["load", "clear"];

/// W3C `webgpu.idl` enum `GPUStoreOp` — 2 values.
pub static SPEC_GPUStoreOp: &[&str] = &["store", "discard"];

/// Find a value's index in a spec table, or `None`.
/// Used by `quanta-cli`'s codegen self-check and by call sites that
/// want to verify a hand-written enum string is in the spec.
pub fn spec_index(table: &[&str], value: &str) -> Option<u32> {
    table.iter().position(|s| *s == value).map(|i| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every string Quanta hands the JS side via `quanta_create_*`
    /// must be a member of the corresponding spec table. If the spec
    /// renames or removes a value, this test fails — clear signal to
    /// migrate the Quanta code.
    #[test]
    fn quanta_strings_are_spec_subsets() {
        // Texture formats Quanta exposes in `Format` enum.
        for s in [
            "rgba8unorm",
            "bgra8unorm",
            "r8unorm",
            "r16float",
            "r32float",
            "rg32float",
            "rgba16float",
            "rgba32float",
            "depth32float",
        ] {
            assert!(
                spec_index(SPEC_GPUTextureFormat, s).is_some(),
                "Quanta format string {:?} not in spec GPUTextureFormat",
                s,
            );
        }
        // Vertex attribute formats.
        for s in [
            "float32",
            "float32x2",
            "float32x3",
            "float32x4",
            "sint32",
            "sint32x2",
            "sint32x3",
            "sint32x4",
            "uint32",
            "uint32x2",
            "uint32x3",
            "uint32x4",
            "unorm8x4",
        ] {
            assert!(
                spec_index(SPEC_GPUVertexFormat, s).is_some(),
                "vertex format {:?}",
                s
            );
        }
        // Primitive topology.
        for s in [
            "point-list",
            "line-list",
            "line-strip",
            "triangle-list",
            "triangle-strip",
        ] {
            assert!(
                spec_index(SPEC_GPUPrimitiveTopology, s).is_some(),
                "topology {:?}",
                s
            );
        }
        // Cull mode.
        for s in ["none", "front", "back"] {
            assert!(spec_index(SPEC_GPUCullMode, s).is_some(), "cull {:?}", s);
        }
        // Blend factor (Quanta uses 10 of these).
        for s in [
            "zero",
            "one",
            "src-alpha",
            "one-minus-src-alpha",
            "dst-alpha",
            "one-minus-dst-alpha",
            "src",
            "one-minus-src",
            "dst",
            "one-minus-dst",
        ] {
            assert!(
                spec_index(SPEC_GPUBlendFactor, s).is_some(),
                "blend factor {:?}",
                s
            );
        }
        // Blend operation.
        for s in ["add", "subtract", "reverse-subtract", "min", "max"] {
            assert!(
                spec_index(SPEC_GPUBlendOperation, s).is_some(),
                "blend op {:?}",
                s
            );
        }
        // Filter, mipmap filter (same string set).
        for s in ["nearest", "linear"] {
            assert!(
                spec_index(SPEC_GPUFilterMode, s).is_some(),
                "filter {:?}",
                s
            );
            assert!(
                spec_index(SPEC_GPUMipmapFilterMode, s).is_some(),
                "mip filter {:?}",
                s
            );
        }
        // Address mode.
        for s in ["clamp-to-edge", "repeat", "mirror-repeat"] {
            assert!(
                spec_index(SPEC_GPUAddressMode, s).is_some(),
                "address {:?}",
                s
            );
        }
        // Compare function.
        for s in [
            "never",
            "less",
            "equal",
            "less-equal",
            "greater",
            "not-equal",
            "greater-equal",
            "always",
        ] {
            assert!(
                spec_index(SPEC_GPUCompareFunction, s).is_some(),
                "compare {:?}",
                s
            );
        }
        // Vertex step mode.
        for s in ["vertex", "instance"] {
            assert!(
                spec_index(SPEC_GPUVertexStepMode, s).is_some(),
                "step {:?}",
                s
            );
        }
        // Index format.
        for s in ["uint16", "uint32"] {
            assert!(
                spec_index(SPEC_GPUIndexFormat, s).is_some(),
                "index fmt {:?}",
                s
            );
        }
        // Load/store op.
        for s in ["load", "clear"] {
            assert!(spec_index(SPEC_GPULoadOp, s).is_some(), "load op {:?}", s);
        }
        for s in ["store", "discard"] {
            assert!(spec_index(SPEC_GPUStoreOp, s).is_some(), "store op {:?}", s);
        }
    }
}

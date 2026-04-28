//! Emit `src/driver/webgpu/generated_codes.rs`.
//!
//! For every project-relevant IDL enum, produces:
//! - A `pub static SPEC_<EnumName>: &[&str]` containing the spec
//!   values in source order.
//! - A `pub fn <enum_name>_index(s: &str) -> Option<u32>` lookup.
//!
//! The hand-written Quanta codes (`mod format`, `mod blend_factor`, …)
//! stay where they are; their integers index into these spec tables
//! at the JS side via `web/src/generated/codes.ts`. Conformance
//! between the two is enforced at codegen time: any string Quanta
//! advertises that isn't in the spec table fails the build.

use crate::parse::ParsedIdl;
use std::fmt::Write;

pub fn emit(parsed: &ParsedIdl) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "//! GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate."
    )
    .unwrap();
    writeln!(&mut out, "//!").unwrap();
    writeln!(
        &mut out,
        "//! Source: web/webgpu.idl  (sha256: {})",
        parsed.source_hash
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! Generator: crates/quanta-codegen (B′ track of the FFI TCB shrink)."
    )
    .unwrap();
    writeln!(&mut out, "//!").unwrap();
    writeln!(
        &mut out,
        "//! Each `SPEC_*` table holds every value that the W3C `webgpu.idl`"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! lists for the corresponding enum, in source order. Quanta's"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! `src/driver/webgpu/ffi.rs` enum-code modules (`format`,"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! `blend_factor`, …) define a *subset* of these strings — every"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! string they expose is required to appear here. The codegen"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! cross-checks at build time; the Rust unit test"
    )
    .unwrap();
    writeln!(
        &mut out,
        "//! `tests::quanta_strings_are_spec_subsets` exercises it again at"
    )
    .unwrap();
    writeln!(&mut out, "//! `cargo test`.").unwrap();
    writeln!(&mut out).unwrap();
    // The generated table names mirror the WebIDL enum names verbatim
    // (`SPEC_GPUTextureFormat`) so reviewers can grep across the IDL,
    // generated Rust, and generated TS without re-translating cases.
    // That trips Rust's UPPER_SNAKE_CASE lint; allow it on this module.
    writeln!(&mut out, "#![allow(dead_code, non_upper_case_globals)]").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "/// SHA-256 of `web/webgpu.idl` at codegen time.").unwrap();
    writeln!(
        &mut out,
        "pub const WEBGPU_IDL_SHA256: &str = \"{}\";",
        parsed.source_hash
    )
    .unwrap();
    writeln!(&mut out).unwrap();

    for e in parsed.enums.iter().filter(|e| e.project_relevant) {
        let const_name = format!("SPEC_{}", e.name);
        writeln!(
            &mut out,
            "/// W3C `webgpu.idl` enum `{}` — {} values.",
            e.name,
            e.values.len()
        )
        .unwrap();
        writeln!(&mut out, "pub static {}: &[&str] = &[", const_name,).unwrap();
        for v in &e.values {
            writeln!(&mut out, "    {:?},", v).unwrap();
        }
        writeln!(&mut out, "];").unwrap();
        writeln!(&mut out).unwrap();
    }

    // Helpers + tests live at the bottom so the constant tables can
    // be scanned without scrolling through them.
    writeln!(
        &mut out,
        "/// Find a value's index in a spec table, or `None`."
    )
    .unwrap();
    writeln!(
        &mut out,
        "/// Used by `quanta-cli`'s codegen self-check and by call sites that"
    )
    .unwrap();
    writeln!(
        &mut out,
        "/// want to verify a hand-written enum string is in the spec."
    )
    .unwrap();
    writeln!(
        &mut out,
        "pub fn spec_index(table: &[&str], value: &str) -> Option<u32> {{"
    )
    .unwrap();
    writeln!(
        &mut out,
        "    table.iter().position(|s| *s == value).map(|i| i as u32)"
    )
    .unwrap();
    writeln!(&mut out, "}}").unwrap();
    writeln!(&mut out).unwrap();

    out.push_str(QUANTA_SUBSET_TEST);
    out
}

/// Hand-written test block appended to the generated file. Runs
/// against the *generated* tables so any spec drift fails `cargo
/// test`. The list of (Quanta name, spec name) pairs lives here
/// because it encodes Quanta's subset selection — a fact about the
/// project, not about the IDL.
const QUANTA_SUBSET_TEST: &str = r#"
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
            "rgba8unorm", "bgra8unorm", "r8unorm", "r16float", "r32float",
            "rg32float", "rgba16float", "rgba32float", "depth32float",
        ] {
            assert!(
                spec_index(SPEC_GPUTextureFormat, s).is_some(),
                "Quanta format string {:?} not in spec GPUTextureFormat", s,
            );
        }
        // Vertex attribute formats.
        for s in [
            "float32", "float32x2", "float32x3", "float32x4",
            "sint32", "sint32x2", "sint32x3", "sint32x4",
            "uint32", "uint32x2", "uint32x3", "uint32x4",
            "unorm8x4",
        ] {
            assert!(spec_index(SPEC_GPUVertexFormat, s).is_some(), "vertex format {:?}", s);
        }
        // Primitive topology.
        for s in ["point-list", "line-list", "line-strip", "triangle-list", "triangle-strip"] {
            assert!(spec_index(SPEC_GPUPrimitiveTopology, s).is_some(), "topology {:?}", s);
        }
        // Cull mode.
        for s in ["none", "front", "back"] {
            assert!(spec_index(SPEC_GPUCullMode, s).is_some(), "cull {:?}", s);
        }
        // Blend factor (Quanta uses 10 of these).
        for s in [
            "zero", "one", "src-alpha", "one-minus-src-alpha",
            "dst-alpha", "one-minus-dst-alpha", "src", "one-minus-src",
            "dst", "one-minus-dst",
        ] {
            assert!(spec_index(SPEC_GPUBlendFactor, s).is_some(), "blend factor {:?}", s);
        }
        // Blend operation.
        for s in ["add", "subtract", "reverse-subtract", "min", "max"] {
            assert!(spec_index(SPEC_GPUBlendOperation, s).is_some(), "blend op {:?}", s);
        }
        // Filter, mipmap filter (same string set).
        for s in ["nearest", "linear"] {
            assert!(spec_index(SPEC_GPUFilterMode, s).is_some(), "filter {:?}", s);
            assert!(spec_index(SPEC_GPUMipmapFilterMode, s).is_some(), "mip filter {:?}", s);
        }
        // Address mode.
        for s in ["clamp-to-edge", "repeat", "mirror-repeat"] {
            assert!(spec_index(SPEC_GPUAddressMode, s).is_some(), "address {:?}", s);
        }
        // Compare function.
        for s in [
            "never", "less", "equal", "less-equal", "greater",
            "not-equal", "greater-equal", "always",
        ] {
            assert!(spec_index(SPEC_GPUCompareFunction, s).is_some(), "compare {:?}", s);
        }
        // Vertex step mode.
        for s in ["vertex", "instance"] {
            assert!(spec_index(SPEC_GPUVertexStepMode, s).is_some(), "step {:?}", s);
        }
        // Index format.
        for s in ["uint16", "uint32"] {
            assert!(spec_index(SPEC_GPUIndexFormat, s).is_some(), "index fmt {:?}", s);
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
"#;

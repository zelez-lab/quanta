//! Emit `specs/verify/lean/Quanta/Idl/WebGpuSpec.lean`.
//!
//! Step **B″** of the FFI TCB shrink track. Walks the same parsed IDL
//! AST that drives the Rust + TS spec tables and produces a Lean
//! `def webGpuSpec : Quanta.Idl.WebGpuSpec` literal. The conformance
//! theorem in `Quanta.Theorems.IdlConformance` then discharges the
//! enum-string component of T1707 (`quanta_abi_faithful`) against
//! that literal — lifting it from axiom to theorem.
//!
//! Why one parsed AST → three emitters: the lockstep hazard between
//! Rust constants, TS arrays, and Lean spec data collapses to a
//! single source. A spec drift that would previously have escaped to
//! runtime now fails `cargo test` (Rust subset check),
//! `assertSpecSubset()` (TS load-time check), *and* `lake build`
//! (Lean conformance theorem) on the next regeneration.
//!
//! The Lean output is intentionally minimal — only what the
//! conformance proof needs. Future B″ commits will widen the surface
//! to dictionary members and method signatures (the `IdlType`,
//! `DictionaryDecl`, and `MethodDecl` shapes already declared in
//! `Quanta.Idl`); when that happens this emitter grows alongside.

use crate::parse::ParsedIdl;
use std::fmt::Write;

pub fn emit(parsed: &ParsedIdl) -> String {
    let mut out = String::new();
    writeln!(
        &mut out,
        "/- GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "Source: web/webgpu.idl  (sha256: {})",
        parsed.source_hash
    )
    .unwrap();
    writeln!(
        &mut out,
        "Generator: crates/lang/quanta-codegen (B″ track of the FFI TCB shrink)."
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "This file is the Lean mirror of the W3C WebGPU IDL — same data"
    )
    .unwrap();
    writeln!(
        &mut out,
        "as `src/webgpu_generated_codes.rs` and `web/src/generated/codes.ts`,"
    )
    .unwrap();
    writeln!(
        &mut out,
        "expressed as a `Quanta.Idl.WebGpuSpec` literal so the conformance"
    )
    .unwrap();
    writeln!(
        &mut out,
        "theorem `Quanta.Theorems.IdlConformance.quanta_strings_in_spec`"
    )
    .unwrap();
    writeln!(
        &mut out,
        "can discharge T1707 (the enum-string component of A11) by"
    )
    .unwrap();
    writeln!(&mut out, "`native_decide` against it.").unwrap();
    writeln!(&mut out, "-/").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "import Quanta.Idl").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "namespace Quanta.Idl").unwrap();
    writeln!(&mut out).unwrap();

    writeln!(
        &mut out,
        "/-- WebGPU IDL spec data, generated from `web/webgpu.idl`."
    )
    .unwrap();
    writeln!(
        &mut out,
        "    Only the project-relevant enums are included — the Quanta"
    )
    .unwrap();
    writeln!(
        &mut out,
        "    FFI does not ship the full IDL surface yet. Future B″"
    )
    .unwrap();
    writeln!(
        &mut out,
        "    passes will widen this with dictionaries and methods. -/"
    )
    .unwrap();
    writeln!(&mut out, "def webGpuSpec : WebGpuSpec :=").unwrap();
    writeln!(&mut out, "  {{ sourceSha256 :=").unwrap();
    writeln!(&mut out, "      \"{}\",", parsed.source_hash).unwrap();
    writeln!(&mut out, "    enums := [").unwrap();

    let kept: Vec<_> = parsed.enums.iter().filter(|e| e.project_relevant).collect();
    for (i, e) in kept.iter().enumerate() {
        writeln!(
            &mut out,
            "      -- W3C `webgpu.idl` enum `{}` — {} values.",
            e.name,
            e.values.len()
        )
        .unwrap();
        writeln!(&mut out, "      {{ name := \"{}\",", e.name).unwrap();
        writeln!(&mut out, "        values := [").unwrap();
        for (j, v) in e.values.iter().enumerate() {
            let comma = if j + 1 < e.values.len() { "," } else { "" };
            writeln!(&mut out, "          \"{}\"{}", escape_lean_string(v), comma).unwrap();
        }
        writeln!(
            &mut out,
            "        ] }}{}",
            if i + 1 < kept.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(&mut out, "    ],").unwrap();
    writeln!(&mut out, "    methods := [").unwrap();
    for (i, m) in parsed.methods.iter().enumerate() {
        let comma = if i + 1 < parsed.methods.len() {
            ","
        } else {
            ""
        };
        let params = if m.params.is_empty() {
            "[]".to_string()
        } else {
            let entries: Vec<String> = m
                .params
                .iter()
                .map(|p| {
                    format!(
                        "{{ typeName := \"{}\", optional := {} }}",
                        escape_lean_string(&p.type_name),
                        if p.optional { "true" } else { "false" },
                    )
                })
                .collect();
            format!("[{}]", entries.join(", "))
        };
        writeln!(
            &mut out,
            "      {{ interfaceName := \"{}\", methodName := \"{}\", \
             requiredArity := {}, maxArity := {}, isVariadic := {}, \
             params := {} }}{}",
            escape_lean_string(&m.interface_name),
            escape_lean_string(&m.method_name),
            m.required_arity,
            m.max_arity,
            if m.is_variadic { "true" } else { "false" },
            params,
            comma,
        )
        .unwrap();
    }
    writeln!(&mut out, "    ] }}").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "end Quanta.Idl").unwrap();

    out
}

/// WebIDL enum values are lowercase ASCII letters, digits, and `-` — no
/// quotes or backslashes — so no escaping is needed in practice. Guard
/// the rare oddball anyway: any future IDL that uses `\` or `"` would
/// otherwise produce malformed Lean source.
fn escape_lean_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            other => out.push(other),
        }
    }
    out
}

//! Emit `web/src/generated/codes.ts`.
//!
//! Mirrors `emit_rust.rs` for the TypeScript side: each project-
//! relevant IDL enum becomes a `readonly string[]` whose elements
//! match the spec source order. `web/src/codes.ts` (hand-written)
//! imports these and asserts that every project string is a member.

use crate::parse::ParsedIdl;
use std::fmt::Write;

pub fn emit(parsed: &ParsedIdl) -> String {
    let mut out = String::new();
    out.push_str("// GENERATED — DO NOT EDIT. Run `quanta codegen webgpu` to regenerate.\n");
    out.push_str("//\n");
    let _ = writeln!(
        &mut out,
        "// Source: web/webgpu.idl  (sha256: {})",
        parsed.source_hash,
    );
    out.push_str("// Generator: crates/lang/quanta-codegen (B′ track of the FFI TCB shrink).\n");
    out.push_str("//\n");
    out.push_str("// Each `SPEC_*` array holds every value that the W3C `webgpu.idl`\n");
    out.push_str("// lists for the corresponding enum, in source order. Quanta's\n");
    out.push_str("// hand-written `web/src/codes.ts` (and the Rust mirror in\n");
    out.push_str("// `src/driver/webgpu/ffi.rs`) define a *subset* — every string they\n");
    out.push_str("// expose is required to appear here. Cross-checked at codegen and at\n");
    out.push_str("// runtime via `assertSpecSubset()` below.\n\n");

    let _ = writeln!(
        &mut out,
        "export const WEBGPU_IDL_SHA256 = {:?};",
        parsed.source_hash,
    );
    out.push('\n');

    for e in parsed.enums.iter().filter(|e| e.project_relevant) {
        let _ = writeln!(
            &mut out,
            "/** W3C `webgpu.idl` enum `{}` — {} values. */",
            e.name,
            e.values.len(),
        );
        let _ = writeln!(
            &mut out,
            "export const SPEC_{}: readonly string[] = [",
            e.name,
        );
        for v in &e.values {
            let _ = writeln!(&mut out, "  {:?},", v);
        }
        out.push_str("];\n\n");
    }

    out.push_str(SUBSET_HELPER);
    out
}

const SUBSET_HELPER: &str = r#"/**
 * Throw if any of `quantaStrings` is missing from `specTable`.
 * Called from `web/src/codes.ts` once at module-init time so a
 * post-deploy spec drift surfaces as a load error, not a silent
 * mis-dispatch on first use.
 */
export function assertSpecSubset(
  enumName: string,
  specTable: readonly string[],
  quantaStrings: readonly string[],
): void {
  for (const s of quantaStrings) {
    if (!specTable.includes(s)) {
      throw new Error(
        `quanta-codegen: ${enumName} value ${JSON.stringify(s)} is not in the spec ` +
          `(spec sha256: ${WEBGPU_IDL_SHA256.slice(0, 12)}…). ` +
          `Either fix Quanta's enum strings or re-run \`quanta codegen webgpu\`.`,
      );
    }
  }
}
"#;

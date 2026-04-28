//! Quanta WebIDL → Rust + TypeScript codegen.
//!
//! Step B′ of the FFI TCB shrink track. Reads the W3C `webgpu.idl`
//! and produces:
//!
//! 1. `src/driver/webgpu/generated_codes.rs` — Rust constants for
//!    every WebGPU enum we use (texture format, blend factor, …).
//!    Each constant's value is a small integer; the integer's
//!    canonical name string is taken straight from the spec.
//! 2. `web/src/generated/codes.ts` — the matching TypeScript tables
//!    that convert each integer back to the IDL string.
//!
//! Both files share the same parsed AST, so the lockstep hazard
//! (Rust says `format::RGBA8UNORM = 0` but TS table[0] says
//! `"rgba8unorm-srgb"`) becomes impossible by construction.
//!
//! Why a separate crate (vs. inlined in `quanta-cli`):
//! - The codegen will eventually be reused for B″ (Lean WebIDL
//!   conformance) and B (WGSL grammar mirror). Same parser, three
//!   targets. Putting the parsing layer in its own crate avoids
//!   coupling the CLI binary to those follow-ons.
//! - It also keeps `quanta-cli`'s dep tree shallow — `weedle` only
//!   pulls in for code generation, not for `quanta build` /
//!   `quanta serve`.
//!
//! Public surface: one `generate(idl_path, project_root)` entry that
//! does end-to-end parse + emit + write. Used by
//! `quanta-cli`'s `codegen` subcommand.

mod emit_rust;
mod emit_ts;
mod parse;

use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Top-level entry: parse the IDL at `idl_path`, generate both
/// outputs, and write them under `project_root`.
///
/// Files written:
/// - `<project_root>/src/webgpu_generated_codes.rs`  (top-level so
///   `cargo test --lib` on native runs the spec-subset tests; the
///   data is wasm32-and-native compatible.)
/// - `<project_root>/web/src/generated/codes.ts`
///
/// Existing files are overwritten. Both outputs include a header
/// comment naming this crate as the source so reviewers can tell
/// at-a-glance not to hand-edit them.
pub fn generate(idl_path: &Path, project_root: &Path) -> Result<()> {
    let idl_text = std::fs::read_to_string(idl_path)?;
    let parsed = parse::parse(&idl_text)?;
    let report = parse::summarize(&parsed);
    eprintln!(
        "[quanta-codegen] parsed {} enums, kept {} project-relevant",
        report.enums_total, report.enums_kept
    );

    let rust_out = project_root.join("src").join("webgpu_generated_codes.rs");
    let ts_dir: PathBuf = project_root.join("web/src/generated");
    let ts_out = ts_dir.join("codes.ts");

    std::fs::create_dir_all(&ts_dir)?;

    let rust_text = emit_rust::emit(&parsed);
    let ts_text = emit_ts::emit(&parsed);
    std::fs::write(&rust_out, rust_text)?;
    std::fs::write(&ts_out, ts_text)?;
    // Re-format the Rust output so it survives `cargo fmt --check`
    // on subsequent regenerations. `rustfmt` is part of every Rust
    // toolchain via rustup; if it's not available we warn but don't
    // fail — the generated file is still functional.
    if let Err(e) = run_rustfmt(&rust_out) {
        eprintln!(
            "[quanta-codegen] warning: rustfmt on {} failed ({}); \
             you may need to run `cargo fmt` manually after codegen.",
            rust_out.display(),
            e
        );
    }
    eprintln!("[quanta-codegen] wrote {}", rust_out.display());
    eprintln!("[quanta-codegen] wrote {}", ts_out.display());
    Ok(())
}

fn run_rustfmt(file: &Path) -> Result<()> {
    let status = std::process::Command::new("rustfmt")
        .arg(file)
        .status()
        .map_err(|e| format!("failed to spawn rustfmt: {e}"))?;
    if !status.success() {
        return Err(format!("rustfmt exited with {status}").into());
    }
    Ok(())
}

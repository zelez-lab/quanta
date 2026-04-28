//! Workspace path discovery.
//!
//! Every subcommand needs to know where the workspace root is so it
//! can find `web/`, `examples/`, and `target/`. We walk up from the
//! current directory looking for the workspace `Cargo.toml`. This
//! also lets the CLI be invoked from any subdirectory.

use std::path::{Path, PathBuf};

use crate::Result;

/// Find the directory containing the workspace `Cargo.toml`.
///
/// Walks upward from the current directory and stops at the first
/// `Cargo.toml` whose contents include `[workspace]`.
pub fn root() -> Result<PathBuf> {
    let mut here = std::env::current_dir()?;
    loop {
        let candidate = here.join("Cargo.toml");
        if candidate.is_file() && is_workspace_manifest(&candidate)? {
            return Ok(here);
        }
        if !here.pop() {
            return Err("not inside a Quanta workspace (no [workspace] Cargo.toml found)".into());
        }
    }
}

fn is_workspace_manifest(path: &Path) -> Result<bool> {
    let text = std::fs::read_to_string(path)?;
    Ok(text.contains("[workspace]"))
}

/// List of valid example crate names, kept in lockstep with
/// `examples/` directory contents. Rather than glob the directory we
/// hard-code so unknown names are rejected with a clear error.
pub const WEB_EXAMPLES: &[&str] = &["web_add_one", "web_triangle", "web_textured"];

/// Validate an example name against the known list (or accept "all").
pub fn resolve_examples(name: &str) -> Result<Vec<&'static str>> {
    if name == "all" {
        return Ok(WEB_EXAMPLES.to_vec());
    }
    for &known in WEB_EXAMPLES {
        if known == name {
            return Ok(vec![known]);
        }
    }
    Err(format!(
        "unknown example '{name}' (valid: {})",
        WEB_EXAMPLES.join(", ")
    )
    .into())
}

/// Cargo crate name corresponding to an `examples/web_<x>/` dir.
/// `web_add_one` → `web-add-one` (Cargo's convention swaps `_` for `-`
/// in package names but not in path components).
pub fn cargo_name(example: &str) -> String {
    example.replace('_', "-")
}

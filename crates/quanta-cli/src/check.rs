//! `quanta check` — run the pre-commit-equivalent checks in one verb.
//!
//! Order:
//! 1. `cargo fmt --all -- --check`  (formatting)
//! 2. `cargo clippy --workspace -- -D warnings`  (lints, native target)
//! 3. `cargo clippy --target wasm32-unknown-unknown --features webgpu \
//!     --no-default-features -- -D warnings`  (lints, wasm target)
//! 4. `tsc --noEmit` inside `web/`  (TypeScript type check)
//!
//! Each step's command is echoed so it's clear which one failed.

use std::path::Path;
use std::process::Command;

use crate::Result;
use crate::workspace;

pub fn run() -> Result<()> {
    let root = workspace::root()?;

    step(
        "cargo fmt --check",
        &root,
        "cargo",
        &["fmt", "--all", "--", "--check"],
    )?;

    step(
        "cargo clippy (native)",
        &root,
        "cargo",
        &["clippy", "--workspace", "--", "-D", "warnings"],
    )?;

    step(
        "cargo clippy (wasm32 + webgpu)",
        &root,
        "cargo",
        &[
            "clippy",
            "--target",
            "wasm32-unknown-unknown",
            "--features",
            "webgpu",
            "--no-default-features",
            "--",
            "-D",
            "warnings",
        ],
    )?;

    let web_dir = root.join("web");
    let local_tsc = web_dir.join("node_modules").join(".bin").join("tsc");
    if local_tsc.exists() {
        step(
            "tsc --noEmit (web/)",
            &web_dir,
            local_tsc.to_str().ok_or("tsc path is not utf-8")?,
            &["--noEmit"],
        )?;
    } else {
        eprintln!(
            "[quanta check] skipping tsc (web/node_modules absent — run `quanta build web` first)"
        );
    }

    eprintln!("[quanta check] all checks passed");
    Ok(())
}

fn step(label: &str, cwd: &Path, program: &str, args: &[&str]) -> Result<()> {
    eprintln!("[quanta check] {label}");
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .map_err(|e| format!("failed to run {program}: {e}"))?;
    if !status.success() {
        return Err(format!("{label} failed (status {status})").into());
    }
    Ok(())
}

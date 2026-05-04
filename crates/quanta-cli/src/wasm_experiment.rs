//! `quanta wasm-experiment` — research harness for the long-term
//! WASM-route translator (steps 058 / 059 / 080).
//!
//! Takes a Rust source file containing a `#[quanta::kernel]` function,
//! wraps it as a `wasm32-unknown-unknown` lib crate, runs `cargo build`,
//! and dumps:
//! - `kernel.wasm` (raw binary)
//! - `kernel.wat` (text form, decoded via `wasm2wat` if available)
//! - `kernel-summary.txt` (op counts, import list, function list)
//!
//! Used to scope what the WASM → KernelOps lowering pass needs to
//! handle. We don't *use* the WASM today — `#[quanta::kernel]` still
//! routes through the syntax-tree parser. This is purely an
//! exploration tool until the lowering pass is built.
//!
//! Usage:
//!   quanta wasm-experiment examples/hello_quanta.rs
//!   quanta wasm-experiment crates/quanta-bench/src/saxpy.rs --out-dir /tmp/wasm

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::Result;

pub(crate) fn run(source: &str, out_dir: &str) -> Result<()> {
    let source_path = Path::new(source);
    if !source_path.is_file() {
        return Err(format!("source file not found: {source}").into());
    }
    let source_text = fs::read_to_string(source_path)?;

    let workspace_root = crate::workspace::root()?;
    let scratch_dir = workspace_root.join("target/wasm-experiment/scratch");
    let out_path = if Path::new(out_dir).is_absolute() {
        PathBuf::from(out_dir)
    } else {
        workspace_root.join(out_dir)
    };
    fs::create_dir_all(&out_path)?;

    eprintln!("[wasm-experiment] source:    {}", source_path.display());
    eprintln!("[wasm-experiment] scratch:   {}", scratch_dir.display());
    eprintln!("[wasm-experiment] out:       {}", out_path.display());

    // Wrap the source in a wasm32 lib crate. We add `quanta` as a path
    // dependency so the macros expand. The kernel attribute is
    // preserved verbatim — that is the input to the future WASM-route
    // translator.
    write_scratch_crate(&scratch_dir, &workspace_root, &source_text)?;

    eprintln!("[wasm-experiment] cargo build --target wasm32-unknown-unknown ...");
    let status = Command::new("cargo")
        .args([
            "build",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
            "--manifest-path",
        ])
        .arg(scratch_dir.join("Cargo.toml"))
        .status()?;
    if !status.success() {
        return Err("cargo build for wasm32 failed".into());
    }

    let wasm_src =
        scratch_dir.join("target/wasm32-unknown-unknown/release/quanta_wasm_experiment.wasm");
    if !wasm_src.is_file() {
        return Err(format!("wasm artifact missing: {}", wasm_src.display()).into());
    }
    let wasm_dst = out_path.join("kernel.wasm");
    fs::copy(&wasm_src, &wasm_dst)?;
    eprintln!("[wasm-experiment] wrote {}", wasm_dst.display());

    // Decode to .wat if wasm2wat is on PATH (best-effort).
    let wat_dst = out_path.join("kernel.wat");
    match Command::new("wasm2wat")
        .arg(&wasm_dst)
        .arg("-o")
        .arg(&wat_dst)
        .status()
    {
        Ok(s) if s.success() => {
            eprintln!("[wasm-experiment] wrote {}", wat_dst.display());
        }
        Ok(_) | Err(_) => {
            eprintln!(
                "[wasm-experiment] wasm2wat not available — install with `brew install wabt` \
                 (macOS) or `apt-get install wabt` (Linux) to also dump kernel.wat"
            );
        }
    }

    // Summary: file size + section breakdown via wasm-objdump if present.
    let summary_dst = out_path.join("kernel-summary.txt");
    let mut summary = String::new();
    summary.push_str(&format!(
        "wasm size: {} bytes\n",
        fs::metadata(&wasm_dst)?.len()
    ));
    if let Ok(out) = Command::new("wasm-objdump")
        .args(["-x", "-d"])
        .arg(&wasm_dst)
        .output()
        && out.status.success()
    {
        summary.push_str("--- wasm-objdump -x ---\n");
        summary.push_str(&String::from_utf8_lossy(&out.stdout));
    }
    fs::write(&summary_dst, summary)?;
    eprintln!("[wasm-experiment] wrote {}", summary_dst.display());

    eprintln!("[wasm-experiment] done");
    Ok(())
}

fn write_scratch_crate(scratch_dir: &Path, workspace_root: &Path, source_text: &str) -> Result<()> {
    let src_dir = scratch_dir.join("src");
    fs::create_dir_all(&src_dir)?;

    let workspace_root_str = workspace_root.to_string_lossy();
    let cargo_toml = format!(
        r#"[package]
name = "quanta-wasm-experiment"
version = "0.0.0"
edition = "2024"

# Stand-alone — must NOT join the parent quanta workspace, otherwise
# cargo refuses the build (parent workspace doesn't list us).
[workspace]

[lib]
crate-type = ["cdylib"]
path = "src/lib.rs"

[dependencies]
quanta = {{ path = "{root}", default-features = false, features = ["jit"] }}

[profile.release]
opt-level = 3
lto = false
codegen-units = 1
panic = "abort"
strip = "debuginfo"
"#,
        root = workspace_root_str,
    );
    fs::write(scratch_dir.join("Cargo.toml"), cargo_toml)?;

    // Strip `fn main(...)` from the user file — lib crate has no entry
    // and the host-side dispatch code references symbols (`init`, etc)
    // that don't exist on wasm32 without the `std` feature. We only
    // care about the kernel function bodies for this experiment.
    let body = strip_main(source_text);

    let lib_rs = format!(
        "//! Auto-generated by `quanta wasm-experiment` — do not edit.\n\
         //!\n\
         //! Wraps a single user-provided source file as a wasm32 lib so\n\
         //! rustc emits the WASM rustc would emit for that kernel after\n\
         //! all monomorphization. The output WASM is the input to the\n\
         //! future WASM-route translator (steps 058 / 059 / 080).\n\
         #![allow(unused, dead_code)]\n\n\
         {body}\n",
        body = body,
    );
    fs::write(src_dir.join("lib.rs"), lib_rs)?;
    Ok(())
}

/// Naively strip a `fn main(...)` definition from the source text by
/// finding the `fn main` token and removing the matching brace block.
/// The wasm-experiment harness only cares about the kernel function;
/// host-side dispatch code shouldn't compile on wasm32 anyway.
fn strip_main(src: &str) -> String {
    let Some(start) = src.find("fn main") else {
        return src.to_string();
    };
    // Find the opening `{` of the function body.
    let after_fn = &src[start..];
    let Some(brace_off) = after_fn.find('{') else {
        return src.to_string();
    };
    let mut depth = 0i32;
    let mut end = None;
    for (i, ch) in after_fn[brace_off..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(brace_off + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(end) = end else {
        return src.to_string();
    };
    let mut out = String::with_capacity(src.len());
    out.push_str(&src[..start]);
    out.push_str(&src[start + end..]);
    out
}

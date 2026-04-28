//! `quanta build web` — compile glue.ts + the wasm crate, stage outputs.
//!
//! Pipeline:
//! 1. Run `tsc` inside `web/` to compile `web/src/*.ts` → `web/dist/*.js`.
//!    `tsc` is invoked through `web/node_modules/.bin/tsc`; if the
//!    `node_modules` dir is missing, we fall back to `npx tsc` and let
//!    npm warn about it.
//! 2. Run `cargo build --target wasm32-unknown-unknown -p web-<name>`
//!    for each requested example.
//! 3. Copy `target/.../web_<name>.wasm` and the contents of
//!    `web/dist/` into `examples/web_<name>/`.

use std::path::Path;
use std::process::Command;

use crate::Result;
use crate::workspace;

/// Build the named example (or "all") at the given profile.
pub fn web(example: &str, profile: &str) -> Result<()> {
    let root = workspace::root()?;
    let examples = workspace::resolve_examples(example)?;

    compile_typescript(&root)?;
    for name in &examples {
        build_wasm(&root, name, profile)?;
        stage(&root, name, profile)?;
    }
    Ok(())
}

fn compile_typescript(root: &Path) -> Result<()> {
    let web_dir = root.join("web");
    if !web_dir.is_dir() {
        return Err("web/ directory not found".into());
    }

    let local_tsc = web_dir.join("node_modules").join(".bin").join("tsc");
    let mut cmd = if local_tsc.exists() {
        let mut c = Command::new(local_tsc);
        c.current_dir(&web_dir);
        c
    } else {
        eprintln!("[quanta build] web/node_modules not found; running `npm install --silent` once");
        run(
            "npm",
            &["install", "--silent", "--no-audit", "--no-fund"],
            &web_dir,
        )?;
        let mut c = Command::new(web_dir.join("node_modules").join(".bin").join("tsc"));
        c.current_dir(&web_dir);
        c
    };

    eprintln!("[quanta build] tsc web/src → web/dist");
    let status = cmd
        .status()
        .map_err(|e| format!("failed to run tsc: {e}"))?;
    if !status.success() {
        return Err(format!("tsc exited with status {status}").into());
    }
    Ok(())
}

fn build_wasm(root: &Path, example: &str, profile: &str) -> Result<()> {
    let crate_name = workspace::cargo_name(example);
    let mut args: Vec<String> = vec![
        "build".into(),
        "--target".into(),
        "wasm32-unknown-unknown".into(),
        "-p".into(),
        crate_name,
    ];
    if profile == "release" {
        args.push("--release".into());
    } else if profile != "dev" && profile != "debug" {
        args.push("--profile".into());
        args.push(profile.into());
    }
    eprintln!("[quanta build] cargo {}", args.join(" "));
    run(
        "cargo",
        &args.iter().map(String::as_str).collect::<Vec<_>>(),
        root,
    )
}

fn stage(root: &Path, example: &str, profile: &str) -> Result<()> {
    let profile_dir = if profile == "dev" || profile == "debug" {
        "debug"
    } else {
        // Custom profiles land in `target/<triple>/<profile>/`; release
        // lands in `target/<triple>/release/`.
        profile
    };
    let wasm_src = root
        .join("target/wasm32-unknown-unknown")
        .join(profile_dir)
        .join(format!("{example}.wasm"));
    let dst = root.join("examples").join(example);
    if !wasm_src.is_file() {
        return Err(format!("wasm artifact missing: {}", wasm_src.display()).into());
    }

    // Clear stale build artifacts in the example dir so a renamed
    // module (e.g. yesterday's `glue.js` after the entry-point rename)
    // doesn't linger and silently shadow the new one. We only delete
    // build-output extensions; index.html, src/, Cargo.toml stay.
    for entry in std::fs::read_dir(&dst)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let path = entry.path();
        let kept = path
            .extension()
            .and_then(|s| s.to_str())
            .is_none_or(|ext| !matches!(ext, "js" | "wasm" | "map"));
        if !kept {
            std::fs::remove_file(&path)?;
        }
    }

    let wasm_dst = dst.join(format!("{example}.wasm"));
    std::fs::copy(&wasm_src, &wasm_dst)?;

    let dist = root.join("web").join("dist");
    for entry in std::fs::read_dir(&dist)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let to = dst.join(entry.file_name());
            std::fs::copy(entry.path(), to)?;
        }
    }
    eprintln!(
        "[quanta build] ready: {}/index.html (wasm + quanta.js in place)",
        dst.display()
    );
    Ok(())
}

fn run(program: &str, args: &[&str], cwd: &Path) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .map_err(|e| format!("failed to run {program}: {e}"))?;
    if !status.success() {
        return Err(format!("{program} exited with status {status}").into());
    }
    Ok(())
}

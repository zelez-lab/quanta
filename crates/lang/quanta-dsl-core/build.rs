fn main() {
    stamp_build_rev();
}

/// Stamp `QUANTA_BUILD_REV` for the stale-compiler handshake.
///
/// "unknown" outside a tracked git checkout — crates.io and vendored
/// copies land there (a vendored copy inside a CONSUMER's repo is
/// detected via `ls-files`: this crate's manifest isn't tracked in
/// that repo) — and both handshake sides then compare equal, silent.
fn stamp_build_rev() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let git = |args: &[&str]| -> Option<String> {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&manifest_dir)
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    };

    let tracked = git(&["ls-files", "--error-unmatch", "Cargo.toml"]).is_some();
    let rev = if tracked {
        git(&["describe", "--always", "--dirty", "--exclude", "*"])
    } else {
        None
    };
    println!(
        "cargo:rustc-env=QUANTA_BUILD_REV={}",
        rev.unwrap_or_else(|| "unknown".to_string())
    );

    // Watch ONLY paths that exist (a missing rerun-if-changed path
    // makes cargo treat the script as always-dirty and the whole
    // dependent tree rebuilds forever), and watch what actually moves:
    // HEAD (branch switches, in the per-worktree git dir) and the
    // current branch's ref + packed-refs (commits, in the COMMON git
    // dir — different from the worktree dir under `git worktree`).
    let exists_then_watch = |p: String| {
        if std::path::Path::new(&p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    };
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        exists_then_watch(format!("{git_dir}/HEAD"));
    }
    if let Some(common) = git(&["rev-parse", "--git-common-dir"]) {
        let common = std::fs::canonicalize(std::path::Path::new(&manifest_dir).join(&common))
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or(common);
        exists_then_watch(format!("{common}/packed-refs"));
        if let Some(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
            exists_then_watch(format!("{common}/{head_ref}"));
        }
    }
}

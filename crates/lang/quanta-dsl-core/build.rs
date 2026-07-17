fn main() {
    stamp_build_rev();
}

/// Stamp `QUANTA_BUILD_REV` for the stale-compiler handshake.
///
/// "unknown" outside a tracked git checkout — crates.io and vendored
/// copies land there (a vendored copy inside a CONSUMER's repo is
/// detected via `ls-files`: this crate's manifest isn't tracked in
/// that repo) — silently: that is the designed state for those builds.
/// But when git *refuses* a checkout that is really there (Windows
/// "dubious ownership" on an elevated-created clone, a broken git),
/// the stamp also lands on "unknown" and the handshake downstream can
/// only WARN — so name git's actual error here, in a cargo warning,
/// where the cause is still visible.
fn stamp_build_rev() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    // Ok(stdout) on success, Err(reason) otherwise — the reason is
    // git's stderr (flattened to one line: cargo:warning is
    // line-oriented) or the spawn error, so a refusal can be named.
    let git = |args: &[&str]| -> Result<String, String> {
        let out = std::process::Command::new("git")
            .args(args)
            .current_dir(&manifest_dir)
            .output()
            .map_err(|e| format!("git not runnable: {e}"))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr)
                .trim()
                .replace('\n', " ");
            return Err(if stderr.is_empty() {
                format!("git {} exited {}", args.first().unwrap_or(&""), out.status)
            } else {
                stderr
            });
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            Err(format!(
                "git {} produced no output",
                args.first().unwrap_or(&"")
            ))
        } else {
            Ok(s)
        }
    };

    let rev = match git(&["rev-parse", "--is-inside-work-tree"]) {
        Ok(_) => {
            if git(&["ls-files", "--error-unmatch", "Cargo.toml"]).is_ok() {
                match git(&["describe", "--always", "--dirty", "--exclude", "*"]) {
                    Ok(rev) => Some(rev),
                    Err(reason) => {
                        println!(
                            "cargo:warning=QUANTA_BUILD_REV stamps `unknown`: git describe \
                             failed in a tracked checkout: {reason}"
                        );
                        None
                    }
                }
            } else {
                // Vendored inside a consumer's repo — by design, silent.
                None
            }
        }
        // A registry/tarball build is simply not a repo — by design,
        // silent.
        Err(reason) if reason.contains("not a git repository") => None,
        Err(reason) => {
            println!(
                "cargo:warning=QUANTA_BUILD_REV stamps `unknown`: git cannot read \
                 {manifest_dir}: {reason} (the stale-compiler handshake will WARN \
                 instead of verifying the compiler rev)"
            );
            None
        }
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
    if let Ok(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        exists_then_watch(format!("{git_dir}/HEAD"));
    }
    if let Ok(common) = git(&["rev-parse", "--git-common-dir"]) {
        let common = std::fs::canonicalize(std::path::Path::new(&manifest_dir).join(&common))
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or(common);
        exists_then_watch(format!("{common}/packed-refs"));
        if let Ok(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
            exists_then_watch(format!("{common}/{head_ref}"));
        }
    }
}

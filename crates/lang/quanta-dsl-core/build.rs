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
    // HEAD (branch switches) and FETCH_HEAD (rewritten by every
    // fetch/pull, even one whose ref update lands only in
    // packed-refs) in the per-worktree git dir; the current branch's
    // loose ref + packed-refs (commits) in the COMMON git dir —
    // different from the worktree dir under `git worktree`.
    //
    // Every path is rebuilt component-wise from git's `/`-separated
    // output (native_path) and assembled with PathBuf::join — NEVER
    // routed through fs::canonicalize: on Windows canonicalize
    // returns a `\\?\`-verbatim path, and verbatim paths switch OFF
    // Win32 normalization including `/`-as-separator, so a form like
    // `\\?\C:\...\.git/packed-refs` stats as nonexistent, the watch
    // is silently dropped, and a `git pull` stops re-stamping
    // QUANTA_BUILD_REV.
    let exists_then_watch = |p: std::path::PathBuf| {
        if p.exists() {
            println!("cargo:rerun-if-changed={}", p.display());
        }
    };
    if let Ok(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        let git_dir = native_path(&git_dir);
        exists_then_watch(git_dir.join("HEAD"));
        exists_then_watch(git_dir.join("FETCH_HEAD"));
    }
    if let Ok(common) = git(&["rev-parse", "--git-common-dir"]) {
        // git prints the common dir relative to the cwd (from this
        // manifest dir: `../../../.git`): resolve with a plain join
        // against the manifest dir plus a lexical `..` collapse;
        // fs::canonicalize only as the symlinked-tree fallback, with
        // the `\\?\` verbatim prefix stripped.
        let common = native_path(&common);
        let common = if common.is_absolute() {
            common
        } else {
            let joined = std::path::Path::new(&manifest_dir).join(&common);
            let lexical = lexical_normalize(&joined);
            if lexical.exists() {
                lexical
            } else {
                std::fs::canonicalize(&joined)
                    .map(strip_verbatim)
                    .unwrap_or(lexical)
            }
        };
        exists_then_watch(common.join("packed-refs"));
        if let Ok(head_ref) = git(&["symbolic-ref", "-q", "HEAD"]) {
            // `refs/heads/<branch>` — `/`-separated: rebuild native
            // before joining under the common dir.
            exists_then_watch(common.join(native_path(&head_ref)));
        }
    }
}

/// Rebuild a git-printed path (`/`-separated even on Windows) from
/// its components, so separators come out native: `C:/r/.git` becomes
/// `C:\r\.git` on Windows, and Unix paths pass through unchanged.
fn native_path(printed: &str) -> std::path::PathBuf {
    std::path::Path::new(printed).components().collect()
}

/// Collapse `.` and `<dir>/..` pairs lexically — a pure-string
/// resolve for git's `../../../.git`-style relative output that
/// involves no fs::canonicalize and therefore can never introduce
/// the Windows `\\?\` verbatim prefix.
fn lexical_normalize(p: &std::path::Path) -> std::path::PathBuf {
    let mut out = std::path::PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => match out.components().next_back() {
                Some(std::path::Component::Normal(_)) => {
                    out.pop();
                }
                _ => out.push(c),
            },
            _ => out.push(c),
        }
    }
    out
}

/// Strip the `\\?\` verbatim prefix fs::canonicalize adds on Windows
/// (`\\?\C:\x` → `C:\x`, `\\?\UNC\srv\share` → `\\srv\share`).
/// Verbatim paths disable `/`-as-separator handling — exactly what
/// broke the rerun watches. Non-Windows paths pass through untouched.
fn strip_verbatim(p: std::path::PathBuf) -> std::path::PathBuf {
    let stripped = {
        let s = p.to_string_lossy();
        if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
            Some(format!(r"\\{rest}"))
        } else {
            s.strip_prefix(r"\\?\").map(str::to_string)
        }
    };
    match stripped {
        Some(s) => std::path::PathBuf::from(s),
        None => p,
    }
}

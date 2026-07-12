//! Compiler binary discovery, invocation, and auto-download.

use quanta_ir::{CompilerOutput, KernelDef};

use super::shader_types::{ShaderParam, ShaderType};

/// Compile a KernelDef to all available targets.
///
/// Strategy:
/// 1. Try quanta-compiler binary (local dev, PATH, cached download, or auto-download)
/// 2. If not found, return empty output with warning
#[cfg(feature = "compute")]
pub fn compile_kernel(kernel: &KernelDef) -> Result<CompilerOutput, String> {
    // A resolvable compiler whose rev provably differs from this build is
    // a HARD error (unless QUANTA_ACCEPT_STALE_COMPILER=1): a stale
    // compiler has emitted invalid SPIR-V that segfaults some drivers, so
    // it must stop the build, not silently JIT/fall back. Checked before
    // invocation so the diagnosis is the rev mismatch, not a downstream
    // symptom. A missing / unloadable / pre-stamp compiler stays soft.
    if let Some(binary) = find_compiler_binary()
        && let CompilerVerdict::RevMismatch(bin_rev) = probe_compiler(&binary)
    {
        return Err(rev_mismatch_error(&binary, &bin_rev));
    }

    // Try calling the compiler binary for full output.
    // find_compiler_binary() handles the full search chain including
    // auto-download from GitHub Releases for crates.io users.
    if let Some(output) = try_compiler_binary(kernel) {
        return Ok(output);
    }

    // No compiler binary found — return empty output.
    // GPU dispatch will fail at runtime, but compilation succeeds.
    Ok(CompilerOutput {
        amd: None,
        nvidia: None,
        spirv: None,
        metallib: None,
        wgsl: None,
    })
}

/// Try to find and call the quanta-compiler binary.
#[cfg(feature = "compute")]
fn try_compiler_binary(kernel: &KernelDef) -> Option<CompilerOutput> {
    let binary = find_compiler_binary()?;
    if !compiler_is_loadable(&binary) {
        return None;
    }

    // Serialize KernelDef to bincode
    let input = quanta_ir::serialize_kernel(kernel);

    // Call the binary: stdin = KernelDef, stdout = CompilerOutput
    let result = std::process::Command::new(&binary)
        .arg("--targets")
        .arg("nvptx,amdgpu")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let mut child = result.ok()?;

    // Write input and explicitly close stdin before reading output
    use std::io::Write;
    {
        let mut stdin = child.stdin.take()?;
        if stdin.write_all(&input).is_err() {
            let _ = child.kill();
            return None;
        }
    } // stdin dropped here → pipe closed → child sees EOF

    // Read output
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("[quanta] compiler failed: {}", stderr);
        return None;
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        // Surface compiler-side warnings (metallib failures, etc.)
        // so build authors see the real reason a kernel target is
        // missing — without this, `metallib: None` silently ships
        // and the runtime panics with a generic
        // "no compiled kernel for vendor Apple".
        eprintln!("{}", stderr);
    }
    let result = quanta_ir::deserialize_output(&output.stdout);
    if let Err(ref e) = result {
        eprintln!("[quanta] deserialize error: {}", e);
    }
    result.ok()
}

/// Find the quanta-compiler binary.
/// Search order:
/// 1. QUANTA_COMPILER env var
/// 2. ../quanta-compiler/target/release/quanta-compiler (development)
/// 3. ../quanta-compiler/target/debug/quanta-compiler (development)
/// 4. quanta-compiler in PATH
/// 5. Cached download in ~/.quanta/bin/
/// 6. Download from GitHub Releases (unless QUANTA_NO_DOWNLOAD=1)
fn find_compiler_binary() -> Option<String> {
    // 1. Environment variable
    if let Ok(path) = std::env::var("QUANTA_COMPILER")
        && std::path::Path::new(&path).exists()
    {
        return Some(path);
    }

    // 2. Development: workspace target directory
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let exe_suffix = if cfg!(windows) { ".exe" } else { "" };
    for sub in &[
        format!("target/release/quanta-compiler{exe_suffix}"),
        format!("../target/release/quanta-compiler{exe_suffix}"),
        format!("../../target/release/quanta-compiler{exe_suffix}"),
        format!("target/debug/quanta-compiler{exe_suffix}"),
        format!("../target/debug/quanta-compiler{exe_suffix}"),
        format!("../../target/debug/quanta-compiler{exe_suffix}"),
    ] {
        let path = std::path::PathBuf::from(&manifest_dir).join(sub);
        if path.exists() {
            return Some(path.to_string_lossy().to_string());
        }
    }

    // 3. PATH — `which` on Unix, `where` on Windows.
    let path_lookup = if cfg!(windows) { "where" } else { "which" };
    if let Ok(output) = std::process::Command::new(path_lookup)
        .arg("quanta-compiler")
        .output()
        && output.status.success()
    {
        // `where` may return multiple paths separated by newlines — take the first.
        let path = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !path.is_empty() {
            return Some(path);
        }
    }

    // 4. Cached download in ~/.quanta/bin/
    if let Some(cached) = find_cached_compiler() {
        return Some(cached);
    }

    // 5. Download from GitHub Releases
    if let Some(downloaded) = download_compiler_binary() {
        return Some(downloaded);
    }

    // The compiler is optional — kernels without precompiled PTX/AMDGPU
    // ISA fall through to the JIT path (`device.wave_jit(...)`) at
    // dispatch time. So this is only a notice, not an error.
    eprintln!(
        "[quanta] note: ahead-of-time LLVM compiler not present; \
         kernels will JIT-compile at runtime instead. \
         Set QUANTA_COMPILER, run `cargo install quanta-compiler`, \
         or upgrade to a release that ships your platform binary."
    );
    None
}

// ============================================================================
// Compiler binary auto-download (for crates.io users)
// ============================================================================

/// Resolve the user's home directory from environment variables.
fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(std::path::PathBuf::from)
}

/// Detect the current compilation target triple.
///
/// macOS Intel (`x86_64-apple-darwin`) is intentionally excluded —
/// Apple discontinued Intel Macs in 2023 and Quanta v0.1 is Apple
/// Silicon-only on macOS. Intel Mac users will hit the "unknown"
/// branch and the JIT fallback covers them at dispatch time.
fn current_target() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    return "aarch64-apple-darwin";
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    return "x86_64-unknown-linux-gnu";
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    return "aarch64-unknown-linux-gnu";
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    return "x86_64-pc-windows-msvc";
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
    )))]
    return "unknown";
}

/// Return the path to the version-pinned cache directory: ~/.quanta/bin/
fn compiler_cache_dir() -> Option<std::path::PathBuf> {
    Some(home_dir()?.join(".quanta").join("bin"))
}

/// Return the expected cached binary path for the current version.
fn cached_compiler_path() -> Option<std::path::PathBuf> {
    let version = env!("CARGO_PKG_VERSION");
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let binary_name = format!("quanta-compiler-{}{}", version, suffix);
    Some(compiler_cache_dir()?.join(binary_name))
}

/// Archive extension for the host triple. Windows ships .zip; everything
/// else ships .tar.gz. Must match the `archive` matrix entry in
/// `.github/workflows/release-compiler.yml`.
fn archive_ext() -> &'static str {
    if cfg!(windows) { "zip" } else { "tar.gz" }
}

/// Check if a previously downloaded compiler binary exists in the cache.
fn find_cached_compiler() -> Option<String> {
    let cached_path = cached_compiler_path()?;
    if cached_path.exists() {
        return Some(cached_path.to_string_lossy().to_string());
    }
    None
}

/// Download the quanta-compiler binary from GitHub Releases.
///
/// Downloads a tar.gz archive matching the current version and target triple,
/// extracts it to ~/.quanta/bin/, and returns the path to the binary.
/// Returns None if download is disabled, fails, or the platform is unsupported.
fn download_compiler_binary() -> Option<String> {
    // Respect QUANTA_NO_DOWNLOAD=1 for CI or offline environments
    if std::env::var("QUANTA_NO_DOWNLOAD").unwrap_or_default() == "1" {
        return None;
    }

    let target = current_target();
    if target == "unknown" {
        eprintln!("[quanta] Unsupported platform for auto-download.");
        return None;
    }

    let version = env!("CARGO_PKG_VERSION");
    let cache_dir = compiler_cache_dir()?;
    std::fs::create_dir_all(&cache_dir).ok()?;

    let cached_path = cached_compiler_path()?;
    let ext = archive_ext();
    let download_path = cache_dir.join(format!("download.{ext}"));

    let url = format!(
        "https://github.com/zelez-lab/quanta/releases/download/v{}/quanta-compiler-{}.{}",
        version, target, ext
    );

    eprintln!(
        "[quanta] fetching ahead-of-time compiler v{} for {}...",
        version, target
    );

    // Download using curl (ships with macOS, Linux, and Windows 10 1803+).
    // Use --silent so a 404 doesn't spew progress noise; we already
    // print our own diagnostic if the download fails.
    let output = std::process::Command::new("curl")
        .args(["-fsSL", "--silent", &url, "-o"])
        .arg(&download_path)
        .output()
        .ok()?;

    if !output.status.success() {
        // Quietly clean up — the caller (find_compiler_binary) will
        // print the single, JIT-aware notice. Spamming the build log
        // here was the old behavior and made users think something
        // was broken when JIT was about to handle it transparently.
        let _ = std::fs::remove_file(&download_path);
        return None;
    }

    // Extract the archive — `tar` ships with Win10 1803+ and unpacks both
    // .tar.gz and .zip. On Unix it's the canonical tool for .tar.gz.
    let extract = std::process::Command::new("tar")
        .args(["xf"])
        .arg(&download_path)
        .current_dir(&cache_dir)
        .output()
        .ok()?;

    // Clean up the archive regardless of extraction result
    let _ = std::fs::remove_file(&download_path);

    if !extract.status.success() {
        let stderr = String::from_utf8_lossy(&extract.stderr);
        eprintln!("[quanta] Extraction failed: {}", stderr.trim());
        return None;
    }

    // The archive is expected to contain a `quanta-compiler[.exe]` binary
    // at its root. Rename to the version-pinned name to avoid mismatches.
    let suffix = if cfg!(windows) { ".exe" } else { "" };
    let extracted = cache_dir.join(format!("quanta-compiler{suffix}"));
    if extracted.exists() {
        if std::fs::rename(&extracted, &cached_path).is_err() {
            eprintln!("[quanta] Failed to rename downloaded binary.");
            return None;
        }
    } else {
        eprintln!("[quanta] Archive did not contain expected 'quanta-compiler{suffix}' binary.");
        return None;
    }

    // Ensure the binary is executable (Unix)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&cached_path, std::fs::Permissions::from_mode(0o755));
    }

    eprintln!("[quanta] Compiler installed to {}", cached_path.display());
    Some(cached_path.to_string_lossy().to_string())
}

// ============================================================================
// Shader compilation (vertex / fragment) via compiler binary
// ============================================================================

/// Output from shader compilation — SPIR-V and metallib binaries.
#[cfg(feature = "render")]
pub(crate) struct ShaderCompileOutput {
    pub(crate) spirv: Option<Vec<u8>>,
    pub(crate) metallib: Option<Vec<u8>>,
    pub(crate) wgsl: Option<String>,
}

/// Compile a vertex or fragment shader via the quanta-compiler binary.
///
/// Serializes the ShaderDef to the compiler's stdin, reads ShaderOutput
/// from stdout. Returns `Ok(None)` if the compiler binary is not found
/// (find_compiler_binary already printed its notice), and `Err` if the
/// compiler was found but failed — the macro turns that into a compile
/// error so a broken shader can never ship silently.
#[cfg(feature = "render")]
pub(crate) fn compile_shader(
    name: &str,
    stage: &str,
    params: &[ShaderParam],
    return_type: &ShaderType,
    body_source: &str,
) -> Result<Option<ShaderCompileOutput>, String> {
    let Some(binary) = find_compiler_binary() else {
        return Ok(None);
    };
    // A binary the dynamic loader kills (downloaded release build whose
    // libLLVM isn't installed) dies before reading stdin — writing the
    // shader into its pipe would race the death and can SIGPIPE the
    // rustc process hosting this macro. Preflight once per path so the
    // piped spawn only ever happens against a binary that can run. The
    // same probe reads the build rev: a PROVEN rev mismatch is fatal here
    // (a stale compiler's invalid SPIR-V segfaults some drivers) unless
    // QUANTA_ACCEPT_STALE_COMPILER=1; the error becomes a compile_error!
    // through the macro path. Unloadable / pre-stamp stays soft.
    match probe_compiler(&binary) {
        CompilerVerdict::RevMismatch(bin_rev) => {
            return Err(rev_mismatch_error(&binary, &bin_rev));
        }
        CompilerVerdict::NotLoadable => return Ok(None),
        CompilerVerdict::Usable => {}
    }

    // Build ShaderDef from the parsed macro arguments
    let shader_def = quanta_ir::ShaderDef {
        name: name.to_string(),
        stage: match stage {
            "vertex" => quanta_ir::ShaderStage::Vertex,
            "fragment" => quanta_ir::ShaderStage::Fragment,
            other => return Err(format!("unknown shader stage `{other}`")),
        },
        params: params
            .iter()
            .map(|p| quanta_ir::ShaderParam {
                name: p.name.clone(),
                ty: shader_type_to_ir(&p.ty),
                is_uniform: p.is_uniform,
                is_slice: p.is_slice,
            })
            .collect(),
        return_type: shader_type_to_ir(return_type),
        body_source: body_source.to_string(),
    };

    let input = quanta_ir::serialize_shader(&shader_def);

    let mut child = std::process::Command::new(&binary)
        .arg("--shader-type")
        .arg(stage)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn shader compiler `{binary}`: {e}"))?;

    use std::io::Write;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| "failed to open shader compiler stdin".to_string())?;
        if let Err(e) = stdin.write_all(&input) {
            // The child died before reading its input — collect its
            // status and classify: a loader kill means "no usable
            // compiler here" (soft), anything else is a real failure.
            drop(stdin);
            if let Ok(output) = child.wait_with_output()
                && is_loader_failure(&output)
            {
                eprintln!(
                    "[quanta] note: shader compiler at {binary} cannot run                          here ({}); shaders will have no precompiled binaries",
                    String::from_utf8_lossy(&output.stderr).trim()
                );
                return Ok(None);
            }
            return Err(format!("failed to write shader to compiler stdin: {e}"));
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("failed to read shader compiler output: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        // A binary that cannot EXECUTE in this environment (a downloaded
        // release build whose dynamic libraries aren't installed — the
        // loader kills it before main) means "no usable compiler here",
        // same as binary-not-found: stay soft so builds without a
        // toolchain still compile with empty shader binaries. Only a
        // compiler that actually ran gets to fail the build.
        if is_loader_failure(&output) {
            eprintln!(
                "[quanta] note: shader compiler at {} cannot run here \
                 ({}); shaders will have no precompiled binaries",
                binary,
                stderr.trim()
            );
            return Ok(None);
        }
        return Err(format!("shader compiler failed: {}", stderr.trim()));
    }
    if !stderr.trim().is_empty() {
        // Surface compiler-side warnings (WGSL emitter gaps, etc.) so
        // build authors see why an optional target is missing.
        eprintln!("{}", stderr.trim());
    }

    let shader_output = quanta_ir::deserialize_shader_output(&output.stdout)
        .map_err(|e| format!("failed to deserialize shader compiler output: {e}"))?;
    Ok(Some(ShaderCompileOutput {
        spirv: shader_output.spirv,
        metallib: shader_output.metallib,
        wgsl: shader_output.wgsl,
    }))
}

/// Outcome of probing a resolved compiler binary once with `--rev`.
#[cfg(any(feature = "compute", feature = "render"))]
#[derive(Clone, Debug, PartialEq)]
enum CompilerVerdict {
    /// Loadable and safe to use: rev matches this build, OR the binary
    /// predates rev stamping (a WARN was already emitted), OR a rev
    /// mismatch was explicitly accepted via `QUANTA_ACCEPT_STALE_COMPILER`.
    Usable,
    /// Loadable but its rev DIFFERS from this build's rev. Fatal: an
    /// invalid-SPIR-V module from a stale compiler segfaults some drivers
    /// (v3dv), so a mismatch must stop the build rather than warn. Carries
    /// the probed rev for the error message.
    RevMismatch(String),
    /// Cannot run here (loader kill / spawn failure). Soft: kernels JIT at
    /// runtime, shaders ship with no precompiled binaries.
    NotLoadable,
}

/// Preflight: probe the resolved compiler binary ONCE and classify it.
///
/// A downloaded release build dynamically linked against a libLLVM that
/// isn't installed is killed by the loader before main() — spawning it
/// with piped stdin then races its death (a broken-pipe write can SIGPIPE
/// the host rustc process on macOS). Running it with null stdin and
/// `--rev` both preflights loadability AND reads the build rev. The
/// verdict is cached per path for the life of the process, so the WARN
/// (pre-stamp case) prints at most once per binary.
///
/// `--rev` with null stdin distinguishes three cases:
/// - a CURRENT binary prints its build rev and exits 0;
/// - an OLD binary (no `--rev` flag) falls through to its stdin loop, sees
///   EOF, and exits non-zero fast — it executed, so it's loadable, but its
///   rev is unknown (predates rev stamping);
/// - a loader-killed binary dies before main.
#[cfg(any(feature = "compute", feature = "render"))]
fn probe_compiler(binary: &str) -> CompilerVerdict {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(String, CompilerVerdict)>> = Mutex::new(None);
    if let Ok(guard) = CACHE.lock()
        && let Some((path, verdict)) = guard.as_ref()
        && path == binary
    {
        return verdict.clone();
    }
    let verdict = match std::process::Command::new(binary)
        .arg("--rev")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
    {
        Ok(output) => {
            if is_loader_failure(&output) {
                eprintln!(
                    "[quanta] note: compiler at {binary} cannot run here ({}); \
                     kernels will JIT at runtime and shaders will have no \
                     precompiled binaries",
                    String::from_utf8_lossy(&output.stderr)
                        .trim()
                        .replace('\n', " ")
                );
                CompilerVerdict::NotLoadable
            } else {
                let own_rev = env!("QUANTA_BUILD_REV");
                let bin_rev = if output.status.success() {
                    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
                } else {
                    None
                };
                match bin_rev.as_deref() {
                    Some(r) if r == own_rev => CompilerVerdict::Usable,
                    Some(r) => {
                        // Provable mismatch. FATAL by default — a stale
                        // compiler has shipped spirv-val-INVALID modules
                        // that segfault v3dv. The escape hatch is for rigs
                        // deliberately pinning a known-compatible compiler.
                        if accept_stale_compiler() {
                            eprintln!(
                                "[quanta] note: quanta-compiler at {binary} is rev {r} but \
                                 this quanta build is rev {own_rev}; proceeding because \
                                 QUANTA_ACCEPT_STALE_COMPILER is set."
                            );
                            CompilerVerdict::Usable
                        } else {
                            CompilerVerdict::RevMismatch(r.to_string())
                        }
                    }
                    None => {
                        // Pre-stamp binary: it ran but doesn't support
                        // `--rev`, so a mismatch CANNOT be proven — stay a
                        // loud warning (not fatal), unlike the provable
                        // mismatch above.
                        eprintln!(
                            "[quanta] WARNING: quanta-compiler at {binary} predates rev \
                             stamping (older than this quanta build, rev {own_rev}) — \
                             kernels and shaders may get STALE codegen. Reinstall it from \
                             the matching checkout."
                        );
                        CompilerVerdict::Usable
                    }
                }
            }
        }
        Err(_) => CompilerVerdict::NotLoadable,
    };
    if let Ok(mut guard) = CACHE.lock() {
        *guard = Some((binary.to_string(), verdict.clone()));
    }
    verdict
}

/// Whether `QUANTA_ACCEPT_STALE_COMPILER` is set to a non-empty value —
/// the operator's opt-out that downgrades a provable rev mismatch from
/// fatal to a note. Documented for rigs that intentionally pin a
/// compatible compiler.
#[cfg(any(feature = "compute", feature = "render"))]
fn accept_stale_compiler() -> bool {
    std::env::var("QUANTA_ACCEPT_STALE_COMPILER")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// The fatal error text for a rev mismatch, naming both revs, the escape
/// hatch, and the pre-stamp asymmetry.
#[cfg(any(feature = "compute", feature = "render"))]
fn rev_mismatch_error(binary: &str, bin_rev: &str) -> String {
    let own_rev = env!("QUANTA_BUILD_REV");
    format!(
        "quanta-compiler at {binary} was built from rev {bin_rev} but this quanta build \
         is rev {own_rev}. A mismatched compiler can emit invalid SPIR-V that crashes \
         some drivers, so this is a hard error. Reinstall the matching compiler: \
         cargo install --path crates/quanta-compiler --locked --force. To proceed anyway \
         (e.g. a rig pinning a known-compatible compiler), set \
         QUANTA_ACCEPT_STALE_COMPILER=1. (A pre-stamp compiler that lacks --rev can't be \
         proven mismatched and only WARNs — this fatal path fires only on a proven \
         difference.)"
    )
}

/// Preflight loadability only — a thin wrapper over [`probe_compiler`]
/// used where a rev mismatch is surfaced separately. `Usable` and a rev
/// MISMATCH both mean the binary LOADS (mismatch is handled by the
/// caller); only `NotLoadable` means it can't run here.
///
/// Used by the compute compile path (`try_compiler_binary`) and the probe
/// tests; the render (`compile_shader`) path matches on `probe_compiler`
/// directly so it can surface a mismatch as a hard error. Gated to
/// `compute`-or-`test` so a render-only non-test build (which never calls
/// it) doesn't flag it unused.
#[cfg(any(feature = "compute", test))]
fn compiler_is_loadable(binary: &str) -> bool {
    !matches!(probe_compiler(binary), CompilerVerdict::NotLoadable)
}

/// Whether a child failure is the binary failing to LOAD in this
/// environment rather than the compiler rejecting its input.
/// Linux ld.so exits 127 with "error while loading shared libraries";
/// macOS dyld aborts with "Library not loaded"; Windows exits with
/// STATUS_DLL_NOT_FOUND (0xC0000135).
#[cfg(any(feature = "compute", feature = "render"))]
fn is_loader_failure(output: &std::process::Output) -> bool {
    let stderr = String::from_utf8_lossy(&output.stderr);
    output.status.code() == Some(127)
        || output.status.code() == Some(0xC0000135u32 as i32)
        || stderr.contains("error while loading shared libraries")
        || stderr.contains("Library not loaded")
}

#[cfg(feature = "render")]
fn shader_type_to_ir(ty: &ShaderType) -> quanta_ir::ShaderType {
    match ty {
        ShaderType::F32 => quanta_ir::ShaderType::F32,
        ShaderType::Vec2 => quanta_ir::ShaderType::Vec2,
        ShaderType::Vec3 => quanta_ir::ShaderType::Vec3,
        ShaderType::Vec4 => quanta_ir::ShaderType::Vec4,
        ShaderType::Mat4 => quanta_ir::ShaderType::Mat4,
        ShaderType::Mat3 => quanta_ir::ShaderType::Mat3,
    }
}

#[cfg(all(test, unix, any(feature = "compute", feature = "render")))]
mod probe_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn fake_compiler(name: &str, script: &str) -> String {
        let path = std::env::temp_dir().join(format!("quanta-probe-{name}-{}", std::process::id()));
        std::fs::write(&path, script).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn probe_accepts_matching_rev() {
        let own = env!("QUANTA_BUILD_REV");
        let path = fake_compiler(
            "match",
            &format!("#!/bin/sh\nif [ \"$1\" = \"--rev\" ]; then echo {own}; exit 0; fi\nexit 1\n"),
        );
        assert!(compiler_is_loadable(&path));
        assert_eq!(probe_compiler(&path), CompilerVerdict::Usable);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn probe_accepts_prestamp_binary_as_loadable() {
        // No --rev support: exits non-zero fast — loadable, rev unknown.
        // Stays Usable (loud WARN only) — a mismatch can't be proven.
        let path = fake_compiler("old", "#!/bin/sh\nexit 1\n");
        assert!(compiler_is_loadable(&path));
        assert_eq!(probe_compiler(&path), CompilerVerdict::Usable);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn probe_rejects_loader_killed_binary() {
        let path = fake_compiler(
            "loader",
            "#!/bin/sh\necho 'error while loading shared libraries: libLLVM.so.22' >&2\nexit 127\n",
        );
        assert!(!compiler_is_loadable(&path));
        assert_eq!(probe_compiler(&path), CompilerVerdict::NotLoadable);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn probe_flags_rev_mismatch_as_fatal() {
        // A binary that prints a DIFFERENT rev than this build must
        // classify as a fatal RevMismatch — the signal that makes
        // compile_kernel / compile_shader return a compile error.
        //
        // Env-gated so the suite can opt out: if a rig runs the tests with
        // QUANTA_ACCEPT_STALE_COMPILER set, the mismatch is downgraded to
        // Usable by design, so this assertion would not hold — skip it.
        if accept_stale_compiler() {
            return;
        }
        let path = fake_compiler(
            "mismatch",
            "#!/bin/sh\nif [ \"$1\" = \"--rev\" ]; then echo deadbeefdeadbeef; exit 0; fi\nexit 1\n",
        );
        // Still LOADABLE (it ran) — the mismatch is surfaced by the caller,
        // not by the loadability wrapper.
        assert!(compiler_is_loadable(&path));
        assert_eq!(
            probe_compiler(&path),
            CompilerVerdict::RevMismatch("deadbeefdeadbeef".to_string())
        );
        std::fs::remove_file(&path).ok();
    }
}

//! MSL → metallib compilation via xcrun.
//!
//! Axiom A1 contract: if xcrun is available and accepts valid MSL,
//! it produces correct metallib. If xcrun fails on macOS, we MUST
//! fail the build — silently skipping would violate the axiom and
//! ship a kernel without an Apple binary.

use std::ffi::OsStr;
use std::process::Output;

/// How a failed `xcrun` spawn is classified.
enum SpawnFailure {
    /// The tool doesn't exist on a non-Apple host — skipping the
    /// metallib is legitimate (cross-compiling from Linux).
    ToolAbsent,
    /// Everything else. A missing xcrun on macOS means a broken
    /// toolchain, and a transient spawn error that survives retries
    /// must fail the build rather than silently ship a SPIR-V-only
    /// artifact (axiom A1).
    Fatal(String),
}

/// Spawn `xcrun` with retries on transient failures.
///
/// A parallel cargo build fans out one `quanta-compiler` per shader,
/// each spawning `xcrun` twice; under that process load `posix_spawn`
/// can fail transiently (EAGAIN/ENOMEM). Before the retry, those
/// errors hit a silent `Ok(None)` arm — the root cause of metallib
/// emission flapping between builds on hosts where Xcode is installed.
fn spawn_xcrun(args: &[&OsStr]) -> Result<Output, SpawnFailure> {
    const ATTEMPTS: u32 = 3;
    let mut delay = std::time::Duration::from_millis(50);
    let mut last_err = None;
    for attempt in 0..ATTEMPTS {
        match std::process::Command::new("xcrun")
            .args(args)
            .stderr(std::process::Stdio::piped())
            .output()
        {
            Ok(o) => return Ok(o),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return if cfg!(target_os = "macos") {
                    Err(SpawnFailure::Fatal(
                        "xcrun not found on macOS — install the Xcode command line \
                         tools (`xcode-select --install`), or set \
                         QUANTA_SKIP_METALLIB=1 to build without Apple binaries"
                            .to_string(),
                    ))
                } else {
                    Err(SpawnFailure::ToolAbsent)
                };
            }
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 < ATTEMPTS {
                    std::thread::sleep(delay);
                    delay *= 3;
                }
            }
        }
    }
    Err(SpawnFailure::Fatal(format!(
        "failed to spawn xcrun after {} attempts: {} — refusing to silently \
         drop the metallib (set QUANTA_SKIP_METALLIB=1 to build without \
         Apple binaries)",
        ATTEMPTS,
        last_err.expect("retry loop ran at least once"),
    )))
}

/// Compile MSL source to metallib binary via xcrun metal + xcrun metallib.
///
/// Returns `Ok(bytes)` on success, `Err(message)` if xcrun fails or cannot
/// be spawned on macOS, `Ok(None)` only if xcrun is not available on a
/// non-Apple host (cross-compiling from Linux) or `QUANTA_SKIP_METALLIB=1`
/// explicitly opts out.
pub fn compile_msl_to_metallib(msl_source: &str) -> Result<Option<Vec<u8>>, String> {
    // Explicit, deterministic opt-out — e.g. a mac without the Metal
    // toolchain cross-compiling to a non-Apple target.
    if std::env::var("QUANTA_SKIP_METALLIB").as_deref() == Ok("1") {
        return Ok(None);
    }

    let unique = format!(
        "{}_{:x}",
        std::process::id(),
        msl_source.len() as u64 ^ (msl_source.as_ptr() as u64)
    );
    let tmp_dir = std::env::temp_dir().join(format!("quanta_metal_{}", unique));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("failed to create temp dir: {}", e))?;

    let msl_path = tmp_dir.join("kernel.metal");
    let air_path = tmp_dir.join("kernel.air");
    let lib_path = tmp_dir.join("kernel.metallib");

    std::fs::write(&msl_path, msl_source)
        .map_err(|e| format!("failed to write MSL source: {}", e))?;

    // MSL → AIR (with aggressive optimization)
    let air_args: Vec<&OsStr> = vec![
        "metal".as_ref(),
        "-c".as_ref(),
        "-std=metal3.1".as_ref(), // mesh shaders, ray tracing, bfloat16
        "-O3".as_ref(),           // maximum optimization
        "-ffast-math".as_ref(),   // allow reassociation, no NaN/inf checks
        msl_path.as_os_str(),
        "-o".as_ref(),
        air_path.as_os_str(),
    ];
    match spawn_xcrun(&air_args) {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            // xcrun found but compilation failed — axiom violation, must not skip
            let err = String::from_utf8_lossy(&o.stderr);
            return Err(format!("xcrun metal failed: {}", err));
        }
        Err(SpawnFailure::ToolAbsent) => return Ok(None), // cross-compiling, skip is OK
        Err(SpawnFailure::Fatal(msg)) => return Err(format!("xcrun metal: {}", msg)),
    };

    // AIR → metallib
    let lib_args: Vec<&OsStr> = vec![
        "metallib".as_ref(),
        air_path.as_os_str(),
        "-o".as_ref(),
        lib_path.as_os_str(),
    ];
    match spawn_xcrun(&lib_args) {
        Ok(o) if o.status.success() => {
            let bytes =
                std::fs::read(&lib_path).map_err(|e| format!("failed to read metallib: {}", e))?;
            Ok(Some(bytes))
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            Err(format!("xcrun metallib failed: {}", err))
        }
        Err(SpawnFailure::ToolAbsent) => Ok(None), // cross-compiling, skip is OK
        Err(SpawnFailure::Fatal(msg)) => Err(format!("xcrun metallib: {}", msg)),
    }
}

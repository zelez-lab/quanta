//! MSL → metallib compilation via xcrun.
//!
//! Axiom A1 contract: if xcrun is available and accepts valid MSL,
//! it produces correct metallib. If xcrun fails on macOS, we MUST
//! fail the build — silently skipping would violate the axiom and
//! ship a kernel without an Apple binary.
//!
//! # Platform variants
//!
//! A shader/kernel compiles to up to three metallib variants, one per
//! Apple platform ([`MetalPlatform`]): the macOS library the desktop
//! driver loads, the iOS-device library, and the iOS-simulator library.
//! iOS rejects a macOS-platform metallib, so a consumer built for an iOS
//! target needs the platform-correct binary embedded — the proc macros
//! cannot learn the consumer's target (that reaches build scripts only),
//! so the compiler emits every variant whose SDK is present and the
//! runtime selects by `cfg` (see `ShaderBinary`/`KernelBinary`).
//!
//! The iOS variants require the corresponding SDK. On a mac with only the
//! Command-Line Tools (no full Xcode) the iOS SDKs are absent, so those
//! variants soft-skip and the build ships macOS-only — today's behavior,
//! no error. `QUANTA_METAL_PLATFORMS` (a comma list of `macos,ios,ios-sim`)
//! overrides which variants are attempted.

use std::ffi::OsStr;
use std::process::Output;

/// The iOS/iOS-simulator deployment floor. Paired with [`METAL_STD`]:
/// `-std=metal3.1` needs an iOS 17.0 (or macOS 14 / Metal 3.1) target.
/// Bumping the std flag means bumping this floor in lockstep.
const IOS_DEPLOYMENT_TARGET: &str = "17.0";

/// The Metal Shading Language standard the emitter targets. Enables mesh
/// shaders, ray tracing, and bfloat16; the macOS invocation has passed
/// this since the metallib path existed. The iOS floor
/// ([`IOS_DEPLOYMENT_TARGET`]) is chosen to satisfy it.
const METAL_STD: &str = "-std=metal3.1";

/// An Apple platform a metallib can target.
///
/// macOS is the original, always-attempted variant. The two iOS variants
/// gate on their SDK. Room is intentionally left for watchOS / tvOS /
/// visionOS: add an enum arm plus its `sdk()` / `air_target()` and the
/// probe/dispatch pick it up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalPlatform {
    /// The desktop metallib (`xcrun metal` with the default macOS SDK).
    MacOs,
    /// iOS device (`-sdk iphoneos`, `air64-apple-ios<floor>`).
    Ios,
    /// iOS simulator (`-sdk iphonesimulator`,
    /// `air64-apple-ios<floor>-simulator`).
    IosSim,
}

impl MetalPlatform {
    /// The three variants, in a stable order (macOS first).
    pub const ALL: [MetalPlatform; 3] = [
        MetalPlatform::MacOs,
        MetalPlatform::Ios,
        MetalPlatform::IosSim,
    ];

    /// The `QUANTA_METAL_PLATFORMS` token that names this platform.
    fn env_token(self) -> &'static str {
        match self {
            MetalPlatform::MacOs => "macos",
            MetalPlatform::Ios => "ios",
            MetalPlatform::IosSim => "ios-sim",
        }
    }

    /// The `-sdk` argument, or `None` for the default (macOS) SDK.
    fn sdk(self) -> Option<&'static str> {
        match self {
            MetalPlatform::MacOs => None,
            MetalPlatform::Ios => Some("iphoneos"),
            MetalPlatform::IosSim => Some("iphonesimulator"),
        }
    }

    /// The `-target` triple, or `None` for the macOS SDK's implied target.
    fn air_target(self) -> Option<String> {
        match self {
            MetalPlatform::MacOs => None,
            MetalPlatform::Ios => Some(format!("air64-apple-ios{IOS_DEPLOYMENT_TARGET}")),
            MetalPlatform::IosSim => {
                Some(format!("air64-apple-ios{IOS_DEPLOYMENT_TARGET}-simulator"))
            }
        }
    }
}

/// The three metallib variants for one shader/kernel. A `None` variant
/// was not produced (its SDK is absent, or the platform was excluded by
/// `QUANTA_METAL_PLATFORMS`, or metallib was skipped entirely).
#[derive(Debug, Clone, Default)]
pub struct MetallibVariants {
    /// The macOS-platform metallib (today's binary).
    pub macos: Option<Vec<u8>>,
    /// The iOS-device metallib.
    pub ios: Option<Vec<u8>>,
    /// The iOS-simulator metallib.
    pub ios_sim: Option<Vec<u8>>,
}

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

/// Probe whether a platform's SDK is installed, caching the answer for
/// the life of the process.
///
/// macOS is assumed present (a mac building any Metal target has the
/// macOS SDK; a non-Apple host takes the `ToolAbsent` path inside the
/// compile call and never reaches here for a real emission). The iOS
/// SDKs are probed once via `xcrun -sdk <name> --show-sdk-path`: exit 0
/// with a non-empty path means present. A Command-Line-Tools-only mac
/// has no iOS SDK, so the probe reports absent and the variant
/// soft-skips.
fn sdk_available(platform: MetalPlatform) -> bool {
    use std::sync::OnceLock;
    static IOS: OnceLock<bool> = OnceLock::new();
    static IOS_SIM: OnceLock<bool> = OnceLock::new();

    let Some(sdk) = platform.sdk() else {
        // macOS: the default SDK. Present wherever an emission runs.
        return true;
    };
    let cell = match platform {
        MetalPlatform::Ios => &IOS,
        MetalPlatform::IosSim => &IOS_SIM,
        MetalPlatform::MacOs => unreachable!("macOS has no -sdk"),
    };
    *cell.get_or_init(|| {
        let args: Vec<&OsStr> = vec!["-sdk".as_ref(), sdk.as_ref(), "--show-sdk-path".as_ref()];
        match spawn_xcrun(&args) {
            Ok(o) => o.status.success() && !o.stdout.is_empty(),
            // A missing xcrun (non-Apple host) or a transient failure
            // means we can't confirm the SDK — treat as absent and
            // soft-skip the variant rather than fail.
            Err(_) => false,
        }
    })
}

/// The set of platforms to attempt this process, honoring
/// `QUANTA_METAL_PLATFORMS`.
///
/// Unset (or empty) means all three. When set, it is a comma list of
/// `macos` / `ios` / `ios-sim`; unknown tokens are ignored. The override
/// only says which variants to *attempt* — a requested-but-SDK-absent
/// iOS variant still soft-skips inside the compile call.
fn requested_platforms() -> Vec<MetalPlatform> {
    match std::env::var("QUANTA_METAL_PLATFORMS") {
        Ok(v) if !v.trim().is_empty() => {
            let tokens: Vec<String> = v
                .split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect();
            MetalPlatform::ALL
                .into_iter()
                .filter(|p| tokens.iter().any(|t| t == p.env_token()))
                .collect()
        }
        _ => MetalPlatform::ALL.to_vec(),
    }
}

/// Compile MSL source to the macOS metallib binary via xcrun.
///
/// Back-compat shim over [`compile_msl_to_metallib_variants`]: returns
/// only the macOS variant (`Ok(bytes)` on success, `Ok(None)` when the
/// toolchain is absent or `QUANTA_SKIP_METALLIB=1`, `Err` on a fatal
/// macOS failure). Callers that want the iOS variants use the plural
/// form directly. The pipelines now call the plural form; this shim is
/// retained for the emit_msl shader tests and as the API modeled by the
/// Verus proof crate — hence it reads as dead code in the release binary.
#[cfg_attr(not(test), allow(dead_code))]
pub fn compile_msl_to_metallib(msl_source: &str) -> Result<Option<Vec<u8>>, String> {
    Ok(compile_msl_to_metallib_variants(msl_source)?.macos)
}

/// Compile MSL source to every requested, SDK-available metallib variant.
///
/// The macOS variant keeps the original retry/fatal semantics of
/// [`compile_msl_to_metallib`]: a `metal`/`metallib` failure on macOS is
/// an error (axiom A1). Each iOS variant runs the same retry/fatal path
/// but *only when its SDK exists*; an absent SDK soft-skips (that field
/// stays `None`) with at most one concise note, never an error — a
/// Command-Line-Tools-only mac still builds macOS-only, exactly as before.
///
/// `QUANTA_SKIP_METALLIB=1` opts out of all variants (returns the
/// all-`None` default), unchanged.
pub fn compile_msl_to_metallib_variants(msl_source: &str) -> Result<MetallibVariants, String> {
    // Explicit, deterministic opt-out — e.g. a mac without the Metal
    // toolchain cross-compiling to a non-Apple target.
    if std::env::var("QUANTA_SKIP_METALLIB").as_deref() == Ok("1") {
        return Ok(MetallibVariants::default());
    }

    let unique = format!(
        "{}_{:x}",
        std::process::id(),
        msl_source.len() as u64 ^ (msl_source.as_ptr() as u64)
    );
    let tmp_dir = std::env::temp_dir().join(format!("quanta_metal_{}", unique));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("failed to create temp dir: {}", e))?;

    let msl_path = tmp_dir.join("kernel.metal");
    std::fs::write(&msl_path, msl_source)
        .map_err(|e| format!("failed to write MSL source: {}", e))?;

    let mut variants = MetallibVariants::default();
    let requested = requested_platforms();
    for platform in MetalPlatform::ALL {
        if !requested.contains(&platform) {
            continue;
        }
        // Skip an iOS variant whose SDK isn't installed. macOS always
        // reports available; a CLT-only mac reports the iOS SDKs absent
        // and lands here with a single note, no error.
        if !sdk_available(platform) {
            eprintln!(
                "[quanta] note: {} SDK not installed — skipping its metallib \
                 (install Xcode to embed it)",
                platform.env_token()
            );
            continue;
        }
        let bytes = compile_one_platform(&tmp_dir, &msl_path, platform)?;
        match platform {
            MetalPlatform::MacOs => variants.macos = bytes,
            MetalPlatform::Ios => variants.ios = bytes,
            MetalPlatform::IosSim => variants.ios_sim = bytes,
        }
    }

    Ok(variants)
}

/// Compile one platform variant: MSL → AIR → metallib.
///
/// Mirrors the original two-step invocation, adding the platform's `-sdk`
/// / `-target` flags when it is an iOS variant. Returns `Ok(Some(bytes))`
/// on success, `Ok(None)` only when xcrun is absent on a non-Apple host,
/// and `Err` on a compile/link failure (fatal per axiom A1 — the caller
/// only reaches this for a platform whose SDK it already confirmed).
fn compile_one_platform(
    tmp_dir: &std::path::Path,
    msl_path: &std::path::Path,
    platform: MetalPlatform,
) -> Result<Option<Vec<u8>>, String> {
    // Per-platform artifact names so parallel variants don't collide.
    let air_path = tmp_dir.join(format!("kernel_{}.air", platform.env_token()));
    let lib_path = tmp_dir.join(format!("kernel_{}.metallib", platform.env_token()));

    // MSL → AIR (with aggressive optimization)
    let sdk = platform.sdk();
    let target = platform.air_target();
    let mut air_args: Vec<&OsStr> = Vec::new();
    if let Some(sdk) = sdk {
        air_args.push("-sdk".as_ref());
        air_args.push(sdk.as_ref());
    }
    air_args.push("metal".as_ref());
    air_args.push("-c".as_ref());
    air_args.push(METAL_STD.as_ref()); // mesh shaders, ray tracing, bfloat16
    if let Some(target) = target.as_deref() {
        air_args.push("-target".as_ref());
        air_args.push(target.as_ref());
    }
    air_args.push("-O3".as_ref()); // maximum optimization
    air_args.push("-ffast-math".as_ref()); // allow reassociation, no NaN/inf checks
    air_args.push(msl_path.as_os_str());
    air_args.push("-o".as_ref());
    air_args.push(air_path.as_os_str());

    match spawn_xcrun(&air_args) {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            // xcrun found but compilation failed — axiom violation, must not skip
            let err = String::from_utf8_lossy(&o.stderr);
            return Err(format!(
                "xcrun metal ({}) failed: {}",
                platform.env_token(),
                err
            ));
        }
        Err(SpawnFailure::ToolAbsent) => return Ok(None), // cross-compiling, skip is OK
        Err(SpawnFailure::Fatal(msg)) => {
            return Err(format!("xcrun metal ({}): {}", platform.env_token(), msg));
        }
    };

    // AIR → metallib. The `-sdk` selects the platform library; no
    // `-target` here (metallib links the already-targeted AIR).
    let mut lib_args: Vec<&OsStr> = Vec::new();
    if let Some(sdk) = sdk {
        lib_args.push("-sdk".as_ref());
        lib_args.push(sdk.as_ref());
    }
    lib_args.push("metallib".as_ref());
    lib_args.push(air_path.as_os_str());
    lib_args.push("-o".as_ref());
    lib_args.push(lib_path.as_os_str());

    match spawn_xcrun(&lib_args) {
        Ok(o) if o.status.success() => {
            let bytes =
                std::fs::read(&lib_path).map_err(|e| format!("failed to read metallib: {}", e))?;
            Ok(Some(bytes))
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            Err(format!(
                "xcrun metallib ({}) failed: {}",
                platform.env_token(),
                err
            ))
        }
        Err(SpawnFailure::ToolAbsent) => Ok(None), // cross-compiling, skip is OK
        Err(SpawnFailure::Fatal(msg)) => Err(format!(
            "xcrun metallib ({}): {}",
            platform.env_token(),
            msg
        )),
    }
}

#[cfg(test)]
#[path = "metallib_tests.rs"]
mod metallib_tests;

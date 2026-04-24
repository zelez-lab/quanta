//! MSL → metallib compilation via xcrun.
//!
//! Axiom A1 contract: if xcrun is available and accepts valid MSL,
//! it produces correct metallib. If xcrun fails on macOS, we MUST
//! fail the build — silently skipping would violate the axiom and
//! ship a kernel without an Apple binary.

/// Compile MSL source to metallib binary via xcrun metal + xcrun metallib.
///
/// Returns `Ok(bytes)` on success, `Err(message)` if xcrun fails on macOS,
/// `Ok(None)` only if xcrun is not available (cross-compiling from Linux).
pub fn compile_msl_to_metallib(msl_source: &str) -> Result<Option<Vec<u8>>, String> {
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
    let air_result = std::process::Command::new("xcrun")
        .args([
            "metal",
            "-c",
            "-std=metal3.1", // mesh shaders, ray tracing, bfloat16
            "-O3",           // maximum optimization
            "-ffast-math",   // allow reassociation, no NaN/inf checks
        ])
        .arg(&msl_path)
        .arg("-o")
        .arg(&air_path)
        .stderr(std::process::Stdio::piped())
        .output();

    match air_result {
        Ok(o) if o.status.success() => {}
        Ok(o) => {
            // xcrun found but compilation failed — axiom violation, must not skip
            let err = String::from_utf8_lossy(&o.stderr);
            return Err(format!("xcrun metal failed: {}", err));
        }
        Err(_) => return Ok(None), // xcrun not found — cross-compiling, skip is OK
    };

    // AIR → metallib
    let lib_result = std::process::Command::new("xcrun")
        .args(["metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&lib_path)
        .stderr(std::process::Stdio::piped())
        .output();

    match lib_result {
        Ok(o) if o.status.success() => {
            let bytes =
                std::fs::read(&lib_path).map_err(|e| format!("failed to read metallib: {}", e))?;
            Ok(Some(bytes))
        }
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            Err(format!("xcrun metallib failed: {}", err))
        }
        Err(_) => Ok(None), // metallib tool not found
    }
}

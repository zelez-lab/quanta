//! MSL → metallib compilation via xcrun.

/// Compile MSL source to metallib binary via xcrun metal + xcrun metallib.
/// Returns None if xcrun is not available (e.g., cross-compiling from Linux).
pub fn compile_msl_to_metallib(msl_source: &str) -> Option<Vec<u8>> {
    // Use process ID + thread ID for unique temp files (avoids race when
    // multiple proc macro expansions compile kernels in parallel).
    let unique = format!(
        "{}_{:x}",
        std::process::id(),
        msl_source.len() as u64 ^ (msl_source.as_ptr() as u64)
    );
    let tmp_dir = std::env::temp_dir().join(format!("quanta_metal_{}", unique));
    std::fs::create_dir_all(&tmp_dir).ok()?;

    let msl_path = tmp_dir.join("kernel.metal");
    let air_path = tmp_dir.join("kernel.air");
    let lib_path = tmp_dir.join("kernel.metallib");

    std::fs::write(&msl_path, msl_source).ok()?;

    // MSL → AIR
    let air_result = std::process::Command::new("xcrun")
        .args(["metal", "-c"])
        .arg(&msl_path)
        .arg("-o")
        .arg(&air_path)
        .stderr(std::process::Stdio::piped())
        .output();

    match air_result {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            eprintln!("[quanta] xcrun metal failed: {}", err);
            return None;
        }
        Err(_) => return None, // xcrun not found
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
        Ok(o) if o.status.success() => std::fs::read(&lib_path).ok(),
        Ok(o) => {
            let err = String::from_utf8_lossy(&o.stderr);
            eprintln!("[quanta] xcrun metallib failed: {}", err);
            None
        }
        Err(_) => None,
    }
}

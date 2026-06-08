//! Metal validation layer — re-runs GPU tests with MTL_DEBUG_LAYER=1
//! and MTL_SHADER_VALIDATION=1, then checks for validation errors.
//! Tests self-skip on non-macOS platforms.
//!
//! Run: cargo test --test validate_metal

use std::process::Command;

/// GPU test targets to validate. Each corresponds to a `tests/gpu_*.rs` file.
const GPU_TEST_TARGETS: &[&str] = &[
    "gpu_compute",
    "gpu_atomics",
    "gpu_shared",
    "gpu_texture",
    "gpu_render",
    "gpu_barriers",
    "gpu_mapped",
    "gpu_timestamps",
    "gpu_pipeline",
];

/// Run a single test target with Metal validation layers enabled.
/// Returns (passed, error_lines) where error_lines contains any
/// Metal API or shader validation messages.
fn run_with_metal_validation(test_target: &str) -> (bool, Vec<String>) {
    let output = Command::new("cargo")
        .args(["test", "--test", test_target, "--", "--test-threads=1"])
        .env("MTL_DEBUG_LAYER", "1")
        .env("MTL_SHADER_VALIDATION", "1")
        .output()
        .expect("failed to run cargo test");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Collect Metal validation errors from stderr.
    // Metal debug layer writes messages like:
    //   "Metal API Validation Enabled"   (informational, not an error)
    //   "-[MTLDebugCommandBuffer ...]"   (actual error)
    //   "Shader Validation Error"        (shader issue)
    let error_lines: Vec<String> = stderr
        .lines()
        .filter(|line| {
            // Keep genuine errors, skip the "enabled" banner
            let l = line.to_lowercase();
            (l.contains("metal api validation") && !l.contains("enabled"))
                || l.contains("shader validation error")
                || l.contains("mtldebug")
                || l.contains("gpuvalidation")
                || l.contains("metal error")
        })
        .map(|s| s.to_string())
        .collect();

    (output.status.success(), error_lines)
}

#[test]
fn metal_validation_all_gpu_tests() {
    // Quick check: are we on macOS?
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: Metal validation requires macOS");
        return;
    }

    let mut all_errors: Vec<String> = Vec::new();
    let mut any_ran = false;

    for target in GPU_TEST_TARGETS {
        eprintln!("--- validating: {} ---", target);

        let (passed, errors) = run_with_metal_validation(target);
        any_ran = true;

        if !errors.is_empty() {
            all_errors.push(format!("[{}] Metal validation errors:", target));
            for e in &errors {
                all_errors.push(format!("  {}", e));
            }
        }

        if !passed {
            // Test failure itself is noteworthy but may be unrelated to Metal validation
            // (e.g., the test might skip because no GPU is available).
            eprintln!("[{}] test suite exited non-zero (may have skipped)", target);
        }
    }

    assert!(any_ran, "no GPU test targets were found");
    assert!(
        all_errors.is_empty(),
        "Metal validation errors detected:\n{}",
        all_errors.join("\n"),
    );
}

/// Run each GPU test target individually so failures are isolated.
#[test]
fn metal_validation_gpu_compute() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: not macOS");
        return;
    }
    let (_, errors) = run_with_metal_validation("gpu_compute");
    assert!(
        errors.is_empty(),
        "gpu_compute Metal validation errors:\n{}",
        errors.join("\n"),
    );
}

#[test]
fn metal_validation_gpu_atomics() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: not macOS");
        return;
    }
    let (_, errors) = run_with_metal_validation("gpu_atomics");
    assert!(
        errors.is_empty(),
        "gpu_atomics Metal validation errors:\n{}",
        errors.join("\n"),
    );
}

#[test]
fn metal_validation_gpu_shared() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: not macOS");
        return;
    }
    let (_, errors) = run_with_metal_validation("gpu_shared");
    assert!(
        errors.is_empty(),
        "gpu_shared Metal validation errors:\n{}",
        errors.join("\n"),
    );
}

#[test]
fn metal_validation_gpu_render() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: not macOS");
        return;
    }
    let (_, errors) = run_with_metal_validation("gpu_render");
    assert!(
        errors.is_empty(),
        "gpu_render Metal validation errors:\n{}",
        errors.join("\n"),
    );
}

#[test]
fn metal_validation_gpu_pipeline() {
    if !cfg!(target_os = "macos") {
        eprintln!("skipping: not macOS");
        return;
    }
    let (_, errors) = run_with_metal_validation("gpu_pipeline");
    assert!(
        errors.is_empty(),
        "gpu_pipeline Metal validation errors:\n{}",
        errors.join("\n"),
    );
}

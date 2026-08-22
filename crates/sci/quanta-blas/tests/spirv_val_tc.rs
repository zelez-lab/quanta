#![cfg(feature = "gpu")]
//! The tensor-core GEMM kernels through the JIT SPIR-V emitter, validated
//! by `spirv-val`. This is the only check the `SPV_KHR_cooperative_matrix`
//! lowering gets on a host without a `VK_KHR_cooperative_matrix` device
//! (Metal hosts; lavapipe < Mesa 24.1). Self-skips when spirv-val is not
//! installed, like the root validators.
//!
//! Shape policy is NOT checked here — a device may enumerate 16×16×16 and
//! refuse this 8×8×8 module at pipeline creation; that is the driver's
//! capability gate, not an emitter defect.

use std::io::Write;
use std::process::{Command, Stdio};

const SPIRV_VAL: &str = "/opt/homebrew/bin/spirv-val";

fn spirv_val(label: &str, words: &[u8]) {
    if !std::path::Path::new(SPIRV_VAL).exists() {
        eprintln!("skipping [{label}]: {SPIRV_VAL} not installed");
        return;
    }
    let mut child = Command::new(SPIRV_VAL)
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn spirv-val");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(words)
        .expect("write spirv-val stdin");
    let out = child.wait_with_output().expect("spirv-val run");
    assert!(
        out.status.success(),
        "[{label}] invalid cooperative-matrix SPIR-V:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn tensor_core_kernels_validate_as_khr_cooperative_matrix() {
    for (label, def) in quanta_blas::mixed_tc::tc_kernel_defs_for_validation() {
        let words = quanta_ir::emit_spirv::emit(&def)
            .unwrap_or_else(|e| panic!("[{label}] JIT SPIR-V emission failed: {e}"));
        // The module must declare the extension — no placeholder path left.
        let text = String::from_utf8_lossy(&words);
        assert!(
            text.contains("SPV_KHR_cooperative_matrix"),
            "[{label}] module does not declare SPV_KHR_cooperative_matrix"
        );
        spirv_val(label, &words);
    }
}

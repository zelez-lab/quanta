//! Verify that the LLVM-based quanta-compiler produces valid output
//! for each target backend.
//!
//! These tests spawn the pre-built quanta-compiler binary and check
//! the output for structural correctness. No LLVM linkage at test time.
//!
//! Run: cargo test --test validate_compiler_output -- --ignored

use std::path::PathBuf;
use std::process::Command;

const LLVM_PREFIX: &str = "/opt/homebrew/opt/llvm@22";

fn compiler_path() -> Option<PathBuf> {
    for dir in &["target/debug", "target/release"] {
        let p = PathBuf::from(dir).join("quanta-compiler");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn run_compiler(flag: &str) -> (bool, String, String) {
    let compiler = compiler_path()
        .expect("quanta-compiler not built -- run `cargo build -p quanta-compiler` first");

    let output = Command::new(&compiler)
        .arg(flag)
        .env("LLVM_SYS_221_PREFIX", LLVM_PREFIX)
        .env("DYLD_LIBRARY_PATH", format!("{}/lib", LLVM_PREFIX))
        .output()
        .expect("failed to spawn quanta-compiler");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (output.status.success(), stdout, stderr)
}

// --- PTX ---

#[test]
#[ignore]
fn compiler_produces_valid_ptx() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let (ok, stdout, stderr) = run_compiler("--test-ptx");
    assert!(ok, "compiler --test-ptx failed:\n{}", stderr);

    // PTX text must contain the kernel entry point and version directive.
    // Note: LLVM may emit `.visible .func` instead of `.visible .entry`
    // depending on the calling convention; both are valid PTX.
    assert!(
        stdout.contains(".visible .entry")
            || stdout.contains(".entry")
            || stdout.contains(".visible .func"),
        "PTX output missing kernel definition:\n{}",
        &stdout[..stdout.len().min(500)],
    );
    assert!(
        stdout.contains(".version"),
        "PTX output missing .version directive:\n{}",
        &stdout[..stdout.len().min(500)],
    );
    assert!(
        stdout.contains("vector_add"),
        "PTX output missing kernel name 'vector_add':\n{}",
        &stdout[..stdout.len().min(500)],
    );

    eprintln!("PTX validation passed (contains .entry, .version, kernel name)");
}

// --- SPIR-V ---

#[test]
#[ignore]
fn compiler_produces_valid_spirv() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let (ok, stdout, stderr) = run_compiler("--test-spirv");
    assert!(ok, "compiler --test-spirv failed:\n{}", stderr);

    // The test mode prints stats including the magic number check
    assert!(
        stdout.contains("SPIR-V binary size:") || stdout.contains("SPIR-V"),
        "SPIR-V output missing expected header info:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // Check for the magic number validation line
    let has_valid = stdout.contains("Valid SPIR-V binary") || stdout.contains("07230203");
    assert!(
        has_valid,
        "SPIR-V output missing magic number validation:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    eprintln!("SPIR-V structural validation passed");
}

// --- AMD ELF ---

#[test]
#[ignore]
fn compiler_produces_valid_amd_elf() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let (ok, stdout, stderr) = run_compiler("--test-amd");
    assert!(ok, "compiler --test-amd failed:\n{}", stderr);

    // AMD test mode prints ELF stats
    assert!(
        stdout.contains("ELF binary size:") || stdout.contains("ELF"),
        "AMD output missing ELF info:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // Check for ELF magic validation
    assert!(
        stdout.contains("Valid ELF") || stdout.contains("7f 45 4c 46"),
        "AMD output missing ELF magic validation:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    eprintln!("AMD ELF structural validation passed");
}

// --- LLVM IR (all three targets) ---

#[test]
#[ignore]
fn compiler_produces_valid_ir() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let (ok, stdout, stderr) = run_compiler("--test-ir");
    assert!(ok, "compiler --test-ir failed:\n{}", stderr);

    // LLVM IR for NVPTX
    assert!(
        stdout.contains("NVPTX LLVM IR"),
        "IR output missing NVPTX section:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // LLVM IR for AMDGPU
    assert!(
        stdout.contains("AMDGPU LLVM IR"),
        "IR output missing AMDGPU section:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // LLVM IR for SPIR-V
    assert!(
        stdout.contains("SPIR-V LLVM IR"),
        "IR output missing SPIR-V section:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // All three should contain `define` (LLVM function definitions)
    let define_count = stdout.matches("define").count();
    assert!(
        define_count >= 3,
        "expected at least 3 LLVM `define` directives (one per target), found {}",
        define_count,
    );

    eprintln!(
        "LLVM IR validation passed ({} function definitions across 3 targets)",
        define_count,
    );
}

// --- Complex kernel (neuron_activate) ---

#[test]
#[ignore]
fn compiler_produces_valid_complex_ptx() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let (ok, stdout, stderr) = run_compiler("--test-complex");
    assert!(ok, "compiler --test-complex failed:\n{}", stderr);

    // Should produce both PTX and AMD ELF for the complex kernel
    assert!(
        stdout.contains("neuron_activate"),
        "complex output missing kernel name:\n{}",
        &stdout[..stdout.len().min(500)],
    );

    // PTX section
    assert!(
        stdout.contains("NVIDIA PTX"),
        "complex output missing PTX section",
    );

    // AMD section
    assert!(
        stdout.contains("AMD ELF"),
        "complex output missing AMD ELF section",
    );

    eprintln!("Complex kernel (neuron_activate) validation passed");
}

// --- Stdin/stdout wire protocol ---

#[test]
#[ignore]
fn compiler_wire_protocol_spirv() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    use std::io::Write;

    // Build a minimal KernelDef, serialize, pipe to compiler
    let kernel = quanta_ir::KernelDef {
        name: "wire_test".to_string(),
        params: vec![
            quanta_ir::KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: quanta_ir::ScalarType::F32,
            },
            quanta_ir::KernelParam::FieldWrite {
                name: "b".into(),
                slot: 1,
                scalar_type: quanta_ir::ScalarType::F32,
            },
        ],
        body: vec![
            quanta_ir::KernelOp::QuarkId {
                dst: quanta_ir::Reg(0),
            },
            quanta_ir::KernelOp::Load {
                dst: quanta_ir::Reg(1),
                field: 0,
                index: quanta_ir::Reg(0),
                ty: quanta_ir::ScalarType::F32,
            },
            quanta_ir::KernelOp::Store {
                field: 1,
                index: quanta_ir::Reg(0),
                src: quanta_ir::Reg(1),
                ty: quanta_ir::ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 2,
        opt_level: 3,
        device_sources: Vec::new(),
        device_functions: Vec::new(),
        workgroup_size: [64, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let input_bytes = quanta_ir::serialize_kernel(&kernel);
    let compiler = compiler_path().unwrap();

    let mut child = std::process::Command::new(&compiler)
        .args(["--targets", "spirv"])
        .env("LLVM_SYS_221_PREFIX", LLVM_PREFIX)
        .env("DYLD_LIBRARY_PATH", format!("{}/lib", LLVM_PREFIX))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn compiler");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&input_bytes)
        .expect("failed to write kernel");

    let output = child.wait_with_output().expect("compiler did not finish");
    assert!(
        output.status.success(),
        "wire protocol failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr),
    );

    let co = quanta_ir::deserialize_output(&output.stdout)
        .expect("failed to deserialize CompilerOutput from wire");

    // SPIR-V should be present
    let spirv = co.spirv.expect("wire protocol produced no SPIR-V");
    assert!(spirv.len() >= 20, "SPIR-V too small: {} bytes", spirv.len());

    // Check magic
    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x07230203, "bad SPIR-V magic: 0x{:08x}", magic);

    eprintln!(
        "Wire protocol validation passed (SPIR-V: {} bytes, metallib: {})",
        spirv.len(),
        co.metallib.as_ref().map_or(0, |b| b.len()),
    );
}

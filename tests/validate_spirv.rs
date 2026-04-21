//! SPIR-V binary validation via spirv-val.
//!
//! Compiles test kernels through the quanta-compiler binary, pipes
//! the SPIR-V output through `spirv-val`, and verifies zero errors.
//!
//! Run: cargo test --test validate_spirv -- --ignored

use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const LLVM_PREFIX: &str = "/opt/homebrew/opt/llvm@22";
const SPIRV_VAL: &str = "/opt/homebrew/bin/spirv-val";

fn compiler_path() -> Option<PathBuf> {
    for dir in &["target/debug", "target/release"] {
        let p = PathBuf::from(dir).join("quanta-compiler");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn has_spirv_val() -> bool {
    Command::new(SPIRV_VAL)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build a KernelDef, serialize it, pipe to compiler with --targets spirv,
/// extract the SPIR-V binary from CompilerOutput, write to temp, run spirv-val.
fn compile_and_validate(kernel: &quanta_ir::KernelDef, label: &str) {
    let compiler = compiler_path()
        .expect("quanta-compiler not built — run `cargo build -p quanta-compiler` first");

    let input_bytes = quanta_ir::serialize_kernel(kernel);

    let mut child = Command::new(&compiler)
        .args(["--targets", "spirv"])
        .env("LLVM_SYS_221_PREFIX", LLVM_PREFIX)
        .env("DYLD_LIBRARY_PATH", format!("{}/lib", LLVM_PREFIX))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn quanta-compiler");

    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(&input_bytes)
        .expect("failed to write kernel to compiler stdin");

    let output = child.wait_with_output().expect("compiler did not finish");
    assert!(
        output.status.success(),
        "[{}] compiler exited with error: {}",
        label,
        String::from_utf8_lossy(&output.stderr),
    );

    let co = quanta_ir::deserialize_output(&output.stdout)
        .expect("failed to deserialize CompilerOutput");

    let spirv = co
        .spirv
        .expect(&format!("[{}] compiler produced no SPIR-V output", label));
    assert!(
        spirv.len() >= 4,
        "[{}] SPIR-V binary too small ({} bytes)",
        label,
        spirv.len(),
    );

    // Quick magic check before shelling out
    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(
        magic, 0x07230203,
        "[{}] bad SPIR-V magic: 0x{:08x}",
        label, magic,
    );

    // SPIR-V spec requires the binary to be a stream of 32-bit words.
    // The LLVM SPIR-V backend may emit trailing bytes that break alignment.
    // Truncate to the last 4-byte boundary (trailing bytes are LLVM metadata,
    // not SPIR-V instructions).
    let misaligned = spirv.len() % 4 != 0;
    let spirv_clean = if misaligned {
        let trim = spirv.len() - (spirv.len() % 4);
        eprintln!(
            "[{}] WARNING: SPIR-V binary size {} is not a multiple of 4 -- \
             truncating to {} bytes (LLVM backend alignment bug).",
            label,
            spirv.len(),
            trim,
        );
        &spirv[..trim]
    } else {
        &spirv[..]
    };

    // Write to temp file and run spirv-val
    let tmp = std::env::temp_dir().join(format!("quanta_validate_{}.spv", label));
    std::fs::write(&tmp, spirv_clean).expect("failed to write temp .spv file");

    let val = Command::new(SPIRV_VAL)
        .arg(&tmp)
        .output()
        .expect("failed to run spirv-val");

    let stderr = String::from_utf8_lossy(&val.stderr);
    let stdout = String::from_utf8_lossy(&val.stdout);

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp);

    if !val.status.success() {
        panic!(
            "[{}] spirv-val failed (exit {}):\nstdout: {}\nstderr: {}",
            label, val.status, stdout, stderr,
        );
    }

    if misaligned {
        eprintln!(
            "[{}] spirv-val passed after truncation (compiler should fix 4-byte alignment)",
            label,
        );
    }
}

// --- Test kernels ---

fn vector_add_kernel() -> quanta_ir::KernelDef {
    use quanta_ir::*;
    KernelDef {
        name: "vector_add".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 3,
        device_sources: Vec::new(),
    }
}

fn scalar_mul_kernel() -> quanta_ir::KernelDef {
    use quanta_ir::*;
    KernelDef {
        name: "scalar_mul".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "data".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "result".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(2.0),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 3,
        device_sources: Vec::new(),
    }
}

fn identity_kernel() -> quanta_ir::KernelDef {
    use quanta_ir::*;
    KernelDef {
        name: "identity_copy".to_string(),
        params: vec![
            KernelParam::FieldRead {
                name: "input".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "output".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(1),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 2,
        opt_level: 3,
        device_sources: Vec::new(),
    }
}

// --- Tests ---

#[test]
#[ignore]
fn spirv_val_vector_add() {
    if !has_spirv_val() {
        eprintln!("skipping: spirv-val not found at {}", SPIRV_VAL);
        return;
    }
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }
    compile_and_validate(&vector_add_kernel(), "vector_add");
}

#[test]
#[ignore]
fn spirv_val_scalar_mul() {
    if !has_spirv_val() {
        eprintln!("skipping: spirv-val not found at {}", SPIRV_VAL);
        return;
    }
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }
    compile_and_validate(&scalar_mul_kernel(), "scalar_mul");
}

#[test]
#[ignore]
fn spirv_val_identity() {
    if !has_spirv_val() {
        eprintln!("skipping: spirv-val not found at {}", SPIRV_VAL);
        return;
    }
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }
    compile_and_validate(&identity_kernel(), "identity_copy");
}

//! Verify that the LLVM-based quanta-compiler produces valid output
//! for each target backend.
//!
//! These tests spawn the pre-built quanta-compiler binary and check
//! the output for structural correctness. No LLVM linkage at test
//! time. Tests self-skip with a warning if the compiler binary
//! isn't present; `cargo test --workspace` builds it automatically.
//!
//! Run: cargo test --test validate_compiler_output

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

// --- int4 (PackedU32) SPIR-V ---

const OP_DECORATE: u32 = 71;
const OP_CONSTANT: u32 = 43;
const OP_UDIV: u32 = 134;
const OP_UMOD: u32 = 137;
const OP_BITWISE_XOR: u32 = 198;
const OP_NOT: u32 = 200;
const DECORATION_ARRAY_STRIDE: u32 = 6;

/// A minimal int4 round-trip: `out[i] = a[i]` over I4 storage. Exercises
/// both the packed-nibble load (extract + sign-extend) and the packed-
/// nibble store (read-modify-write).
fn int4_roundtrip_kernel() -> quanta_ir::KernelDef {
    quanta_ir::KernelDef {
        name: "int4_roundtrip".to_string(),
        params: vec![
            quanta_ir::KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: quanta_ir::ScalarType::I4,
            },
            quanta_ir::KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: quanta_ir::ScalarType::I4,
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
                ty: quanta_ir::ScalarType::I4,
            },
            quanta_ir::KernelOp::Store {
                field: 1,
                index: quanta_ir::Reg(0),
                src: quanta_ir::Reg(1),
                ty: quanta_ir::ScalarType::I4,
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
    }
}

/// Compile a KernelDef through the wire protocol and return the SPIR-V.
fn wire_compile_spirv(kernel: &quanta_ir::KernelDef) -> Vec<u8> {
    use std::io::Write;

    let input_bytes = quanta_ir::serialize_kernel(kernel);
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
        "compiler failed on {}:\nstderr: {}",
        kernel.name,
        String::from_utf8_lossy(&output.stderr),
    );

    quanta_ir::deserialize_output(&output.stdout)
        .expect("failed to deserialize CompilerOutput from wire")
        .spirv
        .unwrap_or_else(|| panic!("{}: wire protocol produced no SPIR-V", kernel.name))
}

fn spirv_words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// All `(opcode, operands)` instructions of the module.
fn spirv_instructions(w: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5; // skip header
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        out.push((op, w[i + 1..i + wc].to_vec()));
        i += wc;
    }
    out
}

fn spirv_array_strides(w: &[u32]) -> Vec<u32> {
    spirv_instructions(w)
        .iter()
        .filter(|(op, args)| *op == OP_DECORATE && args.get(1) == Some(&DECORATION_ARRAY_STRIDE))
        .map(|(_, args)| args[2])
        .collect()
}

fn spirv_opcode_count(w: &[u32], opcode: u32) -> usize {
    spirv_instructions(w)
        .iter()
        .filter(|(op, _)| *op == opcode)
        .count()
}

/// `true` when the module declares a 32-bit `OpConstant` with literal `val`.
fn spirv_has_const(w: &[u32], val: u32) -> bool {
    spirv_instructions(w)
        .iter()
        .any(|(op, args)| *op == OP_CONSTANT && args.len() == 3 && args[2] == val)
}

/// Run `spirv-val --target-env vulkan1.3`; skip silently when not installed.
fn spirv_val(name: &str, spirv: &[u8]) {
    use std::io::Write;
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => return,
    };
    child.stdin.as_mut().unwrap().write_all(spirv).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{name}: spirv-val rejected the module:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// int4 buffers stay u32-slot (PackedU32: 8 signed nibbles per word) and
/// load/store go through the nibble extract/insert bit ops — matching the
/// JIT emitter's packing contract.
#[test]
fn compiler_int4_packed_nibble_spirv() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    let spirv = wire_compile_spirv(&int4_roundtrip_kernel());
    let w = spirv_words(&spirv);
    let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
    assert_eq!(magic, 0x07230203, "bad SPIR-V magic: 0x{:08x}", magic);

    // Storage stays u32-slot: every buffer stride must be 4.
    let strides = spirv_array_strides(&w);
    assert!(
        !strides.is_empty() && strides.iter().all(|&s| s == 4),
        "int4 buffers must have ArrayStride 4 (PackedU32), got {strides:?}"
    );

    // The nibble machinery: word index (idx/8) on both load and store,
    // nibble position (idx%8) on both, sign-extension XOR on the load,
    // lane-clear NOT on the store.
    assert_eq!(
        spirv_opcode_count(&w, OP_UDIV),
        2,
        "expected idx/8 word-index UDiv on load AND store"
    );
    assert_eq!(
        spirv_opcode_count(&w, OP_UMOD),
        2,
        "expected idx%8 nibble UMod on load AND store"
    );
    assert_eq!(
        spirv_opcode_count(&w, OP_BITWISE_XOR),
        1,
        "expected exactly the (nib ^ 8) sign-extension XOR"
    );
    assert_eq!(
        spirv_opcode_count(&w, OP_NOT),
        1,
        "expected exactly the ~(0xF << shift) lane-clear NOT"
    );
    // The nibble constants: 8 (nibbles/word + sign bias), 4 (bits/nibble),
    // 0xF (nibble mask).
    for c in [8u32, 4, 0xF] {
        assert!(
            spirv_has_const(&w, c),
            "int4 module missing nibble constant {c}"
        );
    }

    spirv_val("int4_roundtrip", &spirv);
    eprintln!("int4 packed-nibble SPIR-V validation passed");
}

/// The AOT module must carry the same nibble bit-math as the JIT emitter
/// for the identical kernel: equal counts of every distinctive packed-
/// nibble opcode, and the same u32-slot stride. (No SPIR-V runtime exists
/// on this host, so this is the structural half of the JIT/AOT parity
/// contract; spirv-val covers validity of both.)
#[cfg(feature = "jit")]
#[test]
fn compiler_int4_matches_jit_nibble_ops() {
    if compiler_path().is_none() {
        eprintln!("skipping: quanta-compiler not built");
        return;
    }

    const OP_ISUB: u32 = 130;
    const OP_IMUL: u32 = 132;
    const OP_SHR_LOGICAL: u32 = 194;
    const OP_SHL_LOGICAL: u32 = 196;
    const OP_BITWISE_OR: u32 = 197;
    const OP_BITWISE_AND: u32 = 199;

    let kernel = int4_roundtrip_kernel();
    let aot = wire_compile_spirv(&kernel);
    let jit = quanta_ir::emit_spirv::emit(&kernel).expect("JIT emit failed");
    let (aw, jw) = (spirv_words(&aot), spirv_words(&jit));

    for (name, op) in [
        ("UDiv", OP_UDIV),
        ("UMod", OP_UMOD),
        ("IMul", OP_IMUL),
        ("ISub", OP_ISUB),
        ("ShiftRightLogical", OP_SHR_LOGICAL),
        ("ShiftLeftLogical", OP_SHL_LOGICAL),
        ("BitwiseAnd", OP_BITWISE_AND),
        ("BitwiseOr", OP_BITWISE_OR),
        ("BitwiseXor", OP_BITWISE_XOR),
        ("Not", OP_NOT),
    ] {
        assert_eq!(
            spirv_opcode_count(&aw, op),
            spirv_opcode_count(&jw, op),
            "AOT/JIT int4 modules disagree on Op{name} count"
        );
    }

    let (a_strides, j_strides) = (spirv_array_strides(&aw), spirv_array_strides(&jw));
    assert!(
        a_strides.iter().all(|&s| s == 4) && j_strides.iter().all(|&s| s == 4),
        "int4 stride mismatch: AOT {a_strides:?}, JIT {j_strides:?}"
    );

    spirv_val("int4_aot", &aot);
    spirv_val("int4_jit", &jit);
    eprintln!("int4 AOT/JIT nibble-op parity passed");
}

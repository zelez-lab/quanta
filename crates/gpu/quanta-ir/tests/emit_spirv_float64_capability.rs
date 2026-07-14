//! Regression: a kernel that uses `f64` must declare `OpCapability
//! Float64`.
//!
//! The emitter creates `OpTypeFloat 64` for F64-typed loads, binops and
//! stores. SPIR-V requires the Float64 capability to be declared before
//! any module references a 64-bit float type. When it was missing, some
//! drivers (e.g. lavapipe) passed `vkCreateShaderModule` but then failed
//! `vkCreateComputePipelines` with `VK_ERROR_UNKNOWN` (-13) — the failure
//! that broke the Vulkan `op_matrix` lane on the F64 cases.

#![cfg(feature = "jit")]

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

/// `OpCapability` opcode (SPIR-V §3.32.1).
const OP_CAPABILITY: u16 = 17;
/// `Capability.Float64` enumerant (SPIR-V §3.31).
const CAPABILITY_FLOAT64: u32 = 10;

/// A minimal `out = a + b` kernel over one element of the given scalar
/// type. Shape mirrors `tests/diff/op_matrix.rs::build_binop_def`.
fn add_kernel(ty: ScalarType) -> KernelDef {
    KernelDef {
        name: "add".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Decode the little-endian u32 word stream of a SPIR-V binary, skipping
/// the 5-word header.
fn words(spirv: &[u8]) -> Vec<u32> {
    assert_eq!(spirv.len() % 4, 0, "SPIR-V is not word-aligned");
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Walk the instruction stream and collect the operand of every
/// `OpCapability`.
fn declared_capabilities(spirv: &[u8]) -> Vec<u32> {
    let words = words(spirv);
    let mut caps = Vec::new();
    let mut i = 5; // skip header
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xFFFF) as u16;
        let word_count = (word >> 16) as usize;
        assert!(word_count >= 1, "zero-length SPIR-V instruction");
        if opcode == OP_CAPABILITY {
            caps.push(words[i + 1]);
        }
        i += word_count;
    }
    caps
}

#[test]
fn f64_kernel_declares_float64_capability() {
    let spirv = emit_spirv::emit(&add_kernel(ScalarType::F64)).expect("emit f64 kernel");
    let caps = declared_capabilities(&spirv);
    assert!(
        caps.contains(&CAPABILITY_FLOAT64),
        "f64 kernel must declare OpCapability Float64; got {:?}",
        caps
    );
}

#[test]
fn f32_kernel_does_not_declare_float64_capability() {
    // The capability is only emitted on demand — an f32 kernel must not
    // pull in Float64 (which would needlessly raise the device feature
    // requirement).
    let spirv = emit_spirv::emit(&add_kernel(ScalarType::F32)).expect("emit f32 kernel");
    let caps = declared_capabilities(&spirv);
    assert!(
        !caps.contains(&CAPABILITY_FLOAT64),
        "f32 kernel must not declare Float64; got {:?}",
        caps
    );
}

/// Validate the emitted module with `spirv-val` when it is on PATH. This
/// is the direct check for the original bug: `spirv-val` rejects a module
/// that references `OpTypeFloat 64` without the Float64 capability.
#[test]
fn f64_kernel_passes_spirv_val() {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let spirv = emit_spirv::emit(&add_kernel(ScalarType::F64)).expect("emit f64 kernel");

    let mut child = match Command::new("spirv-val")
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("spirv-val not found on PATH; skipping validation");
            return;
        }
    };

    child
        .stdin
        .take()
        .expect("spirv-val stdin")
        .write_all(&spirv)
        .expect("pipe spirv to spirv-val");
    let output = child.wait_with_output().expect("wait spirv-val");
    assert!(
        output.status.success(),
        "spirv-val rejected the f64 module:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

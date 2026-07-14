//! Regression: numeric `Cast` ops must emit the correct SPIR-V
//! conversion opcode, not a reinterpret.
//!
//! The conversion opcode constants were mis-numbered, so f→int and
//! int→f casts were spelled `OpFConvert` / `OpSConvert` (same-domain
//! conversions). Drivers execute those on mismatched types as a
//! reinterpret: `cast_f32_to_i32(1.0)` returned `0x3F800000` (the bits
//! of 1.0) instead of `1`. The op-matrix Vulkan lane caught it on
//! lavapipe once the pipeline-creation bugs were fixed.
//!
//! These tests pin the emitted opcode by decoding the word stream — no
//! external tooling needed — so the numbering can't silently drift again.

#![cfg(feature = "jit")]

use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

// SPIR-V §3.42 numeric conversion opcodes.
const OP_CONVERT_F_TO_U: u16 = 109;
const OP_CONVERT_F_TO_S: u16 = 110;
const OP_CONVERT_S_TO_F: u16 = 111;
const OP_CONVERT_U_TO_F: u16 = 112;
// The wrong opcodes the bug used to emit, asserted absent.
const OP_S_CONVERT: u16 = 114;
const OP_F_CONVERT: u16 = 115;

fn cast_kernel(from: ScalarType, to: ScalarType) -> KernelDef {
    KernelDef {
        name: "cast".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: from,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: to,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: from,
            },
            KernelOp::Cast {
                dst: Reg(2),
                src: Reg(1),
                from,
                to,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: to,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Collect the opcode (low 16 bits) of every instruction in the module.
fn opcodes(spirv: &[u8]) -> Vec<u16> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut ops = Vec::new();
    let mut i = 5; // skip header
    while i < words.len() {
        let word = words[i];
        let opcode = (word & 0xFFFF) as u16;
        let word_count = (word >> 16) as usize;
        assert!(word_count >= 1);
        ops.push(opcode);
        i += word_count;
    }
    ops
}

fn assert_uses(from: ScalarType, to: ScalarType, expected: u16, forbidden: u16) {
    let spirv = emit_spirv::emit(&cast_kernel(from, to)).expect("emit cast kernel");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&expected),
        "cast {from:?}->{to:?} must emit opcode {expected}; opcodes: {ops:?}"
    );
    assert!(
        !ops.contains(&forbidden),
        "cast {from:?}->{to:?} must not emit the reinterpret opcode {forbidden}"
    );
}

#[test]
fn f32_to_i32_converts() {
    assert_uses(
        ScalarType::F32,
        ScalarType::I32,
        OP_CONVERT_F_TO_S,
        OP_F_CONVERT,
    );
}

#[test]
fn f32_to_u32_converts() {
    assert_uses(
        ScalarType::F32,
        ScalarType::U32,
        OP_CONVERT_F_TO_U,
        OP_F_CONVERT,
    );
}

#[test]
fn i32_to_f32_converts() {
    assert_uses(
        ScalarType::I32,
        ScalarType::F32,
        OP_CONVERT_S_TO_F,
        OP_S_CONVERT,
    );
}

#[test]
fn u32_to_f32_converts() {
    assert_uses(
        ScalarType::U32,
        ScalarType::F32,
        OP_CONVERT_U_TO_F,
        OP_S_CONVERT,
    );
}

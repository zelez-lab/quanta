//! Regression: comparison `Cmp` ops must emit the correct SPIR-V opcode.
//!
//! The signed/unsigned GreaterThanEqual and LessThanEqual opcode pairs
//! were swapped, and `OpFOrdNotEqual` was numbered 181 — which is
//! actually `OpFUnordEqual`, an *inverted* result. Both produce valid
//! SPIR-V (so `spirv-val` passes) but wrong values: `lt_i32` ran as
//! `OpULessThanEqual`, `ne_f32` as `OpFUnordEqual`. Only running the
//! op-matrix on lavapipe surfaced it. These tests pin the emitted opcode
//! by decoding the word stream so the numbering can't drift again.
//!
//! Opcode values verified against `spirv-dis` (SPIR-V §3.42.18).

#![cfg(feature = "jit")]

use quanta_ir::{CmpOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

const OP_IEQUAL: u16 = 170;
const OP_INOT_EQUAL: u16 = 171;
const OP_UGREATER_THAN: u16 = 172;
const OP_SGREATER_THAN: u16 = 173;
const OP_UGREATER_THAN_EQUAL: u16 = 174;
const OP_SGREATER_THAN_EQUAL: u16 = 175;
const OP_ULESS_THAN: u16 = 176;
const OP_SLESS_THAN: u16 = 177;
const OP_ULESS_THAN_EQ: u16 = 178;
const OP_SLESS_THAN_EQUAL: u16 = 179;
const OP_FORD_EQUAL: u16 = 180;
const OP_FORD_NOT_EQUAL: u16 = 182;
const OP_FORD_LESS_THAN: u16 = 184;
const OP_FORD_GREATER_THAN: u16 = 186;

fn cmp_kernel(ty: ScalarType, op: CmpOp) -> KernelDef {
    KernelDef {
        name: "cmp".into(),
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
                scalar_type: ScalarType::U32,
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
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op,
                ty,
            },
            KernelOp::Cast {
                dst: Reg(4),
                src: Reg(3),
                from: ScalarType::Bool,
                to: ScalarType::U32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(4),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 5,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

fn opcodes(spirv: &[u8]) -> Vec<u16> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut ops = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let word = words[i];
        ops.push((word & 0xFFFF) as u16);
        let wc = (word >> 16) as usize;
        assert!(wc >= 1);
        i += wc;
    }
    ops
}

fn assert_cmp(ty: ScalarType, op: CmpOp, expected: u16) {
    let spirv = emit_spirv::emit(&cmp_kernel(ty, op)).expect("emit cmp kernel");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&expected),
        "cmp {op:?} {ty:?} must emit opcode {expected}; opcodes: {ops:?}"
    );
}

#[test]
fn integer_cmp_opcodes() {
    assert_cmp(ScalarType::U32, CmpOp::Eq, OP_IEQUAL);
    assert_cmp(ScalarType::U32, CmpOp::Ne, OP_INOT_EQUAL);
    assert_cmp(ScalarType::U32, CmpOp::Lt, OP_ULESS_THAN);
    assert_cmp(ScalarType::U32, CmpOp::Le, OP_ULESS_THAN_EQ);
    assert_cmp(ScalarType::U32, CmpOp::Gt, OP_UGREATER_THAN);
    assert_cmp(ScalarType::U32, CmpOp::Ge, OP_UGREATER_THAN_EQUAL);
    assert_cmp(ScalarType::I32, CmpOp::Lt, OP_SLESS_THAN);
    assert_cmp(ScalarType::I32, CmpOp::Le, OP_SLESS_THAN_EQUAL);
    assert_cmp(ScalarType::I32, CmpOp::Gt, OP_SGREATER_THAN);
    assert_cmp(ScalarType::I32, CmpOp::Ge, OP_SGREATER_THAN_EQUAL);
}

#[test]
fn float_cmp_opcodes() {
    assert_cmp(ScalarType::F32, CmpOp::Eq, OP_FORD_EQUAL);
    assert_cmp(ScalarType::F32, CmpOp::Ne, OP_FORD_NOT_EQUAL);
    assert_cmp(ScalarType::F32, CmpOp::Lt, OP_FORD_LESS_THAN);
    assert_cmp(ScalarType::F32, CmpOp::Gt, OP_FORD_GREATER_THAN);
}

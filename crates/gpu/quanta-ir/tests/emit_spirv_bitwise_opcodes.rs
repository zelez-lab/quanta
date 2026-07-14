//! Regression: bitwise `BinOp`s must emit the correct SPIR-V opcode.
//!
//! `OpBitwiseOr` / `OpBitwiseXor` / `OpBitwiseAnd` are opcodes 197 / 198 /
//! 199. The emitter had them rotated (AND→197=Or, OR→198=Xor, XOR→199=And),
//! producing valid-but-wrong modules: `bitxor` ran as `OpBitwiseAnd`, and
//! the rotate decomposition's masking `OpBitwiseAnd` ran as `OpBitwiseOr`,
//! corrupting every i64 rotate. Pinned by decoding the word stream.
//!
//! Opcode values verified against `spirv-dis` (SPIR-V §3.42.14).

#![cfg(feature = "jit")]

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
const OP_SHIFT_RIGHT_ARITHMETIC: u16 = 195;
const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
const OP_BITWISE_OR: u16 = 197;
const OP_BITWISE_XOR: u16 = 198;
const OP_BITWISE_AND: u16 = 199;

fn binop_kernel(ty: ScalarType, op: BinOp) -> KernelDef {
    KernelDef {
        name: "op".into(),
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
                op,
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

fn assert_op(ty: ScalarType, op: BinOp, expected: u16) {
    let spirv = emit_spirv::emit(&binop_kernel(ty, op)).expect("emit binop kernel");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&expected),
        "{op:?} {ty:?} must emit opcode {expected}; opcodes: {ops:?}"
    );
}

#[test]
fn bitwise_opcodes() {
    assert_op(ScalarType::U32, BinOp::BitAnd, OP_BITWISE_AND);
    assert_op(ScalarType::U32, BinOp::BitOr, OP_BITWISE_OR);
    assert_op(ScalarType::U32, BinOp::BitXor, OP_BITWISE_XOR);
}

#[test]
fn shift_opcodes() {
    assert_op(ScalarType::U32, BinOp::Shl, OP_SHIFT_LEFT_LOGICAL);
    assert_op(ScalarType::U32, BinOp::Shr, OP_SHIFT_RIGHT_LOGICAL);
    assert_op(ScalarType::I32, BinOp::Shr, OP_SHIFT_RIGHT_ARITHMETIC);
}

/// The rotate decomposition masks the shift amount with `OpBitwiseAnd`;
/// a rotated bitwise opcode silently corrupts every rotate.
#[test]
fn rotate_uses_bitwise_and() {
    let spirv = emit_spirv::emit(&binop_kernel(ScalarType::U32, BinOp::Rotl)).expect("emit rotl");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&OP_BITWISE_AND),
        "rotl must mask with OpBitwiseAnd"
    );
    assert!(
        ops.contains(&OP_BITWISE_OR),
        "rotl must combine halves with OpBitwiseOr"
    );
}

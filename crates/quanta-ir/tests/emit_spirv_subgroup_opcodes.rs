//! Regression: subgroup reduce min/max must emit the correct
//! `OpGroupNonUniform{S,U,F}{Min,Max}` opcode.
//!
//! The min/max block (353..=358) was numbered one too high (SMin→354=UMin,
//! …, FMax→359=BitwiseAnd), so signed subgroup reductions ran as unsigned
//! and float-max emitted a bitwise op. Valid SPIR-V, wrong result. Pinned
//! by decoding the word stream. Values verified against `spirv-dis`.

#![cfg(feature = "jit")]

use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

const OP_GROUP_NON_UNIFORM_SMIN: u16 = 353;
const OP_GROUP_NON_UNIFORM_UMIN: u16 = 354;
const OP_GROUP_NON_UNIFORM_FMIN: u16 = 355;
const OP_GROUP_NON_UNIFORM_SMAX: u16 = 356;
const OP_GROUP_NON_UNIFORM_UMAX: u16 = 357;
const OP_GROUP_NON_UNIFORM_FMAX: u16 = 358;

fn reduce_kernel(ty: ScalarType, max: bool) -> KernelDef {
    let reduce = if max {
        KernelOp::SubgroupReduceMax {
            dst: Reg(2),
            src: Reg(1),
            ty,
        }
    } else {
        KernelOp::SubgroupReduceMin {
            dst: Reg(2),
            src: Reg(1),
            ty,
        }
    };
    KernelDef {
        name: "r".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
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
            reduce,
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty,
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

fn opcodes(spirv: &[u8]) -> Vec<u16> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut ops = Vec::new();
    let mut i = 5;
    while i < words.len() {
        ops.push((words[i] & 0xFFFF) as u16);
        let wc = (words[i] >> 16) as usize;
        assert!(wc >= 1);
        i += wc;
    }
    ops
}

fn assert_reduce(ty: ScalarType, max: bool, expected: u16) {
    let spirv = emit_spirv::emit(&reduce_kernel(ty, max)).expect("emit reduce kernel");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&expected),
        "subgroup reduce (max={max}) {ty:?} must emit opcode {expected}; opcodes: {ops:?}"
    );
}

#[test]
fn subgroup_reduce_min_opcodes() {
    assert_reduce(ScalarType::I32, false, OP_GROUP_NON_UNIFORM_SMIN);
    assert_reduce(ScalarType::U32, false, OP_GROUP_NON_UNIFORM_UMIN);
    assert_reduce(ScalarType::F32, false, OP_GROUP_NON_UNIFORM_FMIN);
}

#[test]
fn subgroup_reduce_max_opcodes() {
    assert_reduce(ScalarType::I32, true, OP_GROUP_NON_UNIFORM_SMAX);
    assert_reduce(ScalarType::U32, true, OP_GROUP_NON_UNIFORM_UMAX);
    assert_reduce(ScalarType::F32, true, OP_GROUP_NON_UNIFORM_FMAX);
}

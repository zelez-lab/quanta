//! Regression tests for T1405 parity in the MSL emitter.
//!
//! When a `KernelOp::Loop`'s `count` register was defined by a Const op
//! with a small positive value (1..=8), the MSL emitter emits
//! `#pragma clang loop unroll(full)` immediately before the `for`
//! statement. The Metal compiler (Clang-based) honors this and fully
//! unrolls. Larger or non-const counts emit the bare `for` with no
//! pragma. Threshold matches the SPIR-V emitter (T1405 fix).

#![cfg(feature = "jit")]

use quanta_ir::{
    ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_msl,
};

const UNROLL_PRAGMA: &str = "#pragma clang loop unroll(full)";

fn loop_kernel_with_count_setup(
    count_setup: Vec<KernelOp>,
    count_reg: Reg,
    next_reg: u32,
) -> KernelDef {
    let mut body = count_setup;
    body.push(KernelOp::Loop {
        count: count_reg,
        iter_reg: Reg(next_reg),
        body: vec![],
    });
    KernelDef {
        name: "test_loop".into(),
        params: vec![KernelParam::FieldWrite {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body,
        body_source: None,
        next_reg: next_reg + 1,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn loop_with_const_count_4_emits_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(4),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(
        msl.contains(UNROLL_PRAGMA),
        "count=4 should emit unroll pragma; got:\n{msl}"
    );
}

#[test]
fn loop_with_const_count_1_emits_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(1),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(msl.contains(UNROLL_PRAGMA));
}

#[test]
fn loop_with_const_count_8_emits_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(8),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(msl.contains(UNROLL_PRAGMA));
}

#[test]
fn loop_with_const_count_9_no_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(9),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(
        !msl.contains(UNROLL_PRAGMA),
        "count=9 must NOT emit unroll pragma; got:\n{msl}"
    );
}

#[test]
fn loop_with_const_count_0_no_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(!msl.contains(UNROLL_PRAGMA));
}

#[test]
fn loop_with_i32_const_count_5_emits_unroll_pragma() {
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::I32(5),
        }],
        Reg(0),
        1,
    );
    let msl = emit_msl::emit(&def).expect("emit should succeed");
    assert!(msl.contains(UNROLL_PRAGMA));
}

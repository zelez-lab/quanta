//! Regression tests for T1405: Loop unroll on small const counts.
//!
//! When a `KernelOp::Loop`'s `count` register was defined by a Const op
//! with a small positive value (1..=8), the SPIR-V emitter applies
//! `LOOP_CONTROL_UNROLL` to the `OpLoopMerge`. Larger or non-const
//! counts fall back to `LOOP_CONTROL_NONE`. This pinning test asserts
//! the loop-control word in the emitted SPIR-V matches expectations.

#![cfg(feature = "jit")]

use quanta_ir::{
    ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv,
};

/// SPIR-V opcode for `OpLoopMerge` (per SPIR-V spec §3.49.5).
const OP_LOOP_MERGE: u16 = 246;
/// `LoopControl.Unroll` mask bit (§3.51.7).
const LOOP_CONTROL_UNROLL: u32 = 0x1;
/// `LoopControl.None`.
const LOOP_CONTROL_NONE: u32 = 0;

/// Build a minimal kernel that consists of a single Loop with a given
/// count source. `count_setup` emits the ops that define the `count`
/// register (e.g. a Const op or a Load).
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

/// Scan a SPIR-V word stream for the FIRST `OpLoopMerge` instruction
/// and return its `LoopControl` operand (the 3rd operand of the op).
fn find_loop_merge_control(spv: &[u8]) -> Option<u32> {
    // SPIR-V is little-endian 32-bit words. The first 5 words are the
    // module header; instructions follow.
    let mut words: Vec<u32> = Vec::with_capacity(spv.len() / 4);
    for chunk in spv.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let mut i = 5;
    while i < words.len() {
        let head = words[i];
        let word_count = (head >> 16) as usize;
        let opcode = (head & 0xFFFF) as u16;
        if word_count == 0 {
            // Malformed; bail out so we don't loop forever.
            return None;
        }
        if opcode == OP_LOOP_MERGE && i + 3 < words.len() {
            // OpLoopMerge: [head, merge_label, continue_label, loop_control, ...]
            return Some(words[i + 3]);
        }
        i += word_count;
    }
    None
}

#[test]
fn loop_with_const_count_4_uses_unroll() {
    // count = 4 (const) → expect LOOP_CONTROL_UNROLL
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(4),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(
        ctrl, LOOP_CONTROL_UNROLL,
        "count=4 should yield LOOP_CONTROL_UNROLL, got 0x{ctrl:x}"
    );
}

#[test]
fn loop_with_const_count_1_uses_unroll() {
    // Boundary: count = 1 (smallest valid unroll candidate).
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(1),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(ctrl, LOOP_CONTROL_UNROLL);
}

#[test]
fn loop_with_const_count_8_uses_unroll() {
    // Boundary: count = 8 (largest unroll candidate).
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(8),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(ctrl, LOOP_CONTROL_UNROLL);
}

#[test]
fn loop_with_const_count_9_no_unroll() {
    // Above threshold: count = 9 → LOOP_CONTROL_NONE.
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(9),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(
        ctrl, LOOP_CONTROL_NONE,
        "count=9 is above the unroll threshold; expected 0, got 0x{ctrl:x}"
    );
}

#[test]
fn loop_with_const_count_0_no_unroll() {
    // Boundary: count = 0 falls out of the 1..=8 range → no unroll.
    // (A zero-count loop is technically degenerate; we don't optimize it.)
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(0),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(ctrl, LOOP_CONTROL_NONE);
}

#[test]
fn loop_with_i32_const_count_5_uses_unroll() {
    // I32 path: tracked separately from U32.
    let def = loop_kernel_with_count_setup(
        vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::I32(5),
        }],
        Reg(0),
        1,
    );
    let spv = emit_spirv::emit(&def).expect("emit should succeed");
    let ctrl = find_loop_merge_control(&spv).expect("OpLoopMerge should be present");
    assert_eq!(ctrl, LOOP_CONTROL_UNROLL);
}

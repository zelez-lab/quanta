//! Regression tests for mutable-register semantics in the MSL emitter.
//!
//! The KernelOp contract is mutable registers: a register may be written
//! more than once (loop-carried accumulator, `Copy` reassignment) and may
//! be written inside a Branch/Loop body and read after the merge. MSL is
//! C++, so the old per-write `<type> rN = <expr>;` emission produced
//! `redefinition of 'rN'` for the second write and `use of undeclared
//! identifier` for cross-scope reads (observed on real Metal with the FFT
//! butterfly and narrow-dtype GEMM kernels). The fix mirrors the SPIR-V
//! emitter: the `reg_mutability` pre-pass flags such registers, which are
//! declared once at function entry and plainly assigned at each write.

#![cfg(feature = "jit")]

use quanta_ir::{BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_msl};

fn kernel(body: Vec<KernelOp>, next_reg: u32) -> KernelDef {
    KernelDef {
        name: "test_mut_regs".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "input".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body,
        body_source: None,
        next_reg,
        opt_level: 3,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Number of *typed declarations* of register `reg` in the MSL text
/// (e.g. `float r3 =` / `uint r3;`), as opposed to plain assignments.
fn decl_count(msl: &str, reg: u32) -> usize {
    const TYPES: [&str; 10] = [
        "float", "double", "half", "uint", "int", "ulong", "long", "ushort", "short", "bool",
    ];
    // Leading space = word boundary (declarations are indented / mid-line
    // after `{ `), so the `int` needle doesn't also match inside `uint`.
    TYPES
        .iter()
        .map(|t| {
            msl.match_indices(&format!(" {} r{} ", t, reg)).count()
                + msl.match_indices(&format!(" {} r{};", t, reg)).count()
        })
        .sum()
}

/// Loop-carried f32 accumulator: `acc = 0.0; for i { acc = acc + x }`.
/// This is the FFT-butterfly / GEMM-accumulator shape that emitted
/// `float rN = ...` twice (entry Const + in-loop BinOp) → `redefinition
/// of 'rN'` on Metal.
#[test]
fn loop_carried_accumulator_is_declared_once() {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::F32(0.0),
        },
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::U32(16),
        },
        KernelOp::Loop {
            count: Reg(3),
            iter_reg: Reg(4),
            body: vec![KernelOp::BinOp {
                dst: Reg(2),
                a: Reg(2),
                b: Reg(1),
                op: BinOp::Add,
                ty: ScalarType::F32,
            }],
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::F32,
        },
    ];
    let msl = emit_msl::emit(&kernel(body, 5)).expect("emit should succeed");
    assert_eq!(
        decl_count(&msl, 2),
        1,
        "loop-carried accumulator r2 must be declared exactly once; got:\n{msl}"
    );
    // The in-loop write must be a plain assignment, not a re-declaration.
    assert!(
        msl.contains("r2 = r2 + r1;") && !msl.contains("float r2 = r2 + r1;"),
        "in-loop write must not re-declare r2; got:\n{msl}"
    );
}

/// Single write inside a Branch arm, read after the merge: the old
/// emitter's declaration was scoped to the `if` block, so the post-merge
/// read was `use of undeclared identifier`.
#[test]
fn branch_arm_write_read_after_merge_is_hoisted() {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::Bool(true),
        },
        KernelOp::Branch {
            cond: Reg(1),
            then_ops: vec![KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(7),
            }],
            else_ops: vec![KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(3),
            }],
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::U32,
        },
    ];
    let msl = emit_msl::emit(&kernel(body, 3)).expect("emit should succeed");
    assert_eq!(
        decl_count(&msl, 2),
        1,
        "branch-written r2 must have a single (hoisted) declaration; got:\n{msl}"
    );
    // Both arm writes are plain assignments.
    assert!(
        msl.contains("r2 = 7u;") && msl.contains("r2 = 3u;"),
        "arm writes must be plain assignments; got:\n{msl}"
    );
    // The hoisted declaration precedes the branch.
    let decl_pos = msl.find("uint r2 ").expect("hoisted uint r2 declaration");
    let branch_pos = msl.find("if (r1)").expect("branch");
    assert!(
        decl_pos < branch_pos,
        "hoisted declaration must dominate the branch; got:\n{msl}"
    );
}

/// A single-def `Copy` destination must still be *declared* — the old
/// emitter emitted a bare `rN = rM;` unconditionally, producing `use of
/// undeclared identifier 'rN'` when the Copy was the register's only
/// write.
#[test]
fn single_def_copy_gets_a_typed_declaration() {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Copy {
            dst: Reg(1),
            src: Reg(0),
            ty: ScalarType::U32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(1),
            ty: ScalarType::U32,
        },
    ];
    let msl = emit_msl::emit(&kernel(body, 2)).expect("emit should succeed");
    assert!(
        msl.contains("uint r1 = r0;"),
        "single-def Copy dst needs a typed declaration; got:\n{msl}"
    );
}

/// Straight-line single-def registers keep the pure-SSA inline
/// declaration — no hoisting, no behavior change.
#[test]
fn single_def_registers_stay_inline_ssa() {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::BinOp {
            dst: Reg(2),
            a: Reg(1),
            b: Reg(1),
            op: BinOp::Mul,
            ty: ScalarType::F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::F32,
        },
    ];
    let msl = emit_msl::emit(&kernel(body, 3)).expect("emit should succeed");
    assert!(
        msl.contains("float r2 = r1 * r1;"),
        "single-def temp keeps its inline declaration; got:\n{msl}"
    );
    // No entry-block zero-init hoists for pure SSA temps.
    assert_eq!(decl_count(&msl, 2), 1);
    assert!(
        !msl.contains("= (float)0;"),
        "no hoisted declarations expected for a pure-SSA kernel; got:\n{msl}"
    );
}

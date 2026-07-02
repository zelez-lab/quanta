//! Regression: the SPIR-V emitter must not mix signed `%int` and unsigned
//! `%uint` 32-bit ints in one instruction or a loop-carried phi.
//!
//! `%int` (OpTypeInt 32 1) and `%uint` (OpTypeInt 32 0) are DISTINCT SPIR-V
//! types. An `OpBitwiseOr %int %uint_value ...` or an `OpPhi %uint %int_value`
//! is invalid SPIR-V — Metal's MSL hid it (implicit int/uint conversion), but
//! Vulkan rejects it at `vkCreateComputePipelines` (`VK_ERROR_UNKNOWN`), which
//! is what broke every integer-heavy kernel (gemm, reductions, index math) on
//! the Raspberry Pi's Vulkan backend. The fix coerces operands to a consistent
//! type (`coerce_to` inserts an `OpBitcast`); these tests pin that it stays
//! valid by running `spirv-val` on modules built to trigger the old bug.

#![cfg(feature = "jit")]

use quanta_ir::{
    BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType, emit_spirv,
};

fn assert_spirv_val(spirv: &[u8]) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut child = match Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
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
        .as_mut()
        .unwrap()
        .write_all(spirv)
        .expect("write spirv to spirv-val");
    let out = child.wait_with_output().expect("spirv-val run");
    assert!(
        out.status.success(),
        "spirv-val rejected the module:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A kernel that mixes u32 index math with an i32 accumulator inside a loop:
/// the exact shape that produced `OpBitwiseOr %int %uint` and `OpPhi %uint
/// %int` in the tiled gemm.
fn mixed_int_loop_kernel() -> KernelDef {
    let iter = Reg(4);
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        // acc: i32, loop-carried
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::I32(0),
        },
        // count: u32
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(4),
        },
        // signed mask constant
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::I32(240),
        },
        KernelOp::Loop {
            count: Reg(2),
            iter_reg: iter,
            body: vec![
                // u32 index math
                KernelOp::Const {
                    dst: Reg(5),
                    value: ConstValue::U32(2),
                },
                KernelOp::BinOp {
                    dst: Reg(6),
                    a: iter,
                    b: Reg(5),
                    op: BinOp::Mul,
                    ty: ScalarType::U32,
                },
                // MIX: bitor an i32 acc with a u32-derived value (i32 result)
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(1),
                    b: Reg(6),
                    op: BinOp::BitOr,
                    ty: ScalarType::I32,
                },
                // signed compare i32 vs i32
                KernelOp::Cmp {
                    dst: Reg(8),
                    a: Reg(7),
                    b: Reg(3),
                    op: CmpOp::Lt,
                    ty: ScalarType::I32,
                },
                // acc = acc + 1 (i32, carried across the loop)
                KernelOp::Const {
                    dst: Reg(9),
                    value: ConstValue::I32(1),
                },
                KernelOp::BinOp {
                    dst: Reg(10),
                    a: Reg(1),
                    b: Reg(9),
                    op: BinOp::Add,
                    ty: ScalarType::I32,
                },
                KernelOp::Copy {
                    dst: Reg(1),
                    src: Reg(10),
                    ty: ScalarType::I32,
                },
            ],
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(1),
            ty: ScalarType::I32,
        },
    ];
    KernelDef {
        name: "mixed_int_loop".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::I32,
        }],
        body,
        body_source: None,
        next_reg: 11,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn mixed_int_signedness_loop_is_valid_spirv() {
    let spirv = emit_spirv::emit(&mixed_int_loop_kernel()).expect("emit");
    assert_spirv_val(&spirv);
}

/// A bitwise op whose operands arrive as different int signedness must not
/// produce a mixed-type instruction.
fn bitor_mixed_kernel() -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(255),
        }, // u32
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::I32(15),
        }, // i32
        // i32-tagged BitOr with a u32 operand → would emit OpBitwiseOr %int %uint
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::BitOr,
            ty: ScalarType::I32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(3),
            ty: ScalarType::I32,
        },
    ];
    KernelDef {
        name: "bitor_mixed".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::I32,
        }],
        body,
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

#[test]
fn bitor_mixed_signedness_is_valid_spirv() {
    let spirv = emit_spirv::emit(&bitor_mixed_kernel()).expect("emit");
    assert_spirv_val(&spirv);
}

// ── Mutable-register (demoted OpVariable) regression shapes ──────────────────
//
// The KernelOp contract is mutable-register semantics: a register may be
// written in a Branch arm / Loop body and read after the merge. The emitter
// used to model registers as pure SSA renames, which (a) leaked whichever
// arm's id was emitted last past a Branch merge (silent miscompile of
// `let idx = if c { i } else { 0 }`) and (b) produced dominance-invalid
// modules when a loop-carried register was read past a bypassable loop
// (`spirv-val`: "ID defined in block X does not dominate its use"). Both
// shapes are now demoted to Function-storage OpVariables; these tests pin
// that the emitted modules stay valid.

/// `idx = if i < n { i } else { 999 }` — a register written in a Branch arm
/// (entry-Const init + re-Copy) and read after the merge.
fn branch_select_kernel() -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        // idx: entry init with the else value
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(999),
        },
        // n
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(5),
        },
        KernelOp::Cmp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(2),
            op: CmpOp::Lt,
            ty: ScalarType::U32,
        },
        KernelOp::Branch {
            cond: Reg(3),
            then_ops: vec![KernelOp::Copy {
                dst: Reg(1),
                src: Reg(0),
                ty: ScalarType::U32,
            }],
            else_ops: vec![],
        },
        // Post-merge read: must observe the then-arm write iff taken.
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(1),
            ty: ScalarType::U32,
        },
    ];
    KernelDef {
        name: "branch_select".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body,
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

#[test]
fn branch_arm_write_read_after_merge_is_valid_spirv() {
    let spirv = emit_spirv::emit(&branch_select_kernel()).expect("emit");
    assert_spirv_val(&spirv);
}

/// A loop-carried f32 accumulator whose Loop sits inside a *bypassable*
/// Branch arm, read after the Branch merge — the `gemm_f32_naive` shape
/// that produced the dominance error (the old emitter re-pointed the
/// register at the loop-header phi, which does not dominate the merge).
fn loop_in_branch_carried_kernel() -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        // acc: f32, loop-carried
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::F32(0.0),
        },
        // trip count
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(4),
        },
        // bound for the bypassable branch
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::U32(100),
        },
        KernelOp::Cmp {
            dst: Reg(4),
            a: Reg(0),
            b: Reg(3),
            op: CmpOp::Lt,
            ty: ScalarType::U32,
        },
        KernelOp::Branch {
            cond: Reg(4),
            then_ops: vec![KernelOp::Loop {
                count: Reg(2),
                iter_reg: Reg(5),
                body: vec![
                    KernelOp::Const {
                        dst: Reg(6),
                        value: ConstValue::F32(1.5),
                    },
                    KernelOp::BinOp {
                        dst: Reg(1),
                        a: Reg(1),
                        b: Reg(6),
                        op: BinOp::Add,
                        ty: ScalarType::F32,
                    },
                ],
            }],
            else_ops: vec![],
        },
        // Post-merge read of the carried register.
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(1),
            ty: ScalarType::F32,
        },
    ];
    KernelDef {
        name: "loop_in_branch_carried".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body,
        body_source: None,
        next_reg: 7,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn loop_carried_reg_read_after_bypassable_loop_is_valid_spirv() {
    let spirv = emit_spirv::emit(&loop_in_branch_carried_kernel()).expect("emit");
    assert_spirv_val(&spirv);
}

/// `out[i] = (u < p) as u32` — a comparison result (a `%bool`) written into a
/// register that is then stored into a `u32` buffer. This is the
/// `fill_bernoulli_u32` shape. OpStore is strictly typed: a `%bool` value into
/// a `%uint` element is invalid SPIR-V (`spirv-val`: "Expected Object type to
/// match Pointer type"). The Store arm must materialize the bool as an int
/// (OpSelect 1/0) first.
fn bool_stored_into_uint_kernel() -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        // v: entry-init 0u32 (the register is demoted; the branch/compare
        // writes it) — mirrors `let v: u32 = if u < p { 1 } else { 0 }`.
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(0),
        },
        // p threshold
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(3),
        },
        // The compare result is a %bool value stored straight into a u32
        // register, then into the u32 buffer.
        KernelOp::Cmp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(2),
            op: CmpOp::Lt,
            ty: ScalarType::U32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(3),
            ty: ScalarType::U32,
        },
    ];
    KernelDef {
        name: "bool_into_uint".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body,
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

#[test]
fn bool_stored_into_uint_buffer_is_valid_spirv() {
    let spirv = emit_spirv::emit(&bool_stored_into_uint_kernel()).expect("emit");
    assert_spirv_val(&spirv);
}

/// `out[i] = ln(x)` with an `f64` element type. The GLSL.std.450
/// transcendentals (Log/Exp/Sin/…) accept only 16/32-bit floats, and
/// f32-emulating them at f64 is silently lossy (it corrupts Box-Muller-style
/// algorithms whose ln() argument can be tiny). The SPIR-V backend therefore
/// *refuses* f64 transcendentals: the emitter returns an error and
/// `validate_for(VULKAN, …)` reports the op as unsupported.
fn f64_transcendental_kernel() -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        // x: an f64 input constant
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::F64(2.5),
        },
        // ln(x) at f64 — must be f32-emulated to stay valid.
        KernelOp::MathCall {
            dst: Reg(2),
            func: MathFn::Log,
            args: vec![Reg(1)],
            ty: ScalarType::F64,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::F64,
        },
    ];
    KernelDef {
        name: "f64_log".into(),
        params: vec![KernelParam::FieldWrite {
            name: "o".into(),
            slot: 0,
            scalar_type: ScalarType::F64,
        }],
        body,
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

#[test]
fn f64_transcendental_is_refused_by_emitter() {
    let err = emit_spirv::emit(&f64_transcendental_kernel())
        .expect_err("f64 transcendental must be refused, not emulated");
    assert!(
        err.contains("f64 transcendental"),
        "unexpected error: {err}"
    );
}

#[test]
fn f64_transcendental_is_reported_unsupported_on_vulkan() {
    use quanta_ir::{caps, validate};
    let report = validate::validate_for(&caps::VULKAN, &f64_transcendental_kernel());
    assert!(
        report
            .issues
            .iter()
            .any(|i| i.reason.contains("f64 transcendental")),
        "validate_for(VULKAN) should flag the f64 transcendental: {report:?}"
    );
}

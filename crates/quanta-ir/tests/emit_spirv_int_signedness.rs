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
    BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv,
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

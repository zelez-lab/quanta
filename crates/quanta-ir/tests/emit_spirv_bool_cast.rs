//! Regression: casting a `bool` (a comparison result) to an integer
//! must not use `OpBitcast`.
//!
//! Every comparison kernel lowers as `Cmp -> bool`, then
//! `Cast(Bool, U32) -> 0 | 1`, then stores the u32. A boolean has no
//! physical bit representation in SPIR-V, so `OpBitcast` from a bool is
//! invalid. The emitter must materialize 0/1 with `OpSelect` instead.
//!
//! When it didn't, `spirv-val` rejected the module and drivers (lavapipe)
//! failed `vkCreateComputePipelines` with `VK_ERROR_UNKNOWN` (-13) — the
//! failure that broke the Vulkan `op_matrix` lane on the very first Cmp
//! case (`op_matrix_eq_u32_...`).

#![cfg(feature = "jit")]

use quanta_ir::{CmpOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

/// `out[0] = (a[0] == b[0]) as u32` over the given operand type. Shape
/// mirrors `tests/diff/op_matrix.rs::build_cmp_def`.
fn eq_to_u32_kernel(operand_ty: ScalarType) -> KernelDef {
    KernelDef {
        name: "eq".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: operand_ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: operand_ty,
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
                ty: operand_ty,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: operand_ty,
            },
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: CmpOp::Eq,
                ty: operand_ty,
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

/// Validate the emitted module with `spirv-val` when present. This is the
/// direct check for the bug: `spirv-val` rejects `OpBitcast` from a bool.
fn assert_spirv_val(spirv: &[u8]) {
    use std::io::Write;
    use std::process::{Command, Stdio};

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
        .write_all(spirv)
        .expect("pipe spirv to spirv-val");
    let output = child.wait_with_output().expect("wait spirv-val");
    assert!(
        output.status.success(),
        "spirv-val rejected the bool-cast module:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn cmp_u32_to_u32_passes_spirv_val() {
    let spirv = emit_spirv::emit(&eq_to_u32_kernel(ScalarType::U32)).expect("emit eq u32 kernel");
    assert_spirv_val(&spirv);
}

#[test]
fn cmp_i32_to_u32_passes_spirv_val() {
    let spirv = emit_spirv::emit(&eq_to_u32_kernel(ScalarType::I32)).expect("emit eq i32 kernel");
    assert_spirv_val(&spirv);
}

#[test]
fn cmp_f32_to_u32_passes_spirv_val() {
    let spirv = emit_spirv::emit(&eq_to_u32_kernel(ScalarType::F32)).expect("emit eq f32 kernel");
    assert_spirv_val(&spirv);
}

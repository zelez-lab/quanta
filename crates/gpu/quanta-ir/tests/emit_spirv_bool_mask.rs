//! Bool→float mask materialization in the SPIR-V emitter.
//!
//! The compare kernels (`qa_compare` and friends) produce their {0, 1}
//! mask as `Cast(Cmp(a, b), Bool → ty)`. `%bool` has no bit
//! representation in SPIR-V, so the cast must lower to
//! `OpSelect one zero`, never to an OpConvert*/OpBitcast on the bool
//! value. The float lane used to fall through to `OpConvertUToF %float
//! %bool` — an invalid module that spirv-val rejects but V3D accepted,
//! materializing `true` as its native all-ones mask and reading it back
//! as 2^32 instead of 1.0 (compare/where_mask results came out scaled
//! by exactly 2^32 on the Pi while Metal and CPU passed).
//!
//! These tests pin (a) validity of the emitted module (spirv-val when
//! available) and (b) the structural shape: no OpConvertUToF /
//! OpConvertSToF anywhere in a compare-mask kernel, an OpSelect
//! present, and the typed 1.0/0.0 constants materialized.

#![cfg(feature = "jit")]

use quanta_ir::{CmpOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

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

fn words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Count instructions with the given opcode in the module.
fn count_opcode(spirv: &[u8], opcode: u32) -> usize {
    let words = words(spirv);
    let mut n = 0;
    let mut i = 5; // skip header
    while i < words.len() {
        let op = words[i] & 0xFFFF;
        let count = (words[i] >> 16) as usize;
        if count == 0 {
            break;
        }
        if op == opcode {
            n += 1;
        }
        i += count;
    }
    n
}

/// True when `value` appears as a single-word OpConstant literal.
fn has_u32_constant(spirv: &[u8], value: u32) -> bool {
    const OP_CONSTANT: u32 = 43;
    let words = words(spirv);
    let mut i = 5;
    while i < words.len() {
        let opcode = words[i] & 0xFFFF;
        let count = (words[i] >> 16) as usize;
        if count == 0 {
            break;
        }
        if opcode == OP_CONSTANT && count == 4 && words[i + 3] == value {
            return true;
        }
        i += count;
    }
    false
}

const OP_CONVERT_S_TO_F: u32 = 111;
const OP_CONVERT_U_TO_F: u32 = 112;
const OP_SELECT: u32 = 169;

/// The exact `qa_compare` kernel shape: Load, Load, Cmp, Cast(Bool→ty),
/// Store — all at the mask's own scalar type.
fn compare_kernel(ty: ScalarType) -> KernelDef {
    KernelDef {
        name: "bool_mask_probe".into(),
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
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: CmpOp::Lt,
                ty,
            },
            KernelOp::Cast {
                dst: Reg(4),
                src: Reg(3),
                from: ScalarType::Bool,
                to: ty,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(4),
                ty,
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

#[test]
fn f32_compare_mask_lowers_to_select() {
    let spirv = emit_spirv::emit(&compare_kernel(ScalarType::F32)).expect("emit");
    assert_spirv_val(&spirv);
    assert_eq!(
        count_opcode(&spirv, OP_CONVERT_U_TO_F) + count_opcode(&spirv, OP_CONVERT_S_TO_F),
        0,
        "Bool→f32 mask cast must not go through OpConvert*ToF — \
         a %bool operand there is invalid SPIR-V, and V3D reads its \
         native all-ones `true` back as 2^32"
    );
    assert!(
        count_opcode(&spirv, OP_SELECT) >= 1,
        "Bool→f32 mask cast must materialize 0.0/1.0 with OpSelect"
    );
    assert!(
        has_u32_constant(&spirv, 1.0f32.to_bits()),
        "typed 1.0f constant missing from the mask select"
    );
    assert!(
        has_u32_constant(&spirv, 0.0f32.to_bits()),
        "typed 0.0f constant missing from the mask select"
    );
}

#[test]
fn u32_compare_mask_still_lowers_to_select() {
    let spirv = emit_spirv::emit(&compare_kernel(ScalarType::U32)).expect("emit");
    assert_spirv_val(&spirv);
    assert!(
        count_opcode(&spirv, OP_SELECT) >= 1,
        "Bool→u32 mask cast must materialize 0/1 with OpSelect"
    );
}

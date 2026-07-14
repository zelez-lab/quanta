//! Folded-dispatch linearization in the SPIR-V emitter.
//!
//! Oversized 1D dispatches fold into a 2D grid (see
//! `quanta_ir::dispatch_fold`); the emitter must bake the matching
//! index linearization into `QuarkId` / `NucleusId`:
//!
//! * `QuarkId`  = `gid.x + gid.y * (FOLD_ROW_GROUPS * wg_x)`
//! * `NucleusId` = `wg_id.x + wg_id.y * FOLD_ROW_GROUPS`
//!
//! These tests pin (a) that the modules stay valid SPIR-V
//! (spirv-val when available), and (b) that the linearization
//! constants actually appear in the emitted words — i.e. the y
//! component participates in the index computation. Runtime
//! equivalence for folded grids is exercised on real Vulkan hardware
//! (V3D / lavapipe); on 1D dispatches `gid.y == 0` makes the formula
//! the identity, which keeps every existing backend's behavior.

#![cfg(feature = "jit")]

use quanta_ir::dispatch_fold::FOLD_ROW_GROUPS;
use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, emit_spirv};

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

/// True when `value` appears as an OpConstant literal in the module.
fn has_u32_constant(spirv: &[u8], value: u32) -> bool {
    const OP_CONSTANT: u32 = 43;
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut i = 5; // skip header
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

fn kernel(workgroup_size: [u32; 3]) -> KernelDef {
    KernelDef {
        name: "fold_probe".into(),
        params: vec![KernelParam::FieldWrite {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::U32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::NucleusId { dst: Reg(1) },
            KernelOp::BinOp {
                dst: Reg(2),
                a: Reg(0),
                b: Reg(1),
                op: quanta_ir::BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size,
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn quark_id_linearizes_with_wg1_row_span() {
    let spirv = emit_spirv::emit(&kernel([1, 1, 1])).expect("emit");
    assert_spirv_val(&spirv);
    // wg_x = 1: both QuarkId and NucleusId use FOLD_ROW_GROUPS.
    assert!(
        has_u32_constant(&spirv, FOLD_ROW_GROUPS),
        "row-span constant {FOLD_ROW_GROUPS} missing — QuarkId/NucleusId \
         no longer linearize the folded grid"
    );
}

#[test]
fn quark_id_linearizes_with_wg256_row_span() {
    let spirv = emit_spirv::emit(&kernel([256, 1, 1])).expect("emit");
    assert_spirv_val(&spirv);
    // QuarkId row span scales by the workgroup width…
    assert!(
        has_u32_constant(&spirv, FOLD_ROW_GROUPS * 256),
        "QuarkId row-span constant {} missing",
        FOLD_ROW_GROUPS * 256
    );
    // …while NucleusId stays at workgroup granularity.
    assert!(
        has_u32_constant(&spirv, FOLD_ROW_GROUPS),
        "NucleusId row-span constant {FOLD_ROW_GROUPS} missing"
    );
}

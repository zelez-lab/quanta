//! Integration: lower hello_quanta's vector_add WASM → KernelDef.
//!
//! The side table here is hand-rolled to match what the macro will
//! eventually emit. Once the macro starts emitting it as a custom
//! WASM section, this test will read it from the WASM and assert
//! the lowering works end-to-end.

use quanta_ir::{KernelOp, KernelParam, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

const HELLO_QUANTA_WASM: &[u8] = include_bytes!("hello_quanta.wasm");

fn vector_add_side_table() -> SideTable {
    SideTable {
        kernel_name: "vector_add".to_string(),
        params: vec![
            ParamSlot {
                wasm_index: 0,
                slot: 0,
                kind: ParamKind::BufferRead,
                scalar: ScalarType::F32,
            },
            ParamSlot {
                wasm_index: 1,
                slot: 1,
                kind: ParamKind::BufferRead,
                scalar: ScalarType::F32,
            },
            ParamSlot {
                wasm_index: 2,
                slot: 2,
                kind: ParamKind::BufferWrite,
                scalar: ScalarType::F32,
            },
        ],
        workgroup_size: [64, 1, 1],
    }
}

#[test]
fn lowers_vector_add_to_kerneldef() {
    let side_table = vector_add_side_table();
    let kernel_def = lower(HELLO_QUANTA_WASM, &side_table).expect("lower vector_add");

    assert_eq!(kernel_def.name, "vector_add");
    assert_eq!(kernel_def.params.len(), 3);

    match &kernel_def.params[0] {
        KernelParam::FieldRead {
            slot: 0,
            scalar_type: ScalarType::F32,
            ..
        } => {}
        other => panic!("expected FieldRead/F32 in slot 0, got {other:?}"),
    }
    match &kernel_def.params[2] {
        KernelParam::FieldWrite {
            slot: 2,
            scalar_type: ScalarType::F32,
            ..
        } => {}
        other => panic!("expected FieldWrite/F32 in slot 2, got {other:?}"),
    }

    let mut saw_quark_id = false;
    let mut load_slots: Vec<u32> = Vec::new();
    let mut store_slots: Vec<u32> = Vec::new();
    let mut saw_f32_add = false;
    for op in &kernel_def.body {
        match op {
            KernelOp::QuarkId { .. } => saw_quark_id = true,
            KernelOp::Load {
                field,
                ty: ScalarType::F32,
                ..
            } => load_slots.push(*field),
            KernelOp::Store {
                field,
                ty: ScalarType::F32,
                ..
            } => store_slots.push(*field),
            KernelOp::BinOp {
                op: quanta_ir::BinOp::Add,
                ty: ScalarType::F32,
                ..
            } => saw_f32_add = true,
            _ => {}
        }
    }
    assert!(saw_quark_id, "expected a QuarkId op");
    assert!(load_slots.contains(&0), "expected Load slot 0 (a)");
    assert!(load_slots.contains(&1), "expected Load slot 1 (b)");
    assert!(store_slots.contains(&2), "expected Store slot 2 (result)");
    assert!(saw_f32_add, "expected BinOp::Add F32");
}

//! Integration: lower the tiled n-body kernel (bench_nbody's
//! `nbody_soa`) WASM → KernelDef.
//!
//! Regression fixture for the address-CSE pattern: rustc CSEs the
//! `&mut vx[idx]` address into a wasm local (`local.get 4; local.get 8;
//! i32.add; local.tee 7`) and then both loads from and stores through
//! that local. The lowering must keep the address symbolic across the
//! local.set/tee instead of trying to commit it to a register.

use quanta_ir::{KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

const NBODY_WASM: &[u8] = include_bytes!("nbody_soa.wasm");

fn nbody_side_table() -> SideTable {
    let mut params: Vec<ParamSlot> = (0u32..4)
        .map(|i| ParamSlot {
            wasm_index: i,
            slot: i,
            kind: ParamKind::BufferRead,
            scalar: ScalarType::F32,
        })
        .collect();
    params.extend((4u32..7).map(|i| ParamSlot {
        wasm_index: i,
        slot: i,
        kind: ParamKind::BufferWrite,
        scalar: ScalarType::F32,
    }));
    params.push(ParamSlot {
        wasm_index: 7,
        slot: 7,
        kind: ParamKind::Scalar,
        scalar: ScalarType::U32,
    });
    SideTable {
        kernel_name: "nbody_soa".to_string(),
        params,
        workgroup_size: [512, 1, 1],
    }
}

#[test]
fn lowers_nbody_soa_to_kerneldef() {
    let side_table = nbody_side_table();
    let def = lower(NBODY_WASM, &side_table).expect("lower nbody_soa");

    assert_eq!(def.name, "nbody_soa");
    assert_eq!(def.params.len(), 8);

    // The IR must pass the structural use-before-def oracle.
    quanta_ir::scope_check::scope_check(&def).expect("scope_check");

    // The epilogue is three read-modify-write accumulations
    // `v{x,y,z}[idx] += a * dt` — each must lower to a Load and a
    // Store on the SAME write slot (4, 5, 6).
    fn count_ops(ops: &[KernelOp], slot: u32) -> (usize, usize) {
        let mut loads = 0;
        let mut stores = 0;
        for op in ops {
            match op {
                KernelOp::Load { field, .. } if *field == slot => loads += 1,
                KernelOp::Store { field, .. } if *field == slot => stores += 1,
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    let (l, s) = count_ops(then_ops, slot);
                    let (l2, s2) = count_ops(else_ops, slot);
                    loads += l + l2;
                    stores += s + s2;
                }
                KernelOp::Loop { body, .. } => {
                    let (l, s) = count_ops(body, slot);
                    loads += l;
                    stores += s;
                }
                _ => {}
            }
        }
        (loads, stores)
    }

    for slot in 4u32..7 {
        let (loads, stores) = count_ops(&def.body, slot);
        assert_eq!(
            (loads, stores),
            (1, 1),
            "slot {slot}: expected exactly one Load + one Store, got {loads} loads / {stores} stores"
        );
    }

    // The shared tiles are read through SharedLoad and written through
    // SharedStore; both must be present.
    fn any_op(ops: &[KernelOp], pred: &dyn Fn(&KernelOp) -> bool) -> bool {
        ops.iter().any(|op| {
            pred(op)
                || match op {
                    KernelOp::Branch {
                        then_ops, else_ops, ..
                    } => any_op(then_ops, pred) || any_op(else_ops, pred),
                    KernelOp::Loop { body, .. } => any_op(body, pred),
                    _ => false,
                }
        })
    }
    assert!(any_op(&def.body, &|op| matches!(
        op,
        KernelOp::SharedStore { .. }
    )));
    assert!(any_op(&def.body, &|op| matches!(
        op,
        KernelOp::SharedLoad { .. }
    )));
}

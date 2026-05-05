//! Integration: lower the cookbook_mandelbrot WASM → KernelDef.
//! Exercises control flow (block / loop / br_if) since Mandelbrot
//! has a `while` loop.

use quanta_ir::{KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

const MANDELBROT_WASM: &[u8] = include_bytes!("mandelbrot.wasm");

fn mandelbrot_side_table() -> SideTable {
    SideTable {
        kernel_name: "mandelbrot".to_string(),
        params: vec![
            ParamSlot {
                wasm_index: 0,
                slot: 0,
                kind: ParamKind::BufferWrite,
                scalar: ScalarType::U32,
            },
            ParamSlot {
                wasm_index: 1,
                slot: 1,
                kind: ParamKind::Scalar,
                scalar: ScalarType::U32,
            },
            ParamSlot {
                wasm_index: 2,
                slot: 2,
                kind: ParamKind::Scalar,
                scalar: ScalarType::U32,
            },
            ParamSlot {
                wasm_index: 3,
                slot: 3,
                kind: ParamKind::Scalar,
                scalar: ScalarType::U32,
            },
        ],
        workgroup_size: [64, 1, 1],
    }
}

#[test]
fn lowers_mandelbrot_to_kerneldef() {
    let side_table = mandelbrot_side_table();
    let kernel_def = lower(MANDELBROT_WASM, &side_table)
        .expect("mandelbrot must lower cleanly end-to-end after panic-call elision");

    assert_eq!(kernel_def.name, "mandelbrot");
    assert_eq!(kernel_def.params.len(), 4);

    let mut saw_quark_id = false;
    let mut saw_loop = false;
    let mut saw_store = false;
    for op in body_ops_recursive(&kernel_def.body) {
        if matches!(op, KernelOp::QuarkId { .. }) {
            saw_quark_id = true;
        }
        if matches!(op, KernelOp::Loop { .. }) {
            saw_loop = true;
        }
        if matches!(op, KernelOp::Store { .. }) {
            saw_store = true;
        }
    }
    assert!(saw_quark_id, "expected QuarkId at top of mandelbrot");
    assert!(saw_loop, "expected at least one Loop (the while)");
    assert!(
        saw_store,
        "expected an i32.store of the iteration count to output buffer"
    );

    // The panic-helper call rustc emits for `idx % d.width` is the
    // canary for panic-call elision. If anything in the lowered body
    // mentions a `panic_const_*` or unreachable, the elision regressed.
    let debug = format!("{:?}", kernel_def.body);
    assert!(
        !debug.contains("panic"),
        "lowered body must not retain any reference to a panic helper"
    );

    // `KernelOp::Break` only makes sense inside a `KernelOp::Loop`.
    // The earlier flat `Break` emission for `br_if` to a non-Loop
    // target tripped the Metal/SPIR-V emitters with "break statement
    // not in loop or switch context". The redirect-chain rewrite
    // turns those into structured `Branch.else_ops`, so any `Break`
    // appearing outside a Loop body is a regression.
    assert!(
        no_break_outside_loop(&kernel_def.body),
        "found KernelOp::Break outside a Loop body — br_if-to-Block rewrite regressed"
    );
}

fn body_ops_recursive(ops: &[KernelOp]) -> Vec<&KernelOp> {
    let mut out = Vec::new();
    for op in ops {
        out.push(op);
        match op {
            KernelOp::Loop { body, .. } => out.extend(body_ops_recursive(body)),
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                out.extend(body_ops_recursive(then_ops));
                out.extend(body_ops_recursive(else_ops));
            }
            _ => {}
        }
    }
    out
}

fn no_break_outside_loop(ops: &[KernelOp]) -> bool {
    for op in ops {
        match op {
            KernelOp::Break => return false,
            KernelOp::Loop { .. } => {
                // Inside a Loop body, Break is fine — skip recursion.
            }
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                if !no_break_outside_loop(then_ops) || !no_break_outside_loop(else_ops) {
                    return false;
                }
            }
            _ => {}
        }
    }
    true
}

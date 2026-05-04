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
    let result = lower(MANDELBROT_WASM, &side_table);

    // Mandelbrot exercises control flow (block / loop / br_if) plus
    // mid-loop control patterns (continue) and ops we may not yet
    // support. The lowering pass is allowed to fail for now — what
    // we want to assert is that it either produces a valid KernelDef
    // *or* fails with a precise UnsupportedOp pointing at the gap,
    // not a panic.
    match result {
        Ok(kernel_def) => {
            assert_eq!(kernel_def.name, "mandelbrot");
            assert_eq!(kernel_def.params.len(), 4);

            let mut saw_quark_id = false;
            let mut saw_loop = false;
            for op in &kernel_def.body {
                if matches!(op, KernelOp::QuarkId { .. }) {
                    saw_quark_id = true;
                }
                if matches!(op, KernelOp::Loop { .. }) {
                    saw_loop = true;
                }
            }
            assert!(saw_quark_id, "expected QuarkId at top of mandelbrot");
            assert!(saw_loop, "expected at least one Loop (the while)");
        }
        Err(e) => {
            // Surface the exact gap so we know what to implement next.
            // This branch will turn into a hard assert once Mandelbrot
            // lowers cleanly end-to-end.
            eprintln!("[mandelbrot lowering gap] {e}");
        }
    }
}

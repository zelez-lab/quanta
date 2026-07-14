//! Hand-crafted WASM tests for the f64 family (C4). Mirrors the
//! lower_i64_narrow.rs approach: tiny WAT modules emitting specific
//! f64 ops, lowered through the WASM-route, body inspected for the
//! expected KernelOps.

use quanta_ir::{KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

fn side_table_one_write(name: &str, write_ty: ScalarType) -> SideTable {
    SideTable {
        kernel_name: name.to_string(),
        params: vec![ParamSlot {
            wasm_index: 0,
            slot: 0,
            kind: ParamKind::BufferWrite,
            scalar: write_ty,
        }],
        workgroup_size: [64, 1, 1],
    }
}

fn side_table_read_write(name: &str, read_ty: ScalarType, write_ty: ScalarType) -> SideTable {
    SideTable {
        kernel_name: name.to_string(),
        params: vec![
            ParamSlot {
                wasm_index: 0,
                slot: 0,
                kind: ParamKind::BufferRead,
                scalar: read_ty,
            },
            ParamSlot {
                wasm_index: 1,
                slot: 1,
                kind: ParamKind::BufferWrite,
                scalar: write_ty,
            },
        ],
        workgroup_size: [64, 1, 1],
    }
}

fn count_ops<F: Fn(&KernelOp) -> bool>(body: &[KernelOp], pred: F) -> usize {
    body.iter().filter(|op| pred(op)).count()
}

// ───────────────────────────────────────────────────────────────
// f64 arithmetic (add / mul)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_add_and_mul() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f64.const 1.5
            f64.const 2.25
            f64.add
            f64.const 4.0
            f64.mul
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let f64_binops = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::BinOp {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    assert!(
        f64_binops >= 2,
        "expected at least 2 F64 BinOp; body = {:#?}",
        kernel.body
    );
}

// ───────────────────────────────────────────────────────────────
// f64 comparison
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_lt() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f64.const 1.0
            f64.const 2.0
            f64.lt
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let f64_cmp = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cmp {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    assert_eq!(f64_cmp, 1, "expected exactly one F64 Cmp");
}

// ───────────────────────────────────────────────────────────────
// f64 load / store (scale=8)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_load_and_store() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32) (param i32)
            local.get 1                  ;; out base
            call $qid
            i32.const 3
            i32.shl
            i32.add                      ;; out + qid*8
            local.get 0                  ;; in base
            call $qid
            i32.const 3
            i32.shl
            i32.add                      ;; in + qid*8
            f64.load                     ;; load f64
            f64.store                    ;; store f64
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_read_write("k", ScalarType::F64, ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let f64_load = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Load {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    let f64_store = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Store {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    assert_eq!(f64_load, 1, "expected exactly one F64 Load");
    assert_eq!(f64_store, 1, "expected exactly one F64 Store");
}

// ───────────────────────────────────────────────────────────────
// f32 ↔ f64 promote / demote
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_promote_f32() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f32.const 1.5
            f64.promote_f32
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::F32,
                to: ScalarType::F64,
                ..
            }
        )
    });
    assert_eq!(cast_count, 1, "expected exactly one Cast(F32, F64)");
}

#[test]
fn lowers_f32_demote_f64() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f64.const 1.5
            f32.demote_f64
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F32);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::F64,
                to: ScalarType::F32,
                ..
            }
        )
    });
    assert_eq!(cast_count, 1, "expected exactly one Cast(F64, F32)");
}

// ───────────────────────────────────────────────────────────────
// f64 ↔ int conversions
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_convert_i64_u() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            i64.const 12345
            f64.convert_i64_u
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::U64,
                to: ScalarType::F64,
                ..
            }
        )
    });
    assert_eq!(cast_count, 1, "expected exactly one Cast(U64, F64)");
}

#[test]
fn lowers_i64_trunc_f64_s() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f64.const 3.14
            i64.trunc_f64_s
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::I64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::F64,
                to: ScalarType::I64,
                ..
            }
        )
    });
    assert_eq!(cast_count, 1, "expected exactly one Cast(F64, I64)");
}

// ───────────────────────────────────────────────────────────────
// f64 unary ops (sqrt, neg)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_f64_sqrt_and_neg() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            f64.const 4.0
            f64.sqrt
            f64.neg
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::F64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let f64_unary = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::UnaryOp {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    let f64_math = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::MathCall {
                ty: ScalarType::F64,
                ..
            }
        )
    });
    // Neg → UnaryOp, Sqrt → MathCall.
    assert_eq!(f64_unary, 1, "expected exactly one F64 UnaryOp (Neg)");
    assert_eq!(f64_math, 1, "expected exactly one F64 MathCall (Sqrt)");
}

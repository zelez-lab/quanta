//! Hand-crafted WASM tests for the i64 narrow load/store family
//! (C2g + C3). rustc/LLVM aggressively constant-folds the natural
//! Rust patterns that would emit these instructions, so to verify
//! the lowering we author tiny WAT modules that emit them directly.
//!
//! Each test:
//!   1. Assembles a WAT source string into WASM bytes.
//!   2. Builds a SideTable describing the kernel's parameters.
//!   3. Calls `lower` and inspects the resulting KernelDef body.
//!
//! The kernels here look nothing like rustc output — they're
//! minimal sequences of WASM ops chosen to put one specific
//! narrow-load or narrow-store on the wire. Production kernels
//! never look like this; these tests pin the lowering shape.

use quanta_ir::{KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

/// Build a SideTable for a kernel with one buffer-write param of
/// the given scalar type, plus an optional buffer-read param.
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

/// Count KernelOps matching a predicate. Used to assert specific
/// op kinds appear (or don't appear) in the lowered body.
fn count_ops<F: Fn(&KernelOp) -> bool>(body: &[KernelOp], pred: F) -> usize {
    body.iter().filter(|op| pred(op)).count()
}

/// Helper: assert the body contains at least one Cast op with
/// the given `to` scalar type.
fn assert_cast_to(body: &[KernelOp], to: ScalarType) {
    let cast_count = count_ops(
        body,
        |op| matches!(op, KernelOp::Cast { to: t, .. } if *t == to),
    );
    assert!(
        cast_count > 0,
        "expected at least one Cast(_, {to:?}) in body; got {body:#?}"
    );
}

// ───────────────────────────────────────────────────────────────
// i64.load32_u  (already covered by C2g, but pinned here for symmetry)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_load32_u() {
    // Read u32 at base+i*4, widen to u64, drop. Buffer is u32.
    // The lowering should emit Load(U32) + Cast(U32, U64).
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 2
            i32.shl
            i32.add
            i64.load32_u
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kernel = lower(&wasm, &side_table).expect("lower");
    // Expect Load(U32) followed by Cast(U32, U64).
    assert_cast_to(&kernel.body, ScalarType::U64);
    // And a Load op with ty=U32.
    let load_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Load {
                ty: ScalarType::U32,
                ..
            }
        )
    });
    assert!(load_count >= 1, "expected at least one Load(U32) op");
}

// ───────────────────────────────────────────────────────────────
// i64.load8_u  (C3 — byte-wide load, zero-extend to u64)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_load8_u() {
    // Read u8 at base+i, widen to u64, drop. Buffer is u8.
    // Lowering should emit Load(U8) + BitAnd(0xFF) + Cast(_, U64).
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 0
            i32.shl
            i32.add
            i64.load8_u
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U8);
    let kernel = lower(&wasm, &side_table).expect("lower");
    // Expect a Cast(_, U64) and a Load op.
    assert_cast_to(&kernel.body, ScalarType::U64);
    let load_count = count_ops(&kernel.body, |op| matches!(op, KernelOp::Load { .. }));
    assert!(load_count >= 1, "expected at least one Load op");
}

// ───────────────────────────────────────────────────────────────
// i64.load8_s  (C3 — byte-wide load, sign-extend to i64)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_load8_s() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 0
            i32.shl
            i32.add
            i64.load8_s
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::I8);
    let kernel = lower(&wasm, &side_table).expect("lower");
    // Signed widen → Cast to I64.
    assert_cast_to(&kernel.body, ScalarType::I64);
}

// ───────────────────────────────────────────────────────────────
// i64.load16_u  (C3 — short-wide load, zero-extend to u64)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_load16_u() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 1
            i32.shl
            i32.add
            i64.load16_u
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U16);
    let kernel = lower(&wasm, &side_table).expect("lower");
    assert_cast_to(&kernel.body, ScalarType::U64);
}

// ───────────────────────────────────────────────────────────────
// i64.store8  (C3 — truncate u64 → u8, store)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_store8() {
    // Compute a u64 constant, store low byte to a u8 buffer at
    // base+i. Lowering should emit Cast(U64, U32) + BitAnd(0xFF)
    // + Store(U8).
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 0
            i32.shl
            i32.add                    ;; address
            i64.const 0x1234567890ABCDEF  ;; value
            i64.store8
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U8);
    let kernel = lower(&wasm, &side_table).expect("lower");
    // Expect Cast(U64, U32) and a Store op.
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::U64,
                to: ScalarType::U32,
                ..
            }
        )
    });
    assert!(
        cast_count >= 1,
        "expected at least one Cast(U64, U32) for the narrow store: {:#?}",
        kernel.body
    );
    let store_count = count_ops(&kernel.body, |op| matches!(op, KernelOp::Store { .. }));
    assert!(store_count >= 1, "expected at least one Store op");
}

// ───────────────────────────────────────────────────────────────
// i64.store16  (C3 — truncate u64 → u16, store)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_store16() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 1
            i32.shl
            i32.add
            i64.const 0x1234567890ABCDEF
            i64.store16
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U16);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::U64,
                to: ScalarType::U32,
                ..
            }
        )
    });
    assert!(
        cast_count >= 1,
        "expected Cast(U64, U32) for the narrow store"
    );
}

// ───────────────────────────────────────────────────────────────
// i64.store32  (C2g — already wired, verified here for completeness)
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_store32() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            local.get 0
            call $qid
            i32.const 2
            i32.shl
            i32.add
            i64.const 0x1234567890ABCDEF
            i64.store32
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let cast_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Cast {
                from: ScalarType::U64,
                to: ScalarType::U32,
                ..
            }
        )
    });
    assert!(
        cast_count >= 1,
        "expected Cast(U64, U32) for the narrow store"
    );
}

// ───────────────────────────────────────────────────────────────
// Full i64.load + i64.store round-trip on a u64 buffer.
// ───────────────────────────────────────────────────────────────

#[test]
fn lowers_i64_load_and_store_full() {
    // Read a u64 at base+i*8, do nothing, store it back.
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32) (param i32)
            local.get 1
            call $qid
            i32.const 3
            i32.shl
            i32.add                     ;; out address
            local.get 0
            call $qid
            i32.const 3
            i32.shl
            i32.add                     ;; in address
            i64.load                    ;; load u64
            i64.store                   ;; store u64
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_read_write("k", ScalarType::U64, ScalarType::U64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let load_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Load {
                ty: ScalarType::U64,
                ..
            }
        )
    });
    let store_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::Store {
                ty: ScalarType::U64,
                ..
            }
        )
    });
    assert_eq!(load_count, 1, "expected exactly one Load(U64)");
    assert_eq!(store_count, 1, "expected exactly one Store(U64)");
}

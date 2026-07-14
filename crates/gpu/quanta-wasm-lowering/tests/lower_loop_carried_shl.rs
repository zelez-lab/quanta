//! Witness for lowering variant #5: a loop-carried local whose
//! update is a constant shift (`d = d << 1`, LLVM's strength-reduced
//! `d *= 2`) must survive lowering.
//!
//! The I32Shl arm keeps `<reg> <const> i32.shl` symbolic
//! (`SymVal::ScaledIdx`) so the buffer-addressing recognizer can
//! consume it. Before the fix, `local.set` parked that symbolic
//! value as a rebinding without writing the local's stable
//! register — sound in straight-line code, but a loop-carried
//! local is read through its stable register on the next
//! iteration, so the update vanished: the induction variable
//! froze and the loop ran to its 10000-iteration fuel cap.
//! Found 2026-06-12 via quanta-prims' segmented-reduce kernel
//! (its 3-shared-array body defeats LLVM's unroll, so the loop
//! reached the lowering — unlike every earlier `d *= 2` kernel,
//! which unrolled away).
//!
//! The WAT mirrors LLVM's exact schedule: exit condition
//! materialized into a local EARLY, induction update between that
//! `local.set` and the backedge `br_if`.

use quanta_ir::{BinOp, KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

fn side_table(name: &str) -> SideTable {
    SideTable {
        kernel_name: name.to_string(),
        params: vec![ParamSlot {
            wasm_index: 0,
            slot: 0,
            kind: ParamKind::BufferWrite,
            scalar: ScalarType::U32,
        }],
        workgroup_size: [256, 1, 1],
    }
}

/// Collect every op in the first Loop body found at the top level.
fn loop_body(body: &[KernelOp]) -> &[KernelOp] {
    body.iter()
        .find_map(|op| match op {
            KernelOp::Loop { body, .. } => Some(body.as_slice()),
            _ => None,
        })
        .expect("kernel body must contain a Loop op")
}

#[test]
fn loop_carried_shl_update_survives() {
    let wat = r#"
        (module
          (import "quanta" "local_id" (func $lid (result i32)))
          (import "quanta" "shared_store_u32" (func $sst (param i32 i32 i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            (local $lane i32) (local $d i32) (local $cond i32)
            call $lid
            local.set $lane
            i32.const 1
            local.set $d
            loop ;; label = @1
              ;; keep d live inside the body
              i32.const 0
              local.get $lane
              local.get $d
              call $sst
              ;; exit cond materialized early (LLVM schedule)
              local.get $d
              i32.const 128
              i32.lt_u
              local.set $cond
              ;; induction update between cond set and backedge —
              ;; the variant #5 shape
              local.get $d
              i32.const 1
              i32.shl
              local.set $d
              local.get $cond
              br_if 0 (;@1;)
            end
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let kernel = lower(&wasm, &side_table("k")).expect("lower");
    let body = loop_body(&kernel.body);

    // The materialized update: a Shl BinOp inside the loop body.
    // Before the fix the symbolic ScaledIdx rebinding swallowed it
    // and the body contained zero shifts.
    let shls = body
        .iter()
        .filter(|op| matches!(op, KernelOp::BinOp { op: BinOp::Shl, .. }))
        .count();
    assert!(
        shls >= 1,
        "loop-carried `d <<= 1` update must materialize a Shl in the \
         Loop body; body = {body:#?}"
    );

    // And the backedge itself must still be the conditional Break.
    let breaks = body
        .iter()
        .filter(|op| {
            matches!(
                op,
                KernelOp::Branch { else_ops, .. }
                    if else_ops.iter().any(|o| matches!(o, KernelOp::Break))
            )
        })
        .count();
    assert_eq!(breaks, 1, "backedge must lower to one conditional Break");
}

#[test]
fn loop_carried_shl_via_tee_survives() {
    // Same shape but the update goes through local.tee (LLVM emits
    // tee when the updated value is also consumed on the spot).
    let wat = r#"
        (module
          (import "quanta" "local_id" (func $lid (result i32)))
          (import "quanta" "shared_store_u32" (func $sst (param i32 i32 i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            (local $lane i32) (local $d i32)
            call $lid
            local.set $lane
            i32.const 1
            local.set $d
            loop ;; label = @1
              i32.const 0
              local.get $lane
              local.get $d
              call $sst
              ;; tee: update d and test the NEW value directly
              local.get $d
              i32.const 1
              i32.shl
              local.tee $d
              i32.const 256
              i32.lt_u
              br_if 0 (;@1;)
            end
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let kernel = lower(&wasm, &side_table("k")).expect("lower");
    let body = loop_body(&kernel.body);
    let shls = body
        .iter()
        .filter(|op| matches!(op, KernelOp::BinOp { op: BinOp::Shl, .. }))
        .count();
    assert!(
        shls >= 1,
        "tee'd loop-carried `d <<= 1` must materialize a Shl; body = {body:#?}"
    );
}

#[test]
fn loop_internal_scratch_offset_stays_symbolic() {
    // The bench_nbody pattern: an addressing offset computed INSIDE
    // the loop, stored in a local that is NOT live-in to the loop
    // (first write is inside the body). It must stay symbolic so the
    // `BufferPtr + ScaledIdx` recognizer still fires — materializing
    // it would push a plain Reg into the add and break addressing.
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            (local $i i32) (local $off i32)
            i32.const 0
            local.set $i
            loop ;; label = @1
              ;; off = i << 2 — scratch, no binding at loop entry
              local.get $i
              i32.const 2
              i32.shl
              local.set $off
              local.get 0
              local.get $off
              i32.add
              i32.const 7
              i32.store
              ;; i += 1; continue while i < 4
              local.get $i
              i32.const 1
              i32.add
              local.tee $i
              i32.const 4
              i32.lt_u
              br_if 0 (;@1;)
            end
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let kernel = lower(&wasm, &side_table("k")).expect("lower");
    let body = loop_body(&kernel.body);
    let stores = body
        .iter()
        .filter(|op| matches!(op, KernelOp::Store { .. }))
        .count();
    assert_eq!(
        stores, 1,
        "in-loop scratch shl-offset must still feed the buffer-store \
         recognizer; body = {body:#?}"
    );
}

#[test]
fn straight_line_scaled_idx_stays_symbolic() {
    // Outside loops the symbolic path must be untouched: a shl used
    // for buffer addressing through a local must still produce a
    // plain Load/Store (the addressing recognizer consumes the
    // ScaledIdx), not a materialized shift.
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (memory 1)
          (func $k (export "k") (param i32)
            (local $off i32)
            call $qid
            i32.const 2
            i32.shl
            local.set $off
            local.get 0
            local.get $off
            i32.add
            i32.const 7
            i32.store
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let kernel = lower(&wasm, &side_table("k")).expect("lower");
    let stores = kernel
        .body
        .iter()
        .filter(|op| matches!(op, KernelOp::Store { .. }))
        .count();
    assert_eq!(
        stores, 1,
        "straight-line shl-addressing through a local must still \
         recognize the buffer store; body = {:#?}",
        kernel.body
    );
}

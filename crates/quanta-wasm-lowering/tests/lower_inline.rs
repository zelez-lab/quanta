//! Hand-crafted WASM tests for same-crate helper-function inlining (C5).
//!
//! Each test authors a tiny WAT module with a kernel that calls a
//! straight-line helper function defined in the same module. The
//! lowering pass should inline the helper, producing a flat KernelDef
//! body. Without C5 these tests would error with "call to defined
//! function … — inlining not yet supported".

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

fn count_ops<F: Fn(&KernelOp) -> bool>(body: &[KernelOp], pred: F) -> usize {
    body.iter().filter(|op| pred(op)).count()
}

// ───────────────────────────────────────────────────────────────
// Trivial helper: takes one i32, returns it + 1.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_trivial_helper() {
    // helper(x) = x + 1.
    // kernel `k` calls helper(quark_id()) and drops the result.
    // After inlining, the kernel body should contain at least one
    // BinOp::Add over u32 (the helper's `+ 1`).
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $helper (param i32) (result i32)
            local.get 0
            i32.const 1
            i32.add
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            call $helper
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kernel = lower(&wasm, &side_table).expect("lower");
    // QuarkId emit + helper's i32.const + i32.add.
    let add_count = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::BinOp {
                op: quanta_ir::BinOp::Add,
                ..
            }
        )
    });
    assert!(
        add_count >= 1,
        "expected at least one BinOp::Add from inlined helper"
    );
    let quark_count = count_ops(&kernel.body, |op| matches!(op, KernelOp::QuarkId { .. }));
    assert_eq!(quark_count, 1, "expected exactly one QuarkId in caller");
}

// ───────────────────────────────────────────────────────────────
// Helper with two params + a local — mirrors splitmix32-style use.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_two_param_helper_with_local() {
    // mix(a, b) = (a * 0x9E37_79B9) ^ b
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $mix (param i32) (param i32) (result i32)
            local.get 0
            i32.const 0x9E3779B9
            i32.mul
            local.get 1
            i32.xor
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            call $qid
            call $mix
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let mul = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::BinOp {
                op: quanta_ir::BinOp::Mul,
                ..
            }
        )
    });
    let xor = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::BinOp {
                op: quanta_ir::BinOp::BitXor,
                ..
            }
        )
    });
    assert!(mul >= 1, "expected at least one Mul from inlined helper");
    assert!(xor >= 1, "expected at least one Xor from inlined helper");
}

// ───────────────────────────────────────────────────────────────
// Helper that takes u64 and does i64 arith — exercises C5 + C2.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_i64_helper() {
    // wide_mul(x) = x * 0xDEAD_BEEF_FFFF_FFFF
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $wide_mul (param i64) (result i64)
            local.get 0
            i64.const 0xDEADBEEFFFFFFFFF
            i64.mul
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            i64.extend_i32_u
            call $wide_mul
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U64);
    let kernel = lower(&wasm, &side_table).expect("lower");
    let i64_mul = count_ops(&kernel.body, |op| {
        matches!(
            op,
            KernelOp::BinOp {
                op: quanta_ir::BinOp::Mul,
                ty: ScalarType::I64,
                ..
            }
        )
    });
    assert_eq!(
        i64_mul, 1,
        "expected exactly one I64 Mul from inlined helper"
    );
}

// ───────────────────────────────────────────────────────────────
// V2 inliner: helper containing structured control flow (a Loop).
// The wrap absorbs the helper's frames so the Loop body composes
// cleanly inside the caller.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_helper_with_loop() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $loopy (param i32) (result i32)
            local.get 0
            loop (result i32)
              local.get 0
              i32.const 1
              i32.add
            end
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            call $loopy
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kdef = lower(&wasm, &side_table).expect("v2 inliner should accept loop");
    let loops = count_ops(&kdef.body, |op| matches!(op, KernelOp::Loop { .. }));
    assert_eq!(loops, 1, "expected exactly one Loop from inlined helper");
}

// ───────────────────────────────────────────────────────────────
// V2 inliner: helper with an `if` block. Mirrors the rustc-emitted
// shape of a divisor-check or panic-guard branch.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_helper_with_if() {
    // helper(x) = if x != 0 { x + 1 } else { 0 }
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $branchy (param i32) (result i32)
            local.get 0
            i32.const 0
            i32.ne
            if (result i32)
              local.get 0
              i32.const 1
              i32.add
            else
              i32.const 0
            end
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            call $branchy
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    let kdef = lower(&wasm, &side_table).expect("v2 inliner should accept if/else");
    let branches = count_ops(&kdef.body, |op| matches!(op, KernelOp::Branch { .. }));
    assert_eq!(branches, 1, "expected exactly one Branch from inlined if");
}

// ───────────────────────────────────────────────────────────────
// V2 inliner: helper with a mid-body `return`. Tail-return falls
// through to the wrap's End; the wrap absorbs it.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_helper_with_return() {
    // helper(x):
    //   block @1
    //     local.get 0
    //     return            ;; inside_block + no loop crossing → no-op
    //   end
    //   unreachable         ;; already a no-op
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $earlyret (param i32) (result i32)
            block (result i32)
              local.get 0
              return
            end
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            call $earlyret
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    lower(&wasm, &side_table).expect("v2 inliner should accept early return");
}

// ───────────────────────────────────────────────────────────────
// V2 inliner: helper that ends in a div-by-zero-style panic guard.
// Real-world rustc shape:
//   block @1                  ; "real-work" wrapper
//     <work, leaving result on stack>
//     local.get arg
//     i32.eqz                 ; rustc emits this before the panic call
//     br_if @1                ; skip to real-work end if non-zero arg
//   end
//   call $panic               ; panic helper, elided by name
//   unreachable               ; no-op
// The wrap's End absorbs the leftover frame. Panic call is elided.
// ───────────────────────────────────────────────────────────────

#[test]
fn inlines_helper_with_panic_guard() {
    let wat = r#"
        (module
          (import "quanta" "quark_id" (func $qid (result i32)))
          (func $_ZN4core9panicking5panic17h0000000000000000E)
          (func $div_checked (param i32) (param i32) (result i32)
            block (result i32)
              local.get 0
              local.get 1
              i32.div_u
              local.get 1
              br_if 0
              call $_ZN4core9panicking5panic17h0000000000000000E
              unreachable
            end
          )
          (memory 1)
          (func $k (export "k") (param i32)
            call $qid
            i32.const 7
            call $div_checked
            drop
          )
        )
    "#;
    let wasm = wat::parse_str(wat).expect("wat parse");
    let side_table = side_table_one_write("k", ScalarType::U32);
    lower(&wasm, &side_table).expect("v2 inliner should elide panic helper");
}

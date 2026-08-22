//! Witness for rustc's `while` shape — the exit-flag route.
//!
//! Every rustc `while cond { body }` lowers to
//!
//! ```wat
//! block $exit
//!   loop $top
//!     ...cond...
//!     br_if $exit        ;; exit: crosses the loop to the block
//!     ...body...
//!     br $top            ;; continue
//!   end
//! end
//! ```
//!
//! The `br_if` targets a Block with exactly one Loop between, which
//! `emit_loop_crossing_exit` lowers as: a flag declared `false` at the
//! block (ahead of the loop op), `Branch { cond, then: [flag := true,
//! Break], else: [] }` at the site, and the block's tail after the loop
//! wrapped in `Branch { flag, then: [], else: tail }` at the block's
//! end. This test pins that op shape — it is the shape the Lean model
//! `lowerInstrsP` (specs/verify/lean/Quanta/Wasm/TranslatePending.lean)
//! reproduces, op for op, in its `while_exit_flag_*` pins.

use quanta_ir::{ConstValue, KernelOp, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

/// `i = 0; while i < n { i += 1 }; out[gid] = i`.
const COUNT_WAT: &str = r#"
(module
  (import "quanta" "quark_id" (func $qid (result i32)))
  (memory 1)
  (func $count (export "count") (param i32 i32) ;; out, n
    (local i32)                                 ;; 2 = i
    i32.const 0
    local.set 2
    block ;; @1 exit
      loop ;; @2 top
        local.get 2
        local.get 1
        i32.ge_u
        br_if 1 (;@1;)
        local.get 2
        i32.const 1
        i32.add
        local.set 2
        br 0 (;@2;)
      end
    end
    call $qid
    i32.const 2
    i32.shl
    local.get 0
    i32.add
    local.get 2
    i32.store
  )
)
"#;

fn side_table_for(name: &str) -> SideTable {
    SideTable {
        kernel_name: name.to_string(),
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
        ],
        workgroup_size: [64, 1, 1],
    }
}

#[test]
fn while_exit_flag_shape() {
    let wasm = wat::parse_str(COUNT_WAT).expect("wat parse");
    let def = lower(&wasm, &side_table_for("count")).expect("lower count");
    quanta_ir::scope_check::scope_check(&def).expect("scope_check");
    eprintln!("{:#?}", def.body);

    // Exactly one Loop at the top level; the op right before it is the
    // flag's declaration, the op right after it is the no-op wrap of
    // the (empty) block tail, on the same flag.
    let loop_pos = def
        .body
        .iter()
        .position(|op| matches!(op, KernelOp::Loop { .. }))
        .expect("a Loop op");
    let flag = match &def.body[loop_pos - 1] {
        KernelOp::Const {
            dst,
            value: ConstValue::Bool(false),
        } => *dst,
        other => panic!("expected the flag declaration before the loop, got {other:?}"),
    };
    match &def.body[loop_pos + 1] {
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            assert_eq!(*cond, flag, "the block-tail wrap reads the flag");
            assert!(
                then_ops.is_empty() && else_ops.is_empty(),
                "empty tail wraps as a no-op"
            );
        }
        other => panic!("expected the block-tail wrap after the loop, got {other:?}"),
    }
    // Inside the loop: the exit site sets the flag and breaks, on the
    // condition, and nothing is nested around the body that follows.
    let KernelOp::Loop { body, .. } = &def.body[loop_pos] else {
        unreachable!()
    };
    let site = body
        .iter()
        .position(|op| {
            matches!(op, KernelOp::Branch { then_ops, else_ops, .. }
                if else_ops.is_empty()
                && matches!(then_ops.as_slice(),
                    [KernelOp::Const { dst, value: ConstValue::Bool(true) }, KernelOp::Break]
                    if *dst == flag))
        })
        .expect("the exit site: Branch { cond, [flag := true, Break], [] }");
    assert!(
        body[site + 1..]
            .iter()
            .any(|op| matches!(op, KernelOp::BinOp { .. })),
        "the loop body (i += 1) follows the exit site sequentially"
    );
    assert!(
        !body[site + 1..]
            .iter()
            .any(|op| matches!(op, KernelOp::Break)),
        "the continue `br 0` emits no op"
    );
}

/// Two exit sites in one loop, and a tail inside the block after the
/// loop: `block { loop { br_if 1; br_if 1; br 0 } ; tail }`.
const TWO_SITES_WAT: &str = r#"
(module
  (import "quanta" "quark_id" (func $qid (result i32)))
  (memory 1)
  (func $two (export "two") (param i32 i32) ;; out, n
    (local i32)
    block ;; @1
      loop ;; @2
        local.get 1
        br_if 1 (;@1;)
        local.get 1
        i32.const 1
        i32.and
        br_if 1 (;@1;)
        br 0 (;@2;)
      end
      i32.const 7
      local.set 2
    end
    call $qid
    i32.const 2
    i32.shl
    local.get 0
    i32.add
    local.get 2
    i32.store
  )
)
"#;

#[test]
fn two_exit_sites_wrap_earlier_outermost() {
    let wasm = wat::parse_str(TWO_SITES_WAT).expect("wat parse");
    let def = lower(&wasm, &side_table_for("two")).expect("lower two");
    quanta_ir::scope_check::scope_check(&def).expect("scope_check");
    eprintln!("{:#?}", def.body);
    let loop_pos = def
        .body
        .iter()
        .position(|op| matches!(op, KernelOp::Loop { .. }))
        .expect("a Loop op");
    // Two declarations ahead of the loop, in site order.
    let flags: Vec<_> = def.body[..loop_pos]
        .iter()
        .filter_map(|op| match op {
            KernelOp::Const {
                dst,
                value: ConstValue::Bool(false),
            } => Some(*dst),
            _ => None,
        })
        .collect();
    assert_eq!(flags.len(), 2, "one flag per exit site");
    // The block tail wraps in the EARLIER site's flag outermost.
    match &def.body[loop_pos + 1] {
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            assert_eq!(*cond, flags[0], "earlier site outermost");
            assert!(then_ops.is_empty());
            match else_ops.as_slice() {
                [
                    KernelOp::Branch {
                        cond: inner,
                        else_ops: tail,
                        ..
                    },
                ] => {
                    assert_eq!(*inner, flags[1], "later site inside");
                    assert!(
                        !tail.is_empty(),
                        "the block tail (local.set 2) is the innermost else"
                    );
                }
                other => panic!("expected the nested wrap, got {other:?}"),
            }
        }
        other => panic!("expected the block-tail wrap after the loop, got {other:?}"),
    }
}

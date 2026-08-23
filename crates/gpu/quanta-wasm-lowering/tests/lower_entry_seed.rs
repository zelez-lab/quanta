//! Witness for the function-entry seeding of declared locals.
//!
//! Production pre-allocates every value-typed declared local's stable
//! register at function entry with a default-zero `Const`, in
//! declaration order, right after the scalar-param loads. This is the
//! discipline the Lean model's `seedLocals`
//! (specs/verify/lean/Quanta/Wasm/Translate.lean) mirrors and the
//! seeded while apex (`framework_preservation_kernel_while2_seeded`)
//! assumes as `LocalsSeeded`: a kernel never FIRST-binds a local
//! mid-stream, so the stable layer is list-invariant through loop
//! bodies. The pin asserts the entry stream op for op.

use quanta_ir::{ConstValue, KernelOp, Reg, ScalarType};
use quanta_wasm_lowering::{ParamKind, ParamSlot, SideTable, lower};

/// `i = 0; acc = 0; while i < n { acc += i; i += 1 }; out[gid] = acc` —
/// the two-local witness of the seeded apex.
const SUM_WAT: &str = r#"
(module
  (import "quanta" "quark_id" (func $qid (result i32)))
  (memory 1)
  (func $sum (export "sum") (param i32 i32) ;; out, n
    (local i32 i32)                         ;; 2 = i, 3 = acc
    i32.const 0
    local.set 2
    i32.const 0
    local.set 3
    block ;; @1 exit
      loop ;; @2 top
        local.get 2
        local.get 1
        i32.ge_u
        br_if 1 (;@1;)
        local.get 3
        local.get 2
        i32.add
        local.set 3
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
    local.get 3
    i32.store
  )
)
"#;

#[test]
fn entry_seeds_declared_locals() {
    let wasm = wat::parse_str(SUM_WAT).expect("wat parse");
    let side_table = SideTable {
        kernel_name: "sum".to_string(),
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
    };
    let def = lower(&wasm, &side_table).expect("lower sum");
    quanta_ir::scope_check::scope_check(&def).expect("scope_check");

    // The stream head, op for op:
    //   [hoisted zero-decls]* Load(param) Const(seed local 2) Const(seed local 3)
    // The leading zero `Const`s are the frame-0 declarations of the
    // per-write fresh registers (`write_local_via_copy` hoists them to
    // the function head so the emitters get `uint rN = 0u;`
    // declarations; the Lean model places the same consts INLINE at
    // the write sites — placement differs, abstract effect does not,
    // because each is overwritten before its first read). The param
    // load and the declared-local seeds follow, registers in
    // allocation order.
    let load_pos = def
        .body
        .iter()
        .position(|op| matches!(op, KernelOp::Load { .. }))
        .expect("a param load");
    for op in &def.body[..load_pos] {
        assert!(
            matches!(
                op,
                KernelOp::Const {
                    value: ConstValue::U32(0) | ConstValue::I32(0),
                    ..
                }
            ),
            "only hoisted zero-declarations may precede the param load, got {op:?}"
        );
    }
    match &def.body[load_pos] {
        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            assert_eq!(*dst, Reg(0), "the scalar param takes the first register");
            assert_eq!(*field, 1, "n lives in constant slot 1");
            assert_eq!(*index, Reg(u32::MAX), "push-constant sentinel");
            assert_eq!(*ty, ScalarType::U32);
        }
        _ => unreachable!(),
    }
    // The two declared-local seeds: consecutive registers after the
    // param's, unsigned zero (`scalar_type_for_wasm_ty(i32) = U32`,
    // the model's `zeroConst` default), declaration order.
    for k in 0..2u32 {
        match &def.body[load_pos + 1 + k as usize] {
            KernelOp::Const {
                dst,
                value: ConstValue::U32(0),
            } => {
                assert_eq!(
                    *dst,
                    Reg(1 + k),
                    "seed registers allocate in declaration order"
                );
            }
            other => panic!("expected zero seed for declared local {k}, got {other:?}"),
        }
    }
    // And the kernel still runs the while shape: one top-level Loop.
    assert_eq!(
        def.body
            .iter()
            .filter(|op| matches!(op, KernelOp::Loop { .. }))
            .count(),
        1
    );
}

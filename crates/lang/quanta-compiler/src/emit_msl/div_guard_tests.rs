//! Integer division/remainder-by-zero guard on the AOT MSL emitter.
//!
//! Mirrors `quanta-ir/tests/emit_msl_div_guard.rs` for the JIT emitter
//! — the twins share one semantics contract. The CPU reference defines
//! x/0 = 0 and x%0 = 0 for every int width; Metal's hardware returns
//! ~0 for u32 x/0, so the emitter must guard. The DIVISOR is
//! substituted before the divide (SIMD may evaluate both ternary
//! sides), then the result selected:
//!
//!   r = (b == 0) ? 0 : (a op ((b == 0) ? 1 : b))
//!
//! with operands cast to the op's `ty` like the Shr/Shl arm
//! (signedness robustness). Floats keep bare IEEE `/`.

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

/// `out[i] = a[i] op b[i]` over `ty` — a=r1, b=r2, result r3.
fn binop_kernel(ty: ScalarType, op: BinOp, name: &str) -> KernelDef {
    let field = |name: &str, slot: u32, write: bool| {
        if write {
            KernelParam::FieldWrite {
                name: name.into(),
                slot,
                scalar_type: ty,
            }
        } else {
            KernelParam::FieldRead {
                name: name.into(),
                slot,
                scalar_type: ty,
            }
        }
    };
    KernelDef {
        name: name.into(),
        params: vec![
            field("a", 0, false),
            field("b", 1, false),
            field("out", 2, true),
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op,
                ty,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

#[test]
fn int_div_and_rem_emit_the_divisor_substituting_ternary() {
    for (ty, op, o, t, tag) in [
        (ScalarType::U32, BinOp::Div, "/", "uint", "u32_div"),
        (ScalarType::U32, BinOp::Rem, "%", "uint", "u32_rem"),
        (ScalarType::I32, BinOp::Div, "/", "int", "i32_div"),
        (ScalarType::I32, BinOp::Rem, "%", "int", "i32_rem"),
        (ScalarType::U8, BinOp::Div, "/", "uint8_t", "u8_div"),
        (ScalarType::I64, BinOp::Rem, "%", "long", "i64_rem"),
    ] {
        let msl = crate::emit_msl::emit(&binop_kernel(ty, op, tag)).unwrap();
        let want = format!(
            "{t} r3 = (({t})r2 == ({t})0) ? ({t})0 : \
             (({t})r1 {o} ((({t})r2 == ({t})0) ? ({t})1 : ({t})r2));"
        );
        assert!(
            msl.contains(&want),
            "{tag}: guarded ternary missing.\nwant: {want}\ngot:\n{msl}"
        );
    }
}

#[test]
fn float_div_stays_bare_ieee() {
    // Floats keep the raw `/` — inf/NaN is their contract; no guard.
    let msl = crate::emit_msl::emit(&binop_kernel(ScalarType::F32, BinOp::Div, "f32_div")).unwrap();
    assert!(
        msl.contains("float r3 = r1 / r2;"),
        "f32_div: expected the bare IEEE divide, got:\n{msl}"
    );
    assert!(
        !msl.contains('?'),
        "f32_div: float division must not grow a ternary guard, got:\n{msl}"
    );
}

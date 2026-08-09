//! Integer division/remainder-by-zero guard on the SPIR-V emitter.
//!
//! The CPU reference defines x/0 = 0 and x%0 = 0 for every int width
//! (`eval_binop`: checked_div/checked_rem style). SPIR-V UDiv/SDiv/
//! UMod/SRem with a zero divisor is undefined BEHAVIOR — and SIMD
//! hardware may evaluate both sides of a select — so the emitter must
//! substitute the DIVISOR before dividing, then select the result:
//!
//!   is_zero = OpIEqual(b, 0)          — canonical-type bit compare
//!   safe_b  = OpSelect(is_zero, 1, b)
//!   q       = OpUDiv/OpSDiv/OpUMod/OpSRem(a, safe_b)
//!   result  = OpSelect(is_zero, 0, q)
//!
//! These tests pin that chain by decoding the word stream for u32, i32,
//! u8 and i64 (all four opcodes), check floats keep a bare FDiv, and
//! run `spirv-val --target-env vulkan1.3` on every module when the
//! validator is on PATH.

#![cfg(feature = "jit")]

use quanta_ir::emit_spirv;
use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

const OP_CONSTANT: u32 = 43;
const OP_U_CONVERT: u32 = 113;
const OP_BITCAST: u32 = 124;
const OP_UDIV: u32 = 134;
const OP_SDIV: u32 = 135;
const OP_UMOD: u32 = 137;
const OP_SREM: u32 = 138;
const OP_SELECT: u32 = 169;
const OP_IEQUAL: u32 = 170;
const OP_FDIV: u32 = 136;

// ── Kernel builder ──────────────────────────────────────────────────────

/// `out[i] = a[i] op b[i]` over `ty` — the op-matrix bind layout
/// (slot 0 = a, slot 1 = b, slot 2 = out).
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

// ── Word-stream decoding ────────────────────────────────────────────────

fn words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn instructions(w: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5; // skip header
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        assert!(wc > 0, "zero-length instruction at word {i}");
        out.push((op, w[i + 1..i + wc].to_vec()));
        i += wc;
    }
    out
}

/// The value of `id` if it is a 32- or 64-bit OpConstant.
fn constant_value(insts: &[(u32, Vec<u32>)], id: u32) -> Option<u64> {
    insts.iter().find_map(|(op, a)| {
        if *op != OP_CONSTANT || a.get(1) != Some(&id) {
            return None;
        }
        match a.len() {
            3 => Some(a[2] as u64),
            4 => Some(a[2] as u64 | ((a[3] as u64) << 32)),
            _ => None,
        }
    })
}

/// Whether `id` is `target`, or reaches it through at most one type
/// coercion (OpBitcast for the signed detour, OpUConvert for a width
/// change) — the emitter's canonical↔op-type seam.
fn resolves_to(insts: &[(u32, Vec<u32>)], id: u32, target: u32) -> bool {
    if id == target {
        return true;
    }
    insts.iter().any(|(op, a)| {
        matches!(*op, OP_BITCAST | OP_U_CONVERT)
            && a.get(1) == Some(&id)
            && a.get(2) == Some(&target)
    })
}

/// Assert the full guard chain for one module and its expected divide
/// opcode; returns nothing but panics with `name` context on any miss.
fn assert_guard_chain(name: &str, spirv: &[u8], div_opcode: u32) {
    let w = words(spirv);
    let insts = instructions(&w);

    // 1. Exactly one OpIEqual — the divisor-against-zero compare.
    let eqs: Vec<&Vec<u32>> = insts
        .iter()
        .filter(|(op, _)| *op == OP_IEQUAL)
        .map(|(_, a)| a)
        .collect();
    assert_eq!(eqs.len(), 1, "{name}: expected exactly one OpIEqual");
    let is_zero = eqs[0][1];
    let b_canon = eqs[0][2];
    let zero_id = eqs[0][3];
    assert_eq!(
        constant_value(&insts, zero_id),
        Some(0),
        "{name}: OpIEqual must compare the divisor against a typed 0"
    );

    // 2. Exactly one divide, of the expected opcode — and none of the
    // other three (signedness/op mix-ups).
    for opcode in [OP_UDIV, OP_SDIV, OP_UMOD, OP_SREM] {
        let count = insts.iter().filter(|(op, _)| *op == opcode).count();
        let want = usize::from(opcode == div_opcode);
        assert_eq!(
            count, want,
            "{name}: expected {want} instruction(s) of opcode {opcode}"
        );
    }
    let (_, div_args) = insts.iter().find(|(op, _)| *op == div_opcode).unwrap();
    let div_result = div_args[1];
    let div_divisor = div_args[3];

    // 3. Two OpSelects on the compare: safe_b feeds the divide, the
    // result select swallows the quotient.
    let selects: Vec<&Vec<u32>> = insts
        .iter()
        .filter(|(op, a)| *op == OP_SELECT && a[2] == is_zero)
        .map(|(_, a)| a)
        .collect();
    assert_eq!(
        selects.len(),
        2,
        "{name}: expected exactly two OpSelect on the is-zero compare"
    );
    let safe_b = selects
        .iter()
        .find(|a| resolves_to(&insts, div_divisor, a[1]))
        .unwrap_or_else(|| panic!("{name}: the divide's divisor must come from an OpSelect"));
    assert_eq!(
        constant_value(&insts, safe_b[3]),
        Some(1),
        "{name}: safe_b must substitute a typed 1 for the zero divisor"
    );
    assert_eq!(
        safe_b[4], b_canon,
        "{name}: safe_b's false arm must be the compared divisor"
    );
    let result_sel = selects
        .iter()
        .find(|a| a[1] != safe_b[1])
        .expect("second select");
    assert_eq!(
        constant_value(&insts, result_sel[3]),
        Some(0),
        "{name}: the result select must yield a typed 0 on b == 0"
    );
    assert!(
        resolves_to(&insts, result_sel[4], div_result),
        "{name}: the result select's false arm must be the quotient"
    );

    spirv_val(name, spirv);
}

/// Run `spirv-val --target-env vulkan1.3` when available; skip silently
/// when it isn't installed. An unarmed validator (no --target-env)
/// misses Vulkan-specific rules — the SeqCst lesson.
fn spirv_val(name: &str, spirv: &[u8]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    let mut child = match child {
        Ok(c) => c,
        Err(_) => return,
    };
    child.stdin.as_mut().unwrap().write_all(spirv).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{name}: spirv-val rejected the module:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ── Tests ───────────────────────────────────────────────────────────────

#[test]
fn div_and_rem_carry_the_zero_divisor_guard() {
    for (ty, op, opcode, tag) in [
        (ScalarType::U32, BinOp::Div, OP_UDIV, "u32_div"),
        (ScalarType::U32, BinOp::Rem, OP_UMOD, "u32_rem"),
        (ScalarType::I32, BinOp::Div, OP_SDIV, "i32_div"),
        (ScalarType::I32, BinOp::Rem, OP_SREM, "i32_rem"),
        (ScalarType::U8, BinOp::Div, OP_UDIV, "u8_div"),
        (ScalarType::U8, BinOp::Rem, OP_UMOD, "u8_rem"),
        (ScalarType::I64, BinOp::Div, OP_SDIV, "i64_div"),
        (ScalarType::I64, BinOp::Rem, OP_SREM, "i64_rem"),
    ] {
        let spirv = emit_spirv::emit(&binop_kernel(ty, op, tag)).unwrap();
        assert_guard_chain(tag, &spirv, opcode);
    }
}

#[test]
fn float_div_stays_bare_ieee() {
    // Floats keep the raw FDiv — inf/NaN is their contract; no guard.
    let spirv = emit_spirv::emit(&binop_kernel(ScalarType::F32, BinOp::Div, "f32_div")).unwrap();
    let w = words(&spirv);
    let insts = instructions(&w);
    assert!(
        insts.iter().any(|(op, _)| *op == OP_FDIV),
        "f32_div: expected OpFDiv"
    );
    assert!(
        !insts.iter().any(|(op, _)| *op == OP_IEQUAL),
        "f32_div: float division must not grow a zero-divisor guard"
    );
    spirv_val("f32_div", &spirv);
}

#[test]
fn non_div_int_ops_grow_no_guard() {
    // The guard is Div/Rem-only: an int Add module carries no compare
    // and no select.
    let spirv = emit_spirv::emit(&binop_kernel(ScalarType::U32, BinOp::Add, "u32_add")).unwrap();
    let w = words(&spirv);
    let insts = instructions(&w);
    assert!(
        !insts
            .iter()
            .any(|(op, _)| *op == OP_IEQUAL || *op == OP_SELECT),
        "u32_add: non-division ops must not grow the guard"
    );
    spirv_val("u32_add", &spirv);
}

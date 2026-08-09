//! Narrow-int (u8 / i8 / u16 / i16) storage contract on the SPIR-V
//! emitter.
//!
//! Narrow-int buffers store native-stride 8-/16-bit elements
//! (ArrayStride 1/2) through the same StorageBuffer8/16BitAccess seam
//! as fp8/bf16, while registers stay the canonical unified u32. The
//! module must declare the matching storage-access capability (and
//! `SPV_KHR_8bit_storage` for the 8-bit types), widen every load into
//! the u32 register (OpUConvert; signed types then sign-extend via
//! shl + arithmetic-shr — the CPU interpreter's widen-on-load
//! semantics), and truncate every store back to the storage element
//! (OpUConvert, ≡ mod 2^w — the wrapping contract). These tests pin
//! the instruction sequences by decoding the word stream, and run the
//! reference validator armed with `--target-env` when `spirv-val` is
//! on PATH.

#![cfg(feature = "jit")]

use quanta_ir::emit_spirv;
use quanta_ir::{BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

const CAP_STORAGE_BUFFER_16BIT: u32 = 4433;
const CAP_STORAGE_BUFFER_8BIT: u32 = 4448;
const OP_EXTENSION: u32 = 10;
const OP_CAPABILITY: u32 = 17;
const OP_TYPE_INT: u32 = 21;
const OP_CONSTANT: u32 = 43;
const OP_LOAD: u32 = 61;
const OP_STORE: u32 = 62;
const OP_DECORATE: u32 = 71;
const OP_U_CONVERT: u32 = 113;
const OP_SELECT: u32 = 169;
const OP_UGREATER_THAN: u32 = 172;
const OP_ULESS_THAN: u32 = 176;
const OP_SLESS_THAN: u32 = 177;
const OP_SHIFT_RIGHT_ARITHMETIC: u32 = 195;
const OP_SHIFT_LEFT_LOGICAL: u32 = 196;
const OP_BITWISE_AND: u32 = 199;
const DECORATION_ARRAY_STRIDE: u32 = 6;

/// `(bit width, signedness, stride, name)` rows for the four narrow
/// int types.
const NARROW_INTS: [(ScalarType, u32, bool, u32, &str); 4] = [
    (ScalarType::U8, 8, false, 1, "u8"),
    (ScalarType::I8, 8, true, 1, "i8"),
    (ScalarType::U16, 16, false, 2, "u16"),
    (ScalarType::I16, 16, true, 2, "i16"),
];

// ── Kernel builders ─────────────────────────────────────────────────────

fn def(name: &str, params: Vec<KernelParam>, body: Vec<KernelOp>, next_reg: u32) -> KernelDef {
    KernelDef {
        name: name.into(),
        params,
        body,
        body_source: None,
        next_reg,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

fn field_read(name: &str, slot: u32, ty: ScalarType) -> KernelParam {
    KernelParam::FieldRead {
        name: name.into(),
        slot,
        scalar_type: ty,
    }
}

fn field_write(name: &str, slot: u32, ty: ScalarType) -> KernelParam {
    KernelParam::FieldWrite {
        name: name.into(),
        slot,
        scalar_type: ty,
    }
}

/// `out[i] = a[i] + 1` over `ty` storage — one narrow load (the widen
/// seam) and one narrow store (the truncate seam) around a canonical
/// u32 add.
fn roundtrip_kernel(ty: ScalarType, name: &str) -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty,
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(1),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::Add,
            ty,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(3),
            ty,
        },
    ];
    def(
        name,
        vec![field_read("a", 0, ty), field_write("out", 1, ty)],
        body,
        4,
    )
}

/// `mask[i] = (a[i] < 0) as u32` over `ty` storage — the compare must
/// see the widen result (sign-extended for i8/i16), so a stored -1
/// orders below 0, not as +255/+65535.
fn cmp_kernel(ty: ScalarType, name: &str) -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty,
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::I32(0),
        },
        KernelOp::Cmp {
            dst: Reg(3),
            a: Reg(1),
            b: Reg(2),
            op: CmpOp::Lt,
            ty,
        },
        KernelOp::Cast {
            dst: Reg(4),
            src: Reg(3),
            from: ScalarType::Bool,
            to: ScalarType::U32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(4),
            ty: ScalarType::U32,
        },
    ];
    def(
        name,
        vec![
            field_read("a", 0, ty),
            field_write("mask", 1, ScalarType::U32),
        ],
        body,
        5,
    )
}

/// `out[i] = (to)(-1i32)` — the int→narrow-int cast must REDUCE in
/// register (mask-then-extend, the CPU `eval_cast` contract), then the
/// store truncates to the element. The whole negative-value round-trip
/// is pinned as an instruction chain in the test below.
fn cast_neg1_kernel(to: ScalarType, name: &str) -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::I32(-1),
        },
        KernelOp::Cast {
            dst: Reg(2),
            src: Reg(1),
            from: ScalarType::I32,
            to,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(0),
            src: Reg(2),
            ty: to,
        },
    ];
    def(name, vec![field_write("out", 0, to)], body, 3)
}

/// A mixed-width kernel: narrow u8 + i16 reads combined into a u32
/// write next to an untouched f32 field — one module carrying strides
/// 1, 2 and 4 side by side.
fn mixed_kernel(name: &str) -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty: ScalarType::U8,
        },
        KernelOp::Load {
            dst: Reg(2),
            field: 1,
            index: Reg(0),
            ty: ScalarType::I16,
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::Add,
            ty: ScalarType::U32,
        },
        KernelOp::Load {
            dst: Reg(4),
            field: 2,
            index: Reg(0),
            ty: ScalarType::F32,
        },
        KernelOp::Cast {
            dst: Reg(5),
            src: Reg(4),
            from: ScalarType::F32,
            to: ScalarType::U32,
        },
        KernelOp::BinOp {
            dst: Reg(6),
            a: Reg(3),
            b: Reg(5),
            op: BinOp::Add,
            ty: ScalarType::U32,
        },
        KernelOp::Store {
            field: 3,
            index: Reg(0),
            src: Reg(6),
            ty: ScalarType::U32,
        },
    ];
    def(
        name,
        vec![
            field_read("bytes", 0, ScalarType::U8),
            field_read("shorts", 1, ScalarType::I16),
            field_read("floats", 2, ScalarType::F32),
            field_write("out", 3, ScalarType::U32),
        ],
        body,
        7,
    )
}

/// The quantized-GEMM shape this seam exists for: a loop-carried f32
/// accumulator over narrow loads cast to f32 (loop phi + widen chain
/// inside the loop body).
fn accum_kernel(in_ty: ScalarType, name: &str) -> KernelDef {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(4),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::F32(0.0),
        },
        KernelOp::Loop {
            count: Reg(1),
            iter_reg: Reg(3),
            body: vec![
                KernelOp::Load {
                    dst: Reg(4),
                    field: 0,
                    index: Reg(3),
                    ty: in_ty,
                },
                KernelOp::Cast {
                    dst: Reg(5),
                    src: Reg(4),
                    from: in_ty,
                    to: ScalarType::F32,
                },
                KernelOp::BinOp {
                    dst: Reg(2),
                    a: Reg(2),
                    b: Reg(5),
                    op: BinOp::Add,
                    ty: ScalarType::F32,
                },
            ],
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(2),
            ty: ScalarType::F32,
        },
    ];
    def(
        name,
        vec![
            field_read("a", 0, in_ty),
            field_write("out", 1, ScalarType::F32),
        ],
        body,
        6,
    )
}

/// `out[i] = a[i] <op> b[i]` over `ty` storage — the width-dependent
/// BinOp seams (saturating clamp, rotate masking).
fn binop_kernel(op: BinOp, ty: ScalarType, name: &str) -> KernelDef {
    let body = vec![
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
    ];
    def(
        name,
        vec![
            field_read("a", 0, ty),
            field_read("b", 1, ty),
            field_write("out", 2, ty),
        ],
        body,
        4,
    )
}

// ── Word-stream decoding ────────────────────────────────────────────────

fn words(spirv: &[u8]) -> Vec<u32> {
    spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// All `(opcode, operands)` instructions of the module.
fn instructions(w: &[u32]) -> Vec<(u32, Vec<u32>)> {
    let mut out = Vec::new();
    let mut i = 5; // skip header
    while i < w.len() {
        let wc = (w[i] >> 16) as usize;
        let op = w[i] & 0xFFFF;
        out.push((op, w[i + 1..i + wc].to_vec()));
        i += wc;
    }
    out
}

fn has_capability(insts: &[(u32, Vec<u32>)], cap: u32) -> bool {
    insts
        .iter()
        .any(|(op, args)| *op == OP_CAPABILITY && args == &[cap])
}

fn decode_string(ws: &[u32]) -> String {
    let mut bytes = Vec::new();
    for w in ws {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn has_extension(insts: &[(u32, Vec<u32>)], name: &str) -> bool {
    insts
        .iter()
        .any(|(op, args)| *op == OP_EXTENSION && decode_string(args) == name)
}

fn array_strides(insts: &[(u32, Vec<u32>)]) -> Vec<u32> {
    insts
        .iter()
        .filter(|(op, args)| *op == OP_DECORATE && args.get(1) == Some(&DECORATION_ARRAY_STRIDE))
        .map(|(_, args)| args[2])
        .collect()
}

/// The id of `OpTypeInt <width> 0`, if declared.
fn type_int(insts: &[(u32, Vec<u32>)], width: u32) -> Option<u32> {
    insts.iter().find_map(|(op, a)| {
        if *op == OP_TYPE_INT && a.len() == 3 && a[1] == width && a[2] == 0 {
            Some(a[0])
        } else {
            None
        }
    })
}

/// The id of the u32-typed `OpConstant <val>`, if present.
fn constant_u32(insts: &[(u32, Vec<u32>)], val: u32) -> Option<u32> {
    let uint = type_int(insts, 32)?;
    insts.iter().find_map(|(op, a)| {
        if *op == OP_CONSTANT && a.len() == 3 && a[0] == uint && a[2] == val {
            Some(a[1])
        } else {
            None
        }
    })
}

/// The result id of the first instruction matching `opcode` whose
/// operand words satisfy `pred` — for the 2/3-operand value ops used
/// here, where word 0 is the result type and word 1 the result id.
fn result_of(insts: &[(u32, Vec<u32>)], opcode: u32, pred: impl Fn(&[u32]) -> bool) -> Option<u32> {
    insts.iter().find_map(|(op, a)| {
        if *op == opcode && a.len() >= 2 && pred(a) {
            Some(a[1])
        } else {
            None
        }
    })
}

/// Follow the load-widen chain for a narrow element type and return the
/// id of the in-register (canonical u32) value: `OpLoad %narrow` →
/// `OpUConvert %uint` → (signed only) `OpShiftLeftLogical` +
/// `OpShiftRightArithmetic` by `32 - bits`.
fn assert_widen_chain(insts: &[(u32, Vec<u32>)], name: &str, bits: u32, signed: bool) -> u32 {
    let narrow_ty = type_int(insts, bits)
        .unwrap_or_else(|| panic!("{name}: module must declare OpTypeInt {bits} 0"));
    let uint_ty = type_int(insts, 32).expect("canonical u32 type");
    let loaded = result_of(insts, OP_LOAD, |a| a[0] == narrow_ty)
        .unwrap_or_else(|| panic!("{name}: no OpLoad of the {bits}-bit storage element"));
    let widened = result_of(insts, OP_U_CONVERT, |a| a[0] == uint_ty && a[2] == loaded)
        .unwrap_or_else(|| panic!("{name}: loaded value must widen to u32 via OpUConvert"));
    if !signed {
        return widened;
    }
    let dist = constant_u32(insts, 32 - bits)
        .unwrap_or_else(|| panic!("{name}: missing shift-distance constant {}", 32 - bits));
    let hi = result_of(insts, OP_SHIFT_LEFT_LOGICAL, |a| {
        a[2] == widened && a[3] == dist
    })
    .unwrap_or_else(|| panic!("{name}: signed widen must shl by {}", 32 - bits));
    result_of(insts, OP_SHIFT_RIGHT_ARITHMETIC, |a| {
        a[2] == hi && a[3] == dist
    })
    .unwrap_or_else(|| panic!("{name}: signed widen must arithmetic-shr by {}", 32 - bits))
}

/// Find the store-truncate seam: an `OpUConvert %narrow` whose result
/// is the object of an `OpStore`. Returns the truncated value id.
fn assert_truncate_store(insts: &[(u32, Vec<u32>)], name: &str, bits: u32) -> u32 {
    let narrow_ty = type_int(insts, bits)
        .unwrap_or_else(|| panic!("{name}: module must declare OpTypeInt {bits} 0"));
    let truncated: Vec<u32> = insts
        .iter()
        .filter_map(|(op, a)| {
            if *op == OP_U_CONVERT && a[0] == narrow_ty {
                Some(a[1])
            } else {
                None
            }
        })
        .collect();
    assert!(
        !truncated.is_empty(),
        "{name}: store must truncate to the {bits}-bit element via OpUConvert"
    );
    *truncated
        .iter()
        .find(|id| {
            insts
                .iter()
                .any(|(op, a)| *op == OP_STORE && a.get(1) == Some(id))
        })
        .unwrap_or_else(|| panic!("{name}: no OpStore consumes the truncated {bits}-bit value"))
}

/// Run `spirv-val --target-env vulkan1.3` when available; skip silently
/// (like the build-time gate) when it isn't installed. An unarmed
/// validator (no --target-env) misses Vulkan-specific rules — the
/// SeqCst lesson.
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
fn narrow_int_buffers_use_native_stride_and_declare_capabilities() {
    for (ty, bits, signed, stride, tag) in NARROW_INTS {
        let name = format!("{tag}_roundtrip");
        let spirv = emit_spirv::emit(&roundtrip_kernel(ty, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);

        let strides = array_strides(&insts);
        assert!(
            strides.contains(&stride),
            "{name}: buffer must have ArrayStride {stride}, got {strides:?}"
        );
        let (cap, other_cap) = if bits == 8 {
            (CAP_STORAGE_BUFFER_8BIT, CAP_STORAGE_BUFFER_16BIT)
        } else {
            (CAP_STORAGE_BUFFER_16BIT, CAP_STORAGE_BUFFER_8BIT)
        };
        assert!(
            has_capability(&insts, cap),
            "{name}: module must declare the {bits}-bit storage capability"
        );
        assert!(
            !has_capability(&insts, other_cap),
            "{name}: module must not declare the unused storage capability"
        );
        assert_eq!(
            has_extension(&insts, "SPV_KHR_8bit_storage"),
            bits == 8,
            "{name}: SPV_KHR_8bit_storage iff 8-bit storage is used"
        );

        assert_widen_chain(&insts, &name, bits, signed);
        assert_truncate_store(&insts, &name, bits);
        spirv_val(&name, &spirv);
    }
}

#[test]
fn signed_compare_orders_on_the_sign_extended_register() {
    // i8/i16: the Lt lowers to OpSLessThan over the sign-extended widen
    // result — a stored -1 orders below 0. u8/u16: OpULessThan, and no
    // sign-extension shifts appear in the module.
    for (ty, bits, signed, _, tag) in NARROW_INTS {
        let name = format!("{tag}_cmp_lt_zero");
        let spirv = emit_spirv::emit(&cmp_kernel(ty, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);

        assert_widen_chain(&insts, &name, bits, signed);
        let (want, reject) = if signed {
            (OP_SLESS_THAN, OP_ULESS_THAN)
        } else {
            (OP_ULESS_THAN, OP_SLESS_THAN)
        };
        assert!(
            insts.iter().any(|(op, _)| *op == want),
            "{name}: expected opcode {want}"
        );
        assert!(
            !insts.iter().any(|(op, _)| *op == reject),
            "{name}: wrong-signedness compare {reject} present"
        );
        if !signed {
            assert!(
                !insts.iter().any(|(op, _)| *op == OP_SHIFT_RIGHT_ARITHMETIC),
                "{name}: unsigned widen must not sign-extend"
            );
        }
        spirv_val(&name, &spirv);
    }
}

#[test]
fn cast_of_negative_one_reduces_then_stores_the_narrow_pattern() {
    // (i8)(-1i32) → register reduction (shl/shr by 32-w of the
    // 0xFFFFFFFF constant — the CPU reference's mask-then-extend) →
    // store truncation to the 8-/16-bit element. The full negative
    // round-trip pinned as a def-use chain.
    for (to, bits, _, _, tag) in [
        (ScalarType::I8, 8u32, true, 1u32, "i8"),
        (ScalarType::I16, 16, true, 2, "i16"),
    ] {
        let name = format!("cast_neg1_to_{tag}");
        let spirv = emit_spirv::emit(&cast_neg1_kernel(to, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);

        let neg1 = constant_u32(&insts, u32::MAX)
            .unwrap_or_else(|| panic!("{name}: missing the 0xFFFFFFFF constant"));
        let dist = constant_u32(&insts, 32 - bits)
            .unwrap_or_else(|| panic!("{name}: missing shift-distance constant"));
        let hi = result_of(&insts, OP_SHIFT_LEFT_LOGICAL, |a| {
            a[2] == neg1 && a[3] == dist
        })
        .unwrap_or_else(|| panic!("{name}: cast must shl the constant by {}", 32 - bits));
        let reduced = result_of(&insts, OP_SHIFT_RIGHT_ARITHMETIC, |a| {
            a[2] == hi && a[3] == dist
        })
        .unwrap_or_else(|| panic!("{name}: cast must arithmetic-shr back"));
        let narrow_ty = type_int(&insts, bits).expect("narrow storage type");
        let truncated = result_of(&insts, OP_U_CONVERT, |a| {
            a[0] == narrow_ty && a[2] == reduced
        })
        .unwrap_or_else(|| panic!("{name}: store must truncate the reduced value"));
        assert!(
            insts
                .iter()
                .any(|(op, a)| *op == OP_STORE && a.get(1) == Some(&truncated)),
            "{name}: no OpStore of the truncated value"
        );
        spirv_val(&name, &spirv);
    }
}

#[test]
fn mixed_narrow_and_wide_kernel_carries_all_strides() {
    let spirv = emit_spirv::emit(&mixed_kernel("mixed_widths")).unwrap();
    let w = words(&spirv);
    let insts = instructions(&w);

    let strides = array_strides(&insts);
    for stride in [1, 2, 4] {
        assert!(
            strides.contains(&stride),
            "mixed_widths: expected ArrayStride {stride}, got {strides:?}"
        );
    }
    assert!(has_capability(&insts, CAP_STORAGE_BUFFER_8BIT));
    assert!(has_capability(&insts, CAP_STORAGE_BUFFER_16BIT));
    assert_widen_chain(&insts, "mixed_widths(u8)", 8, false);
    assert_widen_chain(&insts, "mixed_widths(i16)", 16, true);
    spirv_val("mixed_widths", &spirv);
}

#[test]
fn accumulator_loop_over_narrow_loads_validates() {
    // The quantized-GEMM shape: loop-carried f32 accumulator, narrow
    // load + cast to f32 inside the loop body (widen chain under a
    // loop phi — the historically bug-prone dominance shape).
    for (ty, bits, signed, _, tag) in NARROW_INTS {
        let name = format!("{tag}_accum_loop");
        let spirv = emit_spirv::emit(&accum_kernel(ty, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);
        assert_widen_chain(&insts, &name, bits, signed);
        spirv_val(&name, &spirv);
    }
}

#[test]
fn wide_int_kernels_declare_no_narrow_capabilities() {
    for ty in [ScalarType::U32, ScalarType::I32] {
        let name = format!("{ty:?}_roundtrip");
        let spirv = emit_spirv::emit(&roundtrip_kernel(ty, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);
        assert!(!has_capability(&insts, CAP_STORAGE_BUFFER_8BIT));
        assert!(!has_capability(&insts, CAP_STORAGE_BUFFER_16BIT));
        assert!(!has_extension(&insts, "SPV_KHR_8bit_storage"));
        spirv_val(&name, &spirv);
    }
}

#[test]
fn narrow_satadd_clamps_at_the_type_bound() {
    // u8/u16 SatAdd saturates at the TYPE's bound: the sum rides the
    // 32-bit carrier (which cannot wrap for canonical narrow
    // operands), so the overflow test is `sum > MAX_w` against the
    // narrow max (OpUGreaterThan), selecting the max on overflow —
    // not the wide wrap test `sum < a`.
    for (ty, max, tag) in [
        (ScalarType::U8, 0xFFu32, "u8"),
        (ScalarType::U16, 0xFFFF, "u16"),
    ] {
        let name = format!("{tag}_satadd");
        let spirv = emit_spirv::emit(&binop_kernel(BinOp::SatAdd, ty, &name)).unwrap();
        let w = words(&spirv);
        let insts = instructions(&w);
        let max_id = constant_u32(&insts, max)
            .unwrap_or_else(|| panic!("{name}: narrow max constant {max:#x} must be emitted"));
        let over = result_of(&insts, OP_UGREATER_THAN, |a| a.get(3) == Some(&max_id))
            .unwrap_or_else(|| panic!("{name}: overflow test must be `sum > {max:#x}`"));
        assert!(
            result_of(&insts, OP_SELECT, |a| a.get(2) == Some(&over)
                && a.get(3) == Some(&max_id))
            .is_some(),
            "{name}: overflow must select the narrow max"
        );
        assert!(
            !insts.iter().any(|(op, _)| *op == OP_ULESS_THAN),
            "{name}: the wide wrap test (sum < a) must not appear"
        );
        spirv_val(&name, &spirv);
    }

    // Wide control: u32 SatAdd keeps the wrap test (`sum < a`) and
    // the 32-bit max.
    let spirv =
        emit_spirv::emit(&binop_kernel(BinOp::SatAdd, ScalarType::U32, "u32_satadd")).unwrap();
    let w = words(&spirv);
    let insts = instructions(&w);
    assert!(
        constant_u32(&insts, u32::MAX).is_some(),
        "u32_satadd: 32-bit max constant must be emitted"
    );
    assert!(
        insts.iter().any(|(op, _)| *op == OP_ULESS_THAN),
        "u32_satadd: wide overflow test is `sum < a`"
    );
    assert!(
        !insts.iter().any(|(op, _)| *op == OP_UGREATER_THAN),
        "u32_satadd: no narrow clamp compare on the wide path"
    );
    spirv_val("u32_satadd", &spirv);
}

#[test]
fn narrow_rotate_masks_and_reextends_signed_carriers() {
    // Signed narrow rotates mask the SIGN-extended carrier to the
    // storage width before the shift decomposition (an OpBitwiseAnd
    // against the width mask — without it the logical right shift
    // drags the extension bits into the low w bits), and re-extend
    // the result (a third shl + arithmetic-shr pair beyond the two
    // the loads contribute). Unsigned narrow carriers are already
    // zero-extended: no width mask, no arithmetic shifts.
    for (sty, uty, mask, dist, tag) in [
        (ScalarType::I8, ScalarType::U8, 0xFFu32, 24u32, "i8"),
        (ScalarType::I16, ScalarType::U16, 0xFFFF, 16, "i16"),
    ] {
        for op in [BinOp::Rotl, BinOp::Rotr] {
            let name = format!("{tag}_rot");
            let spirv = emit_spirv::emit(&binop_kernel(op, sty, &name)).unwrap();
            let w = words(&spirv);
            let insts = instructions(&w);
            let mask_id = constant_u32(&insts, mask)
                .unwrap_or_else(|| panic!("{name}: width-mask constant {mask:#x} must exist"));
            assert!(
                result_of(&insts, OP_BITWISE_AND, |a| a.get(3) == Some(&mask_id)).is_some(),
                "{name}: the carrier must be masked to the width before the shifts"
            );
            let dist_id = constant_u32(&insts, dist)
                .unwrap_or_else(|| panic!("{name}: extension distance {dist} must exist"));
            let ashr_count = insts
                .iter()
                .filter(|(op, a)| *op == OP_SHIFT_RIGHT_ARITHMETIC && a.get(3) == Some(&dist_id))
                .count();
            assert_eq!(
                ashr_count, 3,
                "{name}: two load widens + the result re-extension = 3 \
                 arithmetic shifts by {dist}"
            );
            spirv_val(&name, &spirv);

            // Unsigned twin: zero-extended carrier — no width mask,
            // no sign-extension shifts.
            let uname = format!("u{}_rot", &tag[1..]);
            let spirv = emit_spirv::emit(&binop_kernel(op, uty, &uname)).unwrap();
            let w = words(&spirv);
            let insts = instructions(&w);
            assert!(
                !insts.iter().any(|(op, _)| *op == OP_SHIFT_RIGHT_ARITHMETIC),
                "{uname}: no arithmetic shifts on the unsigned path"
            );
            spirv_val(&uname, &spirv);
        }
    }
}

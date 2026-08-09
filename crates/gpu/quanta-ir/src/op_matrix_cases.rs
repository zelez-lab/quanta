//! Per-op differential matrix — the case generator.
//!
//! Shared source of truth for the `op_matrix` test harness (which
//! dispatches each case on the software / Metal / Vulkan lanes) and the
//! WGSL browser audit (`examples/web_diff`, which dispatches through real
//! WebGPU). Every case is a minimal kernel performing one op on two
//! scalar inputs, paired with the CPU-computed expected output.
//!
//! This module is pure case generation: no GPU dispatch, no comparison.
//! Those live test-side / example-side, parameterised over `OpCase`.

use crate::{
    BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, QuantScheme, QuantValue, Reg,
    ScalarType, UnaryOp,
};

/// Typed scalar buffer — one variant per supported scalar width. Inputs
/// and expected outputs are carried as length-1 vectors.
#[derive(Clone, Debug)]
pub enum RawValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    /// Narrow ints, carried at their native width (tight 1-/2-byte
    /// storage on every lane that runs them; WGSL rejects them).
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    /// bfloat16 values carried as their raw 16-bit storage patterns.
    BF16(Vec<u16>),
    /// fp8 values carried as their raw 8-bit storage patterns.
    FP8E5M2(Vec<u8>),
    FP8E4M3(Vec<u8>),
    /// Quantized integer codes (symmetric int8 / int4), carried as i8.
    Q8(Vec<i8>),
    Q4(Vec<i8>),
}

impl RawValues {
    pub fn type_tag(&self) -> &'static str {
        match self {
            RawValues::F32(_) => "f32",
            RawValues::F64(_) => "f64",
            RawValues::U32(_) => "u32",
            RawValues::U64(_) => "u64",
            RawValues::I32(_) => "i32",
            RawValues::I64(_) => "i64",
            RawValues::U8(_) => "u8",
            RawValues::I8(_) => "i8",
            RawValues::U16(_) => "u16",
            RawValues::I16(_) => "i16",
            RawValues::BF16(_) => "bf16",
            RawValues::FP8E5M2(_) => "fp8e5m2",
            RawValues::FP8E4M3(_) => "fp8e4m3",
            RawValues::Q8(_) => "q8",
            RawValues::Q4(_) => "q4",
        }
    }
}

pub const NAME_PREFIX: &str = "op_matrix";

/// One row in the matrix: a single (op, ty, a, b) instance and the
/// CPU-computed expected output.
///
/// `max_ulps` is the comparator tolerance applied to floating-point
/// outputs. Integer ops set it to 0 (bit-exact). Float Add/Sub/Mul
/// are bit-exact on every backend we ship; Div is allowed up to 1
/// ULP — the IEEE 754 spec doesn't pin down rounding of the last
/// bit across compilers for division.
#[derive(Clone, Debug)]
pub struct OpCase {
    pub name: String,
    pub def: KernelDef,
    pub input_a: RawValues,
    pub input_b: RawValues,
    pub expected: RawValues,
    pub max_ulps: u32,
    /// Some cases can't run on every backend yet — e.g. F64 on
    /// Metal is unsupported. The driver skips a case when its
    /// `lane_supports` returns false for the lane under test.
    pub skip_on_metal: bool,
}

/// Build a `KernelDef` of shape:
///
/// ```text
///   r0 = QuarkId               (unused but required for indexing semantics)
///   r1 = Load a[0]
///   r2 = Load b[0]
///   r3 = BinOp { op, ty } r1 r2
///   Store out[0] = r3
/// ```
fn build_binop_def(op_name: &str, ty: ScalarType, op: BinOp) -> KernelDef {
    let kernel_name = format!("{}_{}_{}", NAME_PREFIX, op_name, scalar_tag(ty));
    KernelDef {
        name: kernel_name,
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
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

fn scalar_tag(ty: ScalarType) -> &'static str {
    match ty {
        ScalarType::F16 => "f16",
        ScalarType::BF16 => "bf16",
        ScalarType::FP8E5M2 => "fp8e5m2",
        ScalarType::FP8E4M3 => "fp8e4m3",
        ScalarType::F32 => "f32",
        ScalarType::F64 => "f64",
        ScalarType::U8 => "u8",
        ScalarType::U16 => "u16",
        ScalarType::U32 => "u32",
        ScalarType::U64 => "u64",
        ScalarType::I8 => "i8",
        ScalarType::I16 => "i16",
        ScalarType::I32 => "i32",
        ScalarType::I64 => "i64",
        ScalarType::I4 => "i4",
        ScalarType::Bool => "bool",
    }
}

fn binop_tag(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Rem => "rem",
        BinOp::BitAnd => "bitand",
        BinOp::BitOr => "bitor",
        BinOp::BitXor => "bitxor",
        BinOp::Shl => "shl",
        BinOp::Shr => "shr",
        BinOp::Rotl => "rotl",
        BinOp::Rotr => "rotr",
        BinOp::SatAdd => "satadd",
        BinOp::SatSub => "satsub",
    }
}

fn unaryop_tag(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "neg",
        UnaryOp::BitNot => "bitnot",
        UnaryOp::LogicalNot => "logicalnot",
    }
}

/// Build a `KernelDef` of shape:
///
/// ```text
///   r0 = QuarkId
///   r1 = Load a[0]
///   r2 = Load b[0]              (bound but unused — keeps the
///                                dispatcher uniform with BinOp)
///   r3 = UnaryOp { op, ty } r1
///   Store out[0] = r3
/// ```
fn build_unary_def(op_name: &str, ty: ScalarType, op: UnaryOp) -> KernelDef {
    let kernel_name = format!("{}_{}_{}", NAME_PREFIX, op_name, scalar_tag(ty));
    KernelDef {
        name: kernel_name,
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
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
            KernelOp::UnaryOp {
                dst: Reg(3),
                a: Reg(1),
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

fn cmpop_tag(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    }
}

/// Build a `KernelDef` of shape:
///
/// ```text
///   r0 = QuarkId
///   r1 = Load a[0]      (operand type)
///   r2 = Load b[0]      (operand type)
///   r3 = Cmp(r1, r2, op, operand_type)   -> bool
///   r4 = Cast(r3, Bool, U32)              -> 0 or 1
///   Store out[0] = r4
/// ```
///
/// `out` is a `Field<u32>` carrying the comparison result encoded
/// as 0 / 1, which lets us reuse the standard u32 dispatch path.
fn build_cmp_def(op_name: &str, operand_ty: ScalarType, op: CmpOp) -> KernelDef {
    let kernel_name = format!("{}_{}_{}", NAME_PREFIX, op_name, scalar_tag(operand_ty));
    KernelDef {
        name: kernel_name,
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: operand_ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: operand_ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: operand_ty,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: operand_ty,
            },
            KernelOp::Cmp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op,
                ty: operand_ty,
            },
            KernelOp::Cast {
                dst: Reg(4),
                src: Reg(3),
                from: ScalarType::Bool,
                to: ScalarType::U32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(4),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 5,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Build a `KernelDef` of shape:
///
/// ```text
///   r0 = QuarkId
///   r1 = Load a[0]              (from-type)
///   r2 = Load b[0]              (from-type, unused)
///   r3 = Cast(r1, from, to)
///   Store out[0] = r3           (to-type)
/// ```
///
/// `out` matches the target type. `b` is bound but unused, like in
/// the Unary builder, so the standard pair-dispatch works.
fn build_cast_def(from: ScalarType, to: ScalarType) -> KernelDef {
    let kernel_name = format!(
        "{}_cast_{}_to_{}",
        NAME_PREFIX,
        scalar_tag(from),
        scalar_tag(to)
    );
    KernelDef {
        name: kernel_name,
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: from,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: from,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: to,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: from,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: from,
            },
            KernelOp::Cast {
                dst: Reg(3),
                src: Reg(1),
                from,
                to,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: to,
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

/// Build a `KernelDef` of shape:
///
/// ```text
///   r0 = QuarkId
///   r1 = Load a[0]
///   r2 = Load b[0]              (unused; dispatcher-uniform stub)
///   r3 = Const(c)
///   r4 = BinOp { op, ty } r1 r3
///   Store out[0] = r4
/// ```
///
/// This is the path the `85551fa` float-const bug rode on: the
/// scaling factor (e.g. `1.0f32 / (1 << 24)`) lowered to a
/// `KernelOp::Const`, which the MSL emitter then formatted as text.
/// Without this builder, the matrix never exercises that path —
/// every BinOp case loads both operands from buffers, so the
/// constant emitter is untested.
fn build_const_binop_def(name_suffix: &str, ty: ScalarType, op: BinOp, c: ConstValue) -> KernelDef {
    let kernel_name = format!(
        "{}_{}_{}_const_{}",
        NAME_PREFIX,
        binop_tag(op),
        scalar_tag(ty),
        name_suffix
    );
    KernelDef {
        name: kernel_name,
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
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
            KernelOp::Const {
                dst: Reg(3),
                value: c,
            },
            KernelOp::BinOp {
                dst: Reg(4),
                a: Reg(1),
                b: Reg(3),
                op,
                ty,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(4),
                ty,
            },
        ],
        body_source: None,
        next_reg: 5,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Build a `KernelDef` that exercises the **shift-after-signed-op**
/// path of the `06e764c` bug.
///
/// Kernel shape:
///
/// ```text
///   r0 = QuarkId
///   r1 = Load a[0]                       (uint)
///   r2 = Load b[0]                       (uint, unused)
///   r3 = Cast(r1, U32 -> I32)             (int — but holding `a`'s bit pattern)
///   r4 = Const(8u32)
///   r5 = BinOp::Shr { ty: U32 } r3 r4    (unsigned shift of int-typed reg)
///   Store out[0] = r5                    (uint result)
/// ```
///
/// The plain BinOp matrix loads `a` as U32 directly and shifts it,
/// which the emitter already gets right. To trigger the bug we need
/// a U32-typed result whose **operand register is int-typed** —
/// exactly the WASM-route pattern where `i32.xor` produces an I32
/// register that an unsigned `i32.shr_u` then consumes.
fn build_shr_after_signed_def() -> KernelDef {
    KernelDef {
        name: format!("{}_shr_after_signed", NAME_PREFIX),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Cast {
                dst: Reg(3),
                src: Reg(1),
                from: ScalarType::U32,
                to: ScalarType::I32,
            },
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::U32(8),
            },
            KernelOp::BinOp {
                dst: Reg(5),
                a: Reg(3),
                b: Reg(4),
                op: BinOp::Shr,
                ty: ScalarType::U32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(5),
                ty: ScalarType::U32,
            },
        ],
        body_source: None,
        next_reg: 6,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

// ── Case generators ──────────────────────────────────────────────────
//
// Per-type edge-input lists target the bugs we've seen *and* the
// adjacent cases that would surface a similar regression:
//
//   - sign-bit set (`0x80000000`): catches the shift sign-extension
//     bug fixed in 06e764c.
//   - all-ones (`!0`): catches off-by-one truncation / wrap.
//   - MIN / MAX of signed types: catches overflow on wrapping ops.
//   - small literal pair: catches the trivial case.
//   - zero: catches division/remainder by zero (skipped for Div/Rem).

/// U32 edge-input pairs: `(a, b)`. `b` is the shift amount for
/// shift ops and the second operand otherwise. The same list is
/// used for every op; division ops filter out `b == 0` at
/// generation time.
fn u32_inputs() -> &'static [(u32, u32)] {
    &[
        (0x80000000, 8),
        (0xFFFFFFFF, 1),
        (0x12345678, 4),
        (1, 1),
        (0, 5),
        (5, 0),
        (0x7FFFFFFF, 31),
    ]
}

fn u64_inputs() -> &'static [(u64, u64)] {
    &[
        (0x8000_0000_0000_0000, 32),
        (0xFFFF_FFFF_FFFF_FFFF, 1),
        (0x1234_5678_9ABC_DEF0, 16),
        (1, 1),
        (0, 5),
        (5, 0),
    ]
}

fn i32_inputs() -> &'static [(i32, i32)] {
    &[
        (i32::MIN, 1),
        (i32::MAX, 1),
        (-1, 1),
        (1, 1),
        (0, 5),
        (5, 0),
        (-2_147_483_647, 2),
    ]
}

fn i64_inputs() -> &'static [(i64, i64)] {
    &[
        (i64::MIN, 1),
        (i64::MAX, 1),
        (-1, 1),
        (1, 1),
        (0, 5),
        (5, 0),
    ]
}

/// Apply a `BinOp` on the host side using the same wrapping/saturating
/// semantics the CPU executor uses (`src/driver/cpu/eval.rs`).
/// Returns `None` if the op is undefined for the input (e.g. `Div` by
/// zero) so the caller can skip that case.
fn host_apply_u32(op: BinOp, a: u32, b: u32) -> Option<u32> {
    Some(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div if b == 0 => return None,
        BinOp::Div => a / b,
        BinOp::Rem if b == 0 => return None,
        BinOp::Rem => a % b,
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b),
        BinOp::Shr => a.wrapping_shr(b),
        BinOp::Rotl => a.rotate_left(b),
        BinOp::Rotr => a.rotate_right(b),
        BinOp::SatAdd => a.saturating_add(b),
        BinOp::SatSub => a.saturating_sub(b),
    })
}

fn host_apply_u64(op: BinOp, a: u64, b: u64) -> Option<u64> {
    Some(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div if b == 0 => return None,
        BinOp::Div => a / b,
        BinOp::Rem if b == 0 => return None,
        BinOp::Rem => a % b,
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
        BinOp::Rotl => a.rotate_left(b as u32),
        BinOp::Rotr => a.rotate_right(b as u32),
        BinOp::SatAdd => a.saturating_add(b),
        BinOp::SatSub => a.saturating_sub(b),
    })
}

fn host_apply_i32(op: BinOp, a: i32, b: i32) -> Option<i32> {
    Some(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div if b == 0 => return None,
        // i32::MIN / -1 is UB in C/MSL — skip.
        BinOp::Div if a == i32::MIN && b == -1 => return None,
        BinOp::Div => a / b,
        BinOp::Rem if b == 0 => return None,
        BinOp::Rem if a == i32::MIN && b == -1 => return None,
        BinOp::Rem => a % b,
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
        BinOp::Rotl => (a as u32).rotate_left(b as u32) as i32,
        BinOp::Rotr => (a as u32).rotate_right(b as u32) as i32,
        BinOp::SatAdd => a.saturating_add(b),
        BinOp::SatSub => a.saturating_sub(b),
    })
}

fn host_apply_i64(op: BinOp, a: i64, b: i64) -> Option<i64> {
    Some(match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div if b == 0 => return None,
        BinOp::Div if a == i64::MIN && b == -1 => return None,
        BinOp::Div => a / b,
        BinOp::Rem if b == 0 => return None,
        BinOp::Rem if a == i64::MIN && b == -1 => return None,
        BinOp::Rem => a % b,
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
        BinOp::Rotl => (a as u64).rotate_left(b as u32) as i64,
        BinOp::Rotr => (a as u64).rotate_right(b as u32) as i64,
        BinOp::SatAdd => a.saturating_add(b),
        BinOp::SatSub => a.saturating_sub(b),
    })
}

/// Every BinOp variant that takes two same-type integer operands and
/// produces one of the same type. Excludes saturating ops on signed
/// types only because the CPU executor's signed-sat coverage is
/// untested in this matrix — add when the unsigned matrix proves
/// stable.
const INT_BINOPS: &[BinOp] = &[
    BinOp::Add,
    BinOp::Sub,
    BinOp::Mul,
    BinOp::Div,
    BinOp::Rem,
    BinOp::BitAnd,
    BinOp::BitOr,
    BinOp::BitXor,
    BinOp::Shl,
    BinOp::Shr,
];

/// Saturating ops apply to unsigned integer types in the existing IR
/// surface (see `gpu_saturation.rs` test). They get their own list so
/// the signed generators can omit them.
const UNSIGNED_SAT_OPS: &[BinOp] = &[BinOp::SatAdd, BinOp::SatSub];

/// Rotate ops apply to any integer width. Same shape as INT_BINOPS but
/// kept separate because they take their shift amount mod the type's
/// width and could have different emitter paths.
const ROTATE_OPS: &[BinOp] = &[BinOp::Rotl, BinOp::Rotr];

// Concrete builders. Each integer width gets its own because the
// RawValues variant tag drives the dispatch path in `dispatch_on`.

fn case_u32(op: BinOp, a: u32, b: u32, expected: u32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#010x}_b{:#010x}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::U32),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::U32, op),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![b]),
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_u64(op: BinOp, a: u64, b: u64, expected: u64) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#018x}_b{:#018x}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::U64),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::U64, op),
        input_a: RawValues::U64(vec![a]),
        input_b: RawValues::U64(vec![b]),
        expected: RawValues::U64(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_i32(op: BinOp, a: i32, b: i32, expected: i32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{}_b{}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::I32),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::I32, op),
        input_a: RawValues::I32(vec![a]),
        input_b: RawValues::I32(vec![b]),
        expected: RawValues::I32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_i64(op: BinOp, a: i64, b: i64, expected: i64) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{}_b{}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::I64),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::I64, op),
        input_a: RawValues::I64(vec![a]),
        input_b: RawValues::I64(vec![b]),
        expected: RawValues::I64(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

/// Generate every `(INT_BINOPS ∪ UNSIGNED_SAT_OPS ∪ ROTATE_OPS) ×
/// u32_inputs()` case where the host op is defined.
fn cases_u32() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS.iter().chain(UNSIGNED_SAT_OPS).chain(ROTATE_OPS) {
        for &(a, b) in u32_inputs() {
            if let Some(e) = host_apply_u32(op, a, b) {
                out.push(case_u32(op, a, b, e));
            }
        }
    }
    out
}

fn cases_u64() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS.iter().chain(UNSIGNED_SAT_OPS).chain(ROTATE_OPS) {
        for &(a, b) in u64_inputs() {
            if let Some(e) = host_apply_u64(op, a, b) {
                out.push(case_u64(op, a, b, e));
            }
        }
    }
    out
}

fn cases_i32() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS.iter().chain(ROTATE_OPS) {
        for &(a, b) in i32_inputs() {
            if let Some(e) = host_apply_i32(op, a, b) {
                out.push(case_i32(op, a, b, e));
            }
        }
    }
    out
}

fn cases_i64() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS.iter().chain(ROTATE_OPS) {
        for &(a, b) in i64_inputs() {
            if let Some(e) = host_apply_i64(op, a, b) {
                out.push(case_i64(op, a, b, e));
            }
        }
    }
    out
}

// ── Narrow-int cases (u8 / i8 / u16 / i16) ───────────────────────────
//
// Narrow ints inherit the wide-int wrapping contract at their own
// width. The backends hold different register models — the CPU
// reference (and the SPIR-V unified-u32 SSA) widens narrow loads to
// 32-bit, computes at 32-bit, and truncates at the store; MSL types
// registers narrow and truncates per-op via C assignment — but for
// single-op kernels over in-range loads the models agree: mod 2^w is
// a quotient-ring homomorphism of mod 2^32 for add/sub/mul/neg, and
// div/rem/compare/shift operands loaded from narrow memory are always
// in-range. The host oracles below reuse the 32-bit wide oracles and
// truncate to the storage width, which mirrors the CPU reference
// (`eval.rs` + `write_scalar`) by construction.
//
// Edge-input families per type: width boundaries (0, 1, MAX, MAX−1,
// MIN), the sign-boundary byte/halfword (0x7F/0x80, 0x7FFF/0x8000),
// wrap witnesses (MAX+1 via add, 0−5 via sub, 200·3 / 50000·3 via
// mul), sign-extension probes (0xAA / 0xAAAA), MIN/−1 (defined at
// narrow width — computed at 32-bit, truncated on store, unlike the
// i32 case which stays excluded as C UB), ÷0 / %0 (filtered at
// generation like the wide rows), and shift counts at width−1 /
// width / beyond width. Shift counts stay ≤ 31: every native backend
// shifts at ≥ int width after C integer promotion, so counts ≥ 32
// are UB there — the wide rows observe the same cap.
//
// Excluded op families (narrow-specific; the divergences below were
// measured on real Metal during this matrix's first differential
// exercise of the narrow emitters, and independently confirmed on
// the SPIR-V lane). Rows for them would pin one backend's behavior
// as wrong — skipped-with-witness beats pinned-wrong:
//
//   - Rotl / Rotr: width-dependent, not ring ops. The CPU reference
//     rotates the widened value at 32-bit width and truncates on
//     store, while MSL and SPIR-V rotate at the native width —
//     measured: u8 rotl(0x81, 1) = 0x02 under the reference, 0x03 on
//     Metal (u16 likewise); SPIR-V witness u8 rotl(0x80, 1) = 0x01
//     vs the reference's 0x00. Narrow Rotr on MSL is worse: the
//     emitted `rotate(r, (8) - (r2 % 8))` doesn't compile — the
//     count expression promotes to `int` and the MSL `rotate`
//     overload set (uchar,uchar)/(int,int) is ambiguous. The IR has
//     no pinned narrow-rotate width contract and no array-surface
//     consumer emits narrow rotates; rows land when the contract
//     decision + interpreter alignment ship as a follow-up.
//   - SatAdd / SatSub: same width mismatch, three-way. The reference
//     and SPIR-V saturate in the u32 domain then truncate (u8
//     satadd(200, 200) = 400 → stored 144) while MSL clamps at the
//     narrow bounds (measured: 255). SatSub happens to agree (32-bit
//     saturating_sub of in-range narrow operands clamps to 0 exactly
//     like the narrow clamp), but the family ships together or not
//     at all. Saturating ops are not part of any dtype's array
//     surface; the wide rows keep them for the u32/u64 emitter
//     paths only.
//
// Narrow atomics: not excluded here because they cannot arise — the
// op-matrix generates no atomic cases for any type, and the
// differential atomic kernels (counter / race) are u32-only. Nothing
// emits narrow atomics, and they would be invalid SPIR-V.

/// u8 edge-input pairs `(a, b)`; `b` doubles as the shift count.
fn u8_inputs() -> &'static [(u8, u8)] {
    &[
        (0x80, 8), // sign-bit byte; shift == width
        (0xFF, 1), // MAX; add wraps to 0
        (0xFE, 2), // MAX−1
        (0x12, 4),
        (1, 1),
        (0, 5),    // 0−5 wraps (sub); ÷ and % defined
        (5, 0),    // ÷0 / %0 filtered at generation
        (0x7F, 7), // sign-boundary byte; shift == width−1
        (200, 3),  // 200·3 = 600 wraps to 88
        (0xAA, 9), // sign-extension probe; shift beyond width
    ]
}

fn i8_inputs() -> &'static [(i8, i8)] {
    &[
        (i8::MIN, 1),
        (i8::MAX, 1),
        (-1, 1),
        (1, 1),
        (0, 5),
        (5, 0),
        (i8::MIN, -1), // MIN/−1: defined at narrow width (see block comment)
        (i8::MIN, 7),
        (i8::MAX, 8), // shift == width
        (-1, 8),
        (100, 3), // 100·3 = 300 wraps to 44
        (-86, 2), // 0xAA sign-extension probe
    ]
}

fn u16_inputs() -> &'static [(u16, u16)] {
    &[
        (0x8000, 16), // sign-bit halfword; shift == width
        (0xFFFF, 1),  // MAX; add wraps to 0
        (0xFFFE, 2),  // MAX−1
        (0x1234, 4),
        (1, 1),
        (0, 5),
        (5, 0),
        (0x7FFF, 15), // sign-boundary halfword; shift == width−1
        (50_000, 3),  // 50000·3 = 150000 wraps to 18928
        (0x00FF, 8),  // shl crosses the byte boundary
        (0xAAAA, 17), // sign-extension probe; shift beyond width
    ]
}

fn i16_inputs() -> &'static [(i16, i16)] {
    &[
        (i16::MIN, 1),
        (i16::MAX, 1),
        (-1, 1),
        (1, 1),
        (0, 5),
        (5, 0),
        (i16::MIN, -1),
        (i16::MIN, 15),
        (i16::MAX, 16), // shift == width
        (-1, 16),
        (20_000, 3),  // 20000·3 = 60000 wraps to −5536
        (-21_846, 2), // 0xAAAA sign-extension probe
    ]
}

/// Narrow-unsigned oracle: zero-extend to 32-bit, apply the wide
/// wrapping op, truncate to the storage width — exactly the CPU
/// reference's load / eval / store pipeline. Callers never pass
/// shift counts ≥ 32 (the input lists cap them at width + 1).
fn host_apply_u8(op: BinOp, a: u8, b: u8) -> Option<u8> {
    host_apply_u32(op, a as u32, b as u32).map(|v| v as u8)
}

fn host_apply_u16(op: BinOp, a: u16, b: u16) -> Option<u16> {
    host_apply_u32(op, a as u32, b as u32).map(|v| v as u16)
}

/// Narrow-signed oracle: sign-extend to 32-bit, apply the wide
/// wrapping op, truncate. Shifts by a negative count are filtered
/// like ÷0: after C integer promotion the native backends would
/// shift by a huge/negative int count, which is UB. Note MIN/−1
/// passes through: sign-extended narrow operands never reach
/// i32::MIN, so the wide oracle's UB filter stays dormant and the
/// 32-bit quotient truncates back to MIN — the defined narrow
/// result every backend produces.
fn host_apply_i8(op: BinOp, a: i8, b: i8) -> Option<i8> {
    if matches!(op, BinOp::Shl | BinOp::Shr) && b < 0 {
        return None;
    }
    host_apply_i32(op, a as i32, b as i32).map(|v| v as i8)
}

fn host_apply_i16(op: BinOp, a: i16, b: i16) -> Option<i16> {
    if matches!(op, BinOp::Shl | BinOp::Shr) && b < 0 {
        return None;
    }
    host_apply_i32(op, a as i32, b as i32).map(|v| v as i16)
}

fn case_u8(op: BinOp, a: u8, b: u8, expected: u8) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#04x}_b{:#04x}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::U8),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::U8, op),
        input_a: RawValues::U8(vec![a]),
        input_b: RawValues::U8(vec![b]),
        expected: RawValues::U8(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_i8(op: BinOp, a: i8, b: i8, expected: i8) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{}_b{}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::I8),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::I8, op),
        input_a: RawValues::I8(vec![a]),
        input_b: RawValues::I8(vec![b]),
        expected: RawValues::I8(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_u16(op: BinOp, a: u16, b: u16, expected: u16) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#06x}_b{:#06x}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::U16),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::U16, op),
        input_a: RawValues::U16(vec![a]),
        input_b: RawValues::U16(vec![b]),
        expected: RawValues::U16(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_i16(op: BinOp, a: i16, b: i16, expected: i16) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{}_b{}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::I16),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::I16, op),
        input_a: RawValues::I16(vec![a]),
        input_b: RawValues::I16(vec![b]),
        expected: RawValues::I16(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_u8() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS {
        for &(a, b) in u8_inputs() {
            if let Some(e) = host_apply_u8(op, a, b) {
                out.push(case_u8(op, a, b, e));
            }
        }
    }
    out
}

fn cases_i8() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS {
        for &(a, b) in i8_inputs() {
            if let Some(e) = host_apply_i8(op, a, b) {
                out.push(case_i8(op, a, b, e));
            }
        }
    }
    out
}

fn cases_u16() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS {
        for &(a, b) in u16_inputs() {
            if let Some(e) = host_apply_u16(op, a, b) {
                out.push(case_u16(op, a, b, e));
            }
        }
    }
    out
}

fn cases_i16() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in INT_BINOPS {
        for &(a, b) in i16_inputs() {
            if let Some(e) = host_apply_i16(op, a, b) {
                out.push(case_i16(op, a, b, e));
            }
        }
    }
    out
}

// ── Float cases ──────────────────────────────────────────────────────
//
// The four float BinOps are Add, Sub, Mul, Div. Edge inputs target
// the float-const bug fixed in 85551fa (small magnitudes that the
// MSL `{:.6}` format used to round to literal zero), plus the
// standard FP corners (±0, ±denormal, ±MIN_POSITIVE, ±MAX, ±Inf).
// NaN inputs are excluded for now — `compare_f32` treats NaN-vs-NaN
// as "unranked" and would generate spurious failures. F32 ops on
// finite inputs are bit-exact on every backend (we don't ship
// fast-math today); F32 Div allows 1 ULP per IEEE 754.

const FLOAT_BINOPS: &[BinOp] = &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div];

fn f32_inputs() -> &'static [(f32, f32)] {
    &[
        (0.0, 0.0),
        (-0.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (-1.0, 2.0),
        // The exact constant from the 85551fa bug: 2^-24.
        (0.5, 5.960_464_5e-8),
        (1.0e-30, 1.0e-30), // subnormal-ish
        (f32::MIN_POSITIVE, 2.0),
        (f32::MAX, 0.5),
        (1.0, f32::EPSILON),
        (3.0, 7.0), // Div with non-power-of-two divisor — tests rounding
    ]
}

fn f64_inputs() -> &'static [(f64, f64)] {
    &[
        (0.0, 0.0),
        (1.0, 1.0),
        (1.0, 2.0),
        (-1.0, 2.0),
        // Same shape as the float-const bug at f64 magnitude.
        (0.5, 1.110_223_024_625_156_5e-16),
        (f64::MIN_POSITIVE, 2.0),
        (3.0, 7.0),
    ]
}

/// Apply a float BinOp on the host. Matches the CPU executor's f32
/// path (`src/driver/cpu/eval.rs:11`).
fn host_apply_f32(op: BinOp, a: f32, b: f32) -> Option<f32> {
    Some(match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b, // 0/0 = NaN, x/0 = ±Inf — both representable
        _ => return None,
    })
}

fn host_apply_f64(op: BinOp, a: f64, b: f64) -> Option<f64> {
    Some(match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        _ => return None,
    })
}

/// Float BinOp Div allows ≤ 1 ULP error; other float ops are
/// bit-exact on every backend we ship.
fn float_max_ulps(op: BinOp) -> u32 {
    match op {
        BinOp::Div => 1,
        _ => 0,
    }
}

fn case_f32(op: BinOp, a: f32, b: f32, expected: f32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}_b{:e}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::F32),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::F32, op),
        input_a: RawValues::F32(vec![a]),
        input_b: RawValues::F32(vec![b]),
        expected: RawValues::F32(vec![expected]),
        max_ulps: float_max_ulps(op),
        skip_on_metal: false,
    }
}

fn case_f64(op: BinOp, a: f64, b: f64, expected: f64) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}_b{:e}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::F64),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::F64, op),
        input_a: RawValues::F64(vec![a]),
        input_b: RawValues::F64(vec![b]),
        expected: RawValues::F64(vec![expected]),
        max_ulps: 0, // Software-only path is deterministic.
        // F64 on Metal: MSL has no `double` type. The structural
        // fix is queued for step 082 Layer 4 (capability table).
        // Until then, skip every F64 case on the Metal lane.
        skip_on_metal: true,
    }
}

fn cases_f32() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in FLOAT_BINOPS {
        for &(a, b) in f32_inputs() {
            if let Some(e) = host_apply_f32(op, a, b) {
                // Skip inputs where the expected result is NaN — the
                // comparator treats NaN as unranked.
                if e.is_nan() {
                    continue;
                }
                // Skip subnormal results: Metal defaults to flush-to-
                // zero on subnormals, which is a documented backend
                // behavior, not a bug. Once the capability table
                // (step 082 Layer 4) lands, the FTZ policy becomes a
                // queryable flag and this can be removed.
                if e != 0.0 && e.abs() < f32::MIN_POSITIVE {
                    continue;
                }
                out.push(case_f32(op, a, b, e));
            }
        }
    }
    out
}

// ── bf16 ─────────────────────────────────────────────────────────────

/// bf16 → f32: place the 16 bits into the f32 high half.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// f32 → bf16, round-to-nearest-even (matches the CPU executor and every
/// emitter's pack sequence bit-for-bit).
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    if val.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + bias) >> 16) as u16
}

/// bf16 input pairs (as f32 values that are exactly bf16-representable —
/// low 16 mantissa bits zero — so no input rounding muddies the test).
fn bf16_inputs() -> &'static [(f32, f32)] {
    &[
        (1.0, 2.0),
        (1.5, 0.5),
        (-1.0, 1.0),
        (3.0, 4.0),
        (0.0, 5.0),
        (-2.5, 2.5),
        (100.0, 0.25),
        (-0.0, 1.0),
    ]
}

fn case_bf16(op: BinOp, a: f32, b: f32, expected_bits: u16) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}_b{:e}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(ScalarType::BF16),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), ScalarType::BF16, op),
        input_a: RawValues::BF16(vec![f32_to_bf16(a)]),
        input_b: RawValues::BF16(vec![f32_to_bf16(b)]),
        expected: RawValues::BF16(vec![expected_bits]),
        max_ulps: 0, // bf16 result is packed identically host- and device-side.
        // WGSL/Vulkan/Metal all run bf16 via the portable u32-slot path.
        skip_on_metal: false,
    }
}

fn cases_bf16() -> Vec<OpCase> {
    let mut out = Vec::new();
    // Add/Sub/Mul are exact-then-rounded; Div is too but stays in range.
    for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
        for &(a, b) in bf16_inputs() {
            // The kernel loads bf16 inputs (already rounded), computes in
            // f32, and packs the result. The oracle does the same: unpack
            // the stored inputs, apply the op, pack the result.
            let fa = bf16_to_f32(f32_to_bf16(a));
            let fb = bf16_to_f32(f32_to_bf16(b));
            let r = match op {
                BinOp::Add => fa + fb,
                BinOp::Sub => fa - fb,
                BinOp::Mul => fa * fb,
                BinOp::Div if fb != 0.0 => fa / fb,
                _ => continue,
            };
            if r.is_nan() || (r != 0.0 && r.abs() < f32::MIN_POSITIVE) {
                continue;
            }
            out.push(case_bf16(op, a, b, f32_to_bf16(r)));
        }
    }
    out
}

// ── fp8 (e5m2 / e4m3) ────────────────────────────────────────────────

/// fp8 input pairs — small magnitudes that are exactly representable in
/// both formats, so no input rounding muddies the differential test.
fn fp8_inputs() -> &'static [(f32, f32)] {
    &[
        (1.0, 2.0),
        (1.5, 0.5),
        (-1.0, 1.0),
        (3.0, 4.0),
        (0.0, 5.0),
        (-2.5, 2.5),
        (0.5, 0.25),
        (-0.0, 1.0),
    ]
}

fn case_fp8(eb: u32, mb: u32, op: BinOp, a: f32, b: f32, expected: u8) -> OpCase {
    let sty = if (eb, mb) == (5, 2) {
        ScalarType::FP8E5M2
    } else {
        ScalarType::FP8E4M3
    };
    let wrap: fn(Vec<u8>) -> RawValues = if (eb, mb) == (5, 2) {
        RawValues::FP8E5M2
    } else {
        RawValues::FP8E4M3
    };
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}_b{:e}",
            NAME_PREFIX,
            binop_tag(op),
            scalar_tag(sty),
            a,
            b
        ),
        def: build_binop_def(binop_tag(op), sty, op),
        input_a: wrap(vec![crate::dtype::f32_to_fp8(a, eb, mb)]),
        input_b: wrap(vec![crate::dtype::f32_to_fp8(b, eb, mb)]),
        expected: wrap(vec![expected]),
        max_ulps: 0, // fp8 result is packed identically host- and device-side.
        skip_on_metal: false,
    }
}

fn cases_fp8() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &(eb, mb) in &[crate::dtype::E5M2, crate::dtype::E4M3] {
        for &op in &[BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div] {
            for &(a, b) in fp8_inputs() {
                // Kernel loads fp8 inputs (already rounded), computes in
                // f32, packs the result. The oracle does the same.
                let fa = crate::dtype::fp8_to_f32(crate::dtype::f32_to_fp8(a, eb, mb), eb, mb);
                let fb = crate::dtype::fp8_to_f32(crate::dtype::f32_to_fp8(b, eb, mb), eb, mb);
                let r = match op {
                    BinOp::Add => fa + fb,
                    BinOp::Sub => fa - fb,
                    BinOp::Mul => fa * fb,
                    BinOp::Div if fb != 0.0 => fa / fb,
                    _ => continue,
                };
                if r.is_nan() || r.is_infinite() {
                    continue;
                }
                out.push(case_fp8(
                    eb,
                    mb,
                    op,
                    a,
                    b,
                    crate::dtype::f32_to_fp8(r, eb, mb),
                ));
            }
        }
    }
    out
}

// ── int8 / int4 symmetric quantization ───────────────────────────────
//
// Two kernel shapes prove the round-trip end-to-end:
//   Quantize:   Load f32 a → Const scale → Quantize → Store int code
//   Dequantize: Load int code a → Const scale → Dequantize → Store f32
// The scale rides a `Const` (the op-matrix tests fixed scales), so no
// push-constant plumbing is needed. The oracle uses the identical
// dtype::{quantize_sym, dequantize_sym}.

/// Quantize kernel: `out_code = quantize(a, scale)`.
fn build_quantize_def(scheme: QuantScheme, scale: f32) -> KernelDef {
    let int_ty = scheme.value.storage_scalar();
    let zp = Reg(2); // reuse the scale reg slot for zero_point (Symmetric → unused)
    KernelDef {
        name: format!("{}_quantize_{}", NAME_PREFIX, scalar_tag(int_ty)),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: int_ty,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(scale),
            },
            KernelOp::Quantize {
                dst: Reg(3),
                src: Reg(1),
                scale: Reg(2),
                zero_point: zp,
                scheme,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: int_ty,
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

/// Dequantize kernel: `out_f32 = dequantize(a_code, scale)`.
fn build_dequantize_def(scheme: QuantScheme, scale: f32) -> KernelDef {
    let int_ty = scheme.value.storage_scalar();
    let zp = Reg(2);
    KernelDef {
        name: format!("{}_dequantize_{}", NAME_PREFIX, scalar_tag(int_ty)),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: int_ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: int_ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: int_ty,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(scale),
            },
            KernelOp::Dequantize {
                dst: Reg(3),
                src: Reg(1),
                scale: Reg(2),
                zero_point: zp,
                scheme,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
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

/// f32 inputs spanning in-range, clamp-hi, clamp-lo, and round-to-even
/// ties for a given scale.
fn quant_inputs(scale: f32) -> Vec<f32> {
    vec![
        0.0,
        scale,
        -scale,
        2.0 * scale,
        -3.0 * scale,
        0.5 * scale,  // tie → round to even (0)
        1.5 * scale,  // tie → round to even (2)
        2.5 * scale,  // tie → round to even (2)
        1000.0,       // clamp hi
        -1000.0,      // clamp lo
        0.49 * scale, // → 0
        0.51 * scale, // → 1
    ]
}

fn cases_quant() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &qv in &[QuantValue::Q8S, QuantValue::Q4S] {
        let scheme = QuantScheme::per_tensor_symmetric(qv);
        let int_ty = scheme.value.storage_scalar();
        let bits = qv.bits();
        let scale = 0.5f32;

        // Quantize: f32 → code.
        let q_wrap: fn(Vec<i8>) -> RawValues = match qv {
            QuantValue::Q8S => RawValues::Q8,
            QuantValue::Q4S => RawValues::Q4,
        };
        for &x in &quant_inputs(scale) {
            let code = crate::dtype::quantize_sym(x, scale, bits) as i8;
            out.push(OpCase {
                name: format!("{}_quantize_{}_x{:e}", NAME_PREFIX, scalar_tag(int_ty), x),
                def: build_quantize_def(scheme, scale),
                input_a: RawValues::F32(vec![x]),
                input_b: RawValues::F32(vec![x]),
                expected: q_wrap(vec![code]),
                max_ulps: 0,
                skip_on_metal: false,
            });
        }

        // Dequantize: code → f32, over every representable code.
        let (lo, hi) = scheme.value.range();
        for code in lo..=hi {
            let dq = crate::dtype::dequantize_sym(code, scale);
            out.push(OpCase {
                name: format!(
                    "{}_dequantize_{}_c{}",
                    NAME_PREFIX,
                    scalar_tag(int_ty),
                    code
                ),
                def: build_dequantize_def(scheme, scale),
                input_a: q_wrap(vec![code as i8]),
                input_b: q_wrap(vec![code as i8]),
                expected: RawValues::F32(vec![dq]),
                max_ulps: 0,
                skip_on_metal: false,
            });
        }
    }
    out
}

fn cases_f64() -> Vec<OpCase> {
    let mut out = Vec::new();
    for &op in FLOAT_BINOPS {
        for &(a, b) in f64_inputs() {
            if let Some(e) = host_apply_f64(op, a, b) {
                if e.is_nan() {
                    continue;
                }
                out.push(case_f64(op, a, b, e));
            }
        }
    }
    out
}

// ── Unary cases ──────────────────────────────────────────────────────
//
// UnaryOp::Neg works on signed ints and floats. Unsigned-int Neg in
// the IR is wrapping (two's-complement negation) and matches the
// CPU executor's `-` operator. BitNot is integer-only. LogicalNot
// is bool-only and not currently produced by the WASM-route
// translator, so we skip it from the matrix.

fn case_unary_u32(op: UnaryOp, a: u32, expected: u32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#010x}",
            NAME_PREFIX,
            unaryop_tag(op),
            scalar_tag(ScalarType::U32),
            a
        ),
        def: build_unary_def(unaryop_tag(op), ScalarType::U32, op),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![a]), // unused — see build_unary_def
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_unary_i32(op: UnaryOp, a: i32, expected: i32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{}",
            NAME_PREFIX,
            unaryop_tag(op),
            scalar_tag(ScalarType::I32),
            a
        ),
        def: build_unary_def(unaryop_tag(op), ScalarType::I32, op),
        input_a: RawValues::I32(vec![a]),
        input_b: RawValues::I32(vec![a]),
        expected: RawValues::I32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_unary_f32(op: UnaryOp, a: f32, expected: f32) -> OpCase {
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}",
            NAME_PREFIX,
            unaryop_tag(op),
            scalar_tag(ScalarType::F32),
            a
        ),
        def: build_unary_def(unaryop_tag(op), ScalarType::F32, op),
        input_a: RawValues::F32(vec![a]),
        input_b: RawValues::F32(vec![a]),
        expected: RawValues::F32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_unary() -> Vec<OpCase> {
    let mut out = Vec::new();

    // u32 BitNot: !0u32 = 0xFFFFFFFF, !0xFFFFFFFF = 0, ~mid = bitmask.
    for &a in &[0u32, 0x12345678u32, 0xFFFFFFFFu32, 0x80000000u32] {
        out.push(case_unary_u32(UnaryOp::BitNot, a, !a));
    }
    // u32 Neg: wrapping_neg matches the IR semantics.
    for &a in &[0u32, 1u32, 0x80000000u32, 0xFFFFFFFFu32] {
        out.push(case_unary_u32(UnaryOp::Neg, a, a.wrapping_neg()));
    }

    // i32 Neg: includes i32::MIN which is its own negation under
    // two's-complement wrap (the case most likely to surface a
    // signed-overflow bug).
    for &a in &[0i32, 1i32, -1i32, i32::MAX, i32::MIN, 42, -42] {
        out.push(case_unary_i32(UnaryOp::Neg, a, a.wrapping_neg()));
    }
    // i32 BitNot.
    for &a in &[0i32, -1i32, i32::MIN, i32::MAX, 42] {
        out.push(case_unary_i32(UnaryOp::BitNot, a, !a));
    }

    // f32 Neg: includes ±0 (sign-bit flip must produce the right
    // ±0 representation, not silently collapse to +0).
    for &a in &[
        0.0f32,
        -0.0f32,
        1.0f32,
        -1.0f32,
        f32::MAX,
        f32::MIN_POSITIVE,
    ] {
        out.push(case_unary_f32(UnaryOp::Neg, a, -a));
    }

    out
}

/// Narrow-int Neg + BitNot. Both are width-local (wrapping negation
/// and complement commute with truncation), so the direct narrow
/// Rust op is the oracle. Neg on unsigned wraps: −1 ≡ 255 on u8 —
/// the §3 contract the array surface documents.
fn case_unary_narrow(op: UnaryOp, ty: ScalarType, a_val: RawValues, expected: RawValues) -> OpCase {
    let name_val = match &a_val {
        RawValues::U8(v) => format!("a{:#04x}", v[0]),
        RawValues::I8(v) => format!("a{}", v[0]),
        RawValues::U16(v) => format!("a{:#06x}", v[0]),
        RawValues::I16(v) => format!("a{}", v[0]),
        _ => unreachable!("narrow unary builder fed a wide variant"),
    };
    OpCase {
        name: format!(
            "{}_{}_{}_{}",
            NAME_PREFIX,
            unaryop_tag(op),
            scalar_tag(ty),
            name_val
        ),
        def: build_unary_def(unaryop_tag(op), ty, op),
        input_a: a_val.clone(),
        input_b: a_val,
        expected,
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_unary_narrow() -> Vec<OpCase> {
    let mut out = Vec::new();

    for &a in &[0u8, 1, 0x7F, 0x80, 0xAA, 0xFF] {
        out.push(case_unary_narrow(
            UnaryOp::Neg,
            ScalarType::U8,
            RawValues::U8(vec![a]),
            RawValues::U8(vec![a.wrapping_neg()]),
        ));
        out.push(case_unary_narrow(
            UnaryOp::BitNot,
            ScalarType::U8,
            RawValues::U8(vec![a]),
            RawValues::U8(vec![!a]),
        ));
    }
    // i8 Neg includes i8::MIN, its own negation under two's-complement
    // wrap (the §3 contract; the reference computes −(−128) = 128 at
    // 32-bit and the store truncates back to −128).
    for &a in &[0i8, 1, -1, i8::MIN, i8::MAX, -86] {
        out.push(case_unary_narrow(
            UnaryOp::Neg,
            ScalarType::I8,
            RawValues::I8(vec![a]),
            RawValues::I8(vec![a.wrapping_neg()]),
        ));
        out.push(case_unary_narrow(
            UnaryOp::BitNot,
            ScalarType::I8,
            RawValues::I8(vec![a]),
            RawValues::I8(vec![!a]),
        ));
    }
    for &a in &[0u16, 1, 0x7FFF, 0x8000, 0xAAAA, 0xFFFF] {
        out.push(case_unary_narrow(
            UnaryOp::Neg,
            ScalarType::U16,
            RawValues::U16(vec![a]),
            RawValues::U16(vec![a.wrapping_neg()]),
        ));
        out.push(case_unary_narrow(
            UnaryOp::BitNot,
            ScalarType::U16,
            RawValues::U16(vec![a]),
            RawValues::U16(vec![!a]),
        ));
    }
    for &a in &[0i16, 1, -1, i16::MIN, i16::MAX, -21_846] {
        out.push(case_unary_narrow(
            UnaryOp::Neg,
            ScalarType::I16,
            RawValues::I16(vec![a]),
            RawValues::I16(vec![a.wrapping_neg()]),
        ));
        out.push(case_unary_narrow(
            UnaryOp::BitNot,
            ScalarType::I16,
            RawValues::I16(vec![a]),
            RawValues::I16(vec![!a]),
        ));
    }

    out
}

// ── Cmp cases ────────────────────────────────────────────────────────
//
// Every CmpOp on every scalar type we natively dispatch (U32, I32,
// F32). The kernel emits Cmp → Cast(Bool→U32) → Store; the
// expected output is the bool as 0/1 in a u32 lane. Inputs cover
// equality (a == b), strict ordering on both sides, and the
// sign-bit cases that historically miscompiled signed comparisons.

const CMP_OPS: &[CmpOp] = &[
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Gt,
    CmpOp::Ge,
];

fn host_apply_cmp_u32(op: CmpOp, a: u32, b: u32) -> u32 {
    (match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }) as u32
}

fn host_apply_cmp_i32(op: CmpOp, a: i32, b: i32) -> u32 {
    (match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }) as u32
}

fn host_apply_cmp_f32(op: CmpOp, a: f32, b: f32) -> u32 {
    (match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }) as u32
}

fn case_cmp_u32(op: CmpOp, a: u32, b: u32) -> OpCase {
    let expected = host_apply_cmp_u32(op, a, b);
    OpCase {
        name: format!(
            "{}_{}_{}_a{:#010x}_b{:#010x}",
            NAME_PREFIX,
            cmpop_tag(op),
            scalar_tag(ScalarType::U32),
            a,
            b
        ),
        def: build_cmp_def(cmpop_tag(op), ScalarType::U32, op),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![b]),
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_cmp_i32(op: CmpOp, a: i32, b: i32) -> OpCase {
    let expected = host_apply_cmp_i32(op, a, b);
    OpCase {
        name: format!(
            "{}_{}_{}_a{}_b{}",
            NAME_PREFIX,
            cmpop_tag(op),
            scalar_tag(ScalarType::I32),
            a,
            b
        ),
        def: build_cmp_def(cmpop_tag(op), ScalarType::I32, op),
        input_a: RawValues::I32(vec![a]),
        input_b: RawValues::I32(vec![b]),
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_cmp_f32(op: CmpOp, a: f32, b: f32) -> OpCase {
    let expected = host_apply_cmp_f32(op, a, b);
    OpCase {
        name: format!(
            "{}_{}_{}_a{:e}_b{:e}",
            NAME_PREFIX,
            cmpop_tag(op),
            scalar_tag(ScalarType::F32),
            a,
            b
        ),
        def: build_cmp_def(cmpop_tag(op), ScalarType::F32, op),
        input_a: RawValues::F32(vec![a]),
        input_b: RawValues::F32(vec![b]),
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_cmp() -> Vec<OpCase> {
    let mut out = Vec::new();

    // u32 comparisons including sign-bit values (unsigned, so high
    // bit is just a large magnitude — catches any backend that
    // accidentally signed-compares).
    let u32_pairs: &[(u32, u32)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (0x80000000, 0x7FFFFFFF),
        (0xFFFFFFFF, 0),
        (0x12345678, 0x12345678),
    ];
    for &op in CMP_OPS {
        for &(a, b) in u32_pairs {
            out.push(case_cmp_u32(op, a, b));
        }
    }

    // i32 comparisons exercising signed ordering on negatives.
    let i32_pairs: &[(i32, i32)] = &[
        (0, 0),
        (1, -1),
        (i32::MIN, i32::MAX),
        (i32::MIN, 0),
        (-1, 1),
        (42, 42),
    ];
    for &op in CMP_OPS {
        for &(a, b) in i32_pairs {
            out.push(case_cmp_i32(op, a, b));
        }
    }

    // f32 comparisons (finite only — NaN comparison is well-defined
    // by IEEE 754 but a separate axis we can fold in later).
    let f32_pairs: &[(f32, f32)] = &[
        (0.0, 0.0),
        (-0.0, 0.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (f32::INFINITY, f32::MAX),
        (f32::NEG_INFINITY, f32::INFINITY),
    ];
    for &op in CMP_OPS {
        for &(a, b) in f32_pairs {
            out.push(case_cmp_f32(op, a, b));
        }
    }

    out
}

/// Narrow-int comparisons. The sign-boundary operands (0x80 / 0x8000
/// bit patterns) pin the signedness split: as u8, 0x80 > 0x7F; as
/// i8, the same bits order −128 < 127. A backend that compares at
/// the wrong signedness — or compares sign-extended registers as
/// unsigned — inverts these rows. Output is the shared u32 0/1 lane.
fn case_cmp_narrow(op: CmpOp, ty: ScalarType, a: RawValues, b: RawValues, expected: u32) -> OpCase {
    let pair = match (&a, &b) {
        (RawValues::U8(x), RawValues::U8(y)) => format!("a{:#04x}_b{:#04x}", x[0], y[0]),
        (RawValues::I8(x), RawValues::I8(y)) => format!("a{}_b{}", x[0], y[0]),
        (RawValues::U16(x), RawValues::U16(y)) => format!("a{:#06x}_b{:#06x}", x[0], y[0]),
        (RawValues::I16(x), RawValues::I16(y)) => format!("a{}_b{}", x[0], y[0]),
        _ => unreachable!("narrow cmp builder fed a wide variant"),
    };
    OpCase {
        name: format!(
            "{}_{}_{}_{}",
            NAME_PREFIX,
            cmpop_tag(op),
            scalar_tag(ty),
            pair
        ),
        def: build_cmp_def(cmpop_tag(op), ty, op),
        input_a: a,
        input_b: b,
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_cmp_narrow() -> Vec<OpCase> {
    let mut out = Vec::new();

    let u8_pairs: &[(u8, u8)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (0x80, 0x7F), // unsigned: 128 > 127
        (0xFF, 0),
        (0x12, 0x12),
    ];
    for &op in CMP_OPS {
        for &(a, b) in u8_pairs {
            out.push(case_cmp_narrow(
                op,
                ScalarType::U8,
                RawValues::U8(vec![a]),
                RawValues::U8(vec![b]),
                host_apply_cmp_u32(op, a as u32, b as u32),
            ));
        }
    }

    let i8_pairs: &[(i8, i8)] = &[
        (0, 0),
        (1, -1),
        (i8::MIN, i8::MAX), // same bits as the u8 row, opposite order
        (i8::MIN, 0),
        (-1, 1),
        (42, 42),
    ];
    for &op in CMP_OPS {
        for &(a, b) in i8_pairs {
            out.push(case_cmp_narrow(
                op,
                ScalarType::I8,
                RawValues::I8(vec![a]),
                RawValues::I8(vec![b]),
                host_apply_cmp_i32(op, a as i32, b as i32),
            ));
        }
    }

    let u16_pairs: &[(u16, u16)] = &[
        (0, 0),
        (1, 0),
        (0, 1),
        (0x8000, 0x7FFF),
        (0xFFFF, 0),
        (0x1234, 0x1234),
    ];
    for &op in CMP_OPS {
        for &(a, b) in u16_pairs {
            out.push(case_cmp_narrow(
                op,
                ScalarType::U16,
                RawValues::U16(vec![a]),
                RawValues::U16(vec![b]),
                host_apply_cmp_u32(op, a as u32, b as u32),
            ));
        }
    }

    let i16_pairs: &[(i16, i16)] = &[
        (0, 0),
        (1, -1),
        (i16::MIN, i16::MAX),
        (i16::MIN, 0),
        (-1, 1),
        (42, 42),
    ];
    for &op in CMP_OPS {
        for &(a, b) in i16_pairs {
            out.push(case_cmp_narrow(
                op,
                ScalarType::I16,
                RawValues::I16(vec![a]),
                RawValues::I16(vec![b]),
                host_apply_cmp_i32(op, a as i32, b as i32),
            ));
        }
    }

    out
}

// ── Cast cases ───────────────────────────────────────────────────────
//
// The cast matrix grows quickly with type permutations. We cover
// the pairs the WASM-route translator actually emits (u32↔i32,
// u32↔f32, i32↔f32, and their narrow-int variants) with a small
// handful of edge inputs per pair.

fn host_cast_u32_to_i32(a: u32) -> i32 {
    a as i32
}
fn host_cast_i32_to_u32(a: i32) -> u32 {
    a as u32
}
fn host_cast_u32_to_f32(a: u32) -> f32 {
    a as f32
}
fn host_cast_f32_to_u32(a: f32) -> u32 {
    a as u32
}
fn host_cast_i32_to_f32(a: i32) -> f32 {
    a as f32
}
fn host_cast_f32_to_i32(a: f32) -> i32 {
    a as i32
}

fn case_cast(from_val: RawValues, expected: RawValues, from: ScalarType, to: ScalarType) -> OpCase {
    // For Cast the dummy `b` field must match `from`'s type; copy
    // `from_val` into b.
    OpCase {
        name: format!(
            "{}_cast_{}_to_{}_{}",
            NAME_PREFIX,
            scalar_tag(from),
            scalar_tag(to),
            from_val.type_tag(),
        ),
        def: build_cast_def(from, to),
        input_a: from_val.clone(),
        input_b: from_val,
        expected,
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_cast() -> Vec<OpCase> {
    let mut out = Vec::new();

    // u32 → i32 (bit-pattern reinterpretation).
    for &a in &[0u32, 1u32, 0x7FFFFFFFu32, 0x80000000u32, 0xFFFFFFFFu32] {
        out.push(case_cast(
            RawValues::U32(vec![a]),
            RawValues::I32(vec![host_cast_u32_to_i32(a)]),
            ScalarType::U32,
            ScalarType::I32,
        ));
    }
    // i32 → u32.
    for &a in &[0i32, 1i32, -1i32, i32::MIN, i32::MAX, 42i32, -42i32] {
        out.push(case_cast(
            RawValues::I32(vec![a]),
            RawValues::U32(vec![host_cast_i32_to_u32(a)]),
            ScalarType::I32,
            ScalarType::U32,
        ));
    }

    // u32 → f32 (round to nearest).
    for &a in &[0u32, 1u32, 0xFFFFFFFFu32, 0x80000000u32] {
        out.push(case_cast(
            RawValues::U32(vec![a]),
            RawValues::F32(vec![host_cast_u32_to_f32(a)]),
            ScalarType::U32,
            ScalarType::F32,
        ));
    }
    // f32 → u32 (truncate toward zero; saturate on overflow is
    // platform-defined, so skip out-of-range inputs).
    for &a in &[0.0f32, 1.0f32, 42.5f32, 4294967040.0f32 /* in-range */] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::U32(vec![host_cast_f32_to_u32(a)]),
            ScalarType::F32,
            ScalarType::U32,
        ));
    }

    // i32 → f32 and f32 → i32 (in-range only).
    for &a in &[
        0i32,
        1i32,
        -1i32,
        42i32,
        -42i32,
        1_000_000i32,
        -1_000_000i32,
    ] {
        out.push(case_cast(
            RawValues::I32(vec![a]),
            RawValues::F32(vec![host_cast_i32_to_f32(a)]),
            ScalarType::I32,
            ScalarType::F32,
        ));
    }
    for &a in &[0.0f32, 1.5f32, -1.5f32, 42.0f32, -42.0f32] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::I32(vec![host_cast_f32_to_i32(a)]),
            ScalarType::F32,
            ScalarType::I32,
        ));
    }

    out
}

// ── Narrow-int cast lane ─────────────────────────────────────────────
//
// The astype matrix rows involving a narrow type: narrow → wide
// (zero-extend unsigned sources, sign-extend signed sources), wide →
// narrow (truncate mod 2^w), narrow ↔ narrow (same-width bit-pattern
// reinterpret; cross-width extend-then-truncate), and float ↔ narrow
// (exact widening; in-range truncate-toward-zero the other way).
// Host oracle = the Rust `as` conversion, which matches the CPU
// reference's mask-then-extend + truncate-at-store pipeline on every
// pair generated here.
//
// Deliberately absent, inherited from the wide rows' conventions:
//
//   - Out-of-range / NaN float → int: non-portable per the existing
//     f32 → u32/i32 contract (the reference saturates, the native
//     backends do their native conversion). In-range inputs only.
//   - Negative signed-narrow → u64: the reference zero-extends from
//     the *source* width (i8 −1 → 255), while MSL's C conversion
//     sign-extends to the full 64 bits (measured: i8 −1 → 2^64 − 1)
//     — numpy and the scope contract side with MSL. The reference's
//     `eval_cast` u64 arm masks to the source width for narrow *and*
//     i32 sources alike, so this is a pre-existing wide-int gap
//     surfacing at narrow width, not a narrow regression; rows land
//     when the u64-extension contract is settled. Non-negative
//     signed sources agree on every backend and are pinned below.
//     Signed-narrow → i64 sign-extends identically everywhere and
//     is pinned with negative values.

fn cases_cast_narrow() -> Vec<OpCase> {
    let mut out = Vec::new();

    // u8 → everything.
    for &a in &[0u8, 1, 0x7F, 0x80, 0xFF] {
        let v =
            |x: RawValues, to: ScalarType| case_cast(RawValues::U8(vec![a]), x, ScalarType::U8, to);
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
        out.push(v(RawValues::U32(vec![a as u32]), ScalarType::U32));
        out.push(v(RawValues::I32(vec![a as i32]), ScalarType::I32));
        out.push(v(RawValues::U64(vec![a as u64]), ScalarType::U64));
        out.push(v(RawValues::I64(vec![a as i64]), ScalarType::I64));
        out.push(v(RawValues::F32(vec![a as f32]), ScalarType::F32));
        out.push(v(RawValues::F64(vec![a as f64]), ScalarType::F64));
    }

    // i8 → everything (u64 targets: non-negative sources only — see
    // the block comment).
    for &a in &[0i8, 1, -1, i8::MIN, i8::MAX] {
        let v =
            |x: RawValues, to: ScalarType| case_cast(RawValues::I8(vec![a]), x, ScalarType::I8, to);
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
        out.push(v(RawValues::U32(vec![a as u32]), ScalarType::U32));
        out.push(v(RawValues::I32(vec![a as i32]), ScalarType::I32));
        out.push(v(RawValues::I64(vec![a as i64]), ScalarType::I64));
        out.push(v(RawValues::F32(vec![a as f32]), ScalarType::F32));
        out.push(v(RawValues::F64(vec![a as f64]), ScalarType::F64));
        if a >= 0 {
            out.push(v(RawValues::U64(vec![a as u64]), ScalarType::U64));
        }
    }

    // u16 → everything.
    for &a in &[0u16, 1, 0x7FFF, 0x8000, 0xFFFF] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::U16(vec![a]), x, ScalarType::U16, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
        out.push(v(RawValues::U32(vec![a as u32]), ScalarType::U32));
        out.push(v(RawValues::I32(vec![a as i32]), ScalarType::I32));
        out.push(v(RawValues::U64(vec![a as u64]), ScalarType::U64));
        out.push(v(RawValues::I64(vec![a as i64]), ScalarType::I64));
        out.push(v(RawValues::F32(vec![a as f32]), ScalarType::F32));
        out.push(v(RawValues::F64(vec![a as f64]), ScalarType::F64));
    }

    // i16 → everything (u64 targets: non-negative sources only).
    for &a in &[0i16, 1, -1, i16::MIN, i16::MAX] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::I16(vec![a]), x, ScalarType::I16, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::U32(vec![a as u32]), ScalarType::U32));
        out.push(v(RawValues::I32(vec![a as i32]), ScalarType::I32));
        out.push(v(RawValues::I64(vec![a as i64]), ScalarType::I64));
        out.push(v(RawValues::F32(vec![a as f32]), ScalarType::F32));
        out.push(v(RawValues::F64(vec![a as f64]), ScalarType::F64));
        if a >= 0 {
            out.push(v(RawValues::U64(vec![a as u64]), ScalarType::U64));
        }
    }

    // Wide ints → narrow: truncate mod 2^w. The value list crosses
    // every narrow boundary (sign bytes, full bytes, both halves of
    // the halfword) so a backend extending instead of truncating —
    // or truncating at the wrong width — diverges.
    for &a in &[0u32, 1, 0x80, 0xFF, 0xABCD, 0x8000, 0x12345678, 0xFFFFFFFF] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::U32(vec![a]), x, ScalarType::U32, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
    }
    for &a in &[
        0i32,
        1,
        -1,
        255,
        -128,
        32_767,
        -32_768,
        65_535,
        i32::MIN,
        i32::MAX,
    ] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::I32(vec![a]), x, ScalarType::I32, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
    }
    for &a in &[0u64, 1, 0xFF, 0xFFFF, 0x1234_5678_9ABC_DEF0, u64::MAX] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::U64(vec![a]), x, ScalarType::U64, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
    }
    for &a in &[0i64, 1, -1, 255, -32_768, i64::MIN, i64::MAX] {
        let v = |x: RawValues, to: ScalarType| {
            case_cast(RawValues::I64(vec![a]), x, ScalarType::I64, to)
        };
        out.push(v(RawValues::U8(vec![a as u8]), ScalarType::U8));
        out.push(v(RawValues::I8(vec![a as i8]), ScalarType::I8));
        out.push(v(RawValues::U16(vec![a as u16]), ScalarType::U16));
        out.push(v(RawValues::I16(vec![a as i16]), ScalarType::I16));
    }

    // Float → narrow: truncate toward zero, in-range inputs only
    // (the inherited non-contract excludes out-of-range/NaN). The
    // negative fractional values pin trunc-toward-zero on signed
    // targets; the type MAX/MIN values pin the range endpoints (all
    // exactly representable in f32 at these widths).
    for &a in &[0.0f32, 1.0, 42.5, 200.9, 255.0] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::U8(vec![a as u8]),
            ScalarType::F32,
            ScalarType::U8,
        ));
    }
    for &a in &[0.0f32, 1.5, -1.5, -42.7, 127.0, -128.0] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::I8(vec![a as i8]),
            ScalarType::F32,
            ScalarType::I8,
        ));
    }
    for &a in &[0.0f32, 1.0, 42.5, 60_000.9, 65_535.0] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::U16(vec![a as u16]),
            ScalarType::F32,
            ScalarType::U16,
        ));
    }
    for &a in &[0.0f32, 1.5, -1.5, -3000.7, 32_767.0, -32_768.0] {
        out.push(case_cast(
            RawValues::F32(vec![a]),
            RawValues::I16(vec![a as i16]),
            ScalarType::F32,
            ScalarType::I16,
        ));
    }
    for &a in &[0.0f64, 1.5, 200.9, 255.0] {
        out.push(case_cast(
            RawValues::F64(vec![a]),
            RawValues::U8(vec![a as u8]),
            ScalarType::F64,
            ScalarType::U8,
        ));
    }
    for &a in &[0.0f64, 1.5, -42.7, -128.0, 127.0] {
        out.push(case_cast(
            RawValues::F64(vec![a]),
            RawValues::I8(vec![a as i8]),
            ScalarType::F64,
            ScalarType::I8,
        ));
    }
    for &a in &[0.0f64, 1.5, 60_000.9, 65_535.0] {
        out.push(case_cast(
            RawValues::F64(vec![a]),
            RawValues::U16(vec![a as u16]),
            ScalarType::F64,
            ScalarType::U16,
        ));
    }
    for &a in &[0.0f64, 1.5, -3000.7, -32_768.0, 32_767.0] {
        out.push(case_cast(
            RawValues::F64(vec![a]),
            RawValues::I16(vec![a as i16]),
            ScalarType::F64,
            ScalarType::I16,
        ));
    }

    out
}

// ── Const cases ──────────────────────────────────────────────────────
//
// Exercises the `KernelOp::Const` emit path, which the BinOp cases
// above never touch (they load both operands from buffers). The
// `85551fa` float-const bug rode this path: small constants like
// `1.0f32 / (1 << 24)` were emitted as the literal string
// "0.000000" by the MSL emitter's `{:.6}` format, silently
// collapsing every kernel using such a scaling factor.

fn case_const_f32_mul(name: &str, a: f32, c: f32, expected: f32) -> OpCase {
    OpCase {
        name: format!("{}_const_f32_mul_{}_a{:e}", NAME_PREFIX, name, a),
        def: build_const_binop_def(name, ScalarType::F32, BinOp::Mul, ConstValue::F32(c)),
        input_a: RawValues::F32(vec![a]),
        input_b: RawValues::F32(vec![a]),
        expected: RawValues::F32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn case_const_u32_add(name: &str, a: u32, c: u32, expected: u32) -> OpCase {
    OpCase {
        name: format!("{}_const_u32_add_{}_a{:#010x}", NAME_PREFIX, name, a),
        def: build_const_binop_def(name, ScalarType::U32, BinOp::Add, ConstValue::U32(c)),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![a]),
        expected: RawValues::U32(vec![expected]),
        max_ulps: 0,
        skip_on_metal: false,
    }
}

fn cases_const() -> Vec<OpCase> {
    let mut out = Vec::new();

    // The exact bug case from 85551fa: scale by 1.0 / 2^24.
    // `(0.5_f32) * (1.0_f32 / (1 << 24)_f32) = 2.9802322e-8`.
    let small = 1.0f32 / (1u32 << 24) as f32;
    out.push(case_const_f32_mul(
        "scale_24bit",
        0.5f32,
        small,
        0.5f32 * small,
    ));

    // A few neighbouring magnitudes that the `{:.6}` formatter would
    // also silently round.
    out.push(case_const_f32_mul(
        "scale_1e8",
        4_294_967_295.0f32, // ~2^32
        1.0e-8f32,
        4_294_967_295.0f32 * 1.0e-8f32,
    ));
    out.push(case_const_f32_mul(
        "scale_1e7",
        1000.0f32,
        1.0e-7f32,
        1000.0f32 * 1.0e-7f32,
    ));

    // A small u32 const case for symmetry. Doesn't exercise a known
    // bug today but cheap insurance against analogous regressions in
    // integer const emission.
    out.push(case_const_u32_add(
        "add_42",
        0x12345678u32,
        42u32,
        0x12345678u32.wrapping_add(42),
    ));

    // Narrow-typed BinOp fed by a wide Const register: narrow kernel
    // constants ride ConstValue::U32/I32 (there are no narrow const
    // variants by design — the array kernels' zero/one constants use
    // exactly this shape). The wrap-at-MAX inputs pin that the const
    // operand participates at the narrow op's width.
    out.push(OpCase {
        name: format!("{}_add_u8_const_wrap", NAME_PREFIX),
        def: build_const_binop_def("wrap", ScalarType::U8, BinOp::Add, ConstValue::U32(1)),
        input_a: RawValues::U8(vec![0xFF]),
        input_b: RawValues::U8(vec![0]),
        expected: RawValues::U8(vec![0]),
        max_ulps: 0,
        skip_on_metal: false,
    });
    out.push(OpCase {
        name: format!("{}_add_i8_const_wrap", NAME_PREFIX),
        def: build_const_binop_def("wrap", ScalarType::I8, BinOp::Add, ConstValue::I32(-1)),
        input_a: RawValues::I8(vec![i8::MIN]),
        input_b: RawValues::I8(vec![0]),
        expected: RawValues::I8(vec![i8::MAX]),
        max_ulps: 0,
        skip_on_metal: false,
    });
    out.push(OpCase {
        name: format!("{}_add_u16_const_wrap", NAME_PREFIX),
        def: build_const_binop_def("wrap", ScalarType::U16, BinOp::Add, ConstValue::U32(1)),
        input_a: RawValues::U16(vec![0xFFFF]),
        input_b: RawValues::U16(vec![0]),
        expected: RawValues::U16(vec![0]),
        max_ulps: 0,
        skip_on_metal: false,
    });
    out.push(OpCase {
        name: format!("{}_add_i16_const_wrap", NAME_PREFIX),
        def: build_const_binop_def("wrap", ScalarType::I16, BinOp::Add, ConstValue::I32(-1)),
        input_a: RawValues::I16(vec![i16::MIN]),
        input_b: RawValues::I16(vec![0]),
        expected: RawValues::I16(vec![i16::MAX]),
        max_ulps: 0,
        skip_on_metal: false,
    });

    // Composed-op case: unsigned shift of an int-typed register.
    // Exercises the `06e764c` shift sign-extension path. Without
    // the operand-cast fix in the MSL emitter, `(i32)0x80000000 >>
    // 8` arithmetic-shifts to 0xFF800000 (sign-extended), which
    // then assigns to uint — wrong. With the fix, the operands are
    // cast to uint inside the emit so the shift is logical.
    let a = 0x80000000u32;
    let expected_after_shift = a >> 8; // 0x00800000
    out.push(OpCase {
        name: format!("{}_shr_after_signed_a{:#010x}", NAME_PREFIX, a),
        def: build_shr_after_signed_def(),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![0]),
        expected: RawValues::U32(vec![expected_after_shift]),
        max_ulps: 0,
        skip_on_metal: false,
    });

    out
}

// ── Narrow stride guard ──────────────────────────────────────────────
//
// The recorded trap: a backend binding a narrow field at the wrong
// element width still reads element 0 correctly — only index > 0
// can expose it, and only a long ramp makes every misread visible.
// Each narrow type gets a 256-element elementwise case (every quark
// loads and stores at its own index) and a 256-element cast case
// whose input and output strides differ, the sharpest version of
// the guard. The u8 ramps cover the full byte space; the 16-bit
// ramps step by 257 so both bytes of every element vary.

fn cases_stride_narrow() -> Vec<OpCase> {
    let mut out = Vec::new();
    let n = 256u32;

    let a8: Vec<u8> = (0..n).map(|i| i as u8).collect();
    let b8: Vec<u8> = (0..n)
        .map(|i| (i.wrapping_mul(3).wrapping_add(7)) as u8)
        .collect();
    let e8: Vec<u8> = a8
        .iter()
        .zip(&b8)
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    out.push(OpCase {
        name: format!("{}_add_u8_ramp256", NAME_PREFIX),
        def: build_binop_def(binop_tag(BinOp::Add), ScalarType::U8, BinOp::Add),
        input_a: RawValues::U8(a8.clone()),
        input_b: RawValues::U8(b8),
        expected: RawValues::U8(e8),
        max_ulps: 0,
        skip_on_metal: false,
    });
    // u8 → u16 sign-extension-free widening ramp: 1-byte input
    // stride against 2-byte output stride.
    out.push(case_cast(
        RawValues::U8(a8.clone()),
        RawValues::U16(a8.iter().map(|&x| x as u16).collect()),
        ScalarType::U8,
        ScalarType::U16,
    ));

    let ai8: Vec<i8> = (0..n).map(|i| (i as u8) as i8).collect();
    let bi8: Vec<i8> = (0..n)
        .map(|i| ((i.wrapping_mul(5).wrapping_add(3)) as u8) as i8)
        .collect();
    let ei8: Vec<i8> = ai8
        .iter()
        .zip(&bi8)
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    out.push(OpCase {
        name: format!("{}_add_i8_ramp256", NAME_PREFIX),
        def: build_binop_def(binop_tag(BinOp::Add), ScalarType::I8, BinOp::Add),
        input_a: RawValues::I8(ai8.clone()),
        input_b: RawValues::I8(bi8),
        expected: RawValues::I8(ei8),
        max_ulps: 0,
        skip_on_metal: false,
    });
    // i8 → i16 ramp doubles as an exhaustive sign-extension sweep of
    // the whole byte space.
    out.push(case_cast(
        RawValues::I8(ai8.clone()),
        RawValues::I16(ai8.iter().map(|&x| x as i16).collect()),
        ScalarType::I8,
        ScalarType::I16,
    ));

    let a16: Vec<u16> = (0..n).map(|i| (i * 257) as u16).collect();
    let b16: Vec<u16> = (0..n)
        .map(|i| (i.wrapping_mul(101).wrapping_add(3)) as u16)
        .collect();
    let e16: Vec<u16> = a16
        .iter()
        .zip(&b16)
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    out.push(OpCase {
        name: format!("{}_add_u16_ramp256", NAME_PREFIX),
        def: build_binop_def(binop_tag(BinOp::Add), ScalarType::U16, BinOp::Add),
        input_a: RawValues::U16(a16.clone()),
        input_b: RawValues::U16(b16),
        expected: RawValues::U16(e16),
        max_ulps: 0,
        skip_on_metal: false,
    });
    // u16 → u8 narrowing ramp: 2-byte input stride against 1-byte
    // output stride.
    out.push(case_cast(
        RawValues::U16(a16.clone()),
        RawValues::U8(a16.iter().map(|&x| x as u8).collect()),
        ScalarType::U16,
        ScalarType::U8,
    ));

    let ai16: Vec<i16> = (0..n).map(|i| ((i * 257) as u16) as i16).collect();
    let bi16: Vec<i16> = (0..n)
        .map(|i| ((i.wrapping_mul(89).wrapping_add(11)) as u16) as i16)
        .collect();
    let ei16: Vec<i16> = ai16
        .iter()
        .zip(&bi16)
        .map(|(&x, &y)| x.wrapping_add(y))
        .collect();
    out.push(OpCase {
        name: format!("{}_add_i16_ramp256", NAME_PREFIX),
        def: build_binop_def(binop_tag(BinOp::Add), ScalarType::I16, BinOp::Add),
        input_a: RawValues::I16(ai16.clone()),
        input_b: RawValues::I16(bi16),
        expected: RawValues::I16(ei16),
        max_ulps: 0,
        skip_on_metal: false,
    });
    out.push(case_cast(
        RawValues::I16(ai16.clone()),
        RawValues::I8(ai16.iter().map(|&x| x as i8).collect()),
        ScalarType::I16,
        ScalarType::I8,
    ));

    out
}

/// All BinOp + UnaryOp + Cmp + Cast + Const cases. Order: int BinOp
/// (wide then narrow), float BinOp, unary, cmp, cast, const, then
/// the narrow stride ramps.
pub fn cases() -> Vec<OpCase> {
    let mut all = Vec::new();
    all.extend(cases_u32());
    all.extend(cases_u64());
    all.extend(cases_i32());
    all.extend(cases_i64());
    all.extend(cases_u8());
    all.extend(cases_i8());
    all.extend(cases_u16());
    all.extend(cases_i16());
    all.extend(cases_f32());
    all.extend(cases_f64());
    all.extend(cases_bf16());
    all.extend(cases_fp8());
    all.extend(cases_quant());
    all.extend(cases_unary());
    all.extend(cases_unary_narrow());
    all.extend(cases_cmp());
    all.extend(cases_cmp_narrow());
    all.extend(cases_cast());
    all.extend(cases_cast_narrow());
    all.extend(cases_const());
    all.extend(cases_stride_narrow());
    all
}

//! Per-op differential matrix — step 082 Layer 1.
//!
//! For every `(BinOp, ScalarType, edge-input)` triple the matrix
//! generates a minimal kernel that performs the op on two inputs
//! and writes the result. The kernel is dispatched on every
//! available backend lane; results are compared bit-exact against
//! the CPU reference. This catches the class of silent miscompile
//! bug fixed in `85551fa` (float-const format) and `06e764c`
//! (shift sign-extension): the existing differential CI (saxpy,
//! reduce_sum, counter, race) is too coarse to exercise the
//! per-op signedness matrix.
//!
//! Layout: each case is one quark, two scalar inputs as fields,
//! one scalar output as a field, no push constants. Inputs are
//! materialised as Vec<T> of length 1 so we can reuse the existing
//! Field<T>-based dispatch path without push-const plumbing.

use super::lane::Lane;
use super::output::{RawOutput, RawValues};

use quanta_ir::{BinOp, CmpOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType, UnaryOp};

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

impl OpCase {
    /// Build the `RawOutput` produced by dispatching this case on
    /// the given lane. Caller hands the actual lane buffer.
    pub fn output(&self, lane: Lane, values: RawValues) -> RawOutput {
        RawOutput {
            lane,
            kernel: Box::leak(self.name.clone().into_boxed_str()),
            values,
        }
    }

    /// CPU-computed expected output, packaged as a Reference
    /// RawOutput for the comparator.
    pub fn oracle(&self) -> RawOutput {
        RawOutput {
            lane: Lane::Reference,
            kernel: Box::leak(self.name.clone().into_boxed_str()),
            values: self.expected.clone(),
        }
    }
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

/// All BinOp + UnaryOp + Cmp + Cast cases. Order: int BinOp, float
/// BinOp, unary, cmp, cast.
pub fn cases() -> Vec<OpCase> {
    let mut all = Vec::new();
    all.extend(cases_u32());
    all.extend(cases_u64());
    all.extend(cases_i32());
    all.extend(cases_i64());
    all.extend(cases_f32());
    all.extend(cases_f64());
    all.extend(cases_unary());
    all.extend(cases_cmp());
    all.extend(cases_cast());
    all
}

// ── Per-lane dispatcher ──────────────────────────────────────────────

/// Dispatch one case on the given Gpu, return the raw output buffer.
///
/// Bind layout matches `build_binop_def`: slot 0 = a, slot 1 = b,
/// slot 2 = out. All three are length-1 typed fields.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
pub fn dispatch_on(gpu: &quanta::Gpu, case: &OpCase, lane: Lane) -> RawOutput {
    let bytes = quanta_ir::serialize_kernel(&case.def);
    let mut wave = gpu.wave_jit(&bytes).expect("wave_jit");

    // The dispatcher picks `Field<T>` allocations from the input
    // RawValues variants, and the output `Field<U>` from the
    // expected variant. Cmp produces U32 from any input type
    // (Bool→U32 cast inside the kernel); Cast produces target-type
    // from source-type.
    let values = dispatch_pair_typed(gpu, &mut wave, &case.input_a, &case.input_b, &case.expected);

    case.output(lane, values)
}

/// Match on (input_a, input_b, expected) variant triples and pick
/// the right typed allocation for each field. The four scalar
/// widths × six RawValues variants × asymmetric in/out types gives
/// a 36-arm match in principle; this enumerates only the (in_pair,
/// out) combinations we actually use today and panics on the rest.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_pair_typed(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    in_a: &RawValues,
    in_b: &RawValues,
    expected: &RawValues,
) -> RawValues {
    match (in_a, in_b, expected) {
        // Symmetric: input and output share the type (BinOp, UnaryOp).
        (RawValues::U32(a), RawValues::U32(b), RawValues::U32(_)) => {
            dispatch_pair::<u32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        (RawValues::U64(a), RawValues::U64(b), RawValues::U64(_)) => {
            dispatch_pair::<u64, u64>(gpu, wave, a, b, RawValues::U64)
        }
        (RawValues::I32(a), RawValues::I32(b), RawValues::I32(_)) => {
            dispatch_pair::<i32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::I64(a), RawValues::I64(b), RawValues::I64(_)) => {
            dispatch_pair::<i64, i64>(gpu, wave, a, b, RawValues::I64)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::F32(_)) => {
            dispatch_pair::<f32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        (RawValues::F64(a), RawValues::F64(b), RawValues::F64(_)) => {
            dispatch_pair::<f64, f64>(gpu, wave, a, b, RawValues::F64)
        }
        // Cmp with non-U32 inputs: produces a U32 (0/1) output via
        // Cast(Bool→U32) in the kernel body. The U32-input variant
        // is handled by the symmetric arm above.
        (RawValues::I32(a), RawValues::I32(b), RawValues::U32(_)) => {
            dispatch_pair::<i32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::U32(_)) => {
            dispatch_pair::<f32, u32>(gpu, wave, a, b, RawValues::U32)
        }
        // Cast across types: a∈From, b unused, out∈To. The dispatcher
        // still allocates a `Field<From>` for b because the kernel
        // emits a `Load` from slot 1 even though the result is dead.
        (RawValues::U32(a), RawValues::U32(b), RawValues::I32(_)) => {
            dispatch_pair::<u32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::U32(a), RawValues::U32(b), RawValues::F32(_)) => {
            dispatch_pair::<u32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        (RawValues::F32(a), RawValues::F32(b), RawValues::I32(_)) => {
            dispatch_pair::<f32, i32>(gpu, wave, a, b, RawValues::I32)
        }
        (RawValues::I32(a), RawValues::I32(b), RawValues::F32(_)) => {
            dispatch_pair::<i32, f32>(gpu, wave, a, b, RawValues::F32)
        }
        _ => panic!(
            "op_matrix::read_output: in/out type combo not yet wired \
             (a={}, b={}, out={})",
            in_a.type_tag(),
            in_b.type_tag(),
            expected.type_tag()
        ),
    }
}

/// Allocate `Field<TIn>` × 2 + `Field<TOut>` × 1, upload, bind,
/// dispatch one quark, read back as `Vec<TOut>`. Caller picks the
/// `RawValues` wrapper for the output variant.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_pair<TIn: Copy + 'static, TOut: Copy + 'static>(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[TIn],
    b: &[TIn],
    wrap: fn(Vec<TOut>) -> RawValues,
) -> RawValues {
    let fa = gpu.field::<TIn>(1).unwrap();
    let fb = gpu.field::<TIn>(1).unwrap();
    let fout = gpu.field::<TOut>(1).unwrap();
    fa.write(a).unwrap();
    fb.write(b).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(wave, 1).unwrap();
    pulse.wait().unwrap();
    wrap(fout.read().unwrap())
}

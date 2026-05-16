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

use quanta_ir::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

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

/// All BinOp cases. Order: u32, u64, i32, i64, f32, f64.
pub fn cases() -> Vec<OpCase> {
    let mut all = Vec::new();
    all.extend(cases_u32());
    all.extend(cases_u64());
    all.extend(cases_i32());
    all.extend(cases_i64());
    all.extend(cases_f32());
    all.extend(cases_f64());
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

    let values = match (&case.input_a, &case.input_b) {
        (RawValues::U32(a), RawValues::U32(b)) => {
            dispatch_scalar::<u32>(gpu, &mut wave, a, b, RawValues::U32)
        }
        (RawValues::U64(a), RawValues::U64(b)) => {
            dispatch_scalar::<u64>(gpu, &mut wave, a, b, RawValues::U64)
        }
        (RawValues::I32(a), RawValues::I32(b)) => {
            dispatch_scalar::<i32>(gpu, &mut wave, a, b, RawValues::I32)
        }
        (RawValues::I64(a), RawValues::I64(b)) => {
            dispatch_scalar::<i64>(gpu, &mut wave, a, b, RawValues::I64)
        }
        (RawValues::F32(a), RawValues::F32(b)) => {
            dispatch_scalar::<f32>(gpu, &mut wave, a, b, RawValues::F32)
        }
        (RawValues::F64(a), RawValues::F64(b)) => {
            dispatch_scalar::<f64>(gpu, &mut wave, a, b, RawValues::F64)
        }
        _ => panic!(
            "op_matrix::dispatch_on: input type pair not yet wired (a={}, b={})",
            case.input_a.type_tag(),
            case.input_b.type_tag()
        ),
    };

    case.output(lane, values)
}

/// Allocate two read fields and one write field of type `T`,
/// upload inputs, bind, dispatch one quark, read back. Caller
/// re-wraps the `Vec<T>` in the matching `RawValues` variant.
#[cfg(any(feature = "software", feature = "metal", feature = "vulkan"))]
fn dispatch_scalar<T: Copy + 'static>(
    gpu: &quanta::Gpu,
    wave: &mut quanta::Wave,
    a: &[T],
    b: &[T],
    wrap: fn(Vec<T>) -> RawValues,
) -> RawValues {
    let fa = gpu.field::<T>(1).unwrap();
    let fb = gpu.field::<T>(1).unwrap();
    let fout = gpu.field::<T>(1).unwrap();
    fa.write(a).unwrap();
    fb.write(b).unwrap();
    wave.bind(0, &fa);
    wave.bind(1, &fb);
    wave.bind(2, &fout);
    let mut pulse = gpu.dispatch(wave, 1).unwrap();
    pulse.wait().unwrap();
    wrap(fout.read().unwrap())
}

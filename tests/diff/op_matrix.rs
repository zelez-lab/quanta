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
#[derive(Clone, Debug)]
pub struct OpCase {
    pub name: String,
    pub def: KernelDef,
    pub input_a: RawValues,
    pub input_b: RawValues,
    pub expected: RawValues,
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

/// One U32 `Shr` case — exercises the exact path of the bug fixed
/// in `06e764c`: `0x80000000u32 >> 8` must yield `0x00800000`
/// (logical shift), not `0xFF800000` (arithmetic shift).
pub fn u32_shr_sign_bit() -> OpCase {
    let a = 0x80000000u32;
    let b = 8u32;
    OpCase {
        name: format!(
            "{}_{}_{}",
            NAME_PREFIX,
            binop_tag(BinOp::Shr),
            scalar_tag(ScalarType::U32)
        ),
        def: build_binop_def(binop_tag(BinOp::Shr), ScalarType::U32, BinOp::Shr),
        input_a: RawValues::U32(vec![a]),
        input_b: RawValues::U32(vec![b]),
        expected: RawValues::U32(vec![a >> b]),
    }
}

/// Initial smoke-set: just the one shift case for now. Expanded by
/// later tasks in step 082.
pub fn cases() -> Vec<OpCase> {
    vec![u32_shr_sign_bit()]
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
            let fa = gpu.field::<u32>(1).unwrap();
            let fb = gpu.field::<u32>(1).unwrap();
            let fout = gpu.field::<u32>(1).unwrap();
            fa.write(a).unwrap();
            fb.write(b).unwrap();
            wave.bind(0, &fa);
            wave.bind(1, &fb);
            wave.bind(2, &fout);
            let mut pulse = gpu.dispatch(&wave, 1).unwrap();
            pulse.wait().unwrap();
            RawValues::U32(fout.read().unwrap())
        }
        _ => panic!(
            "op_matrix::dispatch_on: input type pair not yet wired (a={}, b={})",
            case.input_a.type_tag(),
            case.input_b.type_tag()
        ),
    };

    case.output(lane, values)
}

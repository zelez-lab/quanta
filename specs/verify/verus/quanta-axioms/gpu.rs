//! Verus mirror of Lean axioms: `specs/verify/lean/Quanta/Axioms/Gpu.lean`
//!
//! GPU execution model axioms (A3). These are the trusted computing base:
//! SPIR-V instruction semantics, opcode numbers from the SPIR-V 1.6 spec,
//! and the link between opcode and semantic operation.
//!
//! Every constant here is ground truth from the SPIR-V 1.6 specification.
//! Emitter proofs reference these axioms instead of bare numeric literals,
//! so the proof chain is:
//!   axiom (SPIR-V spec) -> emitter mapping -> emitter output -> correctness

use vstd::prelude::*;

verus! {

// ════════════════════════════════════════════════════════════════════════
// A3: SPIR-V instruction semantics (wrapping arithmetic on u32)
// ════════════════════════════════════════════════════════════════════════

/// OpIAdd (128): wrapping integer addition.
pub open spec fn spirv_iadd(a: u32, b: u32) -> u32 {
    a.wrapping_add(b)
}

/// OpISub (130): wrapping integer subtraction.
pub open spec fn spirv_isub(a: u32, b: u32) -> u32 {
    a.wrapping_sub(b)
}

/// OpIMul (132): wrapping integer multiplication.
pub open spec fn spirv_imul(a: u32, b: u32) -> u32 {
    a.wrapping_mul(b)
}

/// OpUDiv (134): unsigned integer division. Division by zero is
/// undefined in SPIR-V; our CPU executor returns 0.
pub open spec fn spirv_udiv(a: u32, b: u32) -> u32 {
    if b == 0u32 { 0u32 } else { a / b }
}

/// OpBitwiseAnd (199): bitwise AND.
pub open spec fn spirv_bitwise_and(a: u32, b: u32) -> u32 {
    a & b
}

/// OpBitwiseOr (197): bitwise OR.
pub open spec fn spirv_bitwise_or(a: u32, b: u32) -> u32 {
    a | b
}

/// OpBitwiseXor (198): bitwise XOR.
pub open spec fn spirv_bitwise_xor(a: u32, b: u32) -> u32 {
    a ^ b
}

/// OpShiftLeftLogical (196): logical left shift.
pub open spec fn spirv_shift_left(a: u32, b: u32) -> u32 {
    if b >= 32u32 { 0u32 } else { a << b }
}

/// OpShiftRightLogical (194): logical right shift.
pub open spec fn spirv_shift_right_logical(a: u32, b: u32) -> u32 {
    if b >= 32u32 { 0u32 } else { a >> b }
}

// ════════════════════════════════════════════════════════════════════════
// A3: SPIR-V opcode numbers (ground truth from SPIR-V 1.6 spec)
// ════════════════════════════════════════════════════════════════════════

// -- Arithmetic opcodes --

pub open spec fn OPCODE_SNEGATE() -> u16 { 126u16 }
pub open spec fn OPCODE_FNEGATE() -> u16 { 127u16 }
pub open spec fn OPCODE_IADD() -> u16 { 128u16 }
pub open spec fn OPCODE_FADD() -> u16 { 129u16 }
pub open spec fn OPCODE_ISUB() -> u16 { 130u16 }
pub open spec fn OPCODE_FSUB() -> u16 { 131u16 }
pub open spec fn OPCODE_IMUL() -> u16 { 132u16 }
pub open spec fn OPCODE_FMUL() -> u16 { 133u16 }
pub open spec fn OPCODE_UDIV() -> u16 { 134u16 }
pub open spec fn OPCODE_SDIV() -> u16 { 135u16 }
pub open spec fn OPCODE_FDIV() -> u16 { 136u16 }
pub open spec fn OPCODE_UMOD() -> u16 { 137u16 }
pub open spec fn OPCODE_SMOD() -> u16 { 138u16 }
pub open spec fn OPCODE_FREM() -> u16 { 140u16 }

// -- Comparison opcodes --

pub open spec fn OPCODE_LOGICAL_NOT() -> u16 { 168u16 }
pub open spec fn OPCODE_IEQUAL() -> u16 { 170u16 }
pub open spec fn OPCODE_INOTEQUAL() -> u16 { 171u16 }
pub open spec fn OPCODE_UGREATERTHAN() -> u16 { 172u16 }
pub open spec fn OPCODE_SGREATERTHAN() -> u16 { 173u16 }
pub open spec fn OPCODE_UGREATERTHANEQUAL() -> u16 { 174u16 }
pub open spec fn OPCODE_SGREATERTHANEQUAL() -> u16 { 175u16 }
pub open spec fn OPCODE_ULESSTHAN() -> u16 { 176u16 }
pub open spec fn OPCODE_SLESSTHAN() -> u16 { 177u16 }
pub open spec fn OPCODE_ULESSTHANEQUAL() -> u16 { 178u16 }
pub open spec fn OPCODE_SLESSTHANEQUAL() -> u16 { 179u16 }
pub open spec fn OPCODE_FORDEQUAL() -> u16 { 180u16 }
pub open spec fn OPCODE_FORDNOTEQUAL() -> u16 { 182u16 }
pub open spec fn OPCODE_FORDLESSTHAN() -> u16 { 184u16 }
pub open spec fn OPCODE_FORDLESSTHANEQUAL() -> u16 { 186u16 }
pub open spec fn OPCODE_FORDGREATERTHAN() -> u16 { 188u16 }
pub open spec fn OPCODE_FORDGREATERTHANEQUAL() -> u16 { 190u16 }

// -- Shift and bitwise opcodes --

pub open spec fn OPCODE_SHIFT_RIGHT_LOGICAL() -> u16 { 194u16 }
pub open spec fn OPCODE_SHIFT_RIGHT_ARITHMETIC() -> u16 { 195u16 }
pub open spec fn OPCODE_SHIFT_LEFT_LOGICAL() -> u16 { 196u16 }
pub open spec fn OPCODE_BITWISE_OR() -> u16 { 197u16 }
pub open spec fn OPCODE_BITWISE_XOR() -> u16 { 198u16 }
pub open spec fn OPCODE_BITWISE_AND() -> u16 { 199u16 }
pub open spec fn OPCODE_NOT() -> u16 { 200u16 }

// -- Convert/cast opcodes --

pub open spec fn OPCODE_CONVERT_FTOS() -> u16 { 110u16 }
pub open spec fn OPCODE_CONVERT_STOF() -> u16 { 111u16 }
pub open spec fn OPCODE_CONVERT_UTOF() -> u16 { 112u16 }
pub open spec fn OPCODE_BITCAST() -> u16 { 124u16 }

// -- Type opcodes --

pub open spec fn OPCODE_TYPE_FLOAT() -> u16 { 22u16 }
pub open spec fn OPCODE_TYPE_VECTOR() -> u16 { 23u16 }

// ════════════════════════════════════════════════════════════════════════
// A3: SpvOp enum -- maps Lean SpvOp to Verus
// ════════════════════════════════════════════════════════════════════════

pub enum SpvOp {
    IAdd,
    FAdd,
    ISub,
    FSub,
    IMul,
    FMul,
    UDiv,
    SDiv,
    FDiv,
    UMod,
    SMod,
    FRem,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeftLogical,
    ShiftRightLogical,
    ShiftRightArithmetic,
}

/// Map SpvOp to its opcode number (SPIR-V 1.6 spec).
pub open spec fn spv_op_opcode(op: SpvOp) -> u16 {
    match op {
        SpvOp::IAdd                => OPCODE_IADD(),
        SpvOp::FAdd                => OPCODE_FADD(),
        SpvOp::ISub                => OPCODE_ISUB(),
        SpvOp::FSub                => OPCODE_FSUB(),
        SpvOp::IMul                => OPCODE_IMUL(),
        SpvOp::FMul                => OPCODE_FMUL(),
        SpvOp::UDiv                => OPCODE_UDIV(),
        SpvOp::SDiv                => OPCODE_SDIV(),
        SpvOp::FDiv                => OPCODE_FDIV(),
        SpvOp::UMod                => OPCODE_UMOD(),
        SpvOp::SMod                => OPCODE_SMOD(),
        SpvOp::FRem                => OPCODE_FREM(),
        SpvOp::BitwiseAnd          => OPCODE_BITWISE_AND(),
        SpvOp::BitwiseOr           => OPCODE_BITWISE_OR(),
        SpvOp::BitwiseXor          => OPCODE_BITWISE_XOR(),
        SpvOp::ShiftLeftLogical    => OPCODE_SHIFT_LEFT_LOGICAL(),
        SpvOp::ShiftRightLogical   => OPCODE_SHIFT_RIGHT_LOGICAL(),
        SpvOp::ShiftRightArithmetic => OPCODE_SHIFT_RIGHT_ARITHMETIC(),
    }
}

/// Evaluate SpvOp on u32 operands (integer ops only; float ops return 0).
/// Mirrors Lean `SpvOp.eval_u32`.
pub open spec fn spv_op_eval_u32(op: SpvOp, a: u32, b: u32) -> u32 {
    match op {
        SpvOp::IAdd       => spirv_iadd(a, b),
        SpvOp::ISub       => spirv_isub(a, b),
        SpvOp::IMul       => spirv_imul(a, b),
        SpvOp::UDiv       => spirv_udiv(a, b),
        SpvOp::BitwiseAnd => spirv_bitwise_and(a, b),
        SpvOp::BitwiseOr  => spirv_bitwise_or(a, b),
        SpvOp::BitwiseXor => spirv_bitwise_xor(a, b),
        SpvOp::ShiftLeftLogical  => spirv_shift_left(a, b),
        SpvOp::ShiftRightLogical => spirv_shift_right_logical(a, b),
        _                 => 0u32, // float/signed ops on u32 = undefined
    }
}

// ════════════════════════════════════════════════════════════════════════
// A3: QBinOp -- user-level IR binary operations
// ════════════════════════════════════════════════════════════════════════

pub enum QBinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
}

/// Map user-level BinOp to the SPIR-V opcode that implements it
/// for unsigned integers. This is what our emitter MUST produce.
/// Mirrors Lean `QBinOp.to_spv_unsigned`.
pub open spec fn qbinop_to_spv_unsigned(op: QBinOp) -> SpvOp {
    match op {
        QBinOp::Add    => SpvOp::IAdd,
        QBinOp::Sub    => SpvOp::ISub,
        QBinOp::Mul    => SpvOp::IMul,
        QBinOp::Div    => SpvOp::UDiv,
        QBinOp::Rem    => SpvOp::UMod,
        QBinOp::BitAnd => SpvOp::BitwiseAnd,
        QBinOp::BitOr  => SpvOp::BitwiseOr,
        QBinOp::BitXor => SpvOp::BitwiseXor,
        QBinOp::Shl    => SpvOp::ShiftLeftLogical,
        QBinOp::Shr    => SpvOp::ShiftRightLogical,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Link proofs: opcode -> semantics
// ════════════════════════════════════════════════════════════════════════

/// OpIAdd (opcode 128) means wrapping add.
proof fn iadd_opcode_means_wrapping_add(a: u32, b: u32)
    ensures
        OPCODE_IADD() == 128u16,
        spirv_iadd(a, b) == a.wrapping_add(b),
{}

/// OpISub (opcode 130) means wrapping sub.
proof fn isub_opcode_means_wrapping_sub(a: u32, b: u32)
    ensures
        OPCODE_ISUB() == 130u16,
        spirv_isub(a, b) == a.wrapping_sub(b),
{}

/// OpIMul (opcode 132) means wrapping mul.
proof fn imul_opcode_means_wrapping_mul(a: u32, b: u32)
    ensures
        OPCODE_IMUL() == 132u16,
        spirv_imul(a, b) == a.wrapping_mul(b),
{}

/// OpBitwiseAnd (opcode 199) means bitwise AND.
proof fn bitwise_and_opcode_means_and(a: u32, b: u32)
    ensures
        OPCODE_BITWISE_AND() == 199u16,
        spirv_bitwise_and(a, b) == (a & b),
{}

/// OpBitwiseOr (opcode 197) means bitwise OR.
proof fn bitwise_or_opcode_means_or(a: u32, b: u32)
    ensures
        OPCODE_BITWISE_OR() == 197u16,
        spirv_bitwise_or(a, b) == (a | b),
{}

/// OpBitwiseXor (opcode 198) means bitwise XOR.
proof fn bitwise_xor_opcode_means_xor(a: u32, b: u32)
    ensures
        OPCODE_BITWISE_XOR() == 198u16,
        spirv_bitwise_xor(a, b) == (a ^ b),
{}

// ════════════════════════════════════════════════════════════════════════
// End-to-end theorems (mirrors Lean)
// ════════════════════════════════════════════════════════════════════════

/// User writes `a + b` -> GPU computes wrapping addition.
proof fn user_add_is_wrapping_add(a: u32, b: u32)
    ensures spv_op_eval_u32(qbinop_to_spv_unsigned(QBinOp::Add), a, b) == spirv_iadd(a, b),
{}

/// User writes `a - b` -> GPU computes wrapping subtraction.
proof fn user_sub_is_wrapping_sub(a: u32, b: u32)
    ensures spv_op_eval_u32(qbinop_to_spv_unsigned(QBinOp::Sub), a, b) == spirv_isub(a, b),
{}

/// User writes `a & b` -> GPU computes bitwise AND.
proof fn user_bitand_is_bitwise_and(a: u32, b: u32)
    ensures spv_op_eval_u32(qbinop_to_spv_unsigned(QBinOp::BitAnd), a, b) == spirv_bitwise_and(a, b),
{}

/// User writes `a | b` -> GPU computes bitwise OR.
proof fn user_bitor_is_bitwise_or(a: u32, b: u32)
    ensures spv_op_eval_u32(qbinop_to_spv_unsigned(QBinOp::BitOr), a, b) == spirv_bitwise_or(a, b),
{}

/// User writes `a ^ b` -> GPU computes bitwise XOR.
proof fn user_bitxor_is_bitwise_xor(a: u32, b: u32)
    ensures spv_op_eval_u32(qbinop_to_spv_unsigned(QBinOp::BitXor), a, b) == spirv_bitwise_xor(a, b),
{}

// ════════════════════════════════════════════════════════════════════════
// Opcode grounding: emitter produces the axiom-specified opcode
// ════════════════════════════════════════════════════════════════════════

/// BitAnd -> opcode 199, grounded in OPCODE_BITWISE_AND.
proof fn bitand_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::BitAnd)) == OPCODE_BITWISE_AND(),
{}

/// BitOr -> opcode 197, grounded in OPCODE_BITWISE_OR.
proof fn bitor_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::BitOr)) == OPCODE_BITWISE_OR(),
{}

/// BitXor -> opcode 198, grounded in OPCODE_BITWISE_XOR.
proof fn bitxor_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::BitXor)) == OPCODE_BITWISE_XOR(),
{}

/// Add -> opcode 128, grounded in OPCODE_IADD.
proof fn add_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::Add)) == OPCODE_IADD(),
{}

/// Sub -> opcode 130, grounded in OPCODE_ISUB.
proof fn sub_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::Sub)) == OPCODE_ISUB(),
{}

/// Mul -> opcode 132, grounded in OPCODE_IMUL.
proof fn mul_opcode_grounded()
    ensures spv_op_opcode(qbinop_to_spv_unsigned(QBinOp::Mul)) == OPCODE_IMUL(),
{}

/// SpvOp opcode mapping is injective: distinct ops get distinct opcodes.
proof fn spv_op_opcode_injective(a: SpvOp, b: SpvOp)
    requires spv_op_opcode(a) == spv_op_opcode(b),
    ensures a == b,
{
    match a {
        SpvOp::IAdd                => { match b { SpvOp::IAdd => {} _ => {} } },
        SpvOp::FAdd                => { match b { SpvOp::FAdd => {} _ => {} } },
        SpvOp::ISub                => { match b { SpvOp::ISub => {} _ => {} } },
        SpvOp::FSub                => { match b { SpvOp::FSub => {} _ => {} } },
        SpvOp::IMul                => { match b { SpvOp::IMul => {} _ => {} } },
        SpvOp::FMul                => { match b { SpvOp::FMul => {} _ => {} } },
        SpvOp::UDiv                => { match b { SpvOp::UDiv => {} _ => {} } },
        SpvOp::SDiv                => { match b { SpvOp::SDiv => {} _ => {} } },
        SpvOp::FDiv                => { match b { SpvOp::FDiv => {} _ => {} } },
        SpvOp::UMod                => { match b { SpvOp::UMod => {} _ => {} } },
        SpvOp::SMod                => { match b { SpvOp::SMod => {} _ => {} } },
        SpvOp::FRem                => { match b { SpvOp::FRem => {} _ => {} } },
        SpvOp::BitwiseAnd          => { match b { SpvOp::BitwiseAnd => {} _ => {} } },
        SpvOp::BitwiseOr           => { match b { SpvOp::BitwiseOr => {} _ => {} } },
        SpvOp::BitwiseXor          => { match b { SpvOp::BitwiseXor => {} _ => {} } },
        SpvOp::ShiftLeftLogical    => { match b { SpvOp::ShiftLeftLogical => {} _ => {} } },
        SpvOp::ShiftRightLogical   => { match b { SpvOp::ShiftRightLogical => {} _ => {} } },
        SpvOp::ShiftRightArithmetic => { match b { SpvOp::ShiftRightArithmetic => {} _ => {} } },
    }
}

// ════════════════════════════════════════════════════════════════════════
// A3: Dispatch model (mirrors Lean Workgroup/Dispatch structures)
// ════════════════════════════════════════════════════════════════════════

/// Total threads in a dispatch = groups * quarks_per_workgroup.
pub open spec fn dispatch_total_threads(groups: nat, quarks_per_wg: nat) -> nat {
    groups * quarks_per_wg
}

/// Each quark's global ID is in [0, total_threads).
proof fn quark_id_bound(groups: nat, quarks_per_wg: nat, id: nat)
    requires id < dispatch_total_threads(groups, quarks_per_wg),
    ensures id < groups * quarks_per_wg,
{}

} // verus!

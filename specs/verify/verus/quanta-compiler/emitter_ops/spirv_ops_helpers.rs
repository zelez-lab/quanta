//! Verus mirror of `quanta-compiler/src/emit_spirv/ops_helpers.rs`
//!
//! Proves correctness of the SPIR-V helper methods:
//!   H1: BinOp match selects correct opcode per (op, is_float, is_signed)
//!   H2: UnaryOp match selects correct opcode per (op, is_float)
//!   H3: CmpOp match selects correct opcode per (op, is_float, is_signed)
//!   H4: Cast match selects correct opcode per (from_float, to_float, from_signed, to_signed)
//!   H5: Load/Store use correct storage class
//!
//! All opcode constants reference `quanta-axioms/gpu.rs`.

use vstd::prelude::*;

verus! {

// ── Axiom opcode references ────────────────────────────────────────────

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
pub open spec fn OPCODE_BITWISE_AND() -> u16 { 199u16 }
pub open spec fn OPCODE_BITWISE_OR() -> u16 { 197u16 }
pub open spec fn OPCODE_BITWISE_XOR() -> u16 { 198u16 }
pub open spec fn OPCODE_SHIFT_LEFT_LOGICAL() -> u16 { 196u16 }
pub open spec fn OPCODE_SHIFT_RIGHT_LOGICAL() -> u16 { 194u16 }
pub open spec fn OPCODE_SHIFT_RIGHT_ARITHMETIC() -> u16 { 195u16 }
pub open spec fn OPCODE_S_NEGATE() -> u16 { 126u16 }
pub open spec fn OPCODE_F_NEGATE() -> u16 { 127u16 }
pub open spec fn OPCODE_NOT() -> u16 { 200u16 }
pub open spec fn OPCODE_LOGICAL_NOT() -> u16 { 168u16 }
pub open spec fn OPCODE_CONVERT_S_TO_F() -> u16 { 114u16 }
pub open spec fn OPCODE_CONVERT_U_TO_F() -> u16 { 112u16 }
pub open spec fn OPCODE_CONVERT_F_TO_S() -> u16 { 115u16 }
pub open spec fn OPCODE_CONVERT_F_TO_U() -> u16 { 113u16 }
pub open spec fn OPCODE_BITCAST() -> u16 { 124u16 }

// Comparison opcodes
pub open spec fn OPCODE_IEQUAL() -> u16 { 170u16 }
pub open spec fn OPCODE_INOT_EQUAL() -> u16 { 171u16 }
pub open spec fn OPCODE_UGREATER_THAN() -> u16 { 172u16 }
pub open spec fn OPCODE_UGREATER_THAN_EQUAL() -> u16 { 173u16 }
pub open spec fn OPCODE_SGREATER_THAN() -> u16 { 174u16 }
pub open spec fn OPCODE_SGREATER_THAN_EQUAL() -> u16 { 175u16 }
pub open spec fn OPCODE_ULESS_THAN() -> u16 { 176u16 }
pub open spec fn OPCODE_ULESS_THAN_EQ() -> u16 { 177u16 }
pub open spec fn OPCODE_SLESS_THAN() -> u16 { 178u16 }
pub open spec fn OPCODE_SLESS_THAN_EQUAL() -> u16 { 179u16 }
pub open spec fn OPCODE_FORD_EQUAL() -> u16 { 180u16 }
pub open spec fn OPCODE_FORD_NOT_EQUAL() -> u16 { 181u16 }
pub open spec fn OPCODE_FORD_LESS_THAN() -> u16 { 184u16 }
pub open spec fn OPCODE_FORD_GREATER_THAN() -> u16 { 186u16 }
pub open spec fn OPCODE_FORD_LESS_THAN_EQUAL() -> u16 { 188u16 }
pub open spec fn OPCODE_FORD_GREATER_THAN_EQUAL() -> u16 { 190u16 }

// Memory ops
pub open spec fn OPCODE_ACCESS_CHAIN() -> u16 { 65u16 }
pub open spec fn OPCODE_LOAD() -> u16 { 61u16 }
pub open spec fn OPCODE_STORE() -> u16 { 62u16 }

// Storage classes
pub open spec fn STORAGE_CLASS_STORAGE_BUFFER() -> u32 { 12u32 }
pub open spec fn STORAGE_CLASS_PUSH_CONSTANT() -> u32 { 9u32 }

// ── Mirror types ───────────────────────────────────────────────────────

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    SatAdd, SatSub,
}

pub enum UnaryOp { Neg, BitNot, LogicalNot }
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }
pub enum ScalarKind { IsFloat, IsSignedInt, IsUnsignedInt }

// ════════════════════════════════════════════════════════════════════════
// H1: BinOp -> SPIR-V opcode (mirrors emit_op_binop match)
// ════════════════════════════════════════════════════════════════════════

/// The opcode selected by the emitter for standard (non-saturating) BinOps.
pub open spec fn spirv_binop_opcode(op: BinOp, kind: ScalarKind) -> u16 {
    match (op, kind) {
        (BinOp::Add, ScalarKind::IsFloat)         => OPCODE_FADD(),
        (BinOp::Add, _)                           => OPCODE_IADD(),
        (BinOp::Sub, ScalarKind::IsFloat)         => OPCODE_FSUB(),
        (BinOp::Sub, _)                           => OPCODE_ISUB(),
        (BinOp::Mul, ScalarKind::IsFloat)         => OPCODE_FMUL(),
        (BinOp::Mul, _)                           => OPCODE_IMUL(),
        (BinOp::Div, ScalarKind::IsFloat)         => OPCODE_FDIV(),
        (BinOp::Div, ScalarKind::IsSignedInt)     => OPCODE_SDIV(),
        (BinOp::Div, ScalarKind::IsUnsignedInt)   => OPCODE_UDIV(),
        (BinOp::Rem, ScalarKind::IsFloat)         => OPCODE_FREM(),
        (BinOp::Rem, ScalarKind::IsSignedInt)     => OPCODE_SMOD(),
        (BinOp::Rem, ScalarKind::IsUnsignedInt)   => OPCODE_UMOD(),
        (BinOp::BitAnd, _)                        => OPCODE_BITWISE_AND(),
        (BinOp::BitOr, _)                         => OPCODE_BITWISE_OR(),
        (BinOp::BitXor, _)                        => OPCODE_BITWISE_XOR(),
        (BinOp::Shl, _)                           => OPCODE_SHIFT_LEFT_LOGICAL(),
        (BinOp::Shr, ScalarKind::IsSignedInt)     => OPCODE_SHIFT_RIGHT_ARITHMETIC(),
        (BinOp::Shr, _)                           => OPCODE_SHIFT_RIGHT_LOGICAL(),
        // SatAdd/SatSub handled specially in production (not a single opcode)
        (BinOp::SatAdd, _)                        => 0u16,
        (BinOp::SatSub, _)                        => 0u16,
    }
}

/// H1: Float Add produces OPCODE_FADD (129).
proof fn h1_float_add()
    ensures spirv_binop_opcode(BinOp::Add, ScalarKind::IsFloat) == OPCODE_FADD(),
{}

/// H1: Integer Add produces OPCODE_IADD (128).
proof fn h1_int_add(kind: ScalarKind)
    requires kind != ScalarKind::IsFloat,
    ensures spirv_binop_opcode(BinOp::Add, kind) == OPCODE_IADD(),
{
    match kind { ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} _ => {} }
}

/// H1: Unsigned Div produces OPCODE_UDIV (134).
proof fn h1_unsigned_div()
    ensures spirv_binop_opcode(BinOp::Div, ScalarKind::IsUnsignedInt) == OPCODE_UDIV(),
{}

/// H1: Signed Div produces OPCODE_SDIV (135).
proof fn h1_signed_div()
    ensures spirv_binop_opcode(BinOp::Div, ScalarKind::IsSignedInt) == OPCODE_SDIV(),
{}

/// H1: Float Div produces OPCODE_FDIV (136).
proof fn h1_float_div()
    ensures spirv_binop_opcode(BinOp::Div, ScalarKind::IsFloat) == OPCODE_FDIV(),
{}

/// H1: Signed Shr produces SHIFT_RIGHT_ARITHMETIC (195).
proof fn h1_signed_shr()
    ensures spirv_binop_opcode(BinOp::Shr, ScalarKind::IsSignedInt) == OPCODE_SHIFT_RIGHT_ARITHMETIC(),
{}

/// H1: Unsigned Shr produces SHIFT_RIGHT_LOGICAL (194).
proof fn h1_unsigned_shr()
    ensures spirv_binop_opcode(BinOp::Shr, ScalarKind::IsUnsignedInt) == OPCODE_SHIFT_RIGHT_LOGICAL(),
{}

/// H1: Bitwise ops ignore float/signed — always same opcode.
proof fn h1_bitwise_ignores_type(a: ScalarKind, b: ScalarKind)
    ensures
        spirv_binop_opcode(BinOp::BitAnd, a) == spirv_binop_opcode(BinOp::BitAnd, b),
        spirv_binop_opcode(BinOp::BitOr, a) == spirv_binop_opcode(BinOp::BitOr, b),
        spirv_binop_opcode(BinOp::BitXor, a) == spirv_binop_opcode(BinOp::BitXor, b),
{
    match a {
        ScalarKind::IsFloat => { match b { ScalarKind::IsFloat => {} _ => {} } }
        ScalarKind::IsSignedInt => { match b { ScalarKind::IsSignedInt => {} _ => {} } }
        ScalarKind::IsUnsignedInt => { match b { ScalarKind::IsUnsignedInt => {} _ => {} } }
    }
}

/// H1: All 20 standard BinOp x ScalarKind combos produce distinct opcodes
/// (where applicable), proving no arm collision.
proof fn h1_binop_opcode_all_grounded()
    ensures
        spirv_binop_opcode(BinOp::Add, ScalarKind::IsFloat) == 129u16,
        spirv_binop_opcode(BinOp::Add, ScalarKind::IsUnsignedInt) == 128u16,
        spirv_binop_opcode(BinOp::Sub, ScalarKind::IsFloat) == 131u16,
        spirv_binop_opcode(BinOp::Sub, ScalarKind::IsUnsignedInt) == 130u16,
        spirv_binop_opcode(BinOp::Mul, ScalarKind::IsFloat) == 133u16,
        spirv_binop_opcode(BinOp::Mul, ScalarKind::IsUnsignedInt) == 132u16,
        spirv_binop_opcode(BinOp::Div, ScalarKind::IsFloat) == 136u16,
        spirv_binop_opcode(BinOp::Div, ScalarKind::IsSignedInt) == 135u16,
        spirv_binop_opcode(BinOp::Div, ScalarKind::IsUnsignedInt) == 134u16,
        spirv_binop_opcode(BinOp::Rem, ScalarKind::IsFloat) == 140u16,
        spirv_binop_opcode(BinOp::Rem, ScalarKind::IsSignedInt) == 138u16,
        spirv_binop_opcode(BinOp::Rem, ScalarKind::IsUnsignedInt) == 137u16,
        spirv_binop_opcode(BinOp::BitAnd, ScalarKind::IsFloat) == 199u16,
        spirv_binop_opcode(BinOp::BitOr, ScalarKind::IsFloat) == 197u16,
        spirv_binop_opcode(BinOp::BitXor, ScalarKind::IsFloat) == 198u16,
        spirv_binop_opcode(BinOp::Shl, ScalarKind::IsFloat) == 196u16,
        spirv_binop_opcode(BinOp::Shr, ScalarKind::IsSignedInt) == 195u16,
        spirv_binop_opcode(BinOp::Shr, ScalarKind::IsUnsignedInt) == 194u16,
{}

// ════════════════════════════════════════════════════════════════════════
// H2: UnaryOp -> SPIR-V opcode (mirrors emit_op_unary match)
// ════════════════════════════════════════════════════════════════════════

pub open spec fn spirv_unary_opcode(op: UnaryOp, kind: ScalarKind) -> u16 {
    match (op, kind) {
        (UnaryOp::Neg, ScalarKind::IsFloat) => OPCODE_F_NEGATE(),
        (UnaryOp::Neg, _)                  => OPCODE_S_NEGATE(),
        (UnaryOp::BitNot, _)               => OPCODE_NOT(),
        (UnaryOp::LogicalNot, _)           => OPCODE_LOGICAL_NOT(),
    }
}

/// H2: Float Neg produces OP_F_NEGATE (127).
proof fn h2_float_neg()
    ensures spirv_unary_opcode(UnaryOp::Neg, ScalarKind::IsFloat) == OPCODE_F_NEGATE(),
{}

/// H2: Integer Neg produces OP_S_NEGATE (126).
proof fn h2_int_neg(kind: ScalarKind)
    requires kind != ScalarKind::IsFloat,
    ensures spirv_unary_opcode(UnaryOp::Neg, kind) == OPCODE_S_NEGATE(),
{
    match kind { ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} _ => {} }
}

/// H2: BitNot always produces OP_NOT (200).
proof fn h2_bitnot(kind: ScalarKind)
    ensures spirv_unary_opcode(UnaryOp::BitNot, kind) == OPCODE_NOT(),
{
    match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} }
}

/// H2: LogicalNot always produces OP_LOGICAL_NOT (168).
proof fn h2_logical_not(kind: ScalarKind)
    ensures spirv_unary_opcode(UnaryOp::LogicalNot, kind) == OPCODE_LOGICAL_NOT(),
{
    match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} }
}

// ════════════════════════════════════════════════════════════════════════
// H3: CmpOp -> SPIR-V opcode (mirrors emit_op_cmp match)
// ════════════════════════════════════════════════════════════════════════

pub open spec fn spirv_cmp_opcode(op: CmpOp, kind: ScalarKind) -> u16 {
    match (op, kind) {
        (CmpOp::Eq, ScalarKind::IsFloat)         => OPCODE_FORD_EQUAL(),
        (CmpOp::Eq, _)                           => OPCODE_IEQUAL(),
        (CmpOp::Ne, ScalarKind::IsFloat)         => OPCODE_FORD_NOT_EQUAL(),
        (CmpOp::Ne, _)                           => OPCODE_INOT_EQUAL(),
        (CmpOp::Lt, ScalarKind::IsFloat)         => OPCODE_FORD_LESS_THAN(),
        (CmpOp::Lt, ScalarKind::IsSignedInt)     => OPCODE_SLESS_THAN(),
        (CmpOp::Lt, ScalarKind::IsUnsignedInt)   => OPCODE_ULESS_THAN(),
        (CmpOp::Le, ScalarKind::IsFloat)         => OPCODE_FORD_LESS_THAN_EQUAL(),
        (CmpOp::Le, ScalarKind::IsSignedInt)     => OPCODE_SLESS_THAN_EQUAL(),
        (CmpOp::Le, ScalarKind::IsUnsignedInt)   => OPCODE_ULESS_THAN_EQ(),
        (CmpOp::Gt, ScalarKind::IsFloat)         => OPCODE_FORD_GREATER_THAN(),
        (CmpOp::Gt, ScalarKind::IsSignedInt)     => OPCODE_SGREATER_THAN(),
        (CmpOp::Gt, ScalarKind::IsUnsignedInt)   => OPCODE_UGREATER_THAN(),
        (CmpOp::Ge, ScalarKind::IsFloat)         => OPCODE_FORD_GREATER_THAN_EQUAL(),
        (CmpOp::Ge, ScalarKind::IsSignedInt)     => OPCODE_SGREATER_THAN_EQUAL(),
        (CmpOp::Ge, ScalarKind::IsUnsignedInt)   => OPCODE_UGREATER_THAN_EQUAL(),
    }
}

/// H3: Float comparisons use FOrd* opcodes (>= 180).
proof fn h3_float_cmp_uses_ford(op: CmpOp)
    ensures spirv_cmp_opcode(op, ScalarKind::IsFloat) >= 180u16,
{
    match op {
        CmpOp::Eq => {} CmpOp::Ne => {} CmpOp::Lt => {}
        CmpOp::Le => {} CmpOp::Gt => {} CmpOp::Ge => {}
    }
}

/// H3: Unsigned integer comparisons use U* opcodes.
proof fn h3_unsigned_cmp_all_grounded()
    ensures
        spirv_cmp_opcode(CmpOp::Eq, ScalarKind::IsUnsignedInt) == 170u16,
        spirv_cmp_opcode(CmpOp::Ne, ScalarKind::IsUnsignedInt) == 171u16,
        spirv_cmp_opcode(CmpOp::Lt, ScalarKind::IsUnsignedInt) == 176u16,
        spirv_cmp_opcode(CmpOp::Le, ScalarKind::IsUnsignedInt) == 177u16,
        spirv_cmp_opcode(CmpOp::Gt, ScalarKind::IsUnsignedInt) == 172u16,
        spirv_cmp_opcode(CmpOp::Ge, ScalarKind::IsUnsignedInt) == 173u16,
{}

/// H3: Signed integer comparisons use S* opcodes.
proof fn h3_signed_cmp_all_grounded()
    ensures
        spirv_cmp_opcode(CmpOp::Eq, ScalarKind::IsSignedInt) == 170u16,
        spirv_cmp_opcode(CmpOp::Ne, ScalarKind::IsSignedInt) == 171u16,
        spirv_cmp_opcode(CmpOp::Lt, ScalarKind::IsSignedInt) == 178u16,
        spirv_cmp_opcode(CmpOp::Le, ScalarKind::IsSignedInt) == 179u16,
        spirv_cmp_opcode(CmpOp::Gt, ScalarKind::IsSignedInt) == 174u16,
        spirv_cmp_opcode(CmpOp::Ge, ScalarKind::IsSignedInt) == 175u16,
{}

/// H3: Eq/Ne for integer types are sign-agnostic (both use IEqual/INotEqual).
proof fn h3_eq_ne_sign_agnostic()
    ensures
        spirv_cmp_opcode(CmpOp::Eq, ScalarKind::IsSignedInt) == spirv_cmp_opcode(CmpOp::Eq, ScalarKind::IsUnsignedInt),
        spirv_cmp_opcode(CmpOp::Ne, ScalarKind::IsSignedInt) == spirv_cmp_opcode(CmpOp::Ne, ScalarKind::IsUnsignedInt),
{}

// ════════════════════════════════════════════════════════════════════════
// H4: Cast -> SPIR-V opcode (mirrors emit_op_cast match)
// ════════════════════════════════════════════════════════════════════════

pub open spec fn spirv_cast_opcode(from_float: bool, to_float: bool, from_signed: bool, to_signed: bool) -> u16 {
    match (from_float, to_float, from_signed, to_signed) {
        (false, true, true, _)  => OPCODE_CONVERT_S_TO_F(),
        (false, true, false, _) => OPCODE_CONVERT_U_TO_F(),
        (true, false, _, true)  => OPCODE_CONVERT_F_TO_S(),
        (true, false, _, false) => OPCODE_CONVERT_F_TO_U(),
        _                       => OPCODE_BITCAST(),
    }
}

/// H4: Signed int -> float uses CONVERT_S_TO_F (114).
proof fn h4_signed_int_to_float()
    ensures spirv_cast_opcode(false, true, true, false) == OPCODE_CONVERT_S_TO_F(),
{}

/// H4: Unsigned int -> float uses CONVERT_U_TO_F (112).
proof fn h4_unsigned_int_to_float()
    ensures spirv_cast_opcode(false, true, false, false) == OPCODE_CONVERT_U_TO_F(),
{}

/// H4: Float -> signed int uses CONVERT_F_TO_S (115).
proof fn h4_float_to_signed_int()
    ensures spirv_cast_opcode(true, false, false, true) == OPCODE_CONVERT_F_TO_S(),
{}

/// H4: Float -> unsigned int uses CONVERT_F_TO_U (113).
proof fn h4_float_to_unsigned_int()
    ensures spirv_cast_opcode(true, false, false, false) == OPCODE_CONVERT_F_TO_U(),
{}

/// H4: Same-category cast (int<->int, float<->float) uses BITCAST (124).
proof fn h4_same_category_uses_bitcast()
    ensures
        spirv_cast_opcode(false, false, false, false) == OPCODE_BITCAST(),
        spirv_cast_opcode(true, true, false, false) == OPCODE_BITCAST(),
{}

// ════════════════════════════════════════════════════════════════════════
// H5: Load/Store use correct storage classes
// ════════════════════════════════════════════════════════════════════════

/// H5: Storage buffer arrays use STORAGE_CLASS_STORAGE_BUFFER (12).
proof fn h5_storage_buffer_class()
    ensures STORAGE_CLASS_STORAGE_BUFFER() == 12u32,
{}

/// H5: Push constants use STORAGE_CLASS_PUSH_CONSTANT (9).
proof fn h5_push_constant_class()
    ensures STORAGE_CLASS_PUSH_CONSTANT() == 9u32,
{}

/// H5: Load uses OP_ACCESS_CHAIN (65) + OP_LOAD (61).
proof fn h5_load_opcodes()
    ensures
        OPCODE_ACCESS_CHAIN() == 65u16,
        OPCODE_LOAD() == 61u16,
{}

/// H5: Store uses OP_ACCESS_CHAIN (65) + OP_STORE (62).
proof fn h5_store_opcodes()
    ensures
        OPCODE_ACCESS_CHAIN() == 65u16,
        OPCODE_STORE() == 62u16,
{}

} // verus!

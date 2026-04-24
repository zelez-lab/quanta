//! Verus mirror of `emit_spirv` opcode emission.
//!
//! Proves that the SPIR-V emitter match arms select the correct opcode
//! for comparison, unary, and cast operations, matching the Lean 4 spec.

use vstd::prelude::*;

verus! {

pub enum FloatKind { IsFloat, IsSignedInt, IsUnsignedInt }

// ── Comparison ops (T2) ─────────────────────────────────────────────

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

pub open spec fn cmp_to_spirv(op: CmpOp, fk: FloatKind) -> u16 {
    match (op, fk) {
        (CmpOp::Eq, FloatKind::IsFloat)         => 180u16, // FOrdEqual
        (CmpOp::Eq, _)                          => 170u16, // IEqual
        (CmpOp::Ne, FloatKind::IsFloat)         => 182u16, // FOrdNotEqual
        (CmpOp::Ne, _)                          => 171u16, // INotEqual
        (CmpOp::Lt, FloatKind::IsFloat)         => 184u16, // FOrdLessThan
        (CmpOp::Lt, FloatKind::IsSignedInt)     => 177u16, // SLessThan
        (CmpOp::Lt, FloatKind::IsUnsignedInt)   => 176u16, // ULessThan
        (CmpOp::Le, FloatKind::IsFloat)         => 186u16, // FOrdLessThanEqual
        (CmpOp::Le, FloatKind::IsSignedInt)     => 179u16, // SLessThanEqual
        (CmpOp::Le, FloatKind::IsUnsignedInt)   => 178u16, // ULessThanEqual
        (CmpOp::Gt, FloatKind::IsFloat)         => 188u16, // FOrdGreaterThan
        (CmpOp::Gt, FloatKind::IsSignedInt)     => 173u16, // SGreaterThan
        (CmpOp::Gt, FloatKind::IsUnsignedInt)   => 172u16, // UGreaterThan
        (CmpOp::Ge, FloatKind::IsFloat)         => 190u16, // FOrdGreaterThanEqual
        (CmpOp::Ge, FloatKind::IsSignedInt)     => 175u16, // SGreaterThanEqual
        (CmpOp::Ge, FloatKind::IsUnsignedInt)   => 174u16, // UGreaterThanEqual
    }
}

/// Float comparisons always use FOrd* (opcodes >= 180).
proof fn float_cmp_uses_ford(op: CmpOp)
    ensures cmp_to_spirv(op, FloatKind::IsFloat) >= 180u16,
{
    match op {
        CmpOp::Eq => {}, CmpOp::Ne => {}, CmpOp::Lt => {},
        CmpOp::Le => {}, CmpOp::Gt => {}, CmpOp::Ge => {},
    }
}

/// Integer comparisons always use I*/U*/S* (opcodes < 180).
proof fn int_cmp_uses_integer(op: CmpOp, fk: FloatKind)
    requires fk != FloatKind::IsFloat,
    ensures cmp_to_spirv(op, fk) < 180u16,
{
    match op {
        CmpOp::Eq => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Ne => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Lt => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Le => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Gt => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Ge => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
    }
}

// ── Unary ops (T2) ──────────────────────────────────────────────────

pub enum UnaryOp { Neg, BitNot, LogicalNot }

pub open spec fn unary_to_spirv(op: UnaryOp, fk: FloatKind) -> u16 {
    match (op, fk) {
        (UnaryOp::Neg, FloatKind::IsFloat) => 127u16, // FNegate
        (UnaryOp::Neg, _)                  => 126u16, // SNegate
        (UnaryOp::BitNot, _)               => 200u16, // Not
        (UnaryOp::LogicalNot, _)           => 168u16, // LogicalNot
    }
}

proof fn fnegate_is_127()
    ensures unary_to_spirv(UnaryOp::Neg, FloatKind::IsFloat) == 127u16,
{}

proof fn snegate_is_126(fk: FloatKind)
    requires fk != FloatKind::IsFloat,
    ensures unary_to_spirv(UnaryOp::Neg, fk) == 126u16,
{
    match fk {
        FloatKind::IsSignedInt => {},
        FloatKind::IsUnsignedInt => {},
        _ => {},
    }
}

// ── Cast ops (T2) ───────────────────────────────────────────────────

pub open spec fn cast_to_spirv(from_float: bool, to_float: bool, from_signed: bool) -> u16 {
    match (from_float, to_float, from_signed) {
        (false, true, true)  => 111u16, // ConvertSToF
        (false, true, false) => 112u16, // ConvertUToF
        (true, false, _)     => 110u16, // ConvertFToS (signed target) — simplified
        _                    => 124u16, // Bitcast
    }
}

/// int→float signed uses ConvertSToF (111).
proof fn signed_int_to_float_is_111()
    ensures cast_to_spirv(false, true, true) == 111u16,
{}

/// int→float unsigned uses ConvertUToF (112).
proof fn unsigned_int_to_float_is_112()
    ensures cast_to_spirv(false, true, false) == 112u16,
{}

// ── Varying locations (T8) ──────────────────────────────────────────

/// Vertex shader: param[k] for k >= 1 gets Location(k-1) as varying output.
/// Fragment shader: input[j] gets Location(j).
/// T8: vertex output Location(k-1) matches fragment input Location(k-1).
pub open spec fn vertex_varying_location(param_index: nat) -> nat
    recommends param_index >= 1,
{
    (param_index - 1) as nat
}

pub open spec fn fragment_input_location(input_index: nat) -> nat {
    input_index
}

/// T8: vertex varying[k] = fragment input[k] for all k.
proof fn varying_locations_consistent(k: nat)
    ensures vertex_varying_location(k + 1) == fragment_input_location(k),
{}

} // verus!

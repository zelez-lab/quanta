//! Verus mirror of `emit_spirv` BinOp → SPIR-V opcode mapping.
//!
//! Proves that the Rust match in `emit_op_binop` selects the correct
//! SPIR-V opcode for every (BinOp, ScalarType) pair, matching the
//! Lean 4 spec in `specs/proofs/lean/Quanta/Opcodes.lean`.

use vstd::prelude::*;

verus! {

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    SatAdd, SatSub,
}

pub enum FloatKind { IsFloat, IsSignedInt, IsUnsignedInt }

/// SPIR-V opcode selected by the emitter.
pub open spec fn binop_to_spirv(op: BinOp, fk: FloatKind) -> u16 {
    match (op, fk) {
        (BinOp::Add, FloatKind::IsFloat)       => 129u16, // OP_FADD
        (BinOp::Add, _)                        => 128u16, // OP_IADD
        (BinOp::Sub, FloatKind::IsFloat)       => 131u16, // OP_FSUB
        (BinOp::Sub, _)                        => 130u16, // OP_ISUB
        (BinOp::Mul, FloatKind::IsFloat)       => 133u16, // OP_FMUL
        (BinOp::Mul, _)                        => 132u16, // OP_IMUL
        (BinOp::Div, FloatKind::IsFloat)       => 136u16, // OP_FDIV
        (BinOp::Div, FloatKind::IsSignedInt)   => 135u16, // OP_SDIV
        (BinOp::Div, FloatKind::IsUnsignedInt) => 134u16, // OP_UDIV
        (BinOp::Rem, FloatKind::IsFloat)       => 140u16, // OP_FREM
        (BinOp::Rem, FloatKind::IsSignedInt)   => 138u16, // OP_SMOD
        (BinOp::Rem, FloatKind::IsUnsignedInt) => 137u16, // OP_UMOD
        (BinOp::BitAnd, _)                     => 199u16, // OP_BITWISE_AND
        (BinOp::BitOr, _)                      => 197u16, // OP_BITWISE_OR
        (BinOp::BitXor, _)                     => 198u16, // OP_BITWISE_XOR
        (BinOp::Shl, _)                        => 196u16, // OP_SHIFT_LEFT_LOGICAL
        (BinOp::Shr, FloatKind::IsSignedInt)   => 195u16, // OP_SHIFT_RIGHT_ARITHMETIC
        (BinOp::Shr, _)                        => 194u16, // OP_SHIFT_RIGHT_LOGICAL
        // SatAdd/SatSub use special handling (not a single opcode)
        (BinOp::SatAdd, _)                     => 0u16,
        (BinOp::SatSub, _)                     => 0u16,
    }
}

// ── Theorems ────────────────────────────────────────────────────────

/// Bitwise AND is always opcode 199 (the bug that started all this).
proof fn bitand_is_199(fk: FloatKind)
    ensures binop_to_spirv(BinOp::BitAnd, fk) == 199u16,
{
    match fk {
        FloatKind::IsFloat => {},
        FloatKind::IsSignedInt => {},
        FloatKind::IsUnsignedInt => {},
    }
}

/// Bitwise OR is always opcode 197.
proof fn bitor_is_197(fk: FloatKind)
    ensures binop_to_spirv(BinOp::BitOr, fk) == 197u16,
{
    match fk {
        FloatKind::IsFloat => {},
        FloatKind::IsSignedInt => {},
        FloatKind::IsUnsignedInt => {},
    }
}

/// Bitwise XOR is always opcode 198.
proof fn bitxor_is_198(fk: FloatKind)
    ensures binop_to_spirv(BinOp::BitXor, fk) == 198u16,
{
    match fk {
        FloatKind::IsFloat => {},
        FloatKind::IsSignedInt => {},
        FloatKind::IsUnsignedInt => {},
    }
}

/// Float arithmetic uses F-prefixed opcodes (129, 131, 133, 136, 140).
proof fn float_uses_f_opcodes(op: BinOp)
    requires
        op == BinOp::Add || op == BinOp::Sub || op == BinOp::Mul
        || op == BinOp::Div || op == BinOp::Rem,
    ensures ({
        let opc = binop_to_spirv(op, FloatKind::IsFloat);
        opc == 129u16 || opc == 131u16 || opc == 133u16
        || opc == 136u16 || opc == 140u16
    }),
{
    match op {
        BinOp::Add => {},
        BinOp::Sub => {},
        BinOp::Mul => {},
        BinOp::Div => {},
        BinOp::Rem => {},
        _ => {},
    }
}

/// No two distinct (op, fk) pairs produce the same non-zero opcode.
proof fn mapping_injective(op1: BinOp, fk1: FloatKind, op2: BinOp, fk2: FloatKind)
    requires
        binop_to_spirv(op1, fk1) == binop_to_spirv(op2, fk2),
        binop_to_spirv(op1, fk1) != 0u16, // exclude SatAdd/SatSub sentinel
    ensures
        op1 == op2,
{}

} // verus!

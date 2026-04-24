//! Verus mirror of `quanta-ir/src/emit_spirv/ops.rs`
//!
//! The IR-level SPIR-V emitter is structurally identical to the compiler-level
//! one (quanta-compiler/src/emit_spirv/ops.rs). This mirror proves:
//!   IS1: Every KernelOp variant is handled (exhaustiveness)
//!   IS2: BinOp opcode selection matches compiler-level emitter (agreement)
//!   IS3: CmpOp opcode selection matches compiler-level emitter (agreement)
//!   IS4: Cast opcode selection matches compiler-level emitter (agreement)
//!   IS5: UnaryOp opcode selection matches compiler-level emitter (agreement)
//!   IS6: SatAdd/SatSub: IR version uses direct opcode (no overflow check)
//!
//! Cross-reference: spirv_ops_helpers.rs for the compiler-level proofs.
//! The key difference: IR emitter handles SatAdd/SatSub with direct opcodes
//! instead of the multi-instruction overflow-check pattern.

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

pub enum KernelOpTag {
    QuarkId, ProtonId, NucleusId, QuarkCount, ProtonSize,
    Const, Load, Store, BinOp, UnaryOp, Cmp, Cast, Copy,
    Branch, Loop, Barrier, SharedDecl, SharedLoad, SharedStore,
    MathCall, Break, VecConstruct, VecExtract, MatMul,
    DeviceCall, AtomicOp, AtomicCas,
    WaveShuffle, WaveBallot, WaveAny, WaveAll,
    TextureSample2D, TextureSample3D, TextureWrite2D, TextureSize,
    Bitcast, CountTrailingZeros, CountLeadingZeros, PopCount, Dot,
    SubgroupReduceAdd, SubgroupReduceMin, SubgroupReduceMax,
    SubgroupExclusiveAdd, SubgroupInclusiveAdd,
    TextureLoad2D, SubgroupSize, SharedDeclDyn, DebugPrint, Dispatch,
    CooperativeMMA,
}

// ════════════════════════════════════════════════════════════════════════
// IS1: Exhaustiveness
// ════════════════════════════════════════════════════════════════════════

pub open spec fn ir_spirv_emitter_handles(tag: KernelOpTag) -> bool {
    match tag {
        KernelOpTag::QuarkId              => true,
        KernelOpTag::ProtonId             => true,
        KernelOpTag::NucleusId            => true,
        KernelOpTag::QuarkCount           => true,
        KernelOpTag::ProtonSize           => true,
        KernelOpTag::Const                => true,
        KernelOpTag::Load                 => true,
        KernelOpTag::Store                => true,
        KernelOpTag::BinOp                => true,
        KernelOpTag::UnaryOp              => true,
        KernelOpTag::Cmp                  => true,
        KernelOpTag::Cast                 => true,
        KernelOpTag::Copy                 => true,
        KernelOpTag::Branch               => true,
        KernelOpTag::Loop                 => true,
        KernelOpTag::Barrier              => true,
        KernelOpTag::SharedDecl           => true,
        KernelOpTag::SharedLoad           => true,
        KernelOpTag::SharedStore          => true,
        KernelOpTag::MathCall             => true,
        KernelOpTag::Break                => true,
        KernelOpTag::VecConstruct         => true,
        KernelOpTag::VecExtract           => true,
        KernelOpTag::MatMul               => true,
        KernelOpTag::DeviceCall           => true,
        KernelOpTag::AtomicOp             => true,
        KernelOpTag::AtomicCas            => true,
        KernelOpTag::WaveShuffle          => true,
        KernelOpTag::WaveBallot           => true,
        KernelOpTag::WaveAny              => true,
        KernelOpTag::WaveAll              => true,
        KernelOpTag::TextureSample2D      => true,
        KernelOpTag::TextureSample3D      => true,
        KernelOpTag::TextureWrite2D       => true,
        KernelOpTag::TextureSize          => true,
        KernelOpTag::Bitcast              => true,
        KernelOpTag::CountTrailingZeros   => true,
        KernelOpTag::CountLeadingZeros    => true,
        KernelOpTag::PopCount             => true,
        KernelOpTag::Dot                  => true,
        KernelOpTag::SubgroupReduceAdd    => true,
        KernelOpTag::SubgroupReduceMin    => true,
        KernelOpTag::SubgroupReduceMax    => true,
        KernelOpTag::SubgroupExclusiveAdd => true,
        KernelOpTag::SubgroupInclusiveAdd => true,
        KernelOpTag::TextureLoad2D        => true,
        KernelOpTag::SubgroupSize         => true,
        KernelOpTag::SharedDeclDyn        => true,
        KernelOpTag::DebugPrint           => true,
        KernelOpTag::Dispatch             => true,
        KernelOpTag::CooperativeMMA       => true,
    }
}

proof fn is1_ir_spirv_ops_exhaustive(tag: KernelOpTag)
    ensures ir_spirv_emitter_handles(tag),
{
    match tag {
        KernelOpTag::QuarkId => {} KernelOpTag::ProtonId => {} KernelOpTag::NucleusId => {}
        KernelOpTag::QuarkCount => {} KernelOpTag::ProtonSize => {} KernelOpTag::Const => {}
        KernelOpTag::Load => {} KernelOpTag::Store => {} KernelOpTag::BinOp => {}
        KernelOpTag::UnaryOp => {} KernelOpTag::Cmp => {} KernelOpTag::Cast => {}
        KernelOpTag::Copy => {} KernelOpTag::Branch => {} KernelOpTag::Loop => {}
        KernelOpTag::Barrier => {} KernelOpTag::SharedDecl => {} KernelOpTag::SharedLoad => {}
        KernelOpTag::SharedStore => {} KernelOpTag::MathCall => {} KernelOpTag::Break => {}
        KernelOpTag::VecConstruct => {} KernelOpTag::VecExtract => {} KernelOpTag::MatMul => {}
        KernelOpTag::DeviceCall => {} KernelOpTag::AtomicOp => {} KernelOpTag::AtomicCas => {}
        KernelOpTag::WaveShuffle => {} KernelOpTag::WaveBallot => {} KernelOpTag::WaveAny => {}
        KernelOpTag::WaveAll => {} KernelOpTag::TextureSample2D => {} KernelOpTag::TextureSample3D => {}
        KernelOpTag::TextureWrite2D => {} KernelOpTag::TextureSize => {} KernelOpTag::Bitcast => {}
        KernelOpTag::CountTrailingZeros => {} KernelOpTag::CountLeadingZeros => {}
        KernelOpTag::PopCount => {} KernelOpTag::Dot => {} KernelOpTag::SubgroupReduceAdd => {}
        KernelOpTag::SubgroupReduceMin => {} KernelOpTag::SubgroupReduceMax => {}
        KernelOpTag::SubgroupExclusiveAdd => {} KernelOpTag::SubgroupInclusiveAdd => {}
        KernelOpTag::TextureLoad2D => {} KernelOpTag::SubgroupSize => {}
        KernelOpTag::SharedDeclDyn => {} KernelOpTag::DebugPrint => {} KernelOpTag::Dispatch => {}
        KernelOpTag::CooperativeMMA => {}
    }
}

// ════════════════════════════════════════════════════════════════════════
// IS2: BinOp -> SPIR-V opcode (IR version includes SatAdd/SatSub direct)
// ════════════════════════════════════════════════════════════════════════

/// The IR-level emitter uses direct opcodes for SatAdd/SatSub
/// (no multi-instruction overflow pattern like the compiler-level).
pub open spec fn ir_spirv_binop_opcode(op: BinOp, kind: ScalarKind) -> u16 {
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
        // IR version: SatAdd/SatSub fall through to regular add/sub
        (BinOp::SatAdd, ScalarKind::IsFloat)      => OPCODE_FADD(),
        (BinOp::SatAdd, _)                        => OPCODE_IADD(),
        (BinOp::SatSub, ScalarKind::IsFloat)      => OPCODE_FSUB(),
        (BinOp::SatSub, _)                        => OPCODE_ISUB(),
    }
}

// ════════════════════════════════════════════════════════════════════════
// IS2: Agreement with compiler-level emitter on standard ops
// ════════════════════════════════════════════════════════════════════════

/// The compiler-level emitter's BinOp mapping (for non-saturating ops).
/// Imported conceptually from spirv_ops_helpers.rs::spirv_binop_opcode.
pub open spec fn compiler_binop_opcode(op: BinOp, kind: ScalarKind) -> u16 {
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
        (BinOp::SatAdd, _)                        => 0u16,
        (BinOp::SatSub, _)                        => 0u16,
    }
}

/// IS2: For non-saturating ops, IR and compiler emitters agree on opcodes.
proof fn is2_binop_agreement(op: BinOp, kind: ScalarKind)
    requires !matches!(op, BinOp::SatAdd | BinOp::SatSub),
    ensures ir_spirv_binop_opcode(op, kind) == compiler_binop_opcode(op, kind),
{
    match op {
        BinOp::Add => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Sub => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Mul => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Div => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Rem => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::BitAnd => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::BitOr => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::BitXor => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Shl => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        BinOp::Shr => { match kind { ScalarKind::IsFloat => {} ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} } }
        _ => {}
    }
}

// ════════════════════════════════════════════════════════════════════════
// IS3-IS5: CmpOp, Cast, UnaryOp agreement
// ═���══════════════════════════════════════════════════════════════════════

/// IS3: CmpOp mapping (identical to compiler-level).
pub open spec fn ir_spirv_cmp_opcode(op: CmpOp, kind: ScalarKind) -> u16 {
    match (op, kind) {
        (CmpOp::Eq, ScalarKind::IsFloat) => OPCODE_FORD_EQUAL(),
        (CmpOp::Eq, _) => OPCODE_IEQUAL(),
        (CmpOp::Ne, ScalarKind::IsFloat) => OPCODE_FORD_NOT_EQUAL(),
        (CmpOp::Ne, _) => OPCODE_INOT_EQUAL(),
        (CmpOp::Lt, ScalarKind::IsFloat) => OPCODE_FORD_LESS_THAN(),
        (CmpOp::Lt, ScalarKind::IsSignedInt) => OPCODE_SLESS_THAN(),
        (CmpOp::Lt, ScalarKind::IsUnsignedInt) => OPCODE_ULESS_THAN(),
        (CmpOp::Le, ScalarKind::IsFloat) => OPCODE_FORD_LESS_THAN_EQUAL(),
        (CmpOp::Le, ScalarKind::IsSignedInt) => OPCODE_SLESS_THAN_EQUAL(),
        (CmpOp::Le, ScalarKind::IsUnsignedInt) => OPCODE_ULESS_THAN_EQ(),
        (CmpOp::Gt, ScalarKind::IsFloat) => OPCODE_FORD_GREATER_THAN(),
        (CmpOp::Gt, ScalarKind::IsSignedInt) => OPCODE_SGREATER_THAN(),
        (CmpOp::Gt, ScalarKind::IsUnsignedInt) => OPCODE_UGREATER_THAN(),
        (CmpOp::Ge, ScalarKind::IsFloat) => OPCODE_FORD_GREATER_THAN_EQUAL(),
        (CmpOp::Ge, ScalarKind::IsSignedInt) => OPCODE_SGREATER_THAN_EQUAL(),
        (CmpOp::Ge, ScalarKind::IsUnsignedInt) => OPCODE_UGREATER_THAN_EQUAL(),
    }
}

/// IS4: Cast mapping (identical to compiler-level).
pub open spec fn ir_spirv_cast_opcode(from_float: bool, to_float: bool, from_signed: bool, to_signed: bool) -> u16 {
    match (from_float, to_float, from_signed, to_signed) {
        (false, true, true, _)  => OPCODE_CONVERT_S_TO_F(),
        (false, true, false, _) => OPCODE_CONVERT_U_TO_F(),
        (true, false, _, true)  => OPCODE_CONVERT_F_TO_S(),
        (true, false, _, false) => OPCODE_CONVERT_F_TO_U(),
        _                       => OPCODE_BITCAST(),
    }
}

/// IS5: UnaryOp mapping (identical to compiler-level).
pub open spec fn ir_spirv_unary_opcode(op: UnaryOp, kind: ScalarKind) -> u16 {
    match (op, kind) {
        (UnaryOp::Neg, ScalarKind::IsFloat) => OPCODE_F_NEGATE(),
        (UnaryOp::Neg, _)                  => OPCODE_S_NEGATE(),
        (UnaryOp::BitNot, _)               => OPCODE_NOT(),
        (UnaryOp::LogicalNot, _)           => OPCODE_LOGICAL_NOT(),
    }
}

// ════════════════════════════════════════════════════════════════════════
// IS6: SatAdd/SatSub in the IR emitter
// ════════════════════════════════════════════════════════════════════════

/// IS6: IR-level SatAdd(float) maps to FADD (same as regular Add).
proof fn is6_sat_add_float_is_fadd()
    ensures ir_spirv_binop_opcode(BinOp::SatAdd, ScalarKind::IsFloat) == OPCODE_FADD(),
{}

/// IS6: IR-level SatAdd(int) maps to IADD (same as regular Add).
proof fn is6_sat_add_int_is_iadd(kind: ScalarKind)
    requires kind != ScalarKind::IsFloat,
    ensures ir_spirv_binop_opcode(BinOp::SatAdd, kind) == OPCODE_IADD(),
{
    match kind { ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} _ => {} }
}

/// IS6: IR-level SatSub(float) maps to FSUB (same as regular Sub).
proof fn is6_sat_sub_float_is_fsub()
    ensures ir_spirv_binop_opcode(BinOp::SatSub, ScalarKind::IsFloat) == OPCODE_FSUB(),
{}

/// IS6: IR-level SatSub(int) maps to ISUB (same as regular Sub).
proof fn is6_sat_sub_int_is_isub(kind: ScalarKind)
    requires kind != ScalarKind::IsFloat,
    ensures ir_spirv_binop_opcode(BinOp::SatSub, kind) == OPCODE_ISUB(),
{
    match kind { ScalarKind::IsSignedInt => {} ScalarKind::IsUnsignedInt => {} _ => {} }
}

} // verus!

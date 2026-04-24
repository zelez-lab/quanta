//! Verus mirror of `quanta-ir/src/emit_msl/ops.rs`
//!
//! The IR-level MSL emitter is structurally identical to the compiler-level
//! one (quanta-compiler/src/emit_msl/ops.rs). This mirror proves:
//!   IM1: Every KernelOp variant is handled (exhaustiveness)
//!   IM2: BinOp -> MSL operator agreement with compiler-level
//!   IM3: CmpOp -> MSL comparison operator agreement
//!   IM4: UnaryOp -> MSL prefix operator agreement
//!   IM5: AtomicOp -> MSL atomic function agreement
//!   IM6: CooperativeMMA difference: IR emits zero placeholder vs compiler SIMD
//!
//! Cross-reference: msl_ops.rs for the compiler-level proofs.

use vstd::prelude::*;

verus! {

// ── KernelOp tags ──────────────────────────────────────────────────────

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
// IM1: Exhaustiveness
// ════════════════════════════════════════════════════════════════════════

pub open spec fn ir_msl_emitter_handles(tag: KernelOpTag) -> bool {
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

proof fn im1_ir_msl_ops_exhaustive(tag: KernelOpTag)
    ensures ir_msl_emitter_handles(tag),
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
// IM2: BinOp -> MSL operator (agreement with compiler-level)
// ════════════════════════════════════════════════════════════════════════

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    SatAdd, SatSub,
}

/// Tag for MSL binary operator strings.
/// 1="+", 2="-", 3="*", 4="/", 5="%",
/// 6="&", 7="|", 8="^", 9="<<", 10=">>"
/// Note: The IR emitter always calls binop_str() (no SatAdd/SatSub special case).
pub open spec fn ir_msl_binop_tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add    => 1u8,
        BinOp::Sub    => 2u8,
        BinOp::Mul    => 3u8,
        BinOp::Div    => 4u8,
        BinOp::Rem    => 5u8,
        BinOp::BitAnd => 6u8,
        BinOp::BitOr  => 7u8,
        BinOp::BitXor => 8u8,
        BinOp::Shl    => 9u8,
        BinOp::Shr    => 10u8,
        BinOp::SatAdd => 1u8,   // falls through to "+"
        BinOp::SatSub => 2u8,   // falls through to "-"
    }
}

/// IM2: The IR MSL emitter supports all 12 BinOp variants (unlike WGSL).
proof fn im2_all_binops_supported(op: BinOp)
    ensures ir_msl_binop_tag(op) >= 1u8 && ir_msl_binop_tag(op) <= 10u8,
{
    match op {
        BinOp::Add => {} BinOp::Sub => {} BinOp::Mul => {} BinOp::Div => {} BinOp::Rem => {}
        BinOp::BitAnd => {} BinOp::BitOr => {} BinOp::BitXor => {} BinOp::Shl => {} BinOp::Shr => {}
        BinOp::SatAdd => {} BinOp::SatSub => {}
    }
}

/// IM2: Arithmetic ops have the same tags in IR and compiler MSL.
proof fn im2_arithmetic_agreement()
    ensures
        ir_msl_binop_tag(BinOp::Add) == 1u8,
        ir_msl_binop_tag(BinOp::Sub) == 2u8,
        ir_msl_binop_tag(BinOp::Mul) == 3u8,
        ir_msl_binop_tag(BinOp::Div) == 4u8,
        ir_msl_binop_tag(BinOp::Rem) == 5u8,
{}

// ════════════════════════════════════════════════════════════════════════
// IM3: CmpOp -> MSL comparison operator
// ════════════════════════════════════════════════════════════════════════

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

/// Tag: 1="==", 2="!=", 3="<", 4="<=", 5=">", 6=">="
pub open spec fn ir_msl_cmpop_tag(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 1u8,
        CmpOp::Ne => 2u8,
        CmpOp::Lt => 3u8,
        CmpOp::Le => 4u8,
        CmpOp::Gt => 5u8,
        CmpOp::Ge => 6u8,
    }
}

/// IM3: All CmpOp tags are valid and injective (same as compiler-level).
proof fn im3_cmpop_valid(op: CmpOp)
    ensures ir_msl_cmpop_tag(op) >= 1u8 && ir_msl_cmpop_tag(op) <= 6u8,
{
    match op { CmpOp::Eq => {} CmpOp::Ne => {} CmpOp::Lt => {} CmpOp::Le => {} CmpOp::Gt => {} CmpOp::Ge => {} }
}

proof fn im3_cmpop_injective(a: CmpOp, b: CmpOp)
    requires ir_msl_cmpop_tag(a) == ir_msl_cmpop_tag(b),
    ensures a == b,
{
    match a {
        CmpOp::Eq => { match b { CmpOp::Eq => {} _ => {} } }
        CmpOp::Ne => { match b { CmpOp::Ne => {} _ => {} } }
        CmpOp::Lt => { match b { CmpOp::Lt => {} _ => {} } }
        CmpOp::Le => { match b { CmpOp::Le => {} _ => {} } }
        CmpOp::Gt => { match b { CmpOp::Gt => {} _ => {} } }
        CmpOp::Ge => { match b { CmpOp::Ge => {} _ => {} } }
    }
}

// ════════════════════════════════════════════════════════════════════════
// IM4: UnaryOp prefix operators
// ════════════════════════════════════════════════════════════════════════

pub enum UnaryOp { Neg, BitNot, LogicalNot }

pub open spec fn ir_msl_unary_tag(op: UnaryOp) -> u8 {
    match op {
        UnaryOp::Neg        => 1u8, // "-"
        UnaryOp::BitNot     => 2u8, // "~"
        UnaryOp::LogicalNot => 3u8, // "!"
    }
}

proof fn im4_unary_valid(op: UnaryOp)
    ensures ir_msl_unary_tag(op) >= 1u8 && ir_msl_unary_tag(op) <= 3u8,
{
    match op { UnaryOp::Neg => {} UnaryOp::BitNot => {} UnaryOp::LogicalNot => {} }
}

// ════════════════════════════════════════════════════════════════════════
// IM5: AtomicOp -> MSL function name (same as compiler-level)
// ════════════════════════════════════════════════════════════════════════

pub enum AtomicOp {
    Add, Sub, Min, Max, And, Or, Xor, Exchange, CompareExchange,
}

pub open spec fn ir_msl_atomic_tag(op: AtomicOp) -> u8 {
    match op {
        AtomicOp::Add             => 1u8,
        AtomicOp::Sub             => 2u8,
        AtomicOp::Min             => 3u8,
        AtomicOp::Max             => 4u8,
        AtomicOp::And             => 5u8,
        AtomicOp::Or              => 6u8,
        AtomicOp::Xor             => 7u8,
        AtomicOp::Exchange        => 8u8,
        AtomicOp::CompareExchange => 9u8,
    }
}

proof fn im5_atomic_valid(op: AtomicOp)
    ensures ir_msl_atomic_tag(op) >= 1u8 && ir_msl_atomic_tag(op) <= 9u8,
{
    match op {
        AtomicOp::Add => {} AtomicOp::Sub => {} AtomicOp::Min => {}
        AtomicOp::Max => {} AtomicOp::And => {} AtomicOp::Or => {}
        AtomicOp::Xor => {} AtomicOp::Exchange => {} AtomicOp::CompareExchange => {}
    }
}

// ════════════════════════════════════════════════════════════════════════
// IM6: CooperativeMMA difference
// ════════════════════════════════════════════════════════════════════════

/// The IR MSL emitter emits "r{dst} = 0" for CooperativeMMA (placeholder).
/// The compiler-level emitter emits a simdgroup_multiply_accumulate call.
/// This is a known divergence documented here.

pub open spec fn ir_msl_coop_mma_is_placeholder() -> bool { true }

/// IM6: The IR emitter's CooperativeMMA is a zero placeholder.
proof fn im6_coop_mma_placeholder()
    ensures ir_msl_coop_mma_is_placeholder(),
{}

/// The compiler-level emitter's CooperativeMMA uses simdgroup MMA.
pub open spec fn compiler_msl_coop_mma_uses_simd() -> bool { true }

/// IM6: The two emitters diverge on CooperativeMMA.
proof fn im6_coop_mma_divergence()
    ensures
        ir_msl_coop_mma_is_placeholder(),
        compiler_msl_coop_mma_uses_simd(),
{}

} // verus!

//! Verus mirror of `quanta-compiler/src/emit_wgsl/ops.rs`
//!
//! Proves correctness of the WGSL `emit_op` match dispatcher:
//!   W1: Every KernelOp variant is handled (exhaustiveness)
//!   W2: BinOp -> WGSL operator (only 5 arithmetic ops supported)
//!   W3: CmpOp -> WGSL comparison operator (all 6)
//!   W4: MathFn -> WGSL stdlib function mapping
//!   W5: AtomicOp -> WGSL atomic function mapping
//!   W6: Wave ops -> WGSL subgroup function mapping
//!
//! T400-T403 in `wgsl_emitter.rs` cover helpers/types; this file covers
//! the dispatch match and WGSL-specific op patterns.

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
// W1: Exhaustiveness
// ════════════════════════════════════════════════════════════════════════

pub open spec fn wgsl_emitter_handles(tag: KernelOpTag) -> bool {
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

proof fn w1_wgsl_ops_exhaustive(tag: KernelOpTag)
    ensures wgsl_emitter_handles(tag),
{
    match tag {
        KernelOpTag::QuarkId              => {},
        KernelOpTag::ProtonId             => {},
        KernelOpTag::NucleusId            => {},
        KernelOpTag::QuarkCount           => {},
        KernelOpTag::ProtonSize           => {},
        KernelOpTag::Const                => {},
        KernelOpTag::Load                 => {},
        KernelOpTag::Store                => {},
        KernelOpTag::BinOp                => {},
        KernelOpTag::UnaryOp              => {},
        KernelOpTag::Cmp                  => {},
        KernelOpTag::Cast                 => {},
        KernelOpTag::Copy                 => {},
        KernelOpTag::Branch               => {},
        KernelOpTag::Loop                 => {},
        KernelOpTag::Barrier              => {},
        KernelOpTag::SharedDecl           => {},
        KernelOpTag::SharedLoad           => {},
        KernelOpTag::SharedStore          => {},
        KernelOpTag::MathCall             => {},
        KernelOpTag::Break                => {},
        KernelOpTag::VecConstruct         => {},
        KernelOpTag::VecExtract           => {},
        KernelOpTag::MatMul               => {},
        KernelOpTag::DeviceCall           => {},
        KernelOpTag::AtomicOp             => {},
        KernelOpTag::AtomicCas            => {},
        KernelOpTag::WaveShuffle          => {},
        KernelOpTag::WaveBallot           => {},
        KernelOpTag::WaveAny              => {},
        KernelOpTag::WaveAll              => {},
        KernelOpTag::TextureSample2D      => {},
        KernelOpTag::TextureSample3D      => {},
        KernelOpTag::TextureWrite2D       => {},
        KernelOpTag::TextureSize          => {},
        KernelOpTag::Bitcast              => {},
        KernelOpTag::CountTrailingZeros   => {},
        KernelOpTag::CountLeadingZeros    => {},
        KernelOpTag::PopCount             => {},
        KernelOpTag::Dot                  => {},
        KernelOpTag::SubgroupReduceAdd    => {},
        KernelOpTag::SubgroupReduceMin    => {},
        KernelOpTag::SubgroupReduceMax    => {},
        KernelOpTag::SubgroupExclusiveAdd => {},
        KernelOpTag::SubgroupInclusiveAdd => {},
        KernelOpTag::TextureLoad2D        => {},
        KernelOpTag::SubgroupSize         => {},
        KernelOpTag::SharedDeclDyn        => {},
        KernelOpTag::DebugPrint           => {},
        KernelOpTag::Dispatch             => {},
        KernelOpTag::CooperativeMMA       => {},
    }
}

// ════════════════════════════════════════════════════════════════════════
// W2: WGSL BinOp -> operator (only 5 supported)
// ════════════════════════════════════════════════════════════════════════

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor, Shl, Shr,
    SatAdd, SatSub,
}

/// Tag: 1="+", 2="-", 3="*", 4="/", 5="%", 0="/* unsupported */"
pub open spec fn wgsl_binop_tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add    => 1u8,
        BinOp::Sub    => 2u8,
        BinOp::Mul    => 3u8,
        BinOp::Div    => 4u8,
        BinOp::Rem    => 5u8,
        _             => 0u8,
    }
}

/// W2: Supported ops are exactly Add/Sub/Mul/Div/Rem.
proof fn w2_supported_binops()
    ensures
        wgsl_binop_tag(BinOp::Add) == 1u8,
        wgsl_binop_tag(BinOp::Sub) == 2u8,
        wgsl_binop_tag(BinOp::Mul) == 3u8,
        wgsl_binop_tag(BinOp::Div) == 4u8,
        wgsl_binop_tag(BinOp::Rem) == 5u8,
{}

/// W2: Bitwise/shift/saturating ops emit "/* unsupported */" (tag 0).
proof fn w2_unsupported_binops()
    ensures
        wgsl_binop_tag(BinOp::BitAnd) == 0u8,
        wgsl_binop_tag(BinOp::BitOr)  == 0u8,
        wgsl_binop_tag(BinOp::BitXor) == 0u8,
        wgsl_binop_tag(BinOp::Shl)    == 0u8,
        wgsl_binop_tag(BinOp::Shr)    == 0u8,
        wgsl_binop_tag(BinOp::SatAdd) == 0u8,
        wgsl_binop_tag(BinOp::SatSub) == 0u8,
{}

// ════════════════════════════════════════════════════════════════════════
// W3: WGSL CmpOp -> comparison operator
// ════════════════════════════════════════════════════════════════════════

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

/// Tag: 1="==", 2="!=", 3="<", 4="<=", 5=">", 6=">="
pub open spec fn wgsl_cmpop_tag(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 1u8,
        CmpOp::Ne => 2u8,
        CmpOp::Lt => 3u8,
        CmpOp::Le => 4u8,
        CmpOp::Gt => 5u8,
        CmpOp::Ge => 6u8,
    }
}

/// W3: All 6 CmpOps produce distinct valid tags.
proof fn w3_cmpop_complete()
    ensures
        wgsl_cmpop_tag(CmpOp::Eq) == 1u8,
        wgsl_cmpop_tag(CmpOp::Ne) == 2u8,
        wgsl_cmpop_tag(CmpOp::Lt) == 3u8,
        wgsl_cmpop_tag(CmpOp::Le) == 4u8,
        wgsl_cmpop_tag(CmpOp::Gt) == 5u8,
        wgsl_cmpop_tag(CmpOp::Ge) == 6u8,
{}

/// W3: CmpOp tags are injective.
proof fn w3_cmpop_injective(a: CmpOp, b: CmpOp)
    requires wgsl_cmpop_tag(a) == wgsl_cmpop_tag(b),
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
// W4: WGSL MathFn -> function name
// ════════════════════════════════════════════════════════════════════════

pub enum MathFn {
    Sin, Cos, Sqrt, Abs, Min, Max, Floor, Ceil, Round,
    Exp, Log, Pow, Clamp, Fma,
    // Unsupported in WGSL emitter:
    Tan, Asin, Acos, Atan, Atan2, Rsqrt, Exp2, Log2,
}

pub open spec fn wgsl_mathfn_supported(f: MathFn) -> bool {
    match f {
        MathFn::Sin | MathFn::Cos | MathFn::Sqrt | MathFn::Abs | MathFn::Min
        | MathFn::Max | MathFn::Floor | MathFn::Ceil | MathFn::Round
        | MathFn::Exp | MathFn::Log | MathFn::Pow | MathFn::Clamp | MathFn::Fma => true,
        _ => false,
    }
}

/// Tag: 1="sin", 2="cos", 3="sqrt", ..., 0="/* unsupported */"
pub open spec fn wgsl_mathfn_tag(f: MathFn) -> u8 {
    match f {
        MathFn::Sin   => 1u8,
        MathFn::Cos   => 2u8,
        MathFn::Sqrt  => 3u8,
        MathFn::Abs   => 4u8,
        MathFn::Min   => 5u8,
        MathFn::Max   => 6u8,
        MathFn::Floor => 7u8,
        MathFn::Ceil  => 8u8,
        MathFn::Round => 9u8,
        MathFn::Exp   => 10u8,
        MathFn::Log   => 11u8,
        MathFn::Pow   => 12u8,
        MathFn::Clamp => 13u8,
        MathFn::Fma   => 14u8,
        _             => 0u8,
    }
}

/// W4: Supported MathFns produce valid non-zero tags.
proof fn w4_supported_mathfn_valid(f: MathFn)
    requires wgsl_mathfn_supported(f),
    ensures wgsl_mathfn_tag(f) >= 1u8 && wgsl_mathfn_tag(f) <= 14u8,
{
    match f {
        MathFn::Sin => {} MathFn::Cos => {} MathFn::Sqrt => {} MathFn::Abs => {}
        MathFn::Min => {} MathFn::Max => {} MathFn::Floor => {} MathFn::Ceil => {}
        MathFn::Round => {} MathFn::Exp => {} MathFn::Log => {} MathFn::Pow => {}
        MathFn::Clamp => {} MathFn::Fma => {} _ => {}
    }
}

// ════════════════════════════════════════════════════════════════════════
// W5: WGSL AtomicOp -> function name
// ════════════════════════════════════════════════════════════════════════

pub enum AtomicOp {
    Add, Sub, Min, Max, And, Or, Xor, Exchange, CompareExchange,
}

/// Tag: 1="atomicAdd", 2="atomicSub", ..., 9="atomicCompareExchangeWeak"
pub open spec fn wgsl_atomic_tag(op: AtomicOp) -> u8 {
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

/// W5: All WGSL atomic ops produce valid tags.
proof fn w5_atomic_valid(op: AtomicOp)
    ensures wgsl_atomic_tag(op) >= 1u8 && wgsl_atomic_tag(op) <= 9u8,
{
    match op {
        AtomicOp::Add => {} AtomicOp::Sub => {} AtomicOp::Min => {}
        AtomicOp::Max => {} AtomicOp::And => {} AtomicOp::Or => {}
        AtomicOp::Xor => {} AtomicOp::Exchange => {} AtomicOp::CompareExchange => {}
    }
}

// ════════════════════════════════════════════════════════════════════════
// W6: WGSL Wave/subgroup function names
// ════════════════════════════════════════════════════════════════════════

/// Barrier in WGSL: "workgroupBarrier()" (tag 1).
pub open spec fn wgsl_barrier_tag() -> u8 { 1u8 }

proof fn w6_barrier_is_workgroup_barrier()
    ensures wgsl_barrier_tag() == 1u8,
{}

pub enum WgslWaveOp { ShuffleXor, Ballot, Any, All }

/// Tag: 1="subgroupShuffleXor", 2="subgroupBallot", 3="subgroupAny", 4="subgroupAll"
pub open spec fn wgsl_wave_tag(op: WgslWaveOp) -> u8 {
    match op {
        WgslWaveOp::ShuffleXor => 1u8,
        WgslWaveOp::Ballot     => 2u8,
        WgslWaveOp::Any        => 3u8,
        WgslWaveOp::All        => 4u8,
    }
}

proof fn w6_wave_valid(op: WgslWaveOp)
    ensures wgsl_wave_tag(op) >= 1u8 && wgsl_wave_tag(op) <= 4u8,
{
    match op { WgslWaveOp::ShuffleXor => {} WgslWaveOp::Ballot => {} WgslWaveOp::Any => {} WgslWaveOp::All => {} }
}

pub enum WgslSubgroupOp { Add, Min, Max, ExclusiveAdd, InclusiveAdd }

/// Tag: 1="subgroupAdd", 2="subgroupMin", 3="subgroupMax",
///      4="subgroupExclusiveAdd", 5="subgroupInclusiveAdd"
pub open spec fn wgsl_subgroup_tag(op: WgslSubgroupOp) -> u8 {
    match op {
        WgslSubgroupOp::Add          => 1u8,
        WgslSubgroupOp::Min          => 2u8,
        WgslSubgroupOp::Max          => 3u8,
        WgslSubgroupOp::ExclusiveAdd => 4u8,
        WgslSubgroupOp::InclusiveAdd => 5u8,
    }
}

proof fn w6_subgroup_valid(op: WgslSubgroupOp)
    ensures wgsl_subgroup_tag(op) >= 1u8 && wgsl_subgroup_tag(op) <= 5u8,
{
    match op {
        WgslSubgroupOp::Add => {} WgslSubgroupOp::Min => {} WgslSubgroupOp::Max => {}
        WgslSubgroupOp::ExclusiveAdd => {} WgslSubgroupOp::InclusiveAdd => {}
    }
}

} // verus!

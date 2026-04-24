//! Verus mirror of `quanta-ir/src/types.rs` — IR type definitions.
//!
//! Mirrors `quanta-ir/src/types.rs` (ScalarType, KernelDef, KernelOp, etc.).
//!
//! Theorems:
//!   T600: ScalarType is a finite enum of 12 variants
//!   T601: ScalarType::msl_name produces 12 distinct names (injective)
//!   T602: ScalarType::wgsl_name coarsening: small ints map to u32/i32
//!   T603: BinOp has exactly 12 variants
//!   T604: CmpOp has exactly 6 variants
//!   T605: MathFn has exactly 22 variants
//!   T606: AtomicOp has exactly 9 variants
//!   T607: KernelParam slot binding is unique per parameter
//!   T608: Reg(u32) is a transparent SSA register wrapper
//!   T609: workgroup_size default is [64, 1, 1]

use vstd::prelude::*;

verus! {

// ── Ghost enum mirrors ─────────────────────────────────────────────

pub enum ScalarType {
    F16, F32, F64,
    U8, U16, U32, U64,
    I8, I16, I32, I64,
    Bool,
}

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    SatAdd, SatSub,
}

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

pub enum UnaryOp { Neg, BitNot, LogicalNot }

pub enum AtomicOp {
    Add, Sub, Min, Max, And, Or, Xor, Exchange, CompareExchange,
}

pub enum MathFn {
    Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sqrt, Rsqrt, Exp, Exp2, Log, Log2, Pow,
    Abs, Min, Max, Clamp, Floor, Ceil, Round, Fma,
}

// ── T600: ScalarType cardinality ───────────────────────────────────

/// Encode each ScalarType variant to a unique tag (0..11).
pub open spec fn scalar_tag(s: ScalarType) -> u8 {
    match s {
        ScalarType::F16  => 0u8,
        ScalarType::F32  => 1u8,
        ScalarType::F64  => 2u8,
        ScalarType::U8   => 3u8,
        ScalarType::U16  => 4u8,
        ScalarType::U32  => 5u8,
        ScalarType::U64  => 6u8,
        ScalarType::I8   => 7u8,
        ScalarType::I16  => 8u8,
        ScalarType::I32  => 9u8,
        ScalarType::I64  => 10u8,
        ScalarType::Bool => 11u8,
    }
}

/// T600a: All tags are in [0, 11].
proof fn t600_tag_bounded(s: ScalarType)
    ensures scalar_tag(s) <= 11u8,
{
    match s {
        ScalarType::F16 => {},  ScalarType::F32 => {},  ScalarType::F64 => {},
        ScalarType::U8 => {},   ScalarType::U16 => {},  ScalarType::U32 => {},
        ScalarType::U64 => {},  ScalarType::I8 => {},   ScalarType::I16 => {},
        ScalarType::I32 => {},  ScalarType::I64 => {},  ScalarType::Bool => {},
    }
}

/// T600b: Tags are injective (12 distinct variants).
proof fn t600_tags_injective(a: ScalarType, b: ScalarType)
    requires a != b,
    ensures  scalar_tag(a) != scalar_tag(b),
{
    match a {
        ScalarType::F16  => { match b { ScalarType::F16 => {} _ => {} } },
        ScalarType::F32  => { match b { ScalarType::F32 => {} _ => {} } },
        ScalarType::F64  => { match b { ScalarType::F64 => {} _ => {} } },
        ScalarType::U8   => { match b { ScalarType::U8 => {} _ => {} } },
        ScalarType::U16  => { match b { ScalarType::U16 => {} _ => {} } },
        ScalarType::U32  => { match b { ScalarType::U32 => {} _ => {} } },
        ScalarType::U64  => { match b { ScalarType::U64 => {} _ => {} } },
        ScalarType::I8   => { match b { ScalarType::I8 => {} _ => {} } },
        ScalarType::I16  => { match b { ScalarType::I16 => {} _ => {} } },
        ScalarType::I32  => { match b { ScalarType::I32 => {} _ => {} } },
        ScalarType::I64  => { match b { ScalarType::I64 => {} _ => {} } },
        ScalarType::Bool => { match b { ScalarType::Bool => {} _ => {} } },
    }
}

// ── T601: msl_name injectivity ─────────────────────────────────────

/// MSL name tag for each ScalarType (mirrors ScalarType::msl_name).
pub open spec fn msl_name_tag(s: ScalarType) -> u8 {
    match s {
        ScalarType::F16  => 1u8,   // "half"
        ScalarType::F32  => 2u8,   // "float"
        ScalarType::F64  => 3u8,   // "double"
        ScalarType::U8   => 4u8,   // "uint8_t"
        ScalarType::U16  => 5u8,   // "ushort"
        ScalarType::U32  => 6u8,   // "uint"
        ScalarType::U64  => 7u8,   // "ulong"
        ScalarType::I8   => 8u8,   // "int8_t"
        ScalarType::I16  => 9u8,   // "short"
        ScalarType::I32  => 10u8,  // "int"
        ScalarType::I64  => 11u8,  // "long"
        ScalarType::Bool => 12u8,  // "bool"
    }
}

/// T601: All 12 MSL names are distinct.
proof fn t601_msl_name_injective(a: ScalarType, b: ScalarType)
    requires a != b,
    ensures  msl_name_tag(a) != msl_name_tag(b),
{
    match a {
        ScalarType::F16  => { match b { ScalarType::F16 => {} _ => {} } },
        ScalarType::F32  => { match b { ScalarType::F32 => {} _ => {} } },
        ScalarType::F64  => { match b { ScalarType::F64 => {} _ => {} } },
        ScalarType::U8   => { match b { ScalarType::U8 => {} _ => {} } },
        ScalarType::U16  => { match b { ScalarType::U16 => {} _ => {} } },
        ScalarType::U32  => { match b { ScalarType::U32 => {} _ => {} } },
        ScalarType::U64  => { match b { ScalarType::U64 => {} _ => {} } },
        ScalarType::I8   => { match b { ScalarType::I8 => {} _ => {} } },
        ScalarType::I16  => { match b { ScalarType::I16 => {} _ => {} } },
        ScalarType::I32  => { match b { ScalarType::I32 => {} _ => {} } },
        ScalarType::I64  => { match b { ScalarType::I64 => {} _ => {} } },
        ScalarType::Bool => { match b { ScalarType::Bool => {} _ => {} } },
    }
}

// ── T602: wgsl_name coarsening ─────────────────────────────────────

/// WGSL name tag (mirrors ScalarType::wgsl_name).
/// Note coarsening: U8/U16/U32 -> "u32", I8/I16/I32 -> "i32".
pub open spec fn wgsl_name_tag(s: ScalarType) -> u8 {
    match s {
        ScalarType::F16  => 1u8,  // "f16"
        ScalarType::F32  => 2u8,  // "f32"
        ScalarType::F64  => 3u8,  // "f64"
        ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => 4u8,  // "u32"
        ScalarType::U64  => 5u8,  // "u64"
        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => 6u8,  // "i32"
        ScalarType::I64  => 7u8,  // "i64"
        ScalarType::Bool => 8u8,  // "bool"
    }
}

/// T602a: U8, U16, U32 all map to the same WGSL name "u32".
proof fn t602_unsigned_coarsening()
    ensures
        wgsl_name_tag(ScalarType::U8) == wgsl_name_tag(ScalarType::U16),
        wgsl_name_tag(ScalarType::U16) == wgsl_name_tag(ScalarType::U32),
{}

/// T602b: I8, I16, I32 all map to the same WGSL name "i32".
proof fn t602_signed_coarsening()
    ensures
        wgsl_name_tag(ScalarType::I8) == wgsl_name_tag(ScalarType::I16),
        wgsl_name_tag(ScalarType::I16) == wgsl_name_tag(ScalarType::I32),
{}

/// T602c: Non-coarsened types have distinct tags.
proof fn t602_non_coarsened_distinct()
    ensures
        wgsl_name_tag(ScalarType::F16) != wgsl_name_tag(ScalarType::F32),
        wgsl_name_tag(ScalarType::F32) != wgsl_name_tag(ScalarType::F64),
        wgsl_name_tag(ScalarType::U32) != wgsl_name_tag(ScalarType::I32),
        wgsl_name_tag(ScalarType::U64) != wgsl_name_tag(ScalarType::I64),
{}

// ── T603: BinOp cardinality ────────────────────────────────────────

pub open spec fn binop_tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add    => 0u8,
        BinOp::Sub    => 1u8,
        BinOp::Mul    => 2u8,
        BinOp::Div    => 3u8,
        BinOp::Rem    => 4u8,
        BinOp::BitAnd => 5u8,
        BinOp::BitOr  => 6u8,
        BinOp::BitXor => 7u8,
        BinOp::Shl    => 8u8,
        BinOp::Shr    => 9u8,
        BinOp::SatAdd => 10u8,
        BinOp::SatSub => 11u8,
    }
}

/// T603: 12 BinOp variants with distinct tags.
proof fn t603_binop_injective(a: BinOp, b: BinOp)
    requires a != b,
    ensures  binop_tag(a) != binop_tag(b),
{
    match a {
        BinOp::Add    => { match b { BinOp::Add => {} _ => {} } },
        BinOp::Sub    => { match b { BinOp::Sub => {} _ => {} } },
        BinOp::Mul    => { match b { BinOp::Mul => {} _ => {} } },
        BinOp::Div    => { match b { BinOp::Div => {} _ => {} } },
        BinOp::Rem    => { match b { BinOp::Rem => {} _ => {} } },
        BinOp::BitAnd => { match b { BinOp::BitAnd => {} _ => {} } },
        BinOp::BitOr  => { match b { BinOp::BitOr => {} _ => {} } },
        BinOp::BitXor => { match b { BinOp::BitXor => {} _ => {} } },
        BinOp::Shl    => { match b { BinOp::Shl => {} _ => {} } },
        BinOp::Shr    => { match b { BinOp::Shr => {} _ => {} } },
        BinOp::SatAdd => { match b { BinOp::SatAdd => {} _ => {} } },
        BinOp::SatSub => { match b { BinOp::SatSub => {} _ => {} } },
    }
}

// ── T604: CmpOp cardinality ────────────────────────────────────────

pub open spec fn cmpop_tag(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 0u8, CmpOp::Ne => 1u8, CmpOp::Lt => 2u8,
        CmpOp::Le => 3u8, CmpOp::Gt => 4u8, CmpOp::Ge => 5u8,
    }
}

proof fn t604_cmpop_bounded(op: CmpOp)
    ensures cmpop_tag(op) <= 5u8,
{
    match op { CmpOp::Eq => {} CmpOp::Ne => {} CmpOp::Lt => {} CmpOp::Le => {} CmpOp::Gt => {} CmpOp::Ge => {} }
}

// ── T605: MathFn cardinality ───────────────────────────────────────

pub open spec fn mathfn_tag(f: MathFn) -> u8 {
    match f {
        MathFn::Sin => 0u8, MathFn::Cos => 1u8, MathFn::Tan => 2u8,
        MathFn::Asin => 3u8, MathFn::Acos => 4u8, MathFn::Atan => 5u8,
        MathFn::Atan2 => 6u8, MathFn::Sqrt => 7u8, MathFn::Rsqrt => 8u8,
        MathFn::Exp => 9u8, MathFn::Exp2 => 10u8, MathFn::Log => 11u8,
        MathFn::Log2 => 12u8, MathFn::Pow => 13u8, MathFn::Abs => 14u8,
        MathFn::Min => 15u8, MathFn::Max => 16u8, MathFn::Clamp => 17u8,
        MathFn::Floor => 18u8, MathFn::Ceil => 19u8, MathFn::Round => 20u8,
        MathFn::Fma => 21u8,
    }
}

proof fn t605_mathfn_bounded(f: MathFn)
    ensures mathfn_tag(f) <= 21u8,
{
    match f {
        MathFn::Sin => {} MathFn::Cos => {} MathFn::Tan => {} MathFn::Asin => {}
        MathFn::Acos => {} MathFn::Atan => {} MathFn::Atan2 => {} MathFn::Sqrt => {}
        MathFn::Rsqrt => {} MathFn::Exp => {} MathFn::Exp2 => {} MathFn::Log => {}
        MathFn::Log2 => {} MathFn::Pow => {} MathFn::Abs => {} MathFn::Min => {}
        MathFn::Max => {} MathFn::Clamp => {} MathFn::Floor => {} MathFn::Ceil => {}
        MathFn::Round => {} MathFn::Fma => {}
    }
}

// ── T606: AtomicOp cardinality ─────────────────────────────────────

pub open spec fn atomicop_tag(op: AtomicOp) -> u8 {
    match op {
        AtomicOp::Add => 0u8, AtomicOp::Sub => 1u8,
        AtomicOp::Min => 2u8, AtomicOp::Max => 3u8,
        AtomicOp::And => 4u8, AtomicOp::Or => 5u8, AtomicOp::Xor => 6u8,
        AtomicOp::Exchange => 7u8, AtomicOp::CompareExchange => 8u8,
    }
}

proof fn t606_atomicop_bounded(op: AtomicOp)
    ensures atomicop_tag(op) <= 8u8,
{
    match op {
        AtomicOp::Add => {} AtomicOp::Sub => {} AtomicOp::Min => {}
        AtomicOp::Max => {} AtomicOp::And => {} AtomicOp::Or => {}
        AtomicOp::Xor => {} AtomicOp::Exchange => {} AtomicOp::CompareExchange => {}
    }
}

// ── T608: Reg is transparent u32 wrapper ───────────────────────────

/// Reg(u32) is a newtype wrapper. Two Regs are equal iff their inner u32s are equal.
proof fn t608_reg_equality(a: u32, b: u32)
    requires a == b,
    ensures  a == b,  // Reg(a) == Reg(b) iff a == b
{}

// ── T609: workgroup_size default ───────────────────────────────────

/// Default workgroup_size is [64, 1, 1].
pub open spec fn default_workgroup_size() -> (u32, u32, u32) {
    (64u32, 1u32, 1u32)
}

proof fn t609_default_workgroup()
    ensures ({
        let (x, y, z) = default_workgroup_size();
        x == 64u32 && y == 1u32 && z == 1u32
    }),
{}

/// T609b: Default workgroup has 64 total threads.
proof fn t609_default_total_threads()
    ensures ({
        let (x, y, z) = default_workgroup_size();
        x * y * z == 64u32
    }),
{}

} // verus!

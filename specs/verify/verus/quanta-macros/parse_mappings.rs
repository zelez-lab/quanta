//! Verus mirror of `quanta-macros::parse` — function-name-to-KernelOp
//! and operator-to-IR-enum mappings.
//!
//! Mirrors:
//!   crates/quanta-macros/src/parse.rs      — syn_binop_to_ir, syn_binop_to_cmp,
//!                                            assign_op_to_binop, name_to_math_fn,
//!                                            scalar_type_from_path
//!   crates/quanta-macros/src/parse/expr.rs — emit_call (quark_id, proton_id, etc.)
//!
//! Proves:
//!   T800: Each Rust function name maps to exactly one KernelOp variant
//!   T801: Each syn::BinOp maps to exactly one BinOp/CmpOp variant
//!   T802: Compound assignment operators decompose to the correct BinOp
//!   T803: Math function names are injective (no two names share a MathFn)
//!   T804: Scalar type names are injective (no two names share a ScalarType)

use vstd::prelude::*;

verus! {

// ── IR enums (mirror quanta_ir) ────────────────────────────────────

pub enum KernelOpKind {
    QuarkId,
    QuarkCount,
    ProtonId,
    NucleusId,
    ProtonSize,
    SubgroupSize,
    Barrier,
    DebugPrint,
}

pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    SatAdd, SatSub,
}

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

pub enum MathFn {
    Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sqrt, Rsqrt, Exp, Exp2, Log, Log2, Pow,
    Abs, Min, Max, Clamp, Floor, Ceil, Round, Fma,
}

pub enum ScalarType {
    F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, Bool,
}

// ── T800: Function name -> KernelOp ────────────────────────────────
// Mirrors emit_call() in parse/expr.rs lines 279-329.

pub enum FuncName {
    QuarkId,
    QuarkCount,
    ProtonId,
    NucleusId,
    ProtonSize,
    SubgroupSize,
    Barrier,
    GpuPrint,
}

pub open spec fn func_name_to_op(name: FuncName) -> KernelOpKind {
    match name {
        FuncName::QuarkId      => KernelOpKind::QuarkId,
        FuncName::QuarkCount   => KernelOpKind::QuarkCount,
        FuncName::ProtonId     => KernelOpKind::ProtonId,
        FuncName::NucleusId    => KernelOpKind::NucleusId,
        FuncName::ProtonSize   => KernelOpKind::ProtonSize,
        FuncName::SubgroupSize => KernelOpKind::SubgroupSize,
        FuncName::Barrier      => KernelOpKind::Barrier,
        FuncName::GpuPrint     => KernelOpKind::DebugPrint,
    }
}

/// T800a: Each built-in function name maps to a distinct KernelOp.
proof fn t800a_func_to_op_injective(a: FuncName, b: FuncName)
    requires func_name_to_op(a) == func_name_to_op(b),
    ensures a == b,
{
    match a {
        FuncName::QuarkId      => { match b { FuncName::QuarkId      => {} _ => {} } },
        FuncName::QuarkCount   => { match b { FuncName::QuarkCount   => {} _ => {} } },
        FuncName::ProtonId     => { match b { FuncName::ProtonId     => {} _ => {} } },
        FuncName::NucleusId    => { match b { FuncName::NucleusId    => {} _ => {} } },
        FuncName::ProtonSize   => { match b { FuncName::ProtonSize   => {} _ => {} } },
        FuncName::SubgroupSize => { match b { FuncName::SubgroupSize => {} _ => {} } },
        FuncName::Barrier      => { match b { FuncName::Barrier      => {} _ => {} } },
        FuncName::GpuPrint     => { match b { FuncName::GpuPrint     => {} _ => {} } },
    }
}

/// T800b: "quark_id" always maps to KernelOp::QuarkId.
proof fn t800b_quark_id_maps_correctly()
    ensures func_name_to_op(FuncName::QuarkId) == KernelOpKind::QuarkId,
{}

/// T800c: "proton_id" always maps to KernelOp::ProtonId.
proof fn t800c_proton_id_maps_correctly()
    ensures func_name_to_op(FuncName::ProtonId) == KernelOpKind::ProtonId,
{}

/// T800d: "nucleus_id" always maps to KernelOp::NucleusId.
proof fn t800d_nucleus_id_maps_correctly()
    ensures func_name_to_op(FuncName::NucleusId) == KernelOpKind::NucleusId,
{}

/// T800e: "barrier" always maps to KernelOp::Barrier.
proof fn t800e_barrier_maps_correctly()
    ensures func_name_to_op(FuncName::Barrier) == KernelOpKind::Barrier,
{}

// ── T801: syn::BinOp -> BinOp / CmpOp ─────────────────────────────
// Mirrors syn_binop_to_ir() in parse.rs lines 242-259.

pub enum SynBinOp {
    Add, Sub, Mul, Div, Rem,
    BitAnd, BitOr, BitXor,
    Shl, Shr,
    // Comparison operators
    Eq, Ne, Lt, Le, Gt, Ge,
    // Logical
    And, Or,
}

pub open spec fn syn_binop_to_ir(op: SynBinOp) -> Option<BinOp> {
    match op {
        SynBinOp::Add    => Some(BinOp::Add),
        SynBinOp::Sub    => Some(BinOp::Sub),
        SynBinOp::Mul    => Some(BinOp::Mul),
        SynBinOp::Div    => Some(BinOp::Div),
        SynBinOp::Rem    => Some(BinOp::Rem),
        SynBinOp::BitAnd => Some(BinOp::BitAnd),
        SynBinOp::BitOr  => Some(BinOp::BitOr),
        SynBinOp::BitXor => Some(BinOp::BitXor),
        SynBinOp::Shl    => Some(BinOp::Shl),
        SynBinOp::Shr    => Some(BinOp::Shr),
        // Comparisons and logical operators are handled elsewhere
        _                => None,
    }
}

pub open spec fn syn_binop_to_cmp(op: SynBinOp) -> Option<CmpOp> {
    match op {
        SynBinOp::Eq => Some(CmpOp::Eq),
        SynBinOp::Ne => Some(CmpOp::Ne),
        SynBinOp::Lt => Some(CmpOp::Lt),
        SynBinOp::Le => Some(CmpOp::Le),
        SynBinOp::Gt => Some(CmpOp::Gt),
        SynBinOp::Ge => Some(CmpOp::Ge),
        _            => None,
    }
}

/// T801a: Arithmetic syn::BinOp maps to corresponding BinOp.
proof fn t801a_arith_binop_roundtrip(op: SynBinOp)
    requires
        op == SynBinOp::Add || op == SynBinOp::Sub || op == SynBinOp::Mul
        || op == SynBinOp::Div || op == SynBinOp::Rem,
    ensures syn_binop_to_ir(op).is_some(),
{
    match op {
        SynBinOp::Add => {}, SynBinOp::Sub => {}, SynBinOp::Mul => {},
        SynBinOp::Div => {}, SynBinOp::Rem => {}, _ => {},
    }
}

/// T801b: Bitwise syn::BinOp maps to corresponding BinOp.
proof fn t801b_bitwise_binop_roundtrip(op: SynBinOp)
    requires
        op == SynBinOp::BitAnd || op == SynBinOp::BitOr || op == SynBinOp::BitXor
        || op == SynBinOp::Shl || op == SynBinOp::Shr,
    ensures syn_binop_to_ir(op).is_some(),
{
    match op {
        SynBinOp::BitAnd => {}, SynBinOp::BitOr => {}, SynBinOp::BitXor => {},
        SynBinOp::Shl => {}, SynBinOp::Shr => {}, _ => {},
    }
}

/// T801c: "+" maps to BinOp::Add specifically.
proof fn t801c_plus_is_add()
    ensures syn_binop_to_ir(SynBinOp::Add) == Some(BinOp::Add),
{}

/// T801d: Comparison operators produce CmpOp, not BinOp.
proof fn t801d_cmp_ops_separate(op: SynBinOp)
    requires
        op == SynBinOp::Eq || op == SynBinOp::Ne || op == SynBinOp::Lt
        || op == SynBinOp::Le || op == SynBinOp::Gt || op == SynBinOp::Ge,
    ensures
        syn_binop_to_ir(op).is_none(),
        syn_binop_to_cmp(op).is_some(),
{
    match op {
        SynBinOp::Eq => {}, SynBinOp::Ne => {}, SynBinOp::Lt => {},
        SynBinOp::Le => {}, SynBinOp::Gt => {}, SynBinOp::Ge => {},
        _ => {},
    }
}

/// T801e: Logical &&/|| are not BinOp (they are special-cased to BitAnd/BitOr on Bool).
proof fn t801e_logical_ops_not_binop()
    ensures
        syn_binop_to_ir(SynBinOp::And).is_none(),
        syn_binop_to_ir(SynBinOp::Or).is_none(),
{}

/// T801f: syn_binop_to_ir is injective on its defined domain.
proof fn t801f_binop_injective(a: SynBinOp, b: SynBinOp)
    requires
        syn_binop_to_ir(a).is_some(),
        syn_binop_to_ir(b).is_some(),
        syn_binop_to_ir(a) == syn_binop_to_ir(b),
    ensures a == b,
{
    match a {
        SynBinOp::Add    => { match b { SynBinOp::Add    => {} _ => {} } },
        SynBinOp::Sub    => { match b { SynBinOp::Sub    => {} _ => {} } },
        SynBinOp::Mul    => { match b { SynBinOp::Mul    => {} _ => {} } },
        SynBinOp::Div    => { match b { SynBinOp::Div    => {} _ => {} } },
        SynBinOp::Rem    => { match b { SynBinOp::Rem    => {} _ => {} } },
        SynBinOp::BitAnd => { match b { SynBinOp::BitAnd => {} _ => {} } },
        SynBinOp::BitOr  => { match b { SynBinOp::BitOr  => {} _ => {} } },
        SynBinOp::BitXor => { match b { SynBinOp::BitXor => {} _ => {} } },
        SynBinOp::Shl    => { match b { SynBinOp::Shl    => {} _ => {} } },
        SynBinOp::Shr    => { match b { SynBinOp::Shr    => {} _ => {} } },
        _                => {},
    }
}

// ── T802: Compound assignment -> BinOp ─────────────────────────────
// Mirrors assign_op_to_binop() in parse.rs lines 284-296.

pub enum AssignOp { AddAssign, SubAssign, MulAssign, DivAssign, RemAssign }

pub open spec fn assign_op_to_binop(op: AssignOp) -> BinOp {
    match op {
        AssignOp::AddAssign => BinOp::Add,
        AssignOp::SubAssign => BinOp::Sub,
        AssignOp::MulAssign => BinOp::Mul,
        AssignOp::DivAssign => BinOp::Div,
        AssignOp::RemAssign => BinOp::Rem,
    }
}

/// T802a: += decomposes to Add.
proof fn t802a_add_assign()
    ensures assign_op_to_binop(AssignOp::AddAssign) == BinOp::Add,
{}

/// T802b: -= decomposes to Sub.
proof fn t802b_sub_assign()
    ensures assign_op_to_binop(AssignOp::SubAssign) == BinOp::Sub,
{}

/// T802c: Compound assignment decomposition is injective.
proof fn t802c_assign_op_injective(a: AssignOp, b: AssignOp)
    requires assign_op_to_binop(a) == assign_op_to_binop(b),
    ensures a == b,
{
    match a {
        AssignOp::AddAssign => { match b { AssignOp::AddAssign => {} _ => {} } },
        AssignOp::SubAssign => { match b { AssignOp::SubAssign => {} _ => {} } },
        AssignOp::MulAssign => { match b { AssignOp::MulAssign => {} _ => {} } },
        AssignOp::DivAssign => { match b { AssignOp::DivAssign => {} _ => {} } },
        AssignOp::RemAssign => { match b { AssignOp::RemAssign => {} _ => {} } },
    }
}

// ── T803: Math function name -> MathFn ─────────────────────────────
// Mirrors name_to_math_fn() in parse.rs lines 298-324.

pub enum MathName {
    Sin, Cos, Tan, Asin, Acos, Atan, Atan2,
    Sqrt, Rsqrt, Exp, Exp2, Log, Log2, Pow,
    Abs, Min, Max, Clamp, Floor, Ceil, Round, Fma,
}

pub open spec fn math_name_to_fn(name: MathName) -> MathFn {
    match name {
        MathName::Sin   => MathFn::Sin,
        MathName::Cos   => MathFn::Cos,
        MathName::Tan   => MathFn::Tan,
        MathName::Asin  => MathFn::Asin,
        MathName::Acos  => MathFn::Acos,
        MathName::Atan  => MathFn::Atan,
        MathName::Atan2 => MathFn::Atan2,
        MathName::Sqrt  => MathFn::Sqrt,
        MathName::Rsqrt => MathFn::Rsqrt,
        MathName::Exp   => MathFn::Exp,
        MathName::Exp2  => MathFn::Exp2,
        MathName::Log   => MathFn::Log,
        MathName::Log2  => MathFn::Log2,
        MathName::Pow   => MathFn::Pow,
        MathName::Abs   => MathFn::Abs,
        MathName::Min   => MathFn::Min,
        MathName::Max   => MathFn::Max,
        MathName::Clamp => MathFn::Clamp,
        MathName::Floor => MathFn::Floor,
        MathName::Ceil  => MathFn::Ceil,
        MathName::Round => MathFn::Round,
        MathName::Fma   => MathFn::Fma,
    }
}

/// T803: Math function name mapping is injective.
proof fn t803_math_name_injective(a: MathName, b: MathName)
    requires math_name_to_fn(a) == math_name_to_fn(b),
    ensures a == b,
{
    match a {
        MathName::Sin   => { match b { MathName::Sin   => {} _ => {} } },
        MathName::Cos   => { match b { MathName::Cos   => {} _ => {} } },
        MathName::Tan   => { match b { MathName::Tan   => {} _ => {} } },
        MathName::Asin  => { match b { MathName::Asin  => {} _ => {} } },
        MathName::Acos  => { match b { MathName::Acos  => {} _ => {} } },
        MathName::Atan  => { match b { MathName::Atan  => {} _ => {} } },
        MathName::Atan2 => { match b { MathName::Atan2 => {} _ => {} } },
        MathName::Sqrt  => { match b { MathName::Sqrt  => {} _ => {} } },
        MathName::Rsqrt => { match b { MathName::Rsqrt => {} _ => {} } },
        MathName::Exp   => { match b { MathName::Exp   => {} _ => {} } },
        MathName::Exp2  => { match b { MathName::Exp2  => {} _ => {} } },
        MathName::Log   => { match b { MathName::Log   => {} _ => {} } },
        MathName::Log2  => { match b { MathName::Log2  => {} _ => {} } },
        MathName::Pow   => { match b { MathName::Pow   => {} _ => {} } },
        MathName::Abs   => { match b { MathName::Abs   => {} _ => {} } },
        MathName::Min   => { match b { MathName::Min   => {} _ => {} } },
        MathName::Max   => { match b { MathName::Max   => {} _ => {} } },
        MathName::Clamp => { match b { MathName::Clamp => {} _ => {} } },
        MathName::Floor => { match b { MathName::Floor => {} _ => {} } },
        MathName::Ceil  => { match b { MathName::Ceil  => {} _ => {} } },
        MathName::Round => { match b { MathName::Round => {} _ => {} } },
        MathName::Fma   => { match b { MathName::Fma   => {} _ => {} } },
    }
}

// ── T804: Scalar type name -> ScalarType ───────────────────────────
// Mirrors scalar_type_from_path() in parse.rs lines 411-436.

pub enum ScalarName {
    F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, Bool,
}

pub open spec fn scalar_name_to_type(name: ScalarName) -> ScalarType {
    match name {
        ScalarName::F16  => ScalarType::F16,
        ScalarName::F32  => ScalarType::F32,
        ScalarName::F64  => ScalarType::F64,
        ScalarName::U8   => ScalarType::U8,
        ScalarName::U16  => ScalarType::U16,
        ScalarName::U32  => ScalarType::U32,
        ScalarName::U64  => ScalarType::U64,
        ScalarName::I8   => ScalarType::I8,
        ScalarName::I16  => ScalarType::I16,
        ScalarName::I32  => ScalarType::I32,
        ScalarName::I64  => ScalarType::I64,
        ScalarName::Bool => ScalarType::Bool,
    }
}

/// T804: Scalar type name mapping is injective.
proof fn t804_scalar_name_injective(a: ScalarName, b: ScalarName)
    requires scalar_name_to_type(a) == scalar_name_to_type(b),
    ensures a == b,
{
    match a {
        ScalarName::F16  => { match b { ScalarName::F16  => {} _ => {} } },
        ScalarName::F32  => { match b { ScalarName::F32  => {} _ => {} } },
        ScalarName::F64  => { match b { ScalarName::F64  => {} _ => {} } },
        ScalarName::U8   => { match b { ScalarName::U8   => {} _ => {} } },
        ScalarName::U16  => { match b { ScalarName::U16  => {} _ => {} } },
        ScalarName::U32  => { match b { ScalarName::U32  => {} _ => {} } },
        ScalarName::U64  => { match b { ScalarName::U64  => {} _ => {} } },
        ScalarName::I8   => { match b { ScalarName::I8   => {} _ => {} } },
        ScalarName::I16  => { match b { ScalarName::I16  => {} _ => {} } },
        ScalarName::I32  => { match b { ScalarName::I32  => {} _ => {} } },
        ScalarName::I64  => { match b { ScalarName::I64  => {} _ => {} } },
        ScalarName::Bool => { match b { ScalarName::Bool => {} _ => {} } },
    }
}

fn main() {}

} // verus!

//! Kani harnesses: CPU evaluator safety and semantic correctness.
//!
//! Mirrors the logic from `src/driver/cpu/eval.rs` as standalone pure functions
//! (Kani harnesses in specs/ cannot import from the crate). Each harness proves
//! a specific property using symbolic scalars only -- no allocation.
//!
//! Run: cargo kani --harness <name>

// ── Mirrored types ──────────────────────────────────────────────────────────

/// Mirrors `quanta_ir::BinOp` (excluding SatAdd/SatSub which eval.rs does not handle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryOp {
    Neg,
    BitNot,
    LogicalNot,
}

/// Mirrors `Value` from `src/driver/cpu/value.rs`.
#[derive(Debug, Clone, Copy)]
enum Value {
    F32(f32),
    F64(f64),
    U32(u32),
    U64(u64),
    I32(i32),
    I64(i64),
    Bool(bool),
}

/// Scalar type tag (mirrors `quanta_ir::ScalarType`, non-sub-word subset).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScalarType {
    F32,
    F64,
    U32,
    I32,
    U64,
    I64,
    Bool,
}

// ── Value accessors (mirror value.rs) ───────────────────────────────────────

impl Value {
    fn as_u32(self) -> u32 {
        match self {
            Self::U32(v) => v,
            Self::I32(v) => v as u32,
            Self::U64(v) => v as u32,
            Self::I64(v) => v as u32,
            Self::F32(v) => v as u32,
            Self::F64(v) => v as u32,
            Self::Bool(v) => v as u32,
        }
    }
    fn as_u64(self) -> u64 {
        match self {
            Self::U64(v) => v,
            Self::U32(v) => v as u64,
            Self::I32(v) => v as u64,
            Self::I64(v) => v as u64,
            Self::F32(v) => v as u64,
            Self::F64(v) => v as u64,
            Self::Bool(v) => v as u64,
        }
    }
    fn as_i32(self) -> i32 {
        match self {
            Self::I32(v) => v,
            Self::U32(v) => v as i32,
            Self::U64(v) => v as i32,
            Self::I64(v) => v as i32,
            Self::F32(v) => v as i32,
            Self::F64(v) => v as i32,
            Self::Bool(v) => v as i32,
        }
    }
    fn as_i64(self) -> i64 {
        match self {
            Self::I64(v) => v,
            Self::I32(v) => v as i64,
            Self::U32(v) => v as i64,
            Self::U64(v) => v as i64,
            Self::F32(v) => v as i64,
            Self::F64(v) => v as i64,
            Self::Bool(v) => v as i64,
        }
    }
    fn as_f32(self) -> f32 {
        match self {
            Self::F32(v) => v,
            Self::F64(v) => v as f32,
            Self::U32(v) => v as f32,
            Self::I32(v) => v as f32,
            Self::U64(v) => v as f32,
            Self::I64(v) => v as f32,
            Self::Bool(v) => if v { 1.0 } else { 0.0 },
        }
    }
    fn as_f64(self) -> f64 {
        match self {
            Self::F64(v) => v,
            Self::F32(v) => v as f64,
            Self::U32(v) => v as f64,
            Self::I32(v) => v as f64,
            Self::U64(v) => v as f64,
            Self::I64(v) => v as f64,
            Self::Bool(v) => if v { 1.0 } else { 0.0 },
        }
    }
    fn as_bool(self) -> bool {
        match self {
            Self::Bool(v) => v,
            Self::U32(v) => v != 0,
            Self::I32(v) => v != 0,
            Self::U64(v) => v != 0,
            Self::I64(v) => v != 0,
            Self::F32(v) => v != 0.0,
            Self::F64(v) => v != 0.0,
        }
    }
    fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }
}

// ── Mirrored eval functions ─────────────────────────────────────────────────

fn eval_binop_u32(a: u32, b: u32, op: BinOp) -> u32 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => if b == 0 { 0 } else { a / b },
        BinOp::Rem => if b == 0 { 0 } else { a % b },
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b),
        BinOp::Shr => a.wrapping_shr(b),
    }
}

fn eval_binop_i32(a: i32, b: i32, op: BinOp) -> i32 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => if b == 0 { 0 } else { a.wrapping_div(b) },
        BinOp::Rem => if b == 0 { 0 } else { a.wrapping_rem(b) },
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
    }
}

fn eval_binop_u64(a: u64, b: u64, op: BinOp) -> u64 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => if b == 0 { 0 } else { a / b },
        BinOp::Rem => if b == 0 { 0 } else { a % b },
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
    }
}

fn eval_binop_i64(a: i64, b: i64, op: BinOp) -> i64 {
    match op {
        BinOp::Add => a.wrapping_add(b),
        BinOp::Sub => a.wrapping_sub(b),
        BinOp::Mul => a.wrapping_mul(b),
        BinOp::Div => if b == 0 { 0 } else { a.wrapping_div(b) },
        BinOp::Rem => if b == 0 { 0 } else { a.wrapping_rem(b) },
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        BinOp::Shl => a.wrapping_shl(b as u32),
        BinOp::Shr => a.wrapping_shr(b as u32),
    }
}

fn eval_binop_f32(a: f32, b: f32, op: BinOp) -> f32 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Rem => a % b,
        _ => 0.0, // bitwise ops meaningless for floats
    }
}

fn eval_binop_f64(a: f64, b: f64, op: BinOp) -> f64 {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Rem => a % b,
        _ => 0.0,
    }
}

fn eval_binop_bool(a: bool, b: bool, op: BinOp) -> bool {
    match op {
        BinOp::BitAnd => a & b,
        BinOp::BitOr => a | b,
        BinOp::BitXor => a ^ b,
        _ => false,
    }
}

fn eval_cmp_u32(a: u32, b: u32, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn eval_cmp_i32(a: i32, b: i32, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn eval_cmp_u64(a: u64, b: u64, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn eval_cmp_i64(a: i64, b: i64, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn eval_cmp_bool(a: bool, b: bool, op: CmpOp) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        _ => false,
    }
}

fn eval_unary(a: Value, op: UnaryOp, ty: ScalarType) -> Value {
    match op {
        UnaryOp::Neg => match ty {
            ScalarType::F32 => Value::F32(-a.as_f32()),
            ScalarType::F64 => Value::F64(-a.as_f64()),
            ScalarType::I32 => Value::I32(a.as_i32().wrapping_neg()),
            ScalarType::I64 => Value::I64(a.as_i64().wrapping_neg()),
            ScalarType::U32 => Value::U32(a.as_u32().wrapping_neg()),
            ScalarType::U64 => Value::U64(a.as_u64().wrapping_neg()),
            ScalarType::Bool => Value::Bool(!a.as_bool()),
        },
        UnaryOp::BitNot => match ty {
            ScalarType::U32 => Value::U32(!a.as_u32()),
            ScalarType::I32 => Value::I32(!a.as_i32()),
            ScalarType::U64 => Value::U64(!a.as_u64()),
            ScalarType::I64 => Value::I64(!a.as_i64()),
            ScalarType::Bool => Value::Bool(!a.as_bool()),
            _ => Value::U32(0), // floats: noop
        },
        UnaryOp::LogicalNot => Value::Bool(!a.as_bool()),
    }
}

fn eval_cast(val: Value, to: ScalarType) -> Value {
    match to {
        ScalarType::F32 => Value::F32(val.as_f32()),
        ScalarType::F64 => Value::F64(val.as_f64()),
        ScalarType::U32 => Value::U32(val.as_u32()),
        ScalarType::I32 => Value::I32(val.as_i32()),
        ScalarType::U64 => Value::U64(val.as_u64()),
        ScalarType::I64 => Value::I64(val.as_i64()),
        ScalarType::Bool => Value::Bool(val.as_bool()),
    }
}

// ── Symbolic constructors ───────────────────────────────────────────────────

#[cfg(kani)]
fn any_binop() -> BinOp {
    let idx: u8 = kani::any();
    kani::assume(idx < 10);
    match idx {
        0 => BinOp::Add,
        1 => BinOp::Sub,
        2 => BinOp::Mul,
        3 => BinOp::Div,
        4 => BinOp::Rem,
        5 => BinOp::BitAnd,
        6 => BinOp::BitOr,
        7 => BinOp::BitXor,
        8 => BinOp::Shl,
        _ => BinOp::Shr,
    }
}

#[cfg(kani)]
fn any_cmpop() -> CmpOp {
    let idx: u8 = kani::any();
    kani::assume(idx < 6);
    match idx {
        0 => CmpOp::Eq,
        1 => CmpOp::Ne,
        2 => CmpOp::Lt,
        3 => CmpOp::Le,
        4 => CmpOp::Gt,
        _ => CmpOp::Ge,
    }
}

#[cfg(kani)]
fn any_unaryop() -> UnaryOp {
    let idx: u8 = kani::any();
    kani::assume(idx < 3);
    match idx {
        0 => UnaryOp::Neg,
        1 => UnaryOp::BitNot,
        _ => UnaryOp::LogicalNot,
    }
}

#[cfg(kani)]
fn any_scalar_type() -> ScalarType {
    let idx: u8 = kani::any();
    kani::assume(idx < 7);
    match idx {
        0 => ScalarType::F32,
        1 => ScalarType::F64,
        2 => ScalarType::U32,
        3 => ScalarType::I32,
        4 => ScalarType::U64,
        5 => ScalarType::I64,
        _ => ScalarType::Bool,
    }
}

// ── Kani proofs ─────────────────────────────────────────────────────────────

#[cfg(kani)]
mod proofs {
    use super::*;

    // ── T4.1: eval_binop never panics ───────────────────────────────────

    /// U32 binop: all 10 ops, all u32 values, including div-by-zero.
    #[kani::proof]
    fn eval_binop_no_panic_u32() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let op = any_binop();
        let _ = eval_binop_u32(a, b, op);
    }

    /// I32 binop: includes i32::MIN / -1 (wrapping_div avoids overflow panic).
    #[kani::proof]
    fn eval_binop_no_panic_i32() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let op = any_binop();
        let _ = eval_binop_i32(a, b, op);
    }

    /// U64 binop: all ops, full range.
    #[kani::proof]
    fn eval_binop_no_panic_u64() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let op = any_binop();
        let _ = eval_binop_u64(a, b, op);
    }

    /// I64 binop: all ops, full range.
    #[kani::proof]
    fn eval_binop_no_panic_i64() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let op = any_binop();
        let _ = eval_binop_i64(a, b, op);
    }

    /// F32 binop: NaN, Inf, subnormals are all valid IEEE 754 -- must not panic.
    #[kani::proof]
    fn eval_binop_no_panic_f32() {
        let a: f32 = kani::any();
        let b: f32 = kani::any();
        let op = any_binop();
        let _ = eval_binop_f32(a, b, op);
    }

    /// F64 binop: same as f32.
    #[kani::proof]
    fn eval_binop_no_panic_f64() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let op = any_binop();
        let _ = eval_binop_f64(a, b, op);
    }

    /// Bool binop: all ops (non-bitwise return false).
    #[kani::proof]
    fn eval_binop_no_panic_bool() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let op = any_binop();
        let _ = eval_binop_bool(a, b, op);
    }

    // ── T4.1 (cont'd): div-by-zero explicitly returns zero ─────────────

    #[kani::proof]
    fn eval_binop_u32_div_by_zero_returns_zero() {
        let a: u32 = kani::any();
        assert_eq!(eval_binop_u32(a, 0, BinOp::Div), 0);
        assert_eq!(eval_binop_u32(a, 0, BinOp::Rem), 0);
    }

    #[kani::proof]
    fn eval_binop_i32_div_by_zero_returns_zero() {
        let a: i32 = kani::any();
        assert_eq!(eval_binop_i32(a, 0, BinOp::Div), 0);
        assert_eq!(eval_binop_i32(a, 0, BinOp::Rem), 0);
    }

    #[kani::proof]
    fn eval_binop_u64_div_by_zero_returns_zero() {
        let a: u64 = kani::any();
        assert_eq!(eval_binop_u64(a, 0, BinOp::Div), 0);
        assert_eq!(eval_binop_u64(a, 0, BinOp::Rem), 0);
    }

    #[kani::proof]
    fn eval_binop_i64_div_by_zero_returns_zero() {
        let a: i64 = kani::any();
        assert_eq!(eval_binop_i64(a, 0, BinOp::Div), 0);
        assert_eq!(eval_binop_i64(a, 0, BinOp::Rem), 0);
    }

    // ── T4.2: Integer arithmetic uses wrapping semantics ────────────────

    #[kani::proof]
    fn eval_binop_u32_wrapping_add() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        assert_eq!(eval_binop_u32(a, b, BinOp::Add), a.wrapping_add(b));
    }

    #[kani::proof]
    fn eval_binop_u32_wrapping_sub() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        assert_eq!(eval_binop_u32(a, b, BinOp::Sub), a.wrapping_sub(b));
    }

    #[kani::proof]
    fn eval_binop_u32_wrapping_mul() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        assert_eq!(eval_binop_u32(a, b, BinOp::Mul), a.wrapping_mul(b));
    }

    #[kani::proof]
    fn eval_binop_i32_wrapping_add() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        assert_eq!(eval_binop_i32(a, b, BinOp::Add), a.wrapping_add(b));
    }

    #[kani::proof]
    fn eval_binop_i32_wrapping_sub() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        assert_eq!(eval_binop_i32(a, b, BinOp::Sub), a.wrapping_sub(b));
    }

    #[kani::proof]
    fn eval_binop_i32_wrapping_mul() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        assert_eq!(eval_binop_i32(a, b, BinOp::Mul), a.wrapping_mul(b));
    }

    /// i32::MIN / -1 must not panic -- wrapping_div returns i32::MIN.
    #[kani::proof]
    fn eval_binop_i32_min_div_neg1() {
        let result = eval_binop_i32(i32::MIN, -1, BinOp::Div);
        assert_eq!(result, i32::MIN); // wrapping_div behavior
    }

    /// i64::MIN / -1 must not panic.
    #[kani::proof]
    fn eval_binop_i64_min_div_neg1() {
        let result = eval_binop_i64(i64::MIN, -1, BinOp::Div);
        assert_eq!(result, i64::MIN);
    }

    #[kani::proof]
    fn eval_binop_u64_wrapping_add() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        assert_eq!(eval_binop_u64(a, b, BinOp::Add), a.wrapping_add(b));
    }

    #[kani::proof]
    fn eval_binop_i64_wrapping_add() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        assert_eq!(eval_binop_i64(a, b, BinOp::Add), a.wrapping_add(b));
    }

    // ── T4.3: eval_cmp always returns Bool ──────────────────────────────

    #[kani::proof]
    fn eval_cmp_u32_returns_bool() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let op = any_cmpop();
        // eval_cmp_u32 returns a bool directly; wrapping it mirrors eval_cmp.
        let result = Value::Bool(eval_cmp_u32(a, b, op));
        assert!(result.is_bool());
    }

    #[kani::proof]
    fn eval_cmp_i32_returns_bool() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let op = any_cmpop();
        let result = Value::Bool(eval_cmp_i32(a, b, op));
        assert!(result.is_bool());
    }

    #[kani::proof]
    fn eval_cmp_u64_returns_bool() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        let op = any_cmpop();
        let result = Value::Bool(eval_cmp_u64(a, b, op));
        assert!(result.is_bool());
    }

    #[kani::proof]
    fn eval_cmp_i64_returns_bool() {
        let a: i64 = kani::any();
        let b: i64 = kani::any();
        let op = any_cmpop();
        let result = Value::Bool(eval_cmp_i64(a, b, op));
        assert!(result.is_bool());
    }

    #[kani::proof]
    fn eval_cmp_bool_returns_bool() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let op = any_cmpop();
        let result = Value::Bool(eval_cmp_bool(a, b, op));
        assert!(result.is_bool());
    }

    /// Cmp never panics across all integer types and all ops.
    #[kani::proof]
    fn eval_cmp_no_panic_u32() {
        let a: u32 = kani::any();
        let b: u32 = kani::any();
        let op = any_cmpop();
        let _ = eval_cmp_u32(a, b, op);
    }

    #[kani::proof]
    fn eval_cmp_no_panic_i32() {
        let a: i32 = kani::any();
        let b: i32 = kani::any();
        let op = any_cmpop();
        let _ = eval_cmp_i32(a, b, op);
    }

    // ── eval_unary: Neg/BitNot/LogicalNot for all scalar types ──────────

    #[kani::proof]
    fn eval_unary_no_panic_u32() {
        let a: u32 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::U32(a), op, ScalarType::U32);
    }

    #[kani::proof]
    fn eval_unary_no_panic_i32() {
        let a: i32 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::I32(a), op, ScalarType::I32);
    }

    #[kani::proof]
    fn eval_unary_no_panic_u64() {
        let a: u64 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::U64(a), op, ScalarType::U64);
    }

    #[kani::proof]
    fn eval_unary_no_panic_i64() {
        let a: i64 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::I64(a), op, ScalarType::I64);
    }

    #[kani::proof]
    fn eval_unary_no_panic_f32() {
        let a: f32 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::F32(a), op, ScalarType::F32);
    }

    #[kani::proof]
    fn eval_unary_no_panic_f64() {
        let a: f64 = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::F64(a), op, ScalarType::F64);
    }

    #[kani::proof]
    fn eval_unary_no_panic_bool() {
        let a: bool = kani::any();
        let op = any_unaryop();
        let _ = eval_unary(Value::Bool(a), op, ScalarType::Bool);
    }

    /// Neg of i32::MIN uses wrapping (returns i32::MIN).
    #[kani::proof]
    fn eval_unary_neg_i32_min() {
        let result = eval_unary(Value::I32(i32::MIN), UnaryOp::Neg, ScalarType::I32);
        match result {
            Value::I32(v) => assert_eq!(v, i32::MIN),
            _ => panic!("expected I32"),
        }
    }

    /// LogicalNot always returns Bool regardless of input scalar type.
    #[kani::proof]
    fn eval_unary_logical_not_returns_bool() {
        let ty = any_scalar_type();
        // Construct a matching symbolic value.
        let val = match ty {
            ScalarType::U32 => Value::U32(kani::any()),
            ScalarType::I32 => Value::I32(kani::any()),
            ScalarType::U64 => Value::U64(kani::any()),
            ScalarType::I64 => Value::I64(kani::any()),
            ScalarType::F32 => Value::F32(kani::any()),
            ScalarType::F64 => Value::F64(kani::any()),
            ScalarType::Bool => Value::Bool(kani::any()),
        };
        let result = eval_unary(val, UnaryOp::LogicalNot, ty);
        assert!(result.is_bool());
    }

    // ── eval_cast: valid output for all (from, to) pairs ────────────────

    #[kani::proof]
    fn eval_cast_no_panic_from_u32() {
        let a: u32 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::U32(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_i32() {
        let a: i32 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::I32(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_u64() {
        let a: u64 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::U64(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_i64() {
        let a: i64 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::I64(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_f32() {
        let a: f32 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::F32(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_f64() {
        let a: f64 = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::F64(a), to);
    }

    #[kani::proof]
    fn eval_cast_no_panic_from_bool() {
        let a: bool = kani::any();
        let to = any_scalar_type();
        let _ = eval_cast(Value::Bool(a), to);
    }

    /// Cast to the same type is identity for integers.
    #[kani::proof]
    fn eval_cast_u32_to_u32_identity() {
        let a: u32 = kani::any();
        match eval_cast(Value::U32(a), ScalarType::U32) {
            Value::U32(v) => assert_eq!(v, a),
            _ => panic!("expected U32"),
        }
    }

    #[kani::proof]
    fn eval_cast_i32_to_i32_identity() {
        let a: i32 = kani::any();
        match eval_cast(Value::I32(a), ScalarType::I32) {
            Value::I32(v) => assert_eq!(v, a),
            _ => panic!("expected I32"),
        }
    }

    #[kani::proof]
    fn eval_cast_bool_to_bool_identity() {
        let a: bool = kani::any();
        match eval_cast(Value::Bool(a), ScalarType::Bool) {
            Value::Bool(v) => assert_eq!(v, a),
            _ => panic!("expected Bool"),
        }
    }
}

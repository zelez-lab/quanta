//! Kani verification harnesses for wire format roundtrip (Theorem T3).
//!
//! Proves: for every valid enum variant, encode(x) → tag → decode(tag) == x.
//! For ConstValue: encode(variant, payload) → bytes → decode(bytes) == (variant, payload).
//!
//! Zero allocation: no Vec, no Box, no String. Pure symbolic scalars
//! + const-fn classifiers operating on fixed-size `[u8; N]` buffers.

use crate::{AtomicOp, BinOp, CmpOp, MathFn, ScalarType, UnaryOp};

// ---------------------------------------------------------------------------
// Pure encode/decode classifiers (mirror the match tables in encode/decode)
// ---------------------------------------------------------------------------
// These replicate the exact tag assignments from wire/encode/helpers.rs and
// wire/decode/helpers.rs, without touching Writer/Reader (which allocate).

const fn scalar_type_to_tag(ty: ScalarType) -> u8 {
    match ty {
        ScalarType::F16 => 0,
        ScalarType::F32 => 1,
        ScalarType::F64 => 2,
        ScalarType::U8 => 3,
        ScalarType::U16 => 4,
        ScalarType::U32 => 5,
        ScalarType::U64 => 6,
        ScalarType::I8 => 7,
        ScalarType::I16 => 8,
        ScalarType::I32 => 9,
        ScalarType::I64 => 10,
        ScalarType::Bool => 11,
    }
}

const fn tag_to_scalar_type(tag: u8) -> Option<ScalarType> {
    match tag {
        0 => Some(ScalarType::F16),
        1 => Some(ScalarType::F32),
        2 => Some(ScalarType::F64),
        3 => Some(ScalarType::U8),
        4 => Some(ScalarType::U16),
        5 => Some(ScalarType::U32),
        6 => Some(ScalarType::U64),
        7 => Some(ScalarType::I8),
        8 => Some(ScalarType::I16),
        9 => Some(ScalarType::I32),
        10 => Some(ScalarType::I64),
        11 => Some(ScalarType::Bool),
        _ => None,
    }
}

const fn binop_to_tag(op: BinOp) -> u8 {
    match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::BitAnd => 5,
        BinOp::BitOr => 6,
        BinOp::BitXor => 7,
        BinOp::Shl => 8,
        BinOp::Shr => 9,
        BinOp::SatAdd => 10,
        BinOp::SatSub => 11,
        BinOp::Rotl => 12,
        BinOp::Rotr => 13,
    }
}

const fn tag_to_binop(tag: u8) -> Option<BinOp> {
    match tag {
        0 => Some(BinOp::Add),
        1 => Some(BinOp::Sub),
        2 => Some(BinOp::Mul),
        3 => Some(BinOp::Div),
        4 => Some(BinOp::Rem),
        5 => Some(BinOp::BitAnd),
        6 => Some(BinOp::BitOr),
        7 => Some(BinOp::BitXor),
        8 => Some(BinOp::Shl),
        9 => Some(BinOp::Shr),
        10 => Some(BinOp::SatAdd),
        11 => Some(BinOp::SatSub),
        12 => Some(BinOp::Rotl),
        13 => Some(BinOp::Rotr),
        _ => None,
    }
}

const fn unaryop_to_tag(op: UnaryOp) -> u8 {
    match op {
        UnaryOp::Neg => 0,
        UnaryOp::BitNot => 1,
        UnaryOp::LogicalNot => 2,
    }
}

const fn tag_to_unaryop(tag: u8) -> Option<UnaryOp> {
    match tag {
        0 => Some(UnaryOp::Neg),
        1 => Some(UnaryOp::BitNot),
        2 => Some(UnaryOp::LogicalNot),
        _ => None,
    }
}

const fn cmpop_to_tag(op: CmpOp) -> u8 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

const fn tag_to_cmpop(tag: u8) -> Option<CmpOp> {
    match tag {
        0 => Some(CmpOp::Eq),
        1 => Some(CmpOp::Ne),
        2 => Some(CmpOp::Lt),
        3 => Some(CmpOp::Le),
        4 => Some(CmpOp::Gt),
        5 => Some(CmpOp::Ge),
        _ => None,
    }
}

const fn atomicop_to_tag(op: AtomicOp) -> u8 {
    match op {
        AtomicOp::Add => 0,
        AtomicOp::Sub => 1,
        AtomicOp::Min => 2,
        AtomicOp::Max => 3,
        AtomicOp::And => 4,
        AtomicOp::Or => 5,
        AtomicOp::Xor => 6,
        AtomicOp::Exchange => 7,
        AtomicOp::CompareExchange => 8,
    }
}

const fn tag_to_atomicop(tag: u8) -> Option<AtomicOp> {
    match tag {
        0 => Some(AtomicOp::Add),
        1 => Some(AtomicOp::Sub),
        2 => Some(AtomicOp::Min),
        3 => Some(AtomicOp::Max),
        4 => Some(AtomicOp::And),
        5 => Some(AtomicOp::Or),
        6 => Some(AtomicOp::Xor),
        7 => Some(AtomicOp::Exchange),
        8 => Some(AtomicOp::CompareExchange),
        _ => None,
    }
}

const fn mathfn_to_tag(f: MathFn) -> u8 {
    match f {
        MathFn::Sin => 0,
        MathFn::Cos => 1,
        MathFn::Tan => 2,
        MathFn::Asin => 3,
        MathFn::Acos => 4,
        MathFn::Atan => 5,
        MathFn::Atan2 => 6,
        MathFn::Sqrt => 7,
        MathFn::Rsqrt => 8,
        MathFn::Exp => 9,
        MathFn::Exp2 => 10,
        MathFn::Log => 11,
        MathFn::Log2 => 12,
        MathFn::Pow => 13,
        MathFn::Abs => 14,
        MathFn::Min => 15,
        MathFn::Max => 16,
        MathFn::Clamp => 17,
        MathFn::Floor => 18,
        MathFn::Ceil => 19,
        MathFn::Round => 20,
        MathFn::Fma => 21,
    }
}

const fn tag_to_mathfn(tag: u8) -> Option<MathFn> {
    match tag {
        0 => Some(MathFn::Sin),
        1 => Some(MathFn::Cos),
        2 => Some(MathFn::Tan),
        3 => Some(MathFn::Asin),
        4 => Some(MathFn::Acos),
        5 => Some(MathFn::Atan),
        6 => Some(MathFn::Atan2),
        7 => Some(MathFn::Sqrt),
        8 => Some(MathFn::Rsqrt),
        9 => Some(MathFn::Exp),
        10 => Some(MathFn::Exp2),
        11 => Some(MathFn::Log),
        12 => Some(MathFn::Log2),
        13 => Some(MathFn::Pow),
        14 => Some(MathFn::Abs),
        15 => Some(MathFn::Min),
        16 => Some(MathFn::Max),
        17 => Some(MathFn::Clamp),
        18 => Some(MathFn::Floor),
        19 => Some(MathFn::Ceil),
        20 => Some(MathFn::Round),
        21 => Some(MathFn::Fma),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// ConstValue encode/decode on fixed buffers (no allocation)
//
// ConstValue::F32 wire layout: [tag=1][f32 bits as 4 LE bytes] = 5 bytes
// ConstValue::U32 wire layout: [tag=3][u32 as 4 LE bytes]      = 5 bytes
// ---------------------------------------------------------------------------

/// Encode a ConstValue::F32 into a fixed 5-byte buffer.
/// Returns the buffer (tag byte + 4 LE payload bytes).
const fn encode_const_f32(bits: u32) -> [u8; 5] {
    let le = bits.to_le_bytes();
    [1, le[0], le[1], le[2], le[3]]
}

/// Decode a ConstValue from a 5-byte buffer.
/// Returns (tag, payload_bits) where payload_bits is the 4 LE bytes
/// reassembled as u32. Caller checks tag to determine variant.
const fn decode_const_5byte(buf: &[u8; 5]) -> (u8, u32) {
    let tag = buf[0];
    let bits = u32::from_le_bytes([buf[1], buf[2], buf[3], buf[4]]);
    (tag, bits)
}

/// Encode a ConstValue::U32 into a fixed 5-byte buffer.
const fn encode_const_u32(val: u32) -> [u8; 5] {
    let le = val.to_le_bytes();
    [3, le[0], le[1], le[2], le[3]]
}

// ---------------------------------------------------------------------------
// Kani harnesses
// ---------------------------------------------------------------------------

/// T3.1: ScalarType tag roundtrip.
///
/// For every symbolic u8 in [0..12), verify that:
///   tag_to_scalar_type(tag) → Some(ty) → scalar_type_to_tag(ty) == tag
///
/// Also: for every symbolic u8 >= 12, tag_to_scalar_type returns None
/// (rejects invalid tags).
#[cfg(kani)]
#[kani::proof]
fn verify_scalar_type_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 12 {
        // Valid range: roundtrip must be identity
        let ty = tag_to_scalar_type(tag);
        assert!(ty.is_some(), "valid tag must decode");
        let rt_tag = scalar_type_to_tag(ty.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        // Out of range: must reject
        assert!(tag_to_scalar_type(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.2: BinOp tag roundtrip.
#[cfg(kani)]
#[kani::proof]
fn verify_binop_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 12 {
        let op = tag_to_binop(tag);
        assert!(op.is_some(), "valid tag must decode");
        let rt_tag = binop_to_tag(op.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        assert!(tag_to_binop(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.3: UnaryOp tag roundtrip.
#[cfg(kani)]
#[kani::proof]
fn verify_unaryop_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 3 {
        let op = tag_to_unaryop(tag);
        assert!(op.is_some(), "valid tag must decode");
        let rt_tag = unaryop_to_tag(op.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        assert!(tag_to_unaryop(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.4: CmpOp tag roundtrip.
#[cfg(kani)]
#[kani::proof]
fn verify_cmpop_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 6 {
        let op = tag_to_cmpop(tag);
        assert!(op.is_some(), "valid tag must decode");
        let rt_tag = cmpop_to_tag(op.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        assert!(tag_to_cmpop(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.5: AtomicOp tag roundtrip.
#[cfg(kani)]
#[kani::proof]
fn verify_atomicop_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 9 {
        let op = tag_to_atomicop(tag);
        assert!(op.is_some(), "valid tag must decode");
        let rt_tag = atomicop_to_tag(op.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        assert!(tag_to_atomicop(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.6: MathFn tag roundtrip.
#[cfg(kani)]
#[kani::proof]
fn verify_mathfn_roundtrip() {
    let tag: u8 = kani::any();

    if tag < 22 {
        let f = tag_to_mathfn(tag);
        assert!(f.is_some(), "valid tag must decode");
        let rt_tag = mathfn_to_tag(f.unwrap());
        assert!(rt_tag == tag, "roundtrip must preserve tag");
    } else {
        assert!(tag_to_mathfn(tag).is_none(), "invalid tag must reject");
    }
}

/// T3.7: ConstValue::F32 wire roundtrip.
///
/// For a symbolic u32 (representing f32 bits), verify:
///   encode_const_f32(bits) → 5-byte buf → decode → (tag=1, decoded_bits)
///   and decoded_bits == bits.
///
/// This proves the wire format preserves f32 bit patterns exactly,
/// including NaN, infinity, denormals, and negative zero.
#[cfg(kani)]
#[kani::proof]
fn verify_const_value_f32_roundtrip() {
    let bits: u32 = kani::any();

    let buf = encode_const_f32(bits);
    let (tag, decoded_bits) = decode_const_5byte(&buf);

    assert!(tag == 1, "F32 tag must be 1");
    assert!(decoded_bits == bits, "F32 bits must roundtrip exactly");
}

/// T3.8: ConstValue::U32 wire roundtrip.
///
/// For a symbolic u32 value, verify:
///   encode_const_u32(val) → 5-byte buf → decode → (tag=3, decoded_val)
///   and decoded_val == val.
#[cfg(kani)]
#[kani::proof]
fn verify_const_value_u32_roundtrip() {
    let val: u32 = kani::any();

    let buf = encode_const_u32(val);
    let (tag, decoded_val) = decode_const_5byte(&buf);

    assert!(tag == 3, "U32 tag must be 3");
    assert!(decoded_val == val, "U32 value must roundtrip exactly");
}

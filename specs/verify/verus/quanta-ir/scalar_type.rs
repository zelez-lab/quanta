//! Verus mirror of `quanta_ir::ScalarType` wire encoding.
//!
//! Abstract: ScalarType is a finite enum of 12 byte-encoded tags.
//! `encode_tag` and `decode_tag` are inverse functions over the
//! 12 known tags.

use vstd::prelude::*;

verus! {

pub enum ScalarType {
    F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, Bool, F16,
}

pub open spec fn encode_tag(s: ScalarType) -> u8 {
    match s {
        ScalarType::F32  => 0u8,
        ScalarType::F64  => 1u8,
        ScalarType::U8   => 2u8,
        ScalarType::U16  => 3u8,
        ScalarType::U32  => 4u8,
        ScalarType::U64  => 5u8,
        ScalarType::I8   => 6u8,
        ScalarType::I16  => 7u8,
        ScalarType::I32  => 8u8,
        ScalarType::I64  => 9u8,
        ScalarType::Bool => 10u8,
        ScalarType::F16  => 11u8,
    }
}

pub open spec fn decode_tag(b: u8) -> Option<ScalarType> {
    match b {
        0u8  => Some(ScalarType::F32),
        1u8  => Some(ScalarType::F64),
        2u8  => Some(ScalarType::U8),
        3u8  => Some(ScalarType::U16),
        4u8  => Some(ScalarType::U32),
        5u8  => Some(ScalarType::U64),
        6u8  => Some(ScalarType::I8),
        7u8  => Some(ScalarType::I16),
        8u8  => Some(ScalarType::I32),
        9u8  => Some(ScalarType::I64),
        10u8 => Some(ScalarType::Bool),
        11u8 => Some(ScalarType::F16),
        _    => None,
    }
}

// ── Theorems ────────────────────────────────────────────────────────

/// Round-trip: encode then decode yields the original variant.
proof fn encode_decode_roundtrip(s: ScalarType)
    ensures decode_tag(encode_tag(s)) == Some(s),
{
    match s {
        ScalarType::F32  => {},
        ScalarType::F64  => {},
        ScalarType::U8   => {},
        ScalarType::U16  => {},
        ScalarType::U32  => {},
        ScalarType::U64  => {},
        ScalarType::I8   => {},
        ScalarType::I16  => {},
        ScalarType::I32  => {},
        ScalarType::I64  => {},
        ScalarType::Bool => {},
        ScalarType::F16  => {},
    }
}

/// Injectivity: distinct variants produce distinct tags.
proof fn encode_injective(a: ScalarType, b: ScalarType)
    requires encode_tag(a) == encode_tag(b),
    ensures a == b,
{
    match a {
        ScalarType::F32  => { match b { ScalarType::F32 => {} _ => {} } },
        ScalarType::F64  => { match b { ScalarType::F64 => {} _ => {} } },
        ScalarType::U8   => { match b { ScalarType::U8  => {} _ => {} } },
        ScalarType::U16  => { match b { ScalarType::U16 => {} _ => {} } },
        ScalarType::U32  => { match b { ScalarType::U32 => {} _ => {} } },
        ScalarType::U64  => { match b { ScalarType::U64 => {} _ => {} } },
        ScalarType::I8   => { match b { ScalarType::I8  => {} _ => {} } },
        ScalarType::I16  => { match b { ScalarType::I16 => {} _ => {} } },
        ScalarType::I32  => { match b { ScalarType::I32 => {} _ => {} } },
        ScalarType::I64  => { match b { ScalarType::I64 => {} _ => {} } },
        ScalarType::Bool => { match b { ScalarType::Bool => {} _ => {} } },
        ScalarType::F16  => { match b { ScalarType::F16 => {} _ => {} } },
    }
}

/// Totality: all tags 0..11 decode to Some.
proof fn all_tags_valid()
    ensures
        forall|b: u8| 0u8 <= b && b <= 11u8 ==> decode_tag(b).is_some(),
{}

/// Rejection: tags >= 12 decode to None.
proof fn invalid_tags_rejected(b: u8)
    requires b >= 12u8,
    ensures decode_tag(b).is_none(),
{}

} // verus!

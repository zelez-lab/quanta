//! Verus mirror of `src/api/error.rs` — QuantaError, QuantaErrorKind.
//!
//! Verified properties:
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T2300 error_kind_exhaustive | All 8 QuantaErrorKind variants are distinct.            |
//! | T2301 with_context_preserves_kind | with_context does not change the error kind.     |
//! | T2302 constructors_correct  | Each convenience constructor creates the right kind.   |
//! | T2303 eq_ignores_context    | PartialEq compares kind only, not context.              |

use vstd::prelude::*;

verus! {

pub enum QuantaErrorKind {
    NoDevice,
    OutOfMemory,
    CompilationFailed,
    SubmitFailed,
    Timeout,
    DeviceLost,
    InvalidParam,
    Internal,
}

pub struct QuantaErrorModel {
    pub kind: QuantaErrorKind,
    pub has_context: bool,
}

/// with_context: sets has_context without changing kind.
pub open spec fn with_context(err: QuantaErrorModel) -> QuantaErrorModel {
    QuantaErrorModel {
        kind: err.kind,
        has_context: true,
    }
}

// ── T2300: Error kinds are pairwise distinct ────────────────────────

pub open spec fn kind_tag(k: QuantaErrorKind) -> nat {
    match k {
        QuantaErrorKind::NoDevice          => 0,
        QuantaErrorKind::OutOfMemory       => 1,
        QuantaErrorKind::CompilationFailed => 2,
        QuantaErrorKind::SubmitFailed      => 3,
        QuantaErrorKind::Timeout           => 4,
        QuantaErrorKind::DeviceLost        => 5,
        QuantaErrorKind::InvalidParam      => 6,
        QuantaErrorKind::Internal          => 7,
    }
}

proof fn t2300_error_kind_injective(a: QuantaErrorKind, b: QuantaErrorKind)
    requires kind_tag(a) == kind_tag(b),
    ensures a == b,
{
    match a {
        QuantaErrorKind::NoDevice          => { match b { QuantaErrorKind::NoDevice => {} _ => {} } },
        QuantaErrorKind::OutOfMemory       => { match b { QuantaErrorKind::OutOfMemory => {} _ => {} } },
        QuantaErrorKind::CompilationFailed => { match b { QuantaErrorKind::CompilationFailed => {} _ => {} } },
        QuantaErrorKind::SubmitFailed      => { match b { QuantaErrorKind::SubmitFailed => {} _ => {} } },
        QuantaErrorKind::Timeout           => { match b { QuantaErrorKind::Timeout => {} _ => {} } },
        QuantaErrorKind::DeviceLost        => { match b { QuantaErrorKind::DeviceLost => {} _ => {} } },
        QuantaErrorKind::InvalidParam      => { match b { QuantaErrorKind::InvalidParam => {} _ => {} } },
        QuantaErrorKind::Internal          => { match b { QuantaErrorKind::Internal => {} _ => {} } },
    }
}

// ── T2301: with_context preserves kind ──────────────────────────────

proof fn t2301_with_context_preserves_kind(err: QuantaErrorModel)
    ensures with_context(err).kind == err.kind,
{}

// ── T2302: Constructors create correct kind ─────────────────────────

pub open spec fn make_error(kind: QuantaErrorKind) -> QuantaErrorModel {
    QuantaErrorModel { kind, has_context: false }
}

proof fn t2302_no_device()
    ensures make_error(QuantaErrorKind::NoDevice).kind == QuantaErrorKind::NoDevice,
{}
proof fn t2302_out_of_memory()
    ensures make_error(QuantaErrorKind::OutOfMemory).kind == QuantaErrorKind::OutOfMemory,
{}
proof fn t2302_submit_failed()
    ensures make_error(QuantaErrorKind::SubmitFailed).kind == QuantaErrorKind::SubmitFailed,
{}
proof fn t2302_timeout()
    ensures make_error(QuantaErrorKind::Timeout).kind == QuantaErrorKind::Timeout,
{}
proof fn t2302_device_lost()
    ensures make_error(QuantaErrorKind::DeviceLost).kind == QuantaErrorKind::DeviceLost,
{}

// ── T2303: PartialEq compares kind only ─────────────────────────────

pub open spec fn error_eq(a: QuantaErrorModel, b: QuantaErrorModel) -> bool {
    a.kind == b.kind
}

/// T2303: Two errors with the same kind are equal regardless of context.
proof fn t2303_eq_ignores_context(kind: QuantaErrorKind)
    ensures ({
        let a = QuantaErrorModel { kind, has_context: false };
        let b = QuantaErrorModel { kind, has_context: true };
        error_eq(a, b)
    }),
{}

} // verus!

//! Verus mirror of `quanta-macros::validate` — kernel validation rules.
//!
//! Mirrors:
//!   crates/quanta-macros/src/validate.rs  — validate_kernel, validate_param_type
//!   crates/quanta-macros/src/kernel_macro.rs — parse_opt_str, KernelAttrs defaults
//!
//! Proves:
//!   T820: Validation rules are sound — valid kernels satisfy all constraints
//!   T821: Optimization level parsing covers all documented values
//!   T822: Workgroup size defaults are spec-compliant

use vstd::prelude::*;

verus! {

// ── Validation predicates ──────────────────────────────────────────

/// Abstract model of a function signature for validation.
pub struct FnSig {
    pub has_return_type: bool,
    pub is_async: bool,
    pub is_unsafe: bool,
    pub has_self: bool,
    pub has_type_generics: bool,
    pub has_lifetime_generics: bool,
    pub has_const_generics: bool,
}

/// A function is GPU-valid iff it satisfies all kernel constraints.
pub open spec fn is_valid_kernel(sig: FnSig) -> bool {
    !sig.has_return_type
    && !sig.is_async
    && !sig.is_unsafe
    && !sig.has_self
    && !sig.has_type_generics
    && !sig.has_lifetime_generics
    // const generics are allowed
}

/// T820a: A function with return type is rejected.
proof fn t820a_return_type_rejected()
    ensures !is_valid_kernel(FnSig {
        has_return_type: true,
        is_async: false,
        is_unsafe: false,
        has_self: false,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

/// T820b: An async function is rejected.
proof fn t820b_async_rejected()
    ensures !is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: true,
        is_unsafe: false,
        has_self: false,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

/// T820c: An unsafe function is rejected.
proof fn t820c_unsafe_rejected()
    ensures !is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: false,
        is_unsafe: true,
        has_self: false,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

/// T820d: A self-receiver function is rejected.
proof fn t820d_self_rejected()
    ensures !is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: false,
        is_unsafe: false,
        has_self: true,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

/// T820e: Type generics are rejected.
proof fn t820e_type_generics_rejected()
    ensures !is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: false,
        is_unsafe: false,
        has_self: false,
        has_type_generics: true,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

/// T820f: A clean kernel with const generics is accepted.
proof fn t820f_const_generic_accepted()
    ensures is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: false,
        is_unsafe: false,
        has_self: false,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: true,
    }),
{}

/// T820g: A minimal valid kernel is accepted.
proof fn t820g_minimal_valid()
    ensures is_valid_kernel(FnSig {
        has_return_type: false,
        is_async: false,
        is_unsafe: false,
        has_self: false,
        has_type_generics: false,
        has_lifetime_generics: false,
        has_const_generics: false,
    }),
{}

// ── T821: Optimization level parsing ───────────────────────────────
// Mirrors parse_opt_str() in kernel_macro.rs lines 235-242.

pub enum OptStr { O0, O1, O2, O3, Num0, Num1, Num2, Num3, Other }

pub open spec fn parse_opt_str(s: OptStr) -> u8 {
    match s {
        OptStr::O0   => 0u8,
        OptStr::O1   => 1u8,
        OptStr::O2   => 2u8,
        OptStr::O3   => 3u8,
        OptStr::Num0 => 0u8,
        OptStr::Num1 => 1u8,
        OptStr::Num2 => 2u8,
        OptStr::Num3 => 3u8,
        OptStr::Other => 3u8, // default to O3
    }
}

/// T821a: All valid opt strings produce values in [0, 3].
proof fn t821a_opt_range(s: OptStr)
    ensures parse_opt_str(s) <= 3u8,
{
    match s {
        OptStr::O0 => {}, OptStr::O1 => {}, OptStr::O2 => {}, OptStr::O3 => {},
        OptStr::Num0 => {}, OptStr::Num1 => {}, OptStr::Num2 => {}, OptStr::Num3 => {},
        OptStr::Other => {},
    }
}

/// T821b: Unrecognized strings default to O3.
proof fn t821b_unknown_defaults_to_o3()
    ensures parse_opt_str(OptStr::Other) == 3u8,
{}

/// T821c: "O0" and "0" produce the same level.
proof fn t821c_o0_equivalence()
    ensures parse_opt_str(OptStr::O0) == parse_opt_str(OptStr::Num0),
{}

// ── T822: Workgroup size defaults ──────────────────────────────────
// Mirrors KernelAttrs::default() in kernel_macro.rs.

pub open spec fn default_workgroup_size() -> (u32, u32, u32) {
    (64u32, 1u32, 1u32)
}

pub open spec fn default_opt_level() -> u8 {
    3u8
}

/// T822a: Default workgroup size is [64, 1, 1].
proof fn t822a_default_workgroup()
    ensures ({
        let (x, y, z) = default_workgroup_size();
        x == 64u32 && y == 1u32 && z == 1u32
    }),
{}

/// T822b: Default workgroup size is non-zero in all dimensions.
proof fn t822b_workgroup_nonzero()
    ensures ({
        let (x, y, z) = default_workgroup_size();
        x > 0u32 && y > 0u32 && z > 0u32
    }),
{}

/// T822c: Total threads = x * y * z = 64 (fits in a single warp).
proof fn t822c_total_threads()
    ensures ({
        let (x, y, z) = default_workgroup_size();
        (x as nat) * (y as nat) * (z as nat) == 64
    }),
{}

/// T822d: Default optimization level is O3.
proof fn t822d_default_opt()
    ensures default_opt_level() == 3u8,
{}

fn main() {}

} // verus!

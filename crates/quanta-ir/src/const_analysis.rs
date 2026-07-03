//! Constant-value analysis over `KernelOp` trees.
//!
//! Used by emitters to recognize `KernelOp::Const`-defined registers
//! and apply target-specific micro-optimizations (e.g. loop unrolling
//! when `Loop.count` is a small known integer).
//!
//! Today: `collect_int_consts` builds a flat `Reg → i64` map over
//! every integer Const op in the body, recursing into `Branch` and
//! `Loop` sub-bodies. The lowering's register-allocation invariant
//! guarantees each `Reg` is defined by at most one op across the
//! whole tree, so the map is well-defined.
//!
//! Float and boolean Const variants are deliberately skipped — they
//! don't feed the optimization sites this module supports.

use crate::types::{ConstValue, KernelOp};
use std::collections::HashMap;

/// Walk an op list (and all `Branch`/`Loop` sub-bodies recursively)
/// and collect every register defined by a U32/U64/I32/I64
/// `KernelOp::Const`. The value is sign-extended to `i64`.
///
/// U64/I64 inputs are truncated to 32 bits during emission by all
/// current SPIR-V/MSL/WGSL emitters, so we record the *truncated*
/// value here to match what the emitters actually see at use sites.
///
/// Used by:
/// - SPIR-V emitter: `LOOP_CONTROL_UNROLL` for `Loop.count ∈ 1..=8`
///   (see `crates/quanta-ir/src/emit_spirv/ops.rs`).
/// - MSL emitter: `#pragma clang loop unroll(full)` for the same.
pub fn collect_int_consts(body: &[KernelOp]) -> HashMap<u32, i64> {
    let mut out = HashMap::new();
    collect_into(body, &mut out);
    out
}

fn collect_into(ops: &[KernelOp], out: &mut HashMap<u32, i64>) {
    for op in ops {
        match op {
            KernelOp::Const { dst, value } => match value {
                ConstValue::U32(v) => {
                    out.insert(dst.0, *v as i64);
                }
                ConstValue::U64(v) => {
                    // Match the SPIR-V emitter's truncate-to-u32 path.
                    out.insert(dst.0, (*v as u32) as i64);
                }
                ConstValue::I32(v) => {
                    out.insert(dst.0, *v as i64);
                }
                ConstValue::I64(v) => {
                    // Match the SPIR-V emitter's truncate-to-i32 path.
                    out.insert(dst.0, (*v as i32) as i64);
                }
                // Floats and Bools don't feed the unroll site.
                _ => {}
            },
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => {
                collect_into(then_ops, out);
                collect_into(else_ops, out);
            }
            KernelOp::Loop { body, .. } => {
                collect_into(body, out);
            }
            _ => {}
        }
    }
}

/// Convenience: should a `Loop` whose `count` register has the given
/// (optional) integer value receive an unroll hint? Boundary: 1..=8.
/// Returns `false` for `None` (unknown), 0 (degenerate), and >8.
///
/// Centralised so SPIR-V / MSL / future WGSL emitters share the
/// same threshold without each restating the comparison.
pub fn should_unroll_loop_count(value: Option<i64>) -> bool {
    matches!(value, Some(v) if (1..=8).contains(&v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BinOp, KernelOp, Reg, ScalarType};

    #[test]
    fn collects_top_level_u32_const() {
        let body = vec![KernelOp::Const {
            dst: Reg(5),
            value: ConstValue::U32(42),
        }];
        let m = collect_int_consts(&body);
        assert_eq!(m.get(&5).copied(), Some(42));
    }

    #[test]
    fn collects_inside_branch_then_and_else() {
        let body = vec![KernelOp::Branch {
            cond: Reg(0),
            then_ops: vec![KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(7),
            }],
            else_ops: vec![KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::I32(-3),
            }],
        }];
        let m = collect_int_consts(&body);
        assert_eq!(m.get(&1).copied(), Some(7));
        assert_eq!(m.get(&2).copied(), Some(-3));
    }

    #[test]
    fn collects_inside_loop_body() {
        let body = vec![KernelOp::Loop {
            count: Reg(10),
            iter_reg: Reg(11),
            body: vec![KernelOp::Const {
                dst: Reg(12),
                value: ConstValue::U32(99),
            }],
        }];
        let m = collect_int_consts(&body);
        assert_eq!(m.get(&12).copied(), Some(99));
    }

    #[test]
    fn skips_non_int_consts() {
        let body = vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::F32(3.25),
            },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::Bool(true),
            },
        ];
        let m = collect_int_consts(&body);
        assert!(m.is_empty());
    }

    #[test]
    fn skips_non_const_ops() {
        let body = vec![KernelOp::BinOp {
            dst: Reg(0),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::Add,
            ty: ScalarType::U32,
        }];
        let m = collect_int_consts(&body);
        assert!(m.is_empty());
    }

    #[test]
    fn should_unroll_boundary_cases() {
        assert!(!should_unroll_loop_count(None));
        assert!(!should_unroll_loop_count(Some(0)));
        assert!(should_unroll_loop_count(Some(1)));
        assert!(should_unroll_loop_count(Some(8)));
        assert!(!should_unroll_loop_count(Some(9)));
        assert!(!should_unroll_loop_count(Some(-1)));
    }
}

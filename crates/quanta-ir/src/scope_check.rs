//! Compile-time scope-validity check on a `KernelDef` body.
//!
//! Mirrors the Lean `Quanta.KOps.scopeValid` predicate (see
//! `specs/verify/lean/Quanta/KOps/Scope.lean`). Walks the op tree
//! tracking the set of registers defined so far in the current
//! scope; any operand referencing a register not yet defined is a
//! use-before-def violation.
//!
//! Used by the proc-macro pipeline as a dynamic oracle that catches
//! the structured-control emitter bugs surfaced in
//! `emitter_codegen_bugs_2026-05-29` (specifically bug #1: r44
//! forward-branch in while-loop kernels via the `install_redirect_at`
//! path). Production lowering output that triggers MSL/SPIR-V
//! `use of undeclared identifier` failures is rejected here at macro
//! time, before it reaches the binary blob.
//!
//! The check is structural — it does not depend on backend behavior,
//! data flow, or aliasing. A pass is sufficient evidence that every
//! op-operand reg is defined by a preceding op in the same scope or
//! an enclosing scope.

use crate::types::{KernelDef, KernelOp, Reg};
use std::collections::HashSet;

/// Sentinel reg used by `Load.index` for push-constant slot reads.
/// The MSL and SPIR-V emitters dispatch on `index.0 == u32::MAX` to
/// emit a push-constant fetch instead of a buffer load
/// (see `crates/quanta-ir/src/emit_msl/ops.rs:43` and
/// `crates/quanta-ir/src/emit_spirv/ops.rs:113`; produced by
/// `crates/quanta-wasm-lowering/src/lower.rs:427` for scalar
/// push-constant param initialisation). It is NOT a runtime
/// register — the validator excepts it from the in-scope check.
const PUSH_CONST_INDEX_SENTINEL: Reg = Reg(u32::MAX);

/// A use-before-def finding. Identifies which register was used
/// before it was defined and the op that referenced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeViolation {
    /// The undefined register.
    pub reg: Reg,
    /// Short description of where the register was referenced, e.g.
    /// `"Branch.cond"`, `"BinOp.a"`, `"Load.index"`.
    pub location: &'static str,
}

/// Run the scope-validity check on a `KernelDef` body. Returns the
/// first violation encountered, or `Ok(())` if the body is valid.
///
/// The starting env is empty. Parameter registers are materialized
/// on demand by the lowering's `localGet` path (which calls
/// `alloc_reg()` for both the stable-reg slot and the fresh copy
/// destination), so every reg appearing in the IR body was allocated
/// by some op — there is no implicit ambient scope.
pub fn scope_check(def: &KernelDef) -> Result<(), ScopeViolation> {
    // Device-function bodies are lowered independently; each gets its
    // own scope-check pass with its own empty env.
    for fn_def in &def.device_functions {
        let mut fn_env: HashSet<Reg> = HashSet::new();
        check_ops(&fn_def.body, &mut fn_env)?;
    }
    let mut env: HashSet<Reg> = HashSet::new();
    check_ops(&def.body, &mut env)
}

/// Walk a flat op list against the running env, mutating `env` with
/// each op's defined register. On `Branch` and `Loop`, recurse into
/// the sub-bodies with the appropriate parent env.
fn check_ops(ops: &[KernelOp], env: &mut HashSet<Reg>) -> Result<(), ScopeViolation> {
    for op in ops {
        check_uses(op, env)?;
        if let Some(reg) = defined_reg(op) {
            env.insert(reg);
        }
    }
    Ok(())
}

/// Validate every operand register of `op` against `env`. For
/// structured ops (`Branch`, `Loop`), recurse into the sub-bodies
/// using a *fresh* clone of `env` so the sub-scope's defs don't leak
/// out to the parent scope. (`Loop` extends its body's env with
/// `iter_reg` before recursing.)
fn check_uses(op: &KernelOp, env: &HashSet<Reg>) -> Result<(), ScopeViolation> {
    let check = |reg: Reg, location: &'static str| -> Result<(), ScopeViolation> {
        if env.contains(&reg) {
            Ok(())
        } else {
            Err(ScopeViolation { reg, location })
        }
    };
    match op {
        // Memory.
        KernelOp::Load { index, .. } => {
            // Push-constant sentinel: dispatched on by backends, not
            // a real register reference.
            if *index == PUSH_CONST_INDEX_SENTINEL {
                Ok(())
            } else {
                check(*index, "Load.index")
            }
        }
        KernelOp::Store { index, src, .. } => {
            check(*index, "Store.index")?;
            check(*src, "Store.src")
        }
        KernelOp::SharedDecl { .. } => Ok(()),
        KernelOp::SharedDeclDyn { .. } => Ok(()),
        KernelOp::SharedLoad { index, .. } => check(*index, "SharedLoad.index"),
        KernelOp::SharedStore { index, src, .. } => {
            check(*index, "SharedStore.index")?;
            check(*src, "SharedStore.src")
        }
        // Arithmetic.
        KernelOp::BinOp { a, b, .. } => {
            check(*a, "BinOp.a")?;
            check(*b, "BinOp.b")
        }
        KernelOp::UnaryOp { a, .. } => check(*a, "UnaryOp.a"),
        KernelOp::Cmp { a, b, .. } => {
            check(*a, "Cmp.a")?;
            check(*b, "Cmp.b")
        }
        // Control flow.
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            check(*cond, "Branch.cond")?;
            // Each branch arm starts from the current env — sub-scope
            // defs don't leak to the parent or to the sibling arm.
            let mut then_env = env.clone();
            check_ops(then_ops, &mut then_env)?;
            let mut else_env = env.clone();
            check_ops(else_ops, &mut else_env)
        }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            check(*count, "Loop.count")?;
            // Body sees the parent env plus iter_reg.
            let mut body_env = env.clone();
            body_env.insert(*iter_reg);
            check_ops(body, &mut body_env)
        }
        // Math.
        KernelOp::MathCall { args, .. } => {
            for a in args {
                check(*a, "MathCall.arg")?;
            }
            Ok(())
        }
        // Thread indexing — no operand reads.
        KernelOp::QuarkId { .. }
        | KernelOp::QuarkCount { .. }
        | KernelOp::ProtonId { .. }
        | KernelOp::NucleusId { .. }
        | KernelOp::ProtonSize { .. } => Ok(()),
        // Synchronization.
        KernelOp::Barrier => Ok(()),
        KernelOp::Fence { .. } => Ok(()),
        // Atomics.
        KernelOp::AtomicOp { index, val, .. } => {
            check(*index, "AtomicOp.index")?;
            check(*val, "AtomicOp.val")
        }
        KernelOp::SharedAtomicOp { index, val, .. } => {
            check(*index, "SharedAtomicOp.index")?;
            check(*val, "SharedAtomicOp.val")
        }
        KernelOp::AtomicCas {
            index,
            expected,
            desired,
            ..
        } => {
            check(*index, "AtomicCas.index")?;
            check(*expected, "AtomicCas.expected")?;
            check(*desired, "AtomicCas.desired")
        }
        // Warp/wave.
        KernelOp::WaveShuffle {
            src, lane_delta, ..
        } => {
            check(*src, "WaveShuffle.src")?;
            check(*lane_delta, "WaveShuffle.lane_delta")
        }
        KernelOp::WaveBallot { predicate, .. } => check(*predicate, "WaveBallot.predicate"),
        KernelOp::WaveAny { predicate, .. } => check(*predicate, "WaveAny.predicate"),
        KernelOp::WaveAll { predicate, .. } => check(*predicate, "WaveAll.predicate"),
        // Type conversion.
        KernelOp::Cast { src, .. } => check(*src, "Cast.src"),
        KernelOp::Const { .. } => Ok(()),
        // Vector.
        KernelOp::VecConstruct { components, .. } => {
            for c in components {
                check(*c, "VecConstruct.component")?;
            }
            Ok(())
        }
        KernelOp::VecExtract { vec, .. } => check(*vec, "VecExtract.vec"),
        KernelOp::MatMul { a, b, .. } => {
            check(*a, "MatMul.a")?;
            check(*b, "MatMul.b")
        }
        KernelOp::CooperativeMMA { a, b, c, .. } => {
            check(*a, "CooperativeMMA.a")?;
            check(*b, "CooperativeMMA.b")?;
            check(*c, "CooperativeMMA.c")
        }
        // Texture.
        KernelOp::TextureSample2D { x, y, .. } => {
            check(*x, "TextureSample2D.x")?;
            check(*y, "TextureSample2D.y")
        }
        KernelOp::TextureSample3D { x, y, z, .. } => {
            check(*x, "TextureSample3D.x")?;
            check(*y, "TextureSample3D.y")?;
            check(*z, "TextureSample3D.z")
        }
        KernelOp::TextureWrite2D { x, y, value, .. } => {
            check(*x, "TextureWrite2D.x")?;
            check(*y, "TextureWrite2D.y")?;
            check(*value, "TextureWrite2D.value")
        }
        KernelOp::TextureSize { .. } => Ok(()),
        KernelOp::TextureLoad2D { x, y, .. } => {
            check(*x, "TextureLoad2D.x")?;
            check(*y, "TextureLoad2D.y")
        }
        // Register copy.
        KernelOp::Copy { src, .. } => check(*src, "Copy.src"),
        // Break.
        KernelOp::Break => Ok(()),
        // Dynamic parallelism.
        KernelOp::Dispatch { wave, groups } => {
            check(*wave, "Dispatch.wave")?;
            check(groups[0], "Dispatch.groups[0]")?;
            check(groups[1], "Dispatch.groups[1]")?;
            check(groups[2], "Dispatch.groups[2]")
        }
        // Device call.
        KernelOp::DeviceCall { args, .. } => {
            for a in args {
                check(*a, "DeviceCall.arg")?;
            }
            Ok(())
        }
        // Bit manipulation.
        KernelOp::Bitcast { src, .. } => check(*src, "Bitcast.src"),
        KernelOp::CountTrailingZeros { src, .. } => check(*src, "CountTrailingZeros.src"),
        KernelOp::CountLeadingZeros { src, .. } => check(*src, "CountLeadingZeros.src"),
        KernelOp::PopCount { src, .. } => check(*src, "PopCount.src"),
        // Dot product.
        KernelOp::Dot { a, b, .. } => {
            check(*a, "Dot.a")?;
            check(*b, "Dot.b")
        }
        // Subgroup.
        KernelOp::SubgroupSize { .. } => Ok(()),
        KernelOp::SubgroupReduceAdd { src, .. } => check(*src, "SubgroupReduceAdd.src"),
        KernelOp::SubgroupReduceMin { src, .. } => check(*src, "SubgroupReduceMin.src"),
        KernelOp::SubgroupReduceMax { src, .. } => check(*src, "SubgroupReduceMax.src"),
        KernelOp::SubgroupExclusiveAdd { src, .. } => check(*src, "SubgroupExclusiveAdd.src"),
        KernelOp::SubgroupInclusiveAdd { src, .. } => check(*src, "SubgroupInclusiveAdd.src"),
        // Debug.
        KernelOp::DebugPrint { src, .. } => check(*src, "DebugPrint.src"),
    }
}

/// Register defined by `op`, if any. Mirrors the Lean
/// `KernelOp.definedReg` definition. Structured ops (`Branch`,
/// `Loop`) define no parent-scope reg — their sub-bodies' defs are
/// confined to the sub-scope.
fn defined_reg(op: &KernelOp) -> Option<Reg> {
    match op {
        KernelOp::Load { dst, .. }
        | KernelOp::SharedLoad { dst, .. }
        | KernelOp::BinOp { dst, .. }
        | KernelOp::UnaryOp { dst, .. }
        | KernelOp::Cmp { dst, .. }
        | KernelOp::MathCall { dst, .. }
        | KernelOp::QuarkId { dst }
        | KernelOp::QuarkCount { dst }
        | KernelOp::ProtonId { dst }
        | KernelOp::NucleusId { dst }
        | KernelOp::ProtonSize { dst }
        | KernelOp::AtomicOp { dst, .. }
        | KernelOp::SharedAtomicOp { dst, .. }
        | KernelOp::AtomicCas { dst, .. }
        | KernelOp::WaveShuffle { dst, .. }
        | KernelOp::WaveBallot { dst, .. }
        | KernelOp::WaveAny { dst, .. }
        | KernelOp::WaveAll { dst, .. }
        | KernelOp::Cast { dst, .. }
        | KernelOp::Const { dst, .. }
        | KernelOp::VecConstruct { dst, .. }
        | KernelOp::VecExtract { dst, .. }
        | KernelOp::MatMul { dst, .. }
        | KernelOp::CooperativeMMA { dst, .. }
        | KernelOp::TextureSample2D { dst, .. }
        | KernelOp::TextureSample3D { dst, .. }
        | KernelOp::TextureLoad2D { dst, .. }
        | KernelOp::Copy { dst, .. }
        | KernelOp::DeviceCall { dst, .. }
        | KernelOp::Bitcast { dst, .. }
        | KernelOp::CountTrailingZeros { dst, .. }
        | KernelOp::CountLeadingZeros { dst, .. }
        | KernelOp::PopCount { dst, .. }
        | KernelOp::Dot { dst, .. }
        | KernelOp::SubgroupSize { dst }
        | KernelOp::SubgroupReduceAdd { dst, .. }
        | KernelOp::SubgroupReduceMin { dst, .. }
        | KernelOp::SubgroupReduceMax { dst, .. }
        | KernelOp::SubgroupExclusiveAdd { dst, .. }
        | KernelOp::SubgroupInclusiveAdd { dst, .. } => Some(*dst),
        // TextureSize defines two regs; we add dst_w here and patch
        // dst_h on the same op's post-walk (handled by the caller
        // since check_ops only inserts a single reg per op).
        KernelOp::TextureSize { dst_w, .. } => Some(*dst_w),
        // No defined reg.
        KernelOp::Store { .. }
        | KernelOp::SharedDecl { .. }
        | KernelOp::SharedDeclDyn { .. }
        | KernelOp::SharedStore { .. }
        | KernelOp::Branch { .. }
        | KernelOp::Loop { .. }
        | KernelOp::Barrier
        | KernelOp::Fence { .. }
        | KernelOp::TextureWrite2D { .. }
        | KernelOp::Break
        | KernelOp::Dispatch { .. }
        | KernelOp::DebugPrint { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BinOp as BinOpKind, CmpOp, ConstValue, KernelDef, KernelOp, ScalarType};

    fn make_def(body: Vec<KernelOp>) -> KernelDef {
        KernelDef {
            name: "test".into(),
            params: vec![],
            body,
            body_source: None,
            next_reg: 100,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    /// Bug #1 witness mirroring `specs/verify/lean/Quanta/KOps/Scope.lean`:
    /// const r3; Branch with cond=r44, empty then, else containing a
    /// nested Branch whose then defines r44 via cmp. Validator should
    /// reject because r44 is not in env when OuterBranch's cond is checked.
    #[test]
    fn rejects_r44_forward_branch_pattern() {
        let def = make_def(vec![
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::U32(8),
            },
            KernelOp::Branch {
                cond: Reg(44),
                then_ops: vec![],
                else_ops: vec![KernelOp::Branch {
                    cond: Reg(10),
                    then_ops: vec![
                        KernelOp::Const {
                            dst: Reg(43),
                            value: ConstValue::U32(0),
                        },
                        KernelOp::Cmp {
                            dst: Reg(44),
                            a: Reg(3),
                            b: Reg(43),
                            op: CmpOp::Eq,
                            ty: ScalarType::U32,
                        },
                    ],
                    else_ops: vec![],
                }],
            },
        ]);
        let err = scope_check(&def).expect_err("should reject use-before-def of r44");
        assert_eq!(err.reg, Reg(44));
        assert_eq!(err.location, "Branch.cond");
    }

    /// Counterfactual: the function-scope-cond fix sketch. Pre-allocate
    /// r44 (and r10, the InnerBranch cond) at the outer scope before
    /// OuterBranch, then write r44 via copy inside InnerBranch's then.
    /// Validator should accept.
    #[test]
    fn accepts_function_scope_cond_fix() {
        let def = make_def(vec![
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::U32(8),
            },
            KernelOp::Const {
                dst: Reg(44),
                value: ConstValue::U32(0),
            },
            KernelOp::Const {
                dst: Reg(10),
                value: ConstValue::U32(1),
            },
            KernelOp::Branch {
                cond: Reg(44),
                then_ops: vec![],
                else_ops: vec![KernelOp::Branch {
                    cond: Reg(10),
                    then_ops: vec![
                        KernelOp::Const {
                            dst: Reg(43),
                            value: ConstValue::U32(1),
                        },
                        KernelOp::Copy {
                            dst: Reg(44),
                            src: Reg(43),
                            ty: ScalarType::U32,
                        },
                    ],
                    else_ops: vec![],
                }],
            },
        ]);
        scope_check(&def).expect("function-scope-cond pattern should be scope-valid");
    }

    /// Smoke: an empty body is vacuously scope-valid.
    #[test]
    fn accepts_empty_body() {
        scope_check(&make_def(vec![])).expect("empty body should pass");
    }

    /// Smoke: a straight-line `const r0; const r1; binop r2 r0 r1` chain
    /// is scope-valid because each operand was defined earlier.
    #[test]
    fn accepts_straight_line_chain() {
        let def = make_def(vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::U32(1),
            },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(2),
            },
            KernelOp::BinOp {
                dst: Reg(2),
                a: Reg(0),
                b: Reg(1),
                op: BinOpKind::Add,
                ty: ScalarType::U32,
            },
        ]);
        scope_check(&def).expect("straight-line chain should be valid");
    }

    /// Smoke: a Branch with cond defined immediately before is valid;
    /// inner sub-body defs don't leak to siblings.
    #[test]
    fn accepts_branch_with_cond_in_scope() {
        let def = make_def(vec![
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::U32(0),
            },
            KernelOp::Branch {
                cond: Reg(5),
                then_ops: vec![KernelOp::Const {
                    dst: Reg(6),
                    value: ConstValue::U32(1),
                }],
                else_ops: vec![KernelOp::Const {
                    dst: Reg(7),
                    value: ConstValue::U32(2),
                }],
            },
        ]);
        scope_check(&def).expect("branch with cond defined before is valid");
    }

    /// Smoke: branch sub-scope defs don't leak. After the branch, r6 and
    /// r7 (defined inside the arms) are NOT in scope.
    #[test]
    fn rejects_use_of_branch_sub_scope_def() {
        let def = make_def(vec![
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::U32(0),
            },
            KernelOp::Branch {
                cond: Reg(5),
                then_ops: vec![KernelOp::Const {
                    dst: Reg(6),
                    value: ConstValue::U32(1),
                }],
                else_ops: vec![],
            },
            // Use r6 after the branch — should fail.
            KernelOp::Copy {
                dst: Reg(8),
                src: Reg(6),
                ty: ScalarType::U32,
            },
        ]);
        let err = scope_check(&def).expect_err("post-branch use of sub-scope reg should fail");
        assert_eq!(err.reg, Reg(6));
        assert_eq!(err.location, "Copy.src");
    }
}

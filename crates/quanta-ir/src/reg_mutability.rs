//! Mutable-register detection for SSA-based emitters.
//!
//! The KernelOp contract is *mutable-register* semantics: a register may be
//! written inside a `Branch` arm or `Loop` body and read after the merge,
//! backed by a dominating entry `Const` init (the wasm-route lowering emits
//! exactly this shape for locals). Emitters that model registers as pure SSA
//! renames (SPIR-V) cannot express that directly: the id assigned in one arm
//! does not dominate the merge, and whichever arm was emitted last would
//! silently "win".
//!
//! [`collect_mutable_regs`] is the pre-pass those emitters run over a kernel
//! (or device-function) body to find the registers that must be demoted to a
//! memory slot (`Function`-storage `OpVariable` + `OpLoad`/`OpStore`,
//! mirroring the LLVM backend's `reg_slots` allocas). A register is demoted
//! when SSA renaming is *not* sufficient, i.e. when either:
//!
//! 1. it is written more than once (the entry-`Const` + re-`Copy` pattern), or
//! 2. its single write happens inside a nested scope (Branch arm / Loop body)
//!    and some read escapes that scope — the write would not dominate the
//!    read.
//!
//! Single-def registers whose reads all sit inside the defining scope (the
//! vast majority of temporaries) stay pure SSA renames.

use std::collections::{BTreeMap, HashMap};

use crate::types::{ConstValue, KernelOp, Reg, ScalarType, UnaryOp};

/// Detect registers that need mutable (memory-slot) semantics.
///
/// `pre_written` seeds registers defined before the body runs (e.g. device
/// function parameters, `(reg_number, scalar_type)`), treated as a write at
/// the outermost scope.
///
/// Returns `reg_number → ScalarType` where the scalar type is the type of the
/// register's *first* write — the element type the memory slot should be
/// declared with (later writes of a different-typed value are coerced at the
/// store). Registers whose first write has no scalar element type (vector
/// constructs) are never demoted.
pub fn collect_mutable_regs(
    pre_written: &[(u32, ScalarType)],
    ops: &[KernelOp],
) -> BTreeMap<u32, ScalarType> {
    let mut scan = MutScan::default();
    for &(reg, ty) in pre_written {
        scan.write(Reg(reg), Some(ty));
    }
    scan.walk(ops);
    scan.demoted
        .into_iter()
        .filter_map(|reg| {
            let (_, ty) = scan.first_write.get(&reg)?;
            (*ty).map(|t| (reg, t))
        })
        .collect()
}

#[derive(Default)]
struct MutScan {
    /// Unique id source for scope path entries.
    next_scope: u32,
    /// Current scope path: one entry per enclosing Branch arm / Loop body.
    path: Vec<u32>,
    /// reg → (scope path of first write, scalar type of first write).
    first_write: HashMap<u32, (Vec<u32>, Option<ScalarType>)>,
    /// Registers requiring memory-slot semantics.
    demoted: std::collections::BTreeSet<u32>,
}

impl MutScan {
    fn write(&mut self, reg: Reg, ty: Option<ScalarType>) {
        match self.first_write.entry(reg.0) {
            std::collections::hash_map::Entry::Occupied(_) => {
                // Second write: SSA renaming can't represent it across
                // scopes; demote.
                self.demoted.insert(reg.0);
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((self.path.clone(), ty));
            }
        }
    }

    fn read(&mut self, reg: Reg) {
        // Reg(u32::MAX) is the push-constant sentinel index, not a register.
        if reg.0 == u32::MAX {
            return;
        }
        if let Some((write_path, _)) = self.first_write.get(&reg.0) {
            // A read is dominated by the (single) write only if it happens
            // at the write's scope or deeper inside it. Anything else (after
            // the enclosing Branch/Loop merged, or in a sibling arm) escapes.
            if !self.path.starts_with(write_path) {
                self.demoted.insert(reg.0);
            }
        }
        // Read before any write: invalid IR; the emitter reports it.
    }

    fn enter_scope(&mut self) {
        self.next_scope += 1;
        self.path.push(self.next_scope);
    }

    fn leave_scope(&mut self) {
        self.path.pop();
    }

    fn walk(&mut self, ops: &[KernelOp]) {
        for op in ops {
            // Reads are processed before the op's write so `dst == src`
            // shapes are handled in program order.
            match op {
                KernelOp::Load { dst, index, ty, .. } => {
                    self.read(*index);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::Store { index, src, .. } => {
                    self.read(*index);
                    self.read(*src);
                }
                KernelOp::SharedDecl { .. } | KernelOp::SharedDeclDyn { .. } => {}
                KernelOp::SharedLoad { dst, index, ty, .. } => {
                    self.read(*index);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::SharedStore { index, src, .. } => {
                    self.read(*index);
                    self.read(*src);
                }
                KernelOp::BinOp { dst, a, b, ty, .. } => {
                    self.read(*a);
                    self.read(*b);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::UnaryOp { dst, a, op, ty } => {
                    self.read(*a);
                    let out_ty = if matches!(op, UnaryOp::LogicalNot) {
                        ScalarType::Bool
                    } else {
                        *ty
                    };
                    self.write(*dst, Some(out_ty));
                }
                KernelOp::Cmp { dst, a, b, .. } => {
                    self.read(*a);
                    self.read(*b);
                    self.write(*dst, Some(ScalarType::Bool));
                }
                KernelOp::Branch {
                    cond,
                    then_ops,
                    else_ops,
                } => {
                    self.read(*cond);
                    self.enter_scope();
                    self.walk(then_ops);
                    self.leave_scope();
                    self.enter_scope();
                    self.walk(else_ops);
                    self.leave_scope();
                }
                KernelOp::Loop {
                    count,
                    iter_reg,
                    body,
                } => {
                    self.read(*count);
                    // The loop counter is defined by the header phi, which
                    // dominates both the body and the merge — model it as a
                    // write at the Loop's own scope.
                    self.write(*iter_reg, Some(ScalarType::U32));
                    self.enter_scope();
                    self.walk(body);
                    self.leave_scope();
                }
                KernelOp::MathCall { dst, args, ty, .. } => {
                    for a in args {
                        self.read(*a);
                    }
                    self.write(*dst, Some(*ty));
                }
                KernelOp::QuarkId { dst }
                | KernelOp::QuarkCount { dst }
                | KernelOp::ProtonId { dst }
                | KernelOp::NucleusId { dst }
                | KernelOp::ProtonSize { dst }
                | KernelOp::SubgroupSize { dst } => {
                    self.write(*dst, Some(ScalarType::U32));
                }
                KernelOp::Barrier | KernelOp::Fence { .. } | KernelOp::Break => {}
                KernelOp::AtomicOp {
                    dst,
                    index,
                    val,
                    ty,
                    ..
                }
                | KernelOp::SharedAtomicOp {
                    dst,
                    index,
                    val,
                    ty,
                    ..
                } => {
                    self.read(*index);
                    self.read(*val);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::AtomicCas {
                    dst,
                    index,
                    expected,
                    desired,
                    ty,
                    ..
                } => {
                    self.read(*index);
                    self.read(*expected);
                    self.read(*desired);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::WaveShuffle {
                    dst,
                    src,
                    lane_delta,
                    ty,
                } => {
                    self.read(*src);
                    self.read(*lane_delta);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::WaveBallot { dst, predicate }
                | KernelOp::WaveAny { dst, predicate }
                | KernelOp::WaveAll { dst, predicate } => {
                    self.read(*predicate);
                    self.write(*dst, Some(ScalarType::U32));
                }
                KernelOp::Cast { dst, src, to, .. } | KernelOp::Bitcast { dst, src, to, .. } => {
                    self.read(*src);
                    self.write(*dst, Some(*to));
                }
                KernelOp::Const { dst, value } => {
                    // Match the SPIR-V emitters' constant emission: F16/BF16/
                    // fp8 constants materialize as f32 body values.
                    let ty = match value {
                        ConstValue::F16(_)
                        | ConstValue::BF16(_)
                        | ConstValue::FP8E5M2(_)
                        | ConstValue::FP8E4M3(_)
                        | ConstValue::F32(_) => ScalarType::F32,
                        ConstValue::F64(_) => ScalarType::F64,
                        ConstValue::U32(_) => ScalarType::U32,
                        ConstValue::U64(_) => ScalarType::U64,
                        ConstValue::I32(_) => ScalarType::I32,
                        ConstValue::I64(_) => ScalarType::I64,
                        ConstValue::Bool(_) => ScalarType::Bool,
                    };
                    self.write(*dst, Some(ty));
                }
                KernelOp::Quantize {
                    dst,
                    src,
                    scale,
                    zero_point,
                    ..
                } => {
                    self.read(*src);
                    self.read(*scale);
                    self.read(*zero_point);
                    self.write(*dst, Some(ScalarType::I32));
                }
                KernelOp::Dequantize {
                    dst,
                    src,
                    scale,
                    zero_point,
                    ..
                } => {
                    self.read(*src);
                    self.read(*scale);
                    self.read(*zero_point);
                    self.write(*dst, Some(ScalarType::F32));
                }
                KernelOp::VecConstruct {
                    dst, components, ..
                } => {
                    for c in components {
                        self.read(*c);
                    }
                    // Vector-typed: no scalar slot type — never demoted.
                    self.write(*dst, None);
                }
                KernelOp::VecExtract { dst, vec, ty, .. } => {
                    self.read(*vec);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::MatMul { dst, a, b, ty, .. } | KernelOp::Dot { dst, a, b, ty, .. } => {
                    self.read(*a);
                    self.read(*b);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::CooperativeMMA {
                    dst, a, b, c, ty, ..
                } => {
                    self.read(*a);
                    self.read(*b);
                    self.read(*c);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::CooperativeMatrixLoad {
                    dst,
                    index,
                    stride,
                    ty,
                    ..
                } => {
                    self.read(*index);
                    self.read(*stride);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::CooperativeMatrixStore {
                    index, stride, src, ..
                } => {
                    self.read(*index);
                    self.read(*stride);
                    self.read(*src);
                }
                KernelOp::TextureSample2D { dst, x, y, ty, .. }
                | KernelOp::TextureLoad2D { dst, x, y, ty, .. } => {
                    self.read(*x);
                    self.read(*y);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::TextureSample3D {
                    dst, x, y, z, ty, ..
                } => {
                    self.read(*x);
                    self.read(*y);
                    self.read(*z);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::TextureWrite2D { x, y, value, .. } => {
                    self.read(*x);
                    self.read(*y);
                    self.read(*value);
                }
                KernelOp::TextureSize { dst_w, dst_h, .. } => {
                    self.write(*dst_w, Some(ScalarType::U32));
                    self.write(*dst_h, Some(ScalarType::U32));
                }
                KernelOp::Copy { dst, src, ty } => {
                    self.read(*src);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::Dispatch { wave, groups } => {
                    self.read(*wave);
                    for g in groups {
                        self.read(*g);
                    }
                }
                KernelOp::DeviceCall { dst, args, ty, .. } => {
                    for a in args {
                        self.read(*a);
                    }
                    self.write(*dst, Some(*ty));
                }
                KernelOp::CountTrailingZeros { dst, src, ty }
                | KernelOp::CountLeadingZeros { dst, src, ty }
                | KernelOp::PopCount { dst, src, ty }
                | KernelOp::SubgroupReduceAdd { dst, src, ty }
                | KernelOp::SubgroupReduceMin { dst, src, ty }
                | KernelOp::SubgroupReduceMax { dst, src, ty }
                | KernelOp::SubgroupExclusiveAdd { dst, src, ty }
                | KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
                    self.read(*src);
                    self.write(*dst, Some(*ty));
                }
                KernelOp::DebugPrint { src, .. } => {
                    self.read(*src);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BinOp, CmpOp};

    #[test]
    fn straight_line_temps_stay_ssa() {
        let ops = vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::BinOp {
                dst: Reg(1),
                a: Reg(0),
                b: Reg(0),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
        ];
        assert!(collect_mutable_regs(&[], &ops).is_empty());
    }

    #[test]
    fn double_write_is_demoted() {
        let ops = vec![
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(0),
            },
            KernelOp::Copy {
                dst: Reg(1),
                src: Reg(1),
                ty: ScalarType::U32,
            },
        ];
        let m = collect_mutable_regs(&[], &ops);
        assert_eq!(m.get(&1), Some(&ScalarType::U32));
    }

    #[test]
    fn branch_arm_write_read_after_merge_is_demoted() {
        // Single write inside a Branch arm, read after the merge.
        let ops = vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::Bool(true),
            },
            KernelOp::Branch {
                cond: Reg(0),
                then_ops: vec![KernelOp::Const {
                    dst: Reg(1),
                    value: ConstValue::U32(7),
                }],
                else_ops: vec![],
            },
            KernelOp::Store {
                field: 0,
                index: Reg(1),
                src: Reg(1),
                ty: ScalarType::U32,
            },
        ];
        let m = collect_mutable_regs(&[], &ops);
        assert_eq!(m.get(&1), Some(&ScalarType::U32));
    }

    #[test]
    fn branch_arm_local_temp_stays_ssa() {
        let ops = vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::Bool(true),
            },
            KernelOp::Branch {
                cond: Reg(0),
                then_ops: vec![
                    KernelOp::Const {
                        dst: Reg(1),
                        value: ConstValue::U32(7),
                    },
                    KernelOp::Store {
                        field: 0,
                        index: Reg(1),
                        src: Reg(1),
                        ty: ScalarType::U32,
                    },
                ],
                else_ops: vec![],
            },
        ];
        assert!(collect_mutable_regs(&[], &ops).is_empty());
    }

    #[test]
    fn loop_counter_stays_ssa_when_read_after_loop() {
        // The header phi dominates the merge; a post-loop read of the
        // counter must NOT demote it.
        let ops = vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::U32(4),
            },
            KernelOp::Loop {
                count: Reg(0),
                iter_reg: Reg(1),
                body: vec![],
            },
            KernelOp::Store {
                field: 0,
                index: Reg(1),
                src: Reg(1),
                ty: ScalarType::U32,
            },
        ];
        assert!(collect_mutable_regs(&[], &ops).is_empty());
    }

    #[test]
    fn loop_carried_register_is_demoted() {
        let ops = vec![
            KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::U32(4),
            },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::F32(0.0),
            },
            KernelOp::Loop {
                count: Reg(0),
                iter_reg: Reg(2),
                body: vec![KernelOp::BinOp {
                    dst: Reg(1),
                    a: Reg(1),
                    b: Reg(1),
                    op: BinOp::Add,
                    ty: ScalarType::F32,
                }],
            },
        ];
        let m = collect_mutable_regs(&[], &ops);
        assert_eq!(m.get(&1), Some(&ScalarType::F32));
        assert!(!m.contains_key(&2));
    }

    #[test]
    fn cmp_dst_records_bool_slot_type() {
        let ops = vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Cmp {
                dst: Reg(1),
                a: Reg(0),
                b: Reg(0),
                op: CmpOp::Lt,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: Reg(1),
                then_ops: vec![KernelOp::Cmp {
                    dst: Reg(1),
                    a: Reg(0),
                    b: Reg(0),
                    op: CmpOp::Eq,
                    ty: ScalarType::U32,
                }],
                else_ops: vec![],
            },
        ];
        let m = collect_mutable_regs(&[], &ops);
        assert_eq!(m.get(&1), Some(&ScalarType::Bool));
    }

    #[test]
    fn prewritten_param_rewrite_is_demoted() {
        let ops = vec![KernelOp::Const {
            dst: Reg(0),
            value: ConstValue::U32(3),
        }];
        let m = collect_mutable_regs(&[(0, ScalarType::U32)], &ops);
        assert_eq!(m.get(&0), Some(&ScalarType::U32));
    }
}

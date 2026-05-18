//! IR validator — step 082 Layer 4.
//!
//! Walks a `KernelDef` and checks every `ScalarType` it mentions
//! against a `BackendCaps` row. If the backend rejects any type
//! used by the kernel, the validator collects every offending
//! op-and-reason pair and returns them all at once. The caller
//! (proc-macro AOT pipeline, JIT driver) uses the result to skip
//! emission cleanly instead of producing nonsense backend output.
//!
//! Exhaustive-not-short-circuit because:
//! - Kernel validation is microsecond-cheap relative to xcrun /
//!   LLVM, so the extra work is negligible.
//! - Listing every problem in one error lets users fix the kernel
//!   in a single edit instead of one-failure-per-build.
//!
//! Only `ScalarType` checks today; per-op feature flags (atomic
//! ordering modes, subgroup uniformity, saturating arithmetic) are
//! a planned extension when concrete need surfaces.

use crate::caps::{BackendCaps, TypeSupport};
use crate::types::{ConstValue, KernelDef, KernelOp, KernelParam, ScalarType};

/// One unsupported-type finding from a validation pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Human-readable description of where the offending type
    /// appears (e.g. "param `out` field-write", "BinOp::Mul",
    /// "Cast from F64").
    pub location: String,
    /// The unsupported scalar type.
    pub ty: ScalarType,
    /// Backend-supplied reason string.
    pub reason: &'static str,
}

/// Aggregate result of a validation pass. `issues` is the union of
/// every unsupported-type finding; empty means the kernel can be
/// emitted for this backend.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ValidationReport {
    pub backend_name: &'static str,
    pub kernel_name: String,
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.issues.is_empty()
    }

    /// One-line summary suitable for build-time logging: lists each
    /// distinct unsupported `ScalarType` once with the occurrence
    /// count, dropping the per-op locations. The full per-issue
    /// detail stays available through `Display`, which JIT drivers
    /// use when surfacing runtime `NotSupported` errors.
    pub fn summary(&self) -> String {
        if self.issues.is_empty() {
            return format!(
                "kernel `{}` validated for backend `{}`",
                self.kernel_name, self.backend_name
            );
        }
        // Group by ScalarType, preserve first-seen order.
        let mut order: Vec<ScalarType> = Vec::new();
        let mut count: Vec<(ScalarType, usize, &'static str)> = Vec::new();
        for issue in &self.issues {
            if let Some(entry) = count.iter_mut().find(|(t, _, _)| *t == issue.ty) {
                entry.1 += 1;
            } else {
                order.push(issue.ty);
                count.push((issue.ty, 1, issue.reason));
            }
        }
        let types_summary: Vec<String> = count
            .iter()
            .map(|(t, n, r)| format!("{:?} ({} sites — {})", t, n, r))
            .collect();
        format!(
            "kernel `{}` cannot be emitted for backend `{}`: {}",
            self.kernel_name,
            self.backend_name,
            types_summary.join(", ")
        )
    }
}

impl core::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "kernel `{}` cannot be emitted for backend `{}`:",
            self.kernel_name, self.backend_name
        )?;
        for issue in &self.issues {
            write!(
                f,
                "\n  - {:?} at {}: {}",
                issue.ty, issue.location, issue.reason
            )?;
        }
        Ok(())
    }
}

/// Validate a kernel against a backend's capability table. Returns
/// a report listing every `ScalarType` use the backend rejects.
pub fn validate_for(caps: &BackendCaps, kernel: &KernelDef) -> ValidationReport {
    let mut report = ValidationReport {
        backend_name: caps.backend.name(),
        kernel_name: kernel.name.clone(),
        issues: Vec::new(),
    };

    // Parameters: each field/uniform/texture declares a scalar type.
    for (i, param) in kernel.params.iter().enumerate() {
        let (name, ty) = param_name_and_type(param);
        check(caps, &mut report, ty, &format!("param[{}] `{}`", i, name));
    }

    // Body: walk every op recursively (Branch / Loop nest sub-ops).
    walk_ops(caps, &mut report, &kernel.body, "body");

    report
}

fn check(caps: &BackendCaps, report: &mut ValidationReport, ty: ScalarType, location: &str) {
    if let TypeSupport::NotSupported(reason) = caps.scalar(ty) {
        report.issues.push(ValidationIssue {
            location: location.to_string(),
            ty,
            reason,
        });
    }
}

fn param_name_and_type(p: &KernelParam) -> (&str, ScalarType) {
    match p {
        KernelParam::FieldRead {
            name, scalar_type, ..
        }
        | KernelParam::FieldWrite {
            name, scalar_type, ..
        }
        | KernelParam::Constant {
            name, scalar_type, ..
        }
        | KernelParam::Texture2DRead {
            name, scalar_type, ..
        }
        | KernelParam::Texture2DWrite {
            name, scalar_type, ..
        }
        | KernelParam::Texture3DRead {
            name, scalar_type, ..
        } => (name, *scalar_type),
    }
}

fn const_value_type(v: &ConstValue) -> ScalarType {
    match v {
        ConstValue::F16(_) => ScalarType::F16,
        ConstValue::F32(_) => ScalarType::F32,
        ConstValue::F64(_) => ScalarType::F64,
        ConstValue::U32(_) => ScalarType::U32,
        ConstValue::U64(_) => ScalarType::U64,
        ConstValue::I32(_) => ScalarType::I32,
        ConstValue::I64(_) => ScalarType::I64,
        ConstValue::Bool(_) => ScalarType::Bool,
    }
}

fn walk_ops(caps: &BackendCaps, report: &mut ValidationReport, ops: &[KernelOp], context: &str) {
    for (i, op) in ops.iter().enumerate() {
        let loc = format!("{}[{}]", context, i);
        walk_op(caps, report, op, &loc);
    }
}

fn walk_op(caps: &BackendCaps, report: &mut ValidationReport, op: &KernelOp, loc: &str) {
    use KernelOp::*;
    match op {
        Load { ty, .. }
        | Store { ty, .. }
        | SharedDecl { ty, .. }
        | SharedDeclDyn { ty, .. }
        | SharedLoad { ty, .. }
        | SharedStore { ty, .. }
        | BinOp { ty, .. }
        | UnaryOp { ty, .. }
        | Cmp { ty, .. }
        | MathCall { ty, .. }
        | AtomicOp { ty, .. }
        | AtomicCas { ty, .. }
        | SharedAtomicOp { ty, .. }
        | WaveShuffle { ty, .. }
        | VecConstruct { ty, .. }
        | VecExtract { ty, .. }
        | MatMul { ty, .. }
        | CooperativeMMA { ty, .. }
        | TextureSample2D { ty, .. }
        | TextureSample3D { ty, .. }
        | TextureLoad2D { ty, .. }
        | TextureWrite2D { ty, .. }
        | Copy { ty, .. }
        | DeviceCall { ty, .. }
        | CountTrailingZeros { ty, .. }
        | CountLeadingZeros { ty, .. }
        | PopCount { ty, .. }
        | Dot { ty, .. }
        | SubgroupReduceAdd { ty, .. }
        | SubgroupReduceMin { ty, .. }
        | SubgroupReduceMax { ty, .. }
        | SubgroupExclusiveAdd { ty, .. }
        | SubgroupInclusiveAdd { ty, .. }
        | DebugPrint { ty, .. } => {
            check(caps, report, *ty, &format!("{}: {}", loc, op_name(op)));
        }
        Cast { from, to, .. } | Bitcast { from, to, .. } => {
            check(
                caps,
                report,
                *from,
                &format!("{}: {} from", loc, op_name(op)),
            );
            check(caps, report, *to, &format!("{}: {} to", loc, op_name(op)));
        }
        Const { value, .. } => {
            check(
                caps,
                report,
                const_value_type(value),
                &format!("{}: Const", loc),
            );
        }
        Branch {
            then_ops, else_ops, ..
        } => {
            walk_ops(caps, report, then_ops, &format!("{}.then", loc));
            walk_ops(caps, report, else_ops, &format!("{}.else", loc));
        }
        Loop { body, .. } => {
            walk_ops(caps, report, body, &format!("{}.body", loc));
        }
        // Ops with no ScalarType field — thread indexing, control
        // flow, fences, ballot/any/all (predicates are bool-typed,
        // not parameterised by ScalarType), texture-size queries
        // (always returns u32). These are universally supported.
        QuarkId { .. }
        | QuarkCount { .. }
        | ProtonId { .. }
        | NucleusId { .. }
        | ProtonSize { .. }
        | Barrier
        | Fence { .. }
        | Break
        | WaveBallot { .. }
        | WaveAny { .. }
        | WaveAll { .. }
        | TextureSize { .. }
        | Dispatch { .. }
        | SubgroupSize { .. } => {}
    }
}

fn op_name(op: &KernelOp) -> &'static str {
    use KernelOp::*;
    match op {
        Load { .. } => "Load",
        Store { .. } => "Store",
        SharedDecl { .. } => "SharedDecl",
        SharedLoad { .. } => "SharedLoad",
        SharedStore { .. } => "SharedStore",
        BinOp { .. } => "BinOp",
        UnaryOp { .. } => "UnaryOp",
        Cmp { .. } => "Cmp",
        Branch { .. } => "Branch",
        Loop { .. } => "Loop",
        MathCall { .. } => "MathCall",
        QuarkId { .. } => "QuarkId",
        QuarkCount { .. } => "QuarkCount",
        ProtonId { .. } => "ProtonId",
        NucleusId { .. } => "NucleusId",
        ProtonSize { .. } => "ProtonSize",
        Barrier => "Barrier",
        Fence { .. } => "Fence",
        AtomicOp { .. } => "AtomicOp",
        AtomicCas { .. } => "AtomicCas",
        SharedAtomicOp { .. } => "SharedAtomicOp",
        WaveShuffle { .. } => "WaveShuffle",
        WaveBallot { .. } => "WaveBallot",
        WaveAny { .. } => "WaveAny",
        WaveAll { .. } => "WaveAll",
        Cast { .. } => "Cast",
        Const { .. } => "Const",
        VecConstruct { .. } => "VecConstruct",
        VecExtract { .. } => "VecExtract",
        MatMul { .. } => "MatMul",
        CooperativeMMA { .. } => "CooperativeMMA",
        TextureSample2D { .. } => "TextureSample2D",
        TextureSample3D { .. } => "TextureSample3D",
        TextureWrite2D { .. } => "TextureWrite2D",
        TextureSize { .. } => "TextureSize",
        Copy { .. } => "Copy",
        Break => "Break",
        Dispatch { .. } => "Dispatch",
        DeviceCall { .. } => "DeviceCall",
        Bitcast { .. } => "Bitcast",
        CountTrailingZeros { .. } => "CountTrailingZeros",
        CountLeadingZeros { .. } => "CountLeadingZeros",
        PopCount { .. } => "PopCount",
        Dot { .. } => "Dot",
        SubgroupSize { .. } => "SubgroupSize",
        SubgroupReduceAdd { .. } => "SubgroupReduceAdd",
        SubgroupReduceMin { .. } => "SubgroupReduceMin",
        SubgroupReduceMax { .. } => "SubgroupReduceMax",
        SubgroupExclusiveAdd { .. } => "SubgroupExclusiveAdd",
        SubgroupInclusiveAdd { .. } => "SubgroupInclusiveAdd",
        TextureLoad2D { .. } => "TextureLoad2D",
        SharedDeclDyn { .. } => "SharedDeclDyn",
        DebugPrint { .. } => "DebugPrint",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::{CPU, METAL, VULKAN, WEBGPU};
    use crate::types::{BinOp, Reg};

    fn kernel_with_body(name: &str, body: Vec<KernelOp>) -> KernelDef {
        KernelDef {
            name: name.into(),
            params: vec![],
            body,
            body_source: None,
            next_reg: 1,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    #[test]
    fn f64_binop_rejected_on_metal() {
        let k = kernel_with_body(
            "f64_mul",
            vec![KernelOp::BinOp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::F64,
            }],
        );
        let report = validate_for(&METAL, &k);
        assert!(!report.is_ok(), "Metal should reject F64 BinOp");
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].ty, ScalarType::F64);
    }

    #[test]
    fn f32_passes_on_every_backend() {
        let k = kernel_with_body(
            "f32_mul",
            vec![KernelOp::BinOp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::F32,
            }],
        );
        for caps in [&METAL, &VULKAN, &WEBGPU, &CPU] {
            assert!(
                validate_for(caps, &k).is_ok(),
                "{} should accept F32",
                caps.backend.name()
            );
        }
    }

    #[test]
    fn u64_rejected_on_webgpu() {
        let k = kernel_with_body(
            "u64_add",
            vec![KernelOp::BinOp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::U64,
            }],
        );
        assert!(
            validate_for(&WEBGPU, &k)
                .issues
                .iter()
                .any(|i| i.ty == ScalarType::U64)
        );
    }

    #[test]
    fn validator_collects_all_unsupported_uses_not_just_first() {
        // Two unrelated F64 ops; Metal should report both.
        let k = kernel_with_body(
            "two_f64_ops",
            vec![
                KernelOp::BinOp {
                    dst: Reg(0),
                    a: Reg(1),
                    b: Reg(2),
                    op: BinOp::Mul,
                    ty: ScalarType::F64,
                },
                KernelOp::UnaryOp {
                    dst: Reg(3),
                    a: Reg(0),
                    op: crate::types::UnaryOp::Neg,
                    ty: ScalarType::F64,
                },
            ],
        );
        let report = validate_for(&METAL, &k);
        assert_eq!(
            report.issues.len(),
            2,
            "expected both F64 ops to be reported, got {:?}",
            report.issues
        );
    }

    #[test]
    fn param_types_validated_too() {
        let mut k = kernel_with_body("read_f64", vec![]);
        k.params.push(KernelParam::FieldRead {
            name: "out".into(),
            slot: 0,
            scalar_type: ScalarType::F64,
        });
        assert!(
            !validate_for(&METAL, &k).is_ok(),
            "F64 param should be flagged on Metal"
        );
    }

    #[test]
    fn const_value_type_inspected() {
        let k = kernel_with_body(
            "f64_const",
            vec![KernelOp::Const {
                dst: Reg(0),
                value: ConstValue::F64(3.14),
            }],
        );
        let report = validate_for(&METAL, &k);
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].ty, ScalarType::F64);
    }

    #[test]
    fn cast_checks_both_from_and_to() {
        let k = kernel_with_body(
            "cast_to_f64",
            vec![KernelOp::Cast {
                dst: Reg(0),
                src: Reg(1),
                from: ScalarType::F32,
                to: ScalarType::F64,
            }],
        );
        let report = validate_for(&METAL, &k);
        // Only `to` is rejected — F32 is fine on Metal.
        assert_eq!(report.issues.len(), 1);
        assert_eq!(report.issues[0].ty, ScalarType::F64);
    }

    #[test]
    fn requires_feature_does_not_reject() {
        // VULKAN.f64 = RequiresFeature("shaderFloat64"). The
        // validator treats RequiresFeature as soft: runtime device
        // caps are the source of truth.
        let k = kernel_with_body(
            "f64_mul",
            vec![KernelOp::BinOp {
                dst: Reg(0),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: ScalarType::F64,
            }],
        );
        assert!(validate_for(&VULKAN, &k).is_ok());
    }

    #[test]
    fn summary_groups_by_type() {
        // Three F64 issues + two I64 issues. The summary should
        // group them so build logs aren't drowned in per-site noise.
        let k = kernel_with_body(
            "mixed_64bit",
            vec![
                KernelOp::Const {
                    dst: Reg(0),
                    value: ConstValue::F64(1.0),
                },
                KernelOp::Const {
                    dst: Reg(1),
                    value: ConstValue::F64(2.0),
                },
                KernelOp::Const {
                    dst: Reg(2),
                    value: ConstValue::F64(3.0),
                },
                KernelOp::Const {
                    dst: Reg(3),
                    value: ConstValue::I64(10),
                },
                KernelOp::Const {
                    dst: Reg(4),
                    value: ConstValue::I64(20),
                },
            ],
        );
        let s = validate_for(&METAL, &k).summary();
        // Metal accepts I64 but rejects F64.
        assert!(
            s.contains("F64 (3 sites"),
            "expected grouped F64 count, got: {}",
            s
        );
        assert!(
            !s.contains("I64"),
            "I64 is Native on Metal; should not appear: {}",
            s
        );
        // One-line: no embedded newlines.
        assert!(!s.contains('\n'), "summary must be single-line: {:?}", s);
    }

    #[test]
    fn summary_ok_for_clean_kernel() {
        let k = kernel_with_body("noop", vec![]);
        let s = validate_for(&METAL, &k).summary();
        assert!(
            s.contains("validated"),
            "expected 'validated' message, got: {}",
            s
        );
    }

    #[test]
    fn nested_branch_body_walked() {
        let inner = KernelOp::BinOp {
            dst: Reg(0),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::Mul,
            ty: ScalarType::F64,
        };
        let k = kernel_with_body(
            "branch_f64",
            vec![KernelOp::Branch {
                cond: Reg(3),
                then_ops: vec![inner.clone()],
                else_ops: vec![inner],
            }],
        );
        let report = validate_for(&METAL, &k);
        assert_eq!(
            report.issues.len(),
            2,
            "expected both then-body and else-body F64 ops to be reported"
        );
    }
}

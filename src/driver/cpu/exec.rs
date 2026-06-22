//! Per-thread kernel execution — walks KernelOp instructions.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use std::collections::HashMap;
use std::sync::Mutex;

use quanta_ir::{BinOp, KernelOp, Reg, ScalarType};

use super::eval::{eval_atomic, eval_binop, eval_cast, eval_cmp, eval_math, eval_unary};
use super::value::{
    Value, read_scalar, read_scalar_at_offset, scalar_size, value_from_const, write_scalar,
};

// ── Execution context ────────────────────────────────────────────────────────

/// Signal to break out of the current loop.
pub(super) struct BreakSignal;

/// CPU subgroup (warp) width. Subgroup reductions on the software lane
/// are resolved cooperatively over chunks of this many lanes within a
/// workgroup, matching typical hardware warp/subgroup width. The prims
/// block kernels derive their warp structure from `subgroup_size()`, so
/// this must equal what [`KernelOp::SubgroupSize`] reports.
pub(super) const SUBGROUP_SIZE: u32 = 32;

/// How a segment is being executed with respect to subgroup ops.
///
/// CPU execution runs every lane of a workgroup through a barrier segment
/// before the next segment, which makes shared memory cooperative — but a
/// subgroup reduction needs *all* lanes' inputs, and a lane reaching the
/// reduce can't see lanes that haven't run yet. We resolve this per warp
/// with two passes:
///
/// - `Collect`: a side-effect-free dry run that records each subgroup op's
///   input value per lane (memory writes suppressed; reads still happen so
///   register values are realistic). Subgroup ops return their own input
///   as a placeholder so dependent code still computes.
/// - `Resolve`: the real run. Subgroup ops return the precomputed
///   cooperative result for this lane+site (from `resolved`); all memory
///   writes happen normally.
///
/// `None`-mode (no subgroup ops in the segment) runs once, normally.
pub(super) enum SubgroupMode<'r> {
    /// Normal single-pass execution (segment has no subgroup ops).
    None,
    /// Dry run: capture subgroup-op (kind, input) per lane, suppress
    /// memory writes.
    Collect {
        /// Per-site `(kind, input)` for THIS lane, pushed in execution
        /// order.
        inputs: &'r mut Vec<(SubgroupKind, Value)>,
    },
    /// Real run: subgroup ops read their resolved result for this lane.
    Resolve {
        /// Per-site resolved values for THIS lane, indexed by site order.
        resolved: &'r [Value],
        /// Next site index to consume (advanced as ops execute).
        cursor: usize,
    },
}

/// The reduction/scan a subgroup op performs across a warp's lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SubgroupKind {
    ReduceAdd,
    ReduceMin,
    ReduceMax,
    InclusiveAdd,
    ExclusiveAdd,
}

/// Per-thread execution state.
///
/// `fields` is shared by reference across all quark workers so that
/// parallel-group dispatch (`wave_dispatch`) can splice work across
/// `available_parallelism()` threads without serialising on a single
/// `&mut`. Each slot's `Mutex<Vec<u8>>` serialises read/write to that
/// field, which gives `AtomicOp` / `AtomicCas` cross-group atomicity
/// for free (lock spans the read-modify-write) and a simple compute-
/// disjoint races-OK model for non-atomic `Load`/`Store`. Most kernel
/// ops are compute (no lock), so the per-op overhead is bounded by
/// field-op density rather than total op count.
pub(super) struct ExecCtx<'a> {
    pub(super) quark_id: u32,
    pub(super) local_id: u32,
    pub(super) group_id: u32,
    pub(super) group_size: u32,
    pub(super) quark_count: u32,
    pub(super) regs: HashMap<u32, Value>,
    pub(super) fields: &'a [Option<Mutex<Vec<u8>>>; 16],
    /// Shared memory per workgroup, keyed by declaration id.
    pub(super) shared: &'a mut HashMap<u32, Vec<u8>>,
    /// Push-constant payload, packed as the SPIR-V / MSL emitters
    /// see it: slot `s` reads from bytes `[s*16 .. s*16+size_of::<T>()]`.
    /// A `KernelOp::Load` with `index = Reg(u32::MAX)` is the
    /// sentinel for a push-constant read.
    pub(super) push_data: &'a [u8; crate::api::wave::PUSH_DATA_CAP],
    /// Subgroup-op resolution mode for the current segment pass.
    pub(super) subgroup: SubgroupMode<'a>,
}

impl ExecCtx<'_> {
    /// Whether memory writes should be suppressed (the `Collect` dry run).
    fn writes_suppressed(&self) -> bool {
        matches!(self.subgroup, SubgroupMode::Collect { .. })
    }
}

pub(super) fn execute_ops(
    ctx: &mut ExecCtx,
    ops: &[KernelOp],
) -> Result<Option<BreakSignal>, String> {
    for op in ops {
        match op {
            KernelOp::QuarkId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.quark_id));
            }
            KernelOp::QuarkCount { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.quark_count));
            }
            KernelOp::ProtonId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.local_id));
            }
            KernelOp::NucleusId { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.group_id));
            }
            KernelOp::ProtonSize { dst } => {
                ctx.regs.insert(dst.0, Value::U32(ctx.group_size));
            }
            KernelOp::Const { dst, value } => {
                ctx.regs.insert(dst.0, value_from_const(value));
            }
            KernelOp::Load {
                dst,
                field,
                index,
                ty,
            } => {
                // index = Reg(u32::MAX) is the SPIR-V/MSL sentinel
                // for "this is a scalar push-constant Load" — see
                // `quanta_wasm_lowering::lower` line ~395. Reading
                // from `ctx.push_data` mirrors what those emitters
                // do at codegen time.
                if index.0 == u32::MAX {
                    let slot = *field as usize;
                    let offset = slot * 16;
                    let val = read_scalar_at_offset(&ctx.push_data[..], offset, ty);
                    ctx.regs.insert(dst.0, val);
                } else {
                    let idx = reg(ctx, index)?;
                    let slot = *field as usize;
                    let lock = ctx.fields[slot]
                        .as_ref()
                        .ok_or_else(|| format!("Load: field slot {slot} not bound"))?;
                    let buf = lock.lock().unwrap();
                    let val = read_scalar(&buf, idx.as_u32(), ty);
                    ctx.regs.insert(dst.0, val);
                }
            }
            KernelOp::Store {
                field,
                index,
                src,
                ty,
            } => {
                let idx = reg(ctx, index)?;
                let val = reg(ctx, src)?;
                // Suppress writes during the subgroup Collect dry run.
                if !ctx.writes_suppressed() {
                    let slot = *field as usize;
                    let lock = ctx.fields[slot]
                        .as_ref()
                        .ok_or_else(|| format!("Store: field slot {slot} not bound"))?;
                    let mut buf = lock.lock().unwrap();
                    write_scalar(&mut buf, idx.as_u32(), val, ty);
                }
            }
            KernelOp::BinOp { dst, a, b, op, ty } => {
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                ctx.regs.insert(dst.0, eval_binop(va, vb, op, ty));
            }
            KernelOp::UnaryOp { dst, a, op, ty } => {
                let va = reg(ctx, a)?;
                ctx.regs.insert(dst.0, eval_unary(va, op, ty));
            }
            KernelOp::Cmp { dst, a, b, op, ty } => {
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                ctx.regs.insert(dst.0, eval_cmp(va, vb, op, ty));
            }
            KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            } => {
                let cv = reg(ctx, cond)?;
                let branch_ops = if cv.as_bool() { then_ops } else { else_ops };
                if let Some(brk) = execute_ops(ctx, branch_ops)? {
                    return Ok(Some(brk));
                }
            }
            KernelOp::Loop {
                count,
                iter_reg,
                body,
            } => {
                let n = reg(ctx, count)?.as_u32();
                'lp: for i in 0..n {
                    ctx.regs.insert(iter_reg.0, Value::U32(i));
                    if let Some(_brk) = execute_ops(ctx, body)? {
                        break 'lp;
                    }
                }
            }
            KernelOp::Break => {
                return Ok(Some(BreakSignal));
            }
            KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            } => {
                let arg_vals: Vec<Value> =
                    args.iter().map(|r| reg(ctx, r)).collect::<Result<_, _>>()?;
                ctx.regs.insert(dst.0, eval_math(func, &arg_vals, ty));
            }
            KernelOp::Cast { dst, src, from, to } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, eval_cast(v, from, to));
            }
            KernelOp::Copy { dst, src, .. } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            // Per-tensor symmetric quantization (the oracle for the GPU
            // emitters). Quantize: f32 → integer code; Dequantize: code →
            // f32. zero_point is 0 for Symmetric (the register is read for
            // shape parity but unused).
            KernelOp::Quantize {
                dst,
                src,
                scale,
                scheme,
                ..
            } => {
                let x = reg(ctx, src)?.as_f32();
                let s = reg(ctx, scale)?.as_f32();
                let q = quanta_ir::dtype::quantize_sym(x, s, scheme.value.bits());
                ctx.regs.insert(dst.0, Value::I32(q));
            }
            KernelOp::Dequantize {
                dst, src, scale, ..
            } => {
                let q = reg(ctx, src)?.as_i32();
                let s = reg(ctx, scale)?.as_f32();
                ctx.regs
                    .insert(dst.0, Value::F32(quanta_ir::dtype::dequantize_sym(q, s)));
            }
            KernelOp::SharedDecl { id, ty, count } => {
                let size = scalar_size(ty) * (*count as usize);
                ctx.shared.entry(*id).or_insert_with(|| vec![0u8; size]);
            }
            KernelOp::SharedLoad { dst, id, index, ty } => {
                let idx = reg(ctx, index)?.as_u32();
                let buf = ctx
                    .shared
                    .get(id)
                    .ok_or_else(|| format!("SharedLoad: shared id {id} not declared"))?;
                let val = read_scalar(buf, idx, ty);
                ctx.regs.insert(dst.0, val);
            }
            KernelOp::SharedStore { id, index, src, ty } => {
                let idx = reg(ctx, index)?.as_u32();
                let val = reg(ctx, src)?;
                // Suppress writes during the subgroup Collect dry run.
                if !ctx.writes_suppressed() {
                    let buf = ctx
                        .shared
                        .get_mut(id)
                        .ok_or_else(|| format!("SharedStore: shared id {id} not declared"))?;
                    write_scalar(buf, idx, val, ty);
                }
            }
            KernelOp::Barrier => {
                // No-op: sequential execution means shared memory is always visible.
            }
            KernelOp::Fence { .. } => {
                // No-op: the CPU interpreter executes opcodes sequentially in
                // a single thread, so every program order is also a memory
                // order. Fence on a non-multithreaded executor is a noop.
            }
            KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
                order: _,
            } => {
                let idx = reg(ctx, index)?.as_u32();
                let operand = reg(ctx, val)?;
                let slot = *field as usize;
                let lock = ctx.fields[slot]
                    .as_ref()
                    .ok_or_else(|| format!("AtomicOp: field slot {slot} not bound"))?;
                // Hold the lock across read-modify-write so concurrent
                // groups see atomic semantics.
                let mut buf = lock.lock().unwrap();
                let old = read_scalar(&buf, idx, ty);
                let (new_val, old_val) = eval_atomic(old, operand, op, ty);
                // Suppress the write during the subgroup Collect dry run
                // (still return the read value so dependent regs compute).
                if !ctx.writes_suppressed() {
                    write_scalar(&mut buf, idx, new_val, ty);
                }
                ctx.regs.insert(dst.0, old_val);
            }
            KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
                success_order: _,
                failure_order: _,
            } => {
                let idx = reg(ctx, index)?.as_u32();
                let exp = reg(ctx, expected)?;
                let des = reg(ctx, desired)?;
                let slot = *field as usize;
                let lock = ctx.fields[slot]
                    .as_ref()
                    .ok_or_else(|| format!("AtomicCas: field slot {slot} not bound"))?;
                // Hold the lock across read-compare-write for CAS atomicity.
                let mut buf = lock.lock().unwrap();
                let old = read_scalar(&buf, idx, ty);
                let old_u64 = old.as_u64();
                let exp_u64 = exp.as_u64();
                // Suppress the write during the subgroup Collect dry run.
                if old_u64 == exp_u64 && !ctx.writes_suppressed() {
                    write_scalar(&mut buf, idx, des, ty);
                }
                ctx.regs.insert(dst.0, old);
            }
            // Shared-memory atomic. Same semantics as AtomicOp but
            // targets the shared-mem HashMap rather than a bound
            // buffer field. Single-thread interpreter has no real
            // concurrency, so the "atomic" here is just sequential
            // read-modify-write — same as SharedStore + SharedLoad
            // but in one step and returning the prior value.
            KernelOp::SharedAtomicOp {
                dst,
                slot,
                index,
                val,
                op,
                ty,
                order: _,
            } => {
                let idx = reg(ctx, index)?.as_u32();
                let operand = reg(ctx, val)?;
                let buf = ctx
                    .shared
                    .get(slot)
                    .ok_or_else(|| format!("SharedAtomicOp: shared id {slot} not declared"))?;
                let old = read_scalar(buf, idx, ty);
                let (new_val, old_val) = eval_atomic(old, operand, op, ty);
                // Suppress the write during the subgroup Collect dry run.
                if !ctx.writes_suppressed() {
                    let buf = ctx.shared.get_mut(slot).unwrap();
                    write_scalar(buf, idx, new_val, ty);
                }
                ctx.regs.insert(dst.0, old_val);
            }
            // Wave/subgroup intrinsics: return identity values in sequential mode
            KernelOp::WaveShuffle { dst, src, .. } => {
                // Single-thread: shuffle returns own value
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::WaveBallot { dst, .. } => {
                // Single-thread ballot: bit 0 set
                ctx.regs.insert(dst.0, Value::U32(1));
            }
            KernelOp::WaveAny { dst, predicate } => {
                let v = reg(ctx, predicate)?;
                ctx.regs.insert(dst.0, Value::Bool(v.as_bool()));
            }
            KernelOp::WaveAll { dst, predicate } => {
                let v = reg(ctx, predicate)?;
                ctx.regs.insert(dst.0, Value::Bool(v.as_bool()));
            }
            KernelOp::SubgroupReduceAdd { dst, src, .. } => {
                let v = reg(ctx, src)?;
                let r = subgroup_value(ctx, SubgroupKind::ReduceAdd, v);
                ctx.regs.insert(dst.0, r);
            }
            KernelOp::SubgroupReduceMin { dst, src, .. } => {
                let v = reg(ctx, src)?;
                let r = subgroup_value(ctx, SubgroupKind::ReduceMin, v);
                ctx.regs.insert(dst.0, r);
            }
            KernelOp::SubgroupReduceMax { dst, src, .. } => {
                let v = reg(ctx, src)?;
                let r = subgroup_value(ctx, SubgroupKind::ReduceMax, v);
                ctx.regs.insert(dst.0, r);
            }
            KernelOp::SubgroupInclusiveAdd { dst, src, .. } => {
                let v = reg(ctx, src)?;
                let r = subgroup_value(ctx, SubgroupKind::InclusiveAdd, v);
                ctx.regs.insert(dst.0, r);
            }
            KernelOp::SubgroupExclusiveAdd { dst, src, .. } => {
                let v = reg(ctx, src)?;
                let r = subgroup_value(ctx, SubgroupKind::ExclusiveAdd, v);
                ctx.regs.insert(dst.0, r);
            }
            // Vector ops
            KernelOp::VecConstruct {
                dst, components, ..
            } => {
                // Store as the first component for simple use cases
                if let Some(first) = components.first() {
                    let v = reg(ctx, first)?;
                    ctx.regs.insert(dst.0, v);
                }
            }
            KernelOp::VecExtract {
                dst,
                vec,
                component,
                ..
            } => {
                // Simplified: we store vectors as their first component
                let v = reg(ctx, vec)?;
                let _ = component;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::MatMul { dst, .. } => {
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::Dot {
                dst,
                a,
                b,
                ty,
                width,
            } => {
                // Simplified dot product: a * b (scalar, not vector)
                let va = reg(ctx, a)?;
                let vb = reg(ctx, b)?;
                let _ = width;
                ctx.regs.insert(dst.0, eval_binop(va, vb, &BinOp::Mul, ty));
            }
            // Texture ops: return zero with no-op
            KernelOp::TextureSample2D { dst, .. }
            | KernelOp::TextureSample3D { dst, .. }
            | KernelOp::TextureLoad2D { dst, .. } => {
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::TextureWrite2D { .. } => {
                // no-op
            }
            KernelOp::TextureSize { dst_w, dst_h, .. } => {
                ctx.regs.insert(dst_w.0, Value::U32(0));
                ctx.regs.insert(dst_h.0, Value::U32(0));
            }
            // Bit manipulation
            KernelOp::Bitcast { dst, src, .. } => {
                let v = reg(ctx, src)?;
                ctx.regs.insert(dst.0, v);
            }
            KernelOp::CountTrailingZeros { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().trailing_zeros()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().trailing_zeros()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            KernelOp::CountLeadingZeros { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().leading_zeros()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().leading_zeros()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            KernelOp::PopCount { dst, src, ty } => {
                let v = reg(ctx, src)?;
                let result = match ty {
                    ScalarType::U32 | ScalarType::I32 => Value::U32(v.as_u32().count_ones()),
                    ScalarType::U64 | ScalarType::I64 => Value::U32(v.as_u64().count_ones()),
                    _ => Value::U32(0),
                };
                ctx.regs.insert(dst.0, result);
            }
            // Dynamic dispatch and device calls: unsupported in V1
            KernelOp::Dispatch { .. } => {
                // Dynamic parallelism is not supported in CPU mode.
            }
            KernelOp::DeviceCall { dst, .. } => {
                // Device function calls require linked function bodies.
                // Return zero for now.
                ctx.regs.insert(dst.0, Value::U32(0));
            }
            KernelOp::CooperativeMMA { dst, .. } => {
                // Cooperative matrix multiply-accumulate: not supported in CPU mode.
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::SubgroupSize { dst } => {
                // Cooperative warp width on the CPU lane (see SUBGROUP_SIZE).
                ctx.regs.insert(dst.0, Value::U32(SUBGROUP_SIZE));
            }
            KernelOp::SharedDeclDyn { id, ty } => {
                // Dynamic shared memory: allocate a default-sized buffer.
                let size = scalar_size(ty) * 64;
                ctx.shared.entry(*id).or_insert_with(|| vec![0u8; size]);
            }
            KernelOp::DebugPrint { .. } => {
                // No-op in CPU mode.
            }
        }
    }
    Ok(None)
}

/// Read a register, returning an error if it hasn't been set.
pub(super) fn reg(ctx: &ExecCtx, r: &Reg) -> Result<Value, String> {
    ctx.regs
        .get(&r.0)
        .copied()
        .ok_or_else(|| format!("register r{} not set", r.0))
}

/// Resolve a subgroup op given its already-read input `v`, per the current
/// [`SubgroupMode`]:
/// - `Collect`: record `v`, return `v` (placeholder for the dry run).
/// - `Resolve`: return the precomputed cooperative result for this site,
///   advancing the cursor; falls back to `v` if the cursor overruns (a
///   site count mismatch — shouldn't happen, but degrade safely).
/// - `None`: return `v` (no cooperative resolution requested).
fn subgroup_value(ctx: &mut ExecCtx, kind: SubgroupKind, v: Value) -> Value {
    match &mut ctx.subgroup {
        SubgroupMode::None => v,
        SubgroupMode::Collect { inputs } => {
            inputs.push((kind, v));
            v
        }
        SubgroupMode::Resolve { resolved, cursor } => {
            let out = resolved.get(*cursor).copied().unwrap_or(v);
            *cursor += 1;
            out
        }
    }
}

/// Does this segment contain any subgroup reduce/scan op (possibly nested
/// in branches/loops)? Drives whether the dispatcher runs the cooperative
/// two-pass path for the segment.
pub(super) fn segment_has_subgroup(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| match op {
        KernelOp::SubgroupReduceAdd { .. }
        | KernelOp::SubgroupReduceMin { .. }
        | KernelOp::SubgroupReduceMax { .. }
        | KernelOp::SubgroupInclusiveAdd { .. }
        | KernelOp::SubgroupExclusiveAdd { .. } => true,
        KernelOp::Branch {
            then_ops, else_ops, ..
        } => segment_has_subgroup(then_ops) || segment_has_subgroup(else_ops),
        KernelOp::Loop { body, .. } => segment_has_subgroup(body),
        _ => false,
    })
}

/// Cooperatively reduce one warp's collected `(kind, input)` site values
/// into the per-lane resolved values. `cohort[lane][site] = (kind, input)`;
/// returns `out[lane][site] = resolved`. All lanes are assumed to hit the
/// same sequence of sites (true for non-divergent subgroup use, which is
/// the only well-defined case on real hardware too).
pub(super) fn resolve_warp(cohort: &[Vec<(SubgroupKind, Value)>]) -> Vec<Vec<Value>> {
    let lanes = cohort.len();
    let mut out: Vec<Vec<Value>> = cohort
        .iter()
        .map(|sites| Vec::with_capacity(sites.len()))
        .collect();
    if lanes == 0 {
        return out;
    }
    let num_sites = cohort.iter().map(|s| s.len()).max().unwrap_or(0);
    for site in 0..num_sites {
        // Gather this site's (kind, input) across the lanes that have it.
        let mut kind = None;
        let mut inputs: Vec<Value> = Vec::with_capacity(lanes);
        let mut present: Vec<usize> = Vec::with_capacity(lanes);
        for (lane, sites) in cohort.iter().enumerate() {
            if let Some((k, v)) = sites.get(site).copied() {
                kind.get_or_insert(k);
                inputs.push(v);
                present.push(lane);
            }
        }
        let kind = kind.unwrap_or(SubgroupKind::ReduceAdd);
        // Compute the per-lane result for the lanes present at this site.
        let results = reduce_site(kind, &inputs);
        for (slot, &lane) in present.iter().enumerate() {
            out[lane].push(results[slot]);
        }
    }
    out
}

/// Apply a subgroup reduction/scan over `inputs` (the lanes present at a
/// site, in lane order). Returns one result per input lane.
fn reduce_site(kind: SubgroupKind, inputs: &[Value]) -> Vec<Value> {
    // Type follows the first input's variant.
    let proto = inputs.first().copied().unwrap_or(Value::U32(0));
    match kind {
        SubgroupKind::ReduceAdd | SubgroupKind::ReduceMin | SubgroupKind::ReduceMax => {
            let acc = fold_all(kind, inputs, proto);
            vec![acc; inputs.len()]
        }
        SubgroupKind::InclusiveAdd => {
            let mut out = Vec::with_capacity(inputs.len());
            let mut acc = identity_add(proto);
            for &v in inputs {
                acc = add_values(acc, v);
                out.push(acc);
            }
            out
        }
        SubgroupKind::ExclusiveAdd => {
            let mut out = Vec::with_capacity(inputs.len());
            let mut acc = identity_add(proto);
            for &v in inputs {
                out.push(acc);
                acc = add_values(acc, v);
            }
            out
        }
    }
}

fn fold_all(kind: SubgroupKind, inputs: &[Value], proto: Value) -> Value {
    let mut it = inputs.iter().copied();
    let mut acc = match it.next() {
        Some(v) => v,
        None => return proto,
    };
    for v in it {
        acc = match kind {
            SubgroupKind::ReduceAdd => add_values(acc, v),
            SubgroupKind::ReduceMin => min_values(acc, v),
            SubgroupKind::ReduceMax => max_values(acc, v),
            _ => add_values(acc, v),
        };
    }
    acc
}

fn identity_add(proto: Value) -> Value {
    match proto {
        Value::F32(_) => Value::F32(0.0),
        Value::I32(_) => Value::I32(0),
        _ => Value::U32(0),
    }
}

fn add_values(a: Value, b: Value) -> Value {
    match a {
        Value::F32(_) => Value::F32(a.as_f32() + b.as_f32()),
        Value::I32(_) => Value::I32(a.as_i32().wrapping_add(b.as_i32())),
        _ => Value::U32(a.as_u32().wrapping_add(b.as_u32())),
    }
}

fn min_values(a: Value, b: Value) -> Value {
    match a {
        Value::F32(_) => Value::F32(a.as_f32().min(b.as_f32())),
        Value::I32(_) => Value::I32(a.as_i32().min(b.as_i32())),
        _ => Value::U32(a.as_u32().min(b.as_u32())),
    }
}

fn max_values(a: Value, b: Value) -> Value {
    match a {
        Value::F32(_) => Value::F32(a.as_f32().max(b.as_f32())),
        Value::I32(_) => Value::I32(a.as_i32().max(b.as_i32())),
        _ => Value::U32(a.as_u32().max(b.as_u32())),
    }
}

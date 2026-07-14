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
/// A texture bound to a compute dispatch. The pixel bytes are snapshotted
/// into a per-slot `Mutex<Vec<u8>>` (mirroring `fields`) so `TextureWrite2D`
/// from parallel groups serialises at the texture; `width`/`height`/`format`
/// drive texel indexing and decode.
pub(super) struct CpuTexSlot {
    pub(super) data: Mutex<Vec<u8>>,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) format: crate::api::types::Format,
}

pub(super) struct ExecCtx<'a> {
    pub(super) quark_id: u32,
    pub(super) local_id: u32,
    pub(super) group_id: u32,
    pub(super) group_size: u32,
    pub(super) quark_count: u32,
    pub(super) regs: HashMap<u32, Value>,
    pub(super) fields: &'a [Option<Mutex<Vec<u8>>>; 16],
    /// Textures bound by slot (parallel to `Wave::texture_bindings`).
    pub(super) textures: &'a [Option<CpuTexSlot>; 16],
    /// Shared memory per workgroup, keyed by declaration id.
    pub(super) shared: &'a mut HashMap<u32, Vec<u8>>,
    /// Push-constant payload, packed as the SPIR-V / MSL emitters
    /// see it: slot `s` reads from bytes `[s*16 .. s*16+size_of::<T>()]`.
    /// A `KernelOp::Load` with `index = Reg(u32::MAX)` is the
    /// sentinel for a push-constant read.
    pub(super) push_data: &'a [u8; crate::api::types::PUSH_DATA_CAP],
    /// Subgroup-op resolution mode for the current segment pass.
    pub(super) subgroup: SubgroupMode<'a>,
}

impl ExecCtx<'_> {
    /// Whether memory writes should be suppressed (the `Collect` dry run).
    fn writes_suppressed(&self) -> bool {
        matches!(self.subgroup, SubgroupMode::Collect { .. })
    }

    /// Read the x-channel (`.x`) of a texel at integer `(x, y)` with
    /// clamp-to-edge addressing. R32Float returns the raw f32; RGBA8 returns
    /// the R channel as unorm (byte/255) — matching the GPU op contract where
    /// `texture_load_2d`/`texture_sample_2d` return the first channel (the
    /// existing RGBA8 read test extracts `.x` on Metal identically). Returns
    /// 0.0 for an unbound slot.
    fn texel_x(&self, slot: u32, x: i64, y: i64) -> f32 {
        let Some(Some(tex)) = self.textures.get(slot as usize) else {
            return 0.0;
        };
        if tex.width == 0 || tex.height == 0 {
            return 0.0;
        }
        let cx = x.clamp(0, tex.width as i64 - 1) as usize;
        let cy = y.clamp(0, tex.height as i64 - 1) as usize;
        let data = tex.data.lock().unwrap();
        let bpp = tex.format.bytes_per_pixel();
        let base = (cy * tex.width as usize + cx) * bpp;
        match tex.format {
            crate::api::types::Format::R32Float => {
                if base + 4 > data.len() {
                    return 0.0;
                }
                f32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]])
            }
            _ => {
                // RGBA8 / BGRA8 and other 8-bit-channel formats: R channel as
                // unorm. (Only R32Float write slots and RGBA8 read slots are
                // exercised; other formats fall through to the first byte.)
                if base >= data.len() {
                    return 0.0;
                }
                data[base] as f32 / 255.0
            }
        }
    }

    /// Write `value` into the x-channel of the R32Float texel at `(x, y)`.
    /// Suppressed during the subgroup `Collect` dry run. Out-of-bounds or
    /// non-R32Float writes are dropped (the format contract restricts writes
    /// to R32Float storage textures, enforced at dispatch).
    fn write_texel_x(&self, slot: u32, x: i64, y: i64, value: f32) {
        if self.writes_suppressed() {
            return;
        }
        let Some(Some(tex)) = self.textures.get(slot as usize) else {
            return;
        };
        if !matches!(tex.format, crate::api::types::Format::R32Float) {
            return;
        }
        if x < 0 || y < 0 || x >= tex.width as i64 || y >= tex.height as i64 {
            return;
        }
        let base = (y as usize * tex.width as usize + x as usize) * 4;
        let mut data = tex.data.lock().unwrap();
        if base + 4 <= data.len() {
            data[base..base + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    /// Read an RGBA8 texel at integer `(x, y)` (clamp-to-edge) as a packed
    /// `0xAABBGGRR` u32 — the four unorm bytes in little-endian R,G,B,A order,
    /// matching the packed-u32 contract of `texture_load_2d_u32` and the SPIR-V
    /// PackUnorm4x8 / MSL pack_float_to_unorm4x8 boundary on the GPU. Returns 0
    /// for an unbound slot. (This is the whole-texel twin of `texel_x`, which
    /// returns only the R channel as an f32 unorm for the sampled read path.)
    fn texel_packed_u32(&self, slot: u32, x: i64, y: i64) -> u32 {
        let Some(Some(tex)) = self.textures.get(slot as usize) else {
            return 0;
        };
        if tex.width == 0 || tex.height == 0 {
            return 0;
        }
        let cx = x.clamp(0, tex.width as i64 - 1) as usize;
        let cy = y.clamp(0, tex.height as i64 - 1) as usize;
        let data = tex.data.lock().unwrap();
        let base = (cy * tex.width as usize + cx) * 4;
        if base + 4 > data.len() {
            return 0;
        }
        u32::from_le_bytes([data[base], data[base + 1], data[base + 2], data[base + 3]])
    }

    /// Write a packed `0xAABBGGRR` u32 into the RGBA8 texel at `(x, y)`,
    /// splitting it into the four little-endian R,G,B,A bytes — the inverse of
    /// `texel_packed_u32`. Suppressed during the subgroup `Collect` dry run;
    /// out-of-bounds or non-RGBA8 writes are dropped (the packed-u32 storage
    /// contract restricts writes to RGBA8 textures, enforced at dispatch).
    fn write_texel_rgba8(&self, slot: u32, x: i64, y: i64, value: u32) {
        if self.writes_suppressed() {
            return;
        }
        let Some(Some(tex)) = self.textures.get(slot as usize) else {
            return;
        };
        if !matches!(tex.format, crate::api::types::Format::RGBA8) {
            return;
        }
        if x < 0 || y < 0 || x >= tex.width as i64 || y >= tex.height as i64 {
            return;
        }
        let base = (y as usize * tex.width as usize + x as usize) * 4;
        let mut data = tex.data.lock().unwrap();
        if base + 4 <= data.len() {
            data[base..base + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// Per-workgroup invariant context for cooperative segment execution.
///
/// Bundles the references every lane's [`ExecCtx`] needs (besides its own
/// `regs` and the shared `HashMap`, which are threaded in per call). Used by
/// the cooperative **barrier-loop** runner: a `Barrier` inside a `Loop` body
/// is a cross-lane sync point, so the loop can't run lane-by-lane as a unit
/// (see `segment_has_barrier_loop`). The runner instead executes the loop
/// **iteration-synchronized**: every lane completes each iteration's
/// pre-barrier sub-segment before any lane starts the post-barrier
/// sub-segment, exactly like the top-level barrier segmenter but per
/// iteration. This makes in-loop shared-memory stores visible across lanes.
pub(super) struct CoopGroup<'a> {
    pub(super) gid: u64,
    pub(super) threads_per_group: u64,
    pub(super) group_size: u32,
    pub(super) quark_count: u32,
    pub(super) fields: &'a [Option<Mutex<Vec<u8>>>; 16],
    pub(super) textures: &'a [Option<CpuTexSlot>; 16],
    pub(super) push_data: &'a [u8; crate::api::types::PUSH_DATA_CAP],
}

impl CoopGroup<'_> {
    fn quark_id(&self, lid: u32) -> u32 {
        (self.gid * self.threads_per_group + lid as u64) as u32
    }

    /// Run `ops` for a single lane against its `regs` and the workgroup
    /// `shared`, normal (non-subgroup) mode. Returns whether the lane hit a
    /// `Break`.
    fn run_lane_ops(
        &self,
        lid: u32,
        regs: &mut HashMap<u32, Value>,
        shared: &mut HashMap<u32, Vec<u8>>,
        ops: &[KernelOp],
    ) -> Result<bool, String> {
        let mut ctx = ExecCtx {
            quark_id: self.quark_id(lid),
            local_id: lid,
            group_id: self.gid as u32,
            group_size: self.group_size,
            quark_count: self.quark_count,
            regs: core::mem::take(regs),
            fields: self.fields,
            textures: self.textures,
            shared,
            push_data: self.push_data,
            subgroup: SubgroupMode::None,
        };
        let broke = execute_ops(&mut ctx, ops)?.is_some();
        *regs = ctx.regs;
        Ok(broke)
    }

    /// Run one op-slice across ALL lanes (a cooperative sub-segment): every
    /// lane executes `ops` before the next sub-segment, so shared writes are
    /// visible to other lanes' subsequent reads. Returns whether any lane
    /// hit a `Break` (uniform across lanes for well-defined kernels).
    fn run_subsegment_all_lanes(
        &self,
        ops: &[KernelOp],
        thread_regs: &mut [HashMap<u32, Value>],
        shared: &mut HashMap<u32, Vec<u8>>,
    ) -> Result<bool, String> {
        let tpg = self.threads_per_group as u32;
        let mut any_break = false;
        for lid in 0..tpg {
            let mut regs = core::mem::take(&mut thread_regs[lid as usize]);
            let broke = self.run_lane_ops(lid, &mut regs, shared, ops)?;
            thread_regs[lid as usize] = regs;
            any_break |= broke;
        }
        Ok(any_break)
    }

    /// Cooperatively execute a segment that may contain a barrier-bearing
    /// loop. Straight-line stretches run per-lane; a barrier-bearing `Loop`
    /// runs iteration-synchronized across all lanes (see the struct doc). A
    /// `Branch` whose taken arm contains a barrier-loop is descended into
    /// cooperatively — its condition must be uniform across lanes (the
    /// inliner's structural guards and any guard around an in-loop barrier
    /// are; a divergent in-loop barrier is GPU UB).
    pub(super) fn run_segment(
        &self,
        segment: &[KernelOp],
        thread_regs: &mut [HashMap<u32, Value>],
        shared: &mut HashMap<u32, Vec<u8>>,
    ) -> Result<(), String> {
        // Walk top-level ops, batching straight-line ops and handling each
        // op that contains a barrier-loop (a `Loop` directly, or a `Branch`
        // whose arm holds one) specially.
        let mut straight: Vec<KernelOp> = Vec::new();
        let flush = |straight: &mut Vec<KernelOp>,
                     thread_regs: &mut [HashMap<u32, Value>],
                     shared: &mut HashMap<u32, Vec<u8>>|
         -> Result<(), String> {
            if !straight.is_empty() {
                self.run_subsegment_all_lanes(straight, thread_regs, shared)?;
                straight.clear();
            }
            Ok(())
        };

        for op in segment {
            match op {
                KernelOp::Loop {
                    count,
                    iter_reg,
                    body,
                } if ops_contain_barrier(body) => {
                    flush(&mut straight, thread_regs, shared)?;
                    self.run_barrier_loop(count, iter_reg, body, thread_regs, shared)?;
                }
                KernelOp::Branch {
                    cond,
                    then_ops,
                    else_ops,
                } if ops_contain_barrier(then_ops) || ops_contain_barrier(else_ops) => {
                    flush(&mut straight, thread_regs, shared)?;
                    self.run_barrier_branch(cond, then_ops, else_ops, thread_regs, shared)?;
                }
                other => straight.push(other.clone()),
            }
        }
        flush(&mut straight, thread_regs, shared)?;
        Ok(())
    }

    /// Cooperatively execute a `Branch` whose taken arm contains a
    /// barrier-loop. The condition is uniform across lanes (required for a
    /// well-defined in-loop barrier), so evaluate it on lane 0 and run the
    /// taken arm cooperatively for every lane via `run_segment`.
    fn run_barrier_branch(
        &self,
        cond: &Reg,
        then_ops: &[KernelOp],
        else_ops: &[KernelOp],
        thread_regs: &mut [HashMap<u32, Value>],
        shared: &mut HashMap<u32, Vec<u8>>,
    ) -> Result<(), String> {
        let take_then = thread_regs
            .first()
            .and_then(|r| r.get(&cond.0))
            .map(|v| v.as_bool())
            .unwrap_or(false);
        let arm = if take_then { then_ops } else { else_ops };
        self.run_segment(arm, thread_regs, shared)
    }

    /// Iteration-synchronized execution of a barrier-bearing loop. The loop
    /// count is uniform across lanes (a divergent in-loop barrier is GPU UB);
    /// the real exit is the body's down-counter `Break`. Each iteration: set
    /// `iter_reg` in every lane, split the body at its barriers, and run each
    /// sub-segment across all lanes. Stop when the body signals `Break`.
    fn run_barrier_loop(
        &self,
        count: &Reg,
        iter_reg: &Reg,
        body: &[KernelOp],
        thread_regs: &mut [HashMap<u32, Value>],
        shared: &mut HashMap<u32, Vec<u8>>,
    ) -> Result<(), String> {
        // The loop's `count` reg bounds the iteration count (the lowerer caps
        // it at a fuel limit and relies on the body's `Break`). Read it from
        // lane 0; all lanes agree (uniform). Fall back to 0 if unset.
        let max_iters = thread_regs
            .first()
            .and_then(|r| r.get(&count.0))
            .map(|v| v.as_u32())
            .unwrap_or(0);

        // Sub-segment the body at its (top-level) barriers once.
        let sub_ranges = subsegment_ranges(body);

        for i in 0..max_iters {
            // Set the iteration register in every lane before this iteration.
            for regs in thread_regs.iter_mut() {
                regs.insert(iter_reg.0, Value::U32(i));
            }
            let mut broke = false;
            for &(s, e) in &sub_ranges {
                let sub = &body[s..e];
                if self.run_subsegment_all_lanes(sub, thread_regs, shared)? {
                    broke = true;
                }
            }
            if broke {
                break;
            }
        }
        Ok(())
    }
}

/// Split a loop body into maximal op-slices between top-level `Barrier` ops
/// (the barriers themselves are skipped). Mirrors `barrier_segment_ranges`
/// but returns ranges over a body slice — used by the cooperative
/// barrier-loop runner to sub-segment each iteration.
fn subsegment_ranges(ops: &[KernelOp]) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for (i, op) in ops.iter().enumerate() {
        if matches!(op, KernelOp::Barrier) {
            ranges.push((start, i));
            start = i + 1;
        }
    }
    ranges.push((start, ops.len()));
    ranges
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
            // Texture load/sample: nearest texel, clamp-to-edge. The GPU compute
            // samplers default to nearest for the existing read test, so sample
            // and load resolve identically on integer coords. R32Float / sampled
            // reads take the `.x` channel as f32; a packed-RGBA8 load (ty=U32)
            // returns the whole texel as a `0xAABBGGRR` u32.
            KernelOp::TextureLoad2D {
                dst,
                texture,
                x,
                y,
                ty,
            }
            | KernelOp::TextureSample2D {
                dst,
                texture,
                x,
                y,
                ty,
            } => {
                let xi = reg(ctx, x)?.as_u32() as i64;
                let yi = reg(ctx, y)?.as_u32() as i64;
                if *ty == ScalarType::U32 {
                    let v = ctx.texel_packed_u32(*texture, xi, yi);
                    ctx.regs.insert(dst.0, Value::U32(v));
                } else {
                    let v = ctx.texel_x(*texture, xi, yi);
                    ctx.regs.insert(dst.0, Value::F32(v));
                }
            }
            // 3D sampling is not implemented on the CPU executor (no compute
            // kernel exercises it); keep the zero contract.
            KernelOp::TextureSample3D { dst, .. } => {
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::TextureWrite2D {
                texture,
                x,
                y,
                value,
                ty,
            } => {
                let xi = reg(ctx, x)?.as_u32() as i64;
                let yi = reg(ctx, y)?.as_u32() as i64;
                // R32Float writes the scalar into the x channel; a packed-RGBA8
                // write (ty=U32) splits the `0xAABBGGRR` u32 into four bytes.
                if *ty == ScalarType::U32 {
                    let v = reg(ctx, value)?.as_u32();
                    ctx.write_texel_rgba8(*texture, xi, yi, v);
                } else {
                    let v = reg(ctx, value)?.as_f32();
                    ctx.write_texel_x(*texture, xi, yi, v);
                }
            }
            KernelOp::TextureSize {
                dst_w,
                dst_h,
                texture,
            } => {
                let (w, h) = match ctx.textures.get(*texture as usize) {
                    Some(Some(tex)) => (tex.width, tex.height),
                    _ => (0, 0),
                };
                ctx.regs.insert(dst_w.0, Value::U32(w));
                ctx.regs.insert(dst_h.0, Value::U32(h));
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
                // Cooperative matrix multiply-accumulate: not supported in CPU
                // mode (the CPU lane reports supports_cooperative_matrix=false,
                // so quanta-blas routes to the scalar tiled GEMM here).
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::CooperativeMatrixLoad { dst, .. } => {
                // Subgroup-collective fragment load: no CPU execution (gated
                // out by capability). Placeholder so the op is total.
                ctx.regs.insert(dst.0, Value::F32(0.0));
            }
            KernelOp::CooperativeMatrixStore { .. } => {
                // Placeholder no-op (gated out on the CPU lane).
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

/// Does this segment contain a `Loop` whose body has a `Barrier`? Such a
/// loop cannot be run lane-by-lane as one unit: the in-loop barrier is a
/// cross-lane synchronization point, so every lane must complete each
/// iteration's pre-barrier work before any lane starts the post-barrier
/// work (otherwise cross-thread shared-memory writes inside the loop aren't
/// visible). The dispatcher routes these segments to the cooperative
/// per-iteration loop runner instead of the plain single-pass loop.
pub(super) fn segment_has_barrier_loop(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| match op {
        KernelOp::Loop { body, .. } => ops_contain_barrier(body),
        KernelOp::Branch {
            then_ops, else_ops, ..
        } => segment_has_barrier_loop(then_ops) || segment_has_barrier_loop(else_ops),
        _ => false,
    })
}

/// Whether `ops` contains a `Barrier` at any nesting depth.
pub(super) fn ops_contain_barrier(ops: &[KernelOp]) -> bool {
    ops.iter().any(|op| match op {
        KernelOp::Barrier => true,
        KernelOp::Branch {
            then_ops, else_ops, ..
        } => ops_contain_barrier(then_ops) || ops_contain_barrier(else_ops),
        KernelOp::Loop { body, .. } => ops_contain_barrier(body),
        _ => false,
    })
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

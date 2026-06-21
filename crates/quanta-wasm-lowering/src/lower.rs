//! Op-by-op lowering: WASM `RawInstr` stream → Quanta `KernelOp` list.
//!
//! WASM is a stack machine; the lowering pass simulates that stack with
//! a symbolic abstract domain. Each value on the simulated stack carries
//! a `SymVal` describing what it represents:
//! - `Reg(r, ty)` — a virtual KernelOp register of given scalar type
//! - `BufferPtr(slot)` — pointer to buffer slot N (tracked from WASM
//!   parameter analysis)
//! - `ScaledIdx { base_reg, scale }` — `base_reg << log2(scale)`,
//!   recognized as the canonical "element index → byte offset" pattern
//!
//! When we see `f32.load` after a `BufferPtr(slot) + ScaledIdx{base, 4}`
//! pattern on the stack, we emit `KernelOp::Load { field: slot,
//! index: base, ty: F32 }`. The resulting register goes back on the
//! stack.
//!
//! This is enough for the simplest kernels (vector_add). Control-flow
//! lowering (block / loop / if / br_if), atomics, shared memory, and
//! intrinsic dispatch beyond `quark_id` come in subsequent commits.

use quanta_ir::{
    AtomicOp, BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MathFn, MemoryOrder, Reg,
    ScalarType,
};

use crate::{
    FunctionBodyInfo, FunctionKind, LoweringError, Module, ParamKind, RawInstr, SideTable, WasmTy,
    find_kernel,
};

/// Symbolic stack value during lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SymVal {
    /// A virtual register. `ty` is the IR scalar type.
    Reg(Reg, ScalarType),
    /// A pointer to a buffer slot. Tracked separately from regs so we
    /// recognize the `<ptr> + <offset> .load` pattern as a
    /// KernelOp::Load.
    BufferPtr(u32),
    /// Result of `<base_reg> << log2(scale)` — the "element index → byte
    /// offset" encoding rustc emits. `scale` is the element size in
    /// bytes (4 for f32, 8 for f64, 1 for u8, ...).
    ScaledIdx { base: Reg, scale: u32 },
    /// A WASM i32 constant — kept as a SymVal so we can recognize
    /// `i32.const 2; i32.shl` as a left-shift-by-2 (= scale by 4).
    I32Const(i32),
    /// A WASM i64 constant. Materialized via `KernelOp::Const` when
    /// committed; no buffer-pattern recognition for the wide form.
    I64Const(i64),
    /// `BufferPtr(slot) + ScaledIdx{base, scale}` — emitted by
    /// recognizing the canonical `<ptr> <byte_offset> i32.add` pattern.
    /// Consumed by the next f32.load/f32.store op into a
    /// `KernelOp::Load` or `KernelOp::Store`.
    BufferAccess { slot: u32, base: Reg, scale: u32 },
    /// Catch-all — when we can't yet ascribe meaning. Lowering an op
    /// that consumes an `Opaque` value falls back to emitting a Cast
    /// or BinOp with ScalarType::I32.
    Opaque(Reg, ScalarType),
}

impl SymVal {
    /// If this is a Reg/Opaque, return the register; otherwise None.
    /// Used when we have to commit a stack value to a real register
    /// (e.g. the operand of a non-memory binop).
    fn as_reg(self) -> Option<(Reg, ScalarType)> {
        match self {
            Self::Reg(r, ty) | Self::Opaque(r, ty) => Some((r, ty)),
            _ => None,
        }
    }
}

/// Per-local information: parameters in slots 0..N, then declared
/// locals N..M. Each local has a current `SymVal` that lives in it
/// (for params, it's their initial assignment; for declared locals,
/// it's the stable_reg seeded with a default at function entry, then
/// re-assigned by `local.set`/`local.tee` via `KernelOp::Copy`).
struct LocalInfo {
    /// Underlying WASM type (i32/i64/f32/f64).
    wasm_ty: WasmTy,
    /// What's in this local right now. None = symbolic-only param
    /// (BufferPtr) that doesn't have a value-register; reads of such
    /// locals rely on the existing buffer-access pattern recognition.
    val: Option<SymVal>,
    /// Pre-allocated register that holds the local's value across
    /// control-flow merges. Allocated unconditionally at function
    /// entry for every value-typed local (declared or scalar param);
    /// `None` for buffer-pointer params. Every `local.set`/`local.tee`
    /// emits `KernelOp::Copy { dst: stable_reg, src: <rhs> }` so the
    /// reg is *always* defined before any post-merge `local.get` reads
    /// it — fixes the "register defined inside Branch.else_ops" bug
    /// when WASM locals are written along multiple paths.
    stable_reg: Option<Reg>,
    /// The Quanta IR scalar type carried by `stable_reg`. Mirrors
    /// `wasm_ty` for declared locals; for scalar params it's the
    /// side-table-declared type (which may be u32/i32/u8/… all of
    /// which are i32 in WASM).
    stable_ty: ScalarType,
}

/// Top-level lowering driver: WASM bytes + side table → KernelDef.
pub fn lower_module(wasm: &[u8], side_table: &SideTable) -> Result<KernelDef, LoweringError> {
    let module = crate::parse_module(wasm)?;
    let (_idx, func) = find_kernel(&module, &side_table.kernel_name)?;
    let body = match &func.kind {
        FunctionKind::Defined(b) => b,
        FunctionKind::Imported { .. } => {
            return Err(LoweringError::ShapeMismatch(format!(
                "kernel `{}` is imported, not defined",
                side_table.kernel_name
            )));
        }
    };
    let sig = module
        .types
        .get(func.type_index as usize)
        .ok_or_else(|| LoweringError::ShapeMismatch("function has no type entry".into()))?;

    // Validate side-table shape against the WASM signature.
    if sig.params.len() != side_table.params.len() {
        return Err(LoweringError::ShapeMismatch(format!(
            "kernel `{}` has {} WASM params but side table has {}",
            side_table.kernel_name,
            sig.params.len(),
            side_table.params.len(),
        )));
    }

    let ctx = LowerCtx::new(side_table, sig.params.to_vec(), body, &module);
    ctx.lower()
}

struct LowerCtx<'a> {
    side_table: &'a SideTable,
    /// WASM parameter types in declaration order.
    param_types: Vec<WasmTy>,
    body: &'a FunctionBodyInfo,
    module: &'a Module,

    locals: Vec<LocalInfo>,
    stack: Vec<SymVal>,
    /// Stack of control-flow frames. Index 0 is the function-level
    /// frame; each block/loop/if pushes a new frame on top. Ops are
    /// always emitted to the topmost frame's `ops`. When a frame
    /// closes (End op), it folds back into its parent — Block flushes
    /// its ops, Loop wraps in a `KernelOp::Loop`, If/Else wrap in a
    /// `KernelOp::Branch`.
    frames: Vec<Frame>,
    next_reg: u32,
    /// Known imported function indices keyed by their import name —
    /// used to recognize `call $quark_id` etc.
    intrinsic_names: Vec<String>,
    /// One entry per OPEN Loop frame (innermost last): the set of
    /// locals read (`local.get`/`local.tee`-observed) since that
    /// loop opened. Drives `local_is_loop_carried` — a write to a
    /// local with a prior in-loop read is loop-carried and must be
    /// materialized through its stable register.
    loop_reads: Vec<std::collections::BTreeSet<u32>>,
}

/// One br/br_if's intent recorded on its target frame.
///
/// No Branch is materialised eagerly; we mark the target frame with
/// WHERE the br_if happened (`sink_position`) and WHAT its cond was.
/// At end-of-frame, the records are walked in reverse and the
/// `frame.ops[sink_position..]` are wrapped in real
/// `Branch{cond, then, else}` ops reflecting the actual taken /
/// fall-through paths.
#[derive(Clone, Copy, Debug)]
struct BrIfRecord {
    /// Index into the target frame's `ops` where the br/br_if fired.
    /// Ops at this position or later are on the br_if's
    /// fall-through path (cond=false for br_if; never reached for
    /// unconditional Br) until the next record's position.
    sink_position: usize,
    /// The condition register for br_if. For unconditional Br
    /// (`is_unconditional = true`), this is unused; we still allocate
    /// a register so the field can stay non-optional.
    cond: Reg,
    /// Distinguishes `Br N` (unconditional) from `BrIf N`. Reserved
    /// for the next session — session 1 only handles BrIf.
    is_unconditional: bool,
}

/// One control-flow frame on the lowering stack.
struct Frame {
    kind: FrameKind,
    ops: Vec<KernelOp>,
    /// Snapshot of `(local_idx → val)` taken at frame entry.
    /// Used at frame close: any local whose current `val` differs
    /// from the snapshot was modified inside this frame, and its
    /// `val` needs to be reset to `stable_reg` so post-frame reads
    /// don't reference a register that was defined inside a
    /// now-closed scope (e.g. inside `Branch.else_ops` or
    /// `Loop.body`).
    ///
    /// Only populated for frames that introduce a scope boundary
    /// (If, Else, Loop). Block frames splice into the parent so
    /// no merge is needed. See workspace design doc at
    /// `roadmap/_design/wasm_local_renaming.md`.
    local_snapshot: Vec<(u32, Option<SymVal>)>,
    /// br/br_if records targeting this frame. See `BrIfRecord`
    /// for the wrap-at-end-of-frame semantics.
    brifs: Vec<BrIfRecord>,
    /// Snapshot of parent.brifs.len() at the moment THIS
    /// frame opened. Used at this frame's close to bump every parent
    /// brif appended during this frame's lifetime by `self.ops.len()`.
    ///
    /// Why: a br/br_if recorded on parent DURING this frame's
    /// lifetime is, semantically, a branch that fires AFTER all of
    /// this frame's content executes. The record's `sink_position`
    /// at record time = parent.ops.len() at that moment, which is
    /// BEFORE this frame's splice contributes. Without bumping, the
    /// resulting wrap engulfs this frame's content, making it
    /// conditional on the brif's cond — wrong semantics for ops
    /// that already ran in this frame before the brif fired.
    parent_brifs_at_open: usize,
}

/// Reduction operator for the `subgroup_reduce` helper. Mirrors
/// the three `SubgroupReduce*` IR ops.
#[derive(Copy, Clone)]
enum SubgroupOp {
    Add,
    Min,
    Max,
}

enum FrameKind {
    /// The outermost frame — the function itself.
    Function,
    /// `block ... end`. WASM blocks are label scopes only; on close
    /// we just splice our ops into the parent.
    Block,
    /// `loop ... end`. Closes into `KernelOp::Loop { count, iter_reg, body }`.
    Loop { count_reg: Reg, iter_reg: Reg },
    /// `if ... end` (no else clause yet). Closes into
    /// `KernelOp::Branch { cond, then_ops: this.ops, else_ops: [] }`.
    If { cond: Reg },
    /// `if ... else ... end`. The `then_ops` we collected before the
    /// `else` are saved here; current `ops` collect the else-arm.
    Else { cond: Reg, then_ops: Vec<KernelOp> },
}

/// Lightweight discriminant tag used to peek at a frame's kind without
/// holding a borrow — needed when we want to inspect the target of a
/// `br`/`br_if` then mutate it.
#[derive(Copy, Clone, Debug)]
enum FrameKindTag {
    Function,
    Block,
    Loop,
    If,
    Else,
}

impl Frame {
    fn kind_discriminant(&self) -> FrameKindTag {
        match self.kind {
            FrameKind::Function => FrameKindTag::Function,
            FrameKind::Block => FrameKindTag::Block,
            FrameKind::Loop { .. } => FrameKindTag::Loop,
            FrameKind::If { .. } => FrameKindTag::If,
            FrameKind::Else { .. } => FrameKindTag::Else,
        }
    }
}

impl<'a> LowerCtx<'a> {
    fn new(
        side_table: &'a SideTable,
        param_types: Vec<WasmTy>,
        body: &'a FunctionBodyInfo,
        module: &'a Module,
    ) -> Self {
        // Seed locals with parameters — each param's initial SymVal
        // depends on its side-table classification.
        let mut locals = Vec::new();
        for (i, ty) in param_types.iter().enumerate() {
            let slot = &side_table.params[i];
            let val = match slot.kind {
                ParamKind::BufferRead | ParamKind::BufferWrite => {
                    Some(SymVal::BufferPtr(slot.slot))
                }
                ParamKind::Scalar => {
                    // Push constants enter as Loads from slot N at
                    // index 0xFFFFFFFF — same convention the legacy
                    // parser uses (see emit_field_access).
                    None
                }
            };
            locals.push(LocalInfo {
                wasm_ty: *ty,
                val,
                stable_reg: None,
                stable_ty: slot.scalar,
            });
        }
        // Append declared locals (after params) — uninitialized.
        for (count, ty) in &body.locals {
            for _ in 0..*count {
                locals.push(LocalInfo {
                    wasm_ty: *ty,
                    val: None,
                    stable_reg: None,
                    stable_ty: scalar_type_for_wasm_ty(*ty),
                });
            }
        }

        // Build the intrinsic name table: indices 0..K of the
        // function namespace are imports; we record their names.
        let mut intrinsic_names = Vec::new();
        for f in &module.functions {
            if let FunctionKind::Imported { name, .. } = &f.kind {
                intrinsic_names.push(name.clone());
            }
        }

        Self {
            side_table,
            param_types,
            body,
            module,
            locals,
            stack: Vec::new(),
            frames: vec![Frame {
                kind: FrameKind::Function,
                ops: Vec::new(),
                local_snapshot: Vec::new(),
                brifs: Vec::new(),
                parent_brifs_at_open: 0,
            }],
            next_reg: 0,
            intrinsic_names,
            loop_reads: Vec::new(),
        }
    }

    fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Append an op to the current (topmost) frame.
    fn emit(&mut self, op: KernelOp) {
        let top = self.frames.len() - 1;
        self.frames[top].ops.push(op);
    }

    /// Append a sequence of ops to a specific frame. Used when an
    /// inner frame closes and its accumulated ops splice into the
    /// parent.
    fn splice_into_frame(target: &mut Frame, ops: impl IntoIterator<Item = KernelOp>) {
        target.ops.extend(ops);
    }

    /// Inspect a frame N levels above the current top (0 = current).
    fn frame_at_depth(&self, depth: u32) -> Option<&Frame> {
        let idx = self.frames.len().checked_sub(1 + depth as usize)?;
        self.frames.get(idx)
    }

    /// Mutable counterpart to `frame_at_depth`.
    fn frame_at_depth_mut(&mut self, depth: u32) -> Option<&mut Frame> {
        let idx = self.frames.len().checked_sub(1 + depth as usize)?;
        self.frames.get_mut(idx)
    }

    /// True if any frame strictly between the top and `depth` is a
    /// `KernelOp::Loop`. Used by `Br`/`BrIf` to decide whether the
    /// branch crosses a loop boundary — if so, a record-and-wrap on
    /// the target would reference a register defined inside the loop
    /// body from outside it. Falls back to a `Break` (or conditional
    /// Break) instead.
    fn has_loop_between_top_and_depth(&self, depth: u32) -> bool {
        self.loops_between_top_and_depth(depth) > 0
    }

    /// True if local `idx` is loop-carried at this point: it has an
    /// upward-exposed read in some open loop — i.e. it was read
    /// inside the loop body BEFORE this write, so on iteration N+1
    /// that read observes iteration N's write. Such a write must go
    /// through the stable register — a symbolic rebinding doesn't
    /// survive the iteration boundary, so the next iteration's
    /// reads (which pull from the stable register via
    /// `force_locals_to_stable`) would see a stale value. See the
    /// ScaledIdx arm in `LocalSet`/`LocalTee`.
    ///
    /// Iteration-local scratch (written before any in-loop read,
    /// like bench_nbody's hoisted `off = i << 2` addressing offset)
    /// keeps symbolic rebinding, which the buffer-addressing
    /// recognizer needs. Note "live-in at Loop entry" would be the
    /// wrong test: the function preamble zero-inits every declared
    /// local, so everything is trivially live-in.
    fn local_is_loop_carried(&self, idx: usize) -> bool {
        self.loop_reads
            .iter()
            .any(|reads| reads.contains(&(idx as u32)))
    }

    /// Number of Loop frames strictly between the top and `depth`.
    fn loops_between_top_and_depth(&self, depth: u32) -> usize {
        let top = self.frames.len();
        let target_idx = match top.checked_sub(1 + depth as usize) {
            Some(i) => i,
            None => return 0,
        };
        // Frames strictly above target, up to and including top-1.
        self.frames[target_idx + 1..]
            .iter()
            .filter(|f| matches!(f.kind, FrameKind::Loop { .. }))
            .count()
    }

    /// Materialize a boolean constant as a `KernelOp::Const` written
    /// into the frame at `depth`. Used by `Br` when no prior br_if
    /// cond exists to record with.
    fn emit_const_bool_to_target(&mut self, depth: u32, dst: Reg, value: bool) {
        let target = self
            .frame_at_depth_mut(depth)
            .expect("caller must verify target depth before emit_const_bool_to_target");
        target.ops.push(KernelOp::Const {
            dst,
            value: ConstValue::Bool(value),
        });
    }

    /// Record a br/br_if's intent on the target frame without
    /// eagerly installing a Branch op. Subsequent emits land at the
    /// target frame's natural sink position. End-of-frame walks the
    /// records in reverse and wraps
    /// `frame.ops[record.sink_position..]` in a real
    /// `Branch{cond, then=[], else=that-range}`.
    ///
    /// Targets: Block frames (Br / BrIf arms) and the Function frame
    /// (top-level Return records an always-true BrIf).
    fn record_br_at(&mut self, depth: u32, cond: Reg, is_unconditional: bool) {
        let target = self
            .frame_at_depth_mut(depth)
            .expect("caller must verify target depth before record_br_at");
        // Position = current end of target's natural ops.
        // Ops emitted AFTER this record live at positions >=
        // sink_position; they execute on the br_if's fall-through path
        // (or, for unconditional Br, are dead code under the source
        // semantics, which the wrap captures by giving them an
        // unreachable enclosing Branch).
        let sink_position = target.ops.len();
        target.brifs.push(BrIfRecord {
            sink_position,
            cond,
            is_unconditional,
        });
    }

    /// Bump the sink_position of every record on the CURRENT top
    /// frame from index `from` onward by `n`. Called when a child
    /// frame closes and contributes `n` ops to this frame: records
    /// made during the child's lifetime semantically fire after the
    /// child's content, so the wrap must start past it.
    fn bump_parent_brifs(&mut self, from: usize, n: usize) {
        let top = self.frames.len() - 1;
        for r in &mut self.frames[top].brifs[from..] {
            r.sink_position += n;
        }
    }

    /// At end-of-frame, wrap the recorded br_ifs into real
    /// `Branch{cond, then=[], else=ops[pos..]}` nests. Walks
    /// records in reverse-position order so each wrap nests inside
    /// the previous one's else_ops.
    ///
    /// Two br_ifs at positions p1 < p2 mean:
    /// - ops[p1..p2-1] execute on !cond1.
    /// - ops[p2..] execute on !cond1 AND !cond2.
    ///
    /// Processing in reverse: first wrap ops[p2..] inside
    /// `Branch{cond2, then=[], else=…}`, replacing target.ops[p2..]
    /// with that single new op. Then wrap ops[p1..] (which now
    /// includes the new inner Branch op as its last element).
    fn reconstruct_block_brifs(
        &self,
        mut ops: Vec<KernelOp>,
        brifs: &[BrIfRecord],
    ) -> Vec<KernelOp> {
        // BrIf wraps as `Branch{cond, then=[], else=tail}` — tail
        // runs only when cond=false (fall-through path).
        //
        // Br (unconditional) wraps as `Branch{cond=prior_brif_cond,
        // then=[tail], else=[]}` — tail runs only when the most
        // recent inner br_if fired (= we reached target's
        // continuation via that path). See Br's record_br_at site
        // for how the record's cond is the inner br_if's cond
        // (recorded as a pointer; we swap arms here to invert the
        // sense, avoiding the need to emit a separate !cond op).
        //
        // When no prior br_if existed at record time, the record
        // cond was set to a const_true: with swapped arms,
        // `then=[tail]` runs unconditionally, which matches strict
        // WASM semantics for an unconditional br that has no
        // earlier br_if to "rescue" the tail.
        for record in brifs.iter().rev() {
            let tail = ops.split_off(record.sink_position);
            if record.is_unconditional {
                ops.push(KernelOp::Branch {
                    cond: record.cond,
                    then_ops: tail,
                    else_ops: Vec::new(),
                });
            } else {
                ops.push(KernelOp::Branch {
                    cond: record.cond,
                    then_ops: Vec::new(),
                    else_ops: tail,
                });
            }
        }
        ops
    }

    /// Materialize the br_if cond into a fresh
    /// register that is (a) zero-init declared in the TARGET frame's
    /// sink before any recorded wrap position, and (b) assigned via
    /// `Copy` at the SOURCE position (current sink, exactly where
    /// wasm evaluated the cond). The end-of-frame wrap then
    /// references the materialized reg, which is structurally in
    /// scope at the wrap and carries the value computed at the
    /// correct point of the iteration.
    ///
    /// This replaces the transitive backward-slice hoist
    /// (`hoist_cond_transitive_v2`, removed 2026-06-12). Hoisting
    /// relocated the cond's whole pure-value slice to the target
    /// frame top, which had two unsoundness modes:
    ///
    /// 1. Slice inputs whose real defining ops lived in INTERMEDIATE
    ///    frames (LLVM sinks computations into inner blocks) were
    ///    invisible to the single-frame backward walk; the
    ///    function-preamble zero-init `Const` declarations satisfied
    ///    scope_check, so the hoisted chain silently read zeros /
    ///    stale values. This was the PTRD distribution-skew bug:
    ///    `ln(v)` ran on v=0.0 from the zero-init, lhs became -inf,
    ///    and the slow-accept fired unconditionally on iteration 0.
    /// 2. The hoisted chain executed unconditionally at the target,
    ///    even on wasm paths that never evaluated the cond.
    ///
    /// Materialization moves no computation, so neither mode exists:
    /// the cond chain stays at its source position, guarded by
    /// exactly the control flow wasm gave it, and only the RESULT
    /// flows through the pre-declared register. If the source path
    /// is skipped by an earlier-firing wrap, the register holds its
    /// zero-init (false) — and the wrap whose tail would read it is
    /// itself inside the skipped region, so the value is never
    /// observed.
    ///
    /// At depth 0 the cond's def already precedes the record
    /// position in the same sink — the wrap references it directly.
    fn materialize_cond_for_v2(&mut self, depth: u32, cond: Reg, ty: ScalarType) -> Reg {
        if depth == 0 {
            return cond;
        }
        let mat = self.alloc_reg();
        let zero = match ty {
            ScalarType::F16 => ConstValue::F16(0),
            // bf16/fp8 zero materializes as an f32 zero (emulated body).
            ScalarType::BF16 | ScalarType::FP8E5M2 | ScalarType::FP8E4M3 => ConstValue::F32(0.0),
            ScalarType::F32 => ConstValue::F32(0.0),
            ScalarType::F64 => ConstValue::F64(0.0),
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => ConstValue::U32(0),
            ScalarType::U64 => ConstValue::U64(0),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I4 => {
                ConstValue::I32(0)
            }
            ScalarType::I64 => ConstValue::I64(0),
            ScalarType::Bool => ConstValue::Bool(false),
        };
        // Declare at the target frame, before the earliest recorded
        // wrap position, so every wrap's tail nests inside a region
        // where the declaration is visible.
        self.insert_decl_at_target(
            depth,
            KernelOp::Const {
                dst: mat,
                value: zero,
            },
        );
        // Assign at the source position. Children closing into the
        // target bump recorded wrap positions past their content
        // (parent_brifs_at_open), so this Copy always executes
        // before the wrap that reads `mat`.
        self.emit(KernelOp::Copy {
            dst: mat,
            src: cond,
            ty,
        });
        mat
    }

    /// Insert a declaration op into the frame at `depth`, before the
    /// earliest recorded wrap position (so every wrap's tail nests
    /// inside a region where the declaration is visible), bumping
    /// existing records past it.
    fn insert_decl_at_target(&mut self, depth: u32, op: KernelOp) {
        let target = self
            .frame_at_depth_mut(depth)
            .expect("caller must verify target depth before insert_decl_at_target");
        let insert_pos = target
            .brifs
            .iter()
            .map(|r| r.sink_position)
            .min()
            .unwrap_or(target.ops.len());
        target.ops.insert(insert_pos, op);
        for r in &mut target.brifs {
            r.sink_position += 1;
        }
    }

    /// br/br_if from inside a loop to a Block target OUTSIDE it.
    ///
    /// A label-less `Break` alone exits the loop but loses WHICH
    /// label the wasm br targeted: any continuation ops between the
    /// `Loop` op and the target's end would run even on the br path.
    /// That was the PTRD early-exit bug — LLVM merges `iter` and
    /// `result` into one local and encodes the two loop exits as brs
    /// to different labels (accept: `local = k; br $outer`, skipping
    /// the `local = 0` exhaustion continuation at $inner's end). The
    /// plain Break ran the zeroing on both paths and every quark
    /// produced 0.
    ///
    /// Encode the label in an exit flag instead:
    ///   - `flag = false` declared at the target frame (re-runs on
    ///     each entry to the target block),
    ///   - `flag = true; Break` at the br site (wrapped in
    ///     `Branch{cond}` for br_if),
    ///   - a recorded br_if{cond: flag} on the target wraps the
    ///     continuation in `Branch{flag, then=[], else=[…]}` at the
    ///     target's End, so it is skipped exactly when this exit
    ///     fired. Frame-close position bumps (Block splice / the +1
    ///     for Loop/If/Else single-op closes) keep the record
    ///     pointing just past the Loop op.
    ///
    /// Caller guarantees: exactly ONE Loop frame between top and
    /// `depth` (a single `Break` exits exactly one loop level), and
    /// the target is a Block.
    fn emit_loop_crossing_exit(&mut self, depth: u32, cond: Option<Reg>) {
        let flag = self.alloc_reg();
        self.insert_decl_at_target(
            depth,
            KernelOp::Const {
                dst: flag,
                value: ConstValue::Bool(false),
            },
        );
        let set_and_break = vec![
            KernelOp::Const {
                dst: flag,
                value: ConstValue::Bool(true),
            },
            KernelOp::Break,
        ];
        match cond {
            Some(c) => self.emit(KernelOp::Branch {
                cond: c,
                then_ops: set_and_break,
                else_ops: Vec::new(),
            }),
            None => {
                for op in set_and_break {
                    self.emit(op);
                }
            }
        }
        // Record on EVERY frame outside the loop up to the target,
        // not just the target: when the exit fires, the Break resumes
        // just after the Loop op, and any live tail in an intermediate
        // block between the loop and the target must be skipped too.
        // (Frames inside the loop are exited by the Break itself.)
        let loop_depth = (0..depth)
            .find(|&d| {
                self.frame_at_depth(d)
                    .is_some_and(|f| matches!(f.kind, FrameKind::Loop { .. }))
            })
            .expect("caller guarantees a Loop frame between top and depth");
        for d in (loop_depth + 1)..=depth {
            self.record_br_at(d, flag, /*is_unconditional=*/ false);
        }
    }

    fn lower(mut self) -> Result<KernelDef, LoweringError> {
        // Initialise scalar push-constant locals: emit one Load from
        // the constant slot at function entry, seed the local with
        // its register. SPIR-V/MSL emitters dispatch push-const reads
        // on `index.0 == u32::MAX` (see emit_spirv/ops.rs:113,
        // emit_msl/ops.rs:43) — the sentinel must be threaded through
        // verbatim, not replaced with a real zero-const register.
        for (i, slot) in self.side_table.params.iter().enumerate() {
            if matches!(slot.kind, ParamKind::Scalar) {
                let dst = self.alloc_reg();
                self.emit(KernelOp::Load {
                    dst,
                    field: slot.slot,
                    index: Reg(u32::MAX),
                    ty: slot.scalar,
                });
                self.locals[i].val = Some(SymVal::Reg(dst, slot.scalar));
                // Scalar params don't need a separate stable_reg: the
                // load above produces a single reg that's already
                // function-entry-defined.
                self.locals[i].stable_reg = Some(dst);
            }
        }

        // Pre-allocate stable registers for every value-typed declared
        // local (after params), with a default-zero initializer.
        // Allocating unconditionally at function entry means every
        // subsequent `local.get` reads a register that was defined
        // before any control-flow split — fixes the SSA-join issue
        // when a local is written along multiple paths and read after
        // the merge.
        let param_count = self.side_table.params.len();
        for i in param_count..self.locals.len() {
            let ty = self.locals[i].stable_ty;
            let dst = self.alloc_reg();
            let init = match ty {
                ScalarType::F16
                | ScalarType::BF16
                | ScalarType::FP8E5M2
                | ScalarType::FP8E4M3
                | ScalarType::F32 => ConstValue::F32(0.0),
                ScalarType::F64 => ConstValue::F64(0.0),
                ScalarType::Bool => ConstValue::Bool(false),
                ScalarType::I8
                | ScalarType::I16
                | ScalarType::I32
                | ScalarType::I64
                | ScalarType::I4 => ConstValue::I32(0),
                ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 => {
                    ConstValue::U32(0)
                }
            };
            self.emit(KernelOp::Const { dst, value: init });
            self.locals[i].stable_reg = Some(dst);
            self.locals[i].val = Some(SymVal::Reg(dst, ty));
        }

        let instrs = self.body.instructions.clone();
        for instr in &instrs {
            self.lower_instr(instr)?;
        }

        Ok(self.into_kernel_def())
    }

    /// Lazily allocate a stable register for a local that didn't get
    /// one at function entry. rustc's optimizer freely reuses local
    /// slots — most notably, a buffer-pointer param's slot can be
    /// recycled as an integer counter once the pointer is no longer
    /// live. The function-entry pass leaves `stable_reg = None` for
    /// buffer-pointer params (they're symbolic, no value-reg yet);
    /// when a `local.set`/`local.tee` later writes a value-typed
    /// SymVal there, we need a stable reg or every read is a
    /// degenerate const.
    fn ensure_stable_reg_for(&mut self, idx: usize, v: &SymVal) {
        if self.locals[idx].stable_reg.is_some() {
            return;
        }
        // Pick the scalar type from the value being stored. For
        // Reg/Opaque carry it; for I32Const default to U32 (matching
        // the function-entry default for declared locals).
        let ty = match v {
            SymVal::Reg(_, ty) | SymVal::Opaque(_, ty) => *ty,
            SymVal::I32Const(_) => ScalarType::U32,
            SymVal::I64Const(_) => ScalarType::U64,
            _ => ScalarType::U32,
        };
        let dst = self.alloc_reg();
        let init = match ty {
            ScalarType::F16
            | ScalarType::BF16
            | ScalarType::FP8E5M2
            | ScalarType::FP8E4M3
            | ScalarType::F32 => ConstValue::F32(0.0),
            ScalarType::F64 => ConstValue::F64(0.0),
            ScalarType::Bool => ConstValue::Bool(false),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I4 => {
                ConstValue::I32(0)
            }
            ScalarType::I64 => ConstValue::I64(0),
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => ConstValue::U32(0),
            ScalarType::U64 => ConstValue::U64(0),
        };
        // Emit the init const at the function-frame level (not the
        // current frame), so the stable reg is defined before any
        // control-flow that might run before this set. We use the
        // function frame's main `ops` directly to avoid the redirect
        // chain on the function frame (the init must be the very
        // first thing that runs).
        let function_frame = &mut self.frames[0];
        function_frame
            .ops
            .insert(0, KernelOp::Const { dst, value: init });
        self.locals[idx].stable_reg = Some(dst);
        self.locals[idx].stable_ty = ty;
    }

    /// Snapshot every local's current `val` for use at frame
    /// close to detect which locals were modified inside the
    /// frame. Cheap — just clones the (idx, val) pairs for
    /// locals that have a binding.
    fn snapshot_locals(&self) -> Vec<(u32, Option<SymVal>)> {
        self.locals
            .iter()
            .enumerate()
            .map(|(idx, info)| (idx as u32, info.val))
            .collect()
    }

    /// Reset locals to the values captured by `snapshot_locals`.
    /// Used at `Else` entry so the else-branch starts from the
    /// same pre-If state the then-branch did.
    fn restore_locals_from_snapshot(&mut self, snapshot: &[(u32, Option<SymVal>)]) {
        for (idx, val) in snapshot {
            if let Some(info) = self.locals.get_mut(*idx as usize) {
                info.val = *val;
            }
        }
    }

    /// Force every local with a stable_reg to read from
    /// stable_reg instead of any fresh-reg binding. Called at
    /// Loop entry so accumulators work across iterations — see
    /// the Loop case in `lower_instr` for the full rationale.
    fn force_locals_to_stable(&mut self) {
        for info in self.locals.iter_mut() {
            if let Some(stable_reg) = info.stable_reg {
                // Only rebind value-typed locals. Buffer-access
                // and other non-value SymVals are consumed by
                // load/store pattern recognition; replacing them
                // with a Reg would corrupt their semantics.
                if matches!(info.val, Some(ref v) if is_value_symval(v)) {
                    info.val = Some(SymVal::Reg(stable_reg, info.stable_ty));
                }
            }
        }
    }

    /// After a scope-introducing frame (If, Else, Loop) closes,
    /// reset the `val` of every local that was modified inside
    /// the frame so post-frame reads use the stable_reg (which
    /// was kept in sync via `write_local_via_copy`'s parallel
    /// Copy). Without this, post-frame `local.get` references a
    /// register that was allocated inside the now-closed scope
    /// (e.g. inside `Branch.else_ops`) and the downstream
    /// emitters produce "register used before definition" errors
    /// or, worse, silently read garbage.
    fn merge_locals_post_frame(&mut self, snapshot: &[(u32, Option<SymVal>)]) {
        for (idx, snap_val) in snapshot {
            let info = match self.locals.get_mut(*idx as usize) {
                Some(i) => i,
                None => continue,
            };
            // Only locals with a stable_reg participate in the
            // merge — buffer-pointer params have no value-reg and
            // their symbolic bindings aren't subject to this
            // aliasing.
            let stable_reg = match info.stable_reg {
                Some(r) => r,
                None => continue,
            };
            // If the current val matches the snapshot, nothing
            // changed inside the frame; skip.
            if info.val == *snap_val {
                continue;
            }
            // Only rebind value-typed bindings. Non-value
            // SymVals (BufferAccess, BufferPtr, ScaledIdx) are
            // consumed by load/store recognizers; replacing
            // them with a Reg corrupts the pattern match.
            let is_value_now = matches!(info.val, Some(ref v) if is_value_symval(v));
            if !is_value_now {
                continue;
            }
            // Reset val to stable_reg. The stable_reg was kept
            // in sync by `write_local_via_copy` on every set
            // inside the frame.
            info.val = Some(SymVal::Reg(stable_reg, info.stable_ty));
        }
    }

    /// Materialize a `local.set` / `local.tee`. Allocates a
    /// **fresh** register per call so successive sets of the same
    /// local don't alias.
    ///
    /// Why fresh-per-set: rustc's wasm codegen sometimes recycles
    /// a wasm-local across SSA-disjoint Rust values (live-range
    /// coalescing in the LLVM backend). If we map every set of
    /// the same local to one stable_reg, the second set clobbers
    /// the first's reads. Fresh-per-set produces SSA-shaped
    /// register output: every set introduces a new dst, every
    /// get reads the latest binding.
    ///
    /// The stable_reg is still allocated at function entry and
    /// kept in sync via a parallel Copy. This is the merge anchor
    /// — code that crosses a control-flow join (e.g. reads after
    /// an `if` branch that updated the local) falls back to
    /// stable_reg via the frame-close logic in `lower_instr`'s
    /// End handler. For straight-line code, the stable_reg copy
    /// is a small redundant cost.
    ///
    /// See `quanta_project/roadmap/_design/wasm_local_renaming.md`
    /// for the full design.
    fn write_local_via_copy(
        &mut self,
        idx: usize,
        v: SymVal,
    ) -> Result<(Reg, ScalarType), LoweringError> {
        let (src, _src_ty) = self.commit(v)?;
        let stable_reg = self.locals[idx].stable_reg.ok_or_else(|| {
            LoweringError::ShapeMismatch(format!(
                "local {idx} has no stable register — buffer-pointer params can't be set"
            ))
        })?;
        let stable_ty = self.locals[idx].stable_ty;

        // Allocate a fresh per-set register. This becomes the
        // post-set binding for the local. The old behaviour
        // (always writing into stable_reg) caused successive sets
        // to alias — see method docstring.
        let fresh = self.alloc_reg();
        // Pre-declare the fresh reg at the function frame so the
        // downstream Metal/WGSL emitters generate the `uint rN =
        // 0u;` declaration that the later `KernelOp::Copy` (which
        // emits `rN = rM;`, assignment-only) depends on. Mirrors
        // the `ensure_stable_reg_for` pattern for stable regs.
        let init = match stable_ty {
            ScalarType::F16
            | ScalarType::BF16
            | ScalarType::FP8E5M2
            | ScalarType::FP8E4M3
            | ScalarType::F32 => ConstValue::F32(0.0),
            ScalarType::F64 => ConstValue::F64(0.0),
            ScalarType::Bool => ConstValue::Bool(false),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I4 => {
                ConstValue::I32(0)
            }
            ScalarType::I64 => ConstValue::I64(0),
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => ConstValue::U32(0),
            ScalarType::U64 => ConstValue::U64(0),
        };
        let function_frame = &mut self.frames[0];
        function_frame.ops.insert(
            0,
            KernelOp::Const {
                dst: fresh,
                value: init,
            },
        );
        self.emit(KernelOp::Copy {
            dst: fresh,
            src,
            ty: stable_ty,
        });
        // Also keep stable_reg in sync — it remains the merge
        // anchor used by post-frame-close reads. The two copies
        // are emitted back-to-back so the IR is still flat.
        self.emit(KernelOp::Copy {
            dst: stable_reg,
            src: fresh,
            ty: stable_ty,
        });

        // Reads of this local now see the fresh register.
        // post-merge code (after a branch closes) will see
        // stable_reg via the frame-close fixup; that fixup
        // re-points locals[idx].val to stable_reg.
        self.locals[idx].val = Some(SymVal::Reg(fresh, stable_ty));
        Ok((fresh, stable_ty))
    }

    fn lower_instr(&mut self, instr: &RawInstr) -> Result<(), LoweringError> {
        match instr {
            RawInstr::LocalGet(idx) => {
                let val = self.locals[*idx as usize].val.ok_or_else(|| {
                    LoweringError::ShapeMismatch(format!("local.get {idx} on uninitialized local"))
                })?;
                // Record the read on every open loop — see
                // `local_is_loop_carried`.
                for reads in &mut self.loop_reads {
                    reads.insert(*idx);
                }
                self.stack.push(val);
            }
            RawInstr::LocalSet(idx) => {
                let v = self.pop()?;
                // Route value-typed SymVals (Reg/Opaque/Const) through
                // the stable register so post-merge reads see a defined
                // value. Buffer/address SymVals (BufferPtr/
                // BufferAccess) carry no scalar value to copy and keep
                // their existing symbolic binding — they're consumed
                // by load/store pattern recognition, not arithmetic.
                //
                // ScaledIdx is the exception: it IS a scalar value
                // (`base << k`, usually a byte offset), kept symbolic
                // only so the buffer-addressing recognizer can see it.
                // A symbolic rebinding is sound for iteration-local
                // scratch — the next local.get re-pushes it — but
                // UNSOUND for a loop-carried local: iteration N+1
                // reads the local through its stable register (see
                // `force_locals_to_stable` at Loop entry), which a
                // symbolic rebinding never writes. LLVM hits this by
                // strength-reducing a loop induction update `d *= 2`
                // to `d = d << 1` (lowering bug variant #5, found
                // 2026-06-12 by the segmented-reduce kernel — the
                // update vanished and the loop ran to its 10000-iter
                // fuel cap). Materialize through the stable register
                // when the local is live-in to an open loop;
                // loop-internal scratch (e.g. bench_nbody's hoisted
                // `off = i << 2` addressing offset) stays symbolic.
                let v = if matches!(v, SymVal::ScaledIdx { .. })
                    && self.local_is_loop_carried(*idx as usize)
                {
                    let (r, ty) = self.commit(v)?;
                    SymVal::Reg(r, ty)
                } else {
                    v
                };
                if is_value_symval(&v) {
                    self.ensure_stable_reg_for(*idx as usize, &v);
                    self.write_local_via_copy(*idx as usize, v)?;
                } else {
                    self.locals[*idx as usize].val = Some(v);
                }
            }
            RawInstr::LocalTee(idx) => {
                let v = self.stack.last().copied().ok_or_else(|| {
                    LoweringError::ShapeMismatch("local.tee on empty stack".into())
                })?;
                // Same ScaledIdx loop-carried materialization as
                // LocalSet above — tee is set + keep-on-stack.
                let v = if matches!(v, SymVal::ScaledIdx { .. })
                    && self.local_is_loop_carried(*idx as usize)
                {
                    let _ = self.pop()?;
                    let (r, ty) = self.commit(v)?;
                    let v = SymVal::Reg(r, ty);
                    self.stack.push(v);
                    v
                } else {
                    v
                };
                if is_value_symval(&v) {
                    let _ = self.pop()?;
                    self.ensure_stable_reg_for(*idx as usize, &v);
                    let (reg, ty) = self.write_local_via_copy(*idx as usize, v)?;
                    // tee leaves the value on stack — the post-tee
                    // value is the stable reg's contents.
                    self.stack.push(SymVal::Reg(reg, ty));
                } else {
                    self.locals[*idx as usize].val = Some(v);
                }
            }
            RawInstr::I32Const(v) => self.stack.push(SymVal::I32Const(*v)),
            RawInstr::F32Const(v) => {
                let dst = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst,
                    value: ConstValue::F32(*v),
                });
                self.stack.push(SymVal::Reg(dst, ScalarType::F32));
            }
            RawInstr::F64Const(v) => {
                let dst = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst,
                    value: ConstValue::F64(*v),
                });
                self.stack.push(SymVal::Reg(dst, ScalarType::F64));
            }

            RawInstr::I32Shl => {
                // Pattern: <reg> <const k> i32.shl → ScaledIdx{base, 1<<k}
                let b = self.pop()?;
                let a = self.pop()?;
                if let (SymVal::Reg(base, _) | SymVal::Opaque(base, _), SymVal::I32Const(k)) =
                    (a, b)
                {
                    let scale = 1u32 << k;
                    self.stack.push(SymVal::ScaledIdx { base, scale });
                } else {
                    // Fall back to a plain shift.
                    let (br, _) = self.commit(a)?;
                    let (kr, _) = self.commit(b)?;
                    let dst = self.alloc_reg();
                    self.emit(KernelOp::BinOp {
                        dst,
                        a: br,
                        b: kr,
                        op: BinOp::Shl,
                        ty: ScalarType::I32,
                    });
                    self.stack.push(SymVal::Reg(dst, ScalarType::I32));
                }
            }

            RawInstr::I32Add => {
                // Pattern: <BufferPtr(slot)> <ScaledIdx{base,scale}>
                // i32.add → keep as a "load address" tag we'll consume
                // when an f32.load/store comes next. We model this by
                // pushing a synthetic "address" SymVal — encoded as a
                // pair via the stack: BufferPtr + ScaledIdx adjacent.
                // But since we can't push two values, we leave them
                // on the stack and let the load/store ops pop both.
                let b = self.pop()?;
                let a = self.pop()?;
                match (a, b) {
                    (SymVal::BufferPtr(slot), SymVal::ScaledIdx { base, scale })
                    | (SymVal::ScaledIdx { base, scale }, SymVal::BufferPtr(slot)) => {
                        // Push as a "buffer access" sentinel — encoded
                        // by pushing the pieces back so the next op
                        // (load/store) recognizes them. We use a
                        // synthetic SymVal for this:
                        self.stack.push(SymVal::BufferAccess { slot, base, scale });
                    }
                    // Chained address arithmetic: rustc may emit
                    // `out + block_offset + pos_offset` as nested
                    // `BufferAccess + ScaledIdx` adds when it has
                    // precomputed part of the byte-offset and the
                    // rest is runtime. Compose the indices.
                    //
                    // Same scale → indices add directly.
                    // Larger BufferAccess scale → rescale its
                    // index to match the smaller ScaledIdx
                    // scale: new_base = ba_base * (ba_scale /
                    // si_scale) + si_base, scale = si_scale.
                    (
                        SymVal::BufferAccess { slot, base, scale },
                        SymVal::ScaledIdx {
                            base: b2,
                            scale: s2,
                        },
                    )
                    | (
                        SymVal::ScaledIdx {
                            base: b2,
                            scale: s2,
                        },
                        SymVal::BufferAccess { slot, base, scale },
                    ) if scale == s2
                        || (scale > s2 && scale % s2 == 0 && (scale / s2).is_power_of_two()) =>
                    {
                        let (final_scale, final_base) = if scale == s2 {
                            // Same-scale add: combine indices.
                            let dst = self.alloc_reg();
                            self.emit(KernelOp::BinOp {
                                dst,
                                a: base,
                                b: b2,
                                op: BinOp::Add,
                                ty: ScalarType::U32,
                            });
                            (scale, dst)
                        } else {
                            // Rescale BufferAccess's base to match
                            // the smaller scale: shift base left
                            // by log2(scale / s2), then add.
                            let shift_amt = (scale / s2).trailing_zeros();
                            let shift_reg = self.alloc_reg();
                            self.emit(KernelOp::Const {
                                dst: shift_reg,
                                value: ConstValue::U32(shift_amt),
                            });
                            let scaled_base = self.alloc_reg();
                            self.emit(KernelOp::BinOp {
                                dst: scaled_base,
                                a: base,
                                b: shift_reg,
                                op: BinOp::Shl,
                                ty: ScalarType::U32,
                            });
                            let dst = self.alloc_reg();
                            self.emit(KernelOp::BinOp {
                                dst,
                                a: scaled_base,
                                b: b2,
                                op: BinOp::Add,
                                ty: ScalarType::U32,
                            });
                            (s2, dst)
                        };
                        self.stack.push(SymVal::BufferAccess {
                            slot,
                            base: final_base,
                            scale: final_scale,
                        });
                    }
                    // BufferAccess + Const offset (negative for
                    // `arr[N] = ...; arr[N-1] = ...` patterns where
                    // rustc folds the -1 into a constant add).
                    (SymVal::BufferAccess { slot, base, scale }, SymVal::I32Const(c))
                    | (SymVal::I32Const(c), SymVal::BufferAccess { slot, base, scale })
                        if (c as i64) % (scale as i64) == 0 =>
                    {
                        let off = c / (scale as i32);
                        let off_reg = self.alloc_reg();
                        self.emit(KernelOp::Const {
                            dst: off_reg,
                            value: ConstValue::I32(off),
                        });
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::BinOp {
                            dst,
                            a: base,
                            b: off_reg,
                            op: BinOp::Add,
                            ty: ScalarType::U32,
                        });
                        self.stack.push(SymVal::BufferAccess {
                            slot,
                            base: dst,
                            scale,
                        });
                    }
                    (a, b) => {
                        let (ar, ty_a) = self.commit(a)?;
                        let (br, ty_b) = self.commit(b)?;
                        let ty = if ty_a == ty_b { ty_a } else { ScalarType::I32 };
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::BinOp {
                            dst,
                            a: ar,
                            b: br,
                            op: BinOp::Add,
                            ty,
                        });
                        self.stack.push(SymVal::Reg(dst, ty));
                    }
                }
            }

            RawInstr::F32Load { offset, .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } if *offset == 0 => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index: base,
                            ty: ScalarType::F32,
                        });
                        self.stack.push(SymVal::Reg(dst, ScalarType::F32));
                    }
                    // Bare BufferPtr — `buf[N]` for compile-time-
                    // constant N. rustc folds N into the memarg
                    // offset (`offset = N * 4` for f32).
                    SymVal::BufferPtr(slot) if (*offset % 4) == 0 => {
                        let index = self.const_index_for_offset(*offset, 4);
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index,
                            ty: ScalarType::F32,
                        });
                        self.stack.push(SymVal::Reg(dst, ScalarType::F32));
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "f32.load on non-buffer address {other:?} (offset={offset})"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::F32Store { offset, .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                let (val_reg, _) = self.commit(val)?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } if *offset == 0 => {
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty: ScalarType::F32,
                        });
                    }
                    // Scaled-pair pattern: rustc folds writes of
                    // `out[id*2]` / `out[id*2+1]` into one address
                    // base `out + id*8` plus immediate offsets 0 and
                    // 4. Reconstruct the element index as
                    // `base * 2 + offset / sizeof(f32)`.
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 8,
                    } if (*offset % 4) == 0 => {
                        let two = self.alloc_reg();
                        self.emit(KernelOp::Const {
                            dst: two,
                            value: ConstValue::U32(2),
                        });
                        let scaled = self.alloc_reg();
                        self.emit(KernelOp::BinOp {
                            dst: scaled,
                            a: base,
                            b: two,
                            op: BinOp::Mul,
                            ty: ScalarType::U32,
                        });
                        let index = if *offset == 0 {
                            scaled
                        } else {
                            let off_reg = self.alloc_reg();
                            self.emit(KernelOp::Const {
                                dst: off_reg,
                                value: ConstValue::U32((*offset / 4) as u32),
                            });
                            let idx = self.alloc_reg();
                            self.emit(KernelOp::BinOp {
                                dst: idx,
                                a: scaled,
                                b: off_reg,
                                op: BinOp::Add,
                                ty: ScalarType::U32,
                            });
                            idx
                        };
                        self.emit(KernelOp::Store {
                            field: slot,
                            index,
                            src: val_reg,
                            ty: ScalarType::F32,
                        });
                    }
                    // Bare BufferPtr — `buf[N] = …` for compile-time-
                    // constant N. rustc folds N into the memarg
                    // offset (`offset = N * 4`).
                    SymVal::BufferPtr(slot) if (*offset % 4) == 0 => {
                        let index = self.const_index_for_offset(*offset, 4);
                        self.emit(KernelOp::Store {
                            field: slot,
                            index,
                            src: val_reg,
                            ty: ScalarType::F32,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "f32.store on non-buffer address {other:?} (offset={offset})"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::F64Load { offset, .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 8,
                    } => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index: base,
                            ty: ScalarType::F64,
                        });
                        self.stack.push(SymVal::Reg(dst, ScalarType::F64));
                    }
                    // Bare BufferPtr — `buf[N]` for compile-time-
                    // constant N (f64 ⇒ offset = N * 8).
                    SymVal::BufferPtr(slot) if (*offset % 8) == 0 => {
                        let index = self.const_index_for_offset(*offset, 8);
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index,
                            ty: ScalarType::F64,
                        });
                        self.stack.push(SymVal::Reg(dst, ScalarType::F64));
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("f64.load on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::F64Store { offset, .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                let (val_reg, _) = self.commit(val)?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 8,
                    } if *offset == 0 => {
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty: ScalarType::F64,
                        });
                    }
                    // Bare BufferPtr — `buf[N] = …` for compile-time-
                    // constant N. rustc folds N into the memarg
                    // offset (`offset = N * 8` for f64).
                    SymVal::BufferPtr(slot) if (*offset % 8) == 0 => {
                        let index = self.const_index_for_offset(*offset, 8);
                        self.emit(KernelOp::Store {
                            field: slot,
                            index,
                            src: val_reg,
                            ty: ScalarType::F64,
                        });
                    }
                    // Scaled-pair pattern: rustc folds writes of
                    // `out[id*2]` / `out[id*2+1]` for an f64 buffer
                    // into one byte-base `out + id*16` plus immediate
                    // offsets 0 and 8. Reconstruct the element index
                    // as `base * 2 + offset / sizeof(f64)`.
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 16,
                    } if (*offset % 8) == 0 => {
                        let two = self.alloc_reg();
                        self.emit(KernelOp::Const {
                            dst: two,
                            value: ConstValue::U32(2),
                        });
                        let scaled = self.alloc_reg();
                        self.emit(KernelOp::BinOp {
                            dst: scaled,
                            a: base,
                            b: two,
                            op: BinOp::Mul,
                            ty: ScalarType::U32,
                        });
                        let index = if *offset == 0 {
                            scaled
                        } else {
                            let off_reg = self.alloc_reg();
                            self.emit(KernelOp::Const {
                                dst: off_reg,
                                value: ConstValue::U32((*offset / 8) as u32),
                            });
                            let idx = self.alloc_reg();
                            self.emit(KernelOp::BinOp {
                                dst: idx,
                                a: scaled,
                                b: off_reg,
                                op: BinOp::Add,
                                ty: ScalarType::U32,
                            });
                            idx
                        };
                        self.emit(KernelOp::Store {
                            field: slot,
                            index,
                            src: val_reg,
                            ty: ScalarType::F64,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "f64.store on non-buffer address {other:?} (offset={offset})"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::I32Load { offset, .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } if *offset == 0 => {
                        let ty = self.scalar_type_for_slot(slot);
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index: base,
                            ty,
                        });
                        self.stack.push(SymVal::Reg(dst, ty));
                    }
                    // Bare BufferPtr — `buf[N]` for compile-time-
                    // constant N. rustc folds the index into the
                    // memarg offset (`offset = N * 4`), so the stack
                    // arg is just the buffer base pointer.
                    SymVal::BufferPtr(slot) if (*offset % 4) == 0 => {
                        let ty = self.scalar_type_for_slot(slot);
                        let index = self.const_index_for_offset(*offset, 4);
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index,
                            ty,
                        });
                        self.stack.push(SymVal::Reg(dst, ty));
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "i32.load on non-buffer address {other:?} (offset={offset})"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::I32Store { offset, .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } if *offset == 0 => {
                        let ty = self.scalar_type_for_slot(slot);
                        let val_reg = self.materialize_for_typed_store(val, ty)?;
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty,
                        });
                    }
                    // Bare BufferPtr — `buf[N] = …` for compile-time-
                    // constant N. rustc folds the index into the
                    // memarg offset (`offset = N * 4`).
                    SymVal::BufferPtr(slot) if (*offset % 4) == 0 => {
                        let ty = self.scalar_type_for_slot(slot);
                        let val_reg = self.materialize_for_typed_store(val, ty)?;
                        let index = self.const_index_for_offset(*offset, 4);
                        self.emit(KernelOp::Store {
                            field: slot,
                            index,
                            src: val_reg,
                            ty,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "i32.store on non-buffer address {other:?} (offset={offset})"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            // Narrow loads (`i32.load8_u`, `i32.load8_s`) — rustc's
            // optimizer narrows `buf[i] & 0xFF` (where `buf` is a u32
            // buffer) to a byte-wide load to save WASM bytes. Since
            // Quanta IR has no byte-addressed loads, we lower these
            // as a regular element load + a synthetic mask: u32 load
            // + And(0xFF) for unsigned, sign-extend (`(x << 24) >> 24`)
            // for signed. Only the bottom byte is meaningful in either
            // case, mirroring WASM's semantics.
            RawInstr::I32Load8U { .. } => self.narrow_load(8, false)?,
            RawInstr::I32Load8S { .. } => self.narrow_load(8, true)?,

            // Narrow stores (`i32.store8`) — only the bottom byte of
            // the value is written. We emit a regular Store after
            // masking the source register to its low byte. Note: this
            // only matches the user's intent when the slot really is
            // a u32 buffer being updated with byte-granular values.
            RawInstr::I32Store8 { .. } => self.narrow_store(8)?,

            // ── i64 memory ops ──────────────────────────────────────
            // Wide i64 load/store — used for `&[u64]` buffer access
            // where rustc emits a single 8-byte memory op. Mirrors
            // the i32.load / i32.store arms but matches scale=8
            // (the rustc-emitted stride for u64 element arrays).
            RawInstr::I64Load { .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 8,
                    } => {
                        let ty = self.scalar_type_for_slot(slot);
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index: base,
                            ty,
                        });
                        self.stack.push(SymVal::Reg(dst, ty));
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("i64.load on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::I64Store { .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 8,
                    } => {
                        let ty = self.scalar_type_for_slot(slot);
                        let val_reg = self.materialize_for_typed_store(val, ty)?;
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("i64.store on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            // Narrow i64 load (`i64.load32_u` / `_s`) — load 4 bytes
            // from a u32-typed buffer slot and zero/sign-extend the
            // result to u64. rustc emits this when a kernel reads
            // `buf[i] as u64` where `buf: &[u32]`. We lower as
            // Load(ty=slot_ty) followed by Cast(slot_ty, U64/I64).
            RawInstr::I64Load32U { .. } => self.narrow_load_widen_i64(32, false)?,
            RawInstr::I64Load32S { .. } => self.narrow_load_widen_i64(32, true)?,
            RawInstr::I64Load16U { .. } => self.narrow_load_widen_i64(16, false)?,
            RawInstr::I64Load16S { .. } => self.narrow_load_widen_i64(16, true)?,
            RawInstr::I64Load8U { .. } => self.narrow_load_widen_i64(8, false)?,
            RawInstr::I64Load8S { .. } => self.narrow_load_widen_i64(8, true)?,

            // Narrow i64 store (`i64.store32`) — write the low 32
            // bits of a u64 value to a u32-typed buffer slot. rustc
            // emits this when a kernel writes `buf[i] = wide as u32`
            // (and the wrap-then-store gets fused). We lower as
            // Cast(U64, U32) followed by Store(ty=slot_ty).
            RawInstr::I64Store32 { .. } => self.narrow_store_truncate_i64(32)?,
            RawInstr::I64Store16 { .. } => self.narrow_store_truncate_i64(16)?,
            RawInstr::I64Store8 { .. } => self.narrow_store_truncate_i64(8)?,

            RawInstr::F32Add => self.bin_op_float(BinOp::Add, ScalarType::F32)?,
            RawInstr::F32Sub => self.bin_op_float(BinOp::Sub, ScalarType::F32)?,
            RawInstr::F32Mul => self.bin_op_float(BinOp::Mul, ScalarType::F32)?,
            RawInstr::F32Div => self.bin_op_float(BinOp::Div, ScalarType::F32)?,

            RawInstr::F64Add => self.bin_op_float(BinOp::Add, ScalarType::F64)?,
            RawInstr::F64Sub => self.bin_op_float(BinOp::Sub, ScalarType::F64)?,
            RawInstr::F64Mul => self.bin_op_float(BinOp::Mul, ScalarType::F64)?,
            RawInstr::F64Div => self.bin_op_float(BinOp::Div, ScalarType::F64)?,

            RawInstr::Call(idx) => {
                let name = self.intrinsic_names.get(*idx as usize).cloned();
                match name.as_deref() {
                    Some("quark_id") => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::QuarkId { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("local_id") => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::ProtonId { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("group_id") => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::NucleusId { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("barrier") => {
                        self.emit(KernelOp::Barrier);
                    }
                    Some("workgroup_size") => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::ProtonSize { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }

                    // Subgroup / wave intrinsics. Each one consumes
                    // its arg(s) from the WASM stack and emits the
                    // matching IR op. Type lives on the op itself so
                    // per-backend emitters dispatch correctly.
                    Some("subgroup_size") => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::SubgroupSize { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("subgroup_id") => {
                        // No dedicated SubgroupId op yet; the lane
                        // index is `proton_id % subgroup_size` per
                        // the cross-backend convention. Emit ProtonId
                        // as the closest analog — refine to a
                        // dedicated op once a backend needs it.
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::ProtonId { dst });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("shuffle_u32") => self.wave_shuffle(ScalarType::U32)?,
                    Some("ballot_u32") => {
                        let predicate = self.pop()?;
                        let (pr, _) = self.commit(predicate)?;
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::WaveBallot { dst, predicate: pr });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("any_u32") => {
                        let predicate = self.pop()?;
                        let (pr, _) = self.commit(predicate)?;
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::WaveAny { dst, predicate: pr });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    Some("all_u32") => {
                        let predicate = self.pop()?;
                        let (pr, _) = self.commit(predicate)?;
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::WaveAll { dst, predicate: pr });
                        self.stack.push(SymVal::Reg(dst, ScalarType::U32));
                    }
                    // Reduce / scan / shuffle across the portable
                    // Tier-1 type set {u32, i32, f32}. Each variant
                    // emits the same IR op with a different
                    // ScalarType — the per-backend emitter picks
                    // the right native subgroup instruction.
                    Some("reduce_add_u32") => {
                        self.subgroup_reduce(SubgroupOp::Add, ScalarType::U32)?
                    }
                    Some("reduce_add_i32") => {
                        self.subgroup_reduce(SubgroupOp::Add, ScalarType::I32)?
                    }
                    Some("reduce_add_f32") => {
                        self.subgroup_reduce(SubgroupOp::Add, ScalarType::F32)?
                    }
                    Some("reduce_min_u32") => {
                        self.subgroup_reduce(SubgroupOp::Min, ScalarType::U32)?
                    }
                    Some("reduce_min_i32") => {
                        self.subgroup_reduce(SubgroupOp::Min, ScalarType::I32)?
                    }
                    Some("reduce_min_f32") => {
                        self.subgroup_reduce(SubgroupOp::Min, ScalarType::F32)?
                    }
                    Some("reduce_max_u32") => {
                        self.subgroup_reduce(SubgroupOp::Max, ScalarType::U32)?
                    }
                    Some("reduce_max_i32") => {
                        self.subgroup_reduce(SubgroupOp::Max, ScalarType::I32)?
                    }
                    Some("reduce_max_f32") => {
                        self.subgroup_reduce(SubgroupOp::Max, ScalarType::F32)?
                    }
                    Some("scan_add_u32") => self.subgroup_scan_inclusive(ScalarType::U32)?,
                    Some("scan_add_i32") => self.subgroup_scan_inclusive(ScalarType::I32)?,
                    Some("scan_add_f32") => self.subgroup_scan_inclusive(ScalarType::F32)?,
                    Some("scan_add_exclusive_u32") => {
                        self.subgroup_scan_exclusive(ScalarType::U32)?
                    }
                    Some("scan_add_exclusive_i32") => {
                        self.subgroup_scan_exclusive(ScalarType::I32)?
                    }
                    Some("scan_add_exclusive_f32") => {
                        self.subgroup_scan_exclusive(ScalarType::F32)?
                    }
                    Some("shuffle_i32") => self.wave_shuffle(ScalarType::I32)?,
                    Some("shuffle_f32") => self.wave_shuffle(ScalarType::F32)?,

                    // Unary f32 math — these match the F32Ext polyfill in
                    // wasm_compile's wrapper plus direct extern calls.
                    Some("sqrt_f32") => self.math_call_unary(MathFn::Sqrt)?,
                    Some("rsqrt_f32") => self.math_call_unary(MathFn::Rsqrt)?,
                    Some("sin_f32") => self.math_call_unary(MathFn::Sin)?,
                    Some("cos_f32") => self.math_call_unary(MathFn::Cos)?,
                    Some("tan_f32") => self.math_call_unary(MathFn::Tan)?,
                    Some("exp_f32") => self.math_call_unary(MathFn::Exp)?,
                    Some("log_f32") => self.math_call_unary(MathFn::Log)?,
                    Some("abs_f32") => self.math_call_unary(MathFn::Abs)?,
                    Some("floor_f32") => self.math_call_unary(MathFn::Floor)?,
                    Some("ceil_f32") => self.math_call_unary(MathFn::Ceil)?,
                    Some("round_f32") => self.math_call_unary(MathFn::Round)?,

                    // Binary f32 math.
                    Some("min_f32") => self.math_call_binary(MathFn::Min)?,
                    Some("max_f32") => self.math_call_binary(MathFn::Max)?,
                    Some("pow_f32") => self.math_call_binary(MathFn::Pow)?,

                    // Ternary f32 math.
                    Some("clamp_f32") => self.math_call_ternary(MathFn::Clamp)?,
                    Some("fma_f32") => self.math_call_ternary(MathFn::Fma)?,

                    // f64 math — same MathFn enum, the `ty` field
                    // on KernelOp::MathCall comes from the popped
                    // operand's ScalarType so per-backend emitters
                    // dispatch f32 vs f64 automatically.
                    Some("sqrt_f64") => self.math_call_unary(MathFn::Sqrt)?,
                    Some("rsqrt_f64") => self.math_call_unary(MathFn::Rsqrt)?,
                    Some("sin_f64") => self.math_call_unary(MathFn::Sin)?,
                    Some("cos_f64") => self.math_call_unary(MathFn::Cos)?,
                    Some("tan_f64") => self.math_call_unary(MathFn::Tan)?,
                    Some("exp_f64") => self.math_call_unary(MathFn::Exp)?,
                    Some("log_f64") => self.math_call_unary(MathFn::Log)?,
                    Some("abs_f64") => self.math_call_unary(MathFn::Abs)?,
                    Some("floor_f64") => self.math_call_unary(MathFn::Floor)?,
                    Some("ceil_f64") => self.math_call_unary(MathFn::Ceil)?,
                    Some("round_f64") => self.math_call_unary(MathFn::Round)?,
                    Some("min_f64") => self.math_call_binary(MathFn::Min)?,
                    Some("max_f64") => self.math_call_binary(MathFn::Max)?,
                    Some("pow_f64") => self.math_call_binary(MathFn::Pow)?,
                    Some("clamp_f64") => self.math_call_ternary(MathFn::Clamp)?,
                    Some("fma_f64") => self.math_call_ternary(MathFn::Fma)?,

                    // Workgroup-shared memory. The `slot` (first arg)
                    // must be a compile-time `i32.const` so we can lift
                    // it into the IR's `id` field. The `index` is a
                    // runtime register.
                    Some("shared_load_f32") => self.shared_load(ScalarType::F32)?,
                    Some("shared_load_u32") => self.shared_load(ScalarType::U32)?,
                    Some("shared_load_i32") => self.shared_load(ScalarType::I32)?,
                    Some("shared_store_f32") => self.shared_store(ScalarType::F32)?,
                    Some("shared_store_u32") => self.shared_store(ScalarType::U32)?,
                    Some("shared_store_i32") => self.shared_store(ScalarType::I32)?,

                    // Atomic RMW family. Args: (addr: *mut T, val: T,
                    // order: u32). `addr` is the BufferAccess SymVal
                    // pushed by `&mut buf[i]` rewriting; we lift its
                    // slot+index into the IR's `field`/`index`. The
                    // order arg is a compile-time const mapped to
                    // `MemoryOrder`.
                    Some("atomic_add_u32") => self.atomic_rmw(AtomicOp::Add, ScalarType::U32)?,
                    Some("atomic_sub_u32") => self.atomic_rmw(AtomicOp::Sub, ScalarType::U32)?,
                    Some("atomic_min_u32") => self.atomic_rmw(AtomicOp::Min, ScalarType::U32)?,
                    Some("atomic_max_u32") => self.atomic_rmw(AtomicOp::Max, ScalarType::U32)?,
                    Some("atomic_and_u32") => self.atomic_rmw(AtomicOp::And, ScalarType::U32)?,
                    Some("atomic_or_u32") => self.atomic_rmw(AtomicOp::Or, ScalarType::U32)?,
                    Some("atomic_xor_u32") => self.atomic_rmw(AtomicOp::Xor, ScalarType::U32)?,
                    Some("atomic_exchange_u32") => {
                        self.atomic_rmw(AtomicOp::Exchange, ScalarType::U32)?
                    }
                    Some("atomic_add_i32") => self.atomic_rmw(AtomicOp::Add, ScalarType::I32)?,
                    Some("atomic_sub_i32") => self.atomic_rmw(AtomicOp::Sub, ScalarType::I32)?,

                    // Shared-memory atomic family. Same arg shape as the
                    // shared_*_u32 load/store family (slot, index, val,
                    // order) but with an RMW operator. Lowers to
                    // `KernelOp::SharedAtomicOp`.
                    Some("atomic_add_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Add, ScalarType::U32)?
                    }
                    Some("atomic_sub_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Sub, ScalarType::U32)?
                    }
                    Some("atomic_min_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Min, ScalarType::U32)?
                    }
                    Some("atomic_max_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Max, ScalarType::U32)?
                    }
                    Some("atomic_and_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::And, ScalarType::U32)?
                    }
                    Some("atomic_or_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Or, ScalarType::U32)?
                    }
                    Some("atomic_xor_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Xor, ScalarType::U32)?
                    }
                    Some("atomic_exchange_shared_u32") => {
                        self.shared_atomic_rmw(AtomicOp::Exchange, ScalarType::U32)?
                    }
                    Some("atomic_add_shared_i32") => {
                        self.shared_atomic_rmw(AtomicOp::Add, ScalarType::I32)?
                    }
                    Some("atomic_sub_shared_i32") => {
                        self.shared_atomic_rmw(AtomicOp::Sub, ScalarType::I32)?
                    }

                    Some("memory_fence") => self.fence_call()?,

                    // Texture access. The slot (first arg) is a
                    // compile-time `i32.const` that lifts into the
                    // IR's `texture: u32` field; remaining args are
                    // runtime registers.
                    Some("texture_load_2d_f32") => self.texture_load_2d(ScalarType::F32)?,
                    Some("texture_sample_2d_f32") => self.texture_sample_2d(ScalarType::F32)?,
                    Some("texture_load_3d_f32") => self.texture_load_3d(ScalarType::F32)?,
                    Some("texture_write_2d_f32") => self.texture_write_2d(ScalarType::F32)?,

                    Some(other) => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("intrinsic call `{other}` not yet lowered"),
                            at: self.body.body_offset,
                        });
                    }
                    None => {
                        // Defined-function call. rustc injects stdlib
                        // helpers (panic family, alloc shims, etc.) at
                        // function indices the lowering pass cannot
                        // inline. We special-case the panic family —
                        // the GPU contract is UB on division by zero,
                        // so the eqz-guarded panic-then-unreachable
                        // tail rustc emits for `%`/`/` is dead code.
                        // Pop the call's args and emit nothing; the
                        // surrounding control flow already routes the
                        // safe path past this region.
                        let resolved = self
                            .module
                            .function_names
                            .get(*idx as usize)
                            .and_then(|n| n.as_deref());
                        if resolved.is_some_and(is_panic_helper) {
                            self.elide_panic_call(*idx)?;
                        } else if self.try_inline_defined_call(*idx)? {
                            // Successfully inlined the callee body —
                            // the callee's return value (if any) is
                            // now on top of self.stack.
                        } else {
                            let detail = match resolved {
                                Some(name) => format!(
                                    "call to defined function {idx} (`{name}`) — \
                                     inlining not yet supported (callee has unsupported \
                                     control flow or is imported)"
                                ),
                                None => format!(
                                    "call to defined function {idx} (no debug \
                                     name) — inlining not yet supported"
                                ),
                            };
                            return Err(LoweringError::UnsupportedOp {
                                op: detail,
                                at: self.body.body_offset,
                            });
                        }
                    }
                }
            }

            RawInstr::Block { .. } => {
                // Block IS a scope boundary even though it splices into
                // the parent: a `br_if N` from inside the block can
                // skip past `end @N`, so any `local.set` inside is
                // potentially skipped. Capture the pre-block snapshot
                // so `end` can call `merge_locals_post_frame` to reset
                // locals[].val to their stable_reg — `local.get` after
                // the block then reads the merge anchor instead of an
                // inner fresh reg whose write may not have executed.
                // (Originally this was elided as "no scope boundary";
                // PTRD-shape kernels with `result = …` inside an
                // unrolled `while … break` produce exactly the broken
                // pattern.)
                let snapshot = self.snapshot_locals();
                let parent_brifs = self.frames.last().map(|f| f.brifs.len()).unwrap_or(0);
                self.frames.push(Frame {
                    kind: FrameKind::Block,
                    ops: Vec::new(),
                    local_snapshot: snapshot,
                    brifs: Vec::new(),
                    parent_brifs_at_open: parent_brifs,
                });
            }

            RawInstr::Loop { .. } => {
                // Quanta's KernelOp::Loop is a bounded `for i in 0..count`.
                // For an unbounded WASM loop we use a sentinel max
                // count (10000, matching the legacy parser's
                // emit_while_loop). The loop body breaks early via
                // br_if when the user's condition becomes false.
                let count_reg = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: count_reg,
                    value: ConstValue::U32(10_000),
                });
                let iter_reg = self.alloc_reg();
                let snapshot = self.snapshot_locals();
                // Force every local's current binding to its
                // stable_reg before entering the loop body.
                // Why: the body lowers ONCE but runs many times.
                // A mutable accumulator (e.g. `sum = sum + x`)
                // needs reads at the start of iter N+1 to see
                // the writes at the end of iter N. Fresh
                // per-set allocates a NEW reg each iteration,
                // but the lowered body can only reference one
                // specific reg name — so each iteration's
                // BinOp reads the WRONG register (the fresh
                // from a phantom "previous lowering"). The
                // stable_reg side-channel gets updated each
                // iteration via the parallel Copy in
                // `write_local_via_copy`, so making reads pull
                // from stable_reg makes accumulators work.
                //
                // Straight-line code with no loop is unaffected
                // (no frame entry → no force).
                self.force_locals_to_stable();
                self.loop_reads.push(std::collections::BTreeSet::new());
                let parent_brifs = self.frames.last().map(|f| f.brifs.len()).unwrap_or(0);
                self.frames.push(Frame {
                    kind: FrameKind::Loop {
                        count_reg,
                        iter_reg,
                    },
                    ops: Vec::new(),
                    local_snapshot: snapshot,
                    brifs: Vec::new(),
                    parent_brifs_at_open: parent_brifs,
                });
            }

            RawInstr::If { .. } => {
                let cond_sv = self.pop()?;
                let (cond, _) = self.commit(cond_sv)?;
                let snapshot = self.snapshot_locals();
                let parent_brifs = self.frames.last().map(|f| f.brifs.len()).unwrap_or(0);
                self.frames.push(Frame {
                    kind: FrameKind::If { cond },
                    ops: Vec::new(),
                    local_snapshot: snapshot,
                    brifs: Vec::new(),
                    parent_brifs_at_open: parent_brifs,
                });
            }

            RawInstr::Else => {
                let frame = self.frames.pop().ok_or_else(|| {
                    LoweringError::ShapeMismatch("Else with empty frame stack".into())
                })?;
                let cond = match frame.kind {
                    FrameKind::If { cond } => cond,
                    _ => {
                        return Err(LoweringError::ShapeMismatch(
                            "Else not preceded by an If frame".into(),
                        ));
                    }
                };
                // Restore bindings to the If-entry snapshot so
                // the else-branch starts from the same baseline.
                // The then-branch's local rebindings are
                // captured in `then_ops`; whatever they updated
                // stays in stable_reg via the parallel Copy in
                // `write_local_via_copy`. The merge at the
                // matching End uses the snapshot to detect which
                // locals to reset post-frame.
                self.restore_locals_from_snapshot(&frame.local_snapshot);
                self.frames.push(Frame {
                    kind: FrameKind::Else {
                        cond,
                        then_ops: frame.ops,
                    },
                    ops: Vec::new(),
                    local_snapshot: frame.local_snapshot,
                    brifs: Vec::new(),
                    // Else inherits the If's parent snapshot.
                    parent_brifs_at_open: frame.parent_brifs_at_open,
                });
            }

            RawInstr::End => {
                let frame = self.frames.pop().ok_or_else(|| {
                    LoweringError::ShapeMismatch("End with empty frame stack".into())
                })?;
                match frame.kind {
                    FrameKind::Function => {
                        // Function-level End — done. Reconstruct any
                        // function-level br_if records (top-level
                        // `return` records an always-true one), then
                        // push back onto the stack so into_kernel_def
                        // can read it.
                        let ops = if !frame.brifs.is_empty() {
                            self.reconstruct_block_brifs(frame.ops, &frame.brifs)
                        } else {
                            frame.ops
                        };
                        self.frames.push(Frame {
                            kind: FrameKind::Function,
                            ops,
                            local_snapshot: Vec::new(),
                            brifs: Vec::new(),
                            parent_brifs_at_open: 0,
                        });
                    }
                    FrameKind::Block => {
                        // If any br/br_if was recorded on this Block
                        // frame, reconstruct nested
                        // Branches around the post-record op ranges
                        // before splicing into the parent.
                        let ops = if !frame.brifs.is_empty() {
                            self.reconstruct_block_brifs(frame.ops, &frame.brifs)
                        } else {
                            frame.ops
                        };
                        // Block was a label scope — splice ops into
                        // the parent.
                        let parent_idx = self.frames.len() - 1;
                        let splice_len = ops.len();
                        Self::splice_into_frame(&mut self.frames[parent_idx], ops);
                        // Any brifs recorded on parent DURING this
                        // Block's lifetime semantically fire AFTER
                        // this Block's ops execute. Bump their
                        // sink_position by splice_len so the wrap
                        // doesn't engulf this Block's content
                        // (= treat that content as having already
                        // run unconditionally before the brif).
                        self.bump_parent_brifs(frame.parent_brifs_at_open, splice_len);
                        // Then merge locals modified inside the block
                        // back to their stable_reg, so reads after the
                        // block (`local.get`) see the merge anchor
                        // rather than an inner fresh reg whose write
                        // may have been skipped by `br_if N`.
                        self.merge_locals_post_frame(&frame.local_snapshot);
                    }
                    FrameKind::Loop {
                        count_reg,
                        iter_reg,
                    } => {
                        // Reads recorded inside this loop also count
                        // for enclosing loops (the insert in LocalGet
                        // already wrote to every open set) — just drop
                        // this loop's own set.
                        self.loop_reads.pop();
                        self.emit(KernelOp::Loop {
                            count: count_reg,
                            iter_reg,
                            body: frame.ops,
                        });
                        // Records made on the parent during this
                        // loop's lifetime (loop-crossing exit flags)
                        // fire after the Loop op executes — bump them
                        // past it, mirroring the Block-close splice
                        // bump.
                        self.bump_parent_brifs(frame.parent_brifs_at_open, 1);
                        // Post-loop: any local set inside the
                        // body now lives only via stable_reg.
                        self.merge_locals_post_frame(&frame.local_snapshot);
                    }
                    FrameKind::If { cond } => {
                        self.emit(KernelOp::Branch {
                            cond,
                            then_ops: frame.ops,
                            else_ops: Vec::new(),
                        });
                        self.bump_parent_brifs(frame.parent_brifs_at_open, 1);
                        self.merge_locals_post_frame(&frame.local_snapshot);
                    }
                    FrameKind::Else { cond, then_ops } => {
                        self.emit(KernelOp::Branch {
                            cond,
                            then_ops,
                            else_ops: frame.ops,
                        });
                        self.bump_parent_brifs(frame.parent_brifs_at_open, 1);
                        self.merge_locals_post_frame(&frame.local_snapshot);
                    }
                }
            }

            RawInstr::Br(depth) => {
                // br N: unconditional jump to end of Nth enclosing
                // label. Cases:
                //   - Target is a Loop: continue. Quanta's structured
                //     Loop wraps automatically, so this is a no-op.
                //   - Target is non-Loop, NO loop is between us and
                //     target: install redirect on target with cond=true
                //     so subsequent ops flow into an unreachable
                //     else-arm.
                //   - Target is non-Loop AND there's a Loop between us
                //     and target: emit `Break` from the loop. The
                //     post-loop redirect can't carry a loop-internal
                //     cond (registers don't escape `KernelOp::Loop.body`)
                //     so we trust the post-loop trajectory is the same
                //     for early-exit and natural-exit. See 5d.3 docs.
                let target_kind = self
                    .frame_at_depth(*depth)
                    .ok_or_else(|| {
                        LoweringError::ShapeMismatch(format!("br {depth} out of range"))
                    })?
                    .kind_discriminant();
                if matches!(target_kind, FrameKindTag::Loop) {
                    // Continue; no-op for structured Loop.
                    return Ok(());
                }
                if self.has_loop_between_top_and_depth(*depth) {
                    if self.loops_between_top_and_depth(*depth) == 1
                        && matches!(target_kind, FrameKindTag::Block)
                    {
                        self.emit_loop_crossing_exit(*depth, None);
                        return Ok(());
                    }
                    // Multi-loop crossings and non-Block targets keep
                    // the label-lossy plain Break (pre-existing 5d.3
                    // semantics: correct when nothing meaningful sits
                    // between the loop and the target's end). Audited
                    // 2026-06-12: zero occurrences on the kernel
                    // surface.
                    self.emit(KernelOp::Break);
                    return Ok(());
                }
                // For Block targets, recognize the canonical wasm
                // `if/else` pattern: `br_if cond M;
                // br N` where M < N. The br_if exits to M's
                // continuation when cond is true; the br exits to
                // N's continuation otherwise. So N's continuation
                // runs when cond was TRUE, M's continuation runs
                // when cond was FALSE.
                //
                // To encode this, look at the most recent brif on a
                // frame strictly INNER than the br's target (depths
                // 0..N-1 from current). If found, use its cond to
                // build a `!cond` reg as the br's wrap cond: the
                // wrap's `else=[N-tail]` then runs when !cond is
                // false = cond is true. ✓
                //
                // If no such brif exists, fall back to const_true
                // (the wrap is dead, matching strict WASM dead-code
                // semantics).
                if !matches!(target_kind, FrameKindTag::Block) {
                    // If/Else/Function-labelled br targets never
                    // occur in LLVM-generated wasm on the current
                    // kernel surface (audited 2026-06-12: zero hits
                    // across workspace tests, prims, rand, examples,
                    // benches). The legacy redirect-chain path that
                    // used to "handle" them produced IR the
                    // scope_check oracle had to catch. Fail loudly
                    // instead; add a record-and-wrap route with a
                    // witness if a real kernel ever hits this.
                    return Err(LoweringError::UnsupportedOp {
                        op: format!(
                            "br to {target_kind:?}-labelled frame at depth {depth} \
                             without an intervening loop"
                        ),
                        at: self.body.body_offset,
                    });
                }
                // Find the most recent br_if record on any frame
                // strictly inner than target's depth. If found,
                // use ITS cond as the unconditional br's record
                // cond; reconstruct will then swap then/else so
                // tail runs when prior_cond=true (= the inner
                // br_if fired and we reached target's continuation
                // via that path).
                //
                // If no prior br_if: emit a const_true at target
                // and use it as cond. The wrap will be dead per
                // strict WASM semantics.
                let mut prior_cond: Option<Reg> = None;
                for d in 0..*depth {
                    if let Some(f) = self.frame_at_depth(d)
                        && let Some(record) = f.brifs.last()
                    {
                        prior_cond = Some(record.cond);
                    }
                }
                let cond_reg = if let Some(prior) = prior_cond {
                    prior
                } else {
                    let dst = self.alloc_reg();
                    self.emit_const_bool_to_target(*depth, dst, true);
                    dst
                };
                self.record_br_at(*depth, cond_reg, /*is_unconditional=*/ true);
            }

            RawInstr::BrIf(depth) => {
                let cond_sv = self.pop()?;
                let (cond, cond_ty) = self.commit(cond_sv)?;
                let target_kind = self
                    .frame_at_depth(*depth)
                    .ok_or_else(|| {
                        LoweringError::ShapeMismatch(format!("br_if {depth} out of range"))
                    })?
                    .kind_discriminant();
                if matches!(target_kind, FrameKindTag::Loop) {
                    // `br_if cond 0` to the enclosing Loop = "continue
                    // if cond, else fall through". rustc emits this at
                    // the bottom of `for`/`while` loops as the
                    // iteration check. Quanta's structured Loop has no
                    // explicit continue, but its body wrapper auto-
                    // continues on natural fall-through. So we model
                    // the inverse: Break when cond is false. The
                    // emitted `Branch { cond, then_ops: [], else_ops:
                    // [Break] }` runs Break only on the !cond path,
                    // letting the cond=true path fall through to the
                    // loop wrap-around.
                    self.emit(KernelOp::Branch {
                        cond,
                        then_ops: Vec::new(),
                        else_ops: vec![KernelOp::Break],
                    });
                    return Ok(());
                }
                // br_if from inside a loop targeting outside: emit
                // `Branch { cond, then_ops: [Break], else_ops: [] }`
                // so the cond register stays inside the Loop body
                // where it was defined. The redirect-chain can't be
                // used here — its `cond` would be referenced from the
                // outer frame, but `KernelOp::Loop.body` encapsulates
                // the cond and prevents codegen from finding it.
                if self.has_loop_between_top_and_depth(*depth) {
                    if self.loops_between_top_and_depth(*depth) == 1
                        && matches!(target_kind, FrameKindTag::Block)
                    {
                        self.emit_loop_crossing_exit(*depth, Some(cond));
                        return Ok(());
                    }
                    // Multi-loop crossings and non-Block targets keep
                    // the label-lossy conditional Break (pre-existing
                    // 5d.3 semantics). Audited 2026-06-12: zero
                    // occurrences on the kernel surface.
                    self.emit(KernelOp::Branch {
                        cond,
                        then_ops: vec![KernelOp::Break],
                        else_ops: Vec::new(),
                    });
                    return Ok(());
                }
                if !matches!(target_kind, FrameKindTag::Block) {
                    // If/Else/Function-labelled br_if targets never
                    // occur in LLVM-generated wasm on the current
                    // kernel surface (audited 2026-06-12: zero hits
                    // across workspace tests, prims, rand, examples,
                    // benches). Fail loudly instead of guessing; add
                    // a record-and-wrap route with a witness if a
                    // real kernel ever hits this.
                    return Err(LoweringError::UnsupportedOp {
                        op: format!(
                            "br_if to {target_kind:?}-labelled frame at depth {depth} \
                             without an intervening loop"
                        ),
                        at: self.body.body_offset,
                    });
                }
                // Every frame between here and the target must also
                // be a Block — each gets a record below, and only
                // Block (and Function) closes reconstruct records.
                for d in 0..*depth {
                    let k = self
                        .frame_at_depth(d)
                        .expect("depth verified above")
                        .kind_discriminant();
                    if !matches!(k, FrameKindTag::Block) {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!(
                                "br_if to Block at depth {depth} crosses a \
                                 {k:?}-labelled frame at depth {d}"
                            ),
                            at: self.body.body_offset,
                        });
                    }
                }
                // Record the br_if on EVERY frame from the current
                // one up to the target. A br_if needn't be the last
                // instruction of its block — LLVM puts live code
                // after it (e.g. an accept path following two reject
                // br_ifs) — so each level's tail after the br_if site
                // must be skipped when the branch fires, not just the
                // target's continuation. One record per level wraps
                // exactly that level's tail in
                // `Branch{cond, then=[], else=tail}` at its End.
                // (Caught 2026-06-12 by the generated host-oracle
                // differential test: the unguarded current-frame tail
                // ran an accept unconditionally.)
                //
                // The cond is materialized into a register declared
                // at the target frame and assigned right here at the
                // source position — every wrap then reads a value
                // computed by exactly the control flow wasm gave it.
                // See materialize_cond_for_v2 for why this replaced
                // the transitive backward-slice hoist.
                let mat = self.materialize_cond_for_v2(*depth, cond, cond_ty);
                for d in 0..=*depth {
                    self.record_br_at(d, mat, /*is_unconditional=*/ false);
                }
            }

            RawInstr::I32Eqz => {
                let a = self.pop()?;
                let (ar, ty) = self.commit(a)?;
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::I32(0),
                });
                let dst = self.alloc_reg();
                self.emit(KernelOp::Cmp {
                    dst,
                    a: ar,
                    b: zero,
                    op: quanta_ir::CmpOp::Eq,
                    ty,
                });
                self.stack.push(SymVal::Reg(dst, ScalarType::Bool));
            }

            RawInstr::I32And => self.bin_op_int(BinOp::BitAnd, ScalarType::I32)?,
            RawInstr::I32Or => self.bin_op_int(BinOp::BitOr, ScalarType::I32)?,
            RawInstr::I32Xor => self.bin_op_int(BinOp::BitXor, ScalarType::I32)?,
            RawInstr::I32Sub => self.bin_op_int(BinOp::Sub, ScalarType::I32)?,
            RawInstr::I32Mul => self.bin_op_int(BinOp::Mul, ScalarType::I32)?,
            RawInstr::I32DivU | RawInstr::I32DivS => {
                self.bin_op_int(BinOp::Div, ScalarType::I32)?
            }
            RawInstr::I32RemU | RawInstr::I32RemS => {
                self.bin_op_int(BinOp::Rem, ScalarType::I32)?
            }
            // Right-shift: WASM distinguishes signed (`i32.shr_s`) from
            // unsigned (`i32.shr_u`) at the instruction level. We
            // forward that distinction to the IR by picking I32 vs U32
            // as the BinOp::Shr operand type — the CPU evaluator and
            // SPIR-V emitter both branch on the sign of the type to
            // pick arithmetic vs logical shift.
            RawInstr::I32ShrU => self.bin_op_int(BinOp::Shr, ScalarType::U32)?,
            RawInstr::I32ShrS => self.bin_op_int(BinOp::Shr, ScalarType::I32)?,
            RawInstr::I32Rotl => self.bin_op_int(BinOp::Rotl, ScalarType::I32)?,
            RawInstr::I32Rotr => self.bin_op_int(BinOp::Rotr, ScalarType::I32)?,

            RawInstr::I32LtU | RawInstr::I32LtS => {
                self.cmp_op_int(quanta_ir::CmpOp::Lt, ScalarType::I32)?
            }
            RawInstr::I32LeU | RawInstr::I32LeS => {
                self.cmp_op_int(quanta_ir::CmpOp::Le, ScalarType::I32)?
            }
            RawInstr::I32GtU | RawInstr::I32GtS => {
                self.cmp_op_int(quanta_ir::CmpOp::Gt, ScalarType::I32)?
            }
            RawInstr::I32GeU | RawInstr::I32GeS => {
                self.cmp_op_int(quanta_ir::CmpOp::Ge, ScalarType::I32)?
            }
            RawInstr::I32Eq => self.cmp_op_int(quanta_ir::CmpOp::Eq, ScalarType::I32)?,
            RawInstr::I32Ne => self.cmp_op_int(quanta_ir::CmpOp::Ne, ScalarType::I32)?,

            // i64 arithmetic + comparison. Mirrors the i32 surface above;
            // dispatched through the same generic bin_op_int/cmp_op_int
            // helpers with the U64 width class.
            RawInstr::I64Const(v) => self.stack.push(SymVal::I64Const(*v)),
            RawInstr::I64Add => self.bin_op_int(BinOp::Add, ScalarType::I64)?,
            RawInstr::I64Sub => self.bin_op_int(BinOp::Sub, ScalarType::I64)?,
            RawInstr::I64Mul => self.bin_op_int(BinOp::Mul, ScalarType::I64)?,
            RawInstr::I64DivU | RawInstr::I64DivS => {
                self.bin_op_int(BinOp::Div, ScalarType::I64)?
            }
            RawInstr::I64RemU | RawInstr::I64RemS => {
                self.bin_op_int(BinOp::Rem, ScalarType::I64)?
            }
            RawInstr::I64And => self.bin_op_int(BinOp::BitAnd, ScalarType::I64)?,
            RawInstr::I64Or => self.bin_op_int(BinOp::BitOr, ScalarType::I64)?,
            RawInstr::I64Xor => self.bin_op_int(BinOp::BitXor, ScalarType::I64)?,
            RawInstr::I64Shl => self.bin_op_int(BinOp::Shl, ScalarType::I64)?,
            RawInstr::I64ShrU => self.bin_op_int(BinOp::Shr, ScalarType::U64)?,
            RawInstr::I64ShrS => self.bin_op_int(BinOp::Shr, ScalarType::I64)?,
            RawInstr::I64Rotl => self.bin_op_int(BinOp::Rotl, ScalarType::I64)?,
            RawInstr::I64Rotr => self.bin_op_int(BinOp::Rotr, ScalarType::I64)?,
            RawInstr::I64Eq => self.cmp_op_int(quanta_ir::CmpOp::Eq, ScalarType::I64)?,
            RawInstr::I64Ne => self.cmp_op_int(quanta_ir::CmpOp::Ne, ScalarType::I64)?,
            RawInstr::I64LtU | RawInstr::I64LtS => {
                self.cmp_op_int(quanta_ir::CmpOp::Lt, ScalarType::I64)?
            }
            RawInstr::I64LeU | RawInstr::I64LeS => {
                self.cmp_op_int(quanta_ir::CmpOp::Le, ScalarType::I64)?
            }
            RawInstr::I64GtU | RawInstr::I64GtS => {
                self.cmp_op_int(quanta_ir::CmpOp::Gt, ScalarType::I64)?
            }
            RawInstr::I64GeU | RawInstr::I64GeS => {
                self.cmp_op_int(quanta_ir::CmpOp::Ge, ScalarType::I64)?
            }
            RawInstr::I64Eqz => {
                let a = self.pop()?;
                let (ar, _) = self.commit(a)?;
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::I64(0),
                });
                let dst = self.alloc_reg();
                self.emit(KernelOp::Cmp {
                    dst,
                    a: ar,
                    b: zero,
                    op: quanta_ir::CmpOp::Eq,
                    ty: ScalarType::I64,
                });
                self.stack.push(SymVal::Reg(dst, ScalarType::Bool));
            }

            RawInstr::F32Lt => self.cmp_op_float(quanta_ir::CmpOp::Lt, ScalarType::F32)?,
            RawInstr::F32Le => self.cmp_op_float(quanta_ir::CmpOp::Le, ScalarType::F32)?,
            RawInstr::F32Gt => self.cmp_op_float(quanta_ir::CmpOp::Gt, ScalarType::F32)?,
            RawInstr::F32Ge => self.cmp_op_float(quanta_ir::CmpOp::Ge, ScalarType::F32)?,
            RawInstr::F32Eq => self.cmp_op_float(quanta_ir::CmpOp::Eq, ScalarType::F32)?,
            RawInstr::F32Ne => self.cmp_op_float(quanta_ir::CmpOp::Ne, ScalarType::F32)?,

            RawInstr::F64Lt => self.cmp_op_float(quanta_ir::CmpOp::Lt, ScalarType::F64)?,
            RawInstr::F64Le => self.cmp_op_float(quanta_ir::CmpOp::Le, ScalarType::F64)?,
            RawInstr::F64Gt => self.cmp_op_float(quanta_ir::CmpOp::Gt, ScalarType::F64)?,
            RawInstr::F64Ge => self.cmp_op_float(quanta_ir::CmpOp::Ge, ScalarType::F64)?,
            RawInstr::F64Eq => self.cmp_op_float(quanta_ir::CmpOp::Eq, ScalarType::F64)?,
            RawInstr::F64Ne => self.cmp_op_float(quanta_ir::CmpOp::Ne, ScalarType::F64)?,

            RawInstr::F32ConvertI32U => self.cast_op(ScalarType::U32, ScalarType::F32)?,
            RawInstr::F32ConvertI32S => self.cast_op(ScalarType::I32, ScalarType::F32)?,
            RawInstr::I32TruncF32U => self.cast_op(ScalarType::F32, ScalarType::U32)?,
            RawInstr::I32TruncF32S => self.cast_op(ScalarType::F32, ScalarType::I32)?,
            // i64 widening / narrowing. ExtendI32U is the `as u64`
            // path (zero-extend); ExtendI32S is the `as i64` from
            // signed (sign-extend); WrapI64 truncates the high 32
            // bits (matches WASM spec — drops upper word regardless
            // of sign).
            RawInstr::I64ExtendI32U => self.cast_op(ScalarType::U32, ScalarType::U64)?,
            RawInstr::I64ExtendI32S => self.cast_op(ScalarType::I32, ScalarType::I64)?,
            RawInstr::I32WrapI64 => self.cast_op(ScalarType::U64, ScalarType::U32)?,

            // f32 ↔ f64 width conversions.
            RawInstr::F64PromoteF32 => self.cast_op(ScalarType::F32, ScalarType::F64)?,
            RawInstr::F32DemoteF64 => self.cast_op(ScalarType::F64, ScalarType::F32)?,
            // f64 ↔ int. Sign of the source picks the from-type for
            // conversions to f64; for truncations to int, the dest
            // signedness picks the to-type. Reinterpret is a bitcast
            // — same width on both sides.
            RawInstr::F64ConvertI32U => self.cast_op(ScalarType::U32, ScalarType::F64)?,
            RawInstr::F64ConvertI32S => self.cast_op(ScalarType::I32, ScalarType::F64)?,
            RawInstr::F64ConvertI64U => self.cast_op(ScalarType::U64, ScalarType::F64)?,
            RawInstr::F64ConvertI64S => self.cast_op(ScalarType::I64, ScalarType::F64)?,
            RawInstr::I32TruncF64U => self.cast_op(ScalarType::F64, ScalarType::U32)?,
            RawInstr::I32TruncF64S => self.cast_op(ScalarType::F64, ScalarType::I32)?,
            RawInstr::I64TruncF64U => self.cast_op(ScalarType::F64, ScalarType::U64)?,
            RawInstr::I64TruncF64S => self.cast_op(ScalarType::F64, ScalarType::I64)?,
            // Saturating trunc variants — same `Cast` lowering. The
            // single-opcode form rustc emits when
            // `nontrapping-fptoint` is enabled (the default since
            // 2020); avoids the multi-instruction manual saturation
            // block whose structured-control lowering exposed
            // bug #1, bug #2, and the substrate redirect bug.
            RawInstr::I32TruncSatF32U => self.cast_op(ScalarType::F32, ScalarType::U32)?,
            RawInstr::I32TruncSatF32S => self.cast_op(ScalarType::F32, ScalarType::I32)?,
            RawInstr::I32TruncSatF64U => self.cast_op(ScalarType::F64, ScalarType::U32)?,
            RawInstr::I32TruncSatF64S => self.cast_op(ScalarType::F64, ScalarType::I32)?,
            RawInstr::I64TruncSatF32U => self.cast_op(ScalarType::F32, ScalarType::U64)?,
            RawInstr::I64TruncSatF32S => self.cast_op(ScalarType::F32, ScalarType::I64)?,
            RawInstr::I64TruncSatF64U => self.cast_op(ScalarType::F64, ScalarType::U64)?,
            RawInstr::I64TruncSatF64S => self.cast_op(ScalarType::F64, ScalarType::I64)?,
            RawInstr::F64ReinterpretI64 => self.cast_op(ScalarType::U64, ScalarType::F64)?,
            RawInstr::I64ReinterpretF64 => self.cast_op(ScalarType::F64, ScalarType::U64)?,

            // Inline f32 math ops. rustc's optimizer collapses calls
            // to F32Ext methods (`.sqrt()`, `.abs()`) into LLVM
            // intrinsics that lower to these WASM operators directly,
            // so the call-table path alone isn't enough to cover them.
            RawInstr::F32Sqrt => self.math_call_unary(MathFn::Sqrt)?,
            RawInstr::F32Abs => self.math_call_unary(MathFn::Abs)?,
            RawInstr::F32Neg => {
                let a = self.pop()?;
                let (ar, ty) = self.commit(a)?;
                let dst = self.alloc_reg();
                self.emit(KernelOp::UnaryOp {
                    dst,
                    a: ar,
                    op: quanta_ir::UnaryOp::Neg,
                    ty,
                });
                self.stack.push(SymVal::Reg(dst, ty));
            }
            RawInstr::F32Min => self.math_call_binary(MathFn::Min)?,
            RawInstr::F32Max => self.math_call_binary(MathFn::Max)?,

            // f64 unary / binary — same shape as the f32 arms above.
            // math_call_unary / math_call_binary are polymorphic via
            // the operand's commit-time type, so they pick up F64 from
            // the popped SymVal automatically.
            RawInstr::F64Sqrt => self.math_call_unary(MathFn::Sqrt)?,
            RawInstr::F64Abs => self.math_call_unary(MathFn::Abs)?,
            RawInstr::F64Neg => {
                let a = self.pop()?;
                let (ar, ty) = self.commit(a)?;
                let dst = self.alloc_reg();
                self.emit(KernelOp::UnaryOp {
                    dst,
                    a: ar,
                    op: quanta_ir::UnaryOp::Neg,
                    ty,
                });
                self.stack.push(SymVal::Reg(dst, ty));
            }
            RawInstr::F64Min => self.math_call_binary(MathFn::Min)?,
            RawInstr::F64Max => self.math_call_binary(MathFn::Max)?,

            // `unreachable` follows an elided panic call as rustc's
            // dead-code marker; `nop` is a literal no-op. Both produce
            // no IR.
            RawInstr::Unreachable | RawInstr::Nop => {}

            // WASM `select` is a value-level ternary: pop (val_a,
            // val_b, cond), push val_a if cond is non-zero else
            // val_b. Quanta IR has no native select, so we model it
            // with a `Branch` whose arms each Copy the chosen value
            // into a destination register that's pre-initialized
            // (unconditionally defined) so the post-Select read
            // doesn't see a possibly-undefined reg — same SSA-join
            // discipline as 5d.2's stable per-local registers.
            RawInstr::Select => {
                let cond_sv = self.pop()?;
                let b_sv = self.pop()?;
                let a_sv = self.pop()?;
                let (cond, _) = self.commit(cond_sv)?;
                let (a_reg, ty) = self.commit(a_sv)?;
                let (b_reg, _) = self.commit(b_sv)?;
                let dst = self.alloc_reg();
                let init = match ty {
                    ScalarType::F16
                    | ScalarType::BF16
                    | ScalarType::FP8E5M2
                    | ScalarType::FP8E4M3
                    | ScalarType::F32 => ConstValue::F32(0.0),
                    ScalarType::F64 => ConstValue::F64(0.0),
                    ScalarType::Bool => ConstValue::Bool(false),
                    ScalarType::I8
                    | ScalarType::I16
                    | ScalarType::I32
                    | ScalarType::I64
                    | ScalarType::I4 => ConstValue::I32(0),
                    ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 => {
                        ConstValue::U32(0)
                    }
                };
                self.emit(KernelOp::Const { dst, value: init });
                self.emit(KernelOp::Branch {
                    cond,
                    then_ops: vec![KernelOp::Copy {
                        dst,
                        src: a_reg,
                        ty,
                    }],
                    else_ops: vec![KernelOp::Copy {
                        dst,
                        src: b_reg,
                        ty,
                    }],
                });
                self.stack.push(SymVal::Reg(dst, ty));
            }

            // `return` is a function-level early exit. Quanta kernels
            // are all `() -> ()` so there's nothing to push.
            //
            //   - At function top-level (no enclosing block/loop):
            //     record an always-true br_if on the function frame;
            //     `into_kernel_def` reconstructs it as
            //     `Branch{true, then=[], else=tail}` so ops after
            //     the Return are genuinely skipped.
            //   - Inside any frame: emit nothing. Subsequent ops in
            //     the same block run if reached (they're WASM
            //     polymorphic-stack dead code, so usually trivial),
            //     and the surrounding control-flow drops them
            //     naturally on the Return path. Ops AFTER the
            //     enclosing block (typically the elided panic for
            //     the alternate path) are dead too.
            //   - Crossing a Loop: emit `Break` so we exit the loop
            //     body cleanly.
            RawInstr::Return => {
                let depth = (self.frames.len() - 1) as u32;
                let inside_block = self.frames.len() > 1;
                if self.has_loop_between_top_and_depth(depth) {
                    // Label-lossy: ops between the loop and function
                    // end still run after the Break. Correct for the
                    // shipped mid-body-Return kernels (the trailing
                    // ops ARE the intended continuation); a
                    // skip-the-rest Return inside a loop would need
                    // the exit-flag treatment of
                    // emit_loop_crossing_exit aimed at the function
                    // frame. Audited 2026-06-12: zero occurrences.
                    self.emit(KernelOp::Break);
                } else if !inside_block {
                    let dst = self.alloc_reg();
                    self.emit_const_bool_to_target(depth, dst, true);
                    self.record_br_at(depth, dst, /*is_unconditional=*/ false);
                }
                // else: inside a block but not crossing a loop —
                // emit nothing. The block's End will close cleanly
                // and any post-block ops are typically panic-elided
                // dead code.
            }

            // `drop` discards the top of the symbolic stack.
            RawInstr::Drop => {
                let _ = self.pop()?;
            }

            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("{other:?}"),
                    at: self.body.body_offset,
                });
            }
        }
        Ok(())
    }

    fn bin_op_float(&mut self, op: BinOp, ty: ScalarType) -> Result<(), LoweringError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, _) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst,
            a: ar,
            b: br,
            op,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    fn bin_op_int(&mut self, op: BinOp, ty: ScalarType) -> Result<(), LoweringError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, _) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst,
            a: ar,
            b: br,
            op,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    fn cmp_op_int(&mut self, op: quanta_ir::CmpOp, ty: ScalarType) -> Result<(), LoweringError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, _) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::Cmp {
            dst,
            a: ar,
            b: br,
            op,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ScalarType::Bool));
        Ok(())
    }

    fn cmp_op_float(&mut self, op: quanta_ir::CmpOp, ty: ScalarType) -> Result<(), LoweringError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, _) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::Cmp {
            dst,
            a: ar,
            b: br,
            op,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ScalarType::Bool));
        Ok(())
    }

    fn cast_op(&mut self, from: ScalarType, to: ScalarType) -> Result<(), LoweringError> {
        let a = self.pop()?;
        let (ar, _) = self.commit(a)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::Cast {
            dst,
            src: ar,
            from,
            to,
        });
        self.stack.push(SymVal::Reg(dst, to));
        Ok(())
    }

    /// Resolve a buffer slot's scalar type from the side table.
    /// Falls back to `U32` if the slot isn't found — that's the
    /// default rustc emits for raw integer pointers.
    fn scalar_type_for_slot(&self, slot: u32) -> ScalarType {
        self.side_table
            .params
            .iter()
            .find(|p| p.slot == slot)
            .map(|p| p.scalar)
            .unwrap_or(ScalarType::U32)
    }

    /// Emit a Const-index register for a buffer Load/Store where the
    /// element address is `BufferPtr + memarg-offset`. Used when
    /// rustc collapses `out[N] = …` (compile-time-constant N) to a
    /// bare BufferPtr plus an immediate memarg offset.
    ///
    /// `byte_offset` is the memarg offset in bytes (which must be a
    /// multiple of `elem_size`); returns a register holding
    /// `byte_offset / elem_size` as `u32`.
    fn const_index_for_offset(&mut self, byte_offset: u64, elem_size: u32) -> Reg {
        let idx = (byte_offset as u32) / elem_size;
        let reg = self.alloc_reg();
        self.emit(KernelOp::Const {
            dst: reg,
            value: ConstValue::U32(idx),
        });
        reg
    }

    /// Coerce a register holding `src_ty` to `dst_ty`. If the types
    /// already match the register is returned unchanged; otherwise a
    /// `KernelOp::Cast` is emitted and the new register is returned.
    ///
    /// Mainly used to widen `Bool` operands going into subgroup
    /// reduce/scan ops — rustc's optimiser can elide a `bool as u32`
    /// cast when it's a no-op at the WASM bytecode level, but the
    /// downstream MSL / WGSL / SPIR-V emitters need the explicit
    /// type because `simd_prefix_inclusive_sum(bool)` and the
    /// equivalents on other backends aren't valid overloads.
    fn coerce_to(&mut self, src: Reg, src_ty: ScalarType, dst_ty: ScalarType) -> Reg {
        if src_ty == dst_ty {
            return src;
        }
        let dst = self.alloc_reg();
        self.emit(KernelOp::Cast {
            dst,
            src,
            from: src_ty,
            to: dst_ty,
        });
        dst
    }

    /// Drop the args of an elided panic helper from the symbolic
    /// stack and emit nothing. Caller has already verified the
    /// callee name belongs to the panic family.
    fn elide_panic_call(&mut self, fn_idx: u32) -> Result<(), LoweringError> {
        let info = self.module.functions.get(fn_idx as usize).ok_or_else(|| {
            LoweringError::ShapeMismatch(format!(
                "panic-helper function index {fn_idx} out of range"
            ))
        })?;
        let sig = self
            .module
            .types
            .get(info.type_index as usize)
            .ok_or_else(|| {
                LoweringError::ShapeMismatch(format!(
                    "panic-helper type index {} out of range",
                    info.type_index
                ))
            })?;
        for _ in 0..sig.params.len() {
            let _ = self.pop()?;
        }
        Ok(())
    }

    fn math_call_unary(&mut self, func: MathFn) -> Result<(), LoweringError> {
        let a = self.pop()?;
        let (ar, ty) = self.commit(a)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::MathCall {
            dst,
            func,
            args: vec![ar],
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    fn math_call_binary(&mut self, func: MathFn) -> Result<(), LoweringError> {
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, ty) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::MathCall {
            dst,
            func,
            args: vec![ar, br],
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Commit a SymVal to a register, but if the slot is f32-typed
    /// and the value is an `i32.const`, reinterpret the bits as f32
    /// rather than treating the int as a numeric value to convert.
    /// rustc's optimizer collapses `buf[i] = 42.0f32` (where `buf`
    /// is `*mut f32`) into `i32.const <bit_pattern>; i32.store` on
    /// the WASM byte-array memory model. Without this, the lowerer
    /// would emit `Store { ty: F32, src: <reg holding 1109917696> }`
    /// and the backend codegens (Metal/SPIR-V/WGSL/MSL) would do an
    /// integer-to-float CONVERSION (1.1099e9) instead of a bitcast
    /// (42.0). Affects every constant float-buffer assignment that
    /// the optimizer reaches.
    fn materialize_for_typed_store(
        &mut self,
        v: SymVal,
        slot_ty: ScalarType,
    ) -> Result<Reg, LoweringError> {
        if let SymVal::I32Const(c) = v
            && matches!(slot_ty, ScalarType::F32 | ScalarType::F16)
        {
            let dst = self.alloc_reg();
            self.emit(KernelOp::Const {
                dst,
                value: ConstValue::F32(f32::from_bits(c as u32)),
            });
            return Ok(dst);
        }
        let (reg, _) = self.commit(v)?;
        Ok(reg)
    }

    /// Lower a narrow WASM load (`i32.load8_u`, `i32.load8_s`,
    /// `i32.load16_u`, `i32.load16_s`) to `Load + mask` (unsigned)
    /// or `Load + (x << k) >> k` (signed). The slot is read at full
    /// element width; the narrowing is recovered by bitwise ops on
    /// the loaded register. `width_bits` is 8 or 16; `signed` controls
    /// whether to sign-extend.
    fn narrow_load(&mut self, width_bits: u32, signed: bool) -> Result<(), LoweringError> {
        let addr = self.pop()?;
        let (slot, base) = match addr {
            SymVal::BufferAccess { slot, base, .. } => (slot, base),
            SymVal::BufferPtr(slot) => {
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::U32(0),
                });
                (slot, zero)
            }
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("narrow i32.load on non-buffer address {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        let slot_ty = self.scalar_type_for_slot(slot);
        let elem_reg = self.alloc_reg();
        self.emit(KernelOp::Load {
            dst: elem_reg,
            field: slot,
            index: base,
            ty: slot_ty,
        });
        let mask_val: u32 = (1u32 << width_bits) - 1;
        let mask_reg = self.alloc_reg();
        self.emit(KernelOp::Const {
            dst: mask_reg,
            value: ConstValue::U32(mask_val),
        });
        let masked = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst: masked,
            a: elem_reg,
            b: mask_reg,
            op: BinOp::BitAnd,
            ty: ScalarType::U32,
        });
        if !signed {
            self.stack.push(SymVal::Reg(masked, ScalarType::U32));
            return Ok(());
        }
        // Signed: sign-extend by `(x << (32 - width)) >> (32 - width)`.
        let shift = 32 - width_bits;
        let shift_reg = self.alloc_reg();
        self.emit(KernelOp::Const {
            dst: shift_reg,
            value: ConstValue::U32(shift),
        });
        let shifted_left = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst: shifted_left,
            a: masked,
            b: shift_reg,
            op: BinOp::Shl,
            ty: ScalarType::I32,
        });
        let final_reg = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst: final_reg,
            a: shifted_left,
            b: shift_reg,
            op: BinOp::Shr,
            ty: ScalarType::I32,
        });
        self.stack.push(SymVal::Reg(final_reg, ScalarType::I32));
        Ok(())
    }

    /// Lower a narrow WASM store (`i32.store8`, `i32.store16`) to
    /// `Store` with the source masked to its low `width_bits`.
    fn narrow_store(&mut self, width_bits: u32) -> Result<(), LoweringError> {
        let val = self.pop()?;
        let addr = self.pop()?;
        let (val_reg, _) = self.commit(val)?;
        let (slot, base) = match addr {
            SymVal::BufferAccess { slot, base, .. } => (slot, base),
            SymVal::BufferPtr(slot) => {
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::U32(0),
                });
                (slot, zero)
            }
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("narrow i32.store on non-buffer address {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        let mask_val: u32 = (1u32 << width_bits) - 1;
        let mask_reg = self.alloc_reg();
        self.emit(KernelOp::Const {
            dst: mask_reg,
            value: ConstValue::U32(mask_val),
        });
        let masked = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst: masked,
            a: val_reg,
            b: mask_reg,
            op: BinOp::BitAnd,
            ty: ScalarType::U32,
        });
        let slot_ty = self.scalar_type_for_slot(slot);
        self.emit(KernelOp::Store {
            field: slot,
            index: base,
            src: masked,
            ty: slot_ty,
        });
        Ok(())
    }

    /// Lower a narrow i64 load (`i64.load{8,16,32}_{u,s}`) — load
    /// the slot's full element, mask to the requested `width_bits`,
    /// optionally sign-extend, then Cast to U64 / I64. Mirrors the
    /// i32 `narrow_load` helper but with a 64-bit result. Accepts
    /// any BufferAccess scale (the slot's element type from the
    /// side-table determines the per-element width; the mask
    /// narrows further as the user requested).
    fn narrow_load_widen_i64(
        &mut self,
        width_bits: u32,
        signed: bool,
    ) -> Result<(), LoweringError> {
        let addr = self.pop()?;
        let (slot, base) = match addr {
            SymVal::BufferAccess { slot, base, .. } => (slot, base),
            SymVal::BufferPtr(slot) => {
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::U32(0),
                });
                (slot, zero)
            }
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("narrow i64 load on non-buffer address {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        let slot_ty = self.scalar_type_for_slot(slot);
        let loaded = self.alloc_reg();
        self.emit(KernelOp::Load {
            dst: loaded,
            field: slot,
            index: base,
            ty: slot_ty,
        });
        // Mask to the requested width in u32-space. For width=32
        // the mask is 0xFFFFFFFF (all ones), effectively a no-op
        // but emitted uniformly for code-simplicity.
        let mask_val: u32 = if width_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << width_bits) - 1
        };
        let mask_reg = self.alloc_reg();
        self.emit(KernelOp::Const {
            dst: mask_reg,
            value: ConstValue::U32(mask_val),
        });
        let masked = self.alloc_reg();
        self.emit(KernelOp::BinOp {
            dst: masked,
            a: loaded,
            b: mask_reg,
            op: BinOp::BitAnd,
            ty: ScalarType::U32,
        });
        let mut narrowed = masked;
        if signed && width_bits < 32 {
            // Sign-extend within u32: `(x << (32 - width)) >> (32 - width)`
            // arithmetic shift. The Cast to I64 below picks up the
            // sign-extended u32 as i32 and widens.
            let shift = 32 - width_bits;
            let shift_reg = self.alloc_reg();
            self.emit(KernelOp::Const {
                dst: shift_reg,
                value: ConstValue::U32(shift),
            });
            let shifted_left = self.alloc_reg();
            self.emit(KernelOp::BinOp {
                dst: shifted_left,
                a: masked,
                b: shift_reg,
                op: BinOp::Shl,
                ty: ScalarType::I32,
            });
            let signed_reg = self.alloc_reg();
            self.emit(KernelOp::BinOp {
                dst: signed_reg,
                a: shifted_left,
                b: shift_reg,
                op: BinOp::Shr,
                ty: ScalarType::I32,
            });
            narrowed = signed_reg;
        }
        // Widen to 64-bit via Cast.
        let (from_ty, to_ty) = if signed {
            (ScalarType::I32, ScalarType::I64)
        } else {
            (ScalarType::U32, ScalarType::U64)
        };
        let widened = self.alloc_reg();
        self.emit(KernelOp::Cast {
            dst: widened,
            src: narrowed,
            from: from_ty,
            to: to_ty,
        });
        self.stack.push(SymVal::Reg(widened, to_ty));
        Ok(())
    }

    /// Lower a narrow i64 store (`i64.store{8,16,32}`) — truncate
    /// the u64 value to its low `width_bits`, then store at the
    /// slot's full element width (slot's element type from the
    /// side-table). The mask narrows the source as the user
    /// requested; the Store itself writes whatever fits the slot.
    fn narrow_store_truncate_i64(&mut self, width_bits: u32) -> Result<(), LoweringError> {
        let val = self.pop()?;
        let addr = self.pop()?;
        let (val_reg, _) = self.commit(val)?;
        let (slot, base) = match addr {
            SymVal::BufferAccess { slot, base, .. } => (slot, base),
            SymVal::BufferPtr(slot) => {
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::U32(0),
                });
                (slot, zero)
            }
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("narrow i64 store on non-buffer address {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        // Truncate u64 → u32 via Cast (drops high word).
        let truncated = self.alloc_reg();
        self.emit(KernelOp::Cast {
            dst: truncated,
            src: val_reg,
            from: ScalarType::U64,
            to: ScalarType::U32,
        });
        // Mask to the requested width. For width=32 the mask is
        // all-ones — skip the BitAnd in that case for slightly
        // cleaner IR.
        let narrowed = if width_bits >= 32 {
            truncated
        } else {
            let mask_val: u32 = (1u32 << width_bits) - 1;
            let mask_reg = self.alloc_reg();
            self.emit(KernelOp::Const {
                dst: mask_reg,
                value: ConstValue::U32(mask_val),
            });
            let masked = self.alloc_reg();
            self.emit(KernelOp::BinOp {
                dst: masked,
                a: truncated,
                b: mask_reg,
                op: BinOp::BitAnd,
                ty: ScalarType::U32,
            });
            masked
        };
        let slot_ty = self.scalar_type_for_slot(slot);
        self.emit(KernelOp::Store {
            field: slot,
            index: base,
            src: narrowed,
            ty: slot_ty,
        });
        Ok(())
    }

    /// Try to inline a same-crate defined-function call. Looks up the
    /// callee's body, sets up its params + locals on top of the
    /// caller's `self.locals`, then walks the callee's instruction
    /// stream — with `LocalGet`/`Set`/`Tee` indices rewritten to point
    /// at the appended slots — through the existing `lower_instr`
    /// machinery.
    ///
    /// Returns `Ok(true)` on success. Returns `Ok(false)` if the
    /// callee can't be inlined under this v1 (contains `Return`, uses
    /// structured control flow, etc.) — caller should fall back to
    /// the unsupported-call error.
    ///
    /// Accepts structured control flow inside the callee body
    /// (`Block`/`Loop`/`If`/`Else`/`Br`/`BrIf`/`Return`) by wrapping
    /// the inlined body in a synthetic outer `Block` frame.
    ///
    /// The wrap serves two purposes:
    ///   1. It isolates the callee's branch targets from the caller's
    ///      frame stack. A callee `Br(N)` targeting its own function
    ///      level lands on the wrap, not on whatever frame the caller
    ///      happened to be in at the call site.
    ///   2. It absorbs the callee's terminal `End`. The function-end
    ///      `End` rustc always emits closes the wrap and splices the
    ///      collected ops into the caller's current frame.
    ///
    /// `Return` already lowers to "emit nothing if inside a block,
    /// not crossing a loop" (see the `Return` arm in `lower_instr`).
    /// With the wrap in place, that fall-through reaches the wrap's
    /// `End` cleanly. Trailing `unreachable` / panic helpers after a
    /// `return` are already silenced by the existing `Unreachable`
    /// no-op + `is_panic_helper` elision.
    fn try_inline_defined_call(&mut self, callee_idx: u32) -> Result<bool, LoweringError> {
        // Look up callee body + signature.
        let callee = match self.module.functions.get(callee_idx as usize) {
            Some(f) => f,
            None => return Ok(false),
        };
        let callee_body = match &callee.kind {
            FunctionKind::Defined(b) => b.clone(),
            FunctionKind::Imported { .. } => return Ok(false),
        };
        let callee_sig = match self.module.types.get(callee.type_index as usize) {
            Some(s) => s.clone(),
            None => return Ok(false),
        };

        // Reserve callee-local slot space, append to self.locals.
        let base_offset = self.locals.len();
        let arity = callee_sig.params.len();
        // Params first (with the right wasm_ty / stable_ty for each).
        for ty in &callee_sig.params {
            self.locals.push(LocalInfo {
                wasm_ty: *ty,
                val: None,
                stable_reg: None,
                stable_ty: scalar_type_for_wasm_ty(*ty),
            });
        }
        // Declared locals after params.
        for (count, ty) in &callee_body.locals {
            for _ in 0..*count {
                self.locals.push(LocalInfo {
                    wasm_ty: *ty,
                    val: None,
                    stable_reg: None,
                    stable_ty: scalar_type_for_wasm_ty(*ty),
                });
            }
        }

        // Pop args off the operand stack in reverse, then assign in
        // declaration order to the freshly appended param slots. WASM
        // call ABI: the last arg is on top of the stack.
        let mut args: Vec<SymVal> = Vec::with_capacity(arity);
        for _ in 0..arity {
            args.push(self.pop()?);
        }
        args.reverse();
        for (i, arg) in args.into_iter().enumerate() {
            let slot = base_offset + i;
            if is_value_symval(&arg) {
                self.ensure_stable_reg_for(slot, &arg);
                self.write_local_via_copy(slot, arg)?;
            } else {
                self.locals[slot].val = Some(arg);
            }
        }

        // Initialise declared-local stable regs to default-zero, same
        // as the function-entry pass does for the top-level kernel.
        let decl_locals_start = base_offset + arity;
        for i in decl_locals_start..self.locals.len() {
            let ty = self.locals[i].stable_ty;
            let dst = self.alloc_reg();
            let init = match ty {
                ScalarType::F16
                | ScalarType::BF16
                | ScalarType::FP8E5M2
                | ScalarType::FP8E4M3
                | ScalarType::F32 => ConstValue::F32(0.0),
                ScalarType::F64 => ConstValue::F64(0.0),
                ScalarType::Bool => ConstValue::Bool(false),
                ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I4 => {
                    ConstValue::I32(0)
                }
                ScalarType::I64 => ConstValue::I64(0),
                ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => ConstValue::U32(0),
                ScalarType::U64 => ConstValue::U64(0),
            };
            self.emit(KernelOp::Const { dst, value: init });
            self.locals[i].stable_reg = Some(dst);
            self.locals[i].val = Some(SymVal::Reg(dst, ty));
        }

        // Push the synthetic wrapping Block. The callee's terminal End
        // closes this frame and splices ops into the caller's sink.
        let parent_brifs = self.frames.last().map(|f| f.brifs.len()).unwrap_or(0);
        self.frames.push(Frame {
            kind: FrameKind::Block,
            ops: Vec::new(),
            local_snapshot: Vec::new(),
            brifs: Vec::new(),
            parent_brifs_at_open: parent_brifs,
        });

        // Walk the callee body, rewriting local indices by base_offset.
        // The terminating End is not skipped — it closes the wrap.
        let base = base_offset as u32;
        for instr in &callee_body.instructions {
            let rewritten = remap_locals(instr, base);
            self.lower_instr(&rewritten)?;
        }

        // Callee's return value (if any) is on top of self.stack now.
        Ok(true)
    }

    /// Lower an `atomic_<op>_<ty>(addr, val, order)` extern call into
    /// a `KernelOp::AtomicOp`. Args on the symbolic stack (top→bottom):
    /// order, val, addr. The order is a compile-time const; addr must
    /// be a `BufferAccess` (produced by `&mut buf[idx]`-style rewrites).
    fn atomic_rmw(&mut self, op: AtomicOp, ty: ScalarType) -> Result<(), LoweringError> {
        let order_sv = self.pop()?;
        let val_sv = self.pop()?;
        let addr_sv = self.pop()?;
        let order = order_const_to_enum(order_sv)?;
        let (val_reg, _) = self.commit(val_sv)?;
        let (field, index) = match addr_sv {
            SymVal::BufferAccess {
                slot,
                base,
                scale: _,
            } => (slot, base),
            // rustc's optimizer drops the offset arithmetic when the
            // index is a compile-time zero (`&mut buf[0]`), leaving
            // just the BufferPtr on the stack. Synthesize a Const 0
            // index so AtomicOp gets a real index Reg.
            SymVal::BufferPtr(slot) => {
                let zero = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: zero,
                    value: ConstValue::U32(0),
                });
                (slot, zero)
            }
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "atomic addr must be `&mut buf[i]` (BufferAccess) or `&mut buf[0]` (BufferPtr), got {other:?}"
                    ),
                    at: self.body.body_offset,
                });
            }
        };
        let dst = self.alloc_reg();
        self.emit(KernelOp::AtomicOp {
            dst,
            field,
            index,
            val: val_reg,
            op,
            ty,
            order,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower an `atomic_<op>_shared_<ty>(slot, index, val, order)`
    /// extern call into a `KernelOp::SharedAtomicOp`. Args on the
    /// symbolic stack (top→bottom): order, val, index, slot. The slot
    /// and order are compile-time consts; index and val are runtime
    /// registers.
    fn shared_atomic_rmw(&mut self, op: AtomicOp, ty: ScalarType) -> Result<(), LoweringError> {
        let order_sv = self.pop()?;
        let val_sv = self.pop()?;
        let index_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let order = order_const_to_enum(order_sv)?;
        let slot = match slot_sv {
            SymVal::I32Const(c) => c as u32,
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!(
                        "shared-atomic slot must be a compile-time constant, got {other:?}"
                    ),
                    at: self.body.body_offset,
                });
            }
        };
        let (index, _) = self.commit(index_sv)?;
        let (val_reg, _) = self.commit(val_sv)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::SharedAtomicOp {
            dst,
            slot,
            index,
            val: val_reg,
            op,
            ty,
            order,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower `texture_load_2d_<ty>(slot, x, y)` to `TextureLoad2D`.
    fn texture_load_2d(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let y_sv = self.pop()?;
        let x_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let texture = const_u32(slot_sv, "texture_load_2d slot")?;
        let (y, _) = self.commit(y_sv)?;
        let (x, _) = self.commit(x_sv)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::TextureLoad2D {
            dst,
            texture,
            x,
            y,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower `texture_sample_2d_<ty>(slot, x, y)` to `TextureSample2D`.
    fn texture_sample_2d(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let y_sv = self.pop()?;
        let x_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let texture = const_u32(slot_sv, "texture_sample_2d slot")?;
        let (y, _) = self.commit(y_sv)?;
        let (x, _) = self.commit(x_sv)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::TextureSample2D {
            dst,
            texture,
            x,
            y,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower `texture_load_3d_<ty>(slot, x, y, z)` to `TextureSample3D`.
    /// (3D unsampled load shares the same IR op as the sampled form;
    /// downstream emitters distinguish via texture kind.)
    fn texture_load_3d(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let z_sv = self.pop()?;
        let y_sv = self.pop()?;
        let x_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let texture = const_u32(slot_sv, "texture_load_3d slot")?;
        let (z, _) = self.commit(z_sv)?;
        let (y, _) = self.commit(y_sv)?;
        let (x, _) = self.commit(x_sv)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower `texture_write_2d_<ty>(slot, x, y, val)` to `TextureWrite2D`.
    /// Returns no value (the call is `-> ()`); we don't push to the
    /// symbolic stack.
    fn texture_write_2d(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let val_sv = self.pop()?;
        let y_sv = self.pop()?;
        let x_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let texture = const_u32(slot_sv, "texture_write_2d slot")?;
        let (value, _) = self.commit(val_sv)?;
        let (y, _) = self.commit(y_sv)?;
        let (x, _) = self.commit(x_sv)?;
        self.emit(KernelOp::TextureWrite2D {
            texture,
            x,
            y,
            value,
            ty,
        });
        Ok(())
    }

    /// Lower a `memory_fence(order)` extern call into `KernelOp::Fence`.
    fn fence_call(&mut self) -> Result<(), LoweringError> {
        let order_sv = self.pop()?;
        let order = order_const_to_enum(order_sv)?;
        self.emit(KernelOp::Fence { order });
        Ok(())
    }

    /// Lower a `reduce_<op>_<ty>(value)` extern call into the
    /// matching `KernelOp::SubgroupReduce*` op.
    fn subgroup_reduce(&mut self, op: SubgroupOp, ty: ScalarType) -> Result<(), LoweringError> {
        let value = self.pop()?;
        let (vr, src_ty) = self.commit(value)?;
        let vr = self.coerce_to(vr, src_ty, ty);
        let dst = self.alloc_reg();
        let kop = match op {
            SubgroupOp::Add => KernelOp::SubgroupReduceAdd { dst, src: vr, ty },
            SubgroupOp::Min => KernelOp::SubgroupReduceMin { dst, src: vr, ty },
            SubgroupOp::Max => KernelOp::SubgroupReduceMax { dst, src: vr, ty },
        };
        self.emit(kop);
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower a `scan_add_<ty>(value)` extern call into a
    /// `KernelOp::SubgroupInclusiveAdd`.
    fn subgroup_scan_inclusive(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let value = self.pop()?;
        let (vr, src_ty) = self.commit(value)?;
        let vr = self.coerce_to(vr, src_ty, ty);
        let dst = self.alloc_reg();
        self.emit(KernelOp::SubgroupInclusiveAdd { dst, src: vr, ty });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower a `scan_add_exclusive_<ty>(value)` extern call into a
    /// `KernelOp::SubgroupExclusiveAdd`.
    fn subgroup_scan_exclusive(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let value = self.pop()?;
        let (vr, src_ty) = self.commit(value)?;
        let vr = self.coerce_to(vr, src_ty, ty);
        let dst = self.alloc_reg();
        self.emit(KernelOp::SubgroupExclusiveAdd { dst, src: vr, ty });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower a `shuffle_<ty>(value, src_lane)` extern call into a
    /// `KernelOp::WaveShuffle`.
    fn wave_shuffle(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let src_lane = self.pop()?;
        let value = self.pop()?;
        let (vr, _) = self.commit(value)?;
        let (lr, _) = self.commit(src_lane)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::WaveShuffle {
            dst,
            src: vr,
            lane_delta: lr,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower a `shared_load_<ty>(slot, index)` extern call into a
    /// `KernelOp::SharedLoad`. The slot must be a compile-time
    /// constant (the IR carries it as `id: u32`); the index is a
    /// runtime register.
    fn shared_load(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let index_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let id = match slot_sv {
            SymVal::I32Const(c) => c as u32,
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("shared_load slot must be a compile-time constant, got {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        let (idx_reg, _) = self.commit(index_sv)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::SharedLoad {
            dst,
            id,
            index: idx_reg,
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Lower a `shared_store_<ty>(slot, index, val)` extern call into
    /// a `KernelOp::SharedStore`. Slot is compile-time-constant.
    fn shared_store(&mut self, ty: ScalarType) -> Result<(), LoweringError> {
        let val_sv = self.pop()?;
        let index_sv = self.pop()?;
        let slot_sv = self.pop()?;
        let id = match slot_sv {
            SymVal::I32Const(c) => c as u32,
            other => {
                return Err(LoweringError::UnsupportedOp {
                    op: format!("shared_store slot must be a compile-time constant, got {other:?}"),
                    at: self.body.body_offset,
                });
            }
        };
        let (idx_reg, _) = self.commit(index_sv)?;
        let (val_reg, _) = self.commit(val_sv)?;
        self.emit(KernelOp::SharedStore {
            id,
            index: idx_reg,
            src: val_reg,
            ty,
        });
        Ok(())
    }

    fn math_call_ternary(&mut self, func: MathFn) -> Result<(), LoweringError> {
        let c = self.pop()?;
        let b = self.pop()?;
        let a = self.pop()?;
        let (ar, ty) = self.commit(a)?;
        let (br, _) = self.commit(b)?;
        let (cr, _) = self.commit(c)?;
        let dst = self.alloc_reg();
        self.emit(KernelOp::MathCall {
            dst,
            func,
            args: vec![ar, br, cr],
            ty,
        });
        self.stack.push(SymVal::Reg(dst, ty));
        Ok(())
    }

    /// Pop a stack value, erroring if the stack is empty.
    fn pop(&mut self) -> Result<SymVal, LoweringError> {
        self.stack
            .pop()
            .ok_or_else(|| LoweringError::ShapeMismatch("stack underflow".into()))
    }

    /// Commit a `SymVal` to a real register. For non-Reg variants
    /// we materialize a Const or carry a placeholder. Returns
    /// `(reg, scalar_type)`.
    fn commit(&mut self, v: SymVal) -> Result<(Reg, ScalarType), LoweringError> {
        match v {
            SymVal::Reg(r, ty) | SymVal::Opaque(r, ty) => Ok((r, ty)),
            SymVal::I32Const(c) => {
                let dst = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst,
                    value: ConstValue::I32(c),
                });
                Ok((dst, ScalarType::I32))
            }
            SymVal::I64Const(c) => {
                let dst = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst,
                    value: ConstValue::I64(c),
                });
                Ok((dst, ScalarType::I64))
            }
            // ScaledIdx represents `base << log2(scale)` — used as a
            // byte offset by the buffer-load/store pattern recognizer.
            // When it surfaces in non-buffer arithmetic (e.g. rustc's
            // optimizer does its own pointer-arith hoisting), we have
            // to materialize the shift back into a real Reg.
            SymVal::ScaledIdx { base, scale } => {
                let log2 = scale.trailing_zeros();
                let shift_amt = self.alloc_reg();
                self.emit(KernelOp::Const {
                    dst: shift_amt,
                    value: ConstValue::U32(log2),
                });
                let dst = self.alloc_reg();
                self.emit(KernelOp::BinOp {
                    dst,
                    a: base,
                    b: shift_amt,
                    op: BinOp::Shl,
                    ty: ScalarType::U32,
                });
                Ok((dst, ScalarType::U32))
            }
            // BufferAccess flowing into a register-typed context
            // (intrinsic call, BinOp, branch condition, etc.) is
            // a buffer Load through implicit materialization.
            // rustc emits this pattern when a let-binding stores
            // a buffer element and a later use crosses control
            // flow:
            //   let x = buf[i];  // BufferAccess
            //   if cond { foo(x) }  // commit fires here
            //
            // Auto-emit `KernelOp::Load { dst, field: slot,
            // index: base, ty: slot_ty }` and return the fresh
            // register. The buffer's scalar type comes from the
            // side table.
            SymVal::BufferAccess { slot, base, .. } => {
                let ty = self.scalar_type_for_slot(slot);
                let dst = self.alloc_reg();
                self.emit(KernelOp::Load {
                    dst,
                    field: slot,
                    index: base,
                    ty,
                });
                Ok((dst, ty))
            }
            SymVal::BufferPtr(_) => Err(LoweringError::UnsupportedOp {
                op: "cannot commit pointer/address SymVal to a register — \
                     buffer pointer arithmetic not yet supported"
                    .into(),
                at: self.body.body_offset,
            }),
        }
    }

    fn into_kernel_def(mut self) -> KernelDef {
        // The function-level frame holds the final ops list.
        let func_frame = self
            .frames
            .pop()
            .expect("function-level frame must be present at end of lowering");
        debug_assert!(
            self.frames.is_empty(),
            "frame stack should be empty after function-level pop"
        );
        let body_ops = func_frame.ops;
        // Build params from the side table.
        let params = self
            .side_table
            .params
            .iter()
            .map(|s| match s.kind {
                ParamKind::BufferRead => KernelParam::FieldRead {
                    name: format!("buf{}", s.slot),
                    slot: s.slot,
                    scalar_type: s.scalar,
                },
                ParamKind::BufferWrite => KernelParam::FieldWrite {
                    name: format!("buf{}", s.slot),
                    slot: s.slot,
                    scalar_type: s.scalar,
                },
                ParamKind::Scalar => KernelParam::Constant {
                    name: format!("s{}", s.slot),
                    slot: s.slot,
                    scalar_type: s.scalar,
                },
            })
            .collect();

        KernelDef {
            name: self.side_table.kernel_name.clone(),
            params,
            body: body_ops,
            body_source: None,
            next_reg: self.next_reg,
            opt_level: 3,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: self.side_table.workgroup_size,
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }
}

/// Pop a SymVal that must be a compile-time `i32.const` and return
/// its u32 value. Used by texture / shared-mem call lowerings where
/// the slot must lift into a static field of the IR op.
fn const_u32(v: SymVal, ctx: &'static str) -> Result<u32, LoweringError> {
    match v {
        SymVal::I32Const(c) => Ok(c as u32),
        other => Err(LoweringError::ShapeMismatch(format!(
            "{ctx} must be a compile-time constant, got {other:?}"
        ))),
    }
}

/// Map a compile-time `i32.const` order argument to the IR's
/// `MemoryOrder` enum. Atomics + fences accept the order as a u32
/// arg whose value mirrors the `ORDER_*` consts in
/// `quanta::intrinsics`.
fn order_const_to_enum(v: SymVal) -> Result<MemoryOrder, LoweringError> {
    match v {
        SymVal::I32Const(c) => match c {
            0 => Ok(MemoryOrder::Relaxed),
            1 => Ok(MemoryOrder::Acquire),
            2 => Ok(MemoryOrder::Release),
            3 => Ok(MemoryOrder::AcqRel),
            4 => Ok(MemoryOrder::SeqCst),
            other => Err(LoweringError::ShapeMismatch(format!(
                "memory-order arg out of range: {other}"
            ))),
        },
        other => Err(LoweringError::ShapeMismatch(format!(
            "memory-order arg must be a compile-time constant, got {other:?}"
        ))),
    }
}

/// True if a SymVal carries a scalar value (committable to a Reg via
/// `commit`). Buffer pointers and address-arithmetic SymVals don't —
/// they're consumed by the load/store pattern recognizer instead.
fn is_value_symval(v: &SymVal) -> bool {
    matches!(
        v,
        SymVal::Reg(..) | SymVal::Opaque(..) | SymVal::I32Const(..) | SymVal::I64Const(..)
    )
}

/// Map a raw WASM value type to the closest Quanta IR scalar type.
/// WASM only carries i32/i64/f32/f64 — signed/unsigned and narrower
/// widths (u8/u16/etc.) are erased by rustc's wasm32 backend. We pick
/// `U32`/`U64` for integer locals because that matches how rustc emits
/// most pointer-arithmetic and unsigned-by-default kernels; for typed
/// per-slot reads, `scalar_type_for_slot` overrides per the side table.
fn scalar_type_for_wasm_ty(ty: WasmTy) -> ScalarType {
    match ty {
        WasmTy::I32 => ScalarType::U32,
        WasmTy::I64 => ScalarType::U64,
        WasmTy::F32 => ScalarType::F32,
        WasmTy::F64 => ScalarType::F64,
    }
}

/// Recognize rustc's panic helpers by mangled-name prefix. Covers
/// the whole panic family (panic_const_*, panic_fmt,
/// panic_bounds_check, …) under both mangling schemes:
///   - Itanium / legacy: `_ZN4core9panicking…`
///   - v0 (rustc 1.59+, default on stable since 1.95): `_RNv…` with
///     `4core9panicking` somewhere in the body. rustc 1.95 ships
///     v0 mangling on by default for wasm32, so missing this prefix
///     causes panic helpers to fall through to the inliner — which
///     then trips on `global.get $__stack_pointer` inside `panic_fmt`.
///
/// On `%`/`/` by zero rustc emits `panic_const_rem_by_zero` /
/// `panic_const_div_by_zero`, guarded by an `i32.eqz; if/br_if`
/// shape — the GPU contract is UB on zero-divide so this region is
/// dead at runtime, and the lowering pass elides the call + the
/// trailing `unreachable`.
fn is_panic_helper(name: &str) -> bool {
    name.starts_with("_ZN4core9panicking")
        || (name.starts_with("_RNv") && name.contains("4core9panicking"))
}

/// Rewrite a `RawInstr`'s local indices by `base_offset` so a callee
/// body can be spliced into a caller without local-index collisions.
/// All other instruction variants pass through unchanged.
fn remap_locals(instr: &RawInstr, base_offset: u32) -> RawInstr {
    match instr {
        RawInstr::LocalGet(i) => RawInstr::LocalGet(i + base_offset),
        RawInstr::LocalSet(i) => RawInstr::LocalSet(i + base_offset),
        RawInstr::LocalTee(i) => RawInstr::LocalTee(i + base_offset),
        other => other.clone(),
    }
}

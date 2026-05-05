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

use quanta_ir::{BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType};

use crate::{
    FunctionBodyInfo, FunctionKind, LoweringError, Module, ParamKind, RawInstr, SideTable, WasmTy,
    find_kernel,
};

/// Symbolic stack value during lowering.
#[derive(Debug, Clone, Copy)]
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
}

/// One control-flow frame on the lowering stack.
struct Frame {
    kind: FrameKind,
    ops: Vec<KernelOp>,
    /// Path through nested `Branch.else_ops` where the *next* op
    /// emitted to this frame should land. Empty = push directly to
    /// `ops`. Each entry is the index of a `KernelOp::Branch` whose
    /// `else_ops` is the next descent step.
    ///
    /// Used to model WASM `br`/`br_if` to non-Loop targets without a
    /// labeled-break primitive in Quanta IR. When `br_if cond N`
    /// targets a Block frame, we push `Branch { cond, then_ops: [],
    /// else_ops: [] }` to that frame's current sink and append its
    /// index to the redirect path. From then on, every op emitted to
    /// that frame (and every inner frame that closes into it) flows
    /// into the Branch's `else_ops` — i.e. runs only when cond is
    /// false. Nests: a second br_if to the same frame chains another
    /// Branch inside the first's `else_ops`.
    redirect: Vec<usize>,
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
#[derive(Copy, Clone)]
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
                redirect: Vec::new(),
            }],
            next_reg: 0,
            intrinsic_names,
        }
    }

    fn alloc_reg(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Append an op to the current (topmost) frame, honoring its
    /// redirect chain. See `Frame::redirect`.
    fn emit(&mut self, op: KernelOp) {
        let top = self.frames.len() - 1;
        Self::sink_at_mut(&mut self.frames[top]).push(op);
    }

    /// Append a sequence of ops to a specific frame's current sink
    /// (honoring its redirect chain). Used when an inner frame closes
    /// and its accumulated ops splice into the parent's sink — the
    /// parent might be in a redirect after a br_if.
    fn splice_into_frame(target: &mut Frame, ops: impl IntoIterator<Item = KernelOp>) {
        let sink = Self::sink_at_mut(target);
        for op in ops {
            sink.push(op);
        }
    }

    /// Resolve a frame's current sink: walks `redirect` indices into
    /// nested `Branch.else_ops`. Empty redirect = `frame.ops`.
    fn sink_at_mut(frame: &mut Frame) -> &mut Vec<KernelOp> {
        let mut sink = &mut frame.ops;
        for &idx in &frame.redirect {
            match &mut sink[idx] {
                KernelOp::Branch { else_ops, .. } => {
                    sink = else_ops;
                }
                other => {
                    panic!("redirect chain pointed at non-Branch op: {other:?} (lowering bug)")
                }
            }
        }
        sink
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
    /// branch crosses a loop boundary — if so, the redirect-chain
    /// approach would leave a register referenced outside the loop
    /// body it was defined in. Falls back to a `Break` (or
    /// conditional Break) instead.
    fn has_loop_between_top_and_depth(&self, depth: u32) -> bool {
        let top = self.frames.len();
        let target_idx = match top.checked_sub(1 + depth as usize) {
            Some(i) => i,
            None => return false,
        };
        // Frames strictly above target, up to and including top-1.
        for f in &self.frames[target_idx + 1..] {
            if matches!(f.kind, FrameKind::Loop { .. }) {
                return true;
            }
        }
        false
    }

    /// Materialize a boolean constant as a `KernelOp::Const` written
    /// into the active sink of the frame at `depth`. Used by `Br` to
    /// install a redirect with cond=true.
    fn emit_const_bool_to_target(&mut self, depth: u32, dst: Reg, value: bool) {
        let target = self
            .frame_at_depth_mut(depth)
            .expect("caller must verify target depth before emit_const_bool_to_target");
        Self::sink_at_mut(target).push(KernelOp::Const {
            dst,
            value: ConstValue::Bool(value),
        });
    }

    /// Install a redirect on the frame at `depth`: append a
    /// `Branch { cond, then_ops: [], else_ops: [] }` to that frame's
    /// active sink and extend its redirect chain to point at the new
    /// Branch's `else_ops`. Subsequent ops emitted to that frame —
    /// and inner frames closing into it — flow into the Branch's
    /// `else_ops`, modeling "skip to end of target frame on cond".
    fn install_redirect_at(&mut self, depth: u32, cond: Reg) {
        let target = self
            .frame_at_depth_mut(depth)
            .expect("caller must verify target depth before install_redirect_at");
        let sink = Self::sink_at_mut(target);
        let new_idx = sink.len();
        sink.push(KernelOp::Branch {
            cond,
            then_ops: Vec::new(),
            else_ops: Vec::new(),
        });
        target.redirect.push(new_idx);
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
                ScalarType::F16 | ScalarType::F32 => ConstValue::F32(0.0),
                ScalarType::F64 => ConstValue::F64(0.0),
                ScalarType::Bool => ConstValue::Bool(false),
                ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                    ConstValue::I32(0)
                }
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

    /// Materialize a `local.set` / `local.tee` against the stable
    /// per-local register: emits `KernelOp::Copy { dst: stable_reg,
    /// src }` and updates `locals[idx].val` to point at the stable
    /// reg. Returns the stable reg + scalar type.
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
        self.emit(KernelOp::Copy {
            dst: stable_reg,
            src,
            ty: stable_ty,
        });
        self.locals[idx].val = Some(SymVal::Reg(stable_reg, stable_ty));
        Ok((stable_reg, stable_ty))
    }

    fn lower_instr(&mut self, instr: &RawInstr) -> Result<(), LoweringError> {
        match instr {
            RawInstr::LocalGet(idx) => {
                let val = self.locals[*idx as usize].val.ok_or_else(|| {
                    LoweringError::ShapeMismatch(format!("local.get {idx} on uninitialized local"))
                })?;
                self.stack.push(val);
            }
            RawInstr::LocalSet(idx) => {
                let v = self.pop()?;
                // Route value-typed SymVals (Reg/Opaque/Const) through
                // the stable register so post-merge reads see a defined
                // value. Buffer/address SymVals (BufferPtr/ScaledIdx/
                // BufferAccess) carry no scalar value to copy and keep
                // their existing symbolic binding — they're consumed
                // by load/store pattern recognition, not arithmetic.
                if self.locals[*idx as usize].stable_reg.is_some() && is_value_symval(&v) {
                    self.write_local_via_copy(*idx as usize, v)?;
                } else {
                    self.locals[*idx as usize].val = Some(v);
                }
            }
            RawInstr::LocalTee(idx) => {
                let v = self.stack.last().copied().ok_or_else(|| {
                    LoweringError::ShapeMismatch("local.tee on empty stack".into())
                })?;
                if self.locals[*idx as usize].stable_reg.is_some() && is_value_symval(&v) {
                    let _ = self.pop()?;
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

            RawInstr::F32Load { .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } => {
                        let dst = self.alloc_reg();
                        self.emit(KernelOp::Load {
                            dst,
                            field: slot,
                            index: base,
                            ty: ScalarType::F32,
                        });
                        self.stack.push(SymVal::Reg(dst, ScalarType::F32));
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("f32.load on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::F32Store { .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                let (val_reg, _) = self.commit(val)?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } => {
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty: ScalarType::F32,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("f32.store on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::I32Load { .. } => {
                let addr = self.pop()?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
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
                            op: format!("i32.load on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::I32Store { .. } => {
                let val = self.pop()?;
                let addr = self.pop()?;
                let (val_reg, _) = self.commit(val)?;
                match addr {
                    SymVal::BufferAccess {
                        slot,
                        base,
                        scale: 4,
                    } => {
                        let ty = self.scalar_type_for_slot(slot);
                        self.emit(KernelOp::Store {
                            field: slot,
                            index: base,
                            src: val_reg,
                            ty,
                        });
                    }
                    other => {
                        return Err(LoweringError::UnsupportedOp {
                            op: format!("i32.store on non-buffer address {other:?}"),
                            at: self.body.body_offset,
                        });
                    }
                }
            }

            RawInstr::F32Add => self.bin_op_float(BinOp::Add)?,
            RawInstr::F32Sub => self.bin_op_float(BinOp::Sub)?,
            RawInstr::F32Mul => self.bin_op_float(BinOp::Mul)?,
            RawInstr::F32Div => self.bin_op_float(BinOp::Div)?,

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
                        } else {
                            let detail = match resolved {
                                Some(name) => format!(
                                    "call to defined function {idx} (`{name}`) — \
                                     inlining not yet supported; rewrite the \
                                     kernel to avoid this call or expose the \
                                     callee as a Quanta intrinsic"
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
                self.frames.push(Frame {
                    kind: FrameKind::Block,
                    ops: Vec::new(),
                    redirect: Vec::new(),
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
                self.frames.push(Frame {
                    kind: FrameKind::Loop {
                        count_reg,
                        iter_reg,
                    },
                    ops: Vec::new(),
                    redirect: Vec::new(),
                });
            }

            RawInstr::If { .. } => {
                let cond_sv = self.pop()?;
                let (cond, _) = self.commit(cond_sv)?;
                self.frames.push(Frame {
                    kind: FrameKind::If { cond },
                    ops: Vec::new(),
                    redirect: Vec::new(),
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
                self.frames.push(Frame {
                    kind: FrameKind::Else {
                        cond,
                        then_ops: frame.ops,
                    },
                    ops: Vec::new(),
                    redirect: Vec::new(),
                });
            }

            RawInstr::End => {
                let frame = self.frames.pop().ok_or_else(|| {
                    LoweringError::ShapeMismatch("End with empty frame stack".into())
                })?;
                match frame.kind {
                    FrameKind::Function => {
                        // Function-level End — done. Push back onto
                        // the stack so into_kernel_def can read it.
                        self.frames.push(Frame {
                            kind: FrameKind::Function,
                            ops: frame.ops,
                            redirect: Vec::new(),
                        });
                    }
                    FrameKind::Block => {
                        // Block was a label scope — splice ops into the
                        // parent's *active sink* (honors any redirect on
                        // the parent set by a prior br/br_if).
                        let parent_idx = self.frames.len() - 1;
                        Self::splice_into_frame(&mut self.frames[parent_idx], frame.ops);
                    }
                    FrameKind::Loop {
                        count_reg,
                        iter_reg,
                    } => {
                        self.emit(KernelOp::Loop {
                            count: count_reg,
                            iter_reg,
                            body: frame.ops,
                        });
                    }
                    FrameKind::If { cond } => {
                        self.emit(KernelOp::Branch {
                            cond,
                            then_ops: frame.ops,
                            else_ops: Vec::new(),
                        });
                    }
                    FrameKind::Else { cond, then_ops } => {
                        self.emit(KernelOp::Branch {
                            cond,
                            then_ops,
                            else_ops: frame.ops,
                        });
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
                    self.emit(KernelOp::Break);
                    return Ok(());
                }
                let dst = self.alloc_reg();
                self.emit_const_bool_to_target(*depth, dst, true);
                self.install_redirect_at(*depth, dst);
            }

            RawInstr::BrIf(depth) => {
                let cond_sv = self.pop()?;
                let (cond, _) = self.commit(cond_sv)?;
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
                    self.emit(KernelOp::Branch {
                        cond,
                        then_ops: vec![KernelOp::Break],
                        else_ops: Vec::new(),
                    });
                    return Ok(());
                }
                self.install_redirect_at(*depth, cond);
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

            RawInstr::I32And => self.bin_op_int(BinOp::BitAnd)?,
            RawInstr::I32Or => self.bin_op_int(BinOp::BitOr)?,
            RawInstr::I32Xor => self.bin_op_int(BinOp::BitXor)?,
            RawInstr::I32Sub => self.bin_op_int(BinOp::Sub)?,
            RawInstr::I32Mul => self.bin_op_int(BinOp::Mul)?,
            RawInstr::I32DivU | RawInstr::I32DivS => self.bin_op_int(BinOp::Div)?,
            RawInstr::I32RemU | RawInstr::I32RemS => self.bin_op_int(BinOp::Rem)?,
            // Right-shift: unsigned and signed map to the same Quanta
            // BinOp::Shr; the slot's scalar type (set elsewhere)
            // determines the codegen-time arithmetic vs logical
            // distinction.
            RawInstr::I32ShrU | RawInstr::I32ShrS => self.bin_op_int(BinOp::Shr)?,

            RawInstr::I32LtU | RawInstr::I32LtS => self.cmp_op_int(quanta_ir::CmpOp::Lt)?,
            RawInstr::I32LeU | RawInstr::I32LeS => self.cmp_op_int(quanta_ir::CmpOp::Le)?,
            RawInstr::I32GtU | RawInstr::I32GtS => self.cmp_op_int(quanta_ir::CmpOp::Gt)?,
            RawInstr::I32GeU | RawInstr::I32GeS => self.cmp_op_int(quanta_ir::CmpOp::Ge)?,
            RawInstr::I32Eq => self.cmp_op_int(quanta_ir::CmpOp::Eq)?,
            RawInstr::I32Ne => self.cmp_op_int(quanta_ir::CmpOp::Ne)?,

            RawInstr::F32Lt => self.cmp_op_float(quanta_ir::CmpOp::Lt)?,
            RawInstr::F32Le => self.cmp_op_float(quanta_ir::CmpOp::Le)?,
            RawInstr::F32Gt => self.cmp_op_float(quanta_ir::CmpOp::Gt)?,
            RawInstr::F32Ge => self.cmp_op_float(quanta_ir::CmpOp::Ge)?,
            RawInstr::F32Eq => self.cmp_op_float(quanta_ir::CmpOp::Eq)?,
            RawInstr::F32Ne => self.cmp_op_float(quanta_ir::CmpOp::Ne)?,

            RawInstr::F32ConvertI32U => self.cast_op(ScalarType::U32, ScalarType::F32)?,
            RawInstr::F32ConvertI32S => self.cast_op(ScalarType::I32, ScalarType::F32)?,
            RawInstr::I32TruncF32U => self.cast_op(ScalarType::F32, ScalarType::U32)?,
            RawInstr::I32TruncF32S => self.cast_op(ScalarType::F32, ScalarType::I32)?,

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

            // `unreachable` follows an elided panic call as rustc's
            // dead-code marker; `nop` is a literal no-op. Both produce
            // no IR.
            RawInstr::Unreachable | RawInstr::Nop => {}

            // `return` is a function-level early exit. Quanta kernels
            // are all `() -> ()` so there's nothing to push; we model
            // it as a redirect on the Function frame (or a `Break` if
            // we're crossing a Loop boundary, since cond regs can't
            // escape a Loop body — same constraint as 5d.3).
            RawInstr::Return => {
                let depth = (self.frames.len() - 1) as u32;
                if self.has_loop_between_top_and_depth(depth) {
                    self.emit(KernelOp::Break);
                } else {
                    let dst = self.alloc_reg();
                    self.emit_const_bool_to_target(depth, dst, true);
                    self.install_redirect_at(depth, dst);
                }
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

    fn bin_op_float(&mut self, op: BinOp) -> Result<(), LoweringError> {
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
            ty: ScalarType::F32,
        });
        self.stack.push(SymVal::Reg(dst, ScalarType::F32));
        Ok(())
    }

    fn bin_op_int(&mut self, op: BinOp) -> Result<(), LoweringError> {
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
            ty: ScalarType::I32,
        });
        self.stack.push(SymVal::Reg(dst, ScalarType::I32));
        Ok(())
    }

    fn cmp_op_int(&mut self, op: quanta_ir::CmpOp) -> Result<(), LoweringError> {
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
            ty: ScalarType::I32,
        });
        self.stack.push(SymVal::Reg(dst, ScalarType::Bool));
        Ok(())
    }

    fn cmp_op_float(&mut self, op: quanta_ir::CmpOp) -> Result<(), LoweringError> {
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
            ty: ScalarType::F32,
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
            SymVal::BufferPtr(_) | SymVal::BufferAccess { .. } => {
                Err(LoweringError::UnsupportedOp {
                    op: "cannot commit pointer/address SymVal to a register — \
                     buffer pointer arithmetic not yet supported"
                        .into(),
                    at: self.body.body_offset,
                })
            }
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

/// True if a SymVal carries a scalar value (committable to a Reg via
/// `commit`). Buffer pointers and address-arithmetic SymVals don't —
/// they're consumed by the load/store pattern recognizer instead.
fn is_value_symval(v: &SymVal) -> bool {
    matches!(
        v,
        SymVal::Reg(..) | SymVal::Opaque(..) | SymVal::I32Const(..)
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

/// Recognize rustc's panic helpers by mangled-name prefix. The
/// `_ZN4core9panicking` Itanium prefix covers the whole panic family
/// (panic_const_*, panic_fmt, panic_bounds_check, …). On `%`/`/` by
/// zero rustc emits `panic_const_rem_by_zero` / `panic_const_div_by_zero`,
/// guarded by an `i32.eqz; if/br_if` shape — the GPU contract is UB on
/// zero-divide so this region is dead at runtime, and the lowering pass
/// elides the call + the trailing `unreachable`.
fn is_panic_helper(name: &str) -> bool {
    name.starts_with("_ZN4core9panicking")
}

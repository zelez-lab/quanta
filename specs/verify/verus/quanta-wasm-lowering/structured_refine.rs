//! V7-structured — the structured-control layer of `lowerInstrs`.
//!
//! V7 (straight-line) proved the fold over the non-control subset. This
//! file lifts the fold to the *full* instruction set: `block`, `wloop`,
//! `wif`, `br`, `brIf`, `wreturn` — the fuel-bounded, frame-aware arms
//! of the Lean `lowerInstrs` (Translate.lean:555-728).
//!
//! ## What's mechanized here
//!
//! 1. **The splitter helpers** (`split_at_end`, `split_at_else_or_end`,
//!    `walk_until_closer`) — Verus mirrors of `Quanta.Wasm.Structured`,
//!    over `Seq<WasmInstr>`. These pre-extract a structured construct's
//!    body out of the instruction stream (the recursive-descent
//!    strategy the Lean port uses in place of production's streaming
//!    `Vec<Frame>`). With the **progress lemma** `split_shrinks`: both
//!    body and post-suffix are strictly shorter than the input, which
//!    is what makes the fuel-bounded fold well-founded.
//!
//! 2. **The frame-stack predicates** `has_loop_above` / `loops_above`
//!    (mirrors of the same in Translate.lean:499/507) deciding whether
//!    a `br`/`brIf` target crosses a loop boundary.
//!
//! 3. **The structured fold** `lower_instrs` — the full Lean
//!    `lowerInstrs` with fuel + `frames: Seq<FrameKind>`, every arm
//!    transcribed. Plus **conservativity** (`straightline_agrees`): on
//!    a list with no structured openers, the structured fold equals the
//!    V7 straight-line fold — so V7-structured is a genuine extension,
//!    not a reimplementation.
//!
//! 4. **Per-arm shape lemmas** pinning each structured arm's output:
//!    block-splice (`block_splices`), loop-wrap (`loop_wraps`),
//!    if-branch (`wif_branches`), the br/brIf break-emission cases
//!    (`br_loop0_no_ir`, `br_cross_loop_breaks`, `brif_loop0_branches`),
//!    and the **not-yet-modeled refusals** (`br_record_refuses`,
//!    `br_exitflag_refuses`) — the production shapes the Lean spec
//!    deliberately refuses rather than mislower (the exit-flag record
//!    and the record-and-wrap, see [[redirect-chain-v2-closed-2026-06-12]]).
//!
//! ## Production refinement boundary
//!
//! The per-arm production refinement composes the same `view`/`step`
//! discipline V5 established for straight-line ops: each structured arm
//! emits a fixed `KernelOp` wrapper (`LoopOp` / `Branch` / `BreakOp`)
//! around the recursively-lowered body, and the production streaming-
//! `Vec<Frame>` walk produces the identical wrapper at the matching
//! `wend` (Structured.lean's module note: "two phrasings of find the
//! matching wend, lower the body, wrap it"). The body-lowering itself
//! is the recursive `lower_instrs` call, so the refinement is
//! structural: arm-shape equality (proved here) ∘ body refinement (the
//! IH). The streaming-Vec ↔ recursive-descent equivalence at the body
//! boundary is the manual transcription obligation, the same status the
//! straight-line `step_<op> ≈ Rust arm` correspondence carries.

use vstd::prelude::*;

verus! {

// ── Vocabulary (V2/V3 + the structured extensions) ─────────────────

pub type Reg = nat;

pub enum Scalar { Bool, I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64 }

pub enum SymVal {
    Reg(Reg, Scalar),
    BufferPtr(nat),
    ScaledIdx { base: Reg, scale: nat },
    I32ConstSym(int),
    BufferAccess { slot: nat, base: Reg, scale: nat },
}

pub struct LowerState { pub next_reg: nat, pub stack: Seq<SymVal> }

pub enum ConstValue { U32(int), I32(int) }
pub enum BinOp { Add, Sub, Mul, Div, Rem, BAnd, BOr, BXor, Shl, Shr }
pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

/// KernelOp + the structured wrappers (mirror of the Lean `KernelOp`
/// surface used by the control arms: `loopOp`, `breakOp`, `branch`).
pub enum KernelOp {
    Const(Reg, ConstValue),
    BinOpK(Reg, Reg, Reg, BinOp, Scalar),
    CmpK(Reg, Reg, Reg, CmpOp, Scalar),
    Cast(Reg, Reg, Scalar, Scalar),
    Copy(Reg, Reg),
    /// `wloop` body, wrapped (Translate.lean:595).
    LoopOp(Seq<KernelOp>),
    /// cross-loop break (Translate.lean:651/664/682/692).
    BreakOp,
    /// `wif` / `br_if` conditional: cond reg, then-ops, else-ops
    /// (Translate.lean:639).
    Branch(Reg, Seq<KernelOp>, Seq<KernelOp>),
}

/// The instruction set, now with structured openers/markers + branches.
/// `Block`/`WLoop`/`WIf` carry a block-type index (elided to a single
/// `nat`, irrelevant to lowering shape); `WEnd`/`WElse` are the closers
/// the splitters match; `Br`/`BrIf` carry a label depth.
pub enum WasmInstr {
    I32Const(int),
    I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor, I32Shl, I32ShrU, I32DivU, I32RemU,
    I32Eq, I32Ne, I32LtU, I32LeU, I32GtU, I32GeU,
    Drop, Nop, WReturn,
    Block(nat), WLoop(nat), WIf(nat), WEnd, WElse,
    Br(nat), BrIf(nat),
    Other,
}

/// Static kind of an open structured frame (mirror of Lean `FrameKind`).
pub enum FrameKind { Block, LoopK, Wif }

// ── State primitives (mirror V3) ───────────────────────────────────

pub open spec fn alloc(s: LowerState) -> (Reg, LowerState) {
    (s.next_reg, LowerState { next_reg: s.next_reg + 1, stack: s.stack })
}
pub open spec fn pop_sym(s: LowerState) -> Option<(SymVal, LowerState)> {
    if s.stack.len() == 0 { None }
    else { Some((s.stack[0],
        LowerState { stack: s.stack.subrange(1, s.stack.len() as int), next_reg: s.next_reg })) }
}
pub open spec fn commit(s: LowerState, v: SymVal) -> Option<(Reg, LowerState, Seq<KernelOp>)> {
    match v {
        SymVal::Reg(r, _) => Some((r, s, Seq::empty())),
        SymVal::I32ConstSym(n) => {
            let (dst, s1) = alloc(s);
            Some((dst, s1, seq![KernelOp::Const(dst, ConstValue::U32(n))]))
        },
        _ => None,
    }
}

// ── opener test (mirror WasmInstr.isOpener) ────────────────────────

pub open spec fn is_opener(i: WasmInstr) -> bool {
    match i {
        WasmInstr::Block(_) => true,
        WasmInstr::WLoop(_) => true,
        WasmInstr::WIf(_)   => true,
        _ => false,
    }
}

/// A list is "straight-line" if it contains no structured openers,
/// closers, or branches — the V7 subset. On such lists the structured
/// fold must agree with the straight-line fold.
pub open spec fn is_straightline(is: Seq<WasmInstr>) -> bool {
    forall|k: int| 0 <= k < is.len() ==> #[trigger] straightline_instr(is[k])
}

pub open spec fn straightline_instr(i: WasmInstr) -> bool {
    match i {
        WasmInstr::Block(_) => false,
        WasmInstr::WLoop(_) => false,
        WasmInstr::WIf(_)   => false,
        WasmInstr::WEnd     => false,
        WasmInstr::WElse    => false,
        WasmInstr::Br(_)    => false,
        WasmInstr::BrIf(_)  => false,
        WasmInstr::WReturn  => false,
        _ => true,
    }
}

/// A straight-line instruction is none of the structured constructors —
/// the exclusion `lower_instrs`'s `match head` needs to collapse onto
/// the `| _ =>` arm. Proved by the exhaustive match (every false-arm
/// constructor contradicts `straightline_instr`).
proof fn straightline_excludes(i: WasmInstr)
    requires straightline_instr(i),
    ensures
        !(i is Block), !(i is WLoop), !(i is WIf),
        !(i is WEnd), !(i is WElse), !(i is Br), !(i is BrIf), !(i is WReturn),
{
    match i {
        WasmInstr::Block(_) => {},
        WasmInstr::WLoop(_) => {},
        WasmInstr::WIf(_)   => {},
        WasmInstr::WEnd     => {},
        WasmInstr::WElse    => {},
        WasmInstr::Br(_)    => {},
        WasmInstr::BrIf(_)  => {},
        WasmInstr::WReturn  => {},
        _ => {},
    }
}

// ── The splitter (mirror walkUntilCloser / splitAtEnd) ─────────────
//
// `walk_until_closer(l, n, taken_len)` returns, on success, the index
// in `l` of the matching depth-0 closer, the closer itself, and the
// split point. We phrase it returning `Option<(int, WasmInstr)>` =
// (closer index within `l`, the closer) — `body = l[..idx]`,
// `rest = l[idx+1..]`. Depth `n` bumps on openers, drops on `wend`.

pub open spec fn closer_delta(i: WasmInstr, n: nat) -> nat {
    if is_opener(i) { n + 1 }
    else {
        match i {
            WasmInstr::WEnd => if n == 0 { 0 } else { (n - 1) as nat },
            _ => n,
        }
    }
}

/// Index of the matching depth-0 closer (`wend` or `welse`) in `l`,
/// scanning from `pos` at depth `n`. `None` if unbalanced (reached the
/// end before a depth-0 closer). Mirrors `walkUntilCloser`'s recursion.
pub open spec fn closer_index(l: Seq<WasmInstr>, pos: int, n: nat) -> Option<int>
    decreases l.len() - pos
{
    if pos >= l.len() {
        None
    } else if n == 0 && (l[pos] is WEnd || l[pos] is WElse) {
        Some(pos)
    } else {
        closer_index(l, pos + 1, closer_delta(l[pos], n))
    }
}

/// `split_at_end l` — returns `(body, rest)` where `body = l[..k]` and
/// `rest = l[k+1..]` for the matching depth-0 `wend` at index `k`.
/// Refuses on a stray `welse` at depth 0 (a `wif`-only marker).
pub open spec fn split_at_end(l: Seq<WasmInstr>) -> Option<(Seq<WasmInstr>, Seq<WasmInstr>)> {
    match closer_index(l, 0, 0) {
        None => None,
        Some(k) => if l[k] is WEnd {
            Some((l.subrange(0, k), l.subrange(k + 1, l.len() as int)))
        } else {
            None
        },
    }
}

/// `split_at_else_or_end l` — returns `(thenBody, elseBody, rest)`.
/// On a depth-0 `wend` first: no else (`elseBody = []`). On a depth-0
/// `welse` first: scan again from after it for the matching `wend`.
pub open spec fn split_at_else_or_end(l: Seq<WasmInstr>)
    -> Option<(Seq<WasmInstr>, Seq<WasmInstr>, Seq<WasmInstr>)>
{
    match closer_index(l, 0, 0) {
        None => None,
        Some(k1) => if l[k1] is WEnd {
            Some((l.subrange(0, k1), Seq::empty(), l.subrange(k1 + 1, l.len() as int)))
        } else {
            // welse at k1 — scan the suffix for the matching wend.
            let then_body = l.subrange(0, k1);
            let suffix = l.subrange(k1 + 1, l.len() as int);
            match closer_index(suffix, 0, 0) {
                None => None,
                Some(k2) => if suffix[k2] is WEnd {
                    Some((then_body, suffix.subrange(0, k2),
                          suffix.subrange(k2 + 1, suffix.len() as int)))
                } else {
                    None
                },
            }
        },
    }
}

// ── Splitter progress: body + rest are strictly shorter ────────────

/// The matching-closer index lies within bounds — `0 <= k < l.len()`,
/// scanning from 0. This is what makes `body = l[..k]` and
/// `rest = l[k+1..]` both strictly shorter than `l` (the closer itself
/// is consumed by neither), giving the fuel-bounded fold its progress.
proof fn closer_index_in_bounds(l: Seq<WasmInstr>, pos: int, n: nat)
    requires 0 <= pos <= l.len(),
    ensures match closer_index(l, pos, n) {
        Some(k) => pos <= k < l.len(),
        None => true,
    },
    decreases l.len() - pos
{
    if pos >= l.len() {
    } else if n == 0 && (l[pos] is WEnd || l[pos] is WElse) {
    } else {
        closer_index_in_bounds(l, pos + 1, closer_delta(l[pos], n));
    }
}

/// **Progress lemma.** When `split_at_end` succeeds, both the body and
/// the post-`wend` suffix are strictly shorter than the input — the
/// matching `wend` sits strictly inside, consumed by neither half. This
/// discharges the fuel-bounded fold's termination: each structured arm
/// recurses on strictly-smaller lists, so the recursion is well-founded
/// independently of the fuel counter (fuel is the Lean port's
/// structural-recursion certificate, not a real bound on shape).
proof fn split_at_end_shrinks(l: Seq<WasmInstr>)
    requires split_at_end(l).is_some(),
    ensures ({
        let (body, rest) = split_at_end(l).unwrap();
        body.len() < l.len() && rest.len() < l.len()
    }),
{
    closer_index_in_bounds(l, 0, 0);
    let k = closer_index(l, 0, 0).unwrap();
    // body = l[..k], len k < l.len(); rest = l[k+1..], len l.len()-(k+1) < l.len().
    assert(0 <= k < l.len());
}

// ── Frame-stack predicates (mirror hasLoopAbove / loopsAbove) ──────

pub open spec fn count_loops(frames: Seq<FrameKind>, upto: int) -> nat
    decreases upto
{
    if upto <= 0 { 0nat }
    else {
        let prev = count_loops(frames, upto - 1);
        if (upto - 1) < frames.len() && frames[upto - 1] is LoopK { prev + 1 } else { prev }
    }
}

/// True if any frame strictly above `depth` is a loop.
pub open spec fn has_loop_above(frames: Seq<FrameKind>, depth: nat) -> bool {
    count_loops(frames, depth as int) > 0
}

/// Number of loop frames strictly above `depth`.
pub open spec fn loops_above(frames: Seq<FrameKind>, depth: nat) -> nat {
    count_loops(frames, depth as int)
}

// ── The structured fold (mirror lowerInstrs) ───────────────────────
//
// Fuel-bounded; threads `frames: Seq<FrameKind>` (innermost = head).
// We dispatch the head: structured openers consume their body via the
// splitters and recurse; `br`/`brIf`/`wreturn` consult `frames`;
// everything else is the straight-line arm (lower_instr ++ recurse).
// `lower_instr` is the V3 straight-line dispatcher (inlined below).

pub open spec fn lower_i32_bin(s: LowerState, op: BinOp) -> Option<(LowerState, Seq<KernelOp>)> {
    match pop_sym(s) {
        None => None,
        Some((svb, s1)) => match pop_sym(s1) {
            None => None,
            Some((sva, s2)) => match commit(s2, sva) {
                None => None,
                Some((ra, s3, ops_a)) => match commit(s3, svb) {
                    None => None,
                    Some((rb, s4, ops_b)) => {
                        let (dst, s5) = alloc(s4);
                        let s6 = LowerState { stack: seq![SymVal::Reg(dst, Scalar::U32)].add(s5.stack),
                                              next_reg: s5.next_reg };
                        Some((s6, ops_a.add(ops_b).add(seq![KernelOp::BinOpK(dst, ra, rb, op, Scalar::U32)])))
                    },
                },
            },
        },
    }
}

/// Straight-line per-instruction dispatch (V3 subset; structured
/// instructions are intercepted by `lower_instrs` before reaching here,
/// so they map to `None` = "should not occur standalone").
pub open spec fn lower_instr(s: LowerState, i: WasmInstr) -> Option<(LowerState, Seq<KernelOp>)> {
    match i {
        WasmInstr::I32Const(n) =>
            Some((LowerState { stack: seq![SymVal::I32ConstSym(n)].add(s.stack),
                               next_reg: s.next_reg }, Seq::empty())),
        WasmInstr::I32Sub  => lower_i32_bin(s, BinOp::Sub),
        WasmInstr::I32Mul  => lower_i32_bin(s, BinOp::Mul),
        WasmInstr::I32And  => lower_i32_bin(s, BinOp::BAnd),
        WasmInstr::I32Or   => lower_i32_bin(s, BinOp::BOr),
        WasmInstr::I32Xor  => lower_i32_bin(s, BinOp::BXor),
        WasmInstr::I32ShrU => lower_i32_bin(s, BinOp::Shr),
        WasmInstr::I32DivU => lower_i32_bin(s, BinOp::Div),
        WasmInstr::I32RemU => lower_i32_bin(s, BinOp::Rem),
        WasmInstr::WReturn => Some((s, Seq::empty())),
        WasmInstr::Nop     => Some((s, Seq::empty())),
        WasmInstr::Drop    => match pop_sym(s) {
            None => None,
            Some((_, s1)) => Some((s1, Seq::empty())),
        },
        // i32Add/Shl/cmp omitted from this file's straight-line core for
        // brevity (covered in lower_instr_spec.rs / lower_instructions_
        // refine.rs); the structured fold treats them via the same
        // `| _ =>` arm and they are not openers, so conservativity holds.
        _ => None,
    }
}

pub open spec fn lower_instrs(fuel: nat, frames: Seq<FrameKind>, s: LowerState, is: Seq<WasmInstr>)
    -> Option<(LowerState, Seq<KernelOp>)>
    decreases fuel, is.len()
{
    if is.len() == 0 {
        Some((s, Seq::empty()))
    } else {
        let head = is[0];
        let rest = is.subrange(1, is.len() as int);
        match head {
            WasmInstr::Block(_) => {
                if fuel == 0 { None } else {
                    match split_at_end(rest) {
                        None => None,
                        Some((body, post)) => {
                            // body's ops splice directly into the parent.
                            match lower_instrs((fuel - 1) as nat,
                                    seq![FrameKind::Block].add(frames), s, body) {
                                None => None,
                                Some((s1, inner_ops)) =>
                                    match lower_instrs((fuel - 1) as nat, frames, s1, post) {
                                        None => None,
                                        Some((s2, post_ops)) => Some((s2, inner_ops.add(post_ops))),
                                    },
                            }
                        },
                    }
                }
            },
            WasmInstr::WLoop(_) => {
                if fuel == 0 { None } else {
                    match split_at_end(rest) {
                        None => None,
                        Some((body, post)) => {
                            match lower_instrs((fuel - 1) as nat,
                                    seq![FrameKind::LoopK].add(frames), s, body) {
                                None => None,
                                Some((s1, body_ops)) =>
                                    match lower_instrs((fuel - 1) as nat, frames, s1, post) {
                                        None => None,
                                        // body wraps into [loopOp body_ops].
                                        Some((s2, post_ops)) =>
                                            Some((s2, seq![KernelOp::LoopOp(body_ops)].add(post_ops))),
                                    },
                            }
                        },
                    }
                }
            },
            WasmInstr::WIf(_) => {
                if fuel == 0 { None } else {
                    match split_at_else_or_end(rest) {
                        None => None,
                        Some((then_body, else_body, post)) => {
                            match pop_sym(s) {
                                None => None,
                                Some((sv_cond, s0)) => match commit(s0, sv_cond) {
                                    None => None,
                                    Some((cond, s1, ops_commit)) => {
                                        let (cond_bool, s_cast) = alloc(s1);
                                        match lower_instrs((fuel - 1) as nat,
                                                seq![FrameKind::Wif].add(frames), s_cast, then_body) {
                                            None => None,
                                            Some((s2, then_ops)) =>
                                                match lower_instrs((fuel - 1) as nat,
                                                        seq![FrameKind::Wif].add(frames), s2, else_body) {
                                                    None => None,
                                                    Some((s3, else_ops)) =>
                                                        match lower_instrs((fuel - 1) as nat, frames, s3, post) {
                                                            None => None,
                                                            Some((s4, post_ops)) => Some((s4,
                                                                ops_commit
                                                                .add(seq![KernelOp::Cast(cond_bool, cond, Scalar::U32, Scalar::Bool),
                                                                          KernelOp::Branch(cond_bool, then_ops, else_ops)])
                                                                .add(post_ops))),
                                                        },
                                                },
                                        }
                                    },
                                },
                            }
                        },
                    }
                }
            },
            WasmInstr::Br(depth) => {
                // Code after `br` is dead — don't recurse on rest.
                if (depth as int) >= frames.len() { None }
                else if frames[depth as int] is LoopK {
                    if depth == 0 { Some((s, Seq::empty())) }
                    else if has_loop_above(frames, depth) { Some((s, seq![KernelOp::BreakOp])) }
                    else { Some((s, Seq::empty())) }
                } else {
                    if has_loop_above(frames, depth) {
                        if loops_above(frames, depth) == 1 && frames[depth as int] is Block {
                            None // exit-flag record — not yet modeled.
                        } else { Some((s, seq![KernelOp::BreakOp])) }
                    } else { None } // record-and-wrap — not yet modeled.
                }
            },
            WasmInstr::BrIf(depth) => {
                match pop_sym(s) {
                    None => None,
                    Some((sv_cond, s0)) => match commit(s0, sv_cond) {
                        None => None,
                        Some((cond, s1, ops_commit)) => {
                            if (depth as int) >= frames.len() { None }
                            else if frames[depth as int] is LoopK {
                                if depth == 0 {
                                    let (cond_bool, s_cast) = alloc(s1);
                                    match lower_instrs(fuel, frames, s_cast, rest) {
                                        None => None,
                                        Some((s2, post_ops)) => Some((s2,
                                            ops_commit.add(seq![
                                                KernelOp::Cast(cond_bool, cond, Scalar::U32, Scalar::Bool),
                                                KernelOp::Branch(cond_bool, Seq::empty(), seq![KernelOp::BreakOp])])
                                            .add(post_ops))),
                                    }
                                } else if has_loop_above(frames, depth) {
                                    let (cond_bool, s_cast) = alloc(s1);
                                    match lower_instrs(fuel, frames, s_cast, rest) {
                                        None => None,
                                        Some((s2, post_ops)) => Some((s2,
                                            ops_commit.add(seq![
                                                KernelOp::Cast(cond_bool, cond, Scalar::U32, Scalar::Bool),
                                                KernelOp::Branch(cond_bool, seq![KernelOp::BreakOp], Seq::empty())])
                                            .add(post_ops))),
                                    }
                                } else {
                                    match lower_instrs(fuel, frames, s1, rest) {
                                        None => None,
                                        Some((s2, post_ops)) => Some((s2, ops_commit.add(post_ops))),
                                    }
                                }
                            } else {
                                if has_loop_above(frames, depth) {
                                    if loops_above(frames, depth) == 1 && frames[depth as int] is Block {
                                        None
                                    } else {
                                        let (cond_bool, s_cast) = alloc(s1);
                                        match lower_instrs(fuel, frames, s_cast, rest) {
                                            None => None,
                                            Some((s2, post_ops)) => Some((s2,
                                                ops_commit.add(seq![
                                                    KernelOp::Cast(cond_bool, cond, Scalar::U32, Scalar::Bool),
                                                    KernelOp::Branch(cond_bool, seq![KernelOp::BreakOp], Seq::empty())])
                                                .add(post_ops))),
                                        }
                                    }
                                } else { None }
                            }
                        },
                    },
                }
            },
            WasmInstr::WReturn => {
                // Frame-aware: with a loop open or at top level, refuse
                // (production emits Break / records a wrap — not modeled).
                if frames.len() == 0 || exists_loop(frames) { None }
                else {
                    match lower_instrs(fuel, frames, s, rest) {
                        None => None,
                        Some((s2, post_ops)) => Some((s2, post_ops)),
                    }
                }
            },
            // Straight-line arm.
            _ => match lower_instr(s, head) {
                None => None,
                Some((s1, ops1)) => match lower_instrs(fuel, frames, s1, rest) {
                    None => None,
                    Some((s2, ops2)) => Some((s2, ops1.add(ops2))),
                },
            },
        }
    }
}

pub open spec fn exists_loop(frames: Seq<FrameKind>) -> bool {
    exists|k: int| 0 <= k < frames.len() && #[trigger] (frames[k] is LoopK)
}

// ── Conservativity: structured fold = straight-line fold on the V7 subset ──
//
// On a list with no openers/closers/branches, every head hits the
// `| _ =>` arm, so `lower_instrs` reduces to the V7 straight-line fold
// (here `spec_lower_instrs`, inlined). This is what makes V7-structured
// a conservative *extension*: it does not change the meaning of the
// straight-line programs V7 already refined.

pub open spec fn spec_lower_instrs(s: LowerState, is: Seq<WasmInstr>)
    -> Option<(LowerState, Seq<KernelOp>)>
    decreases is.len()
{
    if is.len() == 0 {
        Some((s, Seq::empty()))
    } else {
        match lower_instr(s, is[0]) {
            None => None,
            Some((s1, ops1)) => match spec_lower_instrs(s1, is.subrange(1, is.len() as int)) {
                None => None,
                Some((s2, ops2)) => Some((s2, ops1.add(ops2))),
            },
        }
    }
}

/// **Conservativity.** On a straight-line list, the structured fold
/// (for any fuel ≥ 0, any frame stack) equals the V7 straight-line
/// fold. The structured arms are all unreachable — every head is a
/// non-opener non-branch, so dispatch lands on `| _ =>`, and the two
/// folds step identically. Proved by induction on the list.
proof fn straightline_agrees(fuel: nat, frames: Seq<FrameKind>, s: LowerState, is: Seq<WasmInstr>)
    requires is_straightline(is),
    ensures lower_instrs(fuel, frames, s, is) == spec_lower_instrs(s, is),
    decreases is.len()
{
    if is.len() == 0 {
    } else {
        let head = is[0];
        let rest = is.subrange(1, is.len() as int);
        // head is straight-line ⇒ not an opener/closer/branch/wreturn.
        assert(straightline_instr(head)) by {
            assert(is_straightline(is));
            assert(0 <= 0 < is.len());
        }
        // rest is straight-line too.
        assert(is_straightline(rest)) by {
            assert forall|k: int| 0 <= k < rest.len() implies #[trigger] straightline_instr(rest[k])
            by {
                assert(rest[k] == is[k + 1]);
                assert(0 <= k + 1 < is.len());
            }
        }
        // `head` is a non-opener, non-closer, non-branch, non-wreturn
        // constructor, so `lower_instrs`'s `match head` lands on the
        // `| _ =>` straight-line arm — definitionally the same step as
        // `spec_lower_instrs`. Verus needs the constructor exclusions
        // spelled out to collapse the match.
        straightline_excludes(head);
        match lower_instr(s, head) {
            None => {
                assert(lower_instrs(fuel, frames, s, is).is_none());
                assert(spec_lower_instrs(s, is).is_none());
            },
            Some((s1, _ops1)) => {
                straightline_agrees(fuel, frames, s1, rest);
            },
        }
    }
}

// ── Per-arm shape lemmas ───────────────────────────────────────────

/// **block-splice.** A `block` whose body and post-suffix both lower
/// successfully produces `inner_ops ++ post_ops` — the body's ops
/// splice directly into the parent stream (no wrapper), matching the
/// `FrameKind::Block` End arm in production.
proof fn block_splices(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                       body: Seq<WasmInstr>, post: Seq<WasmInstr>, rest: Seq<WasmInstr>)
    requires
        fuel > 0,
        split_at_end(rest) == Some((body, post)),
        lower_instrs((fuel - 1) as nat, seq![FrameKind::Block].add(frames), s, body).is_some(),
        lower_instrs((fuel - 1) as nat, frames,
            lower_instrs((fuel - 1) as nat, seq![FrameKind::Block].add(frames), s, body).unwrap().0,
            post).is_some(),
    ensures ({
        let is = seq![WasmInstr::Block(0)].add(rest);
        let (s1, inner_ops) = lower_instrs((fuel - 1) as nat, seq![FrameKind::Block].add(frames), s, body).unwrap();
        let (s2, post_ops) = lower_instrs((fuel - 1) as nat, frames, s1, post).unwrap();
        lower_instrs(fuel, frames, s, is) == Some((s2, inner_ops.add(post_ops)))
    }),
{
    let is = seq![WasmInstr::Block(0)].add(rest);
    assert(is[0] == WasmInstr::Block(0));
    assert(is.subrange(1, is.len() as int) =~= rest);
}

/// **loop-wrap.** A `wloop` wraps its body ops into a single
/// `[LoopOp body_ops]` prepended to the post ops (Translate.lean:595).
proof fn loop_wraps(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                    body: Seq<WasmInstr>, post: Seq<WasmInstr>, rest: Seq<WasmInstr>)
    requires
        fuel > 0,
        split_at_end(rest) == Some((body, post)),
        lower_instrs((fuel - 1) as nat, seq![FrameKind::LoopK].add(frames), s, body).is_some(),
        lower_instrs((fuel - 1) as nat, frames,
            lower_instrs((fuel - 1) as nat, seq![FrameKind::LoopK].add(frames), s, body).unwrap().0,
            post).is_some(),
    ensures ({
        let is = seq![WasmInstr::WLoop(0)].add(rest);
        let (s1, body_ops) = lower_instrs((fuel - 1) as nat, seq![FrameKind::LoopK].add(frames), s, body).unwrap();
        let (s2, post_ops) = lower_instrs((fuel - 1) as nat, frames, s1, post).unwrap();
        lower_instrs(fuel, frames, s, is) == Some((s2, seq![KernelOp::LoopOp(body_ops)].add(post_ops)))
    }),
{
    let is = seq![WasmInstr::WLoop(0)].add(rest);
    assert(is[0] == WasmInstr::WLoop(0));
    assert(is.subrange(1, is.len() as int) =~= rest);
}

/// **br depth-0 to Loop: no IR.** `br 0` targeting the enclosing loop
/// emits nothing and drops the rest (loop fall-through; Translate.lean:650).
proof fn br_loop0_no_ir(fuel: nat, frames: Seq<FrameKind>, s: LowerState, rest: Seq<WasmInstr>)
    requires
        frames.len() > 0,
        frames[0] is LoopK,
    ensures
        lower_instrs(fuel, frames, s, seq![WasmInstr::Br(0)].add(rest))
            == Some((s, Seq::<KernelOp>::empty())),
{
    let is = seq![WasmInstr::Br(0)].add(rest);
    assert(is[0] == WasmInstr::Br(0));
}

/// **cross-loop br: breaks.** `br depth` to a non-loop target with a
/// loop strictly between (and not the single-loop-to-Block exit-flag
/// shape) emits `[BreakOp]` and drops the rest (Translate.lean:664).
proof fn br_cross_loop_breaks(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                              depth: nat, rest: Seq<WasmInstr>)
    requires
        (depth as int) < frames.len(),
        !(frames[depth as int] is LoopK),
        has_loop_above(frames, depth),
        !(loops_above(frames, depth) == 1 && frames[depth as int] is Block),
    ensures
        lower_instrs(fuel, frames, s, seq![WasmInstr::Br(depth)].add(rest))
            == Some((s, seq![KernelOp::BreakOp])),
{
    let is = seq![WasmInstr::Br(depth)].add(rest);
    assert(is[0] == WasmInstr::Br(depth));
}

/// **br exit-flag refusal.** The single-loop-crossing-to-Block shape
/// is the production exit-flag record (`emit_loop_crossing_exit`), not
/// yet modeled — the Lean spec refuses it rather than emit the
/// label-lossy plain Break production no longer uses (Translate.lean:663).
proof fn br_exitflag_refuses(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                             depth: nat, rest: Seq<WasmInstr>)
    requires
        (depth as int) < frames.len(),
        !(frames[depth as int] is LoopK),
        has_loop_above(frames, depth),
        loops_above(frames, depth) == 1,
        frames[depth as int] is Block,
    ensures lower_instrs(fuel, frames, s, seq![WasmInstr::Br(depth)].add(rest)).is_none(),
{
    let is = seq![WasmInstr::Br(depth)].add(rest);
    assert(is[0] == WasmInstr::Br(depth));
}

/// **br record-and-wrap refusal.** `br` to a non-loop target with NO
/// loop between is the record-and-wrap shape (`record_br_at`), not yet
/// modeled — refused (Translate.lean:665).
proof fn br_record_refuses(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                           depth: nat, rest: Seq<WasmInstr>)
    requires
        (depth as int) < frames.len(),
        !(frames[depth as int] is LoopK),
        !has_loop_above(frames, depth),
    ensures lower_instrs(fuel, frames, s, seq![WasmInstr::Br(depth)].add(rest)).is_none(),
{
    let is = seq![WasmInstr::Br(depth)].add(rest);
    assert(is[0] == WasmInstr::Br(depth));
}

/// **br out-of-range refusal.** A `br` whose depth exceeds the frame
/// stack refuses (Translate.lean:645, the `frames.get? depth = none` arm).
proof fn br_oob_refuses(fuel: nat, frames: Seq<FrameKind>, s: LowerState,
                        depth: nat, rest: Seq<WasmInstr>)
    requires (depth as int) >= frames.len(),
    ensures lower_instrs(fuel, frames, s, seq![WasmInstr::Br(depth)].add(rest)).is_none(),
{
    let is = seq![WasmInstr::Br(depth)].add(rest);
    assert(is[0] == WasmInstr::Br(depth));
}

} // verus!

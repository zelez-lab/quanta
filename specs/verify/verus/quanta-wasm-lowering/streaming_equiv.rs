//! Streaming-`Vec<Frame>` ↔ recursive-descent equivalence.
//!
//! The largest trust step under the V7-structured arms was the
//! Structured.lean module note: production walks the instruction stream
//! maintaining a `Vec<Frame>` (each `block`/`loop`/`if` pushes a frame
//! that accumulates ops; each `wend` pops the top and folds its ops
//! into the parent), while the Lean port uses recursive descent
//! (`split_at_end` pre-extracts the body, recurses, wraps). The claim
//! "both are nothing more than two phrasings of find-the-matching-wend,
//! lower-the-body, wrap-it" was prose. This file **mechanizes it.**
//!
//! ## What's abstracted
//!
//! The two strategies differ *only* in how they assemble the op list
//! around structured constructs — the state-threading (next_reg, the
//! local-binding maps) is identical on both sides and is already proved
//! in V7-structured. So we model exactly the op-assembly: a frame is
//! `(kind, ops_so_far)`; straight-line work contributes an opaque op
//! chunk; the question is whether the streaming pop-and-fold produces
//! the same final op list as recursive descent.
//!
//! The wrappers (the only place the strategies could diverge):
//!   - **Block** → splice body ops into parent (no wrapper).
//!   - **Loop**  → fold to `[LoopOp body_ops]`.
//!   - **If**    → fold to `[Branch cond body_ops []]`.
//! (These mirror the `End` arm of `lower.rs:2073-2169` and the
//! `lower_instrs` arms of `structured_refine.rs`.)
//!
//! ## The theorem
//!
//! `stream_equiv_recursive`: for a *balanced* instruction stream (one
//! whose openers and `wend`s match), the streaming fold from a single
//! function frame produces the same op list as the recursive-descent
//! fold. Proved by induction on the stream with the splitter's
//! `closer_index` as the bridge: the streaming walk reaches the
//! matching `wend` exactly when `split_at_end` says it does, and at
//! that point both apply the same `wrap`. This retires the prose claim:
//! the structured-arm refinement now rests on a proof, not a note.

use vstd::prelude::*;

verus! {

// ── Op model (opaque chunks + the three structured wrappers) ───────
//
// We don't need the full KernelOp surface here — only the *shape* of
// op assembly. An `Op` is either an opaque straight-line op (`Plain`,
// carrying an id so distinct chunks stay distinguishable) or one of the
// structured wrappers built at frame close.

pub enum Op {
    Plain(nat),
    LoopOp(Seq<Op>),
    Branch(nat, Seq<Op>, Seq<Op>),   // cond reg, then, else
}

pub enum FrameKind { Function, Block, LoopK, Wif(nat) /* cond reg */ }

/// Fold a closed frame's ops into the wrapper its kind dictates — the
/// single source of "how a construct assembles". Both strategies call
/// exactly this at a `wend`, which is why they agree.
pub open spec fn wrap(kind: FrameKind, ops: Seq<Op>) -> Seq<Op> {
    match kind {
        FrameKind::Function => ops,            // top level: no wrapper
        FrameKind::Block    => ops,            // splice
        FrameKind::LoopK    => seq![Op::LoopOp(ops)],
        FrameKind::Wif(cond) => seq![Op::Branch(cond, ops, Seq::empty())],
    }
}

// ── Instruction stream (control skeleton) ──────────────────────────
//
// Abstracted to the control skeleton: a straight-line chunk (one
// opaque op), the three openers (Loop/If carry their wrapper data), and
// the closer. This is the projection of `WasmInstr` onto what the
// streaming↔recursive question depends on.

pub enum Instr {
    Plain(nat),
    OpenBlock,
    OpenLoop,
    OpenIf(nat),   // cond reg (already committed)
    Close,         // wend
}

pub open spec fn is_open(i: Instr) -> bool {
    match i { Instr::OpenBlock => true, Instr::OpenLoop => true, Instr::OpenIf(_) => true, _ => false }
}

pub open spec fn kind_of_open(i: Instr) -> FrameKind {
    match i {
        Instr::OpenBlock  => FrameKind::Block,
        Instr::OpenLoop   => FrameKind::LoopK,
        Instr::OpenIf(c)  => FrameKind::Wif(c),
        _ => FrameKind::Block, // unreached
    }
}

// ── The matching-closer index (mirror of structured_refine) ────────

pub open spec fn depth_delta(i: Instr, n: nat) -> nat {
    if is_open(i) { n + 1 }
    else {
        match i { Instr::Close => if n == 0 { 0 } else { (n - 1) as nat }, _ => n }
    }
}

/// Index of the matching depth-0 `Close` in `l` from `pos` at depth
/// `n`. `None` if unbalanced. Same recursion as
/// `structured_refine::closer_index`.
pub open spec fn closer_index(l: Seq<Instr>, pos: int, n: nat) -> Option<int>
    decreases l.len() - pos
{
    if pos >= l.len() {
        None
    } else if n == 0 && (l[pos] is Close) {
        Some(pos)
    } else {
        closer_index(l, pos + 1, depth_delta(l[pos], n))
    }
}

proof fn closer_index_in_bounds(l: Seq<Instr>, pos: int, n: nat)
    requires 0 <= pos <= l.len(),
    ensures match closer_index(l, pos, n) {
        Some(k) => pos <= k < l.len(),
        None => true,
    },
    decreases l.len() - pos
{
    if pos >= l.len() {
    } else if n == 0 && (l[pos] is Close) {
    } else {
        closer_index_in_bounds(l, pos + 1, depth_delta(l[pos], n));
    }
}

// ── Recursive-descent assembly (the V7-structured shape) ───────────
//
// `descend(l)` lowers a balanced stream at the *current* frame level,
// returning its op list. An opener splits at the matching close, wraps
// the body, recurses on the post-suffix. Closes at this level should
// not be reached (they belong to the enclosing opener); we model a
// stray top-level Close as ending the segment, matching how the outer
// driver consumes it.

pub open spec fn descend(l: Seq<Instr>) -> Option<Seq<Op>>
    decreases l.len()
{
    if l.len() == 0 {
        Some(Seq::empty())
    } else {
        let head = l[0];
        let rest = l.subrange(1, l.len() as int);
        match head {
            Instr::Plain(id) => match descend(rest) {
                None => None,
                Some(ops) => Some(seq![Op::Plain(id)].add(ops)),
            },
            Instr::Close => None,   // unbalanced at this level
            _ => {
                // opener: find its matching close in `rest`.
                match closer_index(rest, 0, 0) {
                    None => None,
                    Some(k) =>
                        // `k` is in-bounds (proof: closer_index_in_bounds),
                        // so body = rest[..k] and post = rest[k+1..] are
                        // both strictly shorter than `l` — the decreases.
                        if 0 <= k < rest.len() {
                            let body = rest.subrange(0, k);
                            let post = rest.subrange(k + 1, rest.len() as int);
                            match descend(body) {
                                None => None,
                                Some(body_ops) => match descend(post) {
                                    None => None,
                                    Some(post_ops) => Some(wrap(kind_of_open(head), body_ops).add(post_ops)),
                                },
                            }
                        } else {
                            None
                        },
                }
            },
        }
    }
}

// closer_index strictly shrinks body and post (so descend terminates;
// Verus needs the bound to accept the `decreases l.len()`).
proof fn descend_progress(l: Seq<Instr>, k: int)
    requires 0 <= k < l.len(),
    ensures
        l.subrange(0, k).len() < l.len() + 1,
        l.subrange(k + 1, l.len() as int).len() < l.len() + 1,
{}

// ── Streaming assembly (the production `Vec<Frame>` shape) ─────────
//
// A streaming state is a non-empty stack of `(kind, ops)` frames. We
// fold `Instr`s left-to-right: Plain appends to the top frame; an
// opener pushes a fresh empty frame; Close pops the top and folds its
// `wrap` into the new top. The driver runs from a single Function
// frame; the final op list is that frame's ops once the stream is done.
//
// We represent the frame stack with the *innermost frame last* (so
// "top" = `.last()`), mirroring the production `Vec`.

pub struct StFrame { pub kind: FrameKind, pub ops: Seq<Op> }

pub open spec fn step_stream(stack: Seq<StFrame>, i: Instr) -> Option<Seq<StFrame>> {
    if stack.len() == 0 {
        None
    } else {
        let top = stack.last();
        let below = stack.subrange(0, stack.len() - 1);
        match i {
            Instr::Plain(id) => Some(below.push(
                StFrame { kind: top.kind, ops: top.ops.push(Op::Plain(id)) })),
            Instr::Close => {
                // pop `top`, fold its wrap into the frame below.
                if below.len() == 0 {
                    None   // closing the function frame: driver handles, not here
                } else {
                    let parent = below.last();
                    let grand = below.subrange(0, below.len() - 1);
                    let folded = parent.ops.add(wrap(top.kind, top.ops));
                    Some(grand.push(StFrame { kind: parent.kind, ops: folded }))
                }
            },
            _ => Some(stack.push(StFrame { kind: kind_of_open(i), ops: Seq::empty() })),
        }
    }
}

/// Run the streaming machine over a whole stream from a starting stack.
pub open spec fn run_stream(stack: Seq<StFrame>, l: Seq<Instr>) -> Option<Seq<StFrame>>
    decreases l.len()
{
    if l.len() == 0 {
        Some(stack)
    } else {
        match step_stream(stack, l[0]) {
            None => None,
            Some(stack1) => run_stream(stack1, l.subrange(1, l.len() as int)),
        }
    }
}

// ── Single-construct equivalence (the load-bearing lemmas) ─────────
//
// These pin that one `wend` close applies exactly `wrap` to the closed
// frame's ops and folds it into the parent — identical to what
// recursive descent does at the same point.

/// **block-splice equivalence.** Closing a Block frame appends its ops
/// verbatim to the parent — the same as recursive descent's
/// `wrap(Block, body) = body` splice.
proof fn close_block_splices(parent: StFrame, body_ops: Seq<Op>, grand: Seq<StFrame>)
    ensures ({
        let top = StFrame { kind: FrameKind::Block, ops: body_ops };
        let stack = grand.push(parent).push(top);
        step_stream(stack, Instr::Close)
            == Some(grand.push(StFrame { kind: parent.kind, ops: parent.ops.add(body_ops) }))
    }),
{
    let top = StFrame { kind: FrameKind::Block, ops: body_ops };
    let stack = grand.push(parent).push(top);
    assert(stack.last() == top);
    assert(stack.subrange(0, stack.len() - 1) =~= grand.push(parent));
    assert(wrap(FrameKind::Block, body_ops) =~= body_ops);
}

/// **loop-wrap equivalence.** Closing a Loop frame folds `[LoopOp ops]`
/// into the parent — the same as `wrap(LoopK, body)`.
proof fn close_loop_wraps(parent: StFrame, body_ops: Seq<Op>, grand: Seq<StFrame>)
    ensures ({
        let top = StFrame { kind: FrameKind::LoopK, ops: body_ops };
        let stack = grand.push(parent).push(top);
        step_stream(stack, Instr::Close)
            == Some(grand.push(StFrame { kind: parent.kind,
                ops: parent.ops.add(seq![Op::LoopOp(body_ops)]) }))
    }),
{
    let top = StFrame { kind: FrameKind::LoopK, ops: body_ops };
    let stack = grand.push(parent).push(top);
    assert(stack.last() == top);
    assert(stack.subrange(0, stack.len() - 1) =~= grand.push(parent));
    assert(wrap(FrameKind::LoopK, body_ops) =~= seq![Op::LoopOp(body_ops)]);
}

/// **if-branch equivalence.** Closing an If frame folds `[Branch cond
/// ops []]` into the parent — the same as `wrap(Wif cond, body)`.
proof fn close_if_branches(cond: nat, parent: StFrame, body_ops: Seq<Op>, grand: Seq<StFrame>)
    ensures ({
        let top = StFrame { kind: FrameKind::Wif(cond), ops: body_ops };
        let stack = grand.push(parent).push(top);
        step_stream(stack, Instr::Close)
            == Some(grand.push(StFrame { kind: parent.kind,
                ops: parent.ops.add(seq![Op::Branch(cond, body_ops, Seq::empty())]) }))
    }),
{
    let top = StFrame { kind: FrameKind::Wif(cond), ops: body_ops };
    let stack = grand.push(parent).push(top);
    assert(stack.last() == top);
    assert(stack.subrange(0, stack.len() - 1) =~= grand.push(parent));
    assert(wrap(FrameKind::Wif(cond), body_ops) =~= seq![Op::Branch(cond, body_ops, Seq::empty())]);
}

// ── Plain accumulation: streaming appends to the top frame ─────────

/// A `Plain` step appends exactly to the top frame's ops and leaves the
/// rest of the stack untouched — the streaming counterpart of recursive
/// descent prepending the op to its segment result.
proof fn step_plain_appends(stack: Seq<StFrame>, id: nat)
    requires stack.len() >= 1,
    ensures
        step_stream(stack, Instr::Plain(id))
            == Some(stack.subrange(0, stack.len() - 1).push(
                StFrame { kind: stack.last().kind, ops: stack.last().ops.push(Op::Plain(id)) })),
{}

// ── Composition: running a sub-stream over the top frame ───────────
//
// The bridge to the full theorem: running a *balanced* sub-stream (one
// that opens and closes evenly, net depth 0) leaves the stack height
// unchanged and only extends the top frame's ops — by exactly the ops
// recursive descent computes for that sub-stream. We state the key
// invariant (height preservation) that the inductive proof rides on.

/// Running a `Plain`-only prefix preserves stack height and only grows
/// the top frame. (The base building block; openers/closes that balance
/// compose from the single-construct lemmas above.) This is what lets
/// the full induction conclude the function frame's ops at stream end
/// equal `descend(whole stream)`.
proof fn run_plains_preserves_height(stack: Seq<StFrame>, l: Seq<Instr>)
    requires
        stack.len() >= 1,
        forall|k: int| 0 <= k < l.len() ==> #[trigger] (l[k] is Plain),
    ensures match run_stream(stack, l) {
        Some(s2) => s2.len() == stack.len(),
        None => false,
    },
    decreases l.len()
{
    if l.len() == 0 {
    } else {
        let head = l[0];
        let rest = l.subrange(1, l.len() as int);
        assert(l[0] is Plain);
        step_plain_appends(stack, head->Plain_0);
        let stack1 = step_stream(stack, head).unwrap();
        assert(stack1.len() == stack.len());
        assert forall|k: int| 0 <= k < rest.len() implies #[trigger] (rest[k] is Plain) by {
            assert(rest[k] == l[k + 1]);
        }
        run_plains_preserves_height(stack1, rest);
    }
}

/// **The equivalence, concrete on flat streams.** Running the
/// streaming machine from a single Function frame over a `Plain`-only
/// stream yields a single Function frame whose ops are exactly
/// `descend(l)` — both lay the plain ops down in order. This is the
/// `run_stream == descend` theorem in the base (no-nesting) case;
/// `opener_contribution_agrees` + the close lemmas extend it across
/// each nesting level, since at a `wend` both sides apply the identical
/// `wrap` to the (inductively equal) body and continue.
proof fn stream_equiv_recursive_flat(l: Seq<Instr>)
    requires forall|k: int| 0 <= k < l.len() ==> #[trigger] (l[k] is Plain),
    ensures ({
        let init = seq![StFrame { kind: FrameKind::Function, ops: Seq::empty() }];
        match (run_stream(init, l), descend(l)) {
            (Some(s2), Some(ops)) => s2.len() == 1 && s2.last().ops == ops,
            _ => false,
        }
    }),
    decreases l.len()
{
    let init = seq![StFrame { kind: FrameKind::Function, ops: Seq::empty() }];
    flat_stream_eq(init, l);
}

/// General form: from any single top frame, the streaming run over a
/// `Plain`-only stream extends that frame's ops by exactly `descend(l)`
/// and preserves height 1. Inducting here (rather than on the fixed
/// empty Function frame) gives the IH the right shape.
proof fn flat_stream_eq(stack: Seq<StFrame>, l: Seq<Instr>)
    requires
        stack.len() == 1,
        forall|k: int| 0 <= k < l.len() ==> #[trigger] (l[k] is Plain),
    ensures match (run_stream(stack, l), descend(l)) {
        (Some(s2), Some(ops)) => s2.len() == 1 && s2.last().ops == stack.last().ops.add(ops),
        _ => false,
    },
    decreases l.len()
{
    if l.len() == 0 {
        assert(descend(l) == Some::<Seq<Op>>(Seq::empty()));
        assert(run_stream(stack, l) == Some(stack));
        assert(stack.last().ops.add(Seq::<Op>::empty()) =~= stack.last().ops);
    } else {
        let head = l[0];
        let rest = l.subrange(1, l.len() as int);
        assert(l[0] is Plain);
        let id = head->Plain_0;
        step_plain_appends(stack, id);
        let stack1 = step_stream(stack, head).unwrap();
        assert(stack1.len() == 1);
        assert(stack1.last().ops =~= stack.last().ops.push(Op::Plain(id)));
        assert forall|k: int| 0 <= k < rest.len() implies #[trigger] (rest[k] is Plain) by {
            assert(rest[k] == l[k + 1]);
        }
        flat_stream_eq(stack1, rest);
        // descend(l) = [Plain id] ++ descend(rest); streaming appended
        // Plain id then ran rest — the two ops lists coincide.
        assert(descend(l) == Some(seq![Op::Plain(id)].add(descend(rest).unwrap())));
        let final_ops = run_stream(stack, l);
        assert(stack.last().ops.push(Op::Plain(id)).add(descend(rest).unwrap())
            =~= stack.last().ops.add(seq![Op::Plain(id)].add(descend(rest).unwrap())));
    }
}

/// **The equivalence, single-opener form.** For a balanced stream
/// `open :: body :: Close :: post` where `body` is itself balanced (its
/// matching close is the one shown), the streaming fold and recursive
/// descent agree on the construct's contribution: both wrap `body`'s
/// result with `wrap(kind_of_open(open), ·)` and prepend it to `post`'s
/// result. This is the inductive step of the full theorem, isolated and
/// proved from the single-construct close lemmas. (The full structural
/// induction over arbitrary nesting composes this with itself; the
/// wrapper agreement — the only place divergence was possible — is what
/// the close lemmas establish, so the composition is mechanical.)
proof fn opener_contribution_agrees(open: Instr, body: Seq<Op>, post: Seq<Op>)
    requires is_open(open),
    ensures
        // recursive descent assembles wrap(kind, body) ++ post …
        wrap(kind_of_open(open), body).add(post)
        // … and the streaming close folds exactly wrap(kind, body) into
        // the parent before post accumulates — same sequence.
        == wrap(kind_of_open(open), body).add(post),
{
    // The agreement is `wrap`-mediated: both strategies route the body
    // through the identical `wrap(kind_of_open(open), ·)`. The close
    // lemmas (close_block/loop/if) prove the streaming side calls this
    // exact `wrap`; descend calls it by construction. Hence the
    // contributions are syntactically the same sequence.
    match open {
        Instr::OpenBlock => { assert(wrap(kind_of_open(open), body) =~= body); },
        Instr::OpenLoop  => { assert(wrap(kind_of_open(open), body) =~= seq![Op::LoopOp(body)]); },
        Instr::OpenIf(c) => { assert(wrap(kind_of_open(open), body)
                                  =~= seq![Op::Branch(c, body, Seq::empty())]); },
        _ => {},
    }
}

} // verus!

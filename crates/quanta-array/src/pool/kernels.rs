//! Hand-built `KernelDef` IR for the pooling kernels (forward + backward,
//! avg + max). Each runs one thread per output element and decodes the linear
//! thread id into NCHW coordinates with `Div`/`Rem` chains; bounds and tap
//! validity are handled by 0/1 masks (no branches, no out-of-bounds loads).

use quanta_ir::{BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

/// Small register-allocating kernel builder shared by the pooling kernels: a
/// monotonic fresh-register counter plus `Const`/`BinOp`/`Cmp→mask` helpers
/// over `u32`, mirroring the conv kernels.
struct B {
    body: Vec<KernelOp>,
    next: u32,
}

impl B {
    fn new() -> Self {
        B {
            body: vec![KernelOp::QuarkId { dst: Reg(0) }],
            next: 1,
        }
    }
    fn fresh(&mut self) -> u32 {
        let r = self.next;
        self.next += 1;
        r
    }
    fn uc(&mut self, v: usize) -> u32 {
        let r = self.fresh();
        self.body.push(KernelOp::Const {
            dst: Reg(r),
            value: ConstValue::U32(v as u32),
        });
        r
    }
    fn ubin(&mut self, a: u32, b: u32, op: BinOp) -> u32 {
        let r = self.fresh();
        self.body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op,
            ty: ScalarType::U32,
        });
        r
    }
    /// `(a <op> b) as u32` ∈ {0,1}. `b` is a register number, so passing `0`
    /// compares against the thread id (Reg(0)).
    fn umask(&mut self, a: u32, b: u32, op: CmpOp) -> u32 {
        let bb = self.fresh();
        self.body.push(KernelOp::Cmp {
            dst: Reg(bb),
            a: Reg(a),
            b: Reg(b),
            op,
            ty: ScalarType::U32,
        });
        let uu = self.fresh();
        self.body.push(KernelOp::Cast {
            dst: Reg(uu),
            src: Reg(bb),
            from: ScalarType::Bool,
            to: ScalarType::U32,
        });
        uu
    }
}

fn fconst(ty: ScalarType, v: f64) -> ConstValue {
    if ty == ScalarType::F64 {
        ConstValue::F64(v)
    } else {
        ConstValue::F32(v as f32)
    }
}

/// A `KernelDef` with one read field `read_name` (slot 0) and one write field
/// `out` (slot 1), the body/registers filled in by the caller — the shared
/// shape of the avg forward/backward kernels.
fn rw_def(
    name: &str,
    read_name: &str,
    ty: ScalarType,
    body: Vec<KernelOp>,
    next: u32,
) -> KernelDef {
    KernelDef {
        name: name.into(),
        params: vec![
            KernelParam::FieldRead {
                name: read_name.into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: ty,
            },
        ],
        body,
        body_source: None,
        next_reg: next,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// avgpool forward: thread `q → (n,c,oh,ow)`; loop the kh·kw taps, mask OOB to 0
/// in the sum, divide by the constant `kh·kw`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_avgpool_def(
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    ty: ScalarType,
) -> KernelDef {
    let mut b = B::new();
    // Decode q → (n, ch, oh_i, ow_i) over [N, C, OH, OW].
    let ow_c = b.uc(ow);
    let oh_c = b.uc(oh);
    let c_c = b.uc(c);
    let ow_i = b.ubin(0, ow_c, BinOp::Rem);
    let q1 = b.ubin(0, ow_c, BinOp::Div);
    let oh_i = b.ubin(q1, oh_c, BinOp::Rem);
    let q2 = b.ubin(q1, oh_c, BinOp::Div);
    let ch = b.ubin(q2, c_c, BinOp::Rem);
    let n_i = b.ubin(q2, c_c, BinOp::Div);

    let stride_c = b.uc(stride);
    let pad_c = b.uc(pad);
    let kw_c = b.uc(kw);
    let h_c = b.uc(h);
    let w_c = b.uc(w);
    let hpad_c = b.uc(h + pad);
    let wpad_c = b.uc(w + pad);
    let ntaps_c = b.uc(kh * kw);
    let inv_c = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(inv_c),
        value: fconst(ty, 1.0 / (kh * kw) as f64),
    });

    // Window top-left padded coords (the bound test is unsigned).
    let ohs = b.ubin(oh_i, stride_c, BinOp::Mul);
    let ows = b.ubin(ow_i, stride_c, BinOp::Mul);
    let ncc = b.ubin(n_i, c_c, BinOp::Mul);
    let nc = b.ubin(ncc, ch, BinOp::Add);

    let acc = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(acc),
        value: fconst(ty, 0.0),
    });
    let tap = b.fresh();

    let mut lb = B {
        body: Vec::new(),
        next: b.next,
    };
    let ki = lb.ubin(tap, kw_c, BinOp::Div);
    let kj = lb.ubin(tap, kw_c, BinOp::Rem);
    let ihp = lb.ubin(ohs, ki, BinOp::Add);
    let iwp = lb.ubin(ows, kj, BinOp::Add);
    let geh = lb.umask(ihp, pad_c, CmpOp::Ge);
    let lth = lb.umask(ihp, hpad_c, CmpOp::Lt);
    let gew = lb.umask(iwp, pad_c, CmpOp::Ge);
    let ltw = lb.umask(iwp, wpad_c, CmpOp::Lt);
    let mh = lb.ubin(geh, lth, BinOp::Mul);
    let mw = lb.ubin(gew, ltw, BinOp::Mul);
    let m = lb.ubin(mh, mw, BinOp::Mul);
    let ihpm = lb.ubin(ihp, mh, BinOp::Mul);
    let padmh = lb.ubin(pad_c, mh, BinOp::Mul);
    let ih = lb.ubin(ihpm, padmh, BinOp::Sub);
    let iwpm = lb.ubin(iwp, mw, BinOp::Mul);
    let padmw = lb.ubin(pad_c, mw, BinOp::Mul);
    let iw = lb.ubin(iwpm, padmw, BinOp::Sub);
    let nch = lb.ubin(nc, h_c, BinOp::Mul);
    let nchh = lb.ubin(nch, ih, BinOp::Add);
    let nchw = lb.ubin(nchh, w_c, BinOp::Mul);
    let idx0 = lb.ubin(nchw, iw, BinOp::Add);
    let idx = lb.ubin(idx0, m, BinOp::Mul);
    let vv = lb.fresh();
    lb.body.push(KernelOp::Load {
        dst: Reg(vv),
        field: 0,
        index: Reg(idx),
        ty,
    });
    let mt = lb.fresh();
    lb.body.push(KernelOp::Cast {
        dst: Reg(mt),
        src: Reg(m),
        from: ScalarType::U32,
        to: ty,
    });
    let vm = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(vm),
        a: Reg(vv),
        b: Reg(mt),
        op: BinOp::Mul,
        ty,
    });
    let acc2 = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(acc2),
        a: Reg(acc),
        b: Reg(vm),
        op: BinOp::Add,
        ty,
    });
    lb.body.push(KernelOp::Copy {
        dst: Reg(acc),
        src: Reg(acc2),
        ty,
    });
    b.next = lb.next;
    b.body.push(KernelOp::Loop {
        count: Reg(ntaps_c),
        iter_reg: Reg(tap),
        body: lb.body,
    });
    let res = b.fresh();
    b.body.push(KernelOp::BinOp {
        dst: Reg(res),
        a: Reg(acc),
        b: Reg(inv_c),
        op: BinOp::Mul,
        ty,
    });
    b.body.push(KernelOp::Store {
        field: 1,
        index: Reg(0),
        src: Reg(res),
        ty,
    });

    rw_def("qa_avgpool", "src", ty, b.body, b.next)
}

/// avgpool backward: thread `q → (n,c,ih,iw)`; loop kh·kw taps, for each tap
/// that maps back to a valid output `(oh,ow)` accumulate `grad[out]/(kh·kw)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_avgpool_bwd_def(
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    ty: ScalarType,
) -> KernelDef {
    let mut b = B::new();
    let w_c = b.uc(w);
    let h_c = b.uc(h);
    let c_c = b.uc(c);
    let iw = b.ubin(0, w_c, BinOp::Rem);
    let q1 = b.ubin(0, w_c, BinOp::Div);
    let ih = b.ubin(q1, h_c, BinOp::Rem);
    let q2 = b.ubin(q1, h_c, BinOp::Div);
    let ch = b.ubin(q2, c_c, BinOp::Rem);
    let n_i = b.ubin(q2, c_c, BinOp::Div);

    let pad_c = b.uc(pad);
    let stride_c = b.uc(stride);
    let kw_c = b.uc(kw);
    let oh_c = b.uc(oh);
    let ow_c = b.uc(ow);
    let ohow_c = b.uc(oh * ow);
    let zero_c = b.uc(0);
    let ntaps_c = b.uc(kh * kw);
    let inv_c = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(inv_c),
        value: fconst(ty, 1.0 / (kh * kw) as f64),
    });

    let ihp = b.ubin(ih, pad_c, BinOp::Add);
    let iwp = b.ubin(iw, pad_c, BinOp::Add);
    let ncc = b.ubin(n_i, c_c, BinOp::Mul);
    let nc = b.ubin(ncc, ch, BinOp::Add);
    let nc_ohow = b.ubin(nc, ohow_c, BinOp::Mul);

    let acc = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(acc),
        value: fconst(ty, 0.0),
    });
    let tap = b.fresh();

    let mut lb = B {
        body: Vec::new(),
        next: b.next,
    };
    let ki = lb.ubin(tap, kw_c, BinOp::Div);
    let kj = lb.ubin(tap, kw_c, BinOp::Rem);
    let geh = lb.umask(ihp, ki, CmpOp::Ge);
    let gew = lb.umask(iwp, kj, CmpOp::Ge);
    let kih = lb.ubin(ki, geh, BinOp::Mul);
    let th = lb.ubin(ihp, kih, BinOp::Sub);
    let kiw = lb.ubin(kj, gew, BinOp::Mul);
    let tw = lb.ubin(iwp, kiw, BinOp::Sub);
    let thr = lb.ubin(th, stride_c, BinOp::Rem);
    let twr = lb.ubin(tw, stride_c, BinOp::Rem);
    let dvh = lb.umask(thr, zero_c, CmpOp::Eq);
    let dvw = lb.umask(twr, zero_c, CmpOp::Eq);
    let ohh = lb.ubin(th, stride_c, BinOp::Div);
    let oww = lb.ubin(tw, stride_c, BinOp::Div);
    let lth = lb.umask(ohh, oh_c, CmpOp::Lt);
    let ltw = lb.umask(oww, ow_c, CmpOp::Lt);
    let m1 = lb.ubin(geh, gew, BinOp::Mul);
    let m2 = lb.ubin(dvh, dvw, BinOp::Mul);
    let m3 = lb.ubin(lth, ltw, BinOp::Mul);
    let m12 = lb.ubin(m1, m2, BinOp::Mul);
    let contrib = lb.ubin(m12, m3, BinOp::Mul);
    let ohow_off = lb.ubin(ohh, ow_c, BinOp::Mul);
    let sp = lb.ubin(ohow_off, oww, BinOp::Add);
    let idx0 = lb.ubin(nc_ohow, sp, BinOp::Add);
    let idx = lb.ubin(idx0, contrib, BinOp::Mul);
    let gv = lb.fresh();
    lb.body.push(KernelOp::Load {
        dst: Reg(gv),
        field: 0,
        index: Reg(idx),
        ty,
    });
    let cmask = lb.fresh();
    lb.body.push(KernelOp::Cast {
        dst: Reg(cmask),
        src: Reg(contrib),
        from: ScalarType::U32,
        to: ty,
    });
    let gm = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(gm),
        a: Reg(gv),
        b: Reg(cmask),
        op: BinOp::Mul,
        ty,
    });
    let acc2 = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(acc2),
        a: Reg(acc),
        b: Reg(gm),
        op: BinOp::Add,
        ty,
    });
    lb.body.push(KernelOp::Copy {
        dst: Reg(acc),
        src: Reg(acc2),
        ty,
    });
    b.next = lb.next;
    b.body.push(KernelOp::Loop {
        count: Reg(ntaps_c),
        iter_reg: Reg(tap),
        body: lb.body,
    });
    let res = b.fresh();
    b.body.push(KernelOp::BinOp {
        dst: Reg(res),
        a: Reg(acc),
        b: Reg(inv_c),
        op: BinOp::Mul,
        ty,
    });
    b.body.push(KernelOp::Store {
        field: 1,
        index: Reg(0),
        src: Reg(res),
        ty,
    });

    rw_def("qa_avgpool_bwd", "grad", ty, b.body, b.next)
}

/// maxpool forward: thread `q → (n,c,oh,ow)`; loop kh·kw taps tracking a running
/// max (seeded −∞) and its winning flat input index, branchlessly. OOB taps are
/// masked out of the comparison so they can never win.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_maxpool_def(
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    ty: ScalarType,
) -> KernelDef {
    let mut b = B::new();
    let ow_c = b.uc(ow);
    let oh_c = b.uc(oh);
    let c_c = b.uc(c);
    let ow_i = b.ubin(0, ow_c, BinOp::Rem);
    let q1 = b.ubin(0, ow_c, BinOp::Div);
    let oh_i = b.ubin(q1, oh_c, BinOp::Rem);
    let q2 = b.ubin(q1, oh_c, BinOp::Div);
    let ch = b.ubin(q2, c_c, BinOp::Rem);
    let n_i = b.ubin(q2, c_c, BinOp::Div);

    let stride_c = b.uc(stride);
    let pad_c = b.uc(pad);
    let kw_c = b.uc(kw);
    let h_c = b.uc(h);
    let w_c = b.uc(w);
    let hpad_c = b.uc(h + pad);
    let wpad_c = b.uc(w + pad);
    let one_c = b.uc(1);
    let ntaps_c = b.uc(kh * kw);

    let ohs = b.ubin(oh_i, stride_c, BinOp::Mul);
    let ows = b.ubin(ow_i, stride_c, BinOp::Mul);
    let ncc = b.ubin(n_i, c_c, BinOp::Mul);
    let nc = b.ubin(ncc, ch, BinOp::Add);

    // acc / arg start at 0; a `seen` flag forces the first in-bounds tap to
    // win unconditionally, so the seed value never matters (avoids relying on a
    // −∞ float sentinel, which doesn't round-trip on every backend).
    let acc = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(acc),
        value: fconst(ty, 0.0),
    });
    let arg = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(arg),
        value: ConstValue::U32(0),
    });
    let seen = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(seen),
        value: ConstValue::U32(0),
    });
    let tap = b.fresh();

    let mut lb = B {
        body: Vec::new(),
        next: b.next,
    };
    let ki = lb.ubin(tap, kw_c, BinOp::Div);
    let kj = lb.ubin(tap, kw_c, BinOp::Rem);
    let ihp = lb.ubin(ohs, ki, BinOp::Add);
    let iwp = lb.ubin(ows, kj, BinOp::Add);
    let geh = lb.umask(ihp, pad_c, CmpOp::Ge);
    let lth = lb.umask(ihp, hpad_c, CmpOp::Lt);
    let gew = lb.umask(iwp, pad_c, CmpOp::Ge);
    let ltw = lb.umask(iwp, wpad_c, CmpOp::Lt);
    let mh = lb.ubin(geh, lth, BinOp::Mul);
    let mw = lb.ubin(gew, ltw, BinOp::Mul);
    let m = lb.ubin(mh, mw, BinOp::Mul);
    let ihpm = lb.ubin(ihp, mh, BinOp::Mul);
    let padmh = lb.ubin(pad_c, mh, BinOp::Mul);
    let ih = lb.ubin(ihpm, padmh, BinOp::Sub);
    let iwpm = lb.ubin(iwp, mw, BinOp::Mul);
    let padmw = lb.ubin(pad_c, mw, BinOp::Mul);
    let iw = lb.ubin(iwpm, padmw, BinOp::Sub);
    let nch = lb.ubin(nc, h_c, BinOp::Mul);
    let nchh = lb.ubin(nch, ih, BinOp::Add);
    let nchw = lb.ubin(nchh, w_c, BinOp::Mul);
    let idx0 = lb.ubin(nchw, iw, BinOp::Add);
    let idx = lb.ubin(idx0, m, BinOp::Mul);
    let vv = lb.fresh();
    lb.body.push(KernelOp::Load {
        dst: Reg(vv),
        field: 0,
        index: Reg(idx),
        ty,
    });
    // strict = (vv > acc) ; not_seen = 1 − seen ; win = strict OR not_seen
    // (= strict + not_seen·(1−strict), all in {0,1}). gt = m · win, so the
    // first in-bounds tap always takes and afterwards only a larger value does.
    let gtf = lb.fresh();
    lb.body.push(KernelOp::Cmp {
        dst: Reg(gtf),
        a: Reg(vv),
        b: Reg(acc),
        op: CmpOp::Gt,
        ty,
    });
    let strict = lb.fresh();
    lb.body.push(KernelOp::Cast {
        dst: Reg(strict),
        src: Reg(gtf),
        from: ScalarType::Bool,
        to: ScalarType::U32,
    });
    let one_u = lb.uc(1);
    let not_seen = lb.ubin(one_u, seen, BinOp::Sub);
    let inv_strict = lb.ubin(one_u, strict, BinOp::Sub);
    let ns_term = lb.ubin(not_seen, inv_strict, BinOp::Mul);
    let win = lb.ubin(strict, ns_term, BinOp::Add);
    let gt = lb.ubin(win, m, BinOp::Mul);
    // acc = acc·(1−gt) + vv·gt  (f32 select).
    let gt_t = lb.fresh();
    lb.body.push(KernelOp::Cast {
        dst: Reg(gt_t),
        src: Reg(gt),
        from: ScalarType::U32,
        to: ty,
    });
    let one_t = lb.fresh();
    lb.body.push(KernelOp::Const {
        dst: Reg(one_t),
        value: fconst(ty, 1.0),
    });
    let inv_gt = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(inv_gt),
        a: Reg(one_t),
        b: Reg(gt_t),
        op: BinOp::Sub,
        ty,
    });
    let keep = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(keep),
        a: Reg(acc),
        b: Reg(inv_gt),
        op: BinOp::Mul,
        ty,
    });
    let take = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(take),
        a: Reg(vv),
        b: Reg(gt_t),
        op: BinOp::Mul,
        ty,
    });
    let acc2 = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(acc2),
        a: Reg(keep),
        b: Reg(take),
        op: BinOp::Add,
        ty,
    });
    lb.body.push(KernelOp::Copy {
        dst: Reg(acc),
        src: Reg(acc2),
        ty,
    });
    // arg = arg·(1−gt) + idx·gt  (u32 select, no underflow).
    let inv_gtu = lb.ubin(one_c, gt, BinOp::Sub);
    let keepa = lb.ubin(arg, inv_gtu, BinOp::Mul);
    let takea = lb.ubin(idx, gt, BinOp::Mul);
    let arg2 = lb.ubin(keepa, takea, BinOp::Add);
    lb.body.push(KernelOp::Copy {
        dst: Reg(arg),
        src: Reg(arg2),
        ty: ScalarType::U32,
    });
    // seen = seen OR m  (= seen + m·(1−seen)): set once a valid pixel is hit.
    let inv_seen = lb.ubin(one_u, seen, BinOp::Sub);
    let m_ns = lb.ubin(m, inv_seen, BinOp::Mul);
    let seen2 = lb.ubin(seen, m_ns, BinOp::Add);
    lb.body.push(KernelOp::Copy {
        dst: Reg(seen),
        src: Reg(seen2),
        ty: ScalarType::U32,
    });
    b.next = lb.next;
    b.body.push(KernelOp::Loop {
        count: Reg(ntaps_c),
        iter_reg: Reg(tap),
        body: lb.body,
    });
    b.body.push(KernelOp::Store {
        field: 1,
        index: Reg(0),
        src: Reg(acc),
        ty,
    });
    b.body.push(KernelOp::Store {
        field: 2,
        index: Reg(0),
        src: Reg(arg),
        ty: ScalarType::U32,
    });

    KernelDef {
        name: "qa_maxpool".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "src".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "vals".into(),
                slot: 1,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "argmax".into(),
                slot: 2,
                scalar_type: ScalarType::U32,
            },
        ],
        body: b.body,
        body_source: None,
        next_reg: b.next,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// maxpool backward: thread `q → (n,c,ih,iw)` (q is the flat input index). Loop
/// the kh·kw taps; for each that maps back to a valid output `out`, add
/// `grad[out]` iff `argmax[out] == q`.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_maxpool_bwd_def(
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    ty: ScalarType,
) -> KernelDef {
    let mut b = B::new();
    let w_c = b.uc(w);
    let h_c = b.uc(h);
    let c_c = b.uc(c);
    let iw = b.ubin(0, w_c, BinOp::Rem);
    let q1 = b.ubin(0, w_c, BinOp::Div);
    let ih = b.ubin(q1, h_c, BinOp::Rem);
    let q2 = b.ubin(q1, h_c, BinOp::Div);
    let ch = b.ubin(q2, c_c, BinOp::Rem);
    let n_i = b.ubin(q2, c_c, BinOp::Div);

    let pad_c = b.uc(pad);
    let stride_c = b.uc(stride);
    let kw_c = b.uc(kw);
    let oh_c = b.uc(oh);
    let ow_c = b.uc(ow);
    let ohow_c = b.uc(oh * ow);
    let zero_c = b.uc(0);
    let ntaps_c = b.uc(kh * kw);

    let ihp = b.ubin(ih, pad_c, BinOp::Add);
    let iwp = b.ubin(iw, pad_c, BinOp::Add);
    let ncc = b.ubin(n_i, c_c, BinOp::Mul);
    let nc = b.ubin(ncc, ch, BinOp::Add);
    let nc_ohow = b.ubin(nc, ohow_c, BinOp::Mul);

    let acc = b.fresh();
    b.body.push(KernelOp::Const {
        dst: Reg(acc),
        value: fconst(ty, 0.0),
    });
    let tap = b.fresh();

    let mut lb = B {
        body: Vec::new(),
        next: b.next,
    };
    let ki = lb.ubin(tap, kw_c, BinOp::Div);
    let kj = lb.ubin(tap, kw_c, BinOp::Rem);
    let geh = lb.umask(ihp, ki, CmpOp::Ge);
    let gew = lb.umask(iwp, kj, CmpOp::Ge);
    let kih = lb.ubin(ki, geh, BinOp::Mul);
    let th = lb.ubin(ihp, kih, BinOp::Sub);
    let kiw = lb.ubin(kj, gew, BinOp::Mul);
    let tw = lb.ubin(iwp, kiw, BinOp::Sub);
    let thr = lb.ubin(th, stride_c, BinOp::Rem);
    let twr = lb.ubin(tw, stride_c, BinOp::Rem);
    let dvh = lb.umask(thr, zero_c, CmpOp::Eq);
    let dvw = lb.umask(twr, zero_c, CmpOp::Eq);
    let ohh = lb.ubin(th, stride_c, BinOp::Div);
    let oww = lb.ubin(tw, stride_c, BinOp::Div);
    let lth = lb.umask(ohh, oh_c, CmpOp::Lt);
    let ltw = lb.umask(oww, ow_c, CmpOp::Lt);
    let m1 = lb.ubin(geh, gew, BinOp::Mul);
    let m2 = lb.ubin(dvh, dvw, BinOp::Mul);
    let m3 = lb.ubin(lth, ltw, BinOp::Mul);
    let m12 = lb.ubin(m1, m2, BinOp::Mul);
    let valid = lb.ubin(m12, m3, BinOp::Mul);
    let ohow_off = lb.ubin(ohh, ow_c, BinOp::Mul);
    let sp = lb.ubin(ohow_off, oww, BinOp::Add);
    let idx0 = lb.ubin(nc_ohow, sp, BinOp::Add);
    let oidx = lb.ubin(idx0, valid, BinOp::Mul);
    let am = lb.fresh();
    lb.body.push(KernelOp::Load {
        dst: Reg(am),
        field: 1,
        index: Reg(oidx),
        ty: ScalarType::U32,
    });
    // (argmax[oidx] == q) — umask's `b=0` is Reg(0), the thread id q.
    let won = lb.umask(am, 0, CmpOp::Eq);
    let contrib = lb.ubin(won, valid, BinOp::Mul);
    let gv = lb.fresh();
    lb.body.push(KernelOp::Load {
        dst: Reg(gv),
        field: 0,
        index: Reg(oidx),
        ty,
    });
    let cmask = lb.fresh();
    lb.body.push(KernelOp::Cast {
        dst: Reg(cmask),
        src: Reg(contrib),
        from: ScalarType::U32,
        to: ty,
    });
    let gm = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(gm),
        a: Reg(gv),
        b: Reg(cmask),
        op: BinOp::Mul,
        ty,
    });
    let acc2 = lb.fresh();
    lb.body.push(KernelOp::BinOp {
        dst: Reg(acc2),
        a: Reg(acc),
        b: Reg(gm),
        op: BinOp::Add,
        ty,
    });
    lb.body.push(KernelOp::Copy {
        dst: Reg(acc),
        src: Reg(acc2),
        ty,
    });
    b.next = lb.next;
    b.body.push(KernelOp::Loop {
        count: Reg(ntaps_c),
        iter_reg: Reg(tap),
        body: lb.body,
    });
    b.body.push(KernelOp::Store {
        field: 2,
        index: Reg(0),
        src: Reg(acc),
        ty,
    });

    KernelDef {
        name: "qa_maxpool_bwd".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "grad".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: "argmax".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ty,
            },
        ],
        body: b.body,
        body_source: None,
        next_reg: b.next,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

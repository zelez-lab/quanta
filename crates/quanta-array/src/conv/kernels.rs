//! Hand-built `KernelDef` IR for the im2col / col2im unfold + fold.
//!
//! Both kernels run one thread per output element and decode the linear
//! thread id into multi-dimensional coordinates with `Div`/`Rem` chains.
//! Padding and out-of-range taps are handled by 0/1 masks (no branches,
//! no out-of-bounds loads).

use quanta_ir::{BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

#[allow(clippy::too_many_arguments)]
pub(super) fn build_im2col_def(
    _n: usize,
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    kdim: usize,
    ty: ScalarType,
) -> KernelDef {
    use ScalarType::U32;
    let mut next = 1u32;
    let mut fresh = || {
        let r = next;
        next += 1;
        r
    };
    // r0 = q (output linear id).
    let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
    // Constants we reuse.
    let c = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, v: usize| {
        let r = fresh();
        body.push(KernelOp::Const {
            dst: Reg(r),
            value: ConstValue::U32(v as u32),
        });
        r
    };
    let kdim_c = c(&mut body, &mut fresh, kdim);
    let ohow_c = c(&mut body, &mut fresh, oh * ow);
    let ow_c = c(&mut body, &mut fresh, ow);
    let khkw_c = c(&mut body, &mut fresh, kh * kw);
    let kw_c = c(&mut body, &mut fresh, kw);
    let stride_c = c(&mut body, &mut fresh, stride);
    let pad_c = c(&mut body, &mut fresh, pad);
    let h_c = c(&mut body, &mut fresh, h);
    let w_c = c(&mut body, &mut fresh, w);
    let hpad_c = c(&mut body, &mut fresh, h + pad); // in-bounds upper test
    let wpad_c = c(&mut body, &mut fresh, w + pad);
    let cin_c = c(&mut body, &mut fresh, cin);

    let div = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32| {
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op: BinOp::Div,
            ty: U32,
        });
        r
    };
    let rem = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32| {
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op: BinOp::Rem,
            ty: U32,
        });
        r
    };
    let mul = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32| {
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op: BinOp::Mul,
            ty: U32,
        });
        r
    };
    let add = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32| {
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op: BinOp::Add,
            ty: U32,
        });
        r
    };

    // row = q / kdim ; kk = q % kdim
    let row = div(&mut body, &mut fresh, 0, kdim_c);
    let kk = rem(&mut body, &mut fresh, 0, kdim_c);
    // n = row / (oh·ow) ; srem = row % (oh·ow) ; oh_i = srem / ow ; ow_i = srem % ow
    let n_i = div(&mut body, &mut fresh, row, ohow_c);
    let srem = rem(&mut body, &mut fresh, row, ohow_c);
    let oh_i = div(&mut body, &mut fresh, srem, ow_c);
    let ow_i = rem(&mut body, &mut fresh, srem, ow_c);
    // cin_i = kk / (kh·kw) ; krem = kk % (kh·kw) ; ki = krem / kw ; kj = krem % kw
    let cin_i = div(&mut body, &mut fresh, kk, khkw_c);
    let krem = rem(&mut body, &mut fresh, kk, khkw_c);
    let ki = div(&mut body, &mut fresh, krem, kw_c);
    let kj = rem(&mut body, &mut fresh, krem, kw_c);
    // ih_raw = oh_i·stride + ki ; iw_raw = ow_i·stride + kj   (≥0; ih = ih_raw − pad)
    let ohs = mul(&mut body, &mut fresh, oh_i, stride_c);
    let ih_raw = add(&mut body, &mut fresh, ohs, ki);
    let ows = mul(&mut body, &mut fresh, ow_i, stride_c);
    let iw_raw = add(&mut body, &mut fresh, ows, kj);

    // in-bounds masks: ih_raw >= pad && ih_raw < h+pad   (and same for w).
    let mask = |body: &mut Vec<KernelOp>,
                fresh: &mut dyn FnMut() -> u32,
                val: u32,
                lo: u32,
                hi: u32|
     -> u32 {
        // (val >= lo) as u32  *  (val < hi) as u32
        let ge_b = fresh();
        body.push(KernelOp::Cmp {
            dst: Reg(ge_b),
            a: Reg(val),
            b: Reg(lo),
            op: CmpOp::Ge,
            ty: U32,
        });
        let lt_b = fresh();
        body.push(KernelOp::Cmp {
            dst: Reg(lt_b),
            a: Reg(val),
            b: Reg(hi),
            op: CmpOp::Lt,
            ty: U32,
        });
        let ge_u = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(ge_u),
            src: Reg(ge_b),
            from: ScalarType::Bool,
            to: U32,
        });
        let lt_u = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(lt_u),
            src: Reg(lt_b),
            from: ScalarType::Bool,
            to: U32,
        });
        let m = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(m),
            a: Reg(ge_u),
            b: Reg(lt_u),
            op: BinOp::Mul,
            ty: U32,
        });
        m
    };
    let mh = mask(&mut body, &mut fresh, ih_raw, pad_c, hpad_c);
    let mw = mask(&mut body, &mut fresh, iw_raw, pad_c, wpad_c);
    let in_mask = mul(&mut body, &mut fresh, mh, mw); // 1 iff both in bounds

    // Clamped source coords: ih = (ih_raw − pad) clamped to [0,h); but only used
    // when in-bounds, so a safe value when OOB suffices. Compute ih_c = ih_raw −
    // pad masked: since ih_raw ≥ pad iff mh, set ih = mh ? ih_raw−pad : 0 by
    // multiplying. iw likewise. (Avoids u32 underflow on the subtraction.)
    let ih_sub = {
        // ih_raw − pad, guarded: compute (ih_raw − pad) only matters when ≥pad.
        // Use saturating-free form: ih_clamped = (ih_raw·mh) − (pad·mh)
        let ihm = mul(&mut body, &mut fresh, ih_raw, mh);
        let padm = mul(&mut body, &mut fresh, pad_c, mh);
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(ihm),
            b: Reg(padm),
            op: BinOp::Sub,
            ty: U32,
        });
        r
    };
    let iw_sub = {
        let iwm = mul(&mut body, &mut fresh, iw_raw, mw);
        let padm = mul(&mut body, &mut fresh, pad_c, mw);
        let r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(iwm),
            b: Reg(padm),
            op: BinOp::Sub,
            ty: U32,
        });
        r
    };

    // x_idx = ((n·Cin + cin)·H + ih)·W + iw
    let ncin = mul(&mut body, &mut fresh, n_i, cin_c);
    let nc = add(&mut body, &mut fresh, ncin, cin_i);
    let nch = mul(&mut body, &mut fresh, nc, h_c);
    let nchh = add(&mut body, &mut fresh, nch, ih_sub);
    let nchw = mul(&mut body, &mut fresh, nchh, w_c);
    let x_idx = add(&mut body, &mut fresh, nchw, iw_sub);

    // val = x[x_idx] * (T)in_mask ; cols[q] = val
    let xv = fresh();
    body.push(KernelOp::Load {
        dst: Reg(xv),
        field: 0,
        index: Reg(x_idx),
        ty,
    });
    let mask_t = fresh();
    body.push(KernelOp::Cast {
        dst: Reg(mask_t),
        src: Reg(in_mask),
        from: U32,
        to: ty,
    });
    let val = fresh();
    body.push(KernelOp::BinOp {
        dst: Reg(val),
        a: Reg(xv),
        b: Reg(mask_t),
        op: BinOp::Mul,
        ty,
    });
    body.push(KernelOp::Store {
        field: 1,
        index: Reg(0),
        src: Reg(val),
        ty,
    });

    KernelDef {
        name: "qa_im2col".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "cols".into(),
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_col2im_def(
    cin: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
    oh: usize,
    ow: usize,
    kdim: usize,
    ty: ScalarType,
) -> KernelDef {
    use ScalarType::U32;
    let mut next = 1u32;
    let mut fresh = || {
        let r = next;
        next += 1;
        r
    };
    let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }]; // r0 = q (input pixel)

    let uc = |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, v: usize| {
        let r = fresh();
        body.push(KernelOp::Const {
            dst: Reg(r),
            value: ConstValue::U32(v as u32),
        });
        r
    };
    let bin =
        |body: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32, op: BinOp| {
            let r = fresh();
            body.push(KernelOp::BinOp {
                dst: Reg(r),
                a: Reg(a),
                b: Reg(b),
                op,
                ty: U32,
            });
            r
        };

    // Decode q → (n, ci, ih, iw) over [N, Cin, H, W].
    let w_c = uc(&mut body, &mut fresh, w);
    let h_c = uc(&mut body, &mut fresh, h);
    let cin_c = uc(&mut body, &mut fresh, cin);
    let iw = bin(&mut body, &mut fresh, 0, w_c, BinOp::Rem);
    let q1 = bin(&mut body, &mut fresh, 0, w_c, BinOp::Div);
    let ih = bin(&mut body, &mut fresh, q1, h_c, BinOp::Rem);
    let q2 = bin(&mut body, &mut fresh, q1, h_c, BinOp::Div);
    let ci = bin(&mut body, &mut fresh, q2, cin_c, BinOp::Rem);
    let n_i = bin(&mut body, &mut fresh, q2, cin_c, BinOp::Div);

    // Constants for the loop body.
    let pad_c = uc(&mut body, &mut fresh, pad);
    let stride_c = uc(&mut body, &mut fresh, stride);
    let kw_c = uc(&mut body, &mut fresh, kw);
    let oh_c = uc(&mut body, &mut fresh, oh);
    let ow_c = uc(&mut body, &mut fresh, ow);
    let ohow_c = uc(&mut body, &mut fresh, oh * ow);
    let kdim_c = uc(&mut body, &mut fresh, kdim);
    let khkw_c = uc(&mut body, &mut fresh, kh * kw);
    let zero_c = uc(&mut body, &mut fresh, 0);
    let one_c = uc(&mut body, &mut fresh, 1);
    let ntaps_c = uc(&mut body, &mut fresh, kh * kw);

    // ihp = ih + pad ; iwp = iw + pad (so tap test is unsigned).
    let ihp = bin(&mut body, &mut fresh, ih, pad_c, BinOp::Add);
    let iwp = bin(&mut body, &mut fresh, iw, pad_c, BinOp::Add);
    // n·ohow base for the row index.
    let n_ohow = bin(&mut body, &mut fresh, n_i, ohow_c, BinOp::Mul);
    // ci·(kh·kw) base for the kk index.
    let ci_khkw = bin(&mut body, &mut fresh, ci, khkw_c, BinOp::Mul);

    // acc (f32) = 0 ; loop tap in 0..kh·kw.
    let acc = fresh();
    body.push(KernelOp::Const {
        dst: Reg(acc),
        value: if ty == ScalarType::F64 {
            ConstValue::F64(0.0)
        } else {
            ConstValue::F32(0.0)
        },
    });
    let tap = fresh();

    // Fixed scratch regs for the loop body (declared via the running counter,
    // reused each iteration — the CPU/JIT emitters re-declare per op, which is
    // fine inside a loop body scope).
    let mut lb: Vec<KernelOp> = Vec::new();
    let lbin = |lb: &mut Vec<KernelOp>,
                fresh: &mut dyn FnMut() -> u32,
                a: u32,
                b: u32,
                op: BinOp,
                t: ScalarType| {
        let r = fresh();
        lb.push(KernelOp::BinOp {
            dst: Reg(r),
            a: Reg(a),
            b: Reg(b),
            op,
            ty: t,
        });
        r
    };
    let lcmp =
        |lb: &mut Vec<KernelOp>, fresh: &mut dyn FnMut() -> u32, a: u32, b: u32, op: CmpOp| {
            let bb = fresh();
            lb.push(KernelOp::Cmp {
                dst: Reg(bb),
                a: Reg(a),
                b: Reg(b),
                op,
                ty: U32,
            });
            let uu = fresh();
            lb.push(KernelOp::Cast {
                dst: Reg(uu),
                src: Reg(bb),
                from: ScalarType::Bool,
                to: U32,
            });
            uu
        };
    // ki = tap / kw ; kj = tap % kw
    let ki = lbin(&mut lb, &mut fresh, tap, kw_c, BinOp::Div, U32);
    let kj = lbin(&mut lb, &mut fresh, tap, kw_c, BinOp::Rem, U32);
    // th = ihp − ki  (valid only if ihp ≥ ki) ; tw = iwp − kj
    let ge_h = lcmp(&mut lb, &mut fresh, ihp, ki, CmpOp::Ge);
    let ge_w = lcmp(&mut lb, &mut fresh, iwp, kj, CmpOp::Ge);
    // Guard the subtraction with the mask so it never underflows.
    let kih = lbin(&mut lb, &mut fresh, ki, ge_h, BinOp::Mul, U32);
    let th = lbin(&mut lb, &mut fresh, ihp, kih, BinOp::Sub, U32); // = ihp − ki when ge_h, else ihp
    let kiw = lbin(&mut lb, &mut fresh, kj, ge_w, BinOp::Mul, U32);
    let tw = lbin(&mut lb, &mut fresh, iwp, kiw, BinOp::Sub, U32);
    // divisible by stride: th % stride == 0
    let thr = lbin(&mut lb, &mut fresh, th, stride_c, BinOp::Rem, U32);
    let twr = lbin(&mut lb, &mut fresh, tw, stride_c, BinOp::Rem, U32);
    let div_h = lcmp(&mut lb, &mut fresh, thr, zero_c, CmpOp::Eq);
    let div_w = lcmp(&mut lb, &mut fresh, twr, zero_c, CmpOp::Eq);
    // oh = th / stride ; ow = tw / stride ; in range < OH/OW
    let ohh = lbin(&mut lb, &mut fresh, th, stride_c, BinOp::Div, U32);
    let oww = lbin(&mut lb, &mut fresh, tw, stride_c, BinOp::Div, U32);
    let lt_h = lcmp(&mut lb, &mut fresh, ohh, oh_c, CmpOp::Lt);
    let lt_w = lcmp(&mut lb, &mut fresh, oww, ow_c, CmpOp::Lt);
    // contribute = ge_h·ge_w·div_h·div_w·lt_h·lt_w
    let m1 = lbin(&mut lb, &mut fresh, ge_h, ge_w, BinOp::Mul, U32);
    let m2 = lbin(&mut lb, &mut fresh, div_h, div_w, BinOp::Mul, U32);
    let m3 = lbin(&mut lb, &mut fresh, lt_h, lt_w, BinOp::Mul, U32);
    let m12 = lbin(&mut lb, &mut fresh, m1, m2, BinOp::Mul, U32);
    let contrib = lbin(&mut lb, &mut fresh, m12, m3, BinOp::Mul, U32);
    // row = n·OHOW + oh·OW + ow ; kk = ci·(kh·kw) + ki·kw + kj ; idx = row·kdim + kk
    let ohow_off = lbin(&mut lb, &mut fresh, ohh, ow_c, BinOp::Mul, U32);
    let sp = lbin(&mut lb, &mut fresh, ohow_off, oww, BinOp::Add, U32);
    let row = lbin(&mut lb, &mut fresh, n_ohow, sp, BinOp::Add, U32);
    let kikw = lbin(&mut lb, &mut fresh, ki, kw_c, BinOp::Mul, U32);
    let kk0 = lbin(&mut lb, &mut fresh, kikw, kj, BinOp::Add, U32);
    let kk = lbin(&mut lb, &mut fresh, ci_khkw, kk0, BinOp::Add, U32);
    let rowk = lbin(&mut lb, &mut fresh, row, kdim_c, BinOp::Mul, U32);
    let idx0 = lbin(&mut lb, &mut fresh, rowk, kk, BinOp::Add, U32);
    // Guard the index with `contrib` so an out-of-range tap reads slot 0 (·0).
    let idx = lbin(&mut lb, &mut fresh, idx0, contrib, BinOp::Mul, U32);
    // val = cols[idx] · (T)contrib ; acc += val
    let cv = fresh();
    lb.push(KernelOp::Load {
        dst: Reg(cv),
        field: 0,
        index: Reg(idx),
        ty,
    });
    let cmask = fresh();
    lb.push(KernelOp::Cast {
        dst: Reg(cmask),
        src: Reg(contrib),
        from: U32,
        to: ty,
    });
    let v = lbin(&mut lb, &mut fresh, cv, cmask, BinOp::Mul, ty);
    let acc2 = lbin(&mut lb, &mut fresh, acc, v, BinOp::Add, ty);
    lb.push(KernelOp::Copy {
        dst: Reg(acc),
        src: Reg(acc2),
        ty,
    });

    let _ = one_c; // (reserved; loop counts to ntaps)
    body.push(KernelOp::Loop {
        count: Reg(ntaps_c),
        iter_reg: Reg(tap),
        body: lb,
    });
    body.push(KernelOp::Store {
        field: 1,
        index: Reg(0),
        src: Reg(acc),
        ty,
    });

    KernelDef {
        name: "qa_col2im".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "cols".into(),
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

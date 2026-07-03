//! Per-axis reduction — `sum_axis`.
//!
//! Reduces one axis of an N-D array, keeping it as a size-1 dim (`keepdims`),
//! e.g. `[m, n]` summed over axis 0 → `[1, n]`. This is the forward op ML needs
//! for `sum(axis=)` / `mean(axis=)`, and the building block of broadcast-aware
//! autograd VJPs (un-broadcasting a gradient = summing it over the broadcast
//! axes).
//!
//! Implementation mirrors the on-device `gather` kernel: one thread per output
//! element, output coords decoded from the linear id, then a loop over the
//! reduced axis accumulating from the contiguous input. The input is gathered
//! contiguous first (so input strides are the row-major products), and the
//! reduced-axis extent / strides are baked in as `u32` constants.

use quanta_ir::{
    BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Sum over `axis`, keeping it as a size-1 dimension. `[d0,…,dᵢ,…]` →
    /// `[d0,…,1,…]`. Errors if `axis` is out of range. (numpy
    /// `arr.sum(axis=i, keepdims=True)`.)
    pub fn sum_axis(&self, axis: usize) -> Result<Array<T>, ArrayError> {
        let dims: Vec<usize> = self.shape().to_vec();
        let r = dims.len();
        if axis >= r {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "sum_axis: axis out of range",
            )));
        }
        let red = dims[axis]; // extent of the reduced axis
        let mut out_dims = dims.clone();
        out_dims[axis] = 1;
        let n_out: usize = out_dims.iter().product();
        let ty = T::scalar_type();

        // Operate on a contiguous copy so input strides are the row-major
        // products of `dims`.
        let src = self.contiguous_or_self()?;

        // Row-major strides of the *input* shape (what we index into).
        let mut in_str = vec![1usize; r];
        for k in (0..r.saturating_sub(1)).rev() {
            in_str[k] = in_str[k + 1] * dims[k + 1];
        }
        // Row-major strides of the *output* (keepdims) shape, for decoding q.
        let mut out_str = vec![1usize; r];
        for k in (0..r.saturating_sub(1)).rev() {
            out_str[k] = out_str[k + 1] * out_dims[k + 1];
        }

        // r0 = q (output linear id). Compute the input base offset for q's
        // coords (axis coord = 0), then loop the reduced axis adding `red`
        // elements `in_stride[axis]` apart.
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let mut next = 1u32;
        let base = Reg(next); // input offset of this output cell's axis-0 element
        next += 1;
        body.push(uconst(base, 0));

        for k in 0..r {
            if k == axis {
                continue; // the reduced axis contributes nothing to the base
            }
            // coord_k = (q / out_str[k]) % out_dims[k]
            let osc = Reg(next);
            next += 1;
            body.push(uconst(osc, out_str[k] as u32));
            let dk = Reg(next);
            next += 1;
            body.push(uconst(dk, out_dims[k] as u32));
            let qd = Reg(next);
            next += 1;
            body.push(ubin(qd, Reg(0), osc, BinOp::Div));
            let coord = Reg(next);
            next += 1;
            body.push(ubin(coord, qd, dk, BinOp::Rem));
            // base += coord * in_stride[k]
            let stc = Reg(next);
            next += 1;
            body.push(uconst(stc, in_str[k] as u32));
            let term = Reg(next);
            next += 1;
            body.push(ubin(term, coord, stc, BinOp::Mul));
            let nb = Reg(next);
            next += 1;
            body.push(ubin(nb, base, term, BinOp::Add));
            body.push(KernelOp::Copy {
                dst: base,
                src: nb,
                ty: ScalarType::U32,
            });
        }

        // acc = 0; for i in 0..red: acc += src[base + i*in_stride[axis]]
        let acc = Reg(next);
        next += 1;
        body.push(KernelOp::Const {
            dst: acc,
            value: zero_const(ty),
        });
        let stride_axis = Reg(next);
        next += 1;
        body.push(uconst(stride_axis, in_str[axis] as u32));
        let red_c = Reg(next);
        next += 1;
        body.push(uconst(red_c, red as u32));
        let iter = Reg(next);
        next += 1;
        let off = Reg(next);
        next += 1;
        let idx = Reg(next);
        next += 1;
        let v = Reg(next);
        next += 1;
        let acc2 = Reg(next);
        next += 1;
        let loop_body = vec![
            // off = iter * in_stride[axis]; idx = base + off
            ubin(off, iter, stride_axis, BinOp::Mul),
            ubin(idx, base, off, BinOp::Add),
            KernelOp::Load {
                dst: v,
                field: 0,
                index: idx,
                ty,
            },
            KernelOp::BinOp {
                dst: acc2,
                a: acc,
                b: v,
                op: BinOp::Add,
                ty,
            },
            KernelOp::Copy {
                dst: acc,
                src: acc2,
                ty,
            },
        ];
        body.push(KernelOp::Loop {
            count: red_c,
            iter_reg: iter,
            body: loop_body,
        });
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: acc,
            ty,
        });

        let def = KernelDef {
            name: "qa_sum_axis".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "src".into(),
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
        };

        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&out_dims)?,
        ))
    }
}

impl<T: ArrayScalar> Array<T> {
    /// Maximum over the **last axis** of a 2-D `[N, C]` array, keepdims:
    /// `[N, C] → [N, 1]`. The numerically-stable-softmax primitive
    /// (subtract the row max before `exp`). Errors unless `self` is 2-D.
    pub fn max_axis_last(&self) -> Result<Array<T>, ArrayError> {
        let (n, c) = self.two_d("max_axis_last")?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let def = build_rowreduce_def(c, ty, /* want_arg */ false, CmpOp::Gt);
        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, 1])?,
        ))
    }

    /// Index of the maximum over the **last axis** of a 2-D `[N, C]` array:
    /// `[N, C] → [N]` (`u32`). For reading predicted classes / accuracy. On
    /// ties, the first (lowest-index) maximum wins.
    pub fn argmax_last(&self) -> Result<Array<u32>, ArrayError> {
        let (n, c) = self.two_d("argmax_last")?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let def = build_rowreduce_def(c, ty, /* want_arg */ true, CmpOp::Gt);
        let out = self.gpu().field::<u32>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n])?,
        ))
    }

    /// Minimum over the **last axis** of a 2-D `[N, C]` array, keepdims:
    /// `[N, C] → [N, 1]`. The min-reduction twin of `max_axis_last`.
    pub fn min_axis_last(&self) -> Result<Array<T>, ArrayError> {
        let (n, c) = self.two_d("min_axis_last")?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let def = build_rowreduce_def(c, ty, /* want_arg */ false, CmpOp::Lt);
        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, 1])?,
        ))
    }

    /// Index of the minimum over the **last axis** of a 2-D `[N, C]` array:
    /// `[N, C] → [N]` (`u32`). On ties, the first (lowest-index) minimum wins.
    /// The k-means assignment step (nearest centroid per point).
    pub fn argmin_last(&self) -> Result<Array<u32>, ArrayError> {
        let (n, c) = self.two_d("argmin_last")?;
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let def = build_rowreduce_def(c, ty, /* want_arg */ true, CmpOp::Lt);
        let out = self.gpu().field::<u32>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n])?,
        ))
    }

    fn two_d(&self, who: &str) -> Result<(usize, usize), ArrayError> {
        let d = self.shape();
        if d.len() != 2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                match who {
                    "max_axis_last" => "max_axis_last: input must be 2-D [N, C]",
                    "min_axis_last" => "min_axis_last: input must be 2-D [N, C]",
                    "argmin_last" => "argmin_last: input must be 2-D [N, C]",
                    _ => "argmax_last: input must be 2-D [N, C]",
                },
            )));
        }
        Ok((d[0], d[1]))
    }
}

/// Row-wise reduce over the last axis of `[N, C]`: thread `i` loops `c` over the
/// row tracking a running extremum (and, if `want_arg`, its arg-index)
/// branchlessly with a `seen` flag so the first in-bounds element always takes —
/// no float sentinel. `cmp` picks the extremum: `Gt` → running max, `Lt` →
/// running min. Writes the extremum value (want_arg=false) or its index (true).
fn build_rowreduce_def(c: usize, ty: ScalarType, want_arg: bool, cmp: CmpOp) -> KernelDef {
    let mut next = 1u32;
    let mut fresh = || {
        let r = next;
        next += 1;
        r
    };
    let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }]; // r0 = i (row)
    let c_c = Reg(fresh());
    body.push(uconst(c_c, c as u32));
    let base = Reg(fresh()); // i * C
    body.push(KernelOp::BinOp {
        dst: base,
        a: Reg(0),
        b: c_c,
        op: BinOp::Mul,
        ty: ScalarType::U32,
    });
    let one_c = Reg(fresh());
    body.push(uconst(one_c, 1));

    let acc = Reg(fresh());
    body.push(KernelOp::Const {
        dst: acc,
        value: zero_const(ty),
    });
    let arg = Reg(fresh());
    body.push(uconst(arg, 0));
    let seen = Reg(fresh());
    body.push(uconst(seen, 0));
    let cc = Reg(fresh()); // loop counter over columns

    // loop body: idx = base + cc ; v = src[idx] ;
    //   strict = v > acc ; win = strict OR !seen ; (seen always 1 after first)
    let idx = Reg(fresh());
    let v = Reg(fresh());
    let gtf = Reg(fresh());
    let strict = Reg(fresh());
    let notseen = Reg(fresh());
    let invstrict = Reg(fresh());
    let nsterm = Reg(fresh());
    let win = Reg(fresh());
    let gt_t = Reg(fresh());
    let one_t = Reg(fresh());
    let invgt = Reg(fresh());
    let keep = Reg(fresh());
    let take = Reg(fresh());
    let acc2 = Reg(fresh());
    let invwin = Reg(fresh());
    let keepa = Reg(fresh());
    let takea = Reg(fresh());
    let arg2 = Reg(fresh());

    let mut lb = vec![
        ubin(idx, base, cc, BinOp::Add),
        KernelOp::Load {
            dst: v,
            field: 0,
            index: idx,
            ty,
        },
        // strict = (v `cmp` acc) as u32  (Gt → max, Lt → min)
        KernelOp::Cmp {
            dst: gtf,
            a: v,
            b: acc,
            op: cmp,
            ty,
        },
        KernelOp::Cast {
            dst: strict,
            src: gtf,
            from: ScalarType::Bool,
            to: ScalarType::U32,
        },
        // win = strict + (1-seen)*(1-strict)
        ubin(notseen, one_c, seen, BinOp::Sub),
        ubin(invstrict, one_c, strict, BinOp::Sub),
        ubin(nsterm, notseen, invstrict, BinOp::Mul),
        ubin(win, strict, nsterm, BinOp::Add),
        // acc = acc*(1-win_t) + v*win_t
        KernelOp::Cast {
            dst: gt_t,
            src: win,
            from: ScalarType::U32,
            to: ty,
        },
        KernelOp::Const {
            dst: one_t,
            value: one_const(ty),
        },
        KernelOp::BinOp {
            dst: invgt,
            a: one_t,
            b: gt_t,
            op: BinOp::Sub,
            ty,
        },
        KernelOp::BinOp {
            dst: keep,
            a: acc,
            b: invgt,
            op: BinOp::Mul,
            ty,
        },
        KernelOp::BinOp {
            dst: take,
            a: v,
            b: gt_t,
            op: BinOp::Mul,
            ty,
        },
        KernelOp::BinOp {
            dst: acc2,
            a: keep,
            b: take,
            op: BinOp::Add,
            ty,
        },
        KernelOp::Copy {
            dst: acc,
            src: acc2,
            ty,
        },
        // arg = arg*(1-win) + cc*win  (u32)
        ubin(invwin, one_c, win, BinOp::Sub),
        ubin(keepa, arg, invwin, BinOp::Mul),
        ubin(takea, cc, win, BinOp::Mul),
        ubin(arg2, keepa, takea, BinOp::Add),
        KernelOp::Copy {
            dst: arg,
            src: arg2,
            ty: ScalarType::U32,
        },
        // seen = 1
        KernelOp::Copy {
            dst: seen,
            src: one_c,
            ty: ScalarType::U32,
        },
    ];
    let _ = &mut lb;
    body.push(KernelOp::Loop {
        count: c_c,
        iter_reg: cc,
        body: lb,
    });
    if want_arg {
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: arg,
            ty: ScalarType::U32,
        });
    } else {
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: acc,
            ty,
        });
    }

    let out_ty = if want_arg { ScalarType::U32 } else { ty };
    KernelDef {
        name: "qa_rowreduce".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "src".into(),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: out_ty,
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

fn one_const(ty: ScalarType) -> ConstValue {
    match ty {
        ScalarType::F32 => ConstValue::F32(1.0),
        ScalarType::F64 => ConstValue::F64(1.0),
        ScalarType::I32 => ConstValue::I32(1),
        _ => ConstValue::U32(1),
    }
}

fn zero_const(ty: ScalarType) -> ConstValue {
    match ty {
        ScalarType::F32 => ConstValue::F32(0.0),
        ScalarType::F64 => ConstValue::F64(0.0),
        ScalarType::I32 => ConstValue::I32(0),
        _ => ConstValue::U32(0),
    }
}

fn uconst(dst: Reg, v: u32) -> KernelOp {
    KernelOp::Const {
        dst,
        value: ConstValue::U32(v),
    }
}

fn ubin(dst: Reg, a: Reg, b: Reg, op: BinOp) -> KernelOp {
    KernelOp::BinOp {
        dst,
        a,
        b,
        op,
        ty: ScalarType::U32,
    }
}

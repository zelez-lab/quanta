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
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
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

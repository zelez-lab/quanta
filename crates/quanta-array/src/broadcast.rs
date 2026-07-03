//! Broadcasting elementwise binary ufuncs.
//!
//! When two arrays have different (but broadcast-compatible) shapes, the
//! generated kernel computes each operand's source offset per output
//! element from its broadcast strides. Both operands are materialized to
//! contiguous row-major first, then `quanta_tensor::Layout::broadcast`
//! gives the zero-stride-on-broadcast-axis strides; those strides + the
//! output shape are baked into the kernel as constants.
//!
//! Per output element with linear id `q` and output dims `d[0..r]`
//! (row-major out-strides `os[k] = prod(d[k+1..])`):
//!   coord[k] = (q / os[k]) % d[k]
//!   a_off    = Σ coord[k] * a_stride[k]
//!   b_off    = Σ coord[k] * b_stride[k]
//!   out[q]   = op(a[a_off], b[b_off])

use quanta::GpuType;
use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;

/// numpy broadcast of two shapes: right-align, each axis is max(a, b) where
/// the smaller must be 1. Returns the broadcast output shape.
pub(crate) fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ArrayError> {
    let r = a.len().max(b.len());
    let mut out = vec![0usize; r];
    for i in 0..r {
        let ad = if i < r - a.len() {
            1
        } else {
            a[i - (r - a.len())]
        };
        let bd = if i < r - b.len() {
            1
        } else {
            b[i - (r - b.len())]
        };
        out[i] = if ad == bd {
            ad
        } else if ad == 1 {
            bd
        } else if bd == 1 {
            ad
        } else {
            return Err(ArrayError::Layout(
                quanta_tensor::LayoutError::BroadcastIncompatible {
                    axis: i,
                    self_extent: ad,
                    target_extent: bd,
                },
            ));
        };
    }
    Ok(out)
}

/// How a binary ufunc kernel combines the two loaded operands.
pub(crate) enum Combine {
    Bin(quanta_ir::BinOp),
    Math(quanta_ir::MathFn),
    /// Elementwise comparison → a `{0, 1}` mask of the operand type.
    Cmp(quanta_ir::CmpOp),
}

impl<T: GpuType> Array<T> {
    /// Broadcasting binary ufunc. Used when `self.shape() != rhs.shape()`.
    pub(crate) fn broadcast_binary(
        &self,
        rhs: &Array<T>,
        combine: Combine,
    ) -> Result<Array<T>, ArrayError> {
        let out_shape = broadcast_shape(self.shape(), rhs.shape())?;
        let r = out_shape.len();

        // Materialize each operand to contiguous row-major, then broadcast
        // its row-major layout to the output shape (zero strides on
        // broadcast axes). Reading from offset-0 contiguous buffers.
        let a = self.contiguous_or_self()?;
        let b = rhs.contiguous_or_self()?;
        let a_bc = Layout::row_major(a.shape())?.broadcast(&out_shape)?;
        let b_bc = Layout::row_major(b.shape())?.broadcast(&out_shape)?;
        let a_str = a_bc.strides();
        let b_str = b_bc.strides();

        // Row-major output strides.
        let mut os = vec![1usize; r];
        for k in (0..r.saturating_sub(1)).rev() {
            os[k] = os[k + 1] * out_shape[k + 1];
        }
        let n: usize = out_shape.iter().product();
        let ty = T::scalar_type();

        // Build the index-math body. Registers:
        //   r0 = quark id (q)
        //   r1 = a_off accumulator, r2 = b_off accumulator
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let mut next = 1u32;
        let a_off = Reg(next);
        next += 1;
        let b_off = Reg(next);
        next += 1;
        body.push(uconst(a_off, 0));
        body.push(uconst(b_off, 0));

        for k in 0..r {
            // coord_k = (q / os[k]) % d[k]
            let div_c = Reg(next);
            next += 1;
            body.push(uconst(div_c, os[k] as u32));
            let dimk = Reg(next);
            next += 1;
            body.push(uconst(dimk, out_shape[k] as u32));
            let q_div = Reg(next);
            next += 1;
            body.push(ubin(q_div, Reg(0), div_c, quanta_ir::BinOp::Div));
            let coord = Reg(next);
            next += 1;
            body.push(ubin(coord, q_div, dimk, quanta_ir::BinOp::Rem));

            // a_off += coord * a_str[k]
            if a_str[k] != 0 {
                let as_c = Reg(next);
                next += 1;
                body.push(uconst(as_c, a_str[k] as u32));
                let term = Reg(next);
                next += 1;
                body.push(ubin(term, coord, as_c, quanta_ir::BinOp::Mul));
                let acc = Reg(next);
                next += 1;
                body.push(ubin(acc, a_off, term, quanta_ir::BinOp::Add));
                body.push(KernelOp::Copy {
                    dst: a_off,
                    src: acc,
                    ty: ScalarType::U32,
                });
            }
            // b_off += coord * b_str[k]
            if b_str[k] != 0 {
                let bs_c = Reg(next);
                next += 1;
                body.push(uconst(bs_c, b_str[k] as u32));
                let term = Reg(next);
                next += 1;
                body.push(ubin(term, coord, bs_c, quanta_ir::BinOp::Mul));
                let acc = Reg(next);
                next += 1;
                body.push(ubin(acc, b_off, term, quanta_ir::BinOp::Add));
                body.push(KernelOp::Copy {
                    dst: b_off,
                    src: acc,
                    ty: ScalarType::U32,
                });
            }
        }

        // Load operands at computed offsets, combine, store at q.
        let av = Reg(next);
        next += 1;
        let bv = Reg(next);
        next += 1;
        let res = Reg(next);
        next += 1;
        body.push(KernelOp::Load {
            dst: av,
            field: 0,
            index: a_off,
            ty,
        });
        body.push(KernelOp::Load {
            dst: bv,
            field: 1,
            index: b_off,
            ty,
        });
        match combine {
            Combine::Bin(op) => body.push(KernelOp::BinOp {
                dst: res,
                a: av,
                b: bv,
                op,
                ty,
            }),
            Combine::Math(func) => body.push(KernelOp::MathCall {
                dst: res,
                func,
                args: vec![av, bv],
                ty,
            }),
            Combine::Cmp(op) => {
                // compare → bool, then materialize a {0,1} mask of `ty`.
                let m = Reg(next);
                next += 1;
                body.push(KernelOp::Cmp {
                    dst: m,
                    a: av,
                    b: bv,
                    op,
                    ty,
                });
                body.push(KernelOp::Cast {
                    dst: res,
                    src: m,
                    from: quanta_ir::ScalarType::Bool,
                    to: ty,
                });
            }
        }
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: res,
            ty,
        });

        let def = KernelDef {
            name: "qa_broadcast".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "b".into(),
                    slot: 1,
                    scalar_type: ty,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 2,
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

        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, a.field_ref());
        wave.bind(1, b.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&out_shape)?,
        ))
    }
}

fn uconst(dst: Reg, v: u32) -> KernelOp {
    KernelOp::Const {
        dst,
        value: quanta_ir::ConstValue::U32(v),
    }
}

fn ubin(dst: Reg, a: Reg, b: Reg, op: quanta_ir::BinOp) -> KernelOp {
    KernelOp::BinOp {
        dst,
        a,
        b,
        op,
        ty: ScalarType::U32,
    }
}

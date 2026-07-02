//! `gather_last` — gather along the last axis by a runtime index array.
//!
//! Generalizes [`gather_rows`](crate::Array::gather_rows) (`K = 1`, output
//! squeezed) to `K` picks per row: for input `[…, D]` and index `[…, K]` with
//! matching leading dims, `out[…, j] = input[…, idx[…, j]]`. The dominant
//! data-dependent lookup in ML — embeddings, top-k selection, attention gather.
//! Other axes are reachable by `permute`-ing the target axis last, gathering,
//! and permuting back.
//!
//! One thread per output cell `q` (flat over `[R, K]`, `R` = product of leading
//! dims): `row = q / K`, `col = idx[q]`, `out[q] = input[row·D + col]`. No
//! atomics — every output cell is written once.

use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

/// Zero literal for a scalar type (the loop-carried accumulator seed).
fn zero_const(ty: ScalarType) -> ConstValue {
    match ty {
        ScalarType::F32 => ConstValue::F32(0.0),
        ScalarType::F64 => ConstValue::F64(0.0),
        ScalarType::I32 => ConstValue::I32(0),
        _ => ConstValue::U32(0),
    }
}

impl<T: ArrayScalar> Array<T> {
    /// Gather along the last axis: `input [L…, D]`, `idx [L…, K]` (same leading
    /// dims `L…`) → `out [L…, K]` where `out[l…, j] = input[l…, idx[l…, j]]`.
    /// Indices are assumed in range `[0, D)`.
    pub fn gather_last(&self, idx: &Array<u32>) -> Result<Array<T>, ArrayError> {
        let in_dims = self.shape().to_vec();
        let ix_dims = idx.shape().to_vec();
        if in_dims.is_empty() || ix_dims.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "gather_last: input and idx need at least one axis",
            )));
        }
        if in_dims[..in_dims.len() - 1] != ix_dims[..ix_dims.len() - 1] {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "gather_last: input and idx must share the leading dims",
            )));
        }
        let d = *in_dims.last().unwrap(); // source extent along the last axis
        let k = *ix_dims.last().unwrap(); // picks per row
        let r: usize = in_dims[..in_dims.len() - 1].iter().product(); // rows
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let idxc = idx.contiguous_or_self()?;
        let n_out = r * k;

        let mut next = 1u32;
        let mut fresh = || {
            let x = next;
            next += 1;
            x
        };
        // thread q: row = q / K ; col = idx[q] ; out[q] = input[row*D + col]
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let k_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(k_c),
            value: quanta_ir::ConstValue::U32(k as u32),
        });
        let d_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(d_c),
            value: quanta_ir::ConstValue::U32(d as u32),
        });
        let row = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(row),
            a: Reg(0),
            b: Reg(k_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        // col = idx[q]  (idx is contiguous, same [R,K] flat layout as out)
        let col = fresh();
        body.push(KernelOp::Load {
            dst: Reg(col),
            field: 1,
            index: Reg(0),
            ty: ScalarType::U32,
        });
        // off = row*D + col
        let base = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(base),
            a: Reg(row),
            b: Reg(d_c),
            op: BinOp::Mul,
            ty: ScalarType::U32,
        });
        let off = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(off),
            a: Reg(base),
            b: Reg(col),
            op: BinOp::Add,
            ty: ScalarType::U32,
        });
        let v = fresh();
        body.push(KernelOp::Load {
            dst: Reg(v),
            field: 0,
            index: Reg(off),
            ty,
        });
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(v),
            ty,
        });

        let def = KernelDef {
            name: "qa_gather_last".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "input".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "idx".into(),
                    slot: 1,
                    scalar_type: ScalarType::U32,
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
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, idxc.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&ix_dims)?,
        ))
    }

    /// Adjoint of [`gather_last`](Self::gather_last): scatter-add a gradient
    /// `[L…, K]` back into a `[L…, D]` array. `self` is the gradient; `idx` is
    /// the same index used in the forward. Cell `(row, col)` of the result
    /// accumulates `Σⱼ grad[row, j]` over every pick `j` with `idx[row, j] ==
    /// col` (so repeated picks add up). One thread per output cell loops over
    /// the `K` picks — no atomics.
    pub fn scatter_last_add(&self, idx: &Array<u32>, d: usize) -> Result<Array<T>, ArrayError> {
        let g_dims = self.shape().to_vec();
        if g_dims.is_empty() || g_dims != idx.shape() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "scatter_last_add: grad and idx must have the same shape [L…, K]",
            )));
        }
        let k = *g_dims.last().unwrap();
        let r: usize = g_dims[..g_dims.len() - 1].iter().product();
        let ty = T::scalar_type();
        let grad = self.contiguous_or_self()?;
        let idxc = idx.contiguous_or_self()?;
        let n_out = r * d;

        let mut next = 1u32;
        let mut fresh = || {
            let x = next;
            next += 1;
            x
        };
        let u = |dst: u32, v: u32| KernelOp::Const {
            dst: Reg(dst),
            value: quanta_ir::ConstValue::U32(v),
        };
        // thread q = row*D + col:  row = q / D ; col = q % D
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let d_c = fresh();
        body.push(u(d_c, d as u32));
        let k_c = fresh();
        body.push(u(k_c, k as u32));
        let row = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(row),
            a: Reg(0),
            b: Reg(d_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        let col = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(col),
            a: Reg(0),
            b: Reg(d_c),
            op: BinOp::Rem,
            ty: ScalarType::U32,
        });
        // gbase = row*K  (start of this row's K picks in grad/idx)
        let gbase = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(gbase),
            a: Reg(row),
            b: Reg(k_c),
            op: BinOp::Mul,
            ty: ScalarType::U32,
        });
        // acc = 0 (loop-carried); iter j over [0, K)
        let acc = fresh();
        body.push(KernelOp::Const {
            dst: Reg(acc),
            value: zero_const(ty),
        });
        let iter = fresh();
        let gj = fresh();
        let pick = fresh();
        let hit_b = fresh();
        let hit_t = fresh();
        let gv = fresh();
        let contrib = fresh();
        let acc2 = fresh();
        let loop_body = vec![
            // gj = gbase + j
            KernelOp::BinOp {
                dst: Reg(gj),
                a: Reg(gbase),
                b: Reg(iter),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            // pick = idx[gj]
            KernelOp::Load {
                dst: Reg(pick),
                field: 1,
                index: Reg(gj),
                ty: ScalarType::U32,
            },
            // hit = (pick == col) as T
            KernelOp::Cmp {
                dst: Reg(hit_b),
                a: Reg(pick),
                b: Reg(col),
                op: quanta_ir::CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Cast {
                dst: Reg(hit_t),
                src: Reg(hit_b),
                from: ScalarType::Bool,
                to: ty,
            },
            // gv = grad[gj]
            KernelOp::Load {
                dst: Reg(gv),
                field: 0,
                index: Reg(gj),
                ty,
            },
            // acc += gv * hit
            KernelOp::BinOp {
                dst: Reg(contrib),
                a: Reg(gv),
                b: Reg(hit_t),
                op: BinOp::Mul,
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(acc2),
                a: Reg(acc),
                b: Reg(contrib),
                op: BinOp::Add,
                ty,
            },
            KernelOp::Copy {
                dst: Reg(acc),
                src: Reg(acc2),
                ty,
            },
        ];
        body.push(KernelOp::Loop {
            count: Reg(k_c),
            iter_reg: Reg(iter),
            body: loop_body,
        });
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(acc),
            ty,
        });

        let def = KernelDef {
            name: "qa_scatter_last_add".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "grad".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "idx".into(),
                    slot: 1,
                    scalar_type: ScalarType::U32,
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
        let out = self.gpu().field::<T>(n_out)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, grad.field_ref());
        wave.bind(1, idxc.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;

        let mut out_dims = g_dims;
        *out_dims.last_mut().unwrap() = d;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&out_dims)?,
        ))
    }
}

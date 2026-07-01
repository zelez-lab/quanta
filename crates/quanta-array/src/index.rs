//! Index-array operations — data-dependent gather and its scatter adjoint.
//!
//! Unlike the strided-view materializer in [`gather`](crate::gather), these
//! index into a table with a **runtime index array** (`Load` at a dynamic
//! register). `gather_rows` is the row-wise pick `out[i] = table[i, idx[i]]`
//! (the piece a cross-entropy loss needs: select the logit at each row's label);
//! `scatter_rows_add` is its adjoint (route each row's gradient back to the
//! column it was picked from). Both are one-thread-per-output gather kernels —
//! no atomics.

use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Row-wise gather: for a `[N, C]` table and a `[N]` index array `idx`,
    /// return `[N]` where `out[i] = self[i, idx[i]]`. The core of a
    /// cross-entropy loss (pick each row's log-probability at its label).
    /// Indices are assumed in range `[0, C)`.
    pub fn gather_rows(&self, idx: &Array<u32>) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 2 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "gather_rows: table must be 2-D [N, C]",
            )));
        }
        let (n, c) = (d[0], d[1]);
        if idx.shape() != [n] {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "gather_rows: idx must be shape [N] matching the table rows",
            )));
        }
        let ty = T::scalar_type();
        let table = self.contiguous_or_self()?;
        let idxc = idx.contiguous_or_self()?;

        // thread i: col = idx[i]; out[i] = table[i*C + col]
        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }]; // r0 = i (row)
        let c_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(c_c),
            value: ConstValue::U32(c as u32),
        });
        // col = idx[i]
        let col = fresh();
        body.push(KernelOp::Load {
            dst: Reg(col),
            field: 1,
            index: Reg(0),
            ty: ScalarType::U32,
        });
        // base = i * C ; off = base + col
        let base = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(base),
            a: Reg(0),
            b: Reg(c_c),
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
            name: "qa_gather_rows".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "table".into(),
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
        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, table.field_ref());
        wave.bind(1, idxc.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n])?,
        ))
    }

    /// Adjoint of [`gather_rows`](Self::gather_rows): scatter a `[N]` gradient
    /// back into a `[N, C]` array, placing `grad[i]` at column `idx[i]` and 0
    /// elsewhere. `self` is the `[N]` gradient. One thread per output cell
    /// `(i, col)` writes `grad[i]` iff `idx[i] == col` (a gather — no atomics).
    pub fn scatter_rows_add(&self, idx: &Array<u32>, c: usize) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 1 {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "scatter_rows_add: grad must be 1-D [N]",
            )));
        }
        let n = d[0];
        if idx.shape() != [n] {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "scatter_rows_add: idx must be shape [N]",
            )));
        }
        let ty = T::scalar_type();
        let grad = self.contiguous_or_self()?;
        let idxc = idx.contiguous_or_self()?;
        let n_out = n * c;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        // thread q = i*C + col:  i = q / C ; col = q % C ;
        // out[q] = grad[i] · (idx[i] == col)
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let c_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(c_c),
            value: ConstValue::U32(c as u32),
        });
        let i_r = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(i_r),
            a: Reg(0),
            b: Reg(c_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        let col = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(col),
            a: Reg(0),
            b: Reg(c_c),
            op: BinOp::Rem,
            ty: ScalarType::U32,
        });
        // lab = idx[i]
        let lab = fresh();
        body.push(KernelOp::Load {
            dst: Reg(lab),
            field: 1,
            index: Reg(i_r),
            ty: ScalarType::U32,
        });
        // hit = (lab == col) as u32
        let hitb = fresh();
        body.push(KernelOp::Cmp {
            dst: Reg(hitb),
            a: Reg(lab),
            b: Reg(col),
            op: quanta_ir::CmpOp::Eq,
            ty: ScalarType::U32,
        });
        let hit = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(hit),
            src: Reg(hitb),
            from: ScalarType::Bool,
            to: ty,
        });
        // g = grad[i] ; out[q] = g · hit
        let g = fresh();
        body.push(KernelOp::Load {
            dst: Reg(g),
            field: 0,
            index: Reg(i_r),
            ty,
        });
        let outv = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(outv),
            a: Reg(g),
            b: Reg(hit),
            op: BinOp::Mul,
            ty,
        });
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(outv),
            ty,
        });

        let def = KernelDef {
            name: "qa_scatter_rows".into(),
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
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, c])?,
        ))
    }
}

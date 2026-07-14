//! `select_rows` — gather whole rows of a table by an index array (an
//! embedding lookup), and its scatter-add adjoint (the sparse gradient update).
//!
//! `table [V, E]` + `ids [B]` → `[B, E]` where `out[b] = table[ids[b], :]`. This
//! is the embedding-table lookup at the heart of word2vec and any categorical
//! feature. Its adjoint scatters each output row's gradient back to the table
//! row it was gathered from, accumulating when an id repeats — the "sparse
//! update" (untouched rows get zero).

use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

/// Zero literal for a scalar type (the scatter accumulator seed).
fn zero_const(ty: ScalarType) -> ConstValue {
    match ty {
        ScalarType::F32 => ConstValue::F32(0.0),
        ScalarType::F64 => ConstValue::F64(0.0),
        ScalarType::I32 => ConstValue::I32(0),
        _ => ConstValue::U32(0),
    }
}

impl<T: ArrayScalar> Array<T> {
    /// Select whole rows of this `[V, E]` table by `ids` `[B]`, giving `[B, E]`
    /// where `out[b] = self[ids[b], :]`. The embedding lookup. Ids are assumed
    /// in range `[0, V)`.
    pub fn select_rows(&self, ids: &Array<u32>) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 2 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "select_rows: table must be 2-D [V, E]",
            )));
        }
        if ids.shape().len() != 1 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "select_rows: ids must be 1-D [B]",
            )));
        }
        let e = d[1];
        let b = ids.shape()[0];
        let ty = T::scalar_type();
        let table = self.contiguous_or_self()?;
        let idxc = ids.contiguous_or_self()?;
        let n_out = b * e;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        // thread q = row*E + col:  row = q / E ; col = q % E ;
        //   src = ids[row]*E + col ; out[q] = table[src]
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let e_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(e_c),
            value: ConstValue::U32(e as u32),
        });
        let row = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(row),
            a: Reg(0),
            b: Reg(e_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        let col = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(col),
            a: Reg(0),
            b: Reg(e_c),
            op: BinOp::Rem,
            ty: ScalarType::U32,
        });
        let id = fresh();
        body.push(KernelOp::Load {
            dst: Reg(id),
            field: 1,
            index: Reg(row),
            ty: ScalarType::U32,
        });
        let base = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(base),
            a: Reg(id),
            b: Reg(e_c),
            op: BinOp::Mul,
            ty: ScalarType::U32,
        });
        let src = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(src),
            a: Reg(base),
            b: Reg(col),
            op: BinOp::Add,
            ty: ScalarType::U32,
        });
        let v = fresh();
        body.push(KernelOp::Load {
            dst: Reg(v),
            field: 0,
            index: Reg(src),
            ty,
        });
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(v),
            ty,
        });

        let def = KernelDef {
            name: "qa_select_rows".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "table".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "ids".into(),
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
        wave.bind(0, table.field_ref());
        wave.bind(1, idxc.field_ref());
        wave.bind(2, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[b, e])?,
        ))
    }

    /// Adjoint of [`select_rows`](Self::select_rows): scatter-add a gradient
    /// `[B, E]` (`self`) into a `[V, E]` table at the rows named by `ids` `[B]`,
    /// zero elsewhere. Repeated ids accumulate (the sparse embedding update).
    /// One thread per table cell loops the `B` lookups — no atomics.
    pub fn scatter_rows_into(
        &self,
        ids: &Array<u32>,
        v_rows: usize,
    ) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 2 || ids.shape().len() != 1 || ids.shape()[0] != d[0] {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "scatter_rows_into: grad must be [B, E] and ids [B]",
            )));
        }
        let (b, e) = (d[0], d[1]);
        let ty = T::scalar_type();
        let grad = self.contiguous_or_self()?;
        let idxc = ids.contiguous_or_self()?;
        let n_out = v_rows * e;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        let u = |dst: u32, val: u32| KernelOp::Const {
            dst: Reg(dst),
            value: ConstValue::U32(val),
        };
        // thread q = row*E + col:  row = q / E ; col = q % E
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let e_c = fresh();
        body.push(u(e_c, e as u32));
        let b_c = fresh();
        body.push(u(b_c, b as u32));
        let row = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(row),
            a: Reg(0),
            b: Reg(e_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        let col = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(col),
            a: Reg(0),
            b: Reg(e_c),
            op: BinOp::Rem,
            ty: ScalarType::U32,
        });
        // acc = 0 ; loop bb over [0, B): if ids[bb]==row, acc += grad[bb*E+col]
        let acc = fresh();
        body.push(KernelOp::Const {
            dst: Reg(acc),
            value: zero_const(ty),
        });
        let iter = fresh();
        let idv = fresh();
        let hit_b = fresh();
        let hit_t = fresh();
        let goff = fresh();
        let gidx = fresh();
        let gv = fresh();
        let contrib = fresh();
        let acc2 = fresh();
        let loop_body = vec![
            // id = ids[bb]
            KernelOp::Load {
                dst: Reg(idv),
                field: 1,
                index: Reg(iter),
                ty: ScalarType::U32,
            },
            // hit = (id == row) as T
            KernelOp::Cmp {
                dst: Reg(hit_b),
                a: Reg(idv),
                b: Reg(row),
                op: quanta_ir::CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Cast {
                dst: Reg(hit_t),
                src: Reg(hit_b),
                from: ScalarType::Bool,
                to: ty,
            },
            // gv = grad[bb*E + col]
            KernelOp::BinOp {
                dst: Reg(goff),
                a: Reg(iter),
                b: Reg(e_c),
                op: BinOp::Mul,
                ty: ScalarType::U32,
            },
            KernelOp::BinOp {
                dst: Reg(gidx),
                a: Reg(goff),
                b: Reg(col),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::Load {
                dst: Reg(gv),
                field: 0,
                index: Reg(gidx),
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
            count: Reg(b_c),
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
            name: "qa_scatter_rows_into".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "grad".into(),
                    slot: 0,
                    scalar_type: ty,
                },
                KernelParam::FieldRead {
                    name: "ids".into(),
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
            Layout::row_major(&[v_rows, e])?,
        ))
    }
}

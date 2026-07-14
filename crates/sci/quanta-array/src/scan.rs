//! `cumsum_last` — inclusive prefix sum along the last axis.
//!
//! `out[.., j] = Σ_{i ≤ j} self[.., i]`, row by row (one thread per row loops
//! the `L` columns, carrying a running sum and storing each partial). This is
//! the cumulative/scan the path-dependent examples need — running an
//! Asian-option average along a price path, a CDF from a histogram, etc.

use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

fn zero_const(ty: ScalarType) -> ConstValue {
    match ty {
        ScalarType::F32 => ConstValue::F32(0.0),
        ScalarType::F64 => ConstValue::F64(0.0),
        ScalarType::I32 => ConstValue::I32(0),
        _ => ConstValue::U32(0),
    }
}

impl<T: ArrayScalar> Array<T> {
    /// Inclusive prefix sum along the last axis of a 2-D `[R, L]` array:
    /// `out[r, j] = self[r, 0] + … + self[r, j]`, same shape. (A 1-D `[L]`
    /// array is treated as one row.)
    pub fn cumsum_last(&self) -> Result<Array<T>, ArrayError> {
        let dims = self.shape().to_vec();
        let (r, l) = match dims.len() {
            1 => (1usize, dims[0]),
            2 => (dims[0], dims[1]),
            _ => {
                return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                    "cumsum_last: input must be 1-D [L] or 2-D [R, L]",
                )));
            }
        };
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;

        let mut next = 1u32;
        let mut fresh = || {
            let x = next;
            next += 1;
            x
        };
        // thread r: base = r*L ; acc = 0 ; for j in 0..L { acc += in[base+j];
        //   out[base+j] = acc }
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let l_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(l_c),
            value: ConstValue::U32(l as u32),
        });
        let base = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(base),
            a: Reg(0),
            b: Reg(l_c),
            op: BinOp::Mul,
            ty: ScalarType::U32,
        });
        let acc = fresh();
        body.push(KernelOp::Const {
            dst: Reg(acc),
            value: zero_const(ty),
        });
        let iter = fresh();
        let idx = fresh();
        let v = fresh();
        let acc2 = fresh();
        let loop_body = vec![
            // idx = base + j
            KernelOp::BinOp {
                dst: Reg(idx),
                a: Reg(base),
                b: Reg(iter),
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            // v = in[idx] ; acc += v
            KernelOp::Load {
                dst: Reg(v),
                field: 0,
                index: Reg(idx),
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(acc2),
                a: Reg(acc),
                b: Reg(v),
                op: BinOp::Add,
                ty,
            },
            KernelOp::Copy {
                dst: Reg(acc),
                src: Reg(acc2),
                ty,
            },
            // out[idx] = acc
            KernelOp::Store {
                field: 1,
                index: Reg(idx),
                src: Reg(acc),
                ty,
            },
        ];
        body.push(KernelOp::Loop {
            count: Reg(l_c),
            iter_reg: Reg(iter),
            body: loop_body,
        });

        let def = KernelDef {
            name: "qa_cumsum_last".into(),
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
        let out = self.gpu().field::<T>(r * l)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, src.field_ref());
        wave.bind(1, &out);
        // one thread per row
        self.gpu().dispatch(&wave, r as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&dims)?,
        ))
    }
}

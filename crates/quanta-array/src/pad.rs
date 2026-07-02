//! `pad_axis0` — place a tensor into a larger zero tensor along axis 0.
//!
//! This is the adjoint of [`Array::narrow`](crate::Array::narrow) on axis 0
//! (the minibatch axis): given a gradient of shape `[len, rest…]`, scatter it
//! into a `[out_rows, rest…]` zero tensor at rows `[start, start+len)`. It is
//! also the forward of an axis-0 zero-pad.
//!
//! One thread per output cell `q` (flat, row-major). `row = q / rest`; the cell
//! is in the source window iff `start ≤ row < start+len`, in which case it
//! copies `src[q − start·rest]`, otherwise it writes 0. No atomics: every
//! output cell is written exactly once, and the source cells map one-to-one
//! onto the window.

use quanta_ir::{
    BinOp, CmpOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;
use crate::scalar::ArrayScalar;

impl<T: ArrayScalar> Array<T> {
    /// Scatter `self` (shape `[len, rest…]`) into a zero tensor of shape
    /// `[out_rows, rest…]`, placing its rows at `[start, start+len)` along
    /// axis 0. The adjoint of `narrow(0, start, len)`.
    pub fn pad_axis0(&self, out_rows: usize, start: usize) -> Result<Array<T>, ArrayError> {
        let dims = self.shape().to_vec();
        if dims.is_empty() {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pad_axis0: input must have at least one axis",
            )));
        }
        let len = dims[0];
        let rest: usize = dims[1..].iter().product();
        if start + len > out_rows {
            return Err(ArrayError::Gpu(quanta::QuantaError::invalid_param(
                "pad_axis0: start + len exceeds out_rows",
            )));
        }
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let n_out = out_rows * rest;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        // thread q ∈ [0, out_rows·rest):  row = q / rest
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let rest_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(rest_c),
            value: ConstValue::U32(rest as u32),
        });
        let row = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(row),
            a: Reg(0),
            b: Reg(rest_c),
            op: BinOp::Div,
            ty: ScalarType::U32,
        });
        // in = (row >= start) & (row < start+len), as {0,1}
        let start_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(start_c),
            value: ConstValue::U32(start as u32),
        });
        let end_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(end_c),
            value: ConstValue::U32((start + len) as u32),
        });
        let ge_b = fresh();
        body.push(KernelOp::Cmp {
            dst: Reg(ge_b),
            a: Reg(row),
            b: Reg(start_c),
            op: CmpOp::Ge,
            ty: ScalarType::U32,
        });
        let lt_b = fresh();
        body.push(KernelOp::Cmp {
            dst: Reg(lt_b),
            a: Reg(row),
            b: Reg(end_c),
            op: CmpOp::Lt,
            ty: ScalarType::U32,
        });
        let ge_u = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(ge_u),
            src: Reg(ge_b),
            from: ScalarType::Bool,
            to: ScalarType::U32,
        });
        let lt_u = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(lt_u),
            src: Reg(lt_b),
            from: ScalarType::Bool,
            to: ScalarType::U32,
        });
        let in_u = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(in_u),
            a: Reg(ge_u),
            b: Reg(lt_u),
            op: BinOp::Mul,
            ty: ScalarType::U32,
        });
        // src_q = q - start*rest  (only meaningful when in-window; clamped to 0
        // otherwise so the Load stays in-bounds, then masked out by `in`).
        let off_c = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off_c),
            value: ConstValue::U32((start * rest) as u32),
        });
        // clamped = in ? (q - off) : 0  →  (q - off) * in  is wrong pre-sub, so
        // guard: only subtract when q >= off. Since off = start*rest and a cell
        // is in-window iff q >= off and q < off + len*rest, `q - off` is a valid
        // index there; out-of-window we force src_q = 0 via `in`-masked select.
        let raw_q = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(raw_q),
            a: Reg(0),
            b: Reg(off_c),
            op: BinOp::Sub, // wrapping; only used when q >= off
            ty: ScalarType::U32,
        });
        let src_q = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(src_q),
            a: Reg(raw_q),
            b: Reg(in_u),
            op: BinOp::Mul, // 0 when out-of-window → reads src[0], masked below
            ty: ScalarType::U32,
        });
        let sv = fresh();
        body.push(KernelOp::Load {
            dst: Reg(sv),
            field: 0,
            index: Reg(src_q),
            ty,
        });
        let in_t = fresh();
        body.push(KernelOp::Cast {
            dst: Reg(in_t),
            src: Reg(in_u),
            from: ScalarType::U32,
            to: ty,
        });
        let outv = fresh();
        body.push(KernelOp::BinOp {
            dst: Reg(outv),
            a: Reg(sv),
            b: Reg(in_t),
            op: BinOp::Mul,
            ty,
        });
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(outv),
            ty,
        });

        let def = KernelDef {
            name: "qa_pad_axis0".into(),
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

        let mut out_dims = dims;
        out_dims[0] = out_rows;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&out_dims)?,
        ))
    }
}

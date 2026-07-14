//! Device-side strided gather — the on-GPU `contiguous` kernel.
//!
//! Materializing a strided / transposed / broadcast view used to round-trip
//! through host memory (`to_vec` + `from_slice`). This builds a kernel that
//! does the gather on the device instead, so the data never leaves the GPU:
//!
//! Per output element with linear id `q` and logical dims `d[0..r]`
//! (row-major out-strides `os[k] = prod(d[k+1..])`):
//!   coord[k] = (q / os[k]) % d[k]
//!   src_off  = base_offset + Σ coord[k] * src_stride[k]
//!   out[q]   = src[src_off]
//!
//! Strides and base offset are baked into the kernel as `u32` constants.
//! Views produced by this crate's layout ops have non-negative strides and a
//! non-negative base offset, so `u32` index math is sufficient.

use quanta_core::GpuType;
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;

impl<T: GpuType> Array<T> {
    /// Gather this (possibly strided) view into a fresh contiguous row-major
    /// array **on the device** — no host round-trip. Assumes the layout is
    /// not already the contiguous fast path (callers check `is_contiguous`).
    pub(crate) fn gather_contiguous(&self) -> Result<Array<T>, ArrayError> {
        let dims: Vec<usize> = self.shape().to_vec();
        let r = dims.len();
        let src_str = self.layout().strides();
        let base = self.layout().base_offset();
        let n = self.len();
        let ty = T::scalar_type();

        // Row-major output strides over the logical shape.
        let mut os = vec![1usize; r];
        for k in (0..r.saturating_sub(1)).rev() {
            os[k] = os[k + 1] * dims[k + 1];
        }

        // r0 = quark id (q); src_off accumulator seeded with the base offset.
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let mut next = 1u32;
        let src_off = Reg(next);
        next += 1;
        body.push(uconst(src_off, base as u32));

        for k in 0..r {
            // coord_k = (q / os[k]) % d[k]
            let div_c = Reg(next);
            next += 1;
            body.push(uconst(div_c, os[k] as u32));
            let dimk = Reg(next);
            next += 1;
            body.push(uconst(dimk, dims[k] as u32));
            let q_div = Reg(next);
            next += 1;
            body.push(ubin(q_div, Reg(0), div_c, BinOp::Div));
            let coord = Reg(next);
            next += 1;
            body.push(ubin(coord, q_div, dimk, BinOp::Rem));

            // src_off += coord * src_stride[k]  (skip zero-stride axes)
            if src_str[k] != 0 {
                let st_c = Reg(next);
                next += 1;
                body.push(uconst(st_c, src_str[k] as u32));
                let term = Reg(next);
                next += 1;
                body.push(ubin(term, coord, st_c, BinOp::Mul));
                let acc = Reg(next);
                next += 1;
                body.push(ubin(acc, src_off, term, BinOp::Add));
                body.push(KernelOp::Copy {
                    dst: src_off,
                    src: acc,
                    ty: ScalarType::U32,
                });
            }
        }

        // out[q] = src[src_off]
        let v = Reg(next);
        next += 1;
        body.push(KernelOp::Load {
            dst: v,
            field: 0,
            index: src_off,
            ty,
        });
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: v,
            ty,
        });

        let def = KernelDef {
            name: "qa_gather".into(),
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

        let out = self.gpu().field::<T>(n)?;
        let bytes = serialize_kernel(&def);
        let mut wave = self.gpu().wave_jit(&bytes)?;
        wave.bind(0, self.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&dims)?,
        ))
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

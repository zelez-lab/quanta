//! `upsample2d` — nearest-neighbour 2× (or k×) spatial upsampling for NCHW
//! tensors, and its adjoint. The decoder counterpart to pooling: pooling shrinks
//! `H×W`, upsampling grows it back, so a conv autoencoder can rebuild an image.
//!
//! Forward: `[N, C, H, W] → [N, C, H·k, W·k]`, `out[n,c,i,j] = in[n,c,i/k,j/k]`
//! (each source pixel replicated over a `k×k` block). One thread per output
//! cell. Adjoint: each source pixel sums the `k×k` output block it fed — one
//! thread per input cell, no atomics.

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
    /// Nearest-neighbour upsample an NCHW tensor by integer factor `k`:
    /// `[N, C, H, W] → [N, C, H·k, W·k]`, each pixel replicated over a `k×k`
    /// block.
    pub fn upsample2d(&self, k: usize) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 4 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "upsample2d: input must be 4-D NCHW",
            )));
        }
        if k == 0 {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "upsample2d: factor must be ≥ 1",
            )));
        }
        let (n, c, h, w) = (d[0], d[1], d[2], d[3]);
        let (oh, ow) = (h * k, w * k);
        let ty = T::scalar_type();
        let src = self.contiguous_or_self()?;
        let n_out = n * c * oh * ow;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        let u = |dst: u32, v: usize| KernelOp::Const {
            dst: Reg(dst),
            value: ConstValue::U32(v as u32),
        };
        let bin = |dst: u32, a: u32, b: u32, op: BinOp| KernelOp::BinOp {
            dst: Reg(dst),
            a: Reg(a),
            b: Reg(b),
            op,
            ty: ScalarType::U32,
        };
        // Decode q → (nc, oi, oj) over [N·C, OH, OW]; source (ii, ij) = (oi/k, oj/k).
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let ow_c = fresh();
        body.push(u(ow_c, ow));
        let oh_c = fresh();
        body.push(u(oh_c, oh));
        let k_c = fresh();
        body.push(u(k_c, k));
        let w_c = fresh();
        body.push(u(w_c, w));
        let h_c = fresh();
        body.push(u(h_c, h));

        let oj = fresh();
        body.push(bin(oj, 0, ow_c, BinOp::Rem));
        let q1 = fresh();
        body.push(bin(q1, 0, ow_c, BinOp::Div));
        let oi = fresh();
        body.push(bin(oi, q1, oh_c, BinOp::Rem));
        let nc = fresh();
        body.push(bin(nc, q1, oh_c, BinOp::Div)); // channel-major plane index N·C
        // ii = oi/k ; ij = oj/k
        let ii = fresh();
        body.push(bin(ii, oi, k_c, BinOp::Div));
        let ij = fresh();
        body.push(bin(ij, oj, k_c, BinOp::Div));
        // src = (nc*H + ii)*W + ij
        let t0 = fresh();
        body.push(bin(t0, nc, h_c, BinOp::Mul));
        let t1 = fresh();
        body.push(bin(t1, t0, ii, BinOp::Add));
        let t2 = fresh();
        body.push(bin(t2, t1, w_c, BinOp::Mul));
        let sidx = fresh();
        body.push(bin(sidx, t2, ij, BinOp::Add));
        let v = fresh();
        body.push(KernelOp::Load {
            dst: Reg(v),
            field: 0,
            index: Reg(sidx),
            ty,
        });
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(v),
            ty,
        });

        let def = KernelDef {
            name: "qa_upsample2d".into(),
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
            Layout::row_major(&[n, c, oh, ow])?,
        ))
    }

    /// Adjoint of [`upsample2d`](Self::upsample2d): downsample a `[N, C, H·k,
    /// W·k]` gradient by summing each `k×k` block back to one `[N, C, H, W]`
    /// cell. `self` is the upstream gradient.
    pub fn upsample2d_backward(
        &self,
        k: usize,
        h: usize,
        w: usize,
    ) -> Result<Array<T>, ArrayError> {
        let d = self.shape();
        if d.len() != 4 || d[2] != h * k || d[3] != w * k {
            return Err(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
                "upsample2d_backward: grad must be [N, C, H·k, W·k]",
            )));
        }
        let (n, c) = (d[0], d[1]);
        let (oh, ow) = (h * k, w * k);
        let ty = T::scalar_type();
        let grad = self.contiguous_or_self()?;
        let n_out = n * c * h * w;

        let mut next = 1u32;
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        let u = |dst: u32, v: usize| KernelOp::Const {
            dst: Reg(dst),
            value: ConstValue::U32(v as u32),
        };
        let bin = |dst: u32, a: u32, b: u32, op: BinOp| KernelOp::BinOp {
            dst: Reg(dst),
            a: Reg(a),
            b: Reg(b),
            op,
            ty: ScalarType::U32,
        };
        // thread q → (nc, ii, ij) over [N·C, H, W]; sum grad over the k×k block
        // rows [ii·k, ii·k+k) × cols [ij·k, ij·k+k) of the [.., OH, OW] plane.
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let w_c = fresh();
        body.push(u(w_c, w));
        let h_c = fresh();
        body.push(u(h_c, h));
        let k_c = fresh();
        body.push(u(k_c, k));
        let ow_c = fresh();
        body.push(u(ow_c, ow));

        let ij = fresh();
        body.push(bin(ij, 0, w_c, BinOp::Rem));
        let q1 = fresh();
        body.push(bin(q1, 0, w_c, BinOp::Div));
        let ii = fresh();
        body.push(bin(ii, q1, h_c, BinOp::Rem));
        let nc = fresh();
        body.push(bin(nc, q1, h_c, BinOp::Div));
        // base output row/col of this block
        let oi0 = fresh();
        body.push(bin(oi0, ii, k_c, BinOp::Mul));
        let oj0 = fresh();
        body.push(bin(oj0, ij, k_c, BinOp::Mul));
        // plane base = nc*OH*OW... compute nc*OH then later; OH = H*k
        let oh_c = fresh();
        body.push(u(oh_c, oh));
        let plane = fresh();
        body.push(bin(plane, nc, oh_c, BinOp::Mul)); // nc*OH
        // acc = Σ over ki,kj of grad[(plane + oi0+ki)*OW + oj0+kj]
        let acc = fresh();
        body.push(KernelOp::Const {
            dst: Reg(acc),
            value: zero_const(ty),
        });
        let ki = fresh();
        let oi = fresh();
        let rb0 = fresh();
        let rowbase = fresh();
        let kj = fresh();
        let oj = fresh();
        let gidx = fresh();
        let gv = fresh();
        let acc2 = fresh();
        // inner loop over kj: acc += grad[rowbase + oj0 + kj]
        let inner = vec![
            bin(oj, oj0, kj, BinOp::Add),
            bin(gidx, rowbase, oj, BinOp::Add),
            KernelOp::Load {
                dst: Reg(gv),
                field: 0,
                index: Reg(gidx),
                ty,
            },
            KernelOp::BinOp {
                dst: Reg(acc2),
                a: Reg(acc),
                b: Reg(gv),
                op: BinOp::Add,
                ty,
            },
            KernelOp::Copy {
                dst: Reg(acc),
                src: Reg(acc2),
                ty,
            },
        ];
        // outer loop over ki: oi = oi0+ki ; rowbase = (plane+oi)*OW
        let outer_body = vec![
            bin(oi, oi0, ki, BinOp::Add),
            bin(rb0, plane, oi, BinOp::Add),
            bin(rowbase, rb0, ow_c, BinOp::Mul),
            KernelOp::Loop {
                count: Reg(k_c),
                iter_reg: Reg(kj),
                body: inner,
            },
        ];
        body.push(KernelOp::Loop {
            count: Reg(k_c),
            iter_reg: Reg(ki),
            body: outer_body,
        });
        body.push(KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(acc),
            ty,
        });

        let def = KernelDef {
            name: "qa_upsample2d_backward".into(),
            params: vec![
                KernelParam::FieldRead {
                    name: "grad".into(),
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
        wave.bind(0, grad.field_ref());
        wave.bind(1, &out);
        self.gpu().dispatch(&wave, n_out as u32)?.wait()?;
        Ok(Array::from_parts(
            self.gpu().clone(),
            out,
            Layout::row_major(&[n, c, h, w])?,
        ))
    }
}

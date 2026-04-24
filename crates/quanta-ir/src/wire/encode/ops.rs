//! KernelOp serialization (write_kernel_op + write_kernel_ops).

use crate::KernelOp;

use super::header::Writer;
use super::helpers::{
    write_atomicop, write_binop, write_cmpop, write_const_value, write_mathfn, write_reg,
    write_scalar_type, write_unaryop,
};

// ---------------------------------------------------------------------------
// KernelOp  (47 variants, tags 0..46)
// ---------------------------------------------------------------------------
//
// Tag assignment (sequential):
//   0  Load
//   1  Store
//   2  SharedDecl
//   3  SharedLoad
//   4  SharedStore
//   5  BinOp
//   6  UnaryOp
//   7  Cmp
//   8  Branch
//   9  Loop
//  10  MathCall
//  11  QuarkId
//  12  QuarkCount
//  13  LocalId
//  14  GroupId
//  15  GroupSize
//  16  Barrier
//  17  AtomicOp
//  18  AtomicCas
//  19  WaveShuffle
//  20  WaveBallot
//  21  WaveAny
//  22  WaveAll
//  23  Cast
//  24  Const
//  25  VecConstruct
//  26  VecExtract
//  27  MatMul
//  28  TextureSample2D
//  29  TextureSample3D
//  30  TextureWrite2D
//  31  TextureSize
//  32  Copy
//  33  Break
//  34  Dispatch
//  35  DeviceCall
//  36  Bitcast
//  37  CountTrailingZeros
//  38  CountLeadingZeros
//  39  PopCount
//  40  Dot
//  41  SubgroupReduceAdd
//  42  SubgroupReduceMin
//  43  SubgroupReduceMax
//  44  SubgroupExclusiveAdd
//  45  SubgroupInclusiveAdd
//  46  TextureLoad2D
//  47  SubgroupSize
//  48  SharedDeclDyn
//  49  DebugPrint

fn write_kernel_op(w: &mut Writer, op: &KernelOp) {
    match op {
        // 0 — Load { dst, field, index, ty }
        KernelOp::Load {
            dst,
            field,
            index,
            ty,
        } => {
            w.u8(0);
            write_reg(w, dst);
            w.u32(*field);
            write_reg(w, index);
            write_scalar_type(w, ty);
        }

        // 1 — Store { field, index, src, ty }
        KernelOp::Store {
            field,
            index,
            src,
            ty,
        } => {
            w.u8(1);
            w.u32(*field);
            write_reg(w, index);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 2 — SharedDecl { id, ty, count }
        KernelOp::SharedDecl { id, ty, count } => {
            w.u8(2);
            w.u32(*id);
            write_scalar_type(w, ty);
            w.u32(*count);
        }

        // 3 — SharedLoad { dst, id, index, ty }
        KernelOp::SharedLoad { dst, id, index, ty } => {
            w.u8(3);
            write_reg(w, dst);
            w.u32(*id);
            write_reg(w, index);
            write_scalar_type(w, ty);
        }

        // 4 — SharedStore { id, index, src, ty }
        KernelOp::SharedStore { id, index, src, ty } => {
            w.u8(4);
            w.u32(*id);
            write_reg(w, index);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 5 — BinOp { dst, a, b, op, ty }
        KernelOp::BinOp { dst, a, b, op, ty } => {
            w.u8(5);
            write_reg(w, dst);
            write_reg(w, a);
            write_reg(w, b);
            write_binop(w, op);
            write_scalar_type(w, ty);
        }

        // 6 — UnaryOp { dst, a, op, ty }
        KernelOp::UnaryOp { dst, a, op, ty } => {
            w.u8(6);
            write_reg(w, dst);
            write_reg(w, a);
            write_unaryop(w, op);
            write_scalar_type(w, ty);
        }

        // 7 — Cmp { dst, a, b, op, ty }
        KernelOp::Cmp { dst, a, b, op, ty } => {
            w.u8(7);
            write_reg(w, dst);
            write_reg(w, a);
            write_reg(w, b);
            write_cmpop(w, op);
            write_scalar_type(w, ty);
        }

        // 8 — Branch { cond, then_ops, else_ops }
        KernelOp::Branch {
            cond,
            then_ops,
            else_ops,
        } => {
            w.u8(8);
            write_reg(w, cond);
            write_kernel_ops(w, then_ops);
            write_kernel_ops(w, else_ops);
        }

        // 9 — Loop { count, iter_reg, body }
        KernelOp::Loop {
            count,
            iter_reg,
            body,
        } => {
            w.u8(9);
            write_reg(w, count);
            write_reg(w, iter_reg);
            write_kernel_ops(w, body);
        }

        // 10 — MathCall { dst, func, args, ty }
        KernelOp::MathCall {
            dst,
            func,
            args,
            ty,
        } => {
            w.u8(10);
            write_reg(w, dst);
            write_mathfn(w, func);
            w.u32(args.len() as u32);
            for arg in args {
                write_reg(w, arg);
            }
            write_scalar_type(w, ty);
        }

        // 11 — QuarkId { dst }
        KernelOp::QuarkId { dst } => {
            w.u8(11);
            write_reg(w, dst);
        }

        // 12 — QuarkCount { dst }
        KernelOp::QuarkCount { dst } => {
            w.u8(12);
            write_reg(w, dst);
        }

        // 13 — LocalId { dst }
        KernelOp::LocalId { dst } => {
            w.u8(13);
            write_reg(w, dst);
        }

        // 14 — GroupId { dst }
        KernelOp::GroupId { dst } => {
            w.u8(14);
            write_reg(w, dst);
        }

        // 15 — GroupSize { dst }
        KernelOp::GroupSize { dst } => {
            w.u8(15);
            write_reg(w, dst);
        }

        // 16 — Barrier
        KernelOp::Barrier => {
            w.u8(16);
        }

        // 17 — AtomicOp { dst, field, index, val, op, ty }
        KernelOp::AtomicOp {
            dst,
            field,
            index,
            val,
            op,
            ty,
        } => {
            w.u8(17);
            write_reg(w, dst);
            w.u32(*field);
            write_reg(w, index);
            write_reg(w, val);
            write_atomicop(w, op);
            write_scalar_type(w, ty);
        }

        // 18 — AtomicCas { dst, field, index, expected, desired, ty }
        KernelOp::AtomicCas {
            dst,
            field,
            index,
            expected,
            desired,
            ty,
        } => {
            w.u8(18);
            write_reg(w, dst);
            w.u32(*field);
            write_reg(w, index);
            write_reg(w, expected);
            write_reg(w, desired);
            write_scalar_type(w, ty);
        }

        // 19 — WaveShuffle { dst, src, lane_delta, ty }
        KernelOp::WaveShuffle {
            dst,
            src,
            lane_delta,
            ty,
        } => {
            w.u8(19);
            write_reg(w, dst);
            write_reg(w, src);
            write_reg(w, lane_delta);
            write_scalar_type(w, ty);
        }

        // 20 — WaveBallot { dst, predicate }
        KernelOp::WaveBallot { dst, predicate } => {
            w.u8(20);
            write_reg(w, dst);
            write_reg(w, predicate);
        }

        // 21 — WaveAny { dst, predicate }
        KernelOp::WaveAny { dst, predicate } => {
            w.u8(21);
            write_reg(w, dst);
            write_reg(w, predicate);
        }

        // 22 — WaveAll { dst, predicate }
        KernelOp::WaveAll { dst, predicate } => {
            w.u8(22);
            write_reg(w, dst);
            write_reg(w, predicate);
        }

        // 23 — Cast { dst, src, from, to }
        KernelOp::Cast { dst, src, from, to } => {
            w.u8(23);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, from);
            write_scalar_type(w, to);
        }

        // 24 — Const { dst, value }
        KernelOp::Const { dst, value } => {
            w.u8(24);
            write_reg(w, dst);
            write_const_value(w, value);
        }

        // 25 — VecConstruct { dst, components, ty }
        KernelOp::VecConstruct {
            dst,
            components,
            ty,
        } => {
            w.u8(25);
            write_reg(w, dst);
            w.u32(components.len() as u32);
            for c in components {
                write_reg(w, c);
            }
            write_scalar_type(w, ty);
        }

        // 26 — VecExtract { dst, vec, component, ty }
        KernelOp::VecExtract {
            dst,
            vec,
            component,
            ty,
        } => {
            w.u8(26);
            write_reg(w, dst);
            write_reg(w, vec);
            w.u8(*component);
            write_scalar_type(w, ty);
        }

        // 27 — MatMul { dst, a, b, size, ty }
        KernelOp::MatMul {
            dst,
            a,
            b,
            size,
            ty,
        } => {
            w.u8(27);
            write_reg(w, dst);
            write_reg(w, a);
            write_reg(w, b);
            w.u8(*size);
            write_scalar_type(w, ty);
        }

        // 28 — TextureSample2D { dst, texture, x, y, ty }
        KernelOp::TextureSample2D {
            dst,
            texture,
            x,
            y,
            ty,
        } => {
            w.u8(28);
            write_reg(w, dst);
            w.u32(*texture);
            write_reg(w, x);
            write_reg(w, y);
            write_scalar_type(w, ty);
        }

        // 29 — TextureSample3D { dst, texture, x, y, z, ty }
        KernelOp::TextureSample3D {
            dst,
            texture,
            x,
            y,
            z,
            ty,
        } => {
            w.u8(29);
            write_reg(w, dst);
            w.u32(*texture);
            write_reg(w, x);
            write_reg(w, y);
            write_reg(w, z);
            write_scalar_type(w, ty);
        }

        // 30 — TextureWrite2D { texture, x, y, value, ty }
        KernelOp::TextureWrite2D {
            texture,
            x,
            y,
            value,
            ty,
        } => {
            w.u8(30);
            w.u32(*texture);
            write_reg(w, x);
            write_reg(w, y);
            write_reg(w, value);
            write_scalar_type(w, ty);
        }

        // 31 — TextureSize { dst_w, dst_h, texture }
        KernelOp::TextureSize {
            dst_w,
            dst_h,
            texture,
        } => {
            w.u8(31);
            write_reg(w, dst_w);
            write_reg(w, dst_h);
            w.u32(*texture);
        }

        // 32 — Copy { dst, src, ty }
        KernelOp::Copy { dst, src, ty } => {
            w.u8(32);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 33 — Break
        KernelOp::Break => {
            w.u8(33);
        }

        // 34 — Dispatch { wave, groups }
        KernelOp::Dispatch { wave, groups } => {
            w.u8(34);
            write_reg(w, wave);
            write_reg(w, &groups[0]);
            write_reg(w, &groups[1]);
            write_reg(w, &groups[2]);
        }

        // 35 — DeviceCall { dst, func_name, args, ty }
        KernelOp::DeviceCall {
            dst,
            func_name,
            args,
            ty,
        } => {
            w.u8(35);
            write_reg(w, dst);
            w.str(func_name);
            w.u32(args.len() as u32);
            for arg in args {
                write_reg(w, arg);
            }
            write_scalar_type(w, ty);
        }

        // 36 — Bitcast { dst, src, from, to }
        KernelOp::Bitcast { dst, src, from, to } => {
            w.u8(36);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, from);
            write_scalar_type(w, to);
        }

        // 37 — CountTrailingZeros { dst, src, ty }
        KernelOp::CountTrailingZeros { dst, src, ty } => {
            w.u8(37);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 38 — CountLeadingZeros { dst, src, ty }
        KernelOp::CountLeadingZeros { dst, src, ty } => {
            w.u8(38);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 39 — PopCount { dst, src, ty }
        KernelOp::PopCount { dst, src, ty } => {
            w.u8(39);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 40 — Dot { dst, a, b, ty, width }
        KernelOp::Dot {
            dst,
            a,
            b,
            ty,
            width,
        } => {
            w.u8(40);
            write_reg(w, dst);
            write_reg(w, a);
            write_reg(w, b);
            write_scalar_type(w, ty);
            w.u8(*width);
        }

        // 41 — SubgroupReduceAdd { dst, src, ty }
        KernelOp::SubgroupReduceAdd { dst, src, ty } => {
            w.u8(41);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 42 — SubgroupReduceMin { dst, src, ty }
        KernelOp::SubgroupReduceMin { dst, src, ty } => {
            w.u8(42);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 43 — SubgroupReduceMax { dst, src, ty }
        KernelOp::SubgroupReduceMax { dst, src, ty } => {
            w.u8(43);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 44 — SubgroupExclusiveAdd { dst, src, ty }
        KernelOp::SubgroupExclusiveAdd { dst, src, ty } => {
            w.u8(44);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 45 — SubgroupInclusiveAdd { dst, src, ty }
        KernelOp::SubgroupInclusiveAdd { dst, src, ty } => {
            w.u8(45);
            write_reg(w, dst);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }

        // 46 — TextureLoad2D { dst, texture, x, y, ty }
        KernelOp::TextureLoad2D {
            dst,
            texture,
            x,
            y,
            ty,
        } => {
            w.u8(46);
            write_reg(w, dst);
            w.u32(*texture);
            write_reg(w, x);
            write_reg(w, y);
            write_scalar_type(w, ty);
        }

        // 47 — SubgroupSize { dst }
        KernelOp::SubgroupSize { dst } => {
            w.u8(47);
            write_reg(w, dst);
        }

        // 48 — SharedDeclDyn { id, ty }
        KernelOp::SharedDeclDyn { id, ty } => {
            w.u8(48);
            w.u32(*id);
            write_scalar_type(w, ty);
        }

        // 49 — DebugPrint { src, ty }
        KernelOp::DebugPrint { src, ty } => {
            w.u8(49);
            write_reg(w, src);
            write_scalar_type(w, ty);
        }
        KernelOp::CooperativeMMA {
            dst,
            a,
            b,
            c,
            m,
            n,
            k,
            ty,
        } => {
            w.u8(50);
            write_reg(w, dst);
            write_reg(w, a);
            write_reg(w, b);
            write_reg(w, c);
            w.u8(*m);
            w.u8(*n);
            w.u8(*k);
            write_scalar_type(w, ty);
        }
    }
}

// ---------------------------------------------------------------------------
// Vec<KernelOp> helpers (u32 length prefix)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn write_kernel_ops(w: &mut Writer, ops: &[KernelOp]) {
    w.u32(ops.len() as u32);
    for op in ops {
        write_kernel_op(w, op);
    }
}

//! KernelOp deserialization (read_kernel_op + read_kernel_ops).

extern crate alloc;

use alloc::vec::Vec;

use crate::KernelOp;

use super::header::Reader;
use super::helpers::{
    read_atomicop, read_binop, read_cmpop, read_const_value, read_mathfn, read_memory_order,
    read_reg, read_scalar_type, read_unaryop,
};

// ---------------------------------------------------------------------------
// KernelOp  (50 variants, tags 0..49)
// ---------------------------------------------------------------------------

fn read_kernel_op(r: &mut Reader) -> Result<KernelOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        // 0 — Load
        0 => {
            let dst = read_reg(r)?;
            let field = r.u32()?;
            let index = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::Load {
                dst,
                field,
                index,
                ty,
            })
        }

        // 1 — Store
        1 => {
            let field = r.u32()?;
            let index = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::Store {
                field,
                index,
                src,
                ty,
            })
        }

        // 2 — SharedDecl
        2 => {
            let id = r.u32()?;
            let ty = read_scalar_type(r)?;
            let count = r.u32()?;
            Ok(KernelOp::SharedDecl { id, ty, count })
        }

        // 3 — SharedLoad
        3 => {
            let dst = read_reg(r)?;
            let id = r.u32()?;
            let index = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SharedLoad { dst, id, index, ty })
        }

        // 4 — SharedStore
        4 => {
            let id = r.u32()?;
            let index = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SharedStore { id, index, src, ty })
        }

        // 5 — BinOp
        5 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let b = read_reg(r)?;
            let op = read_binop(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::BinOp { dst, a, b, op, ty })
        }

        // 6 — UnaryOp
        6 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let op = read_unaryop(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::UnaryOp { dst, a, op, ty })
        }

        // 7 — Cmp
        7 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let b = read_reg(r)?;
            let op = read_cmpop(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::Cmp { dst, a, b, op, ty })
        }

        // 8 — Branch
        8 => {
            let cond = read_reg(r)?;
            let then_ops = read_kernel_ops(r)?;
            let else_ops = read_kernel_ops(r)?;
            Ok(KernelOp::Branch {
                cond,
                then_ops,
                else_ops,
            })
        }

        // 9 — Loop
        9 => {
            let count = read_reg(r)?;
            let iter_reg = read_reg(r)?;
            let body = read_kernel_ops(r)?;
            Ok(KernelOp::Loop {
                count,
                iter_reg,
                body,
            })
        }

        // 10 — MathCall
        10 => {
            let dst = read_reg(r)?;
            let func = read_mathfn(r)?;
            let len = r.u32()? as usize;
            let mut args = Vec::with_capacity(len);
            for _ in 0..len {
                args.push(read_reg(r)?);
            }
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::MathCall {
                dst,
                func,
                args,
                ty,
            })
        }

        // 11 — QuarkId
        11 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::QuarkId { dst })
        }

        // 12 — QuarkCount
        12 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::QuarkCount { dst })
        }

        // 13 — ProtonId
        13 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::ProtonId { dst })
        }

        // 14 — NucleusId
        14 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::NucleusId { dst })
        }

        // 15 — ProtonSize
        15 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::ProtonSize { dst })
        }

        // 16 — Barrier
        16 => Ok(KernelOp::Barrier),

        // 17 — AtomicOp
        17 => {
            let dst = read_reg(r)?;
            let field = r.u32()?;
            let index = read_reg(r)?;
            let val = read_reg(r)?;
            let op = read_atomicop(r)?;
            let ty = read_scalar_type(r)?;
            let order = read_memory_order(r)?;
            Ok(KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
                order,
            })
        }

        // 18 — AtomicCas
        18 => {
            let dst = read_reg(r)?;
            let field = r.u32()?;
            let index = read_reg(r)?;
            let expected = read_reg(r)?;
            let desired = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            let success_order = read_memory_order(r)?;
            let failure_order = read_memory_order(r)?;
            Ok(KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
                success_order,
                failure_order,
            })
        }

        // 19 — WaveShuffle
        19 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let lane_delta = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::WaveShuffle {
                dst,
                src,
                lane_delta,
                ty,
            })
        }

        // 20 — WaveBallot
        20 => {
            let dst = read_reg(r)?;
            let predicate = read_reg(r)?;
            Ok(KernelOp::WaveBallot { dst, predicate })
        }

        // 21 — WaveAny
        21 => {
            let dst = read_reg(r)?;
            let predicate = read_reg(r)?;
            Ok(KernelOp::WaveAny { dst, predicate })
        }

        // 22 — WaveAll
        22 => {
            let dst = read_reg(r)?;
            let predicate = read_reg(r)?;
            Ok(KernelOp::WaveAll { dst, predicate })
        }

        // 23 — Cast
        23 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let from = read_scalar_type(r)?;
            let to = read_scalar_type(r)?;
            Ok(KernelOp::Cast { dst, src, from, to })
        }

        // 24 — Const
        24 => {
            let dst = read_reg(r)?;
            let value = read_const_value(r)?;
            Ok(KernelOp::Const { dst, value })
        }

        // 25 — VecConstruct
        25 => {
            let dst = read_reg(r)?;
            let len = r.u32()? as usize;
            let mut components = Vec::with_capacity(len);
            for _ in 0..len {
                components.push(read_reg(r)?);
            }
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::VecConstruct {
                dst,
                components,
                ty,
            })
        }

        // 26 — VecExtract
        26 => {
            let dst = read_reg(r)?;
            let vec = read_reg(r)?;
            let component = r.u8()?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::VecExtract {
                dst,
                vec,
                component,
                ty,
            })
        }

        // 27 — MatMul
        27 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let b = read_reg(r)?;
            let size = r.u8()?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::MatMul {
                dst,
                a,
                b,
                size,
                ty,
            })
        }

        // 28 — TextureSample2D
        28 => {
            let dst = read_reg(r)?;
            let texture = r.u32()?;
            let x = read_reg(r)?;
            let y = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::TextureSample2D {
                dst,
                texture,
                x,
                y,
                ty,
            })
        }

        // 29 — TextureSample3D
        29 => {
            let dst = read_reg(r)?;
            let texture = r.u32()?;
            let x = read_reg(r)?;
            let y = read_reg(r)?;
            let z = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::TextureSample3D {
                dst,
                texture,
                x,
                y,
                z,
                ty,
            })
        }

        // 30 — TextureWrite2D
        30 => {
            let texture = r.u32()?;
            let x = read_reg(r)?;
            let y = read_reg(r)?;
            let value = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::TextureWrite2D {
                texture,
                x,
                y,
                value,
                ty,
            })
        }

        // 31 — TextureSize
        31 => {
            let dst_w = read_reg(r)?;
            let dst_h = read_reg(r)?;
            let texture = r.u32()?;
            Ok(KernelOp::TextureSize {
                dst_w,
                dst_h,
                texture,
            })
        }

        // 32 — Copy
        32 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::Copy { dst, src, ty })
        }

        // 33 — Break
        33 => Ok(KernelOp::Break),

        // 34 — Dispatch
        34 => {
            let wave = read_reg(r)?;
            let g0 = read_reg(r)?;
            let g1 = read_reg(r)?;
            let g2 = read_reg(r)?;
            Ok(KernelOp::Dispatch {
                wave,
                groups: [g0, g1, g2],
            })
        }

        // 35 — DeviceCall
        35 => {
            let dst = read_reg(r)?;
            let func_name = r.str()?;
            let len = r.u32()? as usize;
            let mut args = Vec::with_capacity(len);
            for _ in 0..len {
                args.push(read_reg(r)?);
            }
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::DeviceCall {
                dst,
                func_name,
                args,
                ty,
            })
        }

        // 36 — Bitcast
        36 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let from = read_scalar_type(r)?;
            let to = read_scalar_type(r)?;
            Ok(KernelOp::Bitcast { dst, src, from, to })
        }

        // 37 — CountTrailingZeros
        37 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::CountTrailingZeros { dst, src, ty })
        }

        // 38 — CountLeadingZeros
        38 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::CountLeadingZeros { dst, src, ty })
        }

        // 39 — PopCount
        39 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::PopCount { dst, src, ty })
        }

        // 40 — Dot
        40 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let b = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            let width = r.u8()?;
            Ok(KernelOp::Dot {
                dst,
                a,
                b,
                ty,
                width,
            })
        }

        // 41 — SubgroupReduceAdd
        41 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SubgroupReduceAdd { dst, src, ty })
        }

        // 42 — SubgroupReduceMin
        42 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SubgroupReduceMin { dst, src, ty })
        }

        // 43 — SubgroupReduceMax
        43 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SubgroupReduceMax { dst, src, ty })
        }

        // 44 — SubgroupExclusiveAdd
        44 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SubgroupExclusiveAdd { dst, src, ty })
        }

        // 45 — SubgroupInclusiveAdd
        45 => {
            let dst = read_reg(r)?;
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SubgroupInclusiveAdd { dst, src, ty })
        }

        // 46 — TextureLoad2D
        46 => {
            let dst = read_reg(r)?;
            let texture = r.u32()?;
            let x = read_reg(r)?;
            let y = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::TextureLoad2D {
                dst,
                texture,
                x,
                y,
                ty,
            })
        }

        // 47 — SubgroupSize
        47 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::SubgroupSize { dst })
        }

        // 48 — SharedDeclDyn
        48 => {
            let id = r.u32()?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::SharedDeclDyn { id, ty })
        }

        // 49 — DebugPrint
        49 => {
            let src = read_reg(r)?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::DebugPrint { src, ty })
        }
        50 => {
            let dst = read_reg(r)?;
            let a = read_reg(r)?;
            let b = read_reg(r)?;
            let c = read_reg(r)?;
            let m = r.u8()?;
            let n = r.u8()?;
            let k = r.u8()?;
            let ty = read_scalar_type(r)?;
            Ok(KernelOp::CooperativeMMA {
                dst,
                a,
                b,
                c,
                m,
                n,
                k,
                ty,
            })
        }

        // 51 — Fence
        51 => {
            let order = read_memory_order(r)?;
            Ok(KernelOp::Fence { order })
        }

        // 52 — SharedAtomicOp
        52 => {
            let dst = read_reg(r)?;
            let slot = r.u32()?;
            let index = read_reg(r)?;
            let val = read_reg(r)?;
            let op = read_atomicop(r)?;
            let ty = read_scalar_type(r)?;
            let order = read_memory_order(r)?;
            Ok(KernelOp::SharedAtomicOp {
                dst,
                slot,
                index,
                val,
                op,
                ty,
                order,
            })
        }

        _ => Err("invalid KernelOp tag"),
    }
}

// ---------------------------------------------------------------------------
// Vec<KernelOp> helpers (u32 length prefix)
// ---------------------------------------------------------------------------

pub(in crate::wire) fn read_kernel_ops(r: &mut Reader) -> Result<Vec<KernelOp>, &'static str> {
    let len = r.u32()? as usize;
    let mut ops = Vec::with_capacity(len);
    for _ in 0..len {
        ops.push(read_kernel_op(r)?);
    }
    Ok(ops)
}

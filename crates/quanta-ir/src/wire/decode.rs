//! Binary decoding: Reader + all read_* helpers.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    AtomicOp, BinOp, CmpOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, MathFn,
    Reg, ScalarType, UnaryOp,
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Zero-copy binary reader over a byte slice.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], &'static str> {
        if self.remaining() < n {
            return Err("unexpected end of input");
        }
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    // -- primitives --

    pub(crate) fn u8(&mut self) -> Result<u8, &'static str> {
        let b = self.take(1)?;
        Ok(b[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16, &'static str> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn i32(&mut self) -> Result<i32, &'static str> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn i64(&mut self) -> Result<i64, &'static str> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub(crate) fn f32(&mut self) -> Result<f32, &'static str> {
        let bits = self.u32()?;
        Ok(f32::from_bits(bits))
    }

    pub(crate) fn f64(&mut self) -> Result<f64, &'static str> {
        let bits = self.u64()?;
        Ok(f64::from_bits(bits))
    }

    pub(crate) fn bool_val(&mut self) -> Result<bool, &'static str> {
        let v = self.u8()?;
        match v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid bool tag"),
        }
    }

    // -- composites --

    pub(crate) fn str(&mut self) -> Result<String, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        core::str::from_utf8(b)
            .map(String::from)
            .map_err(|_| "invalid utf-8 in string")
    }

    pub(crate) fn bytes(&mut self) -> Result<Vec<u8>, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        Ok(b.to_vec())
    }

    pub(crate) fn option_str(&mut self) -> Result<Option<String>, &'static str> {
        let tag = self.u8()?;
        match tag {
            0 => Ok(None),
            1 => self.str().map(Some),
            _ => Err("invalid option tag"),
        }
    }

    pub(crate) fn option_bytes(&mut self) -> Result<Option<Vec<u8>>, &'static str> {
        let tag = self.u8()?;
        match tag {
            0 => Ok(None),
            1 => self.bytes().map(Some),
            _ => Err("invalid option tag"),
        }
    }
}

// ---------------------------------------------------------------------------
// ScalarType  (12 variants, tags 0..11)
// ---------------------------------------------------------------------------

pub(crate) fn read_scalar_type(r: &mut Reader) -> Result<ScalarType, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(ScalarType::F16),
        1 => Ok(ScalarType::F32),
        2 => Ok(ScalarType::F64),
        3 => Ok(ScalarType::U8),
        4 => Ok(ScalarType::U16),
        5 => Ok(ScalarType::U32),
        6 => Ok(ScalarType::U64),
        7 => Ok(ScalarType::I8),
        8 => Ok(ScalarType::I16),
        9 => Ok(ScalarType::I32),
        10 => Ok(ScalarType::I64),
        11 => Ok(ScalarType::Bool),
        _ => Err("invalid ScalarType tag"),
    }
}

// ---------------------------------------------------------------------------
// BinOp  (10 variants, tags 0..9)
// ---------------------------------------------------------------------------

fn read_binop(r: &mut Reader) -> Result<BinOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(BinOp::Add),
        1 => Ok(BinOp::Sub),
        2 => Ok(BinOp::Mul),
        3 => Ok(BinOp::Div),
        4 => Ok(BinOp::Rem),
        5 => Ok(BinOp::BitAnd),
        6 => Ok(BinOp::BitOr),
        7 => Ok(BinOp::BitXor),
        8 => Ok(BinOp::Shl),
        9 => Ok(BinOp::Shr),
        _ => Err("invalid BinOp tag"),
    }
}

// ---------------------------------------------------------------------------
// UnaryOp  (3 variants, tags 0..2)
// ---------------------------------------------------------------------------

fn read_unaryop(r: &mut Reader) -> Result<UnaryOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(UnaryOp::Neg),
        1 => Ok(UnaryOp::BitNot),
        2 => Ok(UnaryOp::LogicalNot),
        _ => Err("invalid UnaryOp tag"),
    }
}

// ---------------------------------------------------------------------------
// CmpOp  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

fn read_cmpop(r: &mut Reader) -> Result<CmpOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(CmpOp::Eq),
        1 => Ok(CmpOp::Ne),
        2 => Ok(CmpOp::Lt),
        3 => Ok(CmpOp::Le),
        4 => Ok(CmpOp::Gt),
        5 => Ok(CmpOp::Ge),
        _ => Err("invalid CmpOp tag"),
    }
}

// ---------------------------------------------------------------------------
// AtomicOp  (9 variants, tags 0..8)
// ---------------------------------------------------------------------------

fn read_atomicop(r: &mut Reader) -> Result<AtomicOp, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(AtomicOp::Add),
        1 => Ok(AtomicOp::Sub),
        2 => Ok(AtomicOp::Min),
        3 => Ok(AtomicOp::Max),
        4 => Ok(AtomicOp::And),
        5 => Ok(AtomicOp::Or),
        6 => Ok(AtomicOp::Xor),
        7 => Ok(AtomicOp::Exchange),
        8 => Ok(AtomicOp::CompareExchange),
        _ => Err("invalid AtomicOp tag"),
    }
}

// ---------------------------------------------------------------------------
// MathFn  (21 variants, tags 0..20)
// ---------------------------------------------------------------------------

fn read_mathfn(r: &mut Reader) -> Result<MathFn, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(MathFn::Sin),
        1 => Ok(MathFn::Cos),
        2 => Ok(MathFn::Tan),
        3 => Ok(MathFn::Asin),
        4 => Ok(MathFn::Acos),
        5 => Ok(MathFn::Atan),
        6 => Ok(MathFn::Atan2),
        7 => Ok(MathFn::Sqrt),
        8 => Ok(MathFn::Rsqrt),
        9 => Ok(MathFn::Exp),
        10 => Ok(MathFn::Exp2),
        11 => Ok(MathFn::Log),
        12 => Ok(MathFn::Log2),
        13 => Ok(MathFn::Pow),
        14 => Ok(MathFn::Abs),
        15 => Ok(MathFn::Min),
        16 => Ok(MathFn::Max),
        17 => Ok(MathFn::Clamp),
        18 => Ok(MathFn::Floor),
        19 => Ok(MathFn::Ceil),
        20 => Ok(MathFn::Round),
        21 => Ok(MathFn::Fma),
        _ => Err("invalid MathFn tag"),
    }
}

// ---------------------------------------------------------------------------
// Reg
// ---------------------------------------------------------------------------

fn read_reg(r: &mut Reader) -> Result<Reg, &'static str> {
    r.u32().map(Reg)
}

// ---------------------------------------------------------------------------
// ConstValue  (8 variants, tags 0..7)
// ---------------------------------------------------------------------------

pub(crate) fn read_const_value(r: &mut Reader) -> Result<ConstValue, &'static str> {
    let tag = r.u8()?;
    match tag {
        0 => Ok(ConstValue::F16(r.u16()?)),
        1 => Ok(ConstValue::F32(r.f32()?)),
        2 => Ok(ConstValue::F64(r.f64()?)),
        3 => Ok(ConstValue::U32(r.u32()?)),
        4 => Ok(ConstValue::U64(r.u64()?)),
        5 => Ok(ConstValue::I32(r.i32()?)),
        6 => Ok(ConstValue::I64(r.i64()?)),
        7 => Ok(ConstValue::Bool(r.bool_val()?)),
        _ => Err("invalid ConstValue tag"),
    }
}

// ---------------------------------------------------------------------------
// KernelParam  (6 variants, tags 0..5)
// ---------------------------------------------------------------------------

fn read_kernel_param(r: &mut Reader) -> Result<KernelParam, &'static str> {
    let tag = r.u8()?;
    let name = r.str()?;
    let slot = r.u32()?;
    let scalar_type = read_scalar_type(r)?;
    match tag {
        0 => Ok(KernelParam::FieldRead {
            name,
            slot,
            scalar_type,
        }),
        1 => Ok(KernelParam::FieldWrite {
            name,
            slot,
            scalar_type,
        }),
        2 => Ok(KernelParam::Constant {
            name,
            slot,
            scalar_type,
        }),
        3 => Ok(KernelParam::Texture2DRead {
            name,
            slot,
            scalar_type,
        }),
        4 => Ok(KernelParam::Texture2DWrite {
            name,
            slot,
            scalar_type,
        }),
        5 => Ok(KernelParam::Texture3DRead {
            name,
            slot,
            scalar_type,
        }),
        _ => Err("invalid KernelParam tag"),
    }
}

// ---------------------------------------------------------------------------
// KernelOp  (35 variants, tags 0..34)
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

        // 13 — LocalId
        13 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::LocalId { dst })
        }

        // 14 — GroupId
        14 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::GroupId { dst })
        }

        // 15 — GroupSize
        15 => {
            let dst = read_reg(r)?;
            Ok(KernelOp::GroupSize { dst })
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
            Ok(KernelOp::AtomicOp {
                dst,
                field,
                index,
                val,
                op,
                ty,
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
            Ok(KernelOp::AtomicCas {
                dst,
                field,
                index,
                expected,
                desired,
                ty,
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

        _ => Err("invalid KernelOp tag"),
    }
}

// ---------------------------------------------------------------------------
// Vec<KernelOp> helpers (u32 length prefix)
// ---------------------------------------------------------------------------

fn read_kernel_ops(r: &mut Reader) -> Result<Vec<KernelOp>, &'static str> {
    let len = r.u32()? as usize;
    let mut ops = Vec::with_capacity(len);
    for _ in 0..len {
        ops.push(read_kernel_op(r)?);
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------
// KernelDef
// ---------------------------------------------------------------------------

pub(crate) fn read_kernel_def(r: &mut Reader) -> Result<KernelDef, &'static str> {
    let name = r.str()?;
    let param_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        params.push(read_kernel_param(r)?);
    }
    let body = read_kernel_ops(r)?;
    let body_source = r.option_str()?;
    let next_reg = r.u32()?;
    let opt_level = r.u8()?;
    // device_sources: Vec<String> — appended after opt_level.
    // If there are no remaining bytes (old format), default to empty.
    let device_sources = if r.remaining() > 0 {
        let count = r.u32()? as usize;
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(r.str()?);
        }
        v
    } else {
        Vec::new()
    };
    Ok(KernelDef {
        name,
        params,
        body,
        body_source,
        next_reg,
        opt_level,
        device_sources,
    })
}

// ---------------------------------------------------------------------------
// CompilerOutput
// ---------------------------------------------------------------------------

pub(crate) fn read_compiler_output(r: &mut Reader) -> Result<CompilerOutput, &'static str> {
    let amd = r.option_bytes()?;
    let nvidia = r.option_bytes()?;
    let spirv = r.option_bytes()?;
    let metallib = r.option_bytes()?;
    let msl = r.option_str()?;
    let wgsl = r.option_str()?;
    let llvm_ir = r.option_bytes()?;
    Ok(CompilerOutput {
        amd,
        nvidia,
        spirv,
        metallib,
        msl,
        wgsl,
        llvm_ir,
    })
}

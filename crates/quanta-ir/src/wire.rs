//! Custom binary serialization for quanta-ir types.
//!
//! Replaces serde + bincode with a zero-dependency, no_std-compatible wire
//! format. All integers are little-endian. Strings and byte slices are
//! length-prefixed (u32). Options use a u8 tag (0 = None, 1 = Some). Enums
//! use a u8 discriminant followed by variant-specific fields.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    AtomicOp, BinOp, CmpOp, CompilerOutput, ConstValue, KernelDef, KernelOp, KernelParam, MathFn,
    Reg, ScalarType, UnaryOp,
};

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Append-only binary writer backed by a `Vec<u8>`.
pub struct Writer {
    buf: Vec<u8>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf
    }

    // -- primitives --

    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub fn bool_val(&mut self, v: bool) {
        self.buf.push(v as u8);
    }

    // -- composites --

    pub fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    pub fn option_str(&mut self, v: &Option<String>) {
        match v {
            None => self.u8(0),
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
        }
    }

    pub fn option_bytes(&mut self, v: &Option<Vec<u8>>) {
        match v {
            None => self.u8(0),
            Some(b) => {
                self.u8(1);
                self.bytes(b);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Zero-copy binary reader over a byte slice.
pub struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
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

    pub fn u8(&mut self) -> Result<u8, &'static str> {
        let b = self.take(1)?;
        Ok(b[0])
    }

    pub fn u16(&mut self) -> Result<u16, &'static str> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn i32(&mut self) -> Result<i32, &'static str> {
        let b = self.take(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn i64(&mut self) -> Result<i64, &'static str> {
        let b = self.take(8)?;
        Ok(i64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    pub fn f32(&mut self) -> Result<f32, &'static str> {
        let bits = self.u32()?;
        Ok(f32::from_bits(bits))
    }

    pub fn f64(&mut self) -> Result<f64, &'static str> {
        let bits = self.u64()?;
        Ok(f64::from_bits(bits))
    }

    pub fn bool_val(&mut self) -> Result<bool, &'static str> {
        let v = self.u8()?;
        match v {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err("invalid bool tag"),
        }
    }

    // -- composites --

    pub fn str(&mut self) -> Result<String, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        core::str::from_utf8(b)
            .map(String::from)
            .map_err(|_| "invalid utf-8 in string")
    }

    pub fn bytes(&mut self) -> Result<Vec<u8>, &'static str> {
        let len = self.u32()? as usize;
        let b = self.take(len)?;
        Ok(b.to_vec())
    }

    pub fn option_str(&mut self) -> Result<Option<String>, &'static str> {
        let tag = self.u8()?;
        match tag {
            0 => Ok(None),
            1 => self.str().map(Some),
            _ => Err("invalid option tag"),
        }
    }

    pub fn option_bytes(&mut self) -> Result<Option<Vec<u8>>, &'static str> {
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

fn write_scalar_type(w: &mut Writer, ty: &ScalarType) {
    let tag: u8 = match ty {
        ScalarType::F16 => 0,
        ScalarType::F32 => 1,
        ScalarType::F64 => 2,
        ScalarType::U8 => 3,
        ScalarType::U16 => 4,
        ScalarType::U32 => 5,
        ScalarType::U64 => 6,
        ScalarType::I8 => 7,
        ScalarType::I16 => 8,
        ScalarType::I32 => 9,
        ScalarType::I64 => 10,
        ScalarType::Bool => 11,
    };
    w.u8(tag);
}

fn read_scalar_type(r: &mut Reader) -> Result<ScalarType, &'static str> {
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

fn write_binop(w: &mut Writer, op: &BinOp) {
    let tag: u8 = match op {
        BinOp::Add => 0,
        BinOp::Sub => 1,
        BinOp::Mul => 2,
        BinOp::Div => 3,
        BinOp::Rem => 4,
        BinOp::BitAnd => 5,
        BinOp::BitOr => 6,
        BinOp::BitXor => 7,
        BinOp::Shl => 8,
        BinOp::Shr => 9,
    };
    w.u8(tag);
}

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

fn write_unaryop(w: &mut Writer, op: &UnaryOp) {
    let tag: u8 = match op {
        UnaryOp::Neg => 0,
        UnaryOp::BitNot => 1,
        UnaryOp::LogicalNot => 2,
    };
    w.u8(tag);
}

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

fn write_cmpop(w: &mut Writer, op: &CmpOp) {
    let tag: u8 = match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    };
    w.u8(tag);
}

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

fn write_atomicop(w: &mut Writer, op: &AtomicOp) {
    let tag: u8 = match op {
        AtomicOp::Add => 0,
        AtomicOp::Sub => 1,
        AtomicOp::Min => 2,
        AtomicOp::Max => 3,
        AtomicOp::And => 4,
        AtomicOp::Or => 5,
        AtomicOp::Xor => 6,
        AtomicOp::Exchange => 7,
        AtomicOp::CompareExchange => 8,
    };
    w.u8(tag);
}

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

fn write_mathfn(w: &mut Writer, f: &MathFn) {
    let tag: u8 = match f {
        MathFn::Sin => 0,
        MathFn::Cos => 1,
        MathFn::Tan => 2,
        MathFn::Asin => 3,
        MathFn::Acos => 4,
        MathFn::Atan => 5,
        MathFn::Atan2 => 6,
        MathFn::Sqrt => 7,
        MathFn::Rsqrt => 8,
        MathFn::Exp => 9,
        MathFn::Exp2 => 10,
        MathFn::Log => 11,
        MathFn::Log2 => 12,
        MathFn::Pow => 13,
        MathFn::Abs => 14,
        MathFn::Min => 15,
        MathFn::Max => 16,
        MathFn::Clamp => 17,
        MathFn::Floor => 18,
        MathFn::Ceil => 19,
        MathFn::Round => 20,
        MathFn::Fma => 21,
    };
    w.u8(tag);
}

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

fn write_reg(w: &mut Writer, r: &Reg) {
    w.u32(r.0);
}

fn read_reg(r: &mut Reader) -> Result<Reg, &'static str> {
    r.u32().map(Reg)
}

// ---------------------------------------------------------------------------
// ConstValue  (8 variants, tags 0..7)
// ---------------------------------------------------------------------------

fn write_const_value(w: &mut Writer, cv: &ConstValue) {
    match cv {
        ConstValue::F16(v) => {
            w.u8(0);
            w.u16(*v);
        }
        ConstValue::F32(v) => {
            w.u8(1);
            w.f32(*v);
        }
        ConstValue::F64(v) => {
            w.u8(2);
            w.f64(*v);
        }
        ConstValue::U32(v) => {
            w.u8(3);
            w.u32(*v);
        }
        ConstValue::U64(v) => {
            w.u8(4);
            w.u64(*v);
        }
        ConstValue::I32(v) => {
            w.u8(5);
            w.i32(*v);
        }
        ConstValue::I64(v) => {
            w.u8(6);
            w.i64(*v);
        }
        ConstValue::Bool(v) => {
            w.u8(7);
            w.bool_val(*v);
        }
    }
}

fn read_const_value(r: &mut Reader) -> Result<ConstValue, &'static str> {
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

fn write_kernel_param(w: &mut Writer, p: &KernelParam) {
    match p {
        KernelParam::FieldRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(0);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::FieldWrite {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(1);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Constant {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(2);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture2DRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(3);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture2DWrite {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(4);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
        KernelParam::Texture3DRead {
            name,
            slot,
            scalar_type,
        } => {
            w.u8(5);
            w.str(name);
            w.u32(*slot);
            write_scalar_type(w, scalar_type);
        }
    }
}

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
    }
}

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

        _ => Err("invalid KernelOp tag"),
    }
}

// ---------------------------------------------------------------------------
// Vec<KernelOp> helpers (u32 length prefix)
// ---------------------------------------------------------------------------

fn write_kernel_ops(w: &mut Writer, ops: &[KernelOp]) {
    w.u32(ops.len() as u32);
    for op in ops {
        write_kernel_op(w, op);
    }
}

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

fn write_kernel_def(w: &mut Writer, k: &KernelDef) {
    w.str(&k.name);
    w.u32(k.params.len() as u32);
    for p in &k.params {
        write_kernel_param(w, p);
    }
    write_kernel_ops(w, &k.body);
    w.option_str(&k.body_source);
    w.u32(k.next_reg);
    w.u8(k.opt_level);
}

fn read_kernel_def(r: &mut Reader) -> Result<KernelDef, &'static str> {
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
    Ok(KernelDef {
        name,
        params,
        body,
        body_source,
        next_reg,
        opt_level,
    })
}

// ---------------------------------------------------------------------------
// CompilerOutput
// ---------------------------------------------------------------------------

fn write_compiler_output(w: &mut Writer, o: &CompilerOutput) {
    w.option_bytes(&o.amd);
    w.option_bytes(&o.nvidia);
    w.option_bytes(&o.spirv);
    w.option_bytes(&o.metallib);
    w.option_str(&o.msl);
    w.option_str(&o.wgsl);
    w.option_bytes(&o.llvm_ir);
}

fn read_compiler_output(r: &mut Reader) -> Result<CompilerOutput, &'static str> {
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

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Serialize a [`KernelDef`] to wire bytes.
pub fn serialize_kernel(kernel: &KernelDef) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_kernel_def(&mut w, kernel);
    w.finish()
}

/// Deserialize a [`KernelDef`] from wire bytes.
pub fn deserialize_kernel(bytes: &[u8]) -> Result<KernelDef, &'static str> {
    let mut r = Reader::new(bytes);
    let k = read_kernel_def(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after KernelDef");
    }
    Ok(k)
}

/// Serialize a [`CompilerOutput`] to wire bytes.
pub fn serialize_output(output: &CompilerOutput) -> Vec<u8> {
    let mut w = Writer::with_capacity(256);
    write_compiler_output(&mut w, output);
    w.finish()
}

/// Deserialize a [`CompilerOutput`] from wire bytes.
pub fn deserialize_output(bytes: &[u8]) -> Result<CompilerOutput, &'static str> {
    let mut r = Reader::new(bytes);
    let o = read_compiler_output(&mut r)?;
    if r.remaining() != 0 {
        return Err("trailing bytes after CompilerOutput");
    }
    Ok(o)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty_kernel() {
        let k = KernelDef {
            name: String::from("test_kernel"),
            params: Vec::new(),
            body: Vec::new(),
            body_source: None,
            next_reg: 0,
            opt_level: 3,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k.name, k2.name);
        assert!(k2.params.is_empty());
        assert!(k2.body.is_empty());
        assert!(k2.body_source.is_none());
        assert_eq!(k2.next_reg, 0);
        assert_eq!(k2.opt_level, 3);
    }

    #[test]
    fn roundtrip_kernel_with_body_source() {
        let k = KernelDef {
            name: String::from("k"),
            params: vec![KernelParam::FieldRead {
                name: String::from("input"),
                slot: 0,
                scalar_type: ScalarType::F32,
            }],
            body: Vec::new(),
            body_source: Some(String::from("let x = input[gid];")),
            next_reg: 5,
            opt_level: 2,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body_source, Some(String::from("let x = input[gid];")));
        assert_eq!(k2.opt_level, 2);
    }

    #[test]
    fn roundtrip_kernel_ops() {
        let ops = vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::F32(3.14),
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(2),
                b: Reg(1),
                op: BinOp::Mul,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
            KernelOp::Barrier,
            KernelOp::Break,
        ];
        let k = KernelDef {
            name: String::from("mul_pi"),
            params: vec![
                KernelParam::FieldRead {
                    name: String::from("in"),
                    slot: 0,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::FieldWrite {
                    name: String::from("out"),
                    slot: 1,
                    scalar_type: ScalarType::F32,
                },
            ],
            body: ops,
            body_source: None,
            next_reg: 4,
            opt_level: 3,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 7);
        assert_eq!(k2.next_reg, 4);
    }

    #[test]
    fn roundtrip_branch_and_loop() {
        let k = KernelDef {
            name: String::from("branchy"),
            params: Vec::new(),
            body: vec![
                KernelOp::Branch {
                    cond: Reg(0),
                    then_ops: vec![KernelOp::Barrier],
                    else_ops: vec![KernelOp::Break],
                },
                KernelOp::Loop {
                    count: Reg(1),
                    iter_reg: Reg(2),
                    body: vec![KernelOp::Const {
                        dst: Reg(3),
                        value: ConstValue::Bool(true),
                    }],
                },
            ],
            body_source: None,
            next_reg: 4,
            opt_level: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 2);
    }

    #[test]
    fn roundtrip_compiler_output_empty() {
        let o = CompilerOutput {
            amd: None,
            nvidia: None,
            spirv: None,
            metallib: None,
            msl: None,
            wgsl: None,
            llvm_ir: None,
        };
        let bytes = serialize_output(&o);
        let o2 = deserialize_output(&bytes).unwrap();
        assert!(o2.amd.is_none());
        assert!(o2.msl.is_none());
    }

    #[test]
    fn roundtrip_compiler_output_full() {
        let o = CompilerOutput {
            amd: Some(vec![0xDE, 0xAD]),
            nvidia: Some(vec![0xBE, 0xEF]),
            spirv: Some(vec![0x03, 0x02, 0x23, 0x07]),
            metallib: Some(vec![0x4D, 0x54]),
            msl: Some(String::from("kernel void k() {}")),
            wgsl: Some(String::from("@compute fn k() {}")),
            llvm_ir: Some(vec![0xBC]),
        };
        let bytes = serialize_output(&o);
        let o2 = deserialize_output(&bytes).unwrap();
        assert_eq!(o2.amd, Some(vec![0xDE, 0xAD]));
        assert_eq!(o2.nvidia, Some(vec![0xBE, 0xEF]));
        assert_eq!(o2.msl, Some(String::from("kernel void k() {}")));
        assert_eq!(o2.wgsl, Some(String::from("@compute fn k() {}")));
    }

    #[test]
    fn trailing_bytes_rejected() {
        let k = KernelDef {
            name: String::from("x"),
            params: Vec::new(),
            body: Vec::new(),
            body_source: None,
            next_reg: 0,
            opt_level: 0,
        };
        let mut bytes = serialize_kernel(&k);
        bytes.push(0xFF);
        assert_eq!(
            deserialize_kernel(&bytes).unwrap_err(),
            "trailing bytes after KernelDef"
        );
    }

    #[test]
    fn truncated_input_rejected() {
        let bytes = [0x01]; // too short for any KernelDef
        assert!(deserialize_kernel(&bytes).is_err());
    }

    #[test]
    fn all_scalar_types_roundtrip() {
        use ScalarType::*;
        let types = [F16, F32, F64, U8, U16, U32, U64, I8, I16, I32, I64, Bool];
        for ty in &types {
            let mut w = Writer::new();
            write_scalar_type(&mut w, ty);
            let buf = w.finish();
            let mut r = Reader::new(&buf);
            let ty2 = read_scalar_type(&mut r).unwrap();
            assert_eq!(*ty, ty2);
        }
    }

    #[test]
    fn all_const_values_roundtrip() {
        let values = [
            ConstValue::F16(0x3C00),
            ConstValue::F32(1.0),
            ConstValue::F64(2.0),
            ConstValue::U32(42),
            ConstValue::U64(1_000_000),
            ConstValue::I32(-1),
            ConstValue::I64(-999),
            ConstValue::Bool(true),
        ];
        for cv in &values {
            let mut w = Writer::new();
            write_const_value(&mut w, cv);
            let buf = w.finish();
            let mut r = Reader::new(&buf);
            let _ = read_const_value(&mut r).unwrap();
        }
    }

    #[test]
    fn dispatch_roundtrip() {
        let op = KernelOp::Dispatch {
            wave: Reg(10),
            groups: [Reg(1), Reg(2), Reg(3)],
        };
        let k = KernelDef {
            name: String::from("d"),
            params: Vec::new(),
            body: vec![op],
            body_source: None,
            next_reg: 11,
            opt_level: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 1);
    }

    #[test]
    fn all_kernel_params_roundtrip() {
        let params = vec![
            KernelParam::FieldRead {
                name: String::from("a"),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: String::from("b"),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::Constant {
                name: String::from("c"),
                slot: 2,
                scalar_type: ScalarType::I32,
            },
            KernelParam::Texture2DRead {
                name: String::from("t0"),
                slot: 3,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Texture2DWrite {
                name: String::from("t1"),
                slot: 4,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Texture3DRead {
                name: String::from("t2"),
                slot: 5,
                scalar_type: ScalarType::F16,
            },
        ];
        let k = KernelDef {
            name: String::from("all_params"),
            params,
            body: Vec::new(),
            body_source: None,
            next_reg: 0,
            opt_level: 1,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.params.len(), 6);
    }

    #[test]
    fn texture_ops_roundtrip() {
        let ops = vec![
            KernelOp::TextureSample2D {
                dst: Reg(0),
                texture: 0,
                x: Reg(1),
                y: Reg(2),
                ty: ScalarType::F32,
            },
            KernelOp::TextureSample3D {
                dst: Reg(3),
                texture: 1,
                x: Reg(4),
                y: Reg(5),
                z: Reg(6),
                ty: ScalarType::F16,
            },
            KernelOp::TextureWrite2D {
                texture: 2,
                x: Reg(7),
                y: Reg(8),
                value: Reg(9),
                ty: ScalarType::F32,
            },
            KernelOp::TextureSize {
                dst_w: Reg(10),
                dst_h: Reg(11),
                texture: 0,
            },
        ];
        let k = KernelDef {
            name: String::from("tex"),
            params: Vec::new(),
            body: ops,
            body_source: None,
            next_reg: 12,
            opt_level: 3,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 4);
    }

    #[test]
    fn wave_ops_roundtrip() {
        let ops = vec![
            KernelOp::WaveShuffle {
                dst: Reg(0),
                src: Reg(1),
                lane_delta: Reg(2),
                ty: ScalarType::F32,
            },
            KernelOp::WaveBallot {
                dst: Reg(3),
                predicate: Reg(4),
            },
            KernelOp::WaveAny {
                dst: Reg(5),
                predicate: Reg(6),
            },
            KernelOp::WaveAll {
                dst: Reg(7),
                predicate: Reg(8),
            },
        ];
        let k = KernelDef {
            name: String::from("wave"),
            params: Vec::new(),
            body: ops,
            body_source: None,
            next_reg: 9,
            opt_level: 0,
        };
        let bytes = serialize_kernel(&k);
        let k2 = deserialize_kernel(&bytes).unwrap();
        assert_eq!(k2.body.len(), 4);
    }
}

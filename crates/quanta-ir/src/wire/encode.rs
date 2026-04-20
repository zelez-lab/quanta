//! Binary encoding: Writer + all write_* helpers.

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
pub(crate) struct Writer {
    buf: Vec<u8>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub(crate) fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub(crate) fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
        }
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        self.buf
    }

    // -- primitives --

    pub(crate) fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub(crate) fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn i32(&mut self, v: i32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub(crate) fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub(crate) fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }

    pub(crate) fn bool_val(&mut self, v: bool) {
        self.buf.push(v as u8);
    }

    // -- composites --

    pub(crate) fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn bytes(&mut self, b: &[u8]) {
        self.u32(b.len() as u32);
        self.buf.extend_from_slice(b);
    }

    pub(crate) fn option_str(&mut self, v: &Option<String>) {
        match v {
            None => self.u8(0),
            Some(s) => {
                self.u8(1);
                self.str(s);
            }
        }
    }

    pub(crate) fn option_bytes(&mut self, v: &Option<Vec<u8>>) {
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
// ScalarType  (12 variants, tags 0..11)
// ---------------------------------------------------------------------------

pub(crate) fn write_scalar_type(w: &mut Writer, ty: &ScalarType) {
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

// ---------------------------------------------------------------------------
// Reg
// ---------------------------------------------------------------------------

fn write_reg(w: &mut Writer, r: &Reg) {
    w.u32(r.0);
}

// ---------------------------------------------------------------------------
// ConstValue  (8 variants, tags 0..7)
// ---------------------------------------------------------------------------

pub(crate) fn write_const_value(w: &mut Writer, cv: &ConstValue) {
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
//  35  DeviceCall

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

// ---------------------------------------------------------------------------
// KernelDef
// ---------------------------------------------------------------------------

pub(crate) fn write_kernel_def(w: &mut Writer, k: &KernelDef) {
    w.str(&k.name);
    w.u32(k.params.len() as u32);
    for p in &k.params {
        write_kernel_param(w, p);
    }
    write_kernel_ops(w, &k.body);
    w.option_str(&k.body_source);
    w.u32(k.next_reg);
    w.u8(k.opt_level);
    // device_sources: Vec<String>
    w.u32(k.device_sources.len() as u32);
    for s in &k.device_sources {
        w.str(s);
    }
}

// ---------------------------------------------------------------------------
// CompilerOutput
// ---------------------------------------------------------------------------

pub(crate) fn write_compiler_output(w: &mut Writer, o: &CompilerOutput) {
    w.option_bytes(&o.amd);
    w.option_bytes(&o.nvidia);
    w.option_bytes(&o.spirv);
    w.option_bytes(&o.metallib);
    w.option_str(&o.msl);
    w.option_str(&o.wgsl);
    w.option_bytes(&o.llvm_ir);
}

//! Weight-only symmetric integer quantization — integer codes + scales ↔
//! `Array<f32>`.
//!
//! A quantized rank-2 leaf `[R, C]` is stored as integer codes (int8 at
//! native 1-byte stride, or int4 packed 8 nibbles per `u32` word) plus an
//! f32 scale grid. Scales tile the matrix: tile size `(gr, gc)` gives a
//! scales tensor of shape `[⌈R/gr⌉, ⌈C/gc⌉]`, and element `(r, c)`
//! dequantizes with `scales[(r/gr) · ⌈C/gc⌉ + (c/gc)]`. Every shipped
//! granularity is a choice of `(gr, gc)` over ONE kernel per codes kind:
//!
//! | [`Granularity`] spelling            | `(gr, gc)` |
//! |-------------------------------------|------------|
//! | `PerTensor`                         | `(R, C)`   |
//! | `PerChannel { axis: 0 }`            | `(1, C)`   |
//! | `PerChannel { axis: 1 }`            | `(R, 1)`   |
//! | `Group { axis: 0, size: g }`        | `(g, 1)`   |
//! | `Group { axis: 1, size: g }`        | `(1, g)`   |
//!
//! The arithmetic is `quanta_ir::dtype`'s reference conversions, used
//! directly (never re-spelled):
//!
//! - quantize: `clamp(round_ties_even(x / s), lo, hi)` (`quantize_sym`) —
//!   round-half-to-even, matching WGSL/MSL `round` and SPIR-V `Round`
//!   bit-for-bit;
//! - dequantize: `s · q` (`dequantize_sym`), one f32 multiply on an
//!   exactly-converted int — bit-exact per backend (op-matrix-pinned);
//! - int4 packing: 8 signed nibbles per `u32` word, **low nibble first**
//!   (`int4_pack` / `int4_unpack`, the GPTQ / llama.cpp layout), rows
//!   packed independently (`[R, ⌈C/8⌉]` words, final word of each row
//!   zero-padded).
//!
//! Scales are **max-abs per tile**: `s = max|w| / hi` (`hi` = 127 / 7), a
//! pure function of the weights. In-range inputs then satisfy the
//! round-trip bound `|s · round_te(x/s) − x| ≤ s/2`, the clamp never
//! fires, and code `−128` / `−8` is never produced. An all-zero tile
//! stores `s = 0` with all-zero codes and dequantizes exactly (documented,
//! not an error).
//!
//! [`QuantizedMatrix::dequantize`] is one device dispatch (`Load(I8|I4)` at
//! native stride → scale `Load(F32)` at the grid index →
//! `KernelOp::Dequantize` → `Store(F32)`); [`dequantize_host`] is its
//! bitwise host twin. Mode: symmetric only — no affine constructor exists;
//! the IR's `zero_point` operand and `Affine` arm stay reserved.

use quanta_ir::dtype::{int4_pack, int4_unpack, quant_range, quantize_sym};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, QuantLevel, QuantMode, QuantScheme,
    QuantStore, QuantValue, Reg, ScalarType, serialize_kernel,
};
use quanta_tensor::Layout;

use crate::array::Array;
use crate::error::ArrayError;

// ── Vocabulary ───────────────────────────────────────────────────────────

/// The quantized value type: symmetric int8 (`[-127, 127]` under max-abs
/// scales) or symmetric int4 (`[-7, 7]`), int4 stored packed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantDtype {
    /// Symmetric int8, one code per element at native 1-byte stride.
    Int8,
    /// Symmetric int4, 8 codes per `u32` word (low nibble first).
    Int4,
}

impl QuantDtype {
    /// Code width in bits (8 / 4).
    pub fn bits(self) -> u32 {
        match self {
            QuantDtype::Int8 => 8,
            QuantDtype::Int4 => 4,
        }
    }

    /// The positive end of the symmetric code range (127 / 7).
    fn hi(self) -> i32 {
        quant_range(self.bits()).1
    }

    fn value(self) -> QuantValue {
        match self {
            QuantDtype::Int8 => QuantValue::Q8S,
            QuantDtype::Int4 => QuantValue::Q4S,
        }
    }
}

/// Scale granularity over a rank-2 `[R, C]` leaf. Each spelling maps to a
/// tile size `(gr, gc)` (module docs table); the scales tensor has shape
/// `[⌈R/gr⌉, ⌈C/gc⌉]`. Axis 0 = rows, axis 1 = columns; any other axis,
/// a zero group size, or a group size exceeding the axis extent is a loud
/// [`QuantError::Granularity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale for the whole matrix — `(R, C)`.
    PerTensor,
    /// One scale per row (`axis: 0`) or per column (`axis: 1` — Linear's
    /// per-out-channel, the accuracy default).
    PerChannel { axis: usize },
    /// One scale per `size`-long group along `axis` (`axis: 0, size: g` =
    /// g input rows per output column — the int4 grouped form).
    Group { axis: usize, size: u32 },
}

impl Granularity {
    /// The tile size `(gr, gc)` this spelling denotes over a `[rows, cols]`
    /// matrix (the grid mapping in the module docs).
    pub fn tile(self, rows: usize, cols: usize) -> Result<(usize, usize), ArrayError> {
        match self {
            Granularity::PerTensor => Ok((rows, cols)),
            Granularity::PerChannel { axis: 0 } => Ok((1, cols)),
            Granularity::PerChannel { axis: 1 } => Ok((rows, 1)),
            Granularity::PerChannel { axis } => Err(QuantError::Granularity {
                what: format!("PerChannel {{ axis: {axis} }} — axis must be 0 (rows) or 1 (cols)"),
            }
            .into()),
            Granularity::Group { axis, size } => {
                if axis > 1 {
                    return Err(QuantError::Granularity {
                        what: format!(
                            "Group {{ axis: {axis}, size: {size} }} — axis must be 0 (rows) or 1 (cols)"
                        ),
                    }
                    .into());
                }
                let extent = if axis == 0 { rows } else { cols };
                if size == 0 || size as usize > extent {
                    return Err(QuantError::Granularity {
                        what: format!(
                            "Group {{ axis: {axis}, size: {size} }} — size must be in 1..={extent} \
                             (the axis extent)"
                        ),
                    }
                    .into());
                }
                Ok(if axis == 0 {
                    (size as usize, 1)
                } else {
                    (1, size as usize)
                })
            }
        }
    }

    /// IR scheme level for this granularity. Operand-sourcing metadata under
    /// the register-operand op form — no emitter reads it (verified: the
    /// lowering is `scale · f32(q)` with whatever scale register the kernel
    /// supplies); carried so the kernel states the TRUE scheme.
    fn level(self) -> QuantLevel {
        match self {
            Granularity::PerTensor => QuantLevel::PerTensor,
            Granularity::PerChannel { .. } => QuantLevel::PerChannel,
            Granularity::Group { size, .. } => QuantLevel::Block(size),
        }
    }
}

/// Ceiling division for grid extents.
fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Append a `Const U32` op, returning its register (kernel-builder helper).
fn emit_const(next: &mut u32, body: &mut Vec<KernelOp>, v: u32) -> u32 {
    let r = *next;
    *next += 1;
    body.push(KernelOp::Const {
        dst: Reg(r),
        value: ConstValue::U32(v),
    });
    r
}

/// Append a u32 `BinOp`, returning its register (kernel-builder helper).
fn emit_bin(next: &mut u32, body: &mut Vec<KernelOp>, a: u32, b: u32, op: BinOp) -> u32 {
    let r = *next;
    *next += 1;
    body.push(KernelOp::BinOp {
        dst: Reg(r),
        a: Reg(a),
        b: Reg(b),
        op,
        ty: ScalarType::U32,
    });
    r
}

/// Validate a `[rows, cols]` quantized-leaf shape: rank-2 with nonzero
/// extents (the array shape model rejects zero extents at construction;
/// the host twins enforce the same rule over raw slices).
fn check_dims(rows: usize, cols: usize) -> Result<(), ArrayError> {
    if rows == 0 || cols == 0 {
        return Err(QuantError::Rank {
            shape: vec![rows, cols],
        }
        .into());
    }
    Ok(())
}

// ── Error ────────────────────────────────────────────────────────────────

/// Quantization faults, wrapped as [`ArrayError::Quant`]. Messages are
/// self-contained: they name the offender and state the rule or the
/// workaround. The checkpoint-facing variants (`Format`, `NotSupported`,
/// `Scale`) are declared here so the whole taxonomy lives in one place;
/// the safetensors loader in quanta-nn constructs them with real leaf
/// names.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantError {
    /// Quantize/load of a leaf that is not rank-2 with nonzero extents.
    Rank { shape: Vec<usize> },
    /// Invalid granularity spelling: axis > 1, group size 0, or group size
    /// exceeding the axis extent.
    Granularity { what: String },
    /// Codes / scales / grid disagreement when assembling a
    /// [`QuantizedMatrix`] from parts (the reader seam).
    Grid { what: String },
    /// Checkpoint-format violation (orphan metadata, missing tensor, name
    /// collision, shape/grid/metadata mismatch, bad scheme string, unknown
    /// format version). `leaf` names the offending checkpoint entry.
    Format { leaf: String, what: String },
    /// Resident int8 codes on a backend without narrow 8-bit storage.
    NotSupported { leaf: String, backend: String },
    /// A non-finite or zero scale read from a checkpoint (corrupt or
    /// hostile file). `tile` is the flat index into the scales tensor.
    Scale {
        leaf: String,
        tile: usize,
        value: f32,
    },
}

impl core::fmt::Display for QuantError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantError::Rank { shape } => write!(
                f,
                "quantized leaves must be rank-2 with nonzero extents, got shape {shape:?} \
                 (reshape rank-2 views before quantizing)"
            ),
            QuantError::Granularity { what } => {
                write!(f, "invalid quantization granularity: {what}")
            }
            QuantError::Grid { what } => {
                write!(f, "quantized matrix parts disagree: {what}")
            }
            QuantError::Format { leaf, what } => {
                write!(f, "quantized checkpoint leaf `{leaf}`: {what}")
            }
            QuantError::NotSupported { leaf, backend } => write!(
                f,
                "leaf `{leaf}`: resident int8 codes need narrow 8-bit storage, which the \
                 {backend} backend does not support — load via the f32 path \
                 (dequantize-on-load) instead"
            ),
            QuantError::Scale { leaf, tile, value } => write!(
                f,
                "leaf `{leaf}`: scale at tile {tile} is {value} (non-finite or zero) — \
                 the checkpoint is corrupt or hostile"
            ),
        }
    }
}

impl std::error::Error for QuantError {}

// ── Host reference (the bitwise twin) ────────────────────────────────────

/// Host-side quantized codes: int8 one `i8` per element (`rows · cols`),
/// or int4 packed 8 nibbles per `u32` word, rows packed independently
/// (`rows · ⌈cols/8⌉` words, low nibble first, final word per row
/// zero-padded).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostCodes {
    Int8(Vec<i8>),
    Int4Packed(Vec<u32>),
}

impl HostCodes {
    /// The code dtype this payload carries.
    pub fn dtype(&self) -> QuantDtype {
        match self {
            HostCodes::Int8(_) => QuantDtype::Int8,
            HostCodes::Int4Packed(_) => QuantDtype::Int4,
        }
    }
}

/// Max-abs per-tile scales over `[rows, cols]` weights: `s = max|w| / hi`.
/// Returns the row-major `[⌈rows/gr⌉ · ⌈cols/gc⌉]` scale vector.
fn max_abs_scales(
    w: &[f32],
    rows: usize,
    cols: usize,
    (gr, gc): (usize, usize),
    hi: i32,
) -> Vec<f32> {
    let (tr, tc) = (div_ceil(rows, gr), div_ceil(cols, gc));
    let mut maxabs = vec![0.0f32; tr * tc];
    for r in 0..rows {
        for c in 0..cols {
            let m = &mut maxabs[(r / gr) * tc + (c / gc)];
            *m = m.max(w[r * cols + c].abs());
        }
    }
    maxabs.iter().map(|&m| m / hi as f32).collect()
}

/// Quantize `[rows, cols]` weights on the host: max-abs per-tile scales,
/// then `quanta_ir::dtype::quantize_sym` (round-ties-even + clamp) per
/// element. Returns `(codes, scales)` with scales in row-major
/// `[⌈rows/gr⌉, ⌈cols/gc⌉]` order. The reference the device path is
/// pinned against, and the quantizer [`QuantizedMatrix::quantize`] runs.
pub fn quantize_host(
    w: &[f32],
    rows: usize,
    cols: usize,
    dtype: QuantDtype,
    granularity: Granularity,
) -> Result<(HostCodes, Vec<f32>), ArrayError> {
    check_dims(rows, cols)?;
    if w.len() != rows * cols {
        return Err(ArrayError::LengthMismatch {
            expected: rows * cols,
            got: w.len(),
        });
    }
    let (gr, gc) = granularity.tile(rows, cols)?;
    let tc = div_ceil(cols, gc);
    let scales = max_abs_scales(w, rows, cols, (gr, gc), dtype.hi());
    let bits = dtype.bits();
    // An all-zero tile has scale 0: its codes are 0 by definition (x/0 is
    // not evaluated), and dequantize to exact zeros.
    let code = |r: usize, c: usize| -> i32 {
        let s = scales[(r / gr) * tc + (c / gc)];
        if s == 0.0 {
            0
        } else {
            quantize_sym(w[r * cols + c], s, bits)
        }
    };
    let codes = match dtype {
        QuantDtype::Int8 => {
            let mut q = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    q.push(code(r, c) as i8);
                }
            }
            HostCodes::Int8(q)
        }
        QuantDtype::Int4 => {
            let wpr = div_ceil(cols, 8);
            let mut words = vec![0u32; rows * wpr];
            for r in 0..rows {
                for c in 0..cols {
                    let word = &mut words[r * wpr + c / 8];
                    *word = int4_pack(*word, (c % 8) as u32, code(r, c));
                }
            }
            HostCodes::Int4Packed(words)
        }
    };
    Ok((codes, scales))
}

/// Host dequantize — the bitwise twin of the device kernel: per element,
/// `quanta_ir::dtype::dequantize_sym(q, s)` with the scale at the grid
/// index. Also the engine of the universal (Mode A) load path. `scales`
/// is the row-major `[⌈rows/gr⌉ · ⌈cols/gc⌉]` grid.
pub fn dequantize_host(
    codes: &HostCodes,
    scales: &[f32],
    rows: usize,
    cols: usize,
    granularity: Granularity,
) -> Result<Vec<f32>, ArrayError> {
    check_dims(rows, cols)?;
    let (gr, gc) = granularity.tile(rows, cols)?;
    let (tr, tc) = (div_ceil(rows, gr), div_ceil(cols, gc));
    if scales.len() != tr * tc {
        return Err(ArrayError::LengthMismatch {
            expected: tr * tc,
            got: scales.len(),
        });
    }
    let expected_codes = match codes {
        HostCodes::Int8(q) => (q.len(), rows * cols),
        HostCodes::Int4Packed(words) => (words.len(), rows * div_ceil(cols, 8)),
    };
    if expected_codes.0 != expected_codes.1 {
        return Err(ArrayError::LengthMismatch {
            expected: expected_codes.1,
            got: expected_codes.0,
        });
    }
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let s = scales[(r / gr) * tc + (c / gc)];
            let q = match codes {
                HostCodes::Int8(q) => q[r * cols + c] as i32,
                HostCodes::Int4Packed(words) => {
                    int4_unpack(words[r * div_ceil(cols, 8) + c / 8], (c % 8) as u32)
                }
            };
            out.push(quanta_ir::dtype::dequantize_sym(q, s));
        }
    }
    Ok(out)
}

// ── The device-resident record ───────────────────────────────────────────

/// Device-resident quantized codes: int8 as a native-stride `Array<i8>`
/// of shape `[R, C]`, or int4 packed into `Array<u32>` words of shape
/// `[R, ⌈C/8⌉]` (low nibble first; the logical column count lives on the
/// owning [`QuantizedMatrix`]).
pub enum QuantCodes {
    Int8(Array<i8>),
    Int4Packed(Array<u32>),
}

impl QuantCodes {
    /// The code dtype this payload carries.
    pub fn dtype(&self) -> QuantDtype {
        match self {
            QuantCodes::Int8(_) => QuantDtype::Int8,
            QuantCodes::Int4Packed(_) => QuantDtype::Int4,
        }
    }
}

// Metadata-only Debug (no device readback), variant + storage shape.
impl core::fmt::Debug for QuantCodes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuantCodes::Int8(a) => write!(f, "Int8({:?})", a.shape()),
            QuantCodes::Int4Packed(a) => write!(f, "Int4Packed({:?})", a.shape()),
        }
    }
}

/// A quantized rank-2 weight: integer codes + the f32 scale grid, both
/// device-resident. Built by [`quantize`](QuantizedMatrix::quantize) (in
/// stack) or [`from_parts`](QuantizedMatrix::from_parts) (a checkpoint
/// reader constructing the record from loaded tensors);
/// [`dequantize`](QuantizedMatrix::dequantize) is one device dispatch
/// back to `Array<f32>`.
pub struct QuantizedMatrix {
    codes: QuantCodes,
    scales: Array<f32>,
    rows: usize,
    cols: usize,
    /// Tile size — `granularity.tile(rows, cols)`, cached.
    gr: usize,
    gc: usize,
    granularity: Granularity,
}

// Metadata-only Debug (no device readback).
impl core::fmt::Debug for QuantizedMatrix {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("QuantizedMatrix")
            .field("shape", &[self.rows, self.cols])
            .field("dtype", &self.dtype())
            .field("granularity", &self.granularity)
            .field("tile", &(self.gr, self.gc))
            .finish_non_exhaustive()
    }
}

impl QuantizedMatrix {
    /// Quantize a rank-2 `[R, C]` f32 array: max-abs per-tile scales +
    /// round-ties-even codes (host-side scan over a readback), uploaded as
    /// the device-resident record. Non-rank-2 input is a loud
    /// [`QuantError::Rank`].
    pub fn quantize(
        a: &Array<f32>,
        dtype: QuantDtype,
        granularity: Granularity,
    ) -> Result<Self, ArrayError> {
        if a.rank() != 2 {
            return Err(QuantError::Rank {
                shape: a.shape().to_vec(),
            }
            .into());
        }
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        let (gr, gc) = granularity.tile(rows, cols)?;
        let w = a.to_vec()?;
        let (codes, scales) = quantize_host(&w, rows, cols, dtype, granularity)?;
        let gpu = a.gpu();
        let (tr, tc) = (div_ceil(rows, gr), div_ceil(cols, gc));
        let scales = Array::from_vec(gpu, scales, &[tr, tc])?;
        let codes = match codes {
            HostCodes::Int8(q) => QuantCodes::Int8(Array::from_vec(gpu, q, &[rows, cols])?),
            HostCodes::Int4Packed(words) => {
                QuantCodes::Int4Packed(Array::from_vec(gpu, words, &[rows, div_ceil(cols, 8)])?)
            }
        };
        Ok(QuantizedMatrix {
            codes,
            scales,
            rows,
            cols,
            gr,
            gc,
            granularity,
        })
    }

    /// Assemble the record from already-resident parts — the seam a
    /// checkpoint reader uses after uploading `x.q` / `x.qs` tensors.
    /// `shape` is the logical `[R, C]` (int4's packed words cannot recover
    /// `C` alone). Validation is total and loud: codes shape must be
    /// `[R, C]` (int8) / `[R, ⌈C/8⌉]` (int4), scales shape must be
    /// `[⌈R/gr⌉, ⌈C/gc⌉]` for the stated granularity.
    pub fn from_parts(
        codes: QuantCodes,
        shape: [usize; 2],
        scales: Array<f32>,
        granularity: Granularity,
    ) -> Result<Self, ArrayError> {
        let [rows, cols] = shape;
        check_dims(rows, cols)?;
        let (gr, gc) = granularity.tile(rows, cols)?;
        let expected_codes: [usize; 2] = match codes {
            QuantCodes::Int8(_) => [rows, cols],
            QuantCodes::Int4Packed(_) => [rows, div_ceil(cols, 8)],
        };
        let got_codes = match &codes {
            QuantCodes::Int8(q) => q.shape(),
            QuantCodes::Int4Packed(q) => q.shape(),
        };
        if got_codes != expected_codes {
            return Err(QuantError::Grid {
                what: format!(
                    "codes shape {got_codes:?} does not match {expected_codes:?} for a \
                     [{rows}, {cols}] {:?} matrix",
                    codes.dtype()
                ),
            }
            .into());
        }
        let (tr, tc) = (div_ceil(rows, gr), div_ceil(cols, gc));
        if scales.shape() != [tr, tc] {
            return Err(QuantError::Grid {
                what: format!(
                    "scales shape {:?} does not match the [{tr}, {tc}] grid of \
                     {granularity:?} over a [{rows}, {cols}] matrix",
                    scales.shape()
                ),
            }
            .into());
        }
        Ok(QuantizedMatrix {
            codes,
            scales,
            rows,
            cols,
            gr,
            gc,
            granularity,
        })
    }

    /// The logical `[R, C]` shape.
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    /// The code dtype (int8 / int4).
    pub fn dtype(&self) -> QuantDtype {
        self.codes.dtype()
    }

    /// The granularity this matrix was built with.
    pub fn granularity(&self) -> Granularity {
        self.granularity
    }

    /// The scale tile size `(gr, gc)`.
    pub fn tile(&self) -> (usize, usize) {
        (self.gr, self.gc)
    }

    /// The device-resident codes (for a checkpoint writer or a fused
    /// consumer).
    pub fn codes(&self) -> &QuantCodes {
        &self.codes
    }

    /// The device-resident `[⌈R/gr⌉, ⌈C/gc⌉]` scale grid.
    pub fn scales(&self) -> &Array<f32> {
        &self.scales
    }

    /// Dequantize back to a `[R, C]` f32 array in ONE device dispatch:
    /// per element, load the code at native stride (int8) or its nibble
    /// (int4), load the scale at the grid index, `Dequantize`, store f32.
    /// Bitwise-equal to [`dequantize_host`] on every backend.
    pub fn dequantize(&self) -> Result<Array<f32>, ArrayError> {
        match &self.codes {
            QuantCodes::Int8(q) => self.dispatch_dequant(q, ScalarType::I8),
            QuantCodes::Int4Packed(q) => self.dispatch_dequant(q, ScalarType::I4),
        }
    }

    /// Build, JIT, and dispatch the dequant kernel over `codes` (whose
    /// element type is the physical storage: `i8` native, `u32` packed
    /// nibbles — binding is by slot, the kernel param states the logical
    /// `ScalarType`).
    fn dispatch_dequant<T: quanta_core::GpuType>(
        &self,
        codes: &Array<T>,
        code_ty: ScalarType,
    ) -> Result<Array<f32>, ArrayError> {
        let n = self.rows * self.cols;
        let gpu = codes.gpu();
        let qc = codes.contiguous_or_self()?;
        let scales = self.scales.contiguous_or_self()?;
        let out = gpu.field::<f32>(n)?;
        let bytes = serialize_kernel(&self.dequant_def(code_ty));
        let mut wave = gpu.wave_jit(&bytes)?;
        wave.bind(0, qc.field_ref());
        wave.bind(1, scales.field_ref());
        wave.bind(2, &out);
        gpu.dispatch(&wave, n as u32)?;
        Ok(Array::from_parts(
            gpu.clone(),
            out,
            Layout::row_major(&[self.rows, self.cols])?,
        ))
    }

    /// The dequant kernel, parameterised by `(R, C, gr, gc)` as baked
    /// constants — one kernel shape per codes kind, every granularity a
    /// value change. Thread `i` in `0..R·C`:
    ///
    /// ```text
    ///   r = i / C ; c = i % C
    ///   s = scales[(r / gr) · ⌈C/gc⌉ + (c / gc)]
    ///   q = codes[i]                    (int8: native stride)
    ///   q = codes[r · 8⌈C/8⌉ + c]       (int4: logical nibble index —
    ///                                    word r·⌈C/8⌉ + c/8, nibble c%8)
    ///   out[i] = Dequantize(q, s)       (= s · f32(q))
    /// ```
    fn dequant_def(&self, code_ty: ScalarType) -> KernelDef {
        let scheme = QuantScheme {
            value: self.dtype().value(),
            store: match code_ty {
                ScalarType::I4 => QuantStore::PackedU32,
                _ => QuantStore::Native,
            },
            level: self.granularity.level(),
            mode: QuantMode::Symmetric,
        };
        let mut next = 1u32;
        let mut body = vec![KernelOp::QuarkId { dst: Reg(0) }];
        let c_cols = emit_const(&mut next, &mut body, self.cols as u32);
        let row = emit_bin(&mut next, &mut body, 0, c_cols, BinOp::Div);
        let col = emit_bin(&mut next, &mut body, 0, c_cols, BinOp::Rem);
        // scale index: (row / gr) * ⌈C/gc⌉ + (col / gc)
        let c_gr = emit_const(&mut next, &mut body, self.gr as u32);
        let tile_r = emit_bin(&mut next, &mut body, row, c_gr, BinOp::Div);
        let c_gc = emit_const(&mut next, &mut body, self.gc as u32);
        let tile_c = emit_bin(&mut next, &mut body, col, c_gc, BinOp::Div);
        let c_tc = emit_const(&mut next, &mut body, div_ceil(self.cols, self.gc) as u32);
        let srow = emit_bin(&mut next, &mut body, tile_r, c_tc, BinOp::Mul);
        let sidx = emit_bin(&mut next, &mut body, srow, tile_c, BinOp::Add);
        // code index: linear for int8; the logical nibble index for int4
        // (rows are packed independently, so a row spans 8·⌈C/8⌉ nibbles).
        let qidx = if code_ty == ScalarType::I4 {
            let c_stride = emit_const(&mut next, &mut body, (div_ceil(self.cols, 8) * 8) as u32);
            let base = emit_bin(&mut next, &mut body, row, c_stride, BinOp::Mul);
            emit_bin(&mut next, &mut body, base, col, BinOp::Add)
        } else {
            0
        };
        let mut fresh = || {
            let r = next;
            next += 1;
            r
        };
        let scale = fresh();
        body.push(KernelOp::Load {
            dst: Reg(scale),
            field: 1,
            index: Reg(sidx),
            ty: ScalarType::F32,
        });
        let q = fresh();
        body.push(KernelOp::Load {
            dst: Reg(q),
            field: 0,
            index: Reg(qidx),
            ty: code_ty,
        });
        let dq = fresh();
        body.push(KernelOp::Dequantize {
            dst: Reg(dq),
            src: Reg(q),
            scale: Reg(scale),
            // Symmetric: zero_point is carried for op-shape parity only and
            // read by no backend — reuse the scale register (the op-matrix
            // idiom).
            zero_point: Reg(scale),
            scheme,
        });
        body.push(KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(dq),
            ty: ScalarType::F32,
        });
        KernelDef {
            name: if code_ty == ScalarType::I4 {
                "qa_dequant_i4".into()
            } else {
                "qa_dequant_i8".into()
            },
            params: vec![
                KernelParam::FieldRead {
                    name: "codes".into(),
                    slot: 0,
                    scalar_type: code_ty,
                },
                KernelParam::FieldRead {
                    name: "scales".into(),
                    slot: 1,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 2,
                    scalar_type: ScalarType::F32,
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
        }
    }
}

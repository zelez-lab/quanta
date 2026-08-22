//! npy interop — the single-array numpy format (`np.save` / `np.load`).
//!
//! The format: 6-byte magic `\x93NUMPY`, a version pair, a little-endian
//! header length (u16 in v1.0, u32 in v2.0/v3.0), then an ASCII
//! Python-dict-literal header naming exactly three keys —
//! `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }` — padded
//! with spaces and terminated by `\n`. Raw element bytes follow, C-order
//! unless `fortran_order: True`. Modern numpy pads the header so the data
//! section starts 64-byte aligned; readers must trust the length field,
//! never the alignment. v3.0 differs from v2.0 only in header encoding
//! (utf-8 instead of latin-1), which is observable only with structured
//! field names — rejected here — so one strict grammar covers all three
//! versions.
//!
//! The surface (see `NPY_INTEROP.md` at the crate root for the full
//! declared scope with its deferrals):
//!
//! - [`save`] — any `Array<T>` view (strided / transposed / broadcast /
//!   narrowed included) to npy bytes: its logical row-major content,
//!   always `<`-endian, always `fortran_order: False`, v1.0 upgrading to
//!   v2.0 only on u16 header overflow (numpy's own rule).
//! - [`load`] — the typed load: the file's descr must match `T` exactly,
//!   with one documented widening — `load::<f32>` also accepts `<f2`
//!   (f16 upconverts exactly). Big-endian (`>`) files byteswap at load;
//!   Fortran-order files are host-permuted so the caller always receives
//!   a row-major contiguous array of the file's logical shape.
//! - [`load_dyn`] — the dtype-preserving load for "inspect what Python
//!   wrote": returns the [`NpyArray`] enum matching the file's descr
//!   (`<f2` upconverts to `F32`; `|b1` validates every byte as 0/1 and
//!   lands in `U8`).
//! - [`header`] — introspection without touching the data section.
//!
//! Everything dtype-independent (the strict header grammar, the padded
//! preamble writer, the descr table) lives in the private shared codec
//! (`crate::npy_codec`); the npz container shares it.
//!
//! The house pattern is bytes-level I/O (no file-path wrappers):
//! `std::fs::write(path, npy::save(&a)?)` is the documented one-liner.

use core::fmt;

use quanta_core::Gpu;

use crate::array::Array;
use crate::error::ArrayError;
use crate::npy_codec::{self, NpyDtype};
use crate::scalar::ArrayScalar;

/// Everything the npy preamble says about a file, without touching the
/// data section: the raw `descr` string, the memory order, the shape,
/// the format version, and where the element bytes begin.
///
/// This is the "what is this file?" probe: it reports the header as
/// written (including descrs the typed loaders reject, e.g. `<c8`), so a
/// caller can decide before committing to a load. It is also the natural
/// probe point for a future mmap path — `data_offset` locates the
/// element bytes inside a mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    /// The `descr` value exactly as written (e.g. `"<f4"`).
    pub descr: String,
    /// `true` when the data section is Fortran-order (column-major).
    pub fortran_order: bool,
    /// The shape tuple; empty for a rank-0 scalar file.
    pub shape: Vec<usize>,
    /// The format version pair, `(1, 0)` / `(2, 0)` / `(3, 0)`.
    pub version: (u8, u8),
    /// Byte offset of the first element byte (magic + version + length
    /// field + padded header text).
    pub data_offset: usize,
}

/// Parse the npy preamble of `bytes` — magic, version, and the header
/// dict — without touching the data section. `bytes` may be a prefix of
/// the file as long as it covers the full header.
pub fn header(bytes: &[u8]) -> Result<NpyHeader, ArrayError> {
    crate::npy_codec::parse_header(bytes).map_err(ArrayError::from)
}

/// An npy / npz interop fault. Wrapped as [`ArrayError::Npy`].
///
/// Every message is self-contained: it names the offending entry or byte
/// offset, and states the workaround where one exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NpyError {
    /// Not an npy file, or truncated before the version bytes. Carries
    /// the first bytes of the input for display.
    Magic { first: Vec<u8> },
    /// An unrecognized format version pair.
    Version { major: u8, minor: u8 },
    /// A header fault: dict-grammar violation, a header length
    /// overrunning the buffer, unknown / missing / duplicate keys. `at`
    /// is a file-absolute byte offset.
    Header { at: usize, what: String },
    /// A descr outside the supported table (§4 of the scope doc).
    Dtype { descr: String },
    /// A malformed `=` byte-order mark (`>` files byteswap-load instead).
    ByteOrder { descr: String },
    /// A `|b1` data byte outside {0, 1}.
    BoolValue { at: usize },
    /// A typed `load::<T>` against a different (but supported) descr.
    DtypeMismatch { file: String, requested: String },
    /// A shape with a zero extent (excluded by the shape model).
    EmptyShape { shape: Vec<usize> },
    /// Data section length disagreeing with element-count × width.
    DataLength { expected: usize, got: usize },
    /// A container fault: bad EOCD / central directory, CRC mismatch,
    /// local/CD disagreement, duplicate or non-`.npy` names, ZIP64
    /// markers, corrupt deflate streams — or an entry whose npy payload
    /// fails to decode (the inner fault's message follows the entry
    /// name). `entry` is present wherever a name exists.
    Zip { entry: Option<String>, what: String },
}

/// The descrs the typed loaders accept, for the `Dtype` message.
const SUPPORTED_DESCRS: &str =
    "<f4 <f8 <i4 <u4 <i8 <u8 <i2 <u2 |i1 |u1 <f2 |b1 (and their > big-endian forms)";

/// Why a rejected descr is rejected — the specific reason the message
/// contract promises (pickle exclusion, complex model gap, …).
fn dtype_reason(descr: &str) -> &'static str {
    if descr.starts_with('[') || descr.starts_with('(') {
        return "structured dtypes have no Array representation and are excluded";
    }
    let kind = descr
        .strip_prefix(['<', '>', '|', '='])
        .unwrap_or(descr)
        .chars()
        .next();
    match kind {
        Some('O') => {
            "object arrays embed pickle (arbitrary code execution) and are permanently \
             excluded — re-export the data as a numeric dtype"
        }
        Some('c') => "no complex element type exists in the stack yet",
        Some('S') | Some('U') | Some('a') => "string dtypes have no Array representation",
        Some('V') => "void/structured dtypes have no Array representation",
        _ => "not an element dtype quanta-array represents",
    }
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpyError::Magic { first } => {
                if first.is_empty() {
                    return write!(f, "not an npy file: empty input");
                }
                write!(
                    f,
                    "not an npy file: expected magic \\x93NUMPY, input starts with"
                )?;
                for b in first.iter().take(8) {
                    write!(f, " {b:02x}")?;
                }
                Ok(())
            }
            NpyError::Version { major, minor } => write!(
                f,
                "npy version {major}.{minor} is not supported (accepted: 1.0, 2.0, 3.0)"
            ),
            NpyError::Header { at, what } => write!(f, "npy header: {what} at byte {at}"),
            NpyError::Dtype { descr } => write!(
                f,
                "npy descr {descr:?} is not supported (supported: {SUPPORTED_DESCRS}): {}",
                dtype_reason(descr)
            ),
            NpyError::ByteOrder { descr } => write!(
                f,
                "npy descr {descr:?} uses byte order '=': numpy never writes '=' to files, \
                 so the descr is malformed"
            ),
            NpyError::BoolValue { at } => write!(
                f,
                "npy |b1 data byte at offset {at} is neither 0 nor 1 — numpy never writes \
                 other values; the file is corrupt"
            ),
            NpyError::DtypeMismatch { file, requested } => write!(
                f,
                "npy dtype mismatch: the file holds {file}, the load requested {requested}; \
                 use npy::load_dyn to take the file's dtype"
            ),
            NpyError::EmptyShape { shape } => write!(
                f,
                "npy shape {shape:?} has a zero extent: quanta-array shapes describe data \
                 that exists, so zero-size axes are rejected by design"
            ),
            NpyError::DataLength { expected, got } => write!(
                f,
                "npy data section holds {got} bytes, but the header's shape and descr \
                 require {expected}"
            ),
            NpyError::Zip { entry, what } => match entry {
                Some(name) => write!(f, "npz entry {name:?}: {what}"),
                None => write!(f, "npz archive: {what}"),
            },
        }
    }
}

impl std::error::Error for NpyError {}

// ── The element vocabulary ──────────────────────────────────────────────

mod sealed {
    pub trait Sealed {}
}

/// The `Array` element types npy can carry — a sealed marker over the
/// ten [`ArrayScalar`] types (`f32`/`f64`, `i32`/`u32`, `i64`/`u64`,
/// `u8`/`i8`/`u16`/`i16`). The trait's items are implementation detail;
/// the bound is the API.
pub trait NpyScalar: ArrayScalar + sealed::Sealed {
    #[doc(hidden)]
    const DTYPE: NpyDtype;
    /// Whether `<f2` widens into this type — true for `f32` only, the
    /// one documented widening.
    #[doc(hidden)]
    const WIDENS_F16: bool = false;
    #[doc(hidden)]
    fn write_le(v: Self, out: &mut Vec<u8>);
    #[doc(hidden)]
    fn read_le(chunk: &[u8]) -> Self;
    #[doc(hidden)]
    fn from_f16(bits: u16) -> Self {
        let _ = bits;
        unreachable!("f16 widens only into f32 (the WIDENS_F16 gate)")
    }
}

macro_rules! npy_scalar {
    ($($t:ty => $dt:ident),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl NpyScalar for $t {
            const DTYPE: NpyDtype = NpyDtype::$dt;
            fn write_le(v: Self, out: &mut Vec<u8>) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            fn read_le(chunk: &[u8]) -> Self {
                Self::from_le_bytes(chunk.try_into().expect("chunk is element-width"))
            }
        }
    )*};
}

npy_scalar!(
    f64 => F64, i32 => I32, u32 => U32, i64 => I64, u64 => U64,
    u8 => U8, i8 => I8, u16 => U16, i16 => I16,
);

impl sealed::Sealed for f32 {}
impl NpyScalar for f32 {
    const DTYPE: NpyDtype = NpyDtype::F32;
    const WIDENS_F16: bool = true;
    fn write_le(v: Self, out: &mut Vec<u8>) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn read_le(chunk: &[u8]) -> Self {
        Self::from_le_bytes(chunk.try_into().expect("chunk is element-width"))
    }
    fn from_f16(bits: u16) -> Self {
        f16_to_f32(bits)
    }
}

/// Decode an IEEE half (`<f2`) to `f32` — an exact embedding: signs,
/// subnormals, inf and NaN all preserved. (Mirrors the safetensors
/// decoder in quanta-nn, the crate precedent for f16 upconversion.)
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: value = m·2⁻²⁴ = 1.f × 2^(p−24) with p the
            // mantissa's MSB index — always a NORMAL f32 (exp 103…112).
            let p = 31 - m.leading_zeros();
            let exp32 = 127 + p - 24;
            let man32 = (m << (10 - p)) & 0x3FF;
            sign | (exp32 << 23) | (man32 << 13)
        }
        (0x1F, 0) => sign | 0x7F80_0000,
        (0x1F, m) => sign | 0x7F80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

// ── The dynamic-dtype vocabulary ────────────────────────────────────────

/// A dtype-preserving loaded array — the vocabulary for "inspect what
/// Python wrote" ([`load_dyn`]) and for mixed-dtype npz archives (labels
/// `u64` next to weights `f32`), in both directions.
///
/// Ten variants, one per element type. `<f2` files land in `F32` (the
/// exact upconversion; no f16 array exists to preserve into) and `|b1`
/// files land in `U8` after 0/1 validation. Building one is cheap
/// (`Array` is Arc-backed): `NpyArray::from(array)` shares the buffer.
pub enum NpyArray {
    F32(Array<f32>),
    F64(Array<f64>),
    I32(Array<i32>),
    U32(Array<u32>),
    I64(Array<i64>),
    U64(Array<u64>),
    U8(Array<u8>),
    I8(Array<i8>),
    U16(Array<u16>),
    I16(Array<i16>),
}

/// Apply one expression to whichever variant is held.
macro_rules! with_variant {
    ($self:expr, $a:ident => $body:expr) => {
        match $self {
            NpyArray::F32($a) => $body,
            NpyArray::F64($a) => $body,
            NpyArray::I32($a) => $body,
            NpyArray::U32($a) => $body,
            NpyArray::I64($a) => $body,
            NpyArray::U64($a) => $body,
            NpyArray::U8($a) => $body,
            NpyArray::I8($a) => $body,
            NpyArray::U16($a) => $body,
            NpyArray::I16($a) => $body,
        }
    };
}

/// The tag of a typed array — lets `with_variant!` bodies name the
/// element type's dtype generically.
fn dtype_of<T: NpyScalar>(_: &Array<T>) -> NpyDtype {
    T::DTYPE
}

impl NpyArray {
    /// The canonical descr of the held dtype (`"<f4"`, `"|u1"`, …) — the
    /// string [`save`] writes for it.
    pub fn dtype(&self) -> &'static str {
        with_variant!(self, a => dtype_of(a).canonical_descr())
    }

    /// The held array's shape.
    pub fn shape(&self) -> &[usize] {
        with_variant!(self, a => a.shape())
    }
}

impl fmt::Debug for NpyArray {
    /// Dtype + shape — `Array` itself has no `Debug` (its data is on
    /// the device), so the metadata is the honest picture.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NpyArray({} {:?})", self.dtype(), self.shape())
    }
}

/// Encode whichever variant is held (the npz save path).
pub(crate) fn save_dyn(array: &NpyArray) -> Result<Vec<u8>, ArrayError> {
    with_variant!(array, a => save(a))
}

macro_rules! npy_array_conversions {
    ($($t:ty => $v:ident),* $(,)?) => {$(
        impl From<Array<$t>> for NpyArray {
            fn from(a: Array<$t>) -> NpyArray {
                NpyArray::$v(a)
            }
        }
        impl TryFrom<NpyArray> for Array<$t> {
            type Error = ArrayError;
            /// The typed unwrap; any other held dtype is the loud
            /// [`NpyError::DtypeMismatch`], naming both sides.
            fn try_from(a: NpyArray) -> Result<Array<$t>, ArrayError> {
                match a {
                    NpyArray::$v(a) => Ok(a),
                    other => Err(NpyError::DtypeMismatch {
                        file: other.dtype().to_string(),
                        requested: <$t as NpyScalar>::DTYPE.canonical_descr().to_string(),
                    }
                    .into()),
                }
            }
        }
    )*};
}

npy_array_conversions!(
    f32 => F32, f64 => F64, i32 => I32, u32 => U32, i64 => I64,
    u64 => U64, u8 => U8, i8 => I8, u16 => U16, i16 => I16,
);

// ── Save ────────────────────────────────────────────────────────────────

/// Serialize an array to npy bytes: the padded v1.0 preamble (v2.0 only
/// on u16 header overflow — numpy's own rule) followed by `<`-endian
/// element bytes in C order, `fortran_order: False` always.
///
/// Accepts ANY view — strided / transposed / broadcast / narrowed views
/// serialize their **logical row-major content** (the `to_vec` gather),
/// so a view and its `contiguous()` copy produce identical bytes.
///
/// Bytes-level by design: `std::fs::write(path, npy::save(&a)?)`.
pub fn save<T: NpyScalar>(a: &Array<T>) -> Result<Vec<u8>, ArrayError> {
    let data = a.to_vec()?;
    let mut out = npy_codec::write_header(T::DTYPE.canonical_descr(), false, a.shape());
    out.reserve(data.len() * core::mem::size_of::<T>());
    for v in data {
        T::write_le(v, &mut out);
    }
    Ok(out)
}

// ── Load ────────────────────────────────────────────────────────────────

/// Typed load — the common path, when you know what you saved. The
/// file's descr must map to `T` exactly, with one documented widening:
/// `load::<f32>` also accepts `<f2` (f16 upconverts exactly). Any other
/// supported descr is a loud [`NpyError::DtypeMismatch`] naming both
/// sides; unsupported descrs are [`NpyError::Dtype`] with the reason.
///
/// Big-endian (`>`) descrs byteswap at element width during the load.
/// `fortran_order: True` files are host-permuted into logical row-major
/// — the caller always receives a row-major contiguous `Array` of the
/// file's logical shape.
pub fn load<T: NpyScalar>(gpu: &Gpu, bytes: &[u8]) -> Result<Array<T>, ArrayError> {
    let h = npy_codec::parse_header(bytes)?;
    let descr = npy_codec::parse_descr(&h.descr)?;
    let data = data_section(bytes, &h, descr.dtype.width())?;
    let elems: Vec<T> = if descr.dtype == T::DTYPE {
        decode(data, descr.big_endian)
    } else if descr.dtype == NpyDtype::F16 && T::WIDENS_F16 {
        decode_f16(data, descr.big_endian)
    } else {
        return Err(NpyError::DtypeMismatch {
            file: h.descr.clone(),
            requested: T::DTYPE.canonical_descr().to_string(),
        }
        .into());
    };
    finish(gpu, elems, &h)
}

/// Dtype-preserving load: returns the [`NpyArray`] variant matching the
/// file's descr. `<f2` upconverts to `F32` (no f16 array exists to
/// preserve into); `|b1` validates every data byte as 0/1 — a loud
/// [`NpyError::BoolValue`] otherwise — and lands in `U8`. Layout
/// semantics (byteswap, Fortran permute, shape) are exactly [`load`]'s.
pub fn load_dyn(gpu: &Gpu, bytes: &[u8]) -> Result<NpyArray, ArrayError> {
    let h = npy_codec::parse_header(bytes)?;
    let descr = npy_codec::parse_descr(&h.descr)?;
    let data = data_section(bytes, &h, descr.dtype.width())?;
    let be = descr.big_endian;
    Ok(match descr.dtype {
        NpyDtype::F32 => NpyArray::F32(finish(gpu, decode(data, be), &h)?),
        NpyDtype::F64 => NpyArray::F64(finish(gpu, decode(data, be), &h)?),
        NpyDtype::I32 => NpyArray::I32(finish(gpu, decode(data, be), &h)?),
        NpyDtype::U32 => NpyArray::U32(finish(gpu, decode(data, be), &h)?),
        NpyDtype::I64 => NpyArray::I64(finish(gpu, decode(data, be), &h)?),
        NpyDtype::U64 => NpyArray::U64(finish(gpu, decode(data, be), &h)?),
        NpyDtype::U8 => NpyArray::U8(finish(gpu, decode(data, be), &h)?),
        NpyDtype::I8 => NpyArray::I8(finish(gpu, decode(data, be), &h)?),
        NpyDtype::U16 => NpyArray::U16(finish(gpu, decode(data, be), &h)?),
        NpyDtype::I16 => NpyArray::I16(finish(gpu, decode(data, be), &h)?),
        NpyDtype::F16 => NpyArray::F32(finish(gpu, decode_f16(data, be), &h)?),
        NpyDtype::Bool => {
            if let Some(i) = data.iter().position(|&b| b > 1) {
                return Err(NpyError::BoolValue {
                    at: h.data_offset + i,
                }
                .into());
            }
            NpyArray::U8(finish(gpu, data.to_vec(), &h)?)
        }
    })
}

// ── Shared load plumbing ────────────────────────────────────────────────

/// Validate the shape and locate the data section: zero extents are the
/// specific [`NpyError::EmptyShape`] (the shape-model exclusion), and
/// the section must hold exactly element-count × width bytes
/// ([`NpyError::DataLength`]) — checked before any allocation is sized
/// from the header.
fn data_section<'a>(bytes: &'a [u8], h: &NpyHeader, width: usize) -> Result<&'a [u8], NpyError> {
    if h.shape.contains(&0) {
        return Err(NpyError::EmptyShape {
            shape: h.shape.clone(),
        });
    }
    // The element count can only overflow usize in a hostile header — no
    // real data section can match it, so saturation lands in DataLength.
    let count = h.shape.iter().fold(1usize, |n, &d| n.saturating_mul(d));
    let expected = count.saturating_mul(width);
    let got = bytes.len() - h.data_offset;
    if expected != got {
        return Err(NpyError::DataLength { expected, got });
    }
    Ok(&bytes[h.data_offset..])
}

/// Decode a validated data section at element width, byteswapping each
/// chunk when the descr was big-endian.
fn decode<T: NpyScalar>(data: &[u8], big_endian: bool) -> Vec<T> {
    let w = core::mem::size_of::<T>();
    if big_endian {
        let mut buf = [0u8; 8];
        data.chunks_exact(w)
            .map(|c| {
                for (i, &b) in c.iter().enumerate() {
                    buf[w - 1 - i] = b;
                }
                T::read_le(&buf[..w])
            })
            .collect()
    } else {
        data.chunks_exact(w).map(T::read_le).collect()
    }
}

/// Decode a `<f2` / `>f2` data section through the exact f16 → f32
/// embedding. `T` is `f32` in practice (the `WIDENS_F16` gate).
fn decode_f16<T: NpyScalar>(data: &[u8], big_endian: bool) -> Vec<T> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|c| {
            let bits = if big_endian {
                u16::from_be_bytes(*c)
            } else {
                u16::from_le_bytes(*c)
            };
            T::from_f16(bits)
        })
        .collect()
}

/// Upload decoded elements as a row-major array of the file's logical
/// shape, host-permuting Fortran-order files first (§6 of the scope:
/// load-with-transpose, never reject).
fn finish<T: NpyScalar>(gpu: &Gpu, elems: Vec<T>, h: &NpyHeader) -> Result<Array<T>, ArrayError> {
    let elems = if h.fortran_order && h.shape.len() > 1 {
        c_order_from_fortran(&elems, &h.shape)
    } else {
        // Rank 0 and rank 1 are order-invariant; C order is already
        // logical row-major.
        elems
    };
    Array::from_slice(gpu, &elems, &h.shape)
}

/// Host-permute Fortran-order (first-axis-fastest) elements into logical
/// row-major: walk logical coordinates in row-major order and gather
/// each from its F-order linear index. One pass, at load time only.
fn c_order_from_fortran<T: Copy>(data: &[T], shape: &[usize]) -> Vec<T> {
    let rank = shape.len();
    let mut fstrides = vec![1usize; rank];
    for i in 1..rank {
        fstrides[i] = fstrides[i - 1] * shape[i - 1];
    }
    let mut out = Vec::with_capacity(data.len());
    let mut coord = vec![0usize; rank];
    for _ in 0..data.len() {
        let idx: usize = coord.iter().zip(&fstrides).map(|(&c, &s)| c * s).sum();
        out.push(data[idx]);
        // Increment the mixed-radix coordinate (last axis fastest).
        for axis in (0..rank).rev() {
            coord[axis] += 1;
            if coord[axis] < shape[axis] {
                break;
            }
            coord[axis] = 0;
        }
    }
    out
}

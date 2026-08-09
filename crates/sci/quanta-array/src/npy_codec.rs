//! The shared npy header codec — private substrate under [`crate::npy`]
//! and [`crate::npz`].
//!
//! Three jobs, all dtype-independent:
//!
//! - [`parse_header`] — magic / version / length-field walk plus a strict
//!   recursive-descent parser for the Python-dict-literal header (exactly
//!   the keys `descr` / `fortran_order` / `shape`). Hostile-input-grade:
//!   every length field is bounds-checked before use, error offsets are
//!   file-absolute, and no allocation is driven by an unvalidated size.
//! - [`write_header`] — the padded preamble with the data section 64-byte
//!   aligned (modern numpy's rule, and what keeps our own files eligible
//!   for a future mmap path), auto-upgrading v1.0 → v2.0 exactly when the
//!   padded header text overflows u16 (numpy's own rule).
//! - [`parse_descr`] — the pure string → [`Descr`] table over every
//!   supported descr. No `Array` involvement; the typed load/save layer
//!   matches on the result to pick its element decode.

use crate::npy::{NpyError, NpyHeader};

/// The six magic bytes every npy file starts with.
pub const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Alignment of the data section in files we write.
pub const DATA_ALIGN: usize = 64;

// ── Descr table ─────────────────────────────────────────────────────────

/// The element dtype a supported descr maps to — the tagged vocabulary
/// the typed load/save layer consumes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyDtype {
    F32,
    F64,
    I32,
    U32,
    I64,
    U64,
    U8,
    I8,
    U16,
    I16,
    /// `f2` — loaded with exact upconversion to `f32`; never written.
    F16,
    /// `b1` — loaded as `Array<u8>` with 0/1 validation; never written.
    Bool,
}

impl NpyDtype {
    /// Element width in bytes.
    pub fn width(self) -> usize {
        match self {
            NpyDtype::U8 | NpyDtype::I8 | NpyDtype::Bool => 1,
            NpyDtype::U16 | NpyDtype::I16 | NpyDtype::F16 => 2,
            NpyDtype::F32 | NpyDtype::I32 | NpyDtype::U32 => 4,
            NpyDtype::F64 | NpyDtype::I64 | NpyDtype::U64 => 8,
        }
    }

    /// The descr the save path writes: always little-endian (`<`), with
    /// numpy's `|` convention for one-byte types.
    pub fn canonical_descr(self) -> &'static str {
        match self {
            NpyDtype::F32 => "<f4",
            NpyDtype::F64 => "<f8",
            NpyDtype::I32 => "<i4",
            NpyDtype::U32 => "<u4",
            NpyDtype::I64 => "<i8",
            NpyDtype::U64 => "<u8",
            NpyDtype::U8 => "|u1",
            NpyDtype::I8 => "|i1",
            NpyDtype::U16 => "<u2",
            NpyDtype::I16 => "<i2",
            NpyDtype::F16 => "<f2",
            NpyDtype::Bool => "|b1",
        }
    }
}

/// A parsed, supported descr: the dtype, and whether the file's element
/// bytes are big-endian (byteswap-on-load; never true at width 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Descr {
    pub dtype: NpyDtype,
    pub big_endian: bool,
}

/// Map a descr string to its [`Descr`], or the loud error the taxonomy
/// promises: `=` byte order → [`NpyError::ByteOrder`], everything else
/// outside the table → [`NpyError::Dtype`] with the specific reason.
pub fn parse_descr(descr: &str) -> Result<Descr, NpyError> {
    let unsupported = || NpyError::Dtype {
        descr: descr.to_string(),
    };
    let (order, rest) = match descr.as_bytes().first() {
        Some(b'<') => (b'<', &descr[1..]),
        Some(b'>') => (b'>', &descr[1..]),
        Some(b'|') => (b'|', &descr[1..]),
        Some(b'=') => {
            return Err(NpyError::ByteOrder {
                descr: descr.to_string(),
            });
        }
        // numpy always writes an order character; a bare descr (or an
        // empty one) is not something it produces.
        _ => return Err(unsupported()),
    };
    let dtype = match rest {
        "f4" => NpyDtype::F32,
        "f8" => NpyDtype::F64,
        "i4" => NpyDtype::I32,
        "u4" => NpyDtype::U32,
        "i8" => NpyDtype::I64,
        "u8" => NpyDtype::U64,
        "u1" => NpyDtype::U8,
        "i1" => NpyDtype::I8,
        "u2" => NpyDtype::U16,
        "i2" => NpyDtype::I16,
        "f2" => NpyDtype::F16,
        "b1" => NpyDtype::Bool,
        _ => return Err(unsupported()),
    };
    let big_endian = match (order, dtype.width()) {
        // Byte order is meaningless at width 1; numpy writes `|`, but
        // `<u1` / `>u1` from other writers are unambiguous — accept.
        (_, 1) => false,
        (b'<', _) => false,
        (b'>', _) => true,
        // `|f4` and friends are malformed: numpy never writes `|` on a
        // multi-byte type.
        _ => return Err(unsupported()),
    };
    Ok(Descr { dtype, big_endian })
}

// ── Header parsing ──────────────────────────────────────────────────────

/// Parse the full preamble: magic, version, length field, header dict.
/// `bytes` may be any prefix of a file that covers the header.
pub fn parse_header(bytes: &[u8]) -> Result<NpyHeader, NpyError> {
    if bytes.len() < 8 || &bytes[..6] != MAGIC {
        return Err(NpyError::Magic {
            first: bytes[..bytes.len().min(8)].to_vec(),
        });
    }
    let version = (bytes[6], bytes[7]);
    let (len_field, prefix) = match version {
        (1, 0) => (2usize, 10usize),
        (2, 0) | (3, 0) => (4, 12),
        (major, minor) => return Err(NpyError::Version { major, minor }),
    };
    let Some(len_bytes) = bytes.get(8..8 + len_field) else {
        return Err(NpyError::Header {
            at: 8,
            what: format!("input ends inside the {len_field}-byte header-length field"),
        });
    };
    let header_len = if len_field == 2 {
        u16::from_le_bytes([len_bytes[0], len_bytes[1]]) as usize
    } else {
        u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize
    };
    // Bounds-check the length field before slicing — the header text is
    // borrowed, never allocated, so a hostile length costs nothing.
    let end = match prefix.checked_add(header_len) {
        Some(e) if e <= bytes.len() => e,
        _ => {
            return Err(NpyError::Header {
                at: 8,
                what: format!(
                    "header length {header_len} overruns the {}-byte input",
                    bytes.len()
                ),
            });
        }
    };
    let (descr, fortran_order, shape) = DictParser {
        b: &bytes[prefix..end],
        i: 0,
        base: prefix,
    }
    .parse()?;
    Ok(NpyHeader {
        descr,
        fortran_order,
        shape,
        version,
        data_offset: end,
    })
}

/// Recursive-descent parser for the strict header-dict grammar. Offsets
/// in errors are file-absolute (`base` + local position).
struct DictParser<'a> {
    b: &'a [u8],
    i: usize,
    base: usize,
}

impl DictParser<'_> {
    fn err(&self, at: usize, what: impl Into<String>) -> NpyError {
        NpyError::Header {
            at: self.base + at,
            what: what.into(),
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.b.get(self.i), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.i += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), NpyError> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(self.i, format!("expected '{}'", c as char)))
        }
    }

    /// A quoted string (single or double quotes, numpy writes single).
    /// ASCII-printable only, no escapes: descr strings and the three key
    /// names never need either, so anything else is malformed (v3's
    /// utf-8 delta only shows up in structured names, which we reject).
    fn string(&mut self) -> Result<String, NpyError> {
        let quote = match self.peek() {
            Some(q @ (b'\'' | b'"')) => q,
            _ => return Err(self.err(self.i, "expected a quoted string")),
        };
        self.i += 1;
        let start = self.i;
        loop {
            match self.b.get(self.i) {
                None => return Err(self.err(start - 1, "unterminated string")),
                Some(&c) if c == quote => {
                    let s = core::str::from_utf8(&self.b[start..self.i])
                        .expect("ascii checked per byte")
                        .to_string();
                    self.i += 1;
                    return Ok(s);
                }
                Some(b'\\') => {
                    return Err(self.err(self.i, "escape sequences are not supported"));
                }
                Some(&c) if (0x20..0x7F).contains(&c) => self.i += 1,
                Some(_) => {
                    return Err(self.err(self.i, "non-printable or non-ascii byte in string"));
                }
            }
        }
    }

    fn literal(&mut self, word: &str) -> bool {
        self.skip_ws();
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            true
        } else {
            false
        }
    }

    /// One shape extent: a run of decimal digits, overflow-checked.
    fn extent(&mut self) -> Result<usize, NpyError> {
        self.skip_ws();
        let start = self.i;
        while matches!(self.b.get(self.i), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if start == self.i {
            return Err(self.err(start, "expected a shape extent (decimal digits)"));
        }
        core::str::from_utf8(&self.b[start..self.i])
            .expect("digits are ascii")
            .parse::<usize>()
            .map_err(|_| self.err(start, "shape extent overflows"))
    }

    /// The shape tuple: `()`, `(3,)`, `(3, 4)`. `(3)` is not a Python
    /// tuple and numpy never writes it — rejected.
    fn shape(&mut self) -> Result<Vec<usize>, NpyError> {
        self.expect(b'(')?;
        let mut dims = Vec::new();
        if self.peek() == Some(b')') {
            self.i += 1;
            return Ok(dims);
        }
        loop {
            dims.push(self.extent()?);
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                    if self.peek() == Some(b')') {
                        self.i += 1;
                        return Ok(dims);
                    }
                }
                Some(b')') => {
                    if dims.len() == 1 {
                        return Err(self.err(
                            self.i,
                            "a 1-element shape must be written '(n,)' — '(n)' is not a tuple",
                        ));
                    }
                    self.i += 1;
                    return Ok(dims);
                }
                _ => return Err(self.err(self.i, "expected ',' or ')' in the shape tuple")),
            }
        }
    }

    fn parse(mut self) -> Result<(String, bool, Vec<usize>), NpyError> {
        self.expect(b'{')?;
        let mut descr: Option<String> = None;
        let mut fortran: Option<bool> = None;
        let mut shape: Option<Vec<usize>> = None;
        loop {
            if self.peek() == Some(b'}') {
                self.i += 1;
                break;
            }
            self.skip_ws();
            let key_at = self.i;
            let key = self.string()?;
            self.expect(b':')?;
            match key.as_str() {
                "descr" => {
                    if descr.is_some() {
                        return Err(self.err(key_at, "duplicate key 'descr'"));
                    }
                    // A list/tuple here is a structured descr — a §4
                    // exclusion, reported through the Dtype taxonomy row
                    // with a bounded snippet of the offending value.
                    if matches!(self.peek(), Some(b'[' | b'(')) {
                        let snippet = &self.b[self.i..self.b.len().min(self.i + 48)];
                        return Err(NpyError::Dtype {
                            descr: String::from_utf8_lossy(snippet).into_owned(),
                        });
                    }
                    descr = Some(self.string()?);
                }
                "fortran_order" => {
                    if fortran.is_some() {
                        return Err(self.err(key_at, "duplicate key 'fortran_order'"));
                    }
                    fortran = Some(if self.literal("True") {
                        true
                    } else if self.literal("False") {
                        false
                    } else {
                        return Err(self.err(self.i, "expected True or False"));
                    });
                }
                "shape" => {
                    if shape.is_some() {
                        return Err(self.err(key_at, "duplicate key 'shape'"));
                    }
                    shape = Some(self.shape()?);
                }
                other => {
                    return Err(self.err(
                        key_at,
                        format!(
                            "unknown key {other:?} (expected 'descr', 'fortran_order', 'shape')"
                        ),
                    ));
                }
            }
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(self.err(self.i, "expected ',' or '}'")),
            }
        }
        // Only space padding and the final newline may follow the dict.
        while let Some(&c) = self.b.get(self.i) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            } else {
                return Err(self.err(
                    self.i,
                    "unexpected byte after the closing '}' (header padding must be spaces)",
                ));
            }
        }
        let end = self.i;
        let descr = descr.ok_or_else(|| self.err(end, "missing key 'descr'"))?;
        let fortran = fortran.ok_or_else(|| self.err(end, "missing key 'fortran_order'"))?;
        let shape = shape.ok_or_else(|| self.err(end, "missing key 'shape'"))?;
        Ok((descr, fortran, shape))
    }
}

// ── Header writing ──────────────────────────────────────────────────────

/// Render the full preamble for a save: magic, version, length field,
/// and the dict padded with spaces (plus the final `\n`) so the data
/// section starts [`DATA_ALIGN`]-byte aligned. Writes v1.0, upgrading to
/// v2.0 exactly when the padded header text overflows u16 — numpy's own
/// rule.
pub fn write_header(descr: &str, fortran_order: bool, shape: &[usize]) -> Vec<u8> {
    let mut dict = String::with_capacity(72);
    dict.push_str("{'descr': '");
    dict.push_str(descr);
    dict.push_str("', 'fortran_order': ");
    dict.push_str(if fortran_order { "True" } else { "False" });
    dict.push_str(", 'shape': (");
    for (i, d) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        dict.push_str(&d.to_string());
    }
    if shape.len() == 1 {
        dict.push(',');
    }
    dict.push_str("), }");

    // Padded total for a given preamble prefix (magic 6 + version 2 +
    // length field), including the terminating '\n'.
    let padded = |prefix: usize| (prefix + dict.len() + 1).div_ceil(DATA_ALIGN) * DATA_ALIGN;
    let total_v1 = padded(10);
    let (major, prefix, total) = if total_v1 - 10 <= u16::MAX as usize {
        (1u8, 10, total_v1)
    } else {
        let t = padded(12);
        assert!(
            t - 12 <= u32::MAX as usize,
            "npy header text exceeds u32 — the shape is unrepresentable"
        );
        (2u8, 12, t)
    };
    let text_len = total - prefix;

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(MAGIC);
    out.push(major);
    out.push(0);
    if major == 1 {
        out.extend_from_slice(&(text_len as u16).to_le_bytes());
    } else {
        out.extend_from_slice(&(text_len as u32).to_le_bytes());
    }
    out.extend_from_slice(dict.as_bytes());
    out.resize(total - 1, b' ');
    out.push(b'\n');
    debug_assert_eq!(out.len(), total);
    debug_assert_eq!(out.len() % DATA_ALIGN, 0);
    out
}

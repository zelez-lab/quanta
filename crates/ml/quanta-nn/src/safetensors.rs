//! safetensors interop — the HF weight format, dependency-free.
//!
//! The format: 8 bytes little-endian header length `N`, then `N` bytes
//! of JSON mapping tensor names to `{"dtype", "shape", "data_offsets"}`
//! (offsets relative to the data section that follows), plus an
//! optional `"__metadata__"` string map. [`save_named`] writes `F32`
//! tensors with ascending contiguous offsets; [`load_named`] reads
//! `F32` exactly and upconverts `F16`/`BF16` (what real checkpoints
//! ship) — every other dtype is a loud error naming the tensor.
//!
//! [`save`]/[`load`] lift this to [`ParamTree`]s through the same
//! hierarchical names as [`crate::state`]: load matches by NAME, not
//! order, and a missing, extra, or wrong-shape entry is a loud error
//! naming the path. The JSON reader is a minimal recursive-descent
//! parser for exactly the header grammar (the no-wrapper-crates
//! policy) — strict about the entry keys, correct about string
//! escapes.

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar};
use quanta_core::Gpu;
use std::collections::HashMap;

use crate::layer::ParamTree;

fn bad(msg: String) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        &msg,
    )))
}

/// Everything a safetensors byte string holds: the tensors (as `f32`
/// arrays, in header order) and the optional `__metadata__` map.
pub struct LoadedSafetensors {
    pub tensors: Vec<(String, Array<f32>)>,
    pub metadata: HashMap<String, String>,
}

// ── Save ────────────────────────────────────────────────────────────────

fn json_escape(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Serialize named tensors (any [`DiffScalar`] — values travel as
/// `F32`) plus an optional metadata map. Entries are written in the
/// given order with ascending contiguous data offsets.
pub fn save_named<T: DiffScalar + ToF64>(
    entries: &[(String, Array<T>)],
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>, AutogradError> {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    let mut first = true;
    if let Some(meta) = metadata {
        // Sorted for a deterministic byte stream.
        let mut keys: Vec<_> = meta.keys().collect();
        keys.sort();
        header.push_str("\"__metadata__\":{");
        for (i, k) in keys.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            json_escape(k, &mut header);
            header.push(':');
            json_escape(&meta[*k], &mut header);
        }
        header.push('}');
        first = false;
    }
    for (name, arr) in entries {
        if !first {
            header.push(',');
        }
        first = false;
        let host = arr
            .contiguous()
            .map_err(AutogradError::from)?
            .to_vec()
            .map_err(AutogradError::from)?;
        let start = data.len();
        for v in &host {
            data.extend_from_slice(&(v.to_f64() as f32).to_le_bytes());
        }
        json_escape(name, &mut header);
        header.push_str(":{\"dtype\":\"F32\",\"shape\":[");
        for (i, d) in arr.shape().iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&d.to_string());
        }
        header.push_str(&format!("],\"data_offsets\":[{},{}]}}", start, data.len()));
    }
    header.push('}');
    // Pad the header with spaces to 8-byte alignment (the convention
    // real writers follow so the data section starts aligned).
    while (header.len() + 8) % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    Ok(out)
}

/// Serialize a [`ParamTree`] under its hierarchical names (the
/// [`crate::state`] traversal), no metadata.
pub fn save<T: DiffScalar + ToF64, P: ParamTree<T>>(tree: &P) -> Result<Vec<u8>, AutogradError> {
    save_named(&tree.named_flatten(), None)
}

// ── Header JSON parser (exactly the safetensors grammar) ────────────────

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

#[derive(Debug)]
enum Json {
    Str(String),
    Num(u64),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl<'a> Parser<'a> {
    fn new(b: &'a [u8]) -> Self {
        Parser { b, i: 0 }
    }

    fn err(&self, msg: &str) -> AutogradError {
        bad(format!("safetensors header: {msg} at byte {}", self.i))
    }

    fn skip_ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn expect(&mut self, c: u8) -> Result<(), AutogradError> {
        self.skip_ws();
        if self.i < self.b.len() && self.b[self.i] == c {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", c as char)))
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }

    fn value(&mut self) -> Result<Json, AutogradError> {
        match self.peek() {
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(c) if c.is_ascii_digit() => self.number(),
            _ => Err(self.err("expected a value")),
        }
    }

    fn string(&mut self) -> Result<String, AutogradError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            let Some(&c) = self.b.get(self.i) else {
                return Err(self.err("unterminated string"));
            };
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let Some(&e) = self.b.get(self.i) else {
                        return Err(self.err("dangling escape"));
                    };
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{0008}'),
                        b'f' => s.push('\u{000C}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            let hex = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or_else(|| self.err("truncated \\u escape"))?;
                            let code = u32::from_str_radix(
                                core::str::from_utf8(hex)
                                    .map_err(|_| self.err("non-utf8 \\u escape"))?,
                                16,
                            )
                            .map_err(|_| self.err("bad \\u escape"))?;
                            self.i += 4;
                            // Surrogate pairs are not expected in tensor
                            // names; reject rather than mis-decode.
                            let ch = char::from_u32(code)
                                .ok_or_else(|| self.err("surrogate in \\u escape"))?;
                            s.push(ch);
                        }
                        _ => return Err(self.err("unknown escape")),
                    }
                }
                c => {
                    // Re-assemble multi-byte UTF-8 sequences.
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        let start = self.i - 1;
                        let len = match c {
                            0xC0..=0xDF => 2,
                            0xE0..=0xEF => 3,
                            0xF0..=0xF7 => 4,
                            _ => return Err(self.err("invalid utf-8")),
                        };
                        let bytes = self
                            .b
                            .get(start..start + len)
                            .ok_or_else(|| self.err("truncated utf-8"))?;
                        let st =
                            core::str::from_utf8(bytes).map_err(|_| self.err("invalid utf-8"))?;
                        s.push_str(st);
                        self.i = start + len;
                    }
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, AutogradError> {
        self.skip_ws();
        let start = self.i;
        while self.i < self.b.len() && self.b[self.i].is_ascii_digit() {
            self.i += 1;
        }
        if start == self.i {
            return Err(self.err("expected a number"));
        }
        let s = core::str::from_utf8(&self.b[start..self.i]).unwrap();
        s.parse::<u64>()
            .map(Json::Num)
            .map_err(|_| self.err("integer out of range"))
    }

    fn array(&mut self) -> Result<Json, AutogradError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected ',' or ']'")),
            }
        }
    }

    fn object(&mut self) -> Result<Json, AutogradError> {
        self.expect(b'{')?;
        let mut items = Vec::new();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json::Obj(items));
        }
        loop {
            let key = self.string()?;
            self.expect(b':')?;
            let val = self.value()?;
            items.push((key, val));
            match self.peek() {
                Some(b',') => {
                    self.i += 1;
                }
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(items));
                }
                _ => return Err(self.err("expected ',' or '}'")),
            }
        }
    }
}

// ── Load ────────────────────────────────────────────────────────────────

/// Decode an IEEE half (F16) to f32 — signs, subnormals, inf and NaN
/// all preserved.
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

/// Parse safetensors bytes into named `f32` tensors plus the metadata
/// map. `F32` loads exactly; `F16`/`BF16` upconvert; anything else is
/// a loud error naming the tensor.
pub fn load_named(gpu: &Gpu, bytes: &[u8]) -> Result<LoadedSafetensors, AutogradError> {
    if bytes.len() < 8 {
        return Err(bad("safetensors: shorter than the length prefix".into()));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    // The reference implementation caps headers at 100 MB — a corrupt
    // prefix must not drive a huge allocation.
    if header_len > 100_000_000 || 8 + header_len > bytes.len() {
        return Err(bad(format!(
            "safetensors: header length {header_len} exceeds the file"
        )));
    }
    let header = &bytes[8..8 + header_len];
    let data = &bytes[8 + header_len..];

    let mut p = Parser::new(header);
    let Json::Obj(entries) = p.object()? else {
        unreachable!()
    };

    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();
    for (name, val) in entries {
        if name == "__metadata__" {
            let Json::Obj(meta) = val else {
                return Err(bad("safetensors: __metadata__ must be an object".into()));
            };
            for (k, v) in meta {
                let Json::Str(s) = v else {
                    return Err(bad(format!(
                        "safetensors: __metadata__[{k}] must be a string"
                    )));
                };
                metadata.insert(k, s);
            }
            continue;
        }
        let Json::Obj(fields) = val else {
            return Err(bad(format!("safetensors: entry {name} must be an object")));
        };
        let (mut dtype, mut shape, mut offsets) = (None, None, None);
        for (k, v) in fields {
            match (k.as_str(), v) {
                ("dtype", Json::Str(s)) => dtype = Some(s),
                ("shape", Json::Arr(a)) => {
                    let mut dims = Vec::with_capacity(a.len());
                    for d in a {
                        let Json::Num(n) = d else {
                            return Err(bad(format!("safetensors: {name}: non-integer dim")));
                        };
                        dims.push(n as usize);
                    }
                    shape = Some(dims);
                }
                ("data_offsets", Json::Arr(a)) => {
                    let nums: Vec<u64> = a
                        .iter()
                        .map(|d| match d {
                            Json::Num(n) => Ok(*n),
                            _ => Err(bad(format!("safetensors: {name}: non-integer offset"))),
                        })
                        .collect::<Result<_, _>>()?;
                    if nums.len() != 2 {
                        return Err(bad(format!("safetensors: {name}: offsets must be [s, e]")));
                    }
                    offsets = Some((nums[0] as usize, nums[1] as usize));
                }
                (k, _) => {
                    return Err(bad(format!("safetensors: {name}: unknown key {k:?}")));
                }
            }
        }
        let dtype = dtype.ok_or_else(|| bad(format!("safetensors: {name}: missing dtype")))?;
        let shape = shape.ok_or_else(|| bad(format!("safetensors: {name}: missing shape")))?;
        let (start, end) =
            offsets.ok_or_else(|| bad(format!("safetensors: {name}: missing data_offsets")))?;
        if end < start || end > data.len() {
            return Err(bad(format!(
                "safetensors: {name}: offsets [{start}, {end}] exceed the data section"
            )));
        }
        let count: usize = shape.iter().product::<usize>().max(1);
        let raw = &data[start..end];
        let host: Vec<f32> = match dtype.as_str() {
            "F32" => {
                if raw.len() != count * 4 {
                    return Err(bad(format!(
                        "safetensors: {name}: F32 needs {} bytes, offsets give {}",
                        count * 4,
                        raw.len()
                    )));
                }
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect()
            }
            "F16" => {
                if raw.len() != count * 2 {
                    return Err(bad(format!(
                        "safetensors: {name}: F16 needs {} bytes, offsets give {}",
                        count * 2,
                        raw.len()
                    )));
                }
                raw.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect()
            }
            "BF16" => {
                if raw.len() != count * 2 {
                    return Err(bad(format!(
                        "safetensors: {name}: BF16 needs {} bytes, offsets give {}",
                        count * 2,
                        raw.len()
                    )));
                }
                raw.chunks_exact(2)
                    .map(|c| {
                        f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16)
                    })
                    .collect()
            }
            other => {
                return Err(bad(format!(
                    "safetensors: {name}: dtype {other} not supported (F32/F16/BF16)"
                )));
            }
        };
        let arr = Array::from_slice(gpu, &host, &shape).map_err(AutogradError::from)?;
        tensors.push((name, arr));
    }
    Ok(LoadedSafetensors { tensors, metadata })
}

/// Rebuild a tree of `witness`'s shape from safetensors bytes,
/// matching by NAME (the [`crate::state::load_state`] contract): a
/// missing, extra, or wrong-shape tensor is a loud error naming the
/// path.
pub fn load<T: DiffScalar + ToF64, P: ParamTree<T>>(
    gpu: &Gpu,
    witness: &P,
    bytes: &[u8],
) -> Result<P, AutogradError> {
    let loaded = load_named(gpu, bytes)?;
    let mut by_name: HashMap<String, Array<f32>> = loaded.tensors.into_iter().collect();
    let named = witness.named_flatten();
    let mut leaves = Vec::with_capacity(named.len());
    for (name, want) in &named {
        let got = by_name
            .remove(name)
            .ok_or_else(|| bad(format!("safetensors load: missing tensor {name:?}")))?;
        if got.shape() != want.shape() {
            return Err(bad(format!(
                "safetensors load: {name:?}: shape {:?} in file, tree wants {:?}",
                got.shape(),
                want.shape()
            )));
        }
        let host = got.to_vec().map_err(AutogradError::from)?;
        let t_host: Vec<T> = host.iter().map(|&v| T::from_f64(v as f64)).collect();
        leaves.push(Array::from_slice(gpu, &t_host, got.shape()).map_err(AutogradError::from)?);
    }
    if let Some(extra) = by_name.keys().next() {
        return Err(bad(format!(
            "safetensors load: tensor {extra:?} has no home in the tree"
        )));
    }
    witness.unflatten(&mut leaves.into_iter())
}

//! A complete hand-rolled RFC 8259 parser — the artifact substrate.
//!
//! `tokenizer.json` is a full JSON document (negative exponent floats in
//! Unigram vocabs, escaped surrogate pairs in vocab keys, `null` /
//! `true` / `false` throughout, real nesting), so unlike the
//! constrained-grammar house parsers (`safetensors.rs`, `npy_codec.rs`)
//! this one implements the *closed spec*: there is no JSON document it
//! will meet later that the grammar does not already cover.
//!
//! Deliberate strictness beyond the grammar (§3 of the crate scope):
//!
//! - **Duplicate keys in one object are rejected loudly** — a vocab
//!   with duplicate tokens is corrupt, not a last-wins guess.
//! - **Recursion depth is capped** at [`MAX_DEPTH`] (real artifacts
//!   nest ≤ ~8) so hostile nesting cannot blow the stack.
//! - **Lone surrogates are rejected**; `\uD800`–`\uDBFF` followed by a
//!   low-surrogate escape decodes to the astral codepoint.
//! - **Non-finite numbers are rejected** (`1e999` overflows `f64`;
//!   JSON cannot express infinity, so producing one would misparse).
//!
//! Hostile-input posture, npy-grade: the input is a borrowed `&[u8]`,
//! every index is bounds-checked, and no allocation is driven by an
//! unvalidated size (containers grow by push only). Errors carry byte
//! offsets — the house "at byte N" contract.

use crate::error::TokenizerError;
use std::collections::HashSet;

/// Maximum nesting depth (objects + arrays). Real artifacts nest ≤ ~8;
/// the cap keeps a crafted `[[[[…` from exhausting the stack.
pub const MAX_DEPTH: usize = 64;

/// A parsed JSON value. Objects preserve source order and are
/// duplicate-key-free by construction (the parser rejects duplicates).
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(Number),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

/// A JSON number: the correctly-rounded `f64` (std's `str::parse`) plus
/// the raw source text, so exact-integer consumers (vocab ids) never
/// round-trip through floating point.
#[derive(Debug, Clone, PartialEq)]
pub struct Number {
    /// The value as `f64` — always finite (the parser rejects overflow).
    pub value: f64,
    /// The verbatim number text from the document.
    pub raw: String,
}

impl Number {
    /// The number as `u64`, iff it is spelled as a plain unsigned
    /// decimal integer (no sign, fraction, or exponent). Ids written
    /// `1.0` or `1e2` are rejected — the reference writer never spells
    /// them that way, and accepting them would blur exactness.
    pub fn as_u64(&self) -> Option<u64> {
        if !self.raw.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        self.raw.parse().ok()
    }

    /// [`Self::as_u64`] narrowed to `u32`.
    pub fn as_u32(&self) -> Option<u32> {
        self.as_u64().and_then(|n| u32::try_from(n).ok())
    }

    /// [`Self::as_u64`] narrowed to `usize`.
    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().and_then(|n| usize::try_from(n).ok())
    }
}

impl Value {
    /// The value's kind, for schema error messages ("expected a string,
    /// found null").
    pub fn kind(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "a bool",
            Value::Number(_) => "a number",
            Value::String(_) => "a string",
            Value::Array(_) => "an array",
            Value::Object(_) => "an object",
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        self.as_number().map(|n| n.value)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Object member lookup; `None` on non-objects and absent keys.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// Parse one JSON document. The whole input must be a single value plus
/// optional whitespace — trailing bytes are a loud error.
pub fn parse(bytes: &[u8]) -> Result<Value, TokenizerError> {
    let mut p = Parser {
        b: bytes,
        i: 0,
        depth: 0,
    };
    let v = p.value()?;
    p.skip_ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing data after the top-level value"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn err(&self, msg: &str) -> TokenizerError {
        TokenizerError::Json {
            at: self.i,
            what: msg.to_string(),
        }
    }

    fn err_at(&self, at: usize, msg: String) -> TokenizerError {
        TokenizerError::Json { at, what: msg }
    }

    fn skip_ws(&mut self) {
        while let Some(&c) = self.b.get(self.i) {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.b.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<(), TokenizerError> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", c as char)))
        }
    }

    fn enter(&mut self) -> Result<(), TokenizerError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err(&format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        Ok(())
    }

    fn value(&mut self) -> Result<Value, TokenizerError> {
        match self.peek() {
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Value::String(self.string()?)),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("expected a value")),
            None => Err(self.err("input ends where a value is expected")),
        }
    }

    fn literal(&mut self, word: &'static [u8], v: Value) -> Result<Value, TokenizerError> {
        if self.b.get(self.i..self.i + word.len()) == Some(word) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("expected a value"))
        }
    }

    fn object(&mut self) -> Result<Value, TokenizerError> {
        self.expect(b'{')?;
        self.enter()?;
        let mut entries: Vec<(String, Value)> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        if self.peek() == Some(b'}') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Object(entries));
        }
        loop {
            if self.peek() != Some(b'"') {
                return Err(self.err("expected a string object key"));
            }
            let key_at = self.i;
            let key = self.string()?;
            if !seen.insert(key.clone()) {
                return Err(self.err_at(key_at, format!("duplicate key {key:?} in object")));
            }
            self.expect(b':')?;
            let v = self.value()?;
            entries.push((key, v));
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
        self.depth -= 1;
        Ok(Value::Object(entries))
    }

    fn array(&mut self) -> Result<Value, TokenizerError> {
        self.expect(b'[')?;
        self.enter()?;
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.i += 1;
            self.depth -= 1;
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            match self.peek() {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    break;
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
        self.depth -= 1;
        Ok(Value::Array(items))
    }

    // ── Strings ─────────────────────────────────────────────────────────

    fn string(&mut self) -> Result<String, TokenizerError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            let Some(&c) = self.b.get(self.i) else {
                return Err(self.err("unterminated string"));
            };
            match c {
                b'"' => {
                    self.i += 1;
                    return Ok(s);
                }
                b'\\' => {
                    self.i += 1;
                    self.escape(&mut s)?;
                }
                0x00..=0x1F => {
                    return Err(self.err("unescaped control character in string"));
                }
                0x20..=0x7F => {
                    s.push(c as char);
                    self.i += 1;
                }
                _ => self.utf8_sequence(&mut s)?,
            }
        }
    }

    /// One escape, cursor just past the backslash.
    fn escape(&mut self, s: &mut String) -> Result<(), TokenizerError> {
        let at = self.i - 1;
        let Some(&e) = self.b.get(self.i) else {
            return Err(self.err_at(at, "dangling escape at end of input".to_string()));
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
                let hi = self.hex4()?;
                match hi {
                    0xD800..=0xDBFF => {
                        // A high surrogate must pair with a following
                        // low-surrogate escape — together they name one
                        // astral codepoint.
                        if self.b.get(self.i..self.i + 2) != Some(b"\\u") {
                            return Err(self.err_at(
                                at,
                                format!(
                                    "lone high surrogate \\u{hi:04X} (a low surrogate \
                                     escape must follow)"
                                ),
                            ));
                        }
                        self.i += 2;
                        let lo = self.hex4()?;
                        if !(0xDC00..=0xDFFF).contains(&lo) {
                            return Err(self.err_at(
                                at,
                                format!(
                                    "high surrogate \\u{hi:04X} followed by \\u{lo:04X}, \
                                     which is not a low surrogate"
                                ),
                            ));
                        }
                        let code = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                        // In range by construction: 0x10000..=0x10FFFF,
                        // never a surrogate.
                        s.push(char::from_u32(code).expect("astral codepoint in range"));
                    }
                    0xDC00..=0xDFFF => {
                        return Err(self.err_at(
                            at,
                            format!("lone low surrogate \\u{hi:04X} (no preceding high)"),
                        ));
                    }
                    _ => {
                        // Non-surrogate BMP codepoints are all valid chars.
                        s.push(char::from_u32(hi).expect("BMP non-surrogate is a char"));
                    }
                }
            }
            _ => {
                return Err(self.err_at(at, format!("unknown escape '\\{}'", char::from(e))));
            }
        }
        Ok(())
    }

    /// Four hex digits, cursor just past `\u` (or a continuation `\u`).
    fn hex4(&mut self) -> Result<u32, TokenizerError> {
        let hex = self
            .b
            .get(self.i..self.i + 4)
            .ok_or_else(|| self.err("truncated \\u escape"))?;
        if !hex.iter().all(u8::is_ascii_hexdigit) {
            return Err(self.err("non-hex digit in \\u escape"));
        }
        let text = core::str::from_utf8(hex).expect("hex digits are ASCII");
        let code = u32::from_str_radix(text, 16).expect("4 hex digits fit u32");
        self.i += 4;
        Ok(code)
    }

    /// One multi-byte UTF-8 sequence, cursor on the lead byte.
    /// `str::from_utf8` on the exact slice rejects overlong encodings,
    /// UTF-8-encoded surrogates, and codepoints past U+10FFFF.
    fn utf8_sequence(&mut self, s: &mut String) -> Result<(), TokenizerError> {
        let lead = self.b[self.i];
        let len = match lead {
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => return Err(self.err("invalid UTF-8 lead byte in string")),
        };
        let bytes = self
            .b
            .get(self.i..self.i + len)
            .ok_or_else(|| self.err("truncated UTF-8 sequence in string"))?;
        let text = core::str::from_utf8(bytes)
            .map_err(|_| self.err("invalid UTF-8 sequence in string"))?;
        s.push_str(text);
        self.i += len;
        Ok(())
    }

    // ── Numbers ─────────────────────────────────────────────────────────

    /// The full RFC 8259 number grammar, lexed by hand; value conversion
    /// is std's correctly-rounded `str::parse::<f64>()`.
    fn number(&mut self) -> Result<Value, TokenizerError> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        // Integer part: '0' alone, or a nonzero digit then any digits.
        match self.b.get(self.i) {
            Some(b'0') => {
                self.i += 1;
                if matches!(self.b.get(self.i), Some(c) if c.is_ascii_digit()) {
                    return Err(self.err("leading zero in number"));
                }
            }
            Some(c) if c.is_ascii_digit() => {
                while matches!(self.b.get(self.i), Some(c) if c.is_ascii_digit()) {
                    self.i += 1;
                }
            }
            _ => return Err(self.err("expected a digit")),
        }
        // Fraction.
        if self.b.get(self.i) == Some(&b'.') {
            self.i += 1;
            self.digits1("expected a digit after the decimal point")?;
        }
        // Exponent.
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            self.digits1("expected a digit in the exponent")?;
        }
        let raw = core::str::from_utf8(&self.b[start..self.i]).expect("number lexeme is ASCII");
        let value: f64 = raw
            .parse()
            .map_err(|_| self.err_at(start, format!("number {raw:?} is not representable")))?;
        if !value.is_finite() {
            return Err(self.err_at(start, format!("number {raw} overflows f64")));
        }
        Ok(Value::Number(Number {
            value,
            raw: raw.to_string(),
        }))
    }

    /// At least one digit.
    fn digits1(&mut self, msg: &str) -> Result<(), TokenizerError> {
        if !matches!(self.b.get(self.i), Some(c) if c.is_ascii_digit()) {
            return Err(self.err(msg));
        }
        while matches!(self.b.get(self.i), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
        }
        Ok(())
    }
}

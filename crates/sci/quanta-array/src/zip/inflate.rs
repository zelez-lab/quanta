//! RFC 1951 inflate — hand-rolled, zero-dep, hostile-input-grade.
//!
//! Shape: an LSB-first bit reader over the raw deflate stream, canonical
//! Huffman decoding (the count/offset algorithm from the RFC — bit-serial,
//! auditability over speed), and the three block types: stored, fixed
//! Huffman, dynamic Huffman, with window copies.
//!
//! Hostile-input posture:
//! - every read is bounds-checked; truncation at any bit is a loud error,
//!   never a panic;
//! - the output is capped at the caller's declared uncompressed size (the
//!   central directory's claim) — one byte over is the zip-bomb error;
//! - the declared size never drives a large upfront allocation (reserve
//!   is capped; growth is earned by actual decoded bytes);
//! - oversubscribed Huffman codes are rejected at table build; incomplete
//!   codes are tolerated (zlib emits degenerate single-code distance
//!   tables) but decoding into a gap is a loud error.
//!
//! Trailing bytes after the final block are ignored: the container's
//! compressed-size field delimits the stream, and the CRC + declared-size
//! checks upstream catch real corruption.
//!
//! Errors are plain strings; the zip layer folds them into
//! `NpyError::Zip { entry, what }` with the entry name attached.

/// Cap on the upfront reservation — the declared size is an
/// attacker-controlled claim, so it bounds output, not allocation.
const MAX_INITIAL_RESERVE: usize = 1 << 20;

const MAX_BITS: usize = 15;

/// Length-code bases and extra bits, symbols 257..=285.
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Distance-code bases and extra bits, symbols 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Order in which the code-length-code lengths are transmitted.
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

struct BitReader<'a> {
    b: &'a [u8],
    pos: usize,
    bit: u8,
}

impl BitReader<'_> {
    /// Read `count` bits, LSB-first. `count` ≤ 16.
    fn bits(&mut self, count: u32) -> Result<u32, String> {
        let mut v = 0u32;
        for i in 0..count {
            let byte = *self
                .b
                .get(self.pos)
                .ok_or_else(|| "unexpected end of deflate stream".to_string())?;
            v |= (((byte >> self.bit) & 1) as u32) << i;
            self.bit += 1;
            if self.bit == 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }
        Ok(v)
    }

    /// Discard bits up to the next byte boundary (stored-block entry).
    fn align(&mut self) {
        if self.bit != 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }
}

/// A canonical Huffman code: `count[len]` codes of each length, symbols
/// sorted by (length, symbol value).
struct Huffman {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Huffman {
    /// Build from per-symbol code lengths (0 = unused). Oversubscription
    /// is a hard error; an incomplete code is returned as-is and errors
    /// only if a decode walks into the gap.
    fn build(lengths: &[u16]) -> Result<Huffman, String> {
        let mut count = [0u16; MAX_BITS + 1];
        for &l in lengths {
            debug_assert!(
                l as usize <= MAX_BITS,
                "lengths come from 4-bit/3-bit fields"
            );
            count[l as usize] += 1;
        }
        if count[0] as usize == lengths.len() {
            // No codes at all — legal for an unused distance alphabet.
            return Ok(Huffman {
                count,
                symbol: Vec::new(),
            });
        }
        let mut left: i32 = 1;
        for &c in &count[1..] {
            left <<= 1;
            left -= c as i32;
            if left < 0 {
                return Err("oversubscribed Huffman code".to_string());
            }
        }
        let mut offs = [0u16; MAX_BITS + 1];
        for len in 1..MAX_BITS {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.iter().filter(|&&l| l != 0).count()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Ok(Huffman { count, symbol })
    }

    /// Decode one symbol, bit-serial over the canonical code.
    fn decode(&self, br: &mut BitReader) -> Result<u16, String> {
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        let mut index: u32 = 0;
        for len in 1..=MAX_BITS {
            code |= br.bits(1)?;
            let count = self.count[len] as u32;
            if code < first + count {
                return Ok(self.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("invalid Huffman code in deflate stream".to_string())
    }
}

struct Inflater<'a> {
    br: BitReader<'a>,
    out: Vec<u8>,
    /// The declared uncompressed size — the output cap (zip-bomb guard).
    cap: usize,
}

impl Inflater<'_> {
    fn bomb(&self) -> String {
        format!(
            "output exceeds the declared uncompressed size {} — zip-bomb guard",
            self.cap
        )
    }

    fn stored_block(&mut self) -> Result<(), String> {
        self.br.align();
        let pos = self.br.pos;
        let hdr = self
            .br
            .b
            .get(pos..pos + 4)
            .ok_or_else(|| "truncated stored-block header".to_string())?;
        let len = u16::from_le_bytes([hdr[0], hdr[1]]) as usize;
        let nlen = u16::from_le_bytes([hdr[2], hdr[3]]);
        if len != (!nlen) as usize {
            return Err("stored block LEN/NLEN mismatch".to_string());
        }
        let data = self
            .br
            .b
            .get(pos + 4..pos + 4 + len)
            .ok_or_else(|| "deflate stream ends inside a stored block".to_string())?;
        if self.out.len() + len > self.cap {
            return Err(self.bomb());
        }
        self.out.extend_from_slice(data);
        self.br.pos = pos + 4 + len;
        Ok(())
    }

    fn compressed_block(&mut self, lit: &Huffman, dist: &Huffman) -> Result<(), String> {
        loop {
            let sym = lit.decode(&mut self.br)?;
            match sym {
                0..=255 => {
                    if self.out.len() >= self.cap {
                        return Err(self.bomb());
                    }
                    self.out.push(sym as u8);
                }
                256 => return Ok(()),
                257..=285 => {
                    let idx = (sym - 257) as usize;
                    let len = LEN_BASE[idx] as usize + self.br.bits(LEN_EXTRA[idx])? as usize;
                    let dsym = dist.decode(&mut self.br)? as usize;
                    if dsym > 29 {
                        return Err(format!("invalid distance code {dsym}"));
                    }
                    let d = DIST_BASE[dsym] as usize + self.br.bits(DIST_EXTRA[dsym])? as usize;
                    if d > self.out.len() {
                        return Err(format!(
                            "distance {d} reaches before the start of the output"
                        ));
                    }
                    if self.out.len() + len > self.cap {
                        return Err(self.bomb());
                    }
                    // Overlapping copies are the point (d < len repeats).
                    let start = self.out.len() - d;
                    for k in 0..len {
                        let byte = self.out[start + k];
                        self.out.push(byte);
                    }
                }
                _ => return Err(format!("invalid literal/length symbol {sym}")),
            }
        }
    }

    /// Read the dynamic-block code descriptions and build both tables.
    fn dynamic_tables(&mut self) -> Result<(Huffman, Huffman), String> {
        let hlit = self.br.bits(5)? as usize + 257;
        let hdist = self.br.bits(5)? as usize + 1;
        let hclen = self.br.bits(4)? as usize + 4;
        if hlit > 286 {
            return Err(format!("too many literal/length codes ({hlit})"));
        }
        if hdist > 30 {
            return Err(format!("too many distance codes ({hdist})"));
        }
        let mut cl = [0u16; 19];
        for &sym in CL_ORDER.iter().take(hclen) {
            cl[sym] = self.br.bits(3)? as u16;
        }
        let cl_huff = Huffman::build(&cl)?;

        let mut lengths = vec![0u16; hlit + hdist];
        let mut i = 0;
        while i < lengths.len() {
            let sym = cl_huff.decode(&mut self.br)?;
            match sym {
                0..=15 => {
                    lengths[i] = sym;
                    i += 1;
                }
                16 => {
                    if i == 0 {
                        return Err("length repeat with no previous code length".to_string());
                    }
                    let prev = lengths[i - 1];
                    let n = 3 + self.br.bits(2)? as usize;
                    if i + n > lengths.len() {
                        return Err("code-length repeat overruns the table".to_string());
                    }
                    for _ in 0..n {
                        lengths[i] = prev;
                        i += 1;
                    }
                }
                17 | 18 => {
                    let n = if sym == 17 {
                        3 + self.br.bits(3)? as usize
                    } else {
                        11 + self.br.bits(7)? as usize
                    };
                    if i + n > lengths.len() {
                        return Err("code-length repeat overruns the table".to_string());
                    }
                    i += n; // already zero
                }
                _ => return Err(format!("code-length symbol {sym} out of range")),
            }
        }
        if lengths[256] == 0 {
            return Err("dynamic block has no end-of-block code".to_string());
        }
        Ok((
            Huffman::build(&lengths[..hlit])?,
            Huffman::build(&lengths[hlit..])?,
        ))
    }
}

/// The fixed lit/len and distance codes of RFC 1951 §3.2.6.
fn fixed_tables() -> (Huffman, Huffman) {
    let mut lit = [0u16; 288];
    lit[..144].fill(8);
    lit[144..256].fill(9);
    lit[256..280].fill(7);
    lit[280..].fill(8);
    let dist = [5u16; 32];
    (
        Huffman::build(&lit).expect("the fixed literal table is complete"),
        Huffman::build(&dist).expect("the fixed distance table is complete"),
    )
}

/// Inflate a raw deflate stream, capping the output at `declared_len`
/// (the container's claimed uncompressed size). Returns the decoded
/// bytes; the caller checks the exact-length and CRC contracts.
pub fn inflate(input: &[u8], declared_len: usize) -> Result<Vec<u8>, String> {
    let mut s = Inflater {
        br: BitReader {
            b: input,
            pos: 0,
            bit: 0,
        },
        out: Vec::with_capacity(declared_len.min(MAX_INITIAL_RESERVE)),
        cap: declared_len,
    };
    loop {
        let bfinal = s.br.bits(1)?;
        match s.br.bits(2)? {
            0 => s.stored_block()?,
            1 => {
                let (lit, dist) = fixed_tables();
                s.compressed_block(&lit, &dist)?;
            }
            2 => {
                let (lit, dist) = s.dynamic_tables()?;
                s.compressed_block(&lit, &dist)?;
            }
            _ => return Err("reserved deflate block type 3".to_string()),
        }
        if bfinal == 1 {
            break;
        }
    }
    Ok(s.out)
}

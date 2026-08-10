//! Unicode support for the tokenizer pipeline (scope §4).
//!
//! Everything here runs on vendored generated tables (`tables.rs`, emitted
//! by the pinned `gen_unicode_tables.py` from UCD 16.0.0 — see that file's
//! provenance header). The surface is exactly what the pipeline stages
//! consume:
//!
//! - [`nfc`] / [`nfd`] / [`nfkc`] / [`nfkd`] — UAX #15 normalization
//!   (canonical ordering by combining class, composition with the
//!   Full_Composition_Exclusion set applied, algorithmic Hangul) for the
//!   `NFC`/`NFD`/`NFKC`/`NFKD` normalizers and the decomposition step of
//!   `BertNormalizer`/`StripAccents`.
//! - [`general_category`] / [`category_class`] and the predicates
//!   ([`is_letter`], [`is_number`], [`is_mark`], [`is_nonspacing_mark`],
//!   [`is_whitespace`]) — the `\p{L}`-class regex tables and the
//!   `Mn`-mark check `strip_accents` needs.
//! - [`lowercase`] — full unconditional lowercase mappings (the
//!   `char::to_lowercase` surface the reference's `Lowercase` /
//!   `BertNormalizer` paths use, U+0130 included).
//! - [`script`] — Script ranges for the `UnicodeScripts` pre-tokenizer.
//! - [`graphemes`] — UAX #29 extended grapheme clusters, consumed by the
//!   `Precompiled` charsmap transform below (and by the normalizer layer's
//!   alignment-tracked integration).
//! - [`Charsmap`] — the sentencepiece `Precompiled` double-array-trie
//!   walker. The blob arrives inside artifacts at runtime and is treated
//!   as hostile: every transition is bounds-checked.

mod tables;

pub use tables::Script;

use std::fmt;

// ---------------------------------------------------------------------------
// General categories.
// ---------------------------------------------------------------------------

/// A Unicode general category (`Gc`). Variant order mirrors the generated
/// table's index order (alphabetical). `Cs` is unreachable from a `char`
/// (surrogates are not scalar values) but kept so the table is the full
/// 30-category partition.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::doc_markdown)]
pub enum GeneralCategory {
    /// Control
    Cc,
    /// Format
    Cf,
    /// Unassigned
    Cn,
    /// Private use
    Co,
    /// Surrogate
    Cs,
    /// Lowercase letter
    Ll,
    /// Modifier letter
    Lm,
    /// Other letter
    Lo,
    /// Titlecase letter
    Lt,
    /// Uppercase letter
    Lu,
    /// Spacing mark
    Mc,
    /// Enclosing mark
    Me,
    /// Nonspacing mark
    Mn,
    /// Decimal number
    Nd,
    /// Letter number
    Nl,
    /// Other number
    No,
    /// Connector punctuation
    Pc,
    /// Dash punctuation
    Pd,
    /// Close punctuation
    Pe,
    /// Final punctuation
    Pf,
    /// Initial punctuation
    Pi,
    /// Other punctuation
    Po,
    /// Open punctuation
    Ps,
    /// Currency symbol
    Sc,
    /// Modifier symbol
    Sk,
    /// Math symbol
    Sm,
    /// Other symbol
    So,
    /// Line separator
    Zl,
    /// Paragraph separator
    Zp,
    /// Space separator
    Zs,
}

impl GeneralCategory {
    const ALL: [GeneralCategory; 30] = [
        GeneralCategory::Cc,
        GeneralCategory::Cf,
        GeneralCategory::Cn,
        GeneralCategory::Co,
        GeneralCategory::Cs,
        GeneralCategory::Ll,
        GeneralCategory::Lm,
        GeneralCategory::Lo,
        GeneralCategory::Lt,
        GeneralCategory::Lu,
        GeneralCategory::Mc,
        GeneralCategory::Me,
        GeneralCategory::Mn,
        GeneralCategory::Nd,
        GeneralCategory::Nl,
        GeneralCategory::No,
        GeneralCategory::Pc,
        GeneralCategory::Pd,
        GeneralCategory::Pe,
        GeneralCategory::Pf,
        GeneralCategory::Pi,
        GeneralCategory::Po,
        GeneralCategory::Ps,
        GeneralCategory::Sc,
        GeneralCategory::Sk,
        GeneralCategory::Sm,
        GeneralCategory::So,
        GeneralCategory::Zl,
        GeneralCategory::Zp,
        GeneralCategory::Zs,
    ];

    /// The category's major class (first letter: `L`, `M`, `N`, `P`, `S`,
    /// `Z`, `C`) — the granularity `\p{L}`-style regex classes use.
    pub fn class(self) -> CategoryClass {
        use GeneralCategory as G;
        match self {
            G::Ll | G::Lm | G::Lo | G::Lt | G::Lu => CategoryClass::Letter,
            G::Mc | G::Me | G::Mn => CategoryClass::Mark,
            G::Nd | G::Nl | G::No => CategoryClass::Number,
            G::Pc | G::Pd | G::Pe | G::Pf | G::Pi | G::Po | G::Ps => CategoryClass::Punctuation,
            G::Sc | G::Sk | G::Sm | G::So => CategoryClass::Symbol,
            G::Zl | G::Zp | G::Zs => CategoryClass::Separator,
            G::Cc | G::Cf | G::Cn | G::Co | G::Cs => CategoryClass::Other,
        }
    }
}

/// A major general-category class (`L`, `M`, `N`, `P`, `S`, `Z`, `C`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CategoryClass {
    /// `L*` — letters.
    Letter,
    /// `M*` — combining marks.
    Mark,
    /// `N*` — numbers.
    Number,
    /// `P*` — punctuation.
    Punctuation,
    /// `S*` — symbols.
    Symbol,
    /// `Z*` — separators.
    Separator,
    /// `C*` — control, format, unassigned, private use.
    Other,
}

/// Index of the partition run covering `cp`. `starts[0] == 0`, so the
/// subtraction cannot underflow.
fn partition_index(starts: &[u32], cp: u32) -> usize {
    starts.partition_point(|&s| s <= cp) - 1
}

/// Membership test over parallel sorted inclusive-range tables.
fn in_ranges(starts: &[u32], ends: &[u32], cp: u32) -> bool {
    let i = starts.partition_point(|&s| s <= cp);
    i > 0 && cp <= ends[i - 1]
}

/// The Unicode general category of `c` (unassigned code points are `Cn`).
pub fn general_category(c: char) -> GeneralCategory {
    let i = partition_index(&tables::GENERAL_CATEGORY_STARTS, c as u32);
    GeneralCategory::ALL[tables::GENERAL_CATEGORY_VALUES[i] as usize]
}

/// The major class (`L`/`M`/`N`/`P`/`S`/`Z`/`C`) of `c`'s general category.
pub fn category_class(c: char) -> CategoryClass {
    general_category(c).class()
}

/// `\p{L}` — `c` is a letter (`Lu`, `Ll`, `Lt`, `Lm`, `Lo`).
pub fn is_letter(c: char) -> bool {
    category_class(c) == CategoryClass::Letter
}

/// `\p{N}` — `c` is a number (`Nd`, `Nl`, `No`).
pub fn is_number(c: char) -> bool {
    category_class(c) == CategoryClass::Number
}

/// `\p{M}` — `c` is a combining mark (`Mn`, `Mc`, `Me`).
pub fn is_mark(c: char) -> bool {
    category_class(c) == CategoryClass::Mark
}

/// `c` is a nonspacing mark (`Mn`) — the exact predicate `strip_accents`
/// filters on after NFD (reference semantics: drop `Mn`, keep `Mc`/`Me`).
pub fn is_nonspacing_mark(c: char) -> bool {
    general_category(c) == GeneralCategory::Mn
}

/// The `White_Space` property (PropList.txt) — the `\s` regex class and
/// `char::is_whitespace` surface. Note the reference's *Bert*-flavored
/// whitespace test is narrower (`\t\n\r ` + `Zs`); build that one from
/// [`general_category`].
pub fn is_whitespace(c: char) -> bool {
    in_ranges(
        tables::WHITE_SPACE_STARTS,
        tables::WHITE_SPACE_ENDS,
        c as u32,
    )
}

/// The canonical combining class (`ccc`) of `c`; starters return 0.
pub fn combining_class(c: char) -> u8 {
    let cp = c as u32;
    let i = tables::CCC_STARTS.partition_point(|&s| s <= cp);
    if i > 0 && cp <= tables::CCC_ENDS[i - 1] {
        tables::CCC_VALUES[i - 1]
    } else {
        0
    }
}

/// The script of `c` (Scripts.txt); unassigned / unlisted code points are
/// [`Script::Unknown`].
pub fn script(c: char) -> Script {
    let i = partition_index(&tables::SCRIPT_STARTS, c as u32);
    tables::SCRIPT_BY_INDEX[tables::SCRIPT_VALUES[i] as usize]
}

// ---------------------------------------------------------------------------
// Lowercase.
// ---------------------------------------------------------------------------

/// Iterator returned by [`lowercase`].
pub struct Lowercase(LowercaseRepr);

enum LowercaseRepr {
    Single(Option<char>),
    Pool(std::slice::Iter<'static, u32>),
}

impl Iterator for Lowercase {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        match &mut self.0 {
            LowercaseRepr::Single(c) => c.take(),
            LowercaseRepr::Pool(it) => it.next().map(|&cp| from_table_cp(cp)),
        }
    }
}

/// The full unconditional lowercase mapping of `c` (the reference's
/// per-char `char::to_lowercase` surface): UnicodeData simple mappings
/// plus the unconditional SpecialCasing entries, so `İ` (U+0130) yields
/// `i` followed by U+0307. Chars without a mapping yield themselves.
pub fn lowercase(c: char) -> Lowercase {
    let cp = c as u32;
    if let Ok(i) = tables::LOWERCASE_KEYS.binary_search(&cp) {
        let mapped = cp.wrapping_add_signed(tables::LOWERCASE_DELTAS[i]);
        return Lowercase(LowercaseRepr::Single(Some(from_table_cp(mapped))));
    }
    if let Ok(i) = tables::LOWERCASE_MULTI_KEYS.binary_search(&cp) {
        let v = tables::LOWERCASE_MULTI_VALUES[i];
        let (off, len) = ((v >> 5) as usize, (v & 31) as usize);
        return Lowercase(LowercaseRepr::Pool(
            tables::LOWERCASE_POOL[off..off + len].iter(),
        ));
    }
    Lowercase(LowercaseRepr::Single(Some(c)))
}

/// Generated tables only hold scalar values; this cannot fail on them.
fn from_table_cp(cp: u32) -> char {
    char::from_u32(cp).expect("generated table holds a non-scalar value")
}

// ---------------------------------------------------------------------------
// Normalization (UAX #15).
// ---------------------------------------------------------------------------

const HANGUL_S_BASE: u32 = 0xAC00;
const HANGUL_L_BASE: u32 = 0x1100;
const HANGUL_V_BASE: u32 = 0x1161;
const HANGUL_T_BASE: u32 = 0x11A7;
const HANGUL_L_COUNT: u32 = 19;
const HANGUL_V_COUNT: u32 = 21;
const HANGUL_T_COUNT: u32 = 28;
const HANGUL_N_COUNT: u32 = HANGUL_V_COUNT * HANGUL_T_COUNT;
const HANGUL_S_COUNT: u32 = HANGUL_L_COUNT * HANGUL_N_COUNT;

/// Normalization Form C: canonical decomposition + canonical composition.
pub fn nfc(s: &str) -> String {
    normalize(s, false, true)
}

/// Normalization Form D: canonical decomposition, canonically ordered.
pub fn nfd(s: &str) -> String {
    normalize(s, false, false)
}

/// Normalization Form KC: compatibility decomposition + canonical
/// composition.
pub fn nfkc(s: &str) -> String {
    normalize(s, true, true)
}

/// Normalization Form KD: compatibility decomposition, canonically ordered.
pub fn nfkd(s: &str) -> String {
    normalize(s, true, false)
}

fn normalize(s: &str, compat: bool, compose_after: bool) -> String {
    // All four forms are the identity on pure-ASCII text.
    if s.is_ascii() {
        return s.to_owned();
    }
    let mut buf: Vec<char> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose_char(c, compat, &mut buf);
    }
    if compose_after {
        compose(buf)
    } else {
        buf.into_iter().collect()
    }
}

/// Pushes the full (recursive) decomposition of `c`, keeping `out`
/// canonically ordered (UAX #15 D109: stable insertion by combining class).
fn decompose_char(c: char, compat: bool, out: &mut Vec<char>) {
    let cp = c as u32;
    if cp < 0x80 {
        out.push(c);
        return;
    }
    // Hangul syllables decompose algorithmically (UAX #15 §3.12); the jamo
    // are all starters, so plain pushes keep the canonical order.
    if (HANGUL_S_BASE..HANGUL_S_BASE + HANGUL_S_COUNT).contains(&cp) {
        let s = cp - HANGUL_S_BASE;
        out.push(from_table_cp(HANGUL_L_BASE + s / HANGUL_N_COUNT));
        out.push(from_table_cp(
            HANGUL_V_BASE + (s % HANGUL_N_COUNT) / HANGUL_T_COUNT,
        ));
        if !s.is_multiple_of(HANGUL_T_COUNT) {
            out.push(from_table_cp(HANGUL_T_BASE + s % HANGUL_T_COUNT));
        }
        return;
    }
    if compat && let Ok(i) = tables::COMPAT_DECOMP_KEYS.binary_search(&cp) {
        let v = tables::COMPAT_DECOMP_VALUES[i];
        let (off, len) = ((v >> 5) as usize, (v & 31) as usize);
        for &e in &tables::COMPAT_DECOMP_POOL[off..off + len] {
            decompose_char(from_table_cp(e), compat, out);
        }
        return;
    }
    if let Ok(i) = tables::CANONICAL_DECOMP_KEYS.binary_search(&cp) {
        decompose_char(
            from_table_cp(tables::CANONICAL_DECOMP_FIRSTS[i]),
            compat,
            out,
        );
        let second = tables::CANONICAL_DECOMP_SECONDS[i];
        if second != 0 {
            decompose_char(from_table_cp(second), compat, out);
        }
        return;
    }
    push_canonical_ordered(out, c);
}

fn push_canonical_ordered(out: &mut Vec<char>, c: char) {
    let ccc = combining_class(c);
    if ccc == 0 {
        out.push(c);
        return;
    }
    let mut i = out.len();
    while i > 0 && combining_class(out[i - 1]) > ccc {
        i -= 1;
    }
    out.insert(i, c);
}

/// Canonical composition (UAX #15 D117) over a canonically ordered buffer.
fn compose(chars: Vec<char>) -> String {
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut starter: Option<usize> = None;
    // `None` while nothing sits between the starter and the candidate;
    // otherwise the combining class of the immediately preceding char
    // (D105 blocking: a starter or a class >= the candidate's blocks).
    let mut last_ccc: Option<u8> = None;
    for c in chars {
        let ccc = combining_class(c);
        if let Some(si) = starter {
            let unblocked = match last_ccc {
                None => true,
                Some(prev) => prev < ccc,
            };
            if unblocked && let Some(p) = primary_composite(out[si], c) {
                out[si] = p;
                continue;
            }
        }
        out.push(c);
        if ccc == 0 {
            starter = Some(out.len() - 1);
            last_ccc = None;
        } else {
            last_ccc = Some(ccc);
        }
    }
    out.into_iter().collect()
}

/// The primary composite of `a` + `b`, if any: algorithmic Hangul (L+V,
/// LV+T) plus the table pairs (Full_Composition_Exclusion pre-applied).
fn primary_composite(a: char, b: char) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (HANGUL_L_BASE..HANGUL_L_BASE + HANGUL_L_COUNT).contains(&a)
        && (HANGUL_V_BASE..HANGUL_V_BASE + HANGUL_V_COUNT).contains(&b)
    {
        let lv = HANGUL_S_BASE
            + ((a - HANGUL_L_BASE) * HANGUL_V_COUNT + (b - HANGUL_V_BASE)) * HANGUL_T_COUNT;
        return Some(from_table_cp(lv));
    }
    if (HANGUL_S_BASE..HANGUL_S_BASE + HANGUL_S_COUNT).contains(&a)
        && (a - HANGUL_S_BASE).is_multiple_of(HANGUL_T_COUNT)
        && (HANGUL_T_BASE + 1..HANGUL_T_BASE + HANGUL_T_COUNT).contains(&b)
    {
        return Some(from_table_cp(a + (b - HANGUL_T_BASE)));
    }
    let key = (u64::from(a) << 21) | u64::from(b);
    tables::COMPOSE_KEYS
        .binary_search(&key)
        .ok()
        .map(|i| from_table_cp(tables::COMPOSE_VALUES[i]))
}

// ---------------------------------------------------------------------------
// Grapheme clusters (UAX #29).
// ---------------------------------------------------------------------------

/// UAX #29 Grapheme_Cluster_Break classes; order mirrors the generated
/// table's indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphemeBreak {
    Other,
    Cr,
    Lf,
    Control,
    Extend,
    Zwj,
    RegionalIndicator,
    Prepend,
    SpacingMark,
    L,
    V,
    T,
    Lv,
    Lvt,
}

fn grapheme_break(c: char) -> GraphemeBreak {
    use GraphemeBreak as B;
    const CLASSES: [GraphemeBreak; 14] = [
        B::Other,
        B::Cr,
        B::Lf,
        B::Control,
        B::Extend,
        B::Zwj,
        B::RegionalIndicator,
        B::Prepend,
        B::SpacingMark,
        B::L,
        B::V,
        B::T,
        B::Lv,
        B::Lvt,
    ];
    let i = partition_index(&tables::GRAPHEME_BREAK_STARTS, c as u32);
    CLASSES[tables::GRAPHEME_BREAK_VALUES[i] as usize]
}

fn is_extended_pictographic(c: char) -> bool {
    in_ranges(
        tables::EXTENDED_PICTOGRAPHIC_STARTS,
        tables::EXTENDED_PICTOGRAPHIC_ENDS,
        c as u32,
    )
}

/// Iterates the extended grapheme clusters of `s` (UAX #29 rules GB1-GB13;
/// GB9c, the Indic conjunct-break rule, is intentionally not implemented —
/// every InCB sequence is >= 6 bytes of UTF-8, so the omission cannot
/// change [`Charsmap::transform`]'s reference-parity windows, its consumer
/// here).
pub fn graphemes(s: &str) -> Graphemes<'_> {
    Graphemes { rest: s }
}

/// Iterator returned by [`graphemes`].
pub struct Graphemes<'a> {
    rest: &'a str,
}

impl<'a> Iterator for Graphemes<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        if self.rest.is_empty() {
            return None;
        }
        let end = next_grapheme_boundary(self.rest);
        let (g, rest) = self.rest.split_at(end);
        self.rest = rest;
        Some(g)
    }
}

/// Byte index of the first grapheme boundary after the start of `s`
/// (`s.len()` when `s` is a single cluster). `s` must be non-empty.
fn next_grapheme_boundary(s: &str) -> usize {
    use GraphemeBreak as B;
    let mut iter = s.char_indices();
    let (_, first) = iter.next().expect("caller guarantees non-empty input");
    let mut prev = grapheme_break(first);
    // GB11 left context: does the text end in ExtPict Extend* (`ep_ext`),
    // or in ExtPict Extend* ZWJ (`ep_zwj`)?
    let mut ep_ext = is_extended_pictographic(first);
    let mut ep_zwj = false;
    // GB12/13 left context: length of the trailing regional-indicator run.
    let mut ri_run = usize::from(prev == B::RegionalIndicator);

    for (at, c) in iter {
        let cur = grapheme_break(c);
        let boundary = match (prev, cur) {
            (B::Cr, B::Lf) => false,                                       // GB3
            (B::Control | B::Cr | B::Lf, _) => true,                       // GB4
            (_, B::Control | B::Cr | B::Lf) => true,                       // GB5
            (B::L, B::L | B::V | B::Lv | B::Lvt) => false,                 // GB6
            (B::Lv | B::V, B::V | B::T) => false,                          // GB7
            (B::Lvt | B::T, B::T) => false,                                // GB8
            (_, B::Extend | B::Zwj) => false,                              // GB9
            (_, B::SpacingMark) => false,                                  // GB9a
            (B::Prepend, _) => false,                                      // GB9b
            (B::Zwj, _) if ep_zwj && is_extended_pictographic(c) => false, // GB11
            (B::RegionalIndicator, B::RegionalIndicator) if ri_run % 2 == 1 => false, // GB12/13
            _ => true,                                                     // GB999
        };
        if boundary {
            return at;
        }
        ep_zwj = ep_ext && cur == B::Zwj;
        ep_ext = is_extended_pictographic(c) || (ep_ext && cur == B::Extend);
        ri_run = if cur == B::RegionalIndicator {
            ri_run + 1
        } else {
            0
        };
        prev = cur;
    }
    s.len()
}

// ---------------------------------------------------------------------------
// The sentencepiece `Precompiled` charsmap (double-array trie walker).
// ---------------------------------------------------------------------------

/// Error for a malformed `Precompiled` charsmap blob. Carries the blob
/// byte offset of the fault (the §8 `Charsmap { at, what }` contract; the
/// crate error enum wraps this).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CharsmapError {
    /// Byte offset of the fault inside the decoded blob.
    pub at: usize,
    /// What is wrong at that offset.
    pub what: &'static str,
}

impl fmt::Display for CharsmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "malformed precompiled charsmap at byte {}: {}",
            self.at, self.what
        )
    }
}

impl std::error::Error for CharsmapError {}

/// A parsed sentencepiece `precompiled_charsmap`: a darts-clone
/// double-array trie mapping UTF-8 byte sequences to replacement strings
/// in a NUL-separated pool.
///
/// Wire layout (after the artifact layer's base64 decode):
/// `[trie_byte_len: u32 LE][trie: trie_byte_len/4 x u32 LE][pool: UTF-8]`.
///
/// Semantics mirror the reference (`spm_precompiled`, re-exported by the
/// pinned HF `tokenizers`) exactly, including its quirks: lookups take the
/// *first* (shortest) prefix hit, a NUL input byte terminates the walk,
/// and [`Charsmap::transform`] tries a whole-grapheme replacement only for
/// clusters under 6 UTF-8 bytes before falling back to per-char lookups.
/// Divergence is confined to hostile blobs: where the reference would
/// panic (out-of-bounds transition, replacement offset outside the pool or
/// off a char boundary), the walker treats the lookup as a miss — for any
/// blob the reference accepts end-to-end, behavior is identical.
pub struct Charsmap {
    units: Vec<u32>,
    pool: String,
}

impl fmt::Debug for Charsmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Charsmap")
            .field("units", &self.units.len())
            .field("pool_bytes", &self.pool.len())
            .finish()
    }
}

fn unit_has_leaf(u: u32) -> bool {
    (u >> 8) & 1 == 1
}

fn unit_value(u: u32) -> u32 {
    u & 0x7FFF_FFFF
}

fn unit_label(u: u32) -> u32 {
    u & (0x8000_0000 | 0xFF)
}

fn unit_offset(u: u32) -> u32 {
    (u >> 10) << ((u & (1 << 9)) >> 6)
}

impl Charsmap {
    /// Parses a decoded `precompiled_charsmap` blob. Structural faults
    /// (truncation, oversized or ragged trie length, non-UTF-8 pool) are
    /// loud errors; the trie graph itself is validated lazily — walks are
    /// bounds-checked, so a corrupt graph degrades to missed matches,
    /// never memory unsafety or a panic.
    pub fn parse(blob: &[u8]) -> Result<Charsmap, CharsmapError> {
        let header: &[u8; 4] = blob.first_chunk().ok_or(CharsmapError {
            at: blob.len(),
            what: "truncated header (need 4 bytes)",
        })?;
        let trie_len = u32::from_le_bytes(*header) as usize;
        if !trie_len.is_multiple_of(4) {
            return Err(CharsmapError {
                at: 0,
                what: "trie byte length is not a multiple of 4",
            });
        }
        let trie_end = 4 + trie_len;
        if trie_end > blob.len() {
            return Err(CharsmapError {
                at: 0,
                what: "trie byte length exceeds the blob",
            });
        }
        let units = blob[4..trie_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().expect("chunks_exact yields 4-byte chunks")))
            .collect();
        let pool = std::str::from_utf8(&blob[trie_end..])
            .map_err(|e| CharsmapError {
                at: trie_end + e.valid_up_to(),
                what: "replacement pool is not valid UTF-8",
            })?
            .to_owned();
        Ok(Charsmap { units, pool })
    }

    /// The reference's `results[0]`: the value of the *shortest* trie
    /// prefix of `key`, walking at most one node per input byte.
    fn first_prefix_value(&self, key: &[u8]) -> Option<u32> {
        let mut pos = 0usize;
        let mut unit = *self.units.first()?;
        pos ^= unit_offset(unit) as usize;
        for &b in key {
            if b == 0 {
                // Reference semantics: NUL terminates the walk (the pool
                // uses NUL as its separator).
                break;
            }
            pos ^= usize::from(b);
            unit = *self.units.get(pos)?;
            if unit_label(unit) != u32::from(b) {
                return None;
            }
            pos ^= unit_offset(unit) as usize;
            if unit_has_leaf(unit) {
                return Some(unit_value(*self.units.get(pos)?));
            }
        }
        None
    }

    /// The NUL-terminated pool string at `index`, refused (miss) when the
    /// index is out of bounds or off a char boundary.
    fn replacement(&self, index: u32) -> Option<&str> {
        let start = index as usize;
        let bytes = self.pool.as_bytes();
        if start > bytes.len() {
            return None;
        }
        let len = bytes[start..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(bytes.len() - start);
        self.pool.get(start..start + len)
    }

    /// The replacement for `chunk`, if its shortest trie-prefix hit has
    /// one — the reference's `Precompiled::transform` primitive. The
    /// normalizer layer calls this per grapheme / per char to build
    /// alignment-tracked transformations.
    pub fn transform_chunk(&self, chunk: &str) -> Option<&str> {
        self.first_prefix_value(chunk.as_bytes())
            .and_then(|v| self.replacement(v))
    }

    /// Applies the charsmap to `s` — the reference's `normalize_string`
    /// loop: per extended grapheme cluster, a whole-cluster replacement is
    /// tried only when the cluster is under 6 UTF-8 bytes; otherwise each
    /// char is looked up individually (unmatched chars pass through).
    pub fn transform(&self, s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for grapheme in graphemes(s) {
            if grapheme.len() < 6
                && let Some(rep) = self.transform_chunk(grapheme)
            {
                out.push_str(rep);
                continue;
            }
            for (i, c) in grapheme.char_indices() {
                let part = &grapheme[i..i + c.len_utf8()];
                match self.transform_chunk(part) {
                    Some(rep) => out.push_str(rep),
                    None => out.push(c),
                }
            }
        }
        out
    }
}

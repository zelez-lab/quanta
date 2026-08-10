//! The alignment-tracked normalized string — the ratified offsets core.
//!
//! Every normalizer transformation runs through [`NormalizedString`],
//! which keeps the original text, the normalized text, and a
//! byte-for-byte alignment between them, mirroring the reference's
//! `NormalizedString` (HF tokenizers 0.21.x,
//! `tokenizer/normalizer.rs`) exactly:
//!
//! - `alignments` holds one `(start, end)` pair per byte of the
//!   NORMALIZED string, giving the byte range of the ORIGINAL text that
//!   byte descends from. A fresh string maps every byte of a char to
//!   that char's own range.
//! - [`NormalizedString::transform_range`] is the single primitive every
//!   transformation compiles down to. It consumes an iterator of
//!   `(char, change)` pairs where `change` is the reference's exact
//!   convention: `0` replaces the next not-yet-consumed char of the
//!   range, `1` is an inserted char (it inherits the alignment of the
//!   previous normalized byte), and `-N` replaces the next char AND
//!   removes the `N` chars after it. `initial_offset` chars at the head
//!   of the range are dropped before the first produced char.
//! - Slicing ([`NormalizedString::slice`]) keeps `original_shift` so a
//!   pre-token split still reports offsets into the WHOLE original
//!   input ([`NormalizedString::offsets_original`]).
//!
//! The derived operations (`filter`, `map`, `prepend`, `append`,
//! `lowercase`, the strip family, NFC/NFD/NFKC/NFKD, `replace`,
//! `split`) are ports of the reference methods of the same names,
//! change-streams included. The NFx streams reproduce the
//! `unicode-normalization-alignments` iterator contract over the
//! crate's own vendored tables: decomposition emits the first char of
//! an expansion with change `0` and the rest with `1`, canonical
//! reordering stable-sorts each maximal non-starter run by combining
//! class with the change values travelling with their chars, and
//! composition adds change values (`k + change - 1` per absorbed pair —
//! the reference fork's `recompose.rs` arithmetic).
//!
//! The pattern layer ([`Matcher`], [`char_matches`], [`literal_matches`],
//! [`regex_matches`]) produces the `(span, is_match)` partitions the
//! reference's `Pattern` trait produces: spans tile the input, and a
//! char-predicate pattern yields each matching char as its OWN match
//! (the reference's `impl Pattern for Fn(char) -> bool`), which is what
//! makes `Isolated` punctuation splitting per-character.

use crate::artifact::{PatternConfig, SplitBehavior};
use crate::error::TokenizerError;
use crate::props::TABLES;
use crate::regex::Regex;
use crate::unicode;

/// A byte span plus whether the pattern matched it. Spans tile the
/// input: contiguous, ordered, covering every byte.
pub type MatchSpans = Vec<((usize, usize), bool)>;

// ── The pattern layer ───────────────────────────────────────────────────

/// A compiled `pattern` field: a literal string or a regex over the
/// crate engine. Compiled once at load (loads-means-runs), matched many
/// times.
#[derive(Debug, Clone)]
pub enum Matcher {
    /// Plain substring find (the reference escapes the string into its
    /// regex engine; the match set is identical).
    Literal(String),
    Regex(Regex),
}

impl Matcher {
    /// Compiles an artifact `pattern` config. Regex patterns go through
    /// the crate's closed-construct engine; errors surface as the §8
    /// `RegexConstruct` row.
    pub fn compile(pattern: &PatternConfig) -> Result<Matcher, TokenizerError> {
        match pattern {
            PatternConfig::String(s) => Ok(Matcher::Literal(s.clone())),
            PatternConfig::Regex(r) => Ok(Matcher::Regex(
                Regex::parse(r, &TABLES).map_err(regex_error)?,
            )),
        }
    }

    /// The `(span, is_match)` partition of `text` under this pattern.
    pub fn spans(&self, text: &str) -> Result<MatchSpans, TokenizerError> {
        match self {
            Matcher::Literal(s) => Ok(literal_matches(text, s)),
            Matcher::Regex(re) => regex_matches(text, re),
        }
    }
}

/// Maps an engine error onto the crate taxonomy: every row (parse,
/// unsupported construct, budget trip) carries the pattern, per §8.
pub fn regex_error(e: crate::regex::RegexError) -> TokenizerError {
    use crate::regex::RegexError as R;
    match e {
        R::Parse { pattern, at, what } => TokenizerError::RegexConstruct {
            pattern,
            construct: format!("malformed pattern ({what} at byte {at})"),
        },
        R::UnsupportedConstruct {
            pattern, construct, ..
        } => TokenizerError::RegexConstruct { pattern, construct },
        R::Budget {
            pattern,
            steps,
            input_chars,
        } => TokenizerError::RegexConstruct {
            pattern,
            construct: format!(
                "backtracking budget of {steps} steps exceeded on a {input_chars}-char input"
            ),
        },
    }
}

/// Char-predicate matches: every char satisfying `pred` is its own
/// one-char match (reference `impl Pattern for Fn(char) -> bool`).
pub fn char_matches(text: &str, pred: impl Fn(char) -> bool) -> MatchSpans {
    if text.is_empty() {
        return vec![((0, 0), false)];
    }
    let mut spans = Vec::new();
    let mut gap_start = 0;
    for (b, c) in text.char_indices() {
        if pred(c) {
            if gap_start < b {
                spans.push(((gap_start, b), false));
            }
            spans.push(((b, b + c.len_utf8()), true));
            gap_start = b + c.len_utf8();
        }
    }
    if gap_start < text.len() {
        spans.push(((gap_start, text.len()), false));
    }
    spans
}

/// Literal substring matches, leftmost and non-overlapping. An empty
/// needle matches nothing (the reference's empty-pattern posture).
pub fn literal_matches(text: &str, needle: &str) -> MatchSpans {
    if text.is_empty() {
        return vec![((0, 0), false)];
    }
    if needle.is_empty() {
        return vec![((0, text.len()), false)];
    }
    let mut spans = Vec::new();
    let mut prev = 0;
    let mut at = 0;
    while let Some(found) = text[at..].find(needle) {
        let start = at + found;
        if prev < start {
            spans.push(((prev, start), false));
        }
        spans.push(((start, start + needle.len()), true));
        prev = start + needle.len();
        at = prev;
    }
    if prev < text.len() {
        spans.push(((prev, text.len()), false));
    }
    spans
}

/// Regex matches through the crate engine. Empty matches contribute no
/// span of their own (they cannot arise from corpus patterns; dropping
/// them keeps the tiling invariant).
pub fn regex_matches(text: &str, re: &Regex) -> Result<MatchSpans, TokenizerError> {
    if text.is_empty() {
        return Ok(vec![((0, 0), false)]);
    }
    let mut spans = Vec::new();
    let mut prev = 0;
    for m in re.find_iter(text) {
        let (start, end) = m.map_err(regex_error)?;
        if start > end || start < prev || end == start {
            continue;
        }
        if prev < start {
            spans.push(((prev, start), false));
        }
        spans.push(((start, end), true));
        prev = end;
    }
    if prev < text.len() {
        spans.push(((prev, text.len()), false));
    }
    Ok(spans)
}

/// Swaps match and gap — the reference's `Invert` pattern wrapper,
/// serving `Split { invert: true }` and the `Whitespace` pre-tokenizer.
pub fn invert(mut spans: MatchSpans) -> MatchSpans {
    for (_, is_match) in &mut spans {
        *is_match = !*is_match;
    }
    spans
}

// ── The alignment-tracked string ────────────────────────────────────────

/// The original text, the normalized text, and the byte alignment
/// between them (module docs for the exact semantics).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NormalizedString {
    original: String,
    normalized: String,
    /// One `(start, end)` original-byte range per NORMALIZED byte.
    alignments: Vec<(usize, usize)>,
    /// When this string is a slice of a bigger one: how many original
    /// bytes precede it, so offsets stay absolute.
    original_shift: usize,
}

impl From<&str> for NormalizedString {
    fn from(s: &str) -> Self {
        let alignments = s
            .char_indices()
            .flat_map(|(b, c)| std::iter::repeat_n((b, b + c.len_utf8()), c.len_utf8()))
            .collect();
        NormalizedString {
            original: s.to_string(),
            normalized: s.to_string(),
            alignments,
            original_shift: 0,
        }
    }
}

impl NormalizedString {
    /// The normalized text.
    pub fn get(&self) -> &str {
        &self.normalized
    }

    /// The original text (of this slice).
    pub fn get_original(&self) -> &str {
        &self.original
    }

    /// Normalized length in bytes.
    pub fn len(&self) -> usize {
        self.normalized.len()
    }

    /// Original length in bytes.
    pub fn len_original(&self) -> usize {
        self.original.len()
    }

    /// Whether the NORMALIZED text is empty (empty splits are dropped
    /// by the pre-tokenization layer on this predicate).
    pub fn is_empty(&self) -> bool {
        self.normalized.is_empty()
    }

    /// The absolute byte range this string covers in the whole original
    /// input (identity for a fresh string, shifted for slices).
    pub fn offsets_original(&self) -> (usize, usize) {
        (
            self.original_shift,
            self.original_shift + self.len_original(),
        )
    }

    /// Converts a NORMALIZED byte range to the ORIGINAL byte range it
    /// descends from: the first covered alignment's start to the last's
    /// end (reference `convert_offsets`, normalized arm). Offsets are
    /// relative to this slice's own original text (add
    /// [`Self::offsets_original`]`.0` for absolute ones).
    pub fn convert_offsets(&self, range: std::ops::Range<usize>) -> Option<std::ops::Range<usize>> {
        if range.start == range.end {
            return Some(range);
        }
        if range.start > range.end {
            return None;
        }
        if self.normalized.is_empty() && range == (0..0) {
            return Some(0..self.len_original());
        }
        let covered = self.alignments.get(range)?;
        if covered.is_empty() {
            None
        } else {
            Some(covered[0].0..covered[covered.len() - 1].1)
        }
    }

    /// Converts an ORIGINAL byte range to the NORMALIZED range derived
    /// from it (reference `convert_offsets`, original arm — including
    /// its zero-width-alignment skip at the start edge).
    fn convert_offsets_from_original(
        &self,
        target: std::ops::Range<usize>,
    ) -> Option<std::ops::Range<usize>> {
        if target.start == target.end {
            return Some(target);
        }
        if self.original.is_empty() && target == (0..0) {
            return Some(0..self.len());
        }
        let (mut start, mut end) = (None, None);
        for (i, alignment) in self.alignments.iter().enumerate() {
            if target.end < alignment.1 {
                break;
            }
            if start.is_none() && target.start <= alignment.0 && alignment.0 != alignment.1 {
                start = Some(i);
            }
            if target.end >= alignment.1 {
                end = Some(i + 1);
            }
        }
        match (start, end) {
            (Some(s), None) => Some(s..s),
            (None, Some(e)) => Some(e..e),
            (Some(s), Some(e)) => Some(s..e),
            (None, None) => None,
        }
    }

    /// The single transformation primitive; see the module docs for the
    /// change convention. `n_range` is a NORMALIZED byte range on char
    /// boundaries.
    pub fn transform_range<I>(
        &mut self,
        n_range: std::ops::Range<usize>,
        dest: I,
        initial_offset: usize,
    ) where
        I: IntoIterator<Item = (char, isize)>,
    {
        let mut replaced = self.normalized[n_range.clone()]
            .chars()
            .collect::<Vec<_>>()
            .into_iter();
        let initial_removed: usize = (&mut replaced)
            .take(initial_offset)
            .map(char::len_utf8)
            .sum();

        let mut offset = initial_removed + n_range.start;
        let mut alignments = Vec::with_capacity(n_range.len());
        let mut produced = String::new();
        for (c, changes) in dest {
            let align = if changes.is_positive() {
                if offset < 1 {
                    (0, 0)
                } else {
                    self.alignments[offset - 1]
                }
            } else {
                self.alignments[offset]
            };
            let replaced_char = if changes.is_positive() {
                None
            } else {
                replaced.next()
            };
            let replaced_char_size = replaced_char.map_or(0, char::len_utf8);
            let removed_bytes: usize = if changes.is_negative() {
                (&mut replaced)
                    .take(changes.unsigned_abs())
                    .map(char::len_utf8)
                    .sum()
            } else {
                0
            };
            offset += replaced_char_size + removed_bytes;
            alignments.extend(std::iter::repeat_n(align, c.len_utf8()));
            produced.push(c);
        }
        self.alignments.splice(n_range.clone(), alignments);
        let mut normalized = String::with_capacity(
            self.normalized.len() - (n_range.end - n_range.start) + produced.len(),
        );
        normalized.push_str(&self.normalized[..n_range.start]);
        normalized.push_str(&produced);
        normalized.push_str(&self.normalized[n_range.end..]);
        self.normalized = normalized;
    }

    /// [`Self::transform_range`] over the range derived from the FULL
    /// original text — the reference's `transform` routes through the
    /// original referential, so content with no original ancestry at
    /// the edges stays out of range.
    pub fn transform<I>(&mut self, dest: I, initial_offset: usize)
    where
        I: IntoIterator<Item = (char, isize)>,
    {
        let Some(range) = self.convert_offsets_from_original(0..self.len_original()) else {
            return;
        };
        self.transform_range(range, dest, initial_offset);
    }

    /// Keeps only the chars `keep` accepts. Removed runs fold into the
    /// preceding kept char as `-N`; a removed head becomes
    /// `initial_offset` (reference `filter`).
    pub fn filter(&mut self, keep: impl Fn(char) -> bool) -> &mut Self {
        let mut removed: isize = 0;
        let mut removed_start = 0usize;
        let mut transforms: Vec<(char, isize)> = Vec::with_capacity(self.normalized.len());
        let mut last_kept = None;
        for c in self.normalized.chars() {
            if keep(c) {
                match last_kept {
                    Some(prev) => transforms.push((prev, -removed)),
                    None => removed_start = usize::try_from(removed).unwrap_or(0),
                }
                last_kept = Some(c);
                removed = 0;
            } else {
                removed += 1;
            }
        }
        if let Some(prev) = last_kept {
            transforms.push((prev, -removed));
        }
        self.transform(transforms, removed_start);
        self
    }

    /// Replaces each char through `map` (one-to-one, reference `map`).
    pub fn map_chars(&mut self, map: impl Fn(char) -> char) -> &mut Self {
        let transforms: Vec<(char, isize)> = self.normalized.chars().map(|c| (map(c), 0)).collect();
        self.transform(transforms, 0);
        self
    }

    /// Prepends `s`. The whole prepended text shares the first char's
    /// alignment (reference `prepend`: first prepended char replaces
    /// the head char, everything after — head char included — rides as
    /// insertions). A no-op on an empty string.
    pub fn prepend(&mut self, s: &str) -> &mut Self {
        if let Some(head) = self.normalized.chars().next() {
            let transformations = s
                .chars()
                .enumerate()
                .map(|(i, c)| (c, isize::from(i != 0)))
                .chain(std::iter::once((head, 1)))
                .collect::<Vec<_>>();
            self.transform_range(0..head.len_utf8(), transformations, 0);
        }
        self
    }

    /// Appends `s`; the appended text shares the last char's alignment.
    pub fn append(&mut self, s: &str) -> &mut Self {
        if let Some((b, last)) = self.normalized.char_indices().last() {
            let transformations = std::iter::once((last, 0))
                .chain(s.chars().map(|c| (c, 1)))
                .collect::<Vec<_>>();
            let end = self.normalized.len();
            self.transform_range(b..end, transformations, 0);
        }
        self
    }

    /// Full Unicode lowercase; expansions (e.g. `İ`) ride as insertions
    /// on the source char's alignment (reference `lowercase`, over the
    /// vendored mapping tables).
    pub fn lowercase(&mut self) -> &mut Self {
        let mut new_chars: Vec<(char, isize)> = Vec::with_capacity(self.normalized.len());
        for c in self.normalized.chars() {
            for (i, l) in unicode::lowercase(c).enumerate() {
                new_chars.push((l, isize::from(i > 0)));
            }
        }
        self.transform(new_chars, 0);
        self
    }

    /// Strips leading whitespace.
    pub fn lstrip(&mut self) -> &mut Self {
        self.lrstrip(true, false)
    }

    /// Strips trailing whitespace.
    pub fn rstrip(&mut self) -> &mut Self {
        self.lrstrip(false, true)
    }

    /// Strips both ends.
    pub fn strip(&mut self) -> &mut Self {
        self.lrstrip(true, true)
    }

    fn lrstrip(&mut self, left: bool, right: bool) -> &mut Self {
        let leading = if left {
            self.normalized
                .chars()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };
        let trailing = if right {
            self.normalized
                .chars()
                .rev()
                .take_while(|c| c.is_whitespace())
                .count()
        } else {
            0
        };
        if leading == 0 && trailing == 0 {
            return self;
        }
        let count = self.normalized.chars().count();
        if leading + trailing >= count {
            // Everything is whitespace: clear the string.
            let len = self.len();
            let full = 0..len;
            self.transform_range(full, std::iter::empty(), count);
            return self;
        }
        let last_kept = count - trailing - 1;
        let transformation: Vec<(char, isize)> = self
            .normalized
            .chars()
            .enumerate()
            .filter_map(|(i, c)| {
                if i < leading || i >= count - trailing {
                    None
                } else if i == last_kept {
                    Some((c, -isize::try_from(trailing).unwrap_or(isize::MAX)))
                } else {
                    Some((c, 0))
                }
            })
            .collect();
        self.transform(transformation, leading);
        self
    }

    // ── Unicode normalization forms ─────────────────────────────────────

    /// NFD over the vendored tables, alignment-tracked.
    pub fn nfd(&mut self) -> &mut Self {
        self.apply_nfx(false, false)
    }

    /// NFKD.
    pub fn nfkd(&mut self) -> &mut Self {
        self.apply_nfx(true, false)
    }

    /// NFC.
    pub fn nfc(&mut self) -> &mut Self {
        self.apply_nfx(false, true)
    }

    /// NFKC.
    pub fn nfkc(&mut self) -> &mut Self {
        self.apply_nfx(true, true)
    }

    fn apply_nfx(&mut self, compat: bool, compose: bool) -> &mut Self {
        // All four forms are the identity on pure-ASCII text, and the
        // identity stream would be a no-op transform.
        if self.normalized.is_ascii() {
            return self;
        }
        let mut stream = decomposed_stream(&self.normalized, compat);
        if compose {
            stream = recomposed_stream(stream);
        }
        self.transform(stream, 0);
        self
    }

    // ── Replace ─────────────────────────────────────────────────────────

    /// Replaces every match span with `content`. Content chars ride as
    /// insertions and all replaced chars count as removed — the
    /// reference's `replace` change-stream, so the replacement inherits
    /// the alignment of the LAST byte of the replaced span.
    pub fn replace_spans(&mut self, spans: &MatchSpans, content: &str) {
        let mut offset: isize = 0;
        for &((start, end), is_match) in spans {
            if !is_match {
                continue;
            }
            let range = {
                let s = usize::try_from(start as isize + offset).unwrap_or(0);
                let e = usize::try_from(end as isize + offset).unwrap_or(0);
                s..e
            };
            let removed_chars = self.normalized[range.clone()].chars().count();
            let mut new_len = 0usize;
            let dest: Vec<(char, isize)> = content
                .chars()
                .map(|c| {
                    new_len += c.len_utf8();
                    (c, 1)
                })
                .collect();
            self.transform_range(range, dest, removed_chars);
            offset += new_len as isize - (end - start) as isize;
        }
    }

    /// [`Self::replace_spans`] through a [`Matcher`].
    pub fn replace(&mut self, matcher: &Matcher, content: &str) -> Result<(), TokenizerError> {
        let spans = matcher.spans(&self.normalized)?;
        self.replace_spans(&spans, content);
        Ok(())
    }

    // ── Split / slice ───────────────────────────────────────────────────

    /// Splits on a `(span, is_match)` partition under the given
    /// delimiter behavior (reference `split`: `Removed` drops matches,
    /// `Isolated` keeps them standalone, `MergedWithPrevious`/`Next`
    /// fuse each match run into its neighbor, `Contiguous` fuses
    /// same-kind runs). Empty output slices are kept here; the
    /// pre-tokenization layer drops them.
    pub fn split(&self, matches: MatchSpans, behavior: SplitBehavior) -> Vec<NormalizedString> {
        use SplitBehavior as B;
        let splits: Vec<((usize, usize), bool)> = match behavior {
            B::Isolated => matches
                .into_iter()
                .map(|(offsets, _)| (offsets, false))
                .collect(),
            B::Removed => matches,
            B::Contiguous => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(Vec::new(), |mut acc, (offsets, is_match)| {
                        if is_match == previous_match {
                            if let Some(((_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            B::MergedWithPrevious => {
                let mut previous_match = false;
                matches
                    .into_iter()
                    .fold(Vec::new(), |mut acc, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some(((_, end), _)) = acc.last_mut() {
                                *end = offsets.1;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    })
            }
            B::MergedWithNext => {
                let mut previous_match = false;
                let mut acc = matches.into_iter().rev().fold(
                    Vec::new(),
                    |mut acc: Vec<((usize, usize), bool)>, (offsets, is_match)| {
                        if is_match && !previous_match {
                            if let Some(((start, _), _)) = acc.last_mut() {
                                *start = offsets.0;
                            } else {
                                acc.push((offsets, false));
                            }
                        } else {
                            acc.push((offsets, false));
                        }
                        previous_match = is_match;
                        acc
                    },
                );
                acc.reverse();
                acc
            }
        };
        splits
            .into_iter()
            .filter_map(
                |((start, end), remove)| {
                    if remove { None } else { self.slice(start..end) }
                },
            )
            .collect()
    }

    /// A sub-string of this one over a NORMALIZED byte range, with the
    /// alignments rebased and `original_shift` accumulated so absolute
    /// offsets survive arbitrary nesting.
    pub fn slice(&self, range: std::ops::Range<usize>) -> Option<NormalizedString> {
        if !self.normalized.is_char_boundary(range.start)
            || !self.normalized.is_char_boundary(range.end)
        {
            return None;
        }
        let original_range = self.convert_offsets(range.clone())?;
        let n_shift = original_range.start;
        Some(NormalizedString {
            original: self.original.get(original_range.clone())?.to_string(),
            normalized: self.normalized[range.clone()].to_string(),
            alignments: self.alignments[range]
                .iter()
                .map(|(start, end)| (start - n_shift, end - n_shift))
                .collect(),
            original_shift: self.original_shift + n_shift,
        })
    }
}

// ── NFx change streams ──────────────────────────────────────────────────

/// The full decomposition of `s` as a `(char, change)` stream: per
/// source char, the first decomposed char carries `0` and the rest `1`;
/// each maximal non-starter run is then stable-sorted by combining
/// class with the changes travelling along (the
/// `unicode-normalization-alignments` decomposition iterator contract).
fn decomposed_stream(s: &str, compat: bool) -> Vec<(char, isize)> {
    let mut buf: Vec<(u8, char, isize)> = Vec::with_capacity(s.len());
    let mut single = String::with_capacity(4);
    for c in s.chars() {
        single.clear();
        single.push(c);
        let decomposed = if compat {
            unicode::nfkd(&single)
        } else {
            unicode::nfd(&single)
        };
        for (i, d) in decomposed.chars().enumerate() {
            buf.push((unicode::combining_class(d), d, isize::from(i > 0)));
        }
    }
    let mut i = 0;
    while i < buf.len() {
        if buf[i].0 == 0 {
            i += 1;
            continue;
        }
        let run_start = i;
        while i < buf.len() && buf[i].0 != 0 {
            i += 1;
        }
        buf[run_start..i].sort_by_key(|&(ccc, _, _)| ccc);
    }
    buf.into_iter().map(|(_, c, change)| (c, change)).collect()
}

/// The primary composite of `a` + `b`, if any, derived through the
/// table-driven NFC of the two-char sequence (equivalent to the pair
/// table for every input the composition machine can present: machine
/// composees are starters or stepwise composites, never re-decomposing
/// sequences).
fn compose_pair(a: char, b: char) -> Option<char> {
    let mut s = String::with_capacity(8);
    s.push(a);
    s.push(b);
    let composed = unicode::nfc(&s);
    let mut chars = composed.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Some(c),
        _ => None,
    }
}

/// Canonical composition over a decomposed `(char, change)` stream —
/// a port of the reference fork's `recompose.rs` state machine, change
/// arithmetic included (`k + change - 1` per absorbed pair).
fn recomposed_stream(decomposed: Vec<(char, isize)>) -> Vec<(char, isize)> {
    let mut out: Vec<(char, isize)> = Vec::with_capacity(decomposed.len());
    let mut buffer: Vec<(char, isize)> = Vec::new();
    let mut composee: Option<(char, isize)> = None;
    let mut last_ccc: Option<u8> = None;
    for (ch, change) in decomposed {
        let ch_class = unicode::combining_class(ch);
        let Some(k) = composee else {
            if ch_class != 0 {
                out.push((ch, change));
            } else {
                composee = Some((ch, change));
            }
            continue;
        };
        match last_ccc {
            None => match compose_pair(k.0, ch) {
                Some(r) => composee = Some((r, k.1 + change - 1)),
                None => {
                    if ch_class == 0 {
                        composee = Some((ch, change));
                        out.push(k);
                    } else {
                        buffer.push((ch, change));
                        last_ccc = Some(ch_class);
                    }
                }
            },
            Some(l_class) => {
                if l_class >= ch_class {
                    // Blocked from the composee.
                    if ch_class == 0 {
                        out.push(k);
                        out.append(&mut buffer);
                        composee = Some((ch, change));
                        last_ccc = None;
                    } else {
                        buffer.push((ch, change));
                        last_ccc = Some(ch_class);
                    }
                } else {
                    match compose_pair(k.0, ch) {
                        Some(r) => composee = Some((r, k.1 + change - 1)),
                        None => {
                            buffer.push((ch, change));
                            last_ccc = Some(ch_class);
                        }
                    }
                }
            }
        }
    }
    if let Some(k) = composee {
        out.push(k);
    }
    out.append(&mut buffer);
    out
}

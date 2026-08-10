//! Execution of the normalizer configs over the alignment-tracked
//! string — every §5 normalizer row, ported from the pinned reference
//! (HF tokenizers 0.21.x, `normalizers/*.rs`):
//!
//! - `NFC`/`NFD`/`NFKC`/`NFKD` — [`NormalizedString`]'s alignment-tracked
//!   forms over the vendored tables.
//! - `BertNormalizer` — the four flags in the reference's order:
//!   clean_text (drop NUL/U+FFFD/controls, fold whitespace to space),
//!   handle_chinese_chars (space-pad the CJK ranges, the reference's
//!   exact range list including its `0x2B920` quirk), strip_accents
//!   (NFD then drop `Mn`; a `null` flag FOLLOWS `lowercase` — the
//!   reference's documented coupling), lowercase.
//! - `Lowercase` / `Strip` / `StripAccents` / `Prepend` / `Replace` —
//!   direct ports. `StripAccents` standalone drops ALL marks (`M*`)
//!   with NO decomposition — `unicode-normalization-alignments`'
//!   `is_combining_mark` is General_Category=Mark; the "NFD + drop Mn"
//!   spelling lives only inside `BertNormalizer`.
//! - `Precompiled` — the sentencepiece charsmap over
//!   [`unicode::graphemes`] with the reference's exact change-stream
//!   helper (whole clusters under 6 UTF-8 bytes first, then per-char).
//! - `Nmt` — the fixed codepoint filter + whitespace-fold lists.
//! - `ByteLevel` (normalizer form) — the byte-to-unicode transform of
//!   the pre-tokenizer, without prefix or regex.
//! - `Sequence` — ordered composition, arbitrary nesting.

use crate::artifact::NormalizerConfig;
use crate::error::TokenizerError;
use crate::normalized::{Matcher, NormalizedString};
use crate::pretokenize::byte_to_char;
use crate::unicode::{self, Charsmap};

/// A compiled, executable normalizer. Built once at load from the
/// artifact config ([`Normalizer::compile`]) — regexes parsed, the
/// charsmap blob decoded and validated — so a tokenizer that loads,
/// runs (§7).
#[derive(Debug)]
pub enum Normalizer {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        lowercase: bool,
    },
    Lowercase,
    Strip {
        strip_left: bool,
        strip_right: bool,
    },
    StripAccents,
    Prepend(String),
    Replace {
        matcher: Matcher,
        content: String,
    },
    Precompiled(Charsmap),
    Nmt,
    ByteLevel,
    Sequence(Vec<Normalizer>),
}

impl Normalizer {
    /// Compiles an artifact normalizer config into its executor.
    pub fn compile(config: &NormalizerConfig) -> Result<Normalizer, TokenizerError> {
        Ok(match config {
            NormalizerConfig::Nfc => Normalizer::Nfc,
            NormalizerConfig::Nfd => Normalizer::Nfd,
            NormalizerConfig::Nfkc => Normalizer::Nfkc,
            NormalizerConfig::Nfkd => Normalizer::Nfkd,
            NormalizerConfig::Bert {
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            } => Normalizer::Bert {
                clean_text: *clean_text,
                handle_chinese_chars: *handle_chinese_chars,
                strip_accents: *strip_accents,
                lowercase: *lowercase,
            },
            NormalizerConfig::Lowercase => Normalizer::Lowercase,
            NormalizerConfig::Strip {
                strip_left,
                strip_right,
            } => Normalizer::Strip {
                strip_left: *strip_left,
                strip_right: *strip_right,
            },
            NormalizerConfig::StripAccents => Normalizer::StripAccents,
            NormalizerConfig::Prepend { prepend } => Normalizer::Prepend(prepend.clone()),
            NormalizerConfig::Replace { pattern, content } => Normalizer::Replace {
                matcher: Matcher::compile(pattern)?,
                content: content.clone(),
            },
            NormalizerConfig::Precompiled { charsmap } => {
                Normalizer::Precompiled(Charsmap::parse(charsmap).map_err(|e| {
                    TokenizerError::Charsmap {
                        at: e.at,
                        what: e.what.to_string(),
                    }
                })?)
            }
            NormalizerConfig::Nmt => Normalizer::Nmt,
            NormalizerConfig::ByteLevel => Normalizer::ByteLevel,
            NormalizerConfig::Sequence(inner) => Normalizer::Sequence(
                inner
                    .iter()
                    .map(Normalizer::compile)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// Runs this normalizer over `n`, maintaining alignment.
    pub fn apply(&self, n: &mut NormalizedString) -> Result<(), TokenizerError> {
        match self {
            Normalizer::Nfc => {
                n.nfc();
            }
            Normalizer::Nfd => {
                n.nfd();
            }
            Normalizer::Nfkc => {
                n.nfkc();
            }
            Normalizer::Nfkd => {
                n.nfkd();
            }
            Normalizer::Bert {
                clean_text,
                handle_chinese_chars,
                strip_accents,
                lowercase,
            } => {
                if *clean_text {
                    bert_clean_text(n);
                }
                if *handle_chinese_chars {
                    bert_handle_chinese_chars(n);
                }
                // The reference's documented default: a null strip_accents
                // follows lowercase.
                if strip_accents.unwrap_or(*lowercase) {
                    n.nfd().filter(|c| !unicode::is_nonspacing_mark(c));
                }
                if *lowercase {
                    n.lowercase();
                }
            }
            Normalizer::Lowercase => {
                n.lowercase();
            }
            Normalizer::Strip {
                strip_left,
                strip_right,
            } => match (strip_left, strip_right) {
                (true, true) => {
                    n.strip();
                }
                (true, false) => {
                    n.lstrip();
                }
                (false, true) => {
                    n.rstrip();
                }
                (false, false) => {}
            },
            Normalizer::StripAccents => {
                // Reference `StripAccents`: drop General_Category=Mark,
                // no decomposition step.
                n.filter(|c| !unicode::is_mark(c));
            }
            Normalizer::Prepend(prepend) => {
                if !n.is_empty() {
                    n.prepend(prepend);
                }
            }
            Normalizer::Replace { matcher, content } => {
                n.replace(matcher, content)?;
            }
            Normalizer::Precompiled(charsmap) => {
                precompiled_normalize(charsmap, n);
            }
            Normalizer::Nmt => {
                nmt(n);
            }
            Normalizer::ByteLevel => {
                byte_level_normalize(n);
            }
            Normalizer::Sequence(inner) => {
                for normalizer in inner {
                    normalizer.apply(n)?;
                }
            }
        }
        Ok(())
    }
}

// ── BertNormalizer helpers (reference `normalizers/bert.rs`) ────────────

/// Bert's whitespace test: `\t` `\n` `\r` count as whitespace.
fn bert_is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

/// Bert's control test: `\t` `\n` `\r` are NOT control; everything in
/// the `C*` classes (Cc, Cf, Cn, Co) is.
fn bert_is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && unicode::category_class(c) == unicode::CategoryClass::Other
}

/// The reference's CJK range list — verbatim, including `0x2B920` where
/// Unicode's Extension E starts at `0x2B820` (a reference quirk kept
/// for parity).
fn is_chinese_char(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn bert_clean_text(n: &mut NormalizedString) {
    n.filter(|c| !(c as u32 == 0 || c as u32 == 0xFFFD || bert_is_control(c)))
        .map_chars(|c| if bert_is_whitespace(c) { ' ' } else { c });
}

fn bert_handle_chinese_chars(n: &mut NormalizedString) {
    let mut new_chars: Vec<(char, isize)> = Vec::new();
    for c in n.get().chars() {
        if is_chinese_char(c) {
            new_chars.extend([(' ', 0), (c, 1), (' ', 1)]);
        } else {
            new_chars.push((c, 0));
        }
    }
    n.transform(new_chars, 0);
}

// ── Nmt (reference `normalizers/unicode.rs`) ────────────────────────────

fn nmt(n: &mut NormalizedString) {
    n.filter(|c| {
        !matches!(
            c as u32,
            0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F
        )
    })
    .map_chars(|c| match c as u32 {
        0x0009
        | 0x000A
        | 0x000C
        | 0x000D
        | 0x1680
        | 0x200B..=0x200F
        | 0x2028
        | 0x2029
        | 0x2581
        | 0xFEFF
        | 0xFFFD => ' ',
        _ => c,
    });
}

// ── ByteLevel normalizer form (reference `normalizers/byte_level.rs`) ───

/// The GPT-2 byte-to-unicode transform as a normalizer: every UTF-8
/// byte becomes its printable stand-in char; continuation bytes ride as
/// insertions on their char's alignment.
fn byte_level_normalize(n: &mut NormalizedString) {
    if n.is_empty() {
        return;
    }
    let s = n.get();
    let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
    for c in s.chars() {
        let mut utf8 = [0u8; 4];
        for (i, b) in c.encode_utf8(&mut utf8).bytes().enumerate() {
            transformations.push((byte_to_char(b), isize::from(i > 0)));
        }
    }
    n.transform(transformations, 0);
}

// ── Precompiled (reference `normalizers/precompiled.rs`) ────────────────

/// The reference's change-stream builder for one charsmap replacement:
/// new chars enter as replacements (`0`), a growth flips the last
/// `diff` entries to insertions (`1`), a shrink folds the deficit into
/// the last entry — quirks preserved verbatim (a deletion with nothing
/// emitted yet folds into the PREVIOUS grapheme's last entry, or is
/// dropped at the very start).
fn charsmap_replace(transformations: &mut Vec<(char, isize)>, old_part: &str, new_part: &str) {
    let old_count = old_part.chars().count() as isize;
    let new_count = new_part.chars().count() as isize;
    let diff = new_count - old_count;

    transformations.extend(new_part.chars().map(|c| (c, 0)));

    match diff.cmp(&0) {
        std::cmp::Ordering::Greater => {
            transformations
                .iter_mut()
                .rev()
                .take(diff.unsigned_abs())
                .for_each(|(_, change)| *change = 1);
        }
        std::cmp::Ordering::Less => {
            if let Some((_, change)) = transformations.last_mut() {
                *change += diff;
            }
        }
        std::cmp::Ordering::Equal => {}
    }
}

/// The reference's `Precompiled::normalize` loop: per extended grapheme
/// cluster, a whole-cluster replacement is tried only under 6 UTF-8
/// bytes; otherwise each char is looked up individually and misses pass
/// through.
fn precompiled_normalize(charsmap: &Charsmap, n: &mut NormalizedString) {
    let mut transformations: Vec<(char, isize)> = Vec::with_capacity(n.get().len());
    let mut modified = false;
    for grapheme in unicode::graphemes(n.get()) {
        if grapheme.len() < 6
            && let Some(norm) = charsmap.transform_chunk(grapheme)
        {
            modified = true;
            charsmap_replace(&mut transformations, grapheme, norm);
            continue;
        }
        for (i, c) in grapheme.char_indices() {
            let part = &grapheme[i..i + c.len_utf8()];
            if let Some(norm) = charsmap.transform_chunk(part) {
                modified = true;
                charsmap_replace(&mut transformations, part, norm);
            } else {
                transformations.push((c, 0));
            }
        }
    }
    if modified {
        n.transform(transformations, 0);
    }
}

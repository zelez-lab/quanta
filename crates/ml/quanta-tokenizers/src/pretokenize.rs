//! Execution of the pre-tokenizer configs: offset-carrying splits.
//!
//! [`PreTokenizedString`] is the port of the reference's struct of the
//! same name (HF tokenizers 0.21.x, `tokenizer/pre_tokenizer.rs`): an
//! ordered run of [`Split`]s, each an alignment-tracked
//! [`NormalizedString`] slice of the input plus, once the model has
//! run, its tokens. Splitting only touches splits without tokens (the
//! added-token layer attaches tokens BEFORE pre-tokenization and those
//! splits pass through untouched), empty splits are dropped, and
//! [`PreTokenizedString::into_encoding`] converts every token's offsets from
//! its split's normalized space back to ORIGINAL-text byte offsets
//! through the alignment layer, assigning word ids by split index.
//!
//! Every §5 pre-tokenizer row executes here, ported one-to-one:
//!
//! - `ByteLevel` — optional space prefix (`prepend`), the GPT-2 split
//!   pattern ([`GPT2_SPLIT_PATTERN`]) under `Isolated`, then the
//!   256-byte↔printable-char bijection ([`byte_to_char`] /
//!   [`char_to_byte`] — a hardcoded table, not UCD).
//! - `Whitespace` (`\w+|[^\w\s]+` inverted, `Removed`),
//!   `WhitespaceSplit`, `Punctuation` (ASCII punctuation OR `\p{P}`,
//!   per-char matches), `Digits` (`Isolated` per digit or `Contiguous`
//!   runs), `CharDelimiterSplit`, `BertPreTokenizer` (whitespace
//!   `Removed`, then punctuation `Isolated`).
//! - `Metaspace` — replace spaces by the replacement char, prepend per
//!   scheme (`First` keys on the split's absolute original offset being
//!   0), then `MergedWithNext` on the replacement when `split`.
//! - `Split` — literal or engine regex, all five behaviors, `invert`.
//! - `UnicodeScripts` — sentencepiece script runs (Hiragana/Katakana
//!   and U+30FC fold into Han, space and unassigned are "any", Common /
//!   Inherited break runs exactly as the reference's table does).
//! - `FixedLength` — fixed char-count chunks.
//! - `Sequence` — ordered composition.

use crate::artifact::{PreTokenizerConfig, PrependScheme, SplitBehavior};
use crate::encoding::Encoding;
use crate::error::TokenizerError;
use crate::model::ModelToken;
use crate::normalized::{self, Matcher, NormalizedString, char_matches, regex_matches};
use crate::props::TABLES;
use crate::regex::Regex;
use crate::unicode::{self, Script};
use std::sync::OnceLock;

// ── The GPT-2 byte-level bijection ──────────────────────────────────────

/// The GPT-2 split pattern (openai/gpt-2 `encoder.py`), run on the
/// crate engine — its construct inventory is the §6 closed set.
pub const GPT2_SPLIT_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Builds the GPT-2 byte↔char tables: the 188 printable bytes map to
/// themselves, the rest to `0x100 + n` in byte order — the reference's
/// `bytes_char()` verbatim.
fn byte_char_tables() -> ([char; 256], [Option<u8>; 0x200]) {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut to_char = ['\0'; 256];
    let mut to_byte = [None; 0x200];
    let mut n = 0u32;
    for b in 0u8..=255 {
        let c = if printable(b) {
            char::from(b)
        } else {
            let c = char::from_u32(0x100 + n).expect("BMP scalar");
            n += 1;
            c
        };
        to_char[b as usize] = c;
        to_byte[c as u32 as usize] = Some(b);
    }
    (to_char, to_byte)
}

fn tables() -> &'static ([char; 256], [Option<u8>; 0x200]) {
    static TABLES_CELL: OnceLock<([char; 256], [Option<u8>; 0x200])> = OnceLock::new();
    TABLES_CELL.get_or_init(byte_char_tables)
}

/// The printable stand-in char for byte `b` (`' '` → `Ġ`).
pub fn byte_to_char(b: u8) -> char {
    tables().0[b as usize]
}

/// The byte a stand-in char denotes, if `c` is in the table.
pub fn char_to_byte(c: char) -> Option<u8> {
    let cp = c as u32 as usize;
    if cp < 0x200 { tables().1[cp] } else { None }
}

// ── PreTokenizedString ──────────────────────────────────────────────────

/// One split: an alignment-tracked slice of the input, plus its model
/// tokens once tokenized (added-token splits arrive pre-tokenized).
#[derive(Debug, Clone)]
pub struct Split {
    pub normalized: NormalizedString,
    pub tokens: Option<Vec<ModelToken>>,
}

impl From<NormalizedString> for Split {
    fn from(normalized: NormalizedString) -> Self {
        Split {
            normalized,
            tokens: None,
        }
    }
}

/// One row of [`PreTokenizedString::splits`]: the split's normalized
/// text, its absolute original byte span, and its tokens if any.
pub type SplitView<'a> = (&'a str, (usize, usize), &'a Option<Vec<ModelToken>>);

/// The input under pre-tokenization; see the module docs.
#[derive(Debug, Clone)]
pub struct PreTokenizedString {
    splits: Vec<Split>,
}

impl From<&str> for PreTokenizedString {
    fn from(s: &str) -> Self {
        PreTokenizedString {
            splits: vec![NormalizedString::from(s).into()],
        }
    }
}

impl PreTokenizedString {
    /// Splits every token-less split through `split_fn`; empty results
    /// are dropped, tokenized splits pass through untouched.
    pub fn split<F, S>(&mut self, mut split_fn: F) -> Result<(), TokenizerError>
    where
        F: FnMut(usize, NormalizedString) -> Result<Vec<S>, TokenizerError>,
        S: Into<Split>,
    {
        let mut new_splits = Vec::with_capacity(self.splits.len());
        for (i, split) in self.splits.drain(..).enumerate() {
            if split.tokens.is_some() {
                new_splits.push(split);
                continue;
            }
            for produced in split_fn(i, split.normalized)? {
                let produced: Split = produced.into();
                if !produced.normalized.is_empty() {
                    new_splits.push(produced);
                }
            }
        }
        self.splits = new_splits;
        Ok(())
    }

    /// Applies `normalize` to every token-less split.
    pub fn normalize<F>(&mut self, normalize: F) -> Result<(), TokenizerError>
    where
        F: Fn(&mut NormalizedString) -> Result<(), TokenizerError>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            normalize(&mut split.normalized)?;
        }
        Ok(())
    }

    /// Tokenizes every token-less split.
    pub fn tokenize<F>(&mut self, tokenize: F) -> Result<(), TokenizerError>
    where
        F: Fn(&NormalizedString) -> Result<Vec<ModelToken>, TokenizerError>,
    {
        for split in self.splits.iter_mut().filter(|s| s.tokens.is_none()) {
            split.tokens = Some(tokenize(&split.normalized)?);
        }
        Ok(())
    }

    /// The splits' normalized texts with their ABSOLUTE original byte
    /// offsets (test and inspection surface).
    pub fn splits(&self) -> Vec<SplitView<'_>> {
        self.splits
            .iter()
            .map(|s| {
                (
                    s.normalized.get(),
                    s.normalized.offsets_original(),
                    &s.tokens,
                )
            })
            .collect()
    }

    /// Assembles the [`Encoding`]: per split (in order), each token's
    /// offsets convert from the split's normalized space to ABSOLUTE
    /// original byte offsets through the alignment; word ids are the
    /// split index (or `word_idx` when the caller pins one).
    pub fn into_encoding(
        self,
        word_idx: Option<u32>,
        type_id: u32,
    ) -> Result<Encoding, TokenizerError> {
        if self.splits.is_empty() {
            return Ok(Encoding::default());
        }
        if !self.splits.iter().all(|split| split.tokens.is_some()) {
            return Err(TokenizerError::Encode {
                what: "a split has not been tokenized (internal pipeline fault)".to_string(),
            });
        }
        let mut encoding = Encoding::default();
        for (idx, split) in self.splits.into_iter().enumerate() {
            let normalized = split.normalized;
            let shift = normalized.offsets_original().0;
            for token in split.tokens.expect("checked above") {
                let offsets = normalized
                    .convert_offsets(token.offsets.0..token.offsets.1)
                    .map_or(token.offsets, |range| {
                        (shift + range.start, shift + range.end)
                    });
                encoding.push_token(
                    token.id,
                    token.value,
                    offsets,
                    Some(word_idx.unwrap_or(idx as u32)),
                    type_id,
                );
            }
        }
        Ok(encoding)
    }
}

// ── The executors ───────────────────────────────────────────────────────

/// A compiled, executable pre-tokenizer (regexes parsed at load).
#[derive(Debug)]
pub enum PreTokenizer {
    ByteLevel {
        add_prefix_space: bool,
        /// `trim_offsets` is POST-PROCESSOR semantics; carried in the
        /// artifact on this stage but consumed by `postprocess`.
        trim_offsets: bool,
        regex: Option<Regex>,
    },
    Bert,
    Whitespace(Regex),
    WhitespaceSplit,
    Punctuation(SplitBehavior),
    Digits {
        individual_digits: bool,
    },
    CharDelimiterSplit(char),
    Metaspace {
        replacement: char,
        prepend_scheme: PrependScheme,
        split: bool,
    },
    Split {
        matcher: Matcher,
        behavior: SplitBehavior,
        invert: bool,
    },
    UnicodeScripts,
    FixedLength(usize),
    Sequence(Vec<PreTokenizer>),
}

/// ASCII punctuation OR `\p{P}` — the reference's `is_punc`, shared by
/// `Punctuation` and `BertPreTokenizer`.
fn is_punc(c: char) -> bool {
    c.is_ascii_punctuation() || unicode::category_class(c) == unicode::CategoryClass::Punctuation
}

/// `char::is_numeric` — the `N*` classes, what the reference's `Digits`
/// splits on.
fn is_numeric(c: char) -> bool {
    unicode::is_number(c)
}

impl PreTokenizer {
    /// Compiles an artifact pre-tokenizer config into its executor.
    pub fn compile(config: &PreTokenizerConfig) -> Result<PreTokenizer, TokenizerError> {
        Ok(match config {
            PreTokenizerConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
            } => PreTokenizer::ByteLevel {
                add_prefix_space: *add_prefix_space,
                trim_offsets: *trim_offsets,
                regex: if *use_regex {
                    Some(
                        Regex::parse(GPT2_SPLIT_PATTERN, &TABLES)
                            .map_err(normalized::regex_error)?,
                    )
                } else {
                    None
                },
            },
            PreTokenizerConfig::Bert => PreTokenizer::Bert,
            PreTokenizerConfig::Whitespace => PreTokenizer::Whitespace(
                Regex::parse(r"\w+|[^\w\s]+", &TABLES).map_err(normalized::regex_error)?,
            ),
            PreTokenizerConfig::WhitespaceSplit => PreTokenizer::WhitespaceSplit,
            PreTokenizerConfig::Punctuation { behavior } => PreTokenizer::Punctuation(*behavior),
            PreTokenizerConfig::Digits { individual_digits } => PreTokenizer::Digits {
                individual_digits: *individual_digits,
            },
            PreTokenizerConfig::CharDelimiterSplit { delimiter } => {
                PreTokenizer::CharDelimiterSplit(*delimiter)
            }
            PreTokenizerConfig::Metaspace {
                replacement,
                prepend_scheme,
                split,
            } => PreTokenizer::Metaspace {
                replacement: *replacement,
                prepend_scheme: *prepend_scheme,
                split: *split,
            },
            PreTokenizerConfig::Split {
                pattern,
                behavior,
                invert,
            } => PreTokenizer::Split {
                matcher: Matcher::compile(pattern)?,
                behavior: *behavior,
                invert: *invert,
            },
            PreTokenizerConfig::UnicodeScripts => PreTokenizer::UnicodeScripts,
            PreTokenizerConfig::FixedLength { length } => {
                if *length == 0 {
                    return Err(TokenizerError::Encode {
                        what: "FixedLength pre-tokenizer with length 0 cannot split".to_string(),
                    });
                }
                PreTokenizer::FixedLength(*length)
            }
            PreTokenizerConfig::Sequence(inner) => PreTokenizer::Sequence(
                inner
                    .iter()
                    .map(PreTokenizer::compile)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// Runs this pre-tokenizer over the splits.
    pub fn apply(&self, pretokenized: &mut PreTokenizedString) -> Result<(), TokenizerError> {
        match self {
            PreTokenizer::ByteLevel {
                add_prefix_space,
                trim_offsets: _,
                regex,
            } => {
                pretokenized.split(|_, mut normalized| {
                    if *add_prefix_space && !normalized.get().starts_with(' ') {
                        normalized.prepend(" ");
                    }
                    match regex {
                        Some(re) => {
                            let spans = regex_matches(normalized.get(), re)?;
                            Ok(normalized.split(spans, SplitBehavior::Isolated))
                        }
                        None => Ok(vec![normalized]),
                    }
                })?;
                pretokenized.normalize(|normalized| {
                    let mut transformations: Vec<(char, isize)> =
                        Vec::with_capacity(normalized.get().len());
                    for c in normalized.get().chars() {
                        let mut utf8 = [0u8; 4];
                        for (i, b) in c.encode_utf8(&mut utf8).bytes().enumerate() {
                            transformations.push((byte_to_char(b), isize::from(i > 0)));
                        }
                    }
                    normalized.transform(transformations, 0);
                    Ok(())
                })
            }
            PreTokenizer::Bert => {
                pretokenized.split(|_, normalized| {
                    let spans = char_matches(normalized.get(), char::is_whitespace);
                    Ok(normalized.split(spans, SplitBehavior::Removed))
                })?;
                pretokenized.split(|_, normalized| {
                    let spans = char_matches(normalized.get(), is_punc);
                    Ok(normalized.split(spans, SplitBehavior::Isolated))
                })
            }
            PreTokenizer::Whitespace(re) => pretokenized.split(|_, normalized| {
                let spans = normalized::invert(regex_matches(normalized.get(), re)?);
                Ok(normalized.split(spans, SplitBehavior::Removed))
            }),
            PreTokenizer::WhitespaceSplit => pretokenized.split(|_, normalized| {
                let spans = char_matches(normalized.get(), char::is_whitespace);
                Ok(normalized.split(spans, SplitBehavior::Removed))
            }),
            PreTokenizer::Punctuation(behavior) => pretokenized.split(|_, normalized| {
                let spans = char_matches(normalized.get(), is_punc);
                Ok(normalized.split(spans, *behavior))
            }),
            PreTokenizer::Digits { individual_digits } => {
                let behavior = if *individual_digits {
                    SplitBehavior::Isolated
                } else {
                    SplitBehavior::Contiguous
                };
                pretokenized.split(|_, normalized| {
                    let spans = char_matches(normalized.get(), is_numeric);
                    Ok(normalized.split(spans, behavior))
                })
            }
            PreTokenizer::CharDelimiterSplit(delimiter) => pretokenized.split(|_, normalized| {
                let spans = char_matches(normalized.get(), |c| c == *delimiter);
                Ok(normalized.split(spans, SplitBehavior::Removed))
            }),
            PreTokenizer::Metaspace {
                replacement,
                prepend_scheme,
                split,
            } => {
                let rep = replacement.to_string();
                pretokenized.split(|_, mut normalized| {
                    let spans = char_matches(normalized.get(), |c| c == ' ');
                    normalized.replace_spans(&spans, &rep);
                    let prepend = match prepend_scheme {
                        PrependScheme::Always => !normalized.get().starts_with(*replacement),
                        PrependScheme::First => {
                            !normalized.get().starts_with(*replacement)
                                && normalized.offsets_original().0 == 0
                        }
                        PrependScheme::Never => false,
                    };
                    if prepend {
                        normalized.prepend(&rep);
                    }
                    if *split {
                        let spans = char_matches(normalized.get(), |c| c == *replacement);
                        Ok(normalized.split(spans, SplitBehavior::MergedWithNext))
                    } else {
                        Ok(vec![normalized])
                    }
                })
            }
            PreTokenizer::Split {
                matcher,
                behavior,
                invert,
            } => pretokenized.split(|_, normalized| {
                let mut spans = matcher.spans(normalized.get())?;
                if *invert {
                    spans = normalized::invert(spans);
                }
                Ok(normalized.split(spans, *behavior))
            }),
            PreTokenizer::UnicodeScripts => {
                pretokenized.split(|_, normalized| Ok(unicode_scripts_split(&normalized)))
            }
            PreTokenizer::FixedLength(length) => pretokenized.split(|_, normalized| {
                let text = normalized.get();
                if text.is_empty() {
                    return Ok(vec![]);
                }
                let positions: Vec<(usize, char)> = text.char_indices().collect();
                let mut out = Vec::new();
                for chunk in positions.chunks(*length) {
                    let start = chunk[0].0;
                    let end = chunk[chunk.len() - 1].0 + chunk[chunk.len() - 1].1.len_utf8();
                    if let Some(slice) = normalized.slice(start..end) {
                        out.push(slice);
                    }
                }
                Ok(out)
            }),
            PreTokenizer::Sequence(inner) => {
                for pretok in inner {
                    pretok.apply(pretokenized)?;
                }
                Ok(())
            }
        }
    }
}

// ── UnicodeScripts (reference `unicode_scripts/pre_tokenizer.rs`) ───────

/// The sentencepiece "any" bucket plus a concrete script.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpScript {
    Any,
    Real(Script),
}

/// The reference's `fixed_script`: U+30FC and the kana scripts fold
/// into Han, the space char and unassigned code points are `Any` (the
/// reference table's fallthrough); everything else — Common and
/// Inherited included — is its own run-breaking script.
fn fixed_script(c: char) -> SpScript {
    if c as u32 == 0x30FC {
        return SpScript::Real(Script::Han);
    }
    if c == ' ' {
        return SpScript::Any;
    }
    match unicode::script(c) {
        Script::Hiragana | Script::Katakana => SpScript::Real(Script::Han),
        Script::Unknown => SpScript::Any,
        s => SpScript::Real(s),
    }
}

/// Script-run boundaries exactly as the reference computes them: a
/// boundary lands before every non-`Any` char whose script differs from
/// the previous non-`Any` script. Note the reference quirk this
/// preserves: a leading `Any` run (e.g. spaces) precedes the first
/// boundary and is dropped from the splits.
fn unicode_scripts_split(normalized: &NormalizedString) -> Vec<NormalizedString> {
    let mut last_script: Option<SpScript> = None;
    let mut offset = 0;
    let mut ranges: Vec<usize> = normalized
        .get()
        .chars()
        .filter_map(|c| {
            let script = fixed_script(c);
            let result = if script != SpScript::Any
                && last_script != Some(SpScript::Any)
                && last_script != Some(script)
            {
                Some(offset)
            } else {
                None
            };
            offset += c.len_utf8();
            if script != SpScript::Any {
                last_script = Some(script);
            }
            result
        })
        .collect();
    ranges.push(normalized.get().len());
    ranges
        .windows(2)
        .filter_map(|pair| normalized.slice(pair[0]..pair[1]))
        .collect()
}

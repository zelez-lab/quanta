//! The typed `tokenizer.json` schema layer: parse, validate, hold.
//!
//! [`TokenizerArtifact::from_bytes`] turns artifact bytes into typed
//! config structs for every pipeline family the pinned reference
//! ([`crate::PINNED_REFERENCE`]) serializes — normalizers,
//! pre-tokenizers, models, post-processors, decoders, added tokens,
//! truncation/padding. This layer holds *configuration*, not behavior:
//! the pipeline stages that execute these configs are built on top of
//! it, so an artifact that loads here has already had every stage tag
//! recognized, every field typed, and the vocabulary cross-checked
//! (the §7 loads-means-runs rule for the schema's share of the work).
//!
//! Claim boundaries, enforced loudly:
//!
//! - A `type` tag outside the pinned inventory is
//!   [`TokenizerError::UnknownTag`] naming the family, the tag, and the
//!   pinned reference — format growth is additive work against a
//!   detected marker, never a misparse (§2 of the scope).
//! - BPE `dropout` non-null is [`TokenizerError::UnsupportedField`]:
//!   train-time stochastic augmentation, excluded by the scope (§12);
//!   inference artifacts serialize `null`.
//! - Unknown *fields* inside a known variant are ignored — the pinned
//!   reference's own serde posture (this is how `cache_capacity` and
//!   the legacy `str_rep` are accepted-and-ignored). Absent fields take
//!   the reference builders' defaults, so hand-written minimal
//!   artifacts load the way the reference loads them.
//!
//! Vocabulary validation at load (§8's `Vocab` row): id collisions,
//! ids that do not fit `u32`, BPE merges naming absent tokens (with
//! the reference's `continuing_subword_prefix` arithmetic, bounds-
//! checked), Unigram duplicate pieces and out-of-range `unk_id`.
//! Duplicate tokens in map-shaped vocabs are impossible here — the
//! JSON layer rejects duplicate object keys outright.
//!
//! The `Precompiled` charsmap's base64 is decoded (and therefore
//! validated) at load; the decoded blob is held raw for the normalizer
//! layer's bounds-checked trie walker.

use crate::error::TokenizerError;
use crate::json::{self, Value};
use std::collections::HashSet;

// ── The artifact ────────────────────────────────────────────────────────

/// A fully-typed, validated `tokenizer.json` — the five pipeline stages
/// plus added tokens and saved truncation/padding. Field order mirrors
/// the artifact's top level.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizerArtifact {
    /// The artifact's `version` field (always `"1.0"` today; a newer
    /// version string is a loud schema error, not a guess).
    pub version: Option<String>,
    pub truncation: Option<TruncationConfig>,
    pub padding: Option<PaddingConfig>,
    pub added_tokens: Vec<AddedTokenConfig>,
    pub normalizer: Option<NormalizerConfig>,
    pub pre_tokenizer: Option<PreTokenizerConfig>,
    pub model: ModelConfig,
    pub post_processor: Option<PostProcessorConfig>,
    pub decoder: Option<DecoderConfig>,
}

impl TokenizerArtifact {
    /// Parse and validate a whole `tokenizer.json` from bytes (the
    /// house bytes rule — `std::fs::read` is the caller's one-liner).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let root = json::parse(bytes)?;
        if root.as_object().is_none() {
            return Err(schema(
                "(root)",
                format!("the artifact must be a JSON object, found {}", root.kind()),
            ));
        }
        let version = match nullable(&root, "version") {
            None => None,
            Some(v) => {
                let s = expect_str(v, "version")?;
                if s != "1.0" {
                    return Err(schema(
                        "version",
                        format!(
                            "format version {s:?} is not the pinned reference's \"1.0\" — \
                             a newer tokenizers format needs additive support, refusing it \
                             here prevents a silent misparse"
                        ),
                    ));
                }
                Some(s.to_string())
            }
        };
        let truncation = nullable(&root, "truncation")
            .map(|v| truncation_config(v, "truncation"))
            .transpose()?;
        let padding = nullable(&root, "padding")
            .map(|v| padding_config(v, "padding"))
            .transpose()?;
        let added_tokens = match nullable(&root, "added_tokens") {
            None => Vec::new(),
            Some(v) => added_tokens(v, "added_tokens")?,
        };
        let normalizer = nullable(&root, "normalizer")
            .map(|v| normalizer(v, "normalizer"))
            .transpose()?;
        let pre_tokenizer = nullable(&root, "pre_tokenizer")
            .map(|v| pre_tokenizer(v, "pre_tokenizer"))
            .transpose()?;
        let model = model(
            nullable(&root, "model").ok_or_else(|| {
                schema("model", "is missing — every tokenizer.json carries a model")
            })?,
            "model",
        )?;
        let post_processor = nullable(&root, "post_processor")
            .map(|v| post_processor(v, "post_processor"))
            .transpose()?;
        let decoder = nullable(&root, "decoder")
            .map(|v| decoder(v, "decoder"))
            .transpose()?;
        Ok(TokenizerArtifact {
            version,
            truncation,
            padding,
            added_tokens,
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
        })
    }
}

// ── Config types ────────────────────────────────────────────────────────

/// A `pattern` field: a plain string (find/replace) or a regex routed
/// through the split-regex engine.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternConfig {
    String(String),
    Regex(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizerConfig {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Bert {
        clean_text: bool,
        handle_chinese_chars: bool,
        /// `null` follows `lowercase` — the reference's documented
        /// default, resolved by the normalizer layer, held verbatim here.
        strip_accents: Option<bool>,
        lowercase: bool,
    },
    Lowercase,
    Strip {
        strip_left: bool,
        strip_right: bool,
    },
    StripAccents,
    Prepend {
        prepend: String,
    },
    Replace {
        pattern: PatternConfig,
        content: String,
    },
    /// The sentencepiece charsmap, base64-decoded at load. The blob is
    /// interpreted by the normalizer layer's bounds-checked trie walker.
    Precompiled {
        charsmap: Vec<u8>,
    },
    Nmt,
    ByteLevel,
    Sequence(Vec<NormalizerConfig>),
}

/// `prepend_scheme` for Metaspace (both forms). The legacy
/// `add_prefix_space` serialization maps onto it at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrependScheme {
    Always,
    Never,
    First,
}

/// `behavior` for `Split` / `Punctuation` — the five reference modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitBehavior {
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreTokenizerConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
    /// Tag `BertPreTokenizer`.
    Bert,
    Whitespace,
    WhitespaceSplit,
    Punctuation {
        behavior: SplitBehavior,
    },
    Digits {
        individual_digits: bool,
    },
    CharDelimiterSplit {
        delimiter: char,
    },
    Metaspace {
        replacement: char,
        prepend_scheme: PrependScheme,
        split: bool,
    },
    Split {
        pattern: PatternConfig,
        behavior: SplitBehavior,
        invert: bool,
    },
    UnicodeScripts,
    FixedLength {
        length: usize,
    },
    Sequence(Vec<PreTokenizerConfig>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelConfig {
    /// Tag `BPE`. `cache_capacity` is accepted and ignored (a perf
    /// hint, not semantics); `dropout` non-null is a loud
    /// [`TokenizerError::UnsupportedField`].
    Bpe {
        /// Token → id, source order, duplicate-free (both directions).
        vocab: Vec<(String, u32)>,
        /// Merge pairs. Both serialized spellings load — legacy
        /// `"a b"` strings and current `["a", "b"]` pairs.
        merges: Vec<(String, String)>,
        unk_token: Option<String>,
        continuing_subword_prefix: Option<String>,
        end_of_word_suffix: Option<String>,
        fuse_unk: bool,
        byte_fallback: bool,
        ignore_merges: bool,
    },
    WordPiece {
        vocab: Vec<(String, u32)>,
        unk_token: String,
        continuing_subword_prefix: String,
        max_input_chars_per_word: usize,
    },
    Unigram {
        /// `[piece, logprob]` rows, source order, duplicate-free.
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    },
    WordLevel {
        vocab: Vec<(String, u32)>,
        unk_token: String,
    },
}

/// One `single`/`pair` template entry in `TemplateProcessing`.
#[derive(Debug, Clone, PartialEq)]
pub enum TemplatePiece {
    /// `{"Sequence": {"id": "A"|"B", "type_id": n}}`.
    Sequence { id: SequenceId, type_id: u32 },
    /// `{"SpecialToken": {"id": "[CLS]", "type_id": n}}` — the id must
    /// name an entry in the processor's `special_tokens` map.
    SpecialToken { id: String, type_id: u32 },
}

/// Which input sequence a template piece refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceId {
    A,
    B,
}

/// One `special_tokens` map entry in `TemplateProcessing`:
/// `{"id": …, "ids": […], "tokens": […]}` with `ids` and `tokens` the
/// same length (validated at load, as the reference validates it).
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialTokenConfig {
    pub id: String,
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PostProcessorConfig {
    Template {
        single: Vec<TemplatePiece>,
        pair: Vec<TemplatePiece>,
        special_tokens: Vec<SpecialTokenConfig>,
    },
    /// Tag `BertProcessing` — `(token, id)` for `sep` and `cls`.
    Bert {
        sep: (String, u32),
        cls: (String, u32),
    },
    Roberta {
        sep: (String, u32),
        cls: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    },
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
    Sequence(Vec<PostProcessorConfig>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DecoderConfig {
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
        use_regex: bool,
    },
    WordPiece {
        prefix: String,
        cleanup: bool,
    },
    /// Tag `BPEDecoder`.
    BpeDecoder {
        suffix: String,
    },
    Metaspace {
        replacement: char,
        prepend_scheme: PrependScheme,
        split: bool,
    },
    ByteFallback,
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Replace {
        pattern: PatternConfig,
        content: String,
    },
    Ctc {
        pad_token: String,
        word_delimiter_token: String,
        cleanup: bool,
    },
    Sequence(Vec<DecoderConfig>),
}

/// One `added_tokens` entry — the full AddedVocabulary flag set. When
/// `normalized` is absent it defaults to `!special`, mirroring the
/// reference's `AddedToken::from(content, special)` builder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddedTokenConfig {
    pub id: u32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

/// Truncation/padding side: `Left` or `Right`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Left,
    Right,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    LongestFirst,
    OnlyFirst,
    OnlySecond,
}

/// The saved `truncation` section — loaded values are the active
/// defaults (the saved sections are the saved behavior).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationConfig {
    pub direction: Direction,
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

/// `strategy` in the `padding` section: `"BatchLongest"` or
/// `{"Fixed": n}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    BatchLongest,
    Fixed(usize),
}

/// The saved `padding` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddingConfig {
    pub strategy: PaddingStrategy,
    pub direction: Direction,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: u32,
    pub pad_type_id: u32,
    pub pad_token: String,
}

// ── Schema-walk helpers ─────────────────────────────────────────────────

fn schema(path: &str, what: impl Into<String>) -> TokenizerError {
    TokenizerError::Schema {
        path: path.to_string(),
        what: what.into(),
    }
}

fn join(base: &str, key: &str) -> String {
    format!("{base}.{key}")
}

fn idx(base: &str, i: usize) -> String {
    format!("{base}[{i}]")
}

/// Member lookup with `null` reading as absent (the artifact writes
/// `"normalizer": null` for "no normalizer").
fn nullable<'a>(obj: &'a Value, key: &str) -> Option<&'a Value> {
    match obj.get(key) {
        Some(Value::Null) | None => None,
        Some(v) => Some(v),
    }
}

/// A required field: present and non-null, or a loud schema error.
fn req<'a>(obj: &'a Value, key: &str, path: &str) -> Result<&'a Value, TokenizerError> {
    nullable(obj, key).ok_or_else(|| {
        schema(
            &join(path, key),
            "is missing (required, no reference default)",
        )
    })
}

fn expect_str<'a>(v: &'a Value, path: &str) -> Result<&'a str, TokenizerError> {
    v.as_str()
        .ok_or_else(|| schema(path, format!("expected a string, found {}", v.kind())))
}

fn expect_bool(v: &Value, path: &str) -> Result<bool, TokenizerError> {
    v.as_bool()
        .ok_or_else(|| schema(path, format!("expected a bool, found {}", v.kind())))
}

fn expect_array<'a>(v: &'a Value, path: &str) -> Result<&'a [Value], TokenizerError> {
    v.as_array()
        .ok_or_else(|| schema(path, format!("expected an array, found {}", v.kind())))
}

fn expect_object<'a>(v: &'a Value, path: &str) -> Result<&'a [(String, Value)], TokenizerError> {
    v.as_object()
        .ok_or_else(|| schema(path, format!("expected an object, found {}", v.kind())))
}

fn expect_f64(v: &Value, path: &str) -> Result<f64, TokenizerError> {
    v.as_f64()
        .ok_or_else(|| schema(path, format!("expected a number, found {}", v.kind())))
}

fn expect_u32(v: &Value, path: &str) -> Result<u32, TokenizerError> {
    v.as_number().and_then(json::Number::as_u32).ok_or_else(|| {
        schema(
            path,
            format!("expected an unsigned 32-bit integer, found {}", v.kind()),
        )
    })
}

fn expect_usize(v: &Value, path: &str) -> Result<usize, TokenizerError> {
    v.as_number()
        .and_then(json::Number::as_usize)
        .ok_or_else(|| {
            schema(
                path,
                format!("expected an unsigned integer, found {}", v.kind()),
            )
        })
}

/// A string field holding exactly one character (`delimiter`,
/// Metaspace `replacement`, Strip-decoder `content`).
fn expect_char(v: &Value, path: &str) -> Result<char, TokenizerError> {
    let s = expect_str(v, path)?;
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(schema(
            path,
            format!("expected a single-character string, found {s:?}"),
        )),
    }
}

fn bool_or(obj: &Value, key: &str, path: &str, default: bool) -> Result<bool, TokenizerError> {
    match nullable(obj, key) {
        None => Ok(default),
        Some(v) => expect_bool(v, &join(path, key)),
    }
}

fn u32_or(obj: &Value, key: &str, path: &str, default: u32) -> Result<u32, TokenizerError> {
    match nullable(obj, key) {
        None => Ok(default),
        Some(v) => expect_u32(v, &join(path, key)),
    }
}

fn usize_or(obj: &Value, key: &str, path: &str, default: usize) -> Result<usize, TokenizerError> {
    match nullable(obj, key) {
        None => Ok(default),
        Some(v) => expect_usize(v, &join(path, key)),
    }
}

fn string_or(obj: &Value, key: &str, path: &str, default: &str) -> Result<String, TokenizerError> {
    match nullable(obj, key) {
        None => Ok(default.to_string()),
        Some(v) => Ok(expect_str(v, &join(path, key))?.to_string()),
    }
}

fn opt_string(obj: &Value, key: &str, path: &str) -> Result<Option<String>, TokenizerError> {
    nullable(obj, key)
        .map(|v| expect_str(v, &join(path, key)).map(str::to_string))
        .transpose()
}

/// Split a tagged stage object into its `type` tag and the object.
fn tag<'a>(v: &'a Value, path: &str) -> Result<(&'a str, &'a Value), TokenizerError> {
    expect_object(v, path)?;
    let t = v.get("type").ok_or_else(|| {
        schema(
            &join(path, "type"),
            "is missing — pipeline stages are tagged objects",
        )
    })?;
    Ok((expect_str(t, &join(path, "type"))?, v))
}

// ── Shared field groups ─────────────────────────────────────────────────

/// The shared ByteLevel struct (pre-tokenizer / post-processor /
/// decoder forms all serialize the same three fields). Defaults are the
/// reference builder's (`true` for all three).
fn byte_level_fields(o: &Value, path: &str) -> Result<(bool, bool, bool), TokenizerError> {
    Ok((
        bool_or(o, "add_prefix_space", path, true)?,
        bool_or(o, "trim_offsets", path, true)?,
        bool_or(o, "use_regex", path, true)?,
    ))
}

/// Metaspace fields (pre-tokenizer and decoder forms). Both serialized
/// spellings load: current `prepend_scheme`, and the legacy
/// `add_prefix_space` bool, which maps to `Always`/`Never` when
/// `prepend_scheme` is absent (when both are present the explicit
/// scheme wins). The legacy `str_rep` field is accepted and ignored.
fn metaspace_fields(o: &Value, path: &str) -> Result<(char, PrependScheme, bool), TokenizerError> {
    let replacement = match nullable(o, "replacement") {
        None => '\u{2581}', // ▁ — the reference default
        Some(v) => expect_char(v, &join(path, "replacement"))?,
    };
    let split = bool_or(o, "split", path, true)?;
    let scheme = match nullable(o, "prepend_scheme") {
        Some(v) => prepend_scheme(v, &join(path, "prepend_scheme"))?,
        None => match nullable(o, "add_prefix_space") {
            Some(v) => {
                if expect_bool(v, &join(path, "add_prefix_space"))? {
                    PrependScheme::Always
                } else {
                    PrependScheme::Never
                }
            }
            None => PrependScheme::Always,
        },
    };
    Ok((replacement, scheme, split))
}

fn prepend_scheme(v: &Value, path: &str) -> Result<PrependScheme, TokenizerError> {
    match expect_str(v, path)? {
        "always" => Ok(PrependScheme::Always),
        "never" => Ok(PrependScheme::Never),
        "first" => Ok(PrependScheme::First),
        other => Err(schema(
            path,
            format!(
                "unknown prepend scheme {other:?} (expected \"always\", \"never\", or \
                 \"first\")"
            ),
        )),
    }
}

fn split_behavior(v: &Value, path: &str) -> Result<SplitBehavior, TokenizerError> {
    match expect_str(v, path)? {
        "Removed" => Ok(SplitBehavior::Removed),
        "Isolated" => Ok(SplitBehavior::Isolated),
        "MergedWithPrevious" => Ok(SplitBehavior::MergedWithPrevious),
        "MergedWithNext" => Ok(SplitBehavior::MergedWithNext),
        "Contiguous" => Ok(SplitBehavior::Contiguous),
        other => Err(schema(
            path,
            format!(
                "unknown split behavior {other:?} (expected \"Removed\", \"Isolated\", \
                 \"MergedWithPrevious\", \"MergedWithNext\", or \"Contiguous\")"
            ),
        )),
    }
}

fn pattern(v: &Value, path: &str) -> Result<PatternConfig, TokenizerError> {
    let o = expect_object(v, path)?;
    let [(kind, inner)] = o else {
        return Err(schema(
            path,
            "a pattern is exactly one of {\"String\": …} or {\"Regex\": …}",
        ));
    };
    let text = expect_str(inner, &join(path, kind))?.to_string();
    match kind.as_str() {
        "String" => Ok(PatternConfig::String(text)),
        "Regex" => Ok(PatternConfig::Regex(text)),
        other => Err(schema(
            path,
            format!("unknown pattern kind {other:?} (expected \"String\" or \"Regex\")"),
        )),
    }
}

fn direction_or(obj: &Value, key: &str, path: &str) -> Result<Direction, TokenizerError> {
    match nullable(obj, key) {
        None => Ok(Direction::Right),
        Some(v) => match expect_str(v, &join(path, key))? {
            "Left" => Ok(Direction::Left),
            "Right" => Ok(Direction::Right),
            other => Err(schema(
                &join(path, key),
                format!("unknown direction {other:?} (expected \"Left\" or \"Right\")"),
            )),
        },
    }
}

// ── Base64 (the Precompiled charsmap carrier) ───────────────────────────

/// Standard-alphabet base64 with mandatory `=` padding — what the
/// reference writes for `precompiled_charsmap`. Strict: rejects
/// non-alphabet characters, bad length, misplaced padding, and nonzero
/// trailing bits (a real encoder always emits zero bits). Errors are
/// [`TokenizerError::Charsmap`] with the offset into the base64 text.
fn base64_decode(s: &str, path: &str) -> Result<Vec<u8>, TokenizerError> {
    let bad = |at: usize, what: String| TokenizerError::Charsmap { at, what };
    let b = s.as_bytes();
    if !b.len().is_multiple_of(4) {
        return Err(bad(
            b.len(),
            format!(
                "base64 length {} is not a multiple of 4 (at {path}; the reference \
                 always writes padded base64)",
                b.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    let sextet = |c: u8, at: usize| -> Result<u32, TokenizerError> {
        match c {
            b'A'..=b'Z' => Ok(u32::from(c - b'A')),
            b'a'..=b'z' => Ok(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(c - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(bad(
                at,
                format!("invalid base64 character {:?}", char::from(c)),
            )),
        }
    };
    for (chunk_start, chunk) in (0..b.len()).step_by(4).zip(b.chunks_exact(4)) {
        let pad = match chunk {
            [_, _, b'=', b'='] => 2,
            [_, _, _, b'='] => 1,
            _ => 0,
        };
        if pad > 0 && chunk_start + 4 != b.len() {
            return Err(bad(
                chunk_start,
                "padding '=' before the end of the base64 text".to_string(),
            ));
        }
        if chunk[..2].contains(&b'=') {
            return Err(bad(chunk_start, "misplaced padding '='".to_string()));
        }
        let mut acc: u32 = 0;
        for (j, &c) in chunk[..4 - pad].iter().enumerate() {
            acc = (acc << 6) | sextet(c, chunk_start + j)?;
        }
        match pad {
            0 => out.extend_from_slice(&[(acc >> 16) as u8, (acc >> 8) as u8, acc as u8]),
            1 => {
                if acc & 0x3 != 0 {
                    return Err(bad(
                        chunk_start,
                        "nonzero trailing bits before padding (corrupt base64)".to_string(),
                    ));
                }
                acc >>= 2;
                out.extend_from_slice(&[(acc >> 8) as u8, acc as u8]);
            }
            _ => {
                if acc & 0xF != 0 {
                    return Err(bad(
                        chunk_start,
                        "nonzero trailing bits before padding (corrupt base64)".to_string(),
                    ));
                }
                out.push((acc >> 4) as u8);
            }
        }
    }
    Ok(out)
}

// ── Normalizers ─────────────────────────────────────────────────────────

fn normalizer(v: &Value, path: &str) -> Result<NormalizerConfig, TokenizerError> {
    let (t, o) = tag(v, path)?;
    match t {
        "NFC" => Ok(NormalizerConfig::Nfc),
        "NFD" => Ok(NormalizerConfig::Nfd),
        "NFKC" => Ok(NormalizerConfig::Nfkc),
        "NFKD" => Ok(NormalizerConfig::Nfkd),
        "BertNormalizer" => Ok(NormalizerConfig::Bert {
            clean_text: bool_or(o, "clean_text", path, true)?,
            handle_chinese_chars: bool_or(o, "handle_chinese_chars", path, true)?,
            strip_accents: nullable(o, "strip_accents")
                .map(|v| expect_bool(v, &join(path, "strip_accents")))
                .transpose()?,
            lowercase: bool_or(o, "lowercase", path, true)?,
        }),
        "Lowercase" => Ok(NormalizerConfig::Lowercase),
        "Strip" => Ok(NormalizerConfig::Strip {
            strip_left: bool_or(o, "strip_left", path, true)?,
            strip_right: bool_or(o, "strip_right", path, true)?,
        }),
        "StripAccents" => Ok(NormalizerConfig::StripAccents),
        "Prepend" => Ok(NormalizerConfig::Prepend {
            prepend: expect_str(req(o, "prepend", path)?, &join(path, "prepend"))?.to_string(),
        }),
        "Replace" => Ok(NormalizerConfig::Replace {
            pattern: pattern(req(o, "pattern", path)?, &join(path, "pattern"))?,
            content: expect_str(req(o, "content", path)?, &join(path, "content"))?.to_string(),
        }),
        "Precompiled" => {
            let field = join(path, "precompiled_charsmap");
            let text = expect_str(req(o, "precompiled_charsmap", path)?, &field)?;
            Ok(NormalizerConfig::Precompiled {
                charsmap: base64_decode(text, &field)?,
            })
        }
        "Nmt" => Ok(NormalizerConfig::Nmt),
        "ByteLevel" => Ok(NormalizerConfig::ByteLevel),
        "Sequence" => {
            let arr = expect_array(req(o, "normalizers", path)?, &join(path, "normalizers"))?;
            let inner = arr
                .iter()
                .enumerate()
                .map(|(i, v)| normalizer(v, &idx(&join(path, "normalizers"), i)))
                .collect::<Result<_, _>>()?;
            Ok(NormalizerConfig::Sequence(inner))
        }
        other => Err(TokenizerError::UnknownTag {
            family: "normalizer",
            tag: other.to_string(),
        }),
    }
}

// ── Pre-tokenizers ──────────────────────────────────────────────────────

fn pre_tokenizer(v: &Value, path: &str) -> Result<PreTokenizerConfig, TokenizerError> {
    let (t, o) = tag(v, path)?;
    match t {
        "ByteLevel" => {
            let (add_prefix_space, trim_offsets, use_regex) = byte_level_fields(o, path)?;
            Ok(PreTokenizerConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
            })
        }
        "BertPreTokenizer" => Ok(PreTokenizerConfig::Bert),
        "Whitespace" => Ok(PreTokenizerConfig::Whitespace),
        "WhitespaceSplit" => Ok(PreTokenizerConfig::WhitespaceSplit),
        "Punctuation" => Ok(PreTokenizerConfig::Punctuation {
            behavior: match nullable(o, "behavior") {
                None => SplitBehavior::Isolated,
                Some(v) => split_behavior(v, &join(path, "behavior"))?,
            },
        }),
        "Digits" => Ok(PreTokenizerConfig::Digits {
            individual_digits: bool_or(o, "individual_digits", path, false)?,
        }),
        "CharDelimiterSplit" => Ok(PreTokenizerConfig::CharDelimiterSplit {
            delimiter: expect_char(req(o, "delimiter", path)?, &join(path, "delimiter"))?,
        }),
        "Metaspace" => {
            let (replacement, prepend_scheme, split) = metaspace_fields(o, path)?;
            Ok(PreTokenizerConfig::Metaspace {
                replacement,
                prepend_scheme,
                split,
            })
        }
        "Split" => Ok(PreTokenizerConfig::Split {
            pattern: pattern(req(o, "pattern", path)?, &join(path, "pattern"))?,
            behavior: split_behavior(req(o, "behavior", path)?, &join(path, "behavior"))?,
            invert: bool_or(o, "invert", path, false)?,
        }),
        "UnicodeScripts" => Ok(PreTokenizerConfig::UnicodeScripts),
        "FixedLength" => Ok(PreTokenizerConfig::FixedLength {
            // The reference default; artifacts serialize the field.
            length: usize_or(o, "length", path, 5)?,
        }),
        "Sequence" => {
            let arr = expect_array(req(o, "pretokenizers", path)?, &join(path, "pretokenizers"))?;
            let inner = arr
                .iter()
                .enumerate()
                .map(|(i, v)| pre_tokenizer(v, &idx(&join(path, "pretokenizers"), i)))
                .collect::<Result<_, _>>()?;
            Ok(PreTokenizerConfig::Sequence(inner))
        }
        other => Err(TokenizerError::UnknownTag {
            family: "pre_tokenizer",
            tag: other.to_string(),
        }),
    }
}

// ── Models ──────────────────────────────────────────────────────────────

/// A map-shaped vocab (`{"token": id}`) → ordered rows, with the §8
/// id checks: every id fits `u32`, no two tokens share an id.
/// (Duplicate *tokens* cannot reach here — the JSON layer rejects
/// duplicate object keys.)
fn vocab_u32(v: &Value, path: &str) -> Result<Vec<(String, u32)>, TokenizerError> {
    let entries = expect_object(v, path)?;
    let mut vocab = Vec::with_capacity(entries.len());
    let mut ids: HashSet<u32> = HashSet::with_capacity(entries.len());
    for (token, val) in entries {
        let n = val.as_number().ok_or_else(|| {
            schema(
                &format!("{path}[{token:?}]"),
                format!("expected an integer id, found {}", val.kind()),
            )
        })?;
        let Some(wide) = n.as_u64() else {
            return Err(schema(
                &format!("{path}[{token:?}]"),
                format!("expected an unsigned integer id, found {}", n.raw),
            ));
        };
        let id = u32::try_from(wide).map_err(|_| TokenizerError::Vocab {
            what: format!(
                "token {token:?} has id {wide}, which does not fit in u32 (ids are u32 \
                 by format contract)"
            ),
        })?;
        if !ids.insert(id) {
            return Err(TokenizerError::Vocab {
                what: format!(
                    "id collision: token {token:?} reuses id {id}, already assigned to \
                     another token"
                ),
            });
        }
        vocab.push((token.clone(), id));
    }
    Ok(vocab)
}

/// BPE merges, both serialized spellings. Legacy `"a b"` strings split
/// on their single space (a token containing a space is inexpressible
/// there — that is why the pair spelling exists); `["a", "b"]` pairs
/// are taken verbatim. One artifact uses one spelling — a mixed array
/// fails the reference's untagged deserialization and fails here too.
fn merges(v: &Value, path: &str) -> Result<Vec<(String, String)>, TokenizerError> {
    let arr = expect_array(v, path)?;
    let mut out = Vec::with_capacity(arr.len());
    let mut legacy: Option<bool> = None;
    for (i, m) in arr.iter().enumerate() {
        let p = idx(path, i);
        let (pair, is_legacy) = match m {
            Value::String(s) => {
                let parts: Vec<&str> = s.split(' ').collect();
                let [a, b] = parts.as_slice() else {
                    return Err(schema(
                        &p,
                        format!(
                            "legacy merge {s:?} must be exactly \"left right\" (one \
                             space); tokens containing spaces need the [\"a\", \"b\"] \
                             pair spelling"
                        ),
                    ));
                };
                if a.is_empty() || b.is_empty() {
                    return Err(schema(&p, format!("legacy merge {s:?} has an empty side")));
                }
                ((a.to_string(), b.to_string()), true)
            }
            Value::Array(pair) => {
                let [a, b] = pair.as_slice() else {
                    return Err(schema(
                        &p,
                        format!("a merge pair holds exactly 2 tokens, found {}", pair.len()),
                    ));
                };
                (
                    (
                        expect_str(a, &idx(&p, 0))?.to_string(),
                        expect_str(b, &idx(&p, 1))?.to_string(),
                    ),
                    false,
                )
            }
            other => {
                return Err(schema(
                    &p,
                    format!(
                        "expected a merge (\"a b\" string or [\"a\", \"b\"] pair), found {}",
                        other.kind()
                    ),
                ));
            }
        };
        if *legacy.get_or_insert(is_legacy) != is_legacy {
            return Err(schema(
                &p,
                "mixed merge spellings in one artifact (the reference writes one \
                 spelling per file)",
            ));
        }
        out.push(pair);
    }
    Ok(out)
}

/// The §8 merge cross-check, with the reference's arithmetic: merge
/// `(a, b)` requires `a`, `b`, and `a + b[prefix..]` in the vocab
/// (the reference strips `continuing_subword_prefix` off `b` before
/// concatenating — bounds-checked here so a corrupt artifact errs
/// instead of panicking).
fn check_merges(
    vocab: &[(String, u32)],
    merges: &[(String, String)],
    prefix: Option<&str>,
) -> Result<(), TokenizerError> {
    let tokens: HashSet<&str> = vocab.iter().map(|(t, _)| t.as_str()).collect();
    let prefix_len = prefix.map_or(0, str::len);
    for (a, b) in merges {
        for side in [a, b] {
            if !tokens.contains(side.as_str()) {
                return Err(TokenizerError::Vocab {
                    what: format!(
                        "merge ({a:?}, {b:?}) names token {side:?}, which is not in the \
                         vocab"
                    ),
                });
            }
        }
        let stripped = if prefix_len == 0 {
            b.as_str()
        } else if b.len() >= prefix_len && b.is_char_boundary(prefix_len) {
            &b[prefix_len..]
        } else {
            return Err(TokenizerError::Vocab {
                what: format!(
                    "merge ({a:?}, {b:?}): token {b:?} is shorter than the \
                     continuing_subword_prefix — the artifact is corrupt"
                ),
            });
        };
        let merged = format!("{a}{stripped}");
        if !tokens.contains(merged.as_str()) {
            return Err(TokenizerError::Vocab {
                what: format!(
                    "merge ({a:?}, {b:?}) produces {merged:?}, which is not in the vocab"
                ),
            });
        }
    }
    Ok(())
}

fn model(v: &Value, path: &str) -> Result<ModelConfig, TokenizerError> {
    let (t, o) = tag(v, path)?;
    match t {
        "BPE" => {
            // `dropout` non-null is a named exclusion (§12): train-time
            // stochastic augmentation, entangled with an RNG-policy
            // API; inference artifacts serialize null.
            if nullable(o, "dropout").is_some() {
                return Err(TokenizerError::UnsupportedField {
                    path: join(path, "dropout"),
                    why: "BPE dropout is train-time stochastic augmentation and is \
                          excluded from the inference surface; inference artifacts \
                          serialize null — re-export the tokenizer with dropout disabled"
                        .to_string(),
                });
            }
            // `cache_capacity` (a perf hint) falls under the ignored-
            // unknown-fields posture — accepted, no effect.
            let vocab = vocab_u32(req(o, "vocab", path)?, &join(path, "vocab"))?;
            let merges = merges(req(o, "merges", path)?, &join(path, "merges"))?;
            let continuing_subword_prefix = opt_string(o, "continuing_subword_prefix", path)?;
            check_merges(&vocab, &merges, continuing_subword_prefix.as_deref())?;
            Ok(ModelConfig::Bpe {
                vocab,
                merges,
                unk_token: opt_string(o, "unk_token", path)?,
                continuing_subword_prefix,
                end_of_word_suffix: opt_string(o, "end_of_word_suffix", path)?,
                fuse_unk: bool_or(o, "fuse_unk", path, false)?,
                byte_fallback: bool_or(o, "byte_fallback", path, false)?,
                ignore_merges: bool_or(o, "ignore_merges", path, false)?,
            })
        }
        "WordPiece" => Ok(ModelConfig::WordPiece {
            vocab: vocab_u32(req(o, "vocab", path)?, &join(path, "vocab"))?,
            unk_token: string_or(o, "unk_token", path, "[UNK]")?,
            continuing_subword_prefix: string_or(o, "continuing_subword_prefix", path, "##")?,
            max_input_chars_per_word: usize_or(o, "max_input_chars_per_word", path, 100)?,
        }),
        "Unigram" => {
            let vpath = join(path, "vocab");
            let rows = expect_array(req(o, "vocab", path)?, &vpath)?;
            let mut vocab = Vec::with_capacity(rows.len());
            let mut seen: HashSet<&str> = HashSet::with_capacity(rows.len());
            for (i, row) in rows.iter().enumerate() {
                let p = idx(&vpath, i);
                let cells = expect_array(row, &p)?;
                let [piece, logprob] = cells else {
                    return Err(schema(
                        &p,
                        format!(
                            "a Unigram vocab row is [piece, logprob], found {} cells",
                            cells.len()
                        ),
                    ));
                };
                let piece = expect_str(piece, &idx(&p, 0))?;
                let logprob = expect_f64(logprob, &idx(&p, 1))?;
                if !seen.insert(piece) {
                    return Err(TokenizerError::Vocab {
                        what: format!("duplicate piece {piece:?} in Unigram vocab"),
                    });
                }
                vocab.push((piece.to_string(), logprob));
            }
            let unk_id = nullable(o, "unk_id")
                .map(|v| expect_usize(v, &join(path, "unk_id")))
                .transpose()?;
            if let Some(u) = unk_id
                && u >= vocab.len()
            {
                return Err(TokenizerError::Vocab {
                    what: format!(
                        "Unigram unk_id {u} is out of range (the vocab holds {} pieces)",
                        vocab.len()
                    ),
                });
            }
            Ok(ModelConfig::Unigram {
                vocab,
                unk_id,
                byte_fallback: bool_or(o, "byte_fallback", path, false)?,
            })
        }
        "WordLevel" => Ok(ModelConfig::WordLevel {
            vocab: vocab_u32(req(o, "vocab", path)?, &join(path, "vocab"))?,
            unk_token: string_or(o, "unk_token", path, "<unk>")?,
        }),
        other => Err(TokenizerError::UnknownTag {
            family: "model",
            tag: other.to_string(),
        }),
    }
}

// ── Post-processors ─────────────────────────────────────────────────────

/// A `(token, id)` pair serialized as `["[SEP]", 102]`.
fn token_id_pair(v: &Value, path: &str) -> Result<(String, u32), TokenizerError> {
    let cells = expect_array(v, path)?;
    let [token, id] = cells else {
        return Err(schema(
            path,
            format!("expected [token, id], found {} cells", cells.len()),
        ));
    };
    Ok((
        expect_str(token, &idx(path, 0))?.to_string(),
        expect_u32(id, &idx(path, 1))?,
    ))
}

fn template_piece(v: &Value, path: &str) -> Result<TemplatePiece, TokenizerError> {
    let o = expect_object(v, path)?;
    let [(kind, inner)] = o else {
        return Err(schema(
            path,
            "a template piece is exactly one of {\"Sequence\": …} or {\"SpecialToken\": …}",
        ));
    };
    let p = join(path, kind);
    let type_id = u32_or(inner, "type_id", &p, 0)?;
    match kind.as_str() {
        "Sequence" => {
            let id = match expect_str(req(inner, "id", &p)?, &join(&p, "id"))? {
                "A" => SequenceId::A,
                "B" => SequenceId::B,
                other => {
                    return Err(schema(
                        &join(&p, "id"),
                        format!("unknown sequence id {other:?} (expected \"A\" or \"B\")"),
                    ));
                }
            };
            Ok(TemplatePiece::Sequence { id, type_id })
        }
        "SpecialToken" => Ok(TemplatePiece::SpecialToken {
            id: expect_str(req(inner, "id", &p)?, &join(&p, "id"))?.to_string(),
            type_id,
        }),
        other => Err(schema(
            path,
            format!(
                "unknown template piece kind {other:?} (expected \"Sequence\" or \
                 \"SpecialToken\")"
            ),
        )),
    }
}

fn template_pieces(v: &Value, path: &str) -> Result<Vec<TemplatePiece>, TokenizerError> {
    expect_array(v, path)?
        .iter()
        .enumerate()
        .map(|(i, v)| template_piece(v, &idx(path, i)))
        .collect()
}

fn post_processor(v: &Value, path: &str) -> Result<PostProcessorConfig, TokenizerError> {
    let (t, o) = tag(v, path)?;
    match t {
        "TemplateProcessing" => {
            let single = template_pieces(req(o, "single", path)?, &join(path, "single"))?;
            let pair = template_pieces(req(o, "pair", path)?, &join(path, "pair"))?;
            let st_path = join(path, "special_tokens");
            let mut special_tokens = Vec::new();
            if let Some(map) = nullable(o, "special_tokens") {
                for (name, entry) in expect_object(map, &st_path)? {
                    let p = format!("{st_path}[{name:?}]");
                    let id = expect_str(req(entry, "id", &p)?, &join(&p, "id"))?.to_string();
                    if id != *name {
                        return Err(schema(
                            &p,
                            format!("map key {name:?} disagrees with its entry id {id:?}"),
                        ));
                    }
                    let ids = expect_array(req(entry, "ids", &p)?, &join(&p, "ids"))?
                        .iter()
                        .enumerate()
                        .map(|(i, v)| expect_u32(v, &idx(&join(&p, "ids"), i)))
                        .collect::<Result<Vec<_>, _>>()?;
                    let tokens = expect_array(req(entry, "tokens", &p)?, &join(&p, "tokens"))?
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            expect_str(v, &idx(&join(&p, "tokens"), i)).map(str::to_string)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if ids.len() != tokens.len() {
                        return Err(schema(
                            &p,
                            format!(
                                "ids ({}) and tokens ({}) must be the same length",
                                ids.len(),
                                tokens.len()
                            ),
                        ));
                    }
                    special_tokens.push(SpecialTokenConfig { id, ids, tokens });
                }
            }
            // Every SpecialToken piece must name a declared special —
            // the reference validates this when it builds the
            // processor; holding an undeclared reference would defer a
            // load fault to encode time.
            let declared: HashSet<&str> = special_tokens.iter().map(|s| s.id.as_str()).collect();
            for (which, pieces) in [("single", &single), ("pair", &pair)] {
                for (i, piece) in pieces.iter().enumerate() {
                    if let TemplatePiece::SpecialToken { id, .. } = piece
                        && !declared.contains(id.as_str())
                    {
                        return Err(schema(
                            &idx(&join(path, which), i),
                            format!(
                                "template references special token {id:?}, which is not \
                                 declared in special_tokens"
                            ),
                        ));
                    }
                }
            }
            Ok(PostProcessorConfig::Template {
                single,
                pair,
                special_tokens,
            })
        }
        "BertProcessing" => Ok(PostProcessorConfig::Bert {
            sep: token_id_pair(req(o, "sep", path)?, &join(path, "sep"))?,
            cls: token_id_pair(req(o, "cls", path)?, &join(path, "cls"))?,
        }),
        "RobertaProcessing" => Ok(PostProcessorConfig::Roberta {
            sep: token_id_pair(req(o, "sep", path)?, &join(path, "sep"))?,
            cls: token_id_pair(req(o, "cls", path)?, &join(path, "cls"))?,
            trim_offsets: bool_or(o, "trim_offsets", path, true)?,
            add_prefix_space: bool_or(o, "add_prefix_space", path, true)?,
        }),
        "ByteLevel" => {
            let (add_prefix_space, trim_offsets, use_regex) = byte_level_fields(o, path)?;
            Ok(PostProcessorConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
            })
        }
        "Sequence" => {
            let arr = expect_array(req(o, "processors", path)?, &join(path, "processors"))?;
            let inner = arr
                .iter()
                .enumerate()
                .map(|(i, v)| post_processor(v, &idx(&join(path, "processors"), i)))
                .collect::<Result<_, _>>()?;
            Ok(PostProcessorConfig::Sequence(inner))
        }
        other => Err(TokenizerError::UnknownTag {
            family: "post_processor",
            tag: other.to_string(),
        }),
    }
}

// ── Decoders ────────────────────────────────────────────────────────────

fn decoder(v: &Value, path: &str) -> Result<DecoderConfig, TokenizerError> {
    let (t, o) = tag(v, path)?;
    match t {
        "ByteLevel" => {
            let (add_prefix_space, trim_offsets, use_regex) = byte_level_fields(o, path)?;
            Ok(DecoderConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex,
            })
        }
        "WordPiece" => Ok(DecoderConfig::WordPiece {
            prefix: string_or(o, "prefix", path, "##")?,
            cleanup: bool_or(o, "cleanup", path, true)?,
        }),
        "BPEDecoder" => Ok(DecoderConfig::BpeDecoder {
            suffix: string_or(o, "suffix", path, "</w>")?,
        }),
        "Metaspace" => {
            let (replacement, prepend_scheme, split) = metaspace_fields(o, path)?;
            Ok(DecoderConfig::Metaspace {
                replacement,
                prepend_scheme,
                split,
            })
        }
        "ByteFallback" => Ok(DecoderConfig::ByteFallback),
        "Fuse" => Ok(DecoderConfig::Fuse),
        "Strip" => Ok(DecoderConfig::Strip {
            content: match nullable(o, "content") {
                None => ' ',
                Some(v) => expect_char(v, &join(path, "content"))?,
            },
            start: usize_or(o, "start", path, 0)?,
            stop: usize_or(o, "stop", path, 0)?,
        }),
        "Replace" => Ok(DecoderConfig::Replace {
            pattern: pattern(req(o, "pattern", path)?, &join(path, "pattern"))?,
            content: expect_str(req(o, "content", path)?, &join(path, "content"))?.to_string(),
        }),
        "CTC" => Ok(DecoderConfig::Ctc {
            pad_token: string_or(o, "pad_token", path, "<pad>")?,
            word_delimiter_token: string_or(o, "word_delimiter_token", path, "|")?,
            cleanup: bool_or(o, "cleanup", path, true)?,
        }),
        "Sequence" => {
            let arr = expect_array(req(o, "decoders", path)?, &join(path, "decoders"))?;
            let inner = arr
                .iter()
                .enumerate()
                .map(|(i, v)| decoder(v, &idx(&join(path, "decoders"), i)))
                .collect::<Result<_, _>>()?;
            Ok(DecoderConfig::Sequence(inner))
        }
        other => Err(TokenizerError::UnknownTag {
            family: "decoder",
            tag: other.to_string(),
        }),
    }
}

// ── Added tokens ────────────────────────────────────────────────────────

fn added_tokens(v: &Value, path: &str) -> Result<Vec<AddedTokenConfig>, TokenizerError> {
    let arr = expect_array(v, path)?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, entry) in arr.iter().enumerate() {
        let p = idx(path, i);
        expect_object(entry, &p)?;
        let id_v = req(entry, "id", &p)?;
        let Some(wide) = id_v.as_number().and_then(json::Number::as_u64) else {
            return Err(schema(
                &join(&p, "id"),
                format!("expected an unsigned integer id, found {}", id_v.kind()),
            ));
        };
        let id = u32::try_from(wide).map_err(|_| TokenizerError::Vocab {
            what: format!(
                "added token [{i}] has id {wide}, which does not fit in u32 (ids are \
                 u32 by format contract)"
            ),
        })?;
        let content = expect_str(req(entry, "content", &p)?, &join(&p, "content"))?.to_string();
        let special = bool_or(entry, "special", &p, false)?;
        let normalized = match nullable(entry, "normalized") {
            // The reference builder's default: specials match raw text.
            None => !special,
            Some(v) => expect_bool(v, &join(&p, "normalized"))?,
        };
        out.push(AddedTokenConfig {
            id,
            content,
            single_word: bool_or(entry, "single_word", &p, false)?,
            lstrip: bool_or(entry, "lstrip", &p, false)?,
            rstrip: bool_or(entry, "rstrip", &p, false)?,
            normalized,
            special,
        });
    }
    Ok(out)
}

// ── Truncation / padding ────────────────────────────────────────────────

fn truncation_config(v: &Value, path: &str) -> Result<TruncationConfig, TokenizerError> {
    expect_object(v, path)?;
    let strategy = match nullable(v, "strategy") {
        None => TruncationStrategy::LongestFirst,
        Some(s) => match expect_str(s, &join(path, "strategy"))? {
            "LongestFirst" => TruncationStrategy::LongestFirst,
            "OnlyFirst" => TruncationStrategy::OnlyFirst,
            "OnlySecond" => TruncationStrategy::OnlySecond,
            other => {
                return Err(schema(
                    &join(path, "strategy"),
                    format!(
                        "unknown truncation strategy {other:?} (expected \
                         \"LongestFirst\", \"OnlyFirst\", or \"OnlySecond\")"
                    ),
                ));
            }
        },
    };
    Ok(TruncationConfig {
        direction: direction_or(v, "direction", path)?,
        max_length: usize_or(v, "max_length", path, 512)?,
        strategy,
        stride: usize_or(v, "stride", path, 0)?,
    })
}

fn padding_config(v: &Value, path: &str) -> Result<PaddingConfig, TokenizerError> {
    expect_object(v, path)?;
    let strategy = match nullable(v, "strategy") {
        None => PaddingStrategy::BatchLongest,
        Some(Value::String(s)) if s == "BatchLongest" => PaddingStrategy::BatchLongest,
        Some(s @ Value::Object(_)) => {
            let fixed = req(s, "Fixed", &join(path, "strategy"))?;
            PaddingStrategy::Fixed(expect_usize(
                fixed,
                &join(&join(path, "strategy"), "Fixed"),
            )?)
        }
        Some(other) => {
            return Err(schema(
                &join(path, "strategy"),
                format!(
                    "unknown padding strategy {} (expected \"BatchLongest\" or \
                     {{\"Fixed\": n}})",
                    match other {
                        Value::String(s) => format!("{s:?}"),
                        v => v.kind().to_string(),
                    }
                ),
            ));
        }
    };
    Ok(PaddingConfig {
        strategy,
        direction: direction_or(v, "direction", path)?,
        pad_to_multiple_of: nullable(v, "pad_to_multiple_of")
            .map(|n| expect_usize(n, &join(path, "pad_to_multiple_of")))
            .transpose()?,
        pad_id: u32_or(v, "pad_id", path, 0)?,
        pad_type_id: u32_or(v, "pad_type_id", path, 0)?,
        pad_token: string_or(v, "pad_token", path, "[PAD]")?,
    })
}

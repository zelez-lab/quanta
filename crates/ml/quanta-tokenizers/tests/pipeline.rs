//! Pipeline-lane tests: alignment correctness through stacked
//! normalizers, every pre-tokenizer's splits and offsets, template /
//! Bert / Roberta / ByteLevel post-processing, truncation with stride
//! and overflow, padding, decoders, and the streaming decode — all
//! against a MOCK model (the four model runtimes live behind the
//! `Model` seam and are integrated by their own lanes; the conformance
//! fixtures land with the integrator).
//!
//! Reference vectors cited inline come from HF tokenizers 0.21.x's own
//! unit tests (`pre_tokenizers/*.rs`, `tokenizer/normalizer.rs`).

use quanta_tokenizers::artifact::{
    AddedTokenConfig, DecoderConfig, Direction, ModelConfig, NormalizerConfig, PaddingConfig,
    PaddingStrategy, PatternConfig, PostProcessorConfig, PreTokenizerConfig, PrependScheme,
    SequenceId, SpecialTokenConfig, SplitBehavior, TemplatePiece, TokenizerArtifact,
    TruncationConfig, TruncationStrategy,
};
use quanta_tokenizers::model::{Model, ModelToken};
use quanta_tokenizers::normalize::Normalizer;
use quanta_tokenizers::normalized::NormalizedString;
use quanta_tokenizers::pretokenize::{PreTokenizedString, PreTokenizer};
use quanta_tokenizers::{Encoding, Tokenizer, TokenizerError};
use std::collections::HashMap;

// ── The mock model ──────────────────────────────────────────────────────

/// A lookup mock: a pretoken found in the vocab becomes ONE token; any
/// other pretoken falls back to per-char tokens (unknown chars get id
/// 0). Offsets are byte offsets into the pretoken, per the seam.
struct MockModel {
    vocab: Vec<(String, u32)>,
    map: HashMap<String, u32>,
}

impl MockModel {
    fn new(entries: &[(&str, u32)]) -> Self {
        let vocab: Vec<(String, u32)> = entries.iter().map(|(s, i)| (s.to_string(), *i)).collect();
        let map = vocab.iter().cloned().collect();
        MockModel { vocab, map }
    }
}

impl Model for MockModel {
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError> {
        if let Some(&id) = self.map.get(pretoken) {
            return Ok(vec![ModelToken {
                id,
                value: pretoken.to_string(),
                offsets: (0, pretoken.len()),
            }]);
        }
        Ok(pretoken
            .char_indices()
            .map(|(i, c)| ModelToken {
                id: self.map.get(c.to_string().as_str()).copied().unwrap_or(0),
                value: c.to_string(),
                offsets: (i, i + c.len_utf8()),
            })
            .collect())
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab
            .iter()
            .find(|(_, i)| *i == id)
            .map(|(s, _)| s.as_str())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.map.get(token).copied()
    }
}

/// A bare artifact (WordLevel dummy in the model slot — the mock
/// bypasses `build_model`).
fn bare_artifact() -> TokenizerArtifact {
    TokenizerArtifact {
        version: Some("1.0".to_string()),
        truncation: None,
        padding: None,
        added_tokens: Vec::new(),
        normalizer: None,
        pre_tokenizer: None,
        model: ModelConfig::WordLevel {
            vocab: Vec::new(),
            unk_token: "<unk>".to_string(),
        },
        post_processor: None,
        decoder: None,
    }
}

fn tokenizer(artifact: &TokenizerArtifact, mock: MockModel) -> Tokenizer {
    Tokenizer::from_artifact(artifact, Box::new(mock)).expect("artifact compiles")
}

/// Runs one pre-tokenizer config over `text` and returns
/// `(split text, absolute original offsets)` rows.
fn pretok_splits(config: &PreTokenizerConfig, text: &str) -> Vec<(String, (usize, usize))> {
    let pretok = PreTokenizer::compile(config).expect("config compiles");
    let mut pts = PreTokenizedString::from(text);
    pretok.apply(&mut pts).expect("pre-tokenization succeeds");
    pts.splits()
        .into_iter()
        .map(|(s, offsets, _)| (s.to_string(), offsets))
        .collect()
}

fn owned(rows: &[(&str, (usize, usize))]) -> Vec<(String, (usize, usize))> {
    rows.iter().map(|(s, o)| (s.to_string(), *o)).collect()
}

// ── The alignment core ──────────────────────────────────────────────────

#[test]
fn nfd_tracks_decomposition_back_to_the_original() {
    let mut n = NormalizedString::from("Café");
    n.nfd();
    assert_eq!(n.get(), "Cafe\u{301}");
    // Both the base letter and the combining mark descend from é's
    // original bytes 3..5.
    assert_eq!(n.convert_offsets(3..4), Some(3..5));
    assert_eq!(n.convert_offsets(4..6), Some(3..5));
    assert_eq!(n.convert_offsets(0..1), Some(0..1));
}

#[test]
fn nfc_composes_and_keeps_the_starter_alignment() {
    let mut n = NormalizedString::from("Cafe\u{301}");
    n.nfc();
    assert_eq!(n.get(), "Café");
    // The composed char takes the STARTER's alignment — the reference's
    // transform semantics for a (char, -1) composition entry.
    assert_eq!(n.convert_offsets(3..5), Some(3..4));
}

#[test]
fn nfkc_expansion_shares_the_source_alignment() {
    // ﬁ (U+FB01, 3 bytes) expands to "fi"; both chars align to it —
    // the reference's own `test_nfkc` vector.
    let mut n = NormalizedString::from("\u{fb01}");
    n.nfkc();
    assert_eq!(n.get(), "fi");
    assert_eq!(n.convert_offsets(0..1), Some(0..3));
    assert_eq!(n.convert_offsets(1..2), Some(0..3));
}

#[test]
fn stacked_normalizers_map_offsets_into_the_original_text() {
    // NFD → BertNormalizer-style strip (drop Mn) → lowercase, then a
    // whitespace pre-tokenizer: token offsets land in the ORIGINAL.
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Sequence(vec![
        NormalizerConfig::Nfd,
        NormalizerConfig::StripAccents,
        NormalizerConfig::Lowercase,
    ]));
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    let tok = tokenizer(&artifact, MockModel::new(&[("cafe", 1), ("menu", 2)]));
    let enc = tok.encode("Café Menü", true).unwrap();
    assert_eq!(enc.tokens(), ["cafe", "menu"]);
    assert_eq!(enc.ids(), [1, 2]);
    // "Café" is bytes 0..5 (é is 2 bytes), "Menü" is 6..11.
    assert_eq!(enc.offsets(), [(0, 5), (6, 11)]);
    assert_eq!(enc.word_ids(), [Some(0), Some(1)]);
}

#[test]
fn strip_normalizer_drops_edges_from_the_offsets() {
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Strip {
        strip_left: true,
        strip_right: true,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("hi", 5)]));
    let enc = tok.encode("  hi  ", true).unwrap();
    assert_eq!(enc.tokens(), ["hi"]);
    assert_eq!(enc.offsets(), [(2, 4)]);
}

#[test]
fn prepend_normalizer_rides_on_the_first_char() {
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Prepend {
        prepend: "\u{2581}".to_string(),
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("\u{2581}Hey", 3)]));
    let enc = tok.encode("Hey", true).unwrap();
    assert_eq!(enc.tokens(), ["\u{2581}Hey"]);
    // The prepended char has no original bytes of its own: the token
    // still spans exactly the original text.
    assert_eq!(enc.offsets(), [(0, 3)]);
}

#[test]
fn replace_regex_normalizer_keeps_alignment() {
    // The Llama-style whitespace fold: runs of whitespace become one
    // space; the replacement aligns to the run's last char.
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Replace {
        pattern: PatternConfig::Regex(r"\s+".to_string()),
        content: " ".to_string(),
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("a b", 9)]));
    let enc = tok.encode("a   b", true).unwrap();
    assert_eq!(enc.tokens(), ["a b"]);
    assert_eq!(enc.offsets(), [(0, 5)]);
}

#[test]
fn bert_normalizer_null_strip_accents_follows_lowercase() {
    let strip_on = |lowercase: bool| {
        let config = NormalizerConfig::Bert {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase,
        };
        let n = Normalizer::compile(&config).unwrap();
        let mut s = NormalizedString::from("é");
        n.apply(&mut s).unwrap();
        s.get().to_string()
    };
    // lowercase=true → null strip_accents means ON.
    assert_eq!(strip_on(true), "e");
    // lowercase=false → null strip_accents means OFF (é survives NFC-
    // composed as typed).
    assert_eq!(strip_on(false), "é");
}

#[test]
fn bert_chinese_chars_get_space_padding() {
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Bert {
        clean_text: true,
        handle_chinese_chars: true,
        strip_accents: None,
        lowercase: false,
    });
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Bert);
    let tok = tokenizer(
        &artifact,
        MockModel::new(&[("ab", 1), ("你", 2), ("cd", 3)]),
    );
    let enc = tok.encode("ab你cd", true).unwrap();
    assert_eq!(enc.tokens(), ["ab", "你", "cd"]);
    assert_eq!(enc.offsets(), [(0, 2), (2, 5), (5, 7)]);
}

#[test]
fn standalone_strip_accents_drops_marks_without_decomposition() {
    // The reference's StripAccents filters General_Category=Mark with
    // NO NFD: a precomposed é passes through untouched, a combining
    // mark is dropped.
    let n = Normalizer::compile(&NormalizerConfig::StripAccents).unwrap();
    let mut precomposed = NormalizedString::from("é");
    n.apply(&mut precomposed).unwrap();
    assert_eq!(precomposed.get(), "é");
    let mut decomposed = NormalizedString::from("e\u{301}");
    n.apply(&mut decomposed).unwrap();
    assert_eq!(decomposed.get(), "e");
}

#[test]
fn nmt_filters_and_folds_its_codepoint_lists() {
    let n = Normalizer::compile(&NormalizerConfig::Nmt).unwrap();
    let mut s = NormalizedString::from("a\u{0001}b\u{200B}c");
    n.apply(&mut s).unwrap();
    assert_eq!(s.get(), "ab c");
}

#[test]
fn hangul_composes_and_decomposes_with_alignment() {
    let mut n = NormalizedString::from("\u{1100}\u{1161}\u{11A8}");
    n.nfc();
    assert_eq!(n.get(), "\u{AC01}");
    // The composed syllable consumes all three jamo: its span covers
    // the starter's original bytes per the transform convention.
    let mut back = NormalizedString::from("\u{AC01}");
    back.nfd();
    assert_eq!(back.get(), "\u{1100}\u{1161}\u{11A8}");
    // Every decomposed jamo descends from the original syllable.
    assert_eq!(back.convert_offsets(0..3), Some(0..3));
    assert_eq!(back.convert_offsets(3..6), Some(0..3));
    assert_eq!(back.convert_offsets(6..9), Some(0..3));
}

#[test]
fn lowercase_expansion_rides_on_the_source_char() {
    // İ (U+0130) lowercases to i + combining dot above.
    let mut n = NormalizedString::from("İX");
    n.lowercase();
    assert_eq!(n.get(), "i\u{307}x");
    assert_eq!(n.convert_offsets(0..1), Some(0..2));
    assert_eq!(n.convert_offsets(1..3), Some(0..2));
    assert_eq!(n.convert_offsets(3..4), Some(2..3));
}

#[test]
fn byte_level_normalizer_form_transforms_without_prefix_or_split() {
    let n = Normalizer::compile(&NormalizerConfig::ByteLevel).unwrap();
    let mut s = NormalizedString::from("a é");
    n.apply(&mut s).unwrap();
    assert_eq!(s.get(), "aĠÃ©");
    // The two stand-ins of é both descend from its original bytes.
    let g_end = 1 + 'Ġ'.len_utf8();
    assert_eq!(s.convert_offsets(g_end..s.len()), Some(2..4));
}

#[test]
fn bert_clean_text_drops_controls_and_folds_whitespace() {
    let n = Normalizer::compile(&NormalizerConfig::Bert {
        clean_text: true,
        handle_chinese_chars: false,
        strip_accents: Some(false),
        lowercase: false,
    })
    .unwrap();
    // NUL and U+FFFD are dropped outright; \t folds to a space.
    let mut s = NormalizedString::from("a\u{0000}b\tc\u{FFFD}d");
    n.apply(&mut s).unwrap();
    assert_eq!(s.get(), "ab cd");
}

// ── Precompiled (sentencepiece charsmap) ────────────────────────────────

/// Encodes a darts-clone unit: label in bits 0-7, has-leaf in bit 8,
/// offset in bits 10.. — the same builder as the unicode-lane tests.
fn trie_unit(label: u8, has_leaf: bool, offset: usize) -> u32 {
    assert!(offset < (1 << 21));
    ((offset as u32) << 10) | (u32::from(has_leaf) << 8) | u32::from(label)
}

/// A tiny charsmap blob: "A" → "x" and "e\u{301}" → "E".
fn charsmap_blob() -> Vec<u8> {
    let mut units = vec![0u32; 1024];
    units[0] = trie_unit(0, false, 256);
    units[256 ^ 0x41] = trie_unit(0x41, true, (256 ^ 0x41) ^ 700);
    units[700] = 0; // pool offset of "x"
    units[256 ^ 0x65] = trie_unit(0x65, false, (256 ^ 0x65) ^ 512);
    units[512 ^ 0xCC] = trie_unit(0xCC, false, (512 ^ 0xCC) ^ 64);
    units[64 ^ 0x81] = trie_unit(0x81, true, (64 ^ 0x81) ^ 900);
    units[900] = 2; // pool offset of "E"
    let pool = b"x\0E\0";
    let mut blob = Vec::new();
    blob.extend_from_slice(&u32::try_from(units.len() * 4).unwrap().to_le_bytes());
    for u in units {
        blob.extend_from_slice(&u.to_le_bytes());
    }
    blob.extend_from_slice(pool);
    blob
}

#[test]
fn precompiled_charsmap_normalizes_with_alignment() {
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Precompiled {
        charsmap: charsmap_blob(),
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("ZE!", 4)]));
    // "e\u{301}" is one 3-byte grapheme → whole-cluster hit → "E";
    // the token still spans the whole original.
    let enc = tok.encode("Ze\u{301}!", true).unwrap();
    assert_eq!(enc.tokens(), ["ZE!"]);
    assert_eq!(enc.offsets(), [(0, 5)]);
}

// ── Pre-tokenizers: splits and offsets ──────────────────────────────────

#[test]
fn whitespace_matches_the_reference_vector() {
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::Whitespace, "Hey man!"),
        owned(&[("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))])
    );
}

#[test]
fn whitespace_split_keeps_punctuation_attached() {
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::WhitespaceSplit, "Hey man!"),
        owned(&[("Hey", (0, 3)), ("man!", (4, 8))])
    );
}

#[test]
fn bert_pre_tokenizer_matches_the_reference_vector() {
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::Bert, "Hey friend!     How are you?!?"),
        owned(&[
            ("Hey", (0, 3)),
            ("friend", (4, 10)),
            ("!", (10, 11)),
            ("How", (16, 19)),
            ("are", (20, 23)),
            ("you", (24, 27)),
            ("?", (27, 28)),
            ("!", (28, 29)),
            ("?", (29, 30)),
        ])
    );
}

#[test]
fn punctuation_isolates_each_char() {
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Punctuation {
                behavior: SplitBehavior::Isolated
            },
            "you?!?"
        ),
        owned(&[("you", (0, 3)), ("?", (3, 4)), ("!", (4, 5)), ("?", (5, 6))])
    );
}

#[test]
fn digits_contiguous_and_individual() {
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Digits {
                individual_digits: false
            },
            "Hey 123 friend"
        ),
        owned(&[("Hey ", (0, 4)), ("123", (4, 7)), (" friend", (7, 14))])
    );
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Digits {
                individual_digits: true
            },
            "a12b"
        ),
        owned(&[("a", (0, 1)), ("1", (1, 2)), ("2", (2, 3)), ("b", (3, 4))])
    );
}

#[test]
fn char_delimiter_split_removes_the_delimiter() {
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::CharDelimiterSplit { delimiter: 'x' },
            "axbxc"
        ),
        owned(&[("a", (0, 1)), ("b", (2, 3)), ("c", (4, 5))])
    );
}

#[test]
fn metaspace_replaces_and_prepends_with_original_offsets() {
    // The classic vector: "Hey friend" → ▁Hey (0,3) + ▁friend (3,10) —
    // the prepended ▁ has no original bytes, the replaced one covers
    // the space.
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Metaspace {
                replacement: '\u{2581}',
                prepend_scheme: PrependScheme::Always,
                split: true,
            },
            "Hey friend"
        ),
        owned(&[("\u{2581}Hey", (0, 3)), ("\u{2581}friend", (3, 10))])
    );
    // Never: no prepend on the head word.
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Metaspace {
                replacement: '\u{2581}',
                prepend_scheme: PrependScheme::Never,
                split: true,
            },
            "Hey friend"
        ),
        owned(&[("Hey", (0, 3)), ("\u{2581}friend", (3, 10))])
    );
    // split=false: one piece, spaces replaced in place.
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::Metaspace {
                replacement: '\u{2581}',
                prepend_scheme: PrependScheme::Never,
                split: false,
            },
            "Hey friend"
        ),
        owned(&[("Hey\u{2581}friend", (0, 10))])
    );
}

#[test]
fn metaspace_first_prepends_only_at_the_text_start() {
    // Behind a whitespace splitter, only the split anchored at
    // original offset 0 gets the prepend under `First`.
    let config = PreTokenizerConfig::Sequence(vec![
        PreTokenizerConfig::WhitespaceSplit,
        PreTokenizerConfig::Metaspace {
            replacement: '\u{2581}',
            prepend_scheme: PrependScheme::First,
            split: true,
        },
    ]);
    assert_eq!(
        pretok_splits(&config, "Hey friend"),
        owned(&[("\u{2581}Hey", (0, 3)), ("friend", (4, 10))])
    );
}

#[test]
fn split_behaviors_match_the_reference_table() {
    // The SplitDelimiterBehavior doc table, input "the-final--countdown".
    let case = |behavior, invert, expected: &[(&str, (usize, usize))]| {
        let config = PreTokenizerConfig::Split {
            pattern: PatternConfig::String("-".to_string()),
            behavior,
            invert,
        };
        assert_eq!(
            pretok_splits(&config, "the-final--countdown"),
            owned(expected)
        );
    };
    case(
        SplitBehavior::Removed,
        false,
        &[("the", (0, 3)), ("final", (4, 9)), ("countdown", (11, 20))],
    );
    case(
        SplitBehavior::Isolated,
        false,
        &[
            ("the", (0, 3)),
            ("-", (3, 4)),
            ("final", (4, 9)),
            ("-", (9, 10)),
            ("-", (10, 11)),
            ("countdown", (11, 20)),
        ],
    );
    case(
        SplitBehavior::MergedWithPrevious,
        false,
        &[
            ("the-", (0, 4)),
            ("final-", (4, 10)),
            ("-", (10, 11)),
            ("countdown", (11, 20)),
        ],
    );
    case(
        SplitBehavior::MergedWithNext,
        false,
        &[
            ("the", (0, 3)),
            ("-final", (3, 9)),
            ("-", (9, 10)),
            ("-countdown", (10, 20)),
        ],
    );
    case(
        SplitBehavior::Contiguous,
        false,
        &[
            ("the", (0, 3)),
            ("-", (3, 4)),
            ("final", (4, 9)),
            ("--", (9, 11)),
            ("countdown", (11, 20)),
        ],
    );
}

#[test]
fn split_invert_swaps_matches_and_gaps() {
    let config = PreTokenizerConfig::Split {
        pattern: PatternConfig::Regex(r"\w+".to_string()),
        behavior: SplitBehavior::Removed,
        invert: true,
    };
    // Inverted, the word runs are the GAPS: removing matches keeps
    // them.
    assert_eq!(
        pretok_splits(&config, "the-final--countdown"),
        owned(&[("the", (0, 3)), ("final", (4, 9)), ("countdown", (11, 20))])
    );
}

#[test]
fn split_regex_pattern_rejects_out_of_set_constructs() {
    let config = PreTokenizerConfig::Split {
        pattern: PatternConfig::Regex(r"(?=x)y".to_string()),
        behavior: SplitBehavior::Removed,
        invert: false,
    };
    let err = PreTokenizer::compile(&config).unwrap_err();
    assert!(matches!(err, TokenizerError::RegexConstruct { .. }));
    assert!(err.to_string().contains("lookahead"));
}

#[test]
fn unicode_scripts_matches_the_reference_vectors() {
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::UnicodeScripts, "どこで生れ。Yes"),
        owned(&[("どこで生れ", (0, 15)), ("。", (15, 18)), ("Yes", (18, 21)),])
    );
    // Spaces glue to the surrounding script run.
    assert_eq!(
        pretok_splits(
            &PreTokenizerConfig::UnicodeScripts,
            "Apples are りんご 林檎"
        ),
        owned(&[("Apples are ", (0, 11)), ("りんご 林檎", (11, 27))])
    );
}

#[test]
fn fixed_length_chunks_by_chars() {
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::FixedLength { length: 3 }, "abcdefgh"),
        owned(&[("abc", (0, 3)), ("def", (3, 6)), ("gh", (6, 8))])
    );
    // Multibyte chars count as one.
    assert_eq!(
        pretok_splits(&PreTokenizerConfig::FixedLength { length: 2 }, "héllo"),
        owned(&[("hé", (0, 3)), ("ll", (3, 5)), ("o", (5, 6))])
    );
}

#[test]
fn byte_level_pretokenizer_bijects_and_offsets() {
    let config = PreTokenizerConfig::ByteLevel {
        add_prefix_space: true,
        trim_offsets: true,
        use_regex: true,
    };
    // 'é' (2 bytes) becomes two stand-in chars; the prepended space
    // becomes Ġ with no original bytes of its own.
    assert_eq!(
        pretok_splits(&config, "Hello té"),
        owned(&[("ĠHello", (0, 5)), ("ĠtÃ©", (5, 9))])
    );
}

#[test]
fn sequence_pre_tokenizer_composes_in_order() {
    let config = PreTokenizerConfig::Sequence(vec![
        PreTokenizerConfig::WhitespaceSplit,
        PreTokenizerConfig::Punctuation {
            behavior: SplitBehavior::Isolated,
        },
    ]);
    assert_eq!(
        pretok_splits(&config, "Hey man!"),
        owned(&[("Hey", (0, 3)), ("man", (4, 7)), ("!", (7, 8))])
    );
}

// ── Added tokens ────────────────────────────────────────────────────────

fn added(content: &str, id: u32, special: bool) -> AddedTokenConfig {
    AddedTokenConfig {
        id,
        content: content.to_string(),
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: !special,
        special,
    }
}

#[test]
fn special_tokens_split_the_raw_text_before_the_model() {
    let mut artifact = bare_artifact();
    artifact.added_tokens = vec![added("<|endoftext|>", 50256, true)];
    let tok = tokenizer(&artifact, MockModel::new(&[("hi", 1), ("there", 2)]));
    let enc = tok.encode("hi<|endoftext|>there", true).unwrap();
    assert_eq!(enc.tokens(), ["hi", "<|endoftext|>", "there"]);
    assert_eq!(enc.ids(), [1, 50256, 2]);
    assert_eq!(enc.offsets(), [(0, 2), (2, 15), (15, 20)]);
    // Tokens matched IN the text are not marked special (only
    // post-processor insertions are) — reference semantics.
    assert_eq!(enc.special_tokens_mask(), [0, 0, 0]);
    assert_eq!(enc.word_ids(), [Some(0), Some(1), Some(2)]);
}

#[test]
fn lstrip_rstrip_extend_over_whitespace() {
    let mut artifact = bare_artifact();
    artifact.added_tokens = vec![AddedTokenConfig {
        id: 9,
        content: "<s>".to_string(),
        single_word: false,
        lstrip: true,
        rstrip: true,
        normalized: false,
        special: true,
    }];
    let tok = tokenizer(&artifact, MockModel::new(&[("a", 1), ("b", 2)]));
    let enc = tok.encode("a <s> b", true).unwrap();
    // The token VALUE carries the stripped spaces (reference:
    // the split's text is the token text).
    assert_eq!(enc.tokens(), ["a", " <s> ", "b"]);
    assert_eq!(enc.ids(), [1, 9, 2]);
    assert_eq!(enc.offsets(), [(0, 1), (1, 6), (6, 7)]);
}

#[test]
fn single_word_discards_embedded_matches() {
    let mut artifact = bare_artifact();
    artifact.added_tokens = vec![AddedTokenConfig {
        id: 7,
        content: "ab".to_string(),
        single_word: true,
        lstrip: false,
        rstrip: false,
        normalized: false,
        special: false,
    }];
    let tok = tokenizer(&artifact, MockModel::new(&[("abc ", 1)]));
    let enc = tok.encode("abc ab", true).unwrap();
    assert_eq!(enc.tokens(), ["abc ", "ab"]);
    assert_eq!(enc.ids(), [1, 7]);
    assert_eq!(enc.offsets(), [(0, 4), (4, 6)]);
}

#[test]
fn normalized_added_tokens_match_after_the_normalizer() {
    let mut artifact = bare_artifact();
    artifact.normalizer = Some(NormalizerConfig::Lowercase);
    artifact.added_tokens = vec![AddedTokenConfig {
        id: 42,
        content: "Day".to_string(),
        single_word: false,
        lstrip: false,
        rstrip: false,
        normalized: true,
        special: false,
    }];
    let tok = tokenizer(&artifact, MockModel::new(&[("sunny ", 1)]));
    let enc = tok.encode("sunny DAY", true).unwrap();
    // Content "Day" normalizes to "day"; the lowercased text matches
    // at 6..9 and the offsets point into the ORIGINAL "DAY".
    assert_eq!(enc.tokens(), ["sunny ", "day"]);
    assert_eq!(enc.ids(), [1, 42]);
    assert_eq!(enc.offsets(), [(0, 6), (6, 9)]);
}

// ── Post-processing ─────────────────────────────────────────────────────

fn cls_sep_template() -> PostProcessorConfig {
    PostProcessorConfig::Template {
        single: vec![
            TemplatePiece::SpecialToken {
                id: "[CLS]".to_string(),
                type_id: 0,
            },
            TemplatePiece::Sequence {
                id: SequenceId::A,
                type_id: 0,
            },
            TemplatePiece::SpecialToken {
                id: "[SEP]".to_string(),
                type_id: 0,
            },
        ],
        pair: vec![
            TemplatePiece::SpecialToken {
                id: "[CLS]".to_string(),
                type_id: 0,
            },
            TemplatePiece::Sequence {
                id: SequenceId::A,
                type_id: 0,
            },
            TemplatePiece::SpecialToken {
                id: "[SEP]".to_string(),
                type_id: 0,
            },
            TemplatePiece::Sequence {
                id: SequenceId::B,
                type_id: 1,
            },
            TemplatePiece::SpecialToken {
                id: "[SEP]".to_string(),
                type_id: 1,
            },
        ],
        special_tokens: vec![
            SpecialTokenConfig {
                id: "[CLS]".to_string(),
                ids: vec![101],
                tokens: vec!["[CLS]".to_string()],
            },
            SpecialTokenConfig {
                id: "[SEP]".to_string(),
                ids: vec![102],
                tokens: vec!["[SEP]".to_string()],
            },
        ],
    }
}

#[test]
fn template_processing_single_and_pair() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.post_processor = Some(cls_sep_template());
    let tok = tokenizer(&artifact, MockModel::new(&[("hello", 1), ("world", 2)]));

    let enc = tok.encode("hello world", true).unwrap();
    assert_eq!(enc.ids(), [101, 1, 2, 102]);
    assert_eq!(enc.tokens(), ["[CLS]", "hello", "world", "[SEP]"]);
    assert_eq!(enc.type_ids(), [0, 0, 0, 0]);
    assert_eq!(enc.special_tokens_mask(), [1, 0, 0, 1]);
    assert_eq!(enc.offsets(), [(0, 0), (0, 5), (6, 11), (0, 0)]);
    assert_eq!(enc.word_ids(), [None, Some(0), Some(1), None]);
    assert_eq!(enc.attention_mask(), [1, 1, 1, 1]);

    // Without specials the template contributes nothing.
    let bare = tok.encode("hello world", false).unwrap();
    assert_eq!(bare.ids(), [1, 2]);

    let pair = tok.encode_pair("hello", "world", true).unwrap();
    assert_eq!(pair.ids(), [101, 1, 102, 2, 102]);
    assert_eq!(pair.type_ids(), [0, 0, 0, 1, 1]);
    assert_eq!(pair.special_tokens_mask(), [1, 0, 1, 0, 1]);
    // Pair offsets index each side's OWN original text.
    assert_eq!(pair.offsets(), [(0, 0), (0, 5), (0, 0), (0, 5), (0, 0)]);
}

#[test]
fn bert_processing_wraps_single_and_pair() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.post_processor = Some(PostProcessorConfig::Bert {
        sep: ("[SEP]".to_string(), 102),
        cls: ("[CLS]".to_string(), 101),
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("a", 1), ("b", 2)]));
    let enc = tok.encode("a", true).unwrap();
    assert_eq!(enc.ids(), [101, 1, 102]);
    assert_eq!(enc.type_ids(), [0, 0, 0]);
    let pair = tok.encode_pair("a", "b", true).unwrap();
    assert_eq!(pair.ids(), [101, 1, 102, 2, 102]);
    assert_eq!(pair.type_ids(), [0, 0, 0, 1, 1]);
    assert_eq!(pair.special_tokens_mask(), [1, 0, 1, 0, 1]);
}

#[test]
fn roberta_processing_double_sep_and_zero_type_ids() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.post_processor = Some(PostProcessorConfig::Roberta {
        sep: ("</s>".to_string(), 2),
        cls: ("<s>".to_string(), 0),
        trim_offsets: true,
        add_prefix_space: true,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("a", 10), ("b", 11)]));
    let pair = tok.encode_pair("a", "b", true).unwrap();
    assert_eq!(pair.ids(), [0, 10, 2, 2, 11, 2]);
    // RoBERTa uses no segment ids at all.
    assert_eq!(pair.type_ids(), [0, 0, 0, 0, 0, 0]);
    assert_eq!(pair.special_tokens_mask(), [1, 0, 1, 1, 0, 1]);
}

#[test]
fn byte_level_trim_offsets_shrinks_space_prefixes() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::ByteLevel {
        add_prefix_space: true,
        trim_offsets: true,
        use_regex: true,
    });
    artifact.post_processor = Some(PostProcessorConfig::ByteLevel {
        add_prefix_space: true,
        trim_offsets: true,
        use_regex: true,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("ĠHello", 1), ("Ġworld", 2)]));
    let enc = tok.encode("Hello world", true).unwrap();
    assert_eq!(enc.tokens(), ["ĠHello", "Ġworld"]);
    // First token: its Ġ is the pipeline's own prefix — kept. Second:
    // the Ġ covers the real space at byte 5 — trimmed off.
    assert_eq!(enc.offsets(), [(0, 5), (6, 11)]);
}

// ── Truncation / padding ────────────────────────────────────────────────

/// Ten single-char tokens via the per-char mock fallback.
fn ten_tokens(truncation: Option<TruncationConfig>) -> Encoding {
    let mut artifact = bare_artifact();
    artifact.truncation = truncation;
    let tok = tokenizer(&artifact, MockModel::new(&[]));
    tok.encode("abcdefghij", true).unwrap()
}

#[test]
fn truncation_right_with_stride_builds_overlapping_windows() {
    let enc = ten_tokens(Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 6,
        strategy: TruncationStrategy::LongestFirst,
        stride: 2,
    }));
    let texts = |e: &Encoding| e.tokens().join("");
    assert_eq!(texts(&enc), "abcdef");
    assert_eq!(enc.overflowing().len(), 1);
    assert_eq!(texts(&enc.overflowing()[0]), "efghij");
    // Offsets survive into the windows.
    assert_eq!(enc.overflowing()[0].offsets()[0], (4, 5));
}

#[test]
fn truncation_left_keeps_the_tail() {
    let enc = ten_tokens(Some(TruncationConfig {
        direction: Direction::Left,
        max_length: 6,
        strategy: TruncationStrategy::LongestFirst,
        stride: 2,
    }));
    assert_eq!(enc.tokens().join(""), "efghij");
    assert_eq!(enc.overflowing().len(), 1);
    assert_eq!(enc.overflowing()[0].tokens().join(""), "abcdef");
}

#[test]
fn truncation_stride_must_stay_below_max_len() {
    let mut artifact = bare_artifact();
    artifact.truncation = Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 4,
        strategy: TruncationStrategy::LongestFirst,
        stride: 4,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[]));
    let err = tok.encode("abcdefghij", true).unwrap_err();
    assert!(matches!(err, TokenizerError::Encode { .. }));
    assert!(err.to_string().contains("stride"));
}

#[test]
fn longest_first_pair_splits_the_budget() {
    let run = |a: &str, b: &str, max: usize| {
        let mut artifact = bare_artifact();
        artifact.truncation = Some(TruncationConfig {
            direction: Direction::Right,
            max_length: max,
            strategy: TruncationStrategy::LongestFirst,
            stride: 0,
        });
        let tok = tokenizer(&artifact, MockModel::new(&[]));
        let enc = tok.encode_pair(a, b, true).unwrap();
        enc.ids().len()
    };
    // Only the longer side truncates: 2 + 8 vs max 6 → 2 + 4.
    assert_eq!(run("ab", "cdefghij", 6), 6);
    // Both truncate to half: 8 + 8 vs max 6 → 3 + 3.
    assert_eq!(run("abcdefgh", "ijklmnop", 6), 6);
}

#[test]
fn only_second_requires_and_cuts_the_pair() {
    let mut artifact = bare_artifact();
    artifact.truncation = Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 5,
        strategy: TruncationStrategy::OnlySecond,
        stride: 0,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[]));
    let enc = tok.encode_pair("abc", "defgh", true).unwrap();
    assert_eq!(enc.tokens().join(""), "abcde");
    // The second sequence too short to absorb the excess is a loud
    // error, not silence.
    let err = tok.encode_pair("abcdefgh", "ij", true).unwrap_err();
    assert!(err.to_string().contains("second"));
}

#[test]
fn truncation_headroom_covers_template_specials() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.post_processor = Some(cls_sep_template());
    artifact.truncation = Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 4,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("a", 1), ("b", 2), ("c", 3)]));
    let enc = tok.encode("a b c", true).unwrap();
    // max 4 with [CLS]/[SEP] leaves 2 model tokens.
    assert_eq!(enc.ids(), [101, 1, 2, 102]);
}

#[test]
fn template_reaches_the_overflow_windows() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.post_processor = Some(cls_sep_template());
    artifact.truncation = Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 3,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });
    let tok = tokenizer(&artifact, MockModel::new(&[("a", 1), ("b", 2)]));
    let enc = tok.encode("a b", true).unwrap();
    assert_eq!(enc.ids(), [101, 1, 102]);
    assert_eq!(enc.overflowing().len(), 1);
    // The overflow window is templated too (the merge cartesian).
    assert_eq!(enc.overflowing()[0].ids(), [101, 2, 102]);
}

#[test]
fn batch_padding_variants() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    artifact.padding = Some(PaddingConfig {
        strategy: PaddingStrategy::BatchLongest,
        direction: Direction::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    });
    let mock = || MockModel::new(&[("a", 1), ("b", 2), ("c", 3)]);
    let tok = tokenizer(&artifact, mock());
    let batch = tok.encode_batch(&["a b c", "a"], true).unwrap();
    assert_eq!(batch[0].ids(), [1, 2, 3]);
    assert_eq!(batch[1].ids(), [1, 0, 0]);
    assert_eq!(batch[1].tokens(), ["a", "[PAD]", "[PAD]"]);
    assert_eq!(batch[1].attention_mask(), [1, 0, 0]);
    assert_eq!(batch[1].special_tokens_mask(), [0, 1, 1]);
    assert_eq!(batch[1].word_ids(), [Some(0), None, None]);

    // Fixed size, left side, rounded to a multiple.
    artifact.padding = Some(PaddingConfig {
        strategy: PaddingStrategy::Fixed(5),
        direction: Direction::Left,
        pad_to_multiple_of: Some(4),
        pad_id: 9,
        pad_type_id: 1,
        pad_token: "<pad>".to_string(),
    });
    let tok = tokenizer(&artifact, mock());
    let enc = tok.encode("a b", true).unwrap();
    // Fixed(5) rounds up to 8.
    assert_eq!(enc.ids(), [9, 9, 9, 9, 9, 9, 1, 2]);
    assert_eq!(enc.type_ids()[0], 1);
    assert_eq!(enc.attention_mask(), [0, 0, 0, 0, 0, 0, 1, 1]);
}

#[test]
fn set_truncation_and_padding_override_the_artifact() {
    let mut artifact = bare_artifact();
    artifact.truncation = Some(TruncationConfig {
        direction: Direction::Right,
        max_length: 2,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
    });
    let mut tok = tokenizer(&artifact, MockModel::new(&[]));
    assert_eq!(tok.encode("abcd", true).unwrap().ids().len(), 2);
    tok.set_truncation(None);
    assert_eq!(tok.encode("abcd", true).unwrap().ids().len(), 4);
    assert!(tok.padding().is_none());
    tok.set_padding(Some(PaddingConfig {
        strategy: PaddingStrategy::Fixed(6),
        direction: Direction::Right,
        pad_to_multiple_of: None,
        pad_id: 0,
        pad_type_id: 0,
        pad_token: "[PAD]".to_string(),
    }));
    assert_eq!(tok.encode("abcd", true).unwrap().ids().len(), 6);
}

// ── Encoding helpers ────────────────────────────────────────────────────

#[test]
fn alignment_helpers_walk_tokens_words_and_chars() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::Whitespace);
    let tok = tokenizer(&artifact, MockModel::new(&[("hey", 1), ("you", 2)]));
    let enc = tok.encode("hey you", true).unwrap();
    assert_eq!(enc.token_to_chars(1), Some((4, 7)));
    assert_eq!(enc.token_to_word(1), Some(1));
    assert_eq!(enc.word_to_tokens(0), Some((0, 1)));
    assert_eq!(enc.word_to_chars(1), Some((4, 7)));
    assert_eq!(enc.char_to_token(5), Some(1));
    assert_eq!(enc.char_to_word(0), Some(0));
    assert_eq!(enc.char_to_token(3), None); // the space belongs to no token
}

#[test]
fn empty_input_encodes_to_an_empty_encoding() {
    let artifact = bare_artifact();
    let tok = tokenizer(&artifact, MockModel::new(&[]));
    let enc = tok.encode("", true).unwrap();
    assert!(enc.is_empty());
    assert_eq!(enc.ids(), [] as [u32; 0]);
}

// ── Decoders ────────────────────────────────────────────────────────────

/// A decode-only tokenizer: vocab entries + a decoder chain.
fn decode_tok(entries: &[(&str, u32)], decoder: DecoderConfig) -> Tokenizer {
    let mut artifact = bare_artifact();
    artifact.decoder = Some(decoder);
    tokenizer(&artifact, MockModel::new(entries))
}

#[test]
fn byte_level_decoder_reassembles_split_multibyte_chars() {
    // é is bytes C3 A9 → stand-ins Ã and ©, split across two ids.
    let tok = decode_tok(
        &[("Hello", 1), ("Ġworld", 2), ("Ã", 3), ("©", 4)],
        DecoderConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
    );
    assert_eq!(tok.decode(&[1, 2], false).unwrap(), "Hello world");
    assert_eq!(tok.decode(&[1, 3, 4], false).unwrap(), "Helloé");
}

#[test]
fn wordpiece_decoder_joins_continuations() {
    let tok = decode_tok(
        &[("I", 1), ("##d", 2), ("##k", 3), ("you", 4), ("!", 5)],
        DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true,
        },
    );
    // Continuations fuse, cleanup folds " !".
    assert_eq!(tok.decode(&[1, 2, 3, 4, 5], false).unwrap(), "Idk you!");
}

#[test]
fn bpe_decoder_maps_suffix_to_spaces() {
    let tok = decode_tok(
        &[("the</w>", 1), ("end</w>", 2)],
        DecoderConfig::BpeDecoder {
            suffix: "</w>".to_string(),
        },
    );
    assert_eq!(tok.decode(&[1, 2], false).unwrap(), "the end");
}

#[test]
fn metaspace_decoder_drops_the_leading_marker() {
    let tok = decode_tok(
        &[("\u{2581}Hey", 1), ("\u{2581}friend", 2)],
        DecoderConfig::Metaspace {
            replacement: '\u{2581}',
            prepend_scheme: PrependScheme::Always,
            split: true,
        },
    );
    assert_eq!(tok.decode(&[1, 2], false).unwrap(), "Hey friend");
}

#[test]
fn byte_fallback_decoder_rebuilds_bytes() {
    let tok = decode_tok(
        &[("<0xC3>", 1), ("<0xA9>", 2), ("!", 3), ("<0xFF>", 4)],
        DecoderConfig::ByteFallback,
    );
    assert_eq!(tok.decode(&[1, 2, 3], false).unwrap(), "é!");
    // An invalid byte run becomes one replacement char per byte.
    assert_eq!(tok.decode(&[4, 3], false).unwrap(), "\u{FFFD}!");
}

#[test]
fn fuse_strip_and_replace_decoders() {
    let tok = decode_tok(
        &[("_a_", 1), ("_b__", 2)],
        DecoderConfig::Sequence(vec![
            DecoderConfig::Strip {
                content: '_',
                start: 1,
                stop: 2,
            },
            DecoderConfig::Replace {
                pattern: PatternConfig::String("b".to_string()),
                content: "B".to_string(),
            },
            DecoderConfig::Fuse,
        ]),
    );
    assert_eq!(tok.decode(&[1, 2], false).unwrap(), "aB");
}

#[test]
fn ctc_decoder_collapses_duplicates() {
    let tok = decode_tok(
        &[
            ("h", 1),
            ("e", 2),
            ("l", 3),
            ("<pad>", 4),
            ("|", 5),
            ("o", 6),
        ],
        DecoderConfig::Ctc {
            pad_token: "<pad>".to_string(),
            word_delimiter_token: "|".to_string(),
            cleanup: true,
        },
    );
    assert_eq!(
        tok.decode(&[1, 1, 2, 4, 3, 3, 4, 3, 6, 5, 6], false)
            .unwrap(),
        "hello o"
    );
}

#[test]
fn decode_skips_specials_only_on_request_and_errs_on_unknown_ids() {
    let mut artifact = bare_artifact();
    artifact.added_tokens = vec![added("<s>", 0, true)];
    let tok = tokenizer(&artifact, MockModel::new(&[("hi", 1)]));
    assert_eq!(tok.decode(&[0, 1], false).unwrap(), "<s> hi");
    assert_eq!(tok.decode(&[0, 1], true).unwrap(), "hi");
    let err = tok.decode(&[1, 777], false).unwrap_err();
    assert!(matches!(err, TokenizerError::Decode { id: 777, .. }));
    assert!(err.to_string().contains("777"));
}

// ── DecodeStream ────────────────────────────────────────────────────────

#[test]
fn decode_stream_holds_split_multibyte_bytes() {
    let tok = decode_tok(
        &[("Ã", 1), ("©", 2), ("hi", 3)],
        DecoderConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        },
    );
    let mut stream = tok.decode_stream(false);
    // First half of é: invalid alone — held.
    assert_eq!(stream.step(1).unwrap(), None);
    // Completing byte arrives: the whole char comes out.
    assert_eq!(stream.step(2).unwrap(), Some("é".to_string()));
    assert_eq!(stream.step(3).unwrap(), Some("hi".to_string()));
}

#[test]
fn decode_stream_concatenation_equals_whole_decode() {
    // The scope's conformance property, run long enough to cross the
    // window re-anchors several times (the pinned reference's own
    // implementation panics on this input at step 10).
    let entries: Vec<(String, u32)> = (0..10)
        .map(|i| (format!("\u{2581}w{i}"), u32::try_from(i).expect("small id")))
        .collect();
    let entry_refs: Vec<(&str, u32)> = entries.iter().map(|(s, i)| (s.as_str(), *i)).collect();
    let tok = decode_tok(
        &entry_refs,
        DecoderConfig::Metaspace {
            replacement: '\u{2581}',
            prepend_scheme: PrependScheme::Always,
            split: true,
        },
    );
    let ids: Vec<u32> = (0..10).chain(0..5).collect();
    let mut stream = tok.decode_stream(false);
    let mut collected = String::new();
    for &id in &ids {
        if let Some(piece) = stream.step(id).unwrap() {
            collected.push_str(&piece);
        }
    }
    assert_eq!(collected, tok.decode(&ids, false).unwrap());
}

// ── Round trip + lookups ────────────────────────────────────────────────

#[test]
fn byte_level_round_trip_through_encode_and_decode() {
    let mut artifact = bare_artifact();
    artifact.pre_tokenizer = Some(PreTokenizerConfig::ByteLevel {
        add_prefix_space: false,
        trim_offsets: true,
        use_regex: true,
    });
    artifact.decoder = Some(DecoderConfig::ByteLevel {
        add_prefix_space: false,
        trim_offsets: true,
        use_regex: true,
    });
    // Per-char fallback tokenizes each byte-level char; ids don't
    // matter for the round trip, tokens do — so give every stand-in
    // char of the input an id.
    let text = "café time";
    let mut entries: Vec<(String, u32)> = Vec::new();
    for (i, c) in "cafÃ©Ġtime".chars().enumerate() {
        entries.push((c.to_string(), u32::try_from(i).expect("small id") + 1));
    }
    let entry_refs: Vec<(&str, u32)> = entries.iter().map(|(s, i)| (s.as_str(), *i)).collect();
    let tok = tokenizer(&artifact, MockModel::new(&entry_refs));
    let enc = tok.encode(text, true).unwrap();
    assert_eq!(tok.decode(enc.ids(), false).unwrap(), text);
}

#[test]
fn lookup_quartet_overlays_added_tokens() {
    let mut artifact = bare_artifact();
    artifact.model = ModelConfig::WordLevel {
        vocab: vec![("hi".to_string(), 0), ("there".to_string(), 1)],
        unk_token: "<unk>".to_string(),
    };
    artifact.added_tokens = vec![added("<s>", 2, true)];
    let tok = tokenizer(&artifact, MockModel::new(&[("hi", 0), ("there", 1)]));
    assert_eq!(tok.token_to_id("<s>"), Some(2));
    assert_eq!(tok.token_to_id("hi"), Some(0));
    assert_eq!(tok.id_to_token(2), Some("<s>"));
    assert_eq!(tok.id_to_token(1), Some("there"));
    assert_eq!(tok.vocab_size(false), 2);
    assert_eq!(tok.vocab_size(true), 3);
    let vocab = tok.get_vocab(true);
    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab.get("<s>"), Some(&2));
    assert_eq!(tok.get_vocab(false).len(), 2);
}

#[test]
fn fixed_length_zero_is_a_loud_error() {
    let err = PreTokenizer::compile(&PreTokenizerConfig::FixedLength { length: 0 }).unwrap_err();
    assert!(err.to_string().contains("FixedLength"));
}

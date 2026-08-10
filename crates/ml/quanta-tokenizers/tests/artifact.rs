//! The typed schema layer against hand-built minimal artifacts: every
//! family's variant inventory, both serialized spellings where the
//! format has two, the reference-builder defaults for absent fields,
//! the `UnknownTag` claim boundary per family, and every §8 error row
//! the schema layer owns — with the message-contract checks (the
//! `loud_errors_name_the_problem` pattern).

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::artifact::{
    AddedTokenConfig, DecoderConfig, Direction, ModelConfig, NormalizerConfig, PaddingStrategy,
    PatternConfig, PostProcessorConfig, PreTokenizerConfig, PrependScheme, SequenceId,
    SplitBehavior, TemplatePiece, TokenizerArtifact, TruncationStrategy,
};

fn load(doc: &str) -> Result<TokenizerArtifact, TokenizerError> {
    TokenizerArtifact::from_bytes(doc.as_bytes())
}

fn load_ok(doc: &str) -> TokenizerArtifact {
    load(doc).unwrap_or_else(|e| panic!("artifact must load, got: {e}"))
}

/// A valid artifact with one section swapped in; WordLevel is the
/// filler model.
fn with_section(section: &str, body: &str) -> String {
    format!(
        r#"{{"version":"1.0","{section}":{body},"model":{{"type":"WordLevel","vocab":{{"a":0}},"unk_token":"<unk>"}}}}"#
    )
}

fn with_model(body: &str) -> String {
    format!(r#"{{"version":"1.0","model":{body}}}"#)
}

fn norm(body: &str) -> NormalizerConfig {
    load_ok(&with_section("normalizer", body))
        .normalizer
        .unwrap()
}

fn pre(body: &str) -> PreTokenizerConfig {
    load_ok(&with_section("pre_tokenizer", body))
        .pre_tokenizer
        .unwrap()
}

fn post(body: &str) -> PostProcessorConfig {
    load_ok(&with_section("post_processor", body))
        .post_processor
        .unwrap()
}

fn dec(body: &str) -> DecoderConfig {
    load_ok(&with_section("decoder", body)).decoder.unwrap()
}

// ── Top level ───────────────────────────────────────────────────────────

#[test]
fn minimal_artifact_loads_with_absent_sections_as_none() {
    let a = load_ok(r#"{"model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"<unk>"}}"#);
    assert_eq!(a.version, None);
    assert!(a.truncation.is_none());
    assert!(a.padding.is_none());
    assert!(a.added_tokens.is_empty());
    assert!(a.normalizer.is_none());
    assert!(a.pre_tokenizer.is_none());
    assert!(a.post_processor.is_none());
    assert!(a.decoder.is_none());
}

#[test]
fn null_sections_read_as_absent() {
    let a = load_ok(
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":null,
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"a":0},"unk_token":"<unk>"}}"#,
    );
    assert_eq!(a.version.as_deref(), Some("1.0"));
    assert!(a.normalizer.is_none());
    assert!(a.decoder.is_none());
}

#[test]
fn future_version_is_a_loud_schema_error() {
    let doc = r#"{"version":"2.0","model":{"type":"WordLevel","vocab":{},"unk_token":"x"}}"#;
    let Err(TokenizerError::Schema { path, what }) = load(doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "version");
    assert!(what.contains("\"1.0\""), "{what}");
}

#[test]
fn root_must_be_an_object() {
    let Err(TokenizerError::Schema { path, what }) = load("[1, 2]") else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "(root)");
    assert!(what.contains("array"), "{what}");
}

#[test]
fn missing_model_is_loud() {
    let Err(TokenizerError::Schema { path, what }) = load(r#"{"version":"1.0"}"#) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "model");
    assert!(what.contains("missing"), "{what}");
}

#[test]
fn missing_type_tag_is_loud() {
    let Err(TokenizerError::Schema { path, .. }) = load(&with_model(r#"{"vocab":{}}"#)) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "model.type");
}

// ── Models: BPE ─────────────────────────────────────────────────────────

const BPE_VOCAB: &str = r#"{"a":0,"b":1,"ab":2}"#;

#[test]
fn bpe_legacy_merge_spelling() {
    let doc = with_model(&format!(
        r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":["a b"]}}"#
    ));
    let ModelConfig::Bpe {
        vocab,
        merges,
        unk_token,
        continuing_subword_prefix,
        end_of_word_suffix,
        fuse_unk,
        byte_fallback,
        ignore_merges,
    } = load_ok(&doc).model
    else {
        panic!("must be BPE");
    };
    assert_eq!(
        vocab,
        [("a", 0u32), ("b", 1), ("ab", 2)].map(|(t, i)| (t.to_string(), i))
    );
    assert_eq!(merges, vec![("a".to_string(), "b".to_string())]);
    // Reference-builder defaults for absent fields.
    assert_eq!(unk_token, None);
    assert_eq!(continuing_subword_prefix, None);
    assert_eq!(end_of_word_suffix, None);
    assert!(!fuse_unk && !byte_fallback && !ignore_merges);
}

#[test]
fn bpe_pair_merge_spelling() {
    let doc = with_model(&format!(
        r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":[["a","b"]]}}"#
    ));
    let ModelConfig::Bpe { merges, .. } = load_ok(&doc).model else {
        panic!("must be BPE");
    };
    assert_eq!(merges, vec![("a".to_string(), "b".to_string())]);
}

#[test]
fn bpe_mixed_merge_spellings_are_rejected() {
    let doc = with_model(&format!(
        r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":["a b",["a","b"]]}}"#
    ));
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "model.merges[1]");
    assert!(what.contains("mixed"), "{what}");
}

#[test]
fn bpe_legacy_merge_must_be_one_space_two_sides() {
    for bad in [r#"["a b c"]"#, r#"["ab"]"#, r#"[" b"]"#, r#"["a "]"#] {
        let doc = with_model(&format!(
            r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":{bad}}}"#
        ));
        assert!(
            matches!(load(&doc), Err(TokenizerError::Schema { .. })),
            "merges {bad} must be rejected"
        );
    }
}

#[test]
fn bpe_full_flag_set_loads() {
    // With prefix "##": merge (a, ##b) produces "ab" — the reference's
    // prefix-stripping arithmetic.
    let doc = with_model(
        r###"{"type":"BPE","dropout":null,"unk_token":"<unk>",
            "continuing_subword_prefix":"##","end_of_word_suffix":"</w>",
            "fuse_unk":true,"byte_fallback":true,"ignore_merges":true,
            "cache_capacity":10000,
            "vocab":{"<unk>":0,"a":1,"##b":2,"ab":3},"merges":[["a","##b"]]}"###,
    );
    let ModelConfig::Bpe {
        unk_token,
        continuing_subword_prefix,
        end_of_word_suffix,
        fuse_unk,
        byte_fallback,
        ignore_merges,
        ..
    } = load_ok(&doc).model
    else {
        panic!("must be BPE");
    };
    assert_eq!(unk_token.as_deref(), Some("<unk>"));
    assert_eq!(continuing_subword_prefix.as_deref(), Some("##"));
    assert_eq!(end_of_word_suffix.as_deref(), Some("</w>"));
    assert!(fuse_unk && byte_fallback && ignore_merges);
}

#[test]
fn bpe_dropout_nonnull_is_unsupported() {
    let doc = with_model(&format!(
        r#"{{"type":"BPE","dropout":0.1,"vocab":{BPE_VOCAB},"merges":[]}}"#
    ));
    let Err(TokenizerError::UnsupportedField { path, why }) = load(&doc) else {
        panic!("must fail with UnsupportedField");
    };
    assert_eq!(path, "model.dropout");
    assert!(why.contains("train-time"), "{why}");
    let msg = TokenizerError::UnsupportedField { path, why }.to_string();
    assert!(msg.contains("model.dropout"), "{msg}");
}

#[test]
fn bpe_merge_naming_absent_token_is_loud() {
    let doc = with_model(&format!(
        r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":[["a","z"]]}}"#
    ));
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("\"z\""), "{what}");
    assert!(what.contains("not in the vocab"), "{what}");
}

#[test]
fn bpe_merge_producing_absent_token_is_loud() {
    // "b" and "a" exist, but "ba" does not.
    let doc = with_model(&format!(
        r#"{{"type":"BPE","vocab":{BPE_VOCAB},"merges":[["b","a"]]}}"#
    ));
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("\"ba\""), "{what}");
    assert!(what.contains("produces"), "{what}");
}

#[test]
fn bpe_requires_vocab_and_merges() {
    for missing in [
        r#"{"type":"BPE","merges":[]}"#,
        r#"{"type":"BPE","vocab":{}}"#,
    ] {
        assert!(
            matches!(
                load(&with_model(missing)),
                Err(TokenizerError::Schema { .. })
            ),
            "{missing} must be rejected"
        );
    }
}

// ── Vocab validation (shared map shape) ─────────────────────────────────

#[test]
fn vocab_id_collision_is_loud() {
    let doc = with_model(r#"{"type":"WordLevel","vocab":{"a":0,"b":0},"unk_token":"a"}"#);
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("id collision"), "{what}");
    assert!(what.contains("\"b\""), "{what}");
}

#[test]
fn vocab_id_beyond_u32_is_loud() {
    let doc = with_model(r#"{"type":"WordLevel","vocab":{"a":4294967296},"unk_token":"a"}"#);
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("4294967296"), "{what}");
    assert!(what.contains("u32"), "{what}");
}

#[test]
fn vocab_id_must_be_a_plain_unsigned_integer() {
    for bad in [
        r#"{"a":1.5}"#,
        r#"{"a":-1}"#,
        r#"{"a":"1"}"#,
        r#"{"a":1e2}"#,
    ] {
        let doc = with_model(&format!(
            r#"{{"type":"WordLevel","vocab":{bad},"unk_token":"a"}}"#
        ));
        let Err(TokenizerError::Schema { path, .. }) = load(&doc) else {
            panic!("vocab {bad} must fail with Schema");
        };
        assert_eq!(path, r#"model.vocab["a"]"#);
    }
}

#[test]
fn duplicate_vocab_tokens_are_caught_by_the_json_layer() {
    let doc = with_model(r#"{"type":"WordLevel","vocab":{"a":0,"a":1},"unk_token":"a"}"#);
    let Err(TokenizerError::Json { what, .. }) = load(&doc) else {
        panic!("must fail with Json");
    };
    assert!(what.contains("duplicate key"), "{what}");
}

// ── Models: WordPiece / Unigram / WordLevel ─────────────────────────────

#[test]
fn wordpiece_explicit_and_default_fields() {
    let full = with_model(
        r#"{"type":"WordPiece","vocab":{"[UNK]":0,"a":1},"unk_token":"[UNK]",
            "continuing_subword_prefix":"++","max_input_chars_per_word":50}"#,
    );
    let ModelConfig::WordPiece {
        unk_token,
        continuing_subword_prefix,
        max_input_chars_per_word,
        ..
    } = load_ok(&full).model
    else {
        panic!("must be WordPiece");
    };
    assert_eq!(unk_token, "[UNK]");
    assert_eq!(continuing_subword_prefix, "++");
    assert_eq!(max_input_chars_per_word, 50);

    // Absent fields take the reference builder's defaults.
    let minimal = with_model(r#"{"type":"WordPiece","vocab":{"x":0}}"#);
    let ModelConfig::WordPiece {
        unk_token,
        continuing_subword_prefix,
        max_input_chars_per_word,
        ..
    } = load_ok(&minimal).model
    else {
        panic!("must be WordPiece");
    };
    assert_eq!(unk_token, "[UNK]");
    assert_eq!(continuing_subword_prefix, "##");
    assert_eq!(max_input_chars_per_word, 100);
}

#[test]
fn unigram_minimal() {
    let doc = with_model(
        r#"{"type":"Unigram","vocab":[["<unk>",-2.0],["a",-1.5],["ab",-13.629]],"unk_id":0}"#,
    );
    let ModelConfig::Unigram {
        vocab,
        unk_id,
        byte_fallback,
    } = load_ok(&doc).model
    else {
        panic!("must be Unigram");
    };
    assert_eq!(vocab.len(), 3);
    assert_eq!(vocab[2], ("ab".to_string(), -13.629));
    assert_eq!(unk_id, Some(0));
    assert!(!byte_fallback);
}

#[test]
fn unigram_null_unk_id_reads_as_none() {
    let doc =
        with_model(r#"{"type":"Unigram","vocab":[["a",-1.0]],"unk_id":null,"byte_fallback":true}"#);
    let ModelConfig::Unigram {
        unk_id,
        byte_fallback,
        ..
    } = load_ok(&doc).model
    else {
        panic!("must be Unigram");
    };
    assert_eq!(unk_id, None);
    assert!(byte_fallback);
}

#[test]
fn unigram_unk_id_out_of_range_is_loud() {
    let doc = with_model(r#"{"type":"Unigram","vocab":[["a",-1.0]],"unk_id":5}"#);
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("unk_id 5"), "{what}");
    assert!(what.contains("1 pieces"), "{what}");
}

#[test]
fn unigram_duplicate_piece_is_loud() {
    let doc = with_model(r#"{"type":"Unigram","vocab":[["a",-1.0],["a",-2.0]]}"#);
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("duplicate piece"), "{what}");
}

#[test]
fn unigram_malformed_rows_are_loud() {
    for bad in [
        r#"[["a",-1.0,3]]"#,
        r#"[["a"]]"#,
        r#"[[-1.0,"a"]]"#,
        r#"["a"]"#,
    ] {
        let doc = with_model(&format!(r#"{{"type":"Unigram","vocab":{bad}}}"#));
        assert!(
            matches!(load(&doc), Err(TokenizerError::Schema { .. })),
            "vocab {bad} must be rejected"
        );
    }
}

#[test]
fn wordlevel_default_unk_token() {
    let doc = with_model(r#"{"type":"WordLevel","vocab":{"hello":0}}"#);
    let ModelConfig::WordLevel { vocab, unk_token } = load_ok(&doc).model else {
        panic!("must be WordLevel");
    };
    assert_eq!(vocab, vec![("hello".to_string(), 0)]);
    assert_eq!(unk_token, "<unk>");
}

#[test]
fn model_unknown_tag_is_the_claim_boundary() {
    let Err(e) = load(&with_model(r#"{"type":"SuperBPE","vocab":{}}"#)) else {
        panic!("must fail");
    };
    let TokenizerError::UnknownTag { family, tag } = &e else {
        panic!("must be UnknownTag, got: {e}");
    };
    assert_eq!(*family, "model");
    assert_eq!(tag, "SuperBPE");
    let msg = e.to_string();
    assert!(msg.contains("SuperBPE"), "{msg}");
    assert!(msg.contains("0.21"), "{msg}");
}

// ── Normalizers ─────────────────────────────────────────────────────────

#[test]
fn fieldless_normalizer_variants() {
    for (body, expect) in [
        (r#"{"type":"NFC"}"#, NormalizerConfig::Nfc),
        (r#"{"type":"NFD"}"#, NormalizerConfig::Nfd),
        (r#"{"type":"NFKC"}"#, NormalizerConfig::Nfkc),
        (r#"{"type":"NFKD"}"#, NormalizerConfig::Nfkd),
        (r#"{"type":"Lowercase"}"#, NormalizerConfig::Lowercase),
        (r#"{"type":"StripAccents"}"#, NormalizerConfig::StripAccents),
        (r#"{"type":"Nmt"}"#, NormalizerConfig::Nmt),
        (r#"{"type":"ByteLevel"}"#, NormalizerConfig::ByteLevel),
    ] {
        assert_eq!(norm(body), expect, "{body}");
    }
}

#[test]
fn bert_normalizer_with_null_strip_accents() {
    // bert-base-uncased's exact serialization shape.
    let got = norm(
        r#"{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,
            "strip_accents":null,"lowercase":true}"#,
    );
    assert_eq!(
        got,
        NormalizerConfig::Bert {
            clean_text: true,
            handle_chinese_chars: true,
            strip_accents: None,
            lowercase: true,
        }
    );
    let explicit = norm(r#"{"type":"BertNormalizer","strip_accents":false,"lowercase":false}"#);
    assert_eq!(
        explicit,
        NormalizerConfig::Bert {
            clean_text: true, // reference default
            handle_chinese_chars: true,
            strip_accents: Some(false),
            lowercase: false,
        }
    );
}

#[test]
fn strip_prepend_replace_normalizers() {
    assert_eq!(
        norm(r#"{"type":"Strip","strip_left":true,"strip_right":false}"#),
        NormalizerConfig::Strip {
            strip_left: true,
            strip_right: false
        }
    );
    assert_eq!(
        norm(r#"{"type":"Prepend","prepend":"▁"}"#),
        NormalizerConfig::Prepend {
            prepend: "▁".to_string()
        }
    );
    // Llama's " " → "▁" replace, String-pattern form.
    assert_eq!(
        norm(r#"{"type":"Replace","pattern":{"String":" "},"content":"▁"}"#),
        NormalizerConfig::Replace {
            pattern: PatternConfig::String(" ".to_string()),
            content: "▁".to_string()
        }
    );
    // Regex-pattern form routes through the engine at pipeline time.
    assert_eq!(
        norm(r#"{"type":"Replace","pattern":{"Regex":" {2,}"},"content":" "}"#),
        NormalizerConfig::Replace {
            pattern: PatternConfig::Regex(" {2,}".to_string()),
            content: " ".to_string()
        }
    );
}

#[test]
fn pattern_must_be_string_or_regex() {
    let doc = with_section(
        "normalizer",
        r#"{"type":"Replace","pattern":{"Glob":"*"},"content":""}"#,
    );
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "normalizer.pattern");
    assert!(what.contains("Glob"), "{what}");
}

#[test]
fn precompiled_charsmap_decodes_base64() {
    for (b64, bytes) in [
        ("", &b""[..]),
        ("TWFu", b"Man"),
        ("TWE=", b"Ma"),
        ("TQ==", b"M"),
        ("AAECAwQFBgc=", &[0u8, 1, 2, 3, 4, 5, 6, 7][..]),
    ] {
        let got = norm(&format!(
            r#"{{"type":"Precompiled","precompiled_charsmap":"{b64}"}}"#
        ));
        assert_eq!(
            got,
            NormalizerConfig::Precompiled {
                charsmap: bytes.to_vec()
            },
            "{b64}"
        );
    }
}

#[test]
fn corrupt_charsmap_base64_is_loud_with_offset() {
    for bad in ["TWF", "TW!u", "TQ=x", "=AAA", "TQ==TWFu", "TR=="] {
        let doc = with_section(
            "normalizer",
            &format!(r#"{{"type":"Precompiled","precompiled_charsmap":"{bad}"}}"#),
        );
        let Err(e) = load(&doc) else {
            panic!("base64 {bad:?} must fail");
        };
        let TokenizerError::Charsmap { .. } = &e else {
            panic!("base64 {bad:?} must fail with Charsmap, got: {e}");
        };
        let msg = e.to_string();
        assert!(msg.contains("at offset"), "{msg}");
    }
}

#[test]
fn normalizer_sequence_nests() {
    let got = norm(
        r#"{"type":"Sequence","normalizers":[
            {"type":"NFD"},
            {"type":"Sequence","normalizers":[{"type":"Lowercase"}]},
            {"type":"StripAccents"}]}"#,
    );
    assert_eq!(
        got,
        NormalizerConfig::Sequence(vec![
            NormalizerConfig::Nfd,
            NormalizerConfig::Sequence(vec![NormalizerConfig::Lowercase]),
            NormalizerConfig::StripAccents,
        ])
    );
}

#[test]
fn normalizer_unknown_tag() {
    let doc = with_section("normalizer", r#"{"type":"Fancy"}"#);
    assert!(matches!(
        load(&doc),
        Err(TokenizerError::UnknownTag {
            family: "normalizer",
            ..
        })
    ));
}

#[test]
fn nested_unknown_tag_inside_sequence_is_still_loud() {
    let doc = with_section(
        "normalizer",
        r#"{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Fancy"}]}"#,
    );
    assert!(matches!(
        load(&doc),
        Err(TokenizerError::UnknownTag {
            family: "normalizer",
            ..
        })
    ));
}

// ── Pre-tokenizers ──────────────────────────────────────────────────────

#[test]
fn byte_level_pre_tokenizer_explicit_and_defaults() {
    // gpt2's exact serialization shape.
    assert_eq!(
        pre(r#"{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true}"#),
        PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            trim_offsets: true,
            use_regex: true, // absent in older artifacts — reference default
        }
    );
    assert_eq!(
        pre(r#"{"type":"ByteLevel"}"#),
        PreTokenizerConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    );
}

#[test]
fn fieldless_pre_tokenizer_variants() {
    for (body, expect) in [
        (r#"{"type":"BertPreTokenizer"}"#, PreTokenizerConfig::Bert),
        (r#"{"type":"Whitespace"}"#, PreTokenizerConfig::Whitespace),
        (
            r#"{"type":"WhitespaceSplit"}"#,
            PreTokenizerConfig::WhitespaceSplit,
        ),
        (
            r#"{"type":"UnicodeScripts"}"#,
            PreTokenizerConfig::UnicodeScripts,
        ),
    ] {
        assert_eq!(pre(body), expect, "{body}");
    }
}

#[test]
fn punctuation_defaults_to_isolated() {
    assert_eq!(
        pre(r#"{"type":"Punctuation"}"#),
        PreTokenizerConfig::Punctuation {
            behavior: SplitBehavior::Isolated
        }
    );
    assert_eq!(
        pre(r#"{"type":"Punctuation","behavior":"Removed"}"#),
        PreTokenizerConfig::Punctuation {
            behavior: SplitBehavior::Removed
        }
    );
}

#[test]
fn digits_and_char_delimiter_and_fixed_length() {
    assert_eq!(
        pre(r#"{"type":"Digits","individual_digits":true}"#),
        PreTokenizerConfig::Digits {
            individual_digits: true
        }
    );
    assert_eq!(
        pre(r#"{"type":"CharDelimiterSplit","delimiter":"|"}"#),
        PreTokenizerConfig::CharDelimiterSplit { delimiter: '|' }
    );
    assert_eq!(
        pre(r#"{"type":"FixedLength","length":3}"#),
        PreTokenizerConfig::FixedLength { length: 3 }
    );
}

#[test]
fn delimiter_must_be_one_char() {
    let doc = with_section(
        "pre_tokenizer",
        r#"{"type":"CharDelimiterSplit","delimiter":"ab"}"#,
    );
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "pre_tokenizer.delimiter");
    assert!(what.contains("single-character"), "{what}");
}

#[test]
fn metaspace_current_spelling() {
    assert_eq!(
        pre(r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":false}"#),
        PreTokenizerConfig::Metaspace {
            replacement: '▁',
            prepend_scheme: PrependScheme::First,
            split: false,
        }
    );
    // Bare Metaspace: all reference defaults.
    assert_eq!(
        pre(r#"{"type":"Metaspace"}"#),
        PreTokenizerConfig::Metaspace {
            replacement: '▁',
            prepend_scheme: PrependScheme::Always,
            split: true,
        }
    );
}

#[test]
fn metaspace_legacy_add_prefix_space_spelling() {
    // The legacy serialization the reference still accepts (str_rep is
    // accepted-and-ignored).
    assert_eq!(
        pre(r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":true,"str_rep":"▁"}"#),
        PreTokenizerConfig::Metaspace {
            replacement: '▁',
            prepend_scheme: PrependScheme::Always,
            split: true,
        }
    );
    assert_eq!(
        pre(r#"{"type":"Metaspace","replacement":"▁","add_prefix_space":false}"#),
        PreTokenizerConfig::Metaspace {
            replacement: '▁',
            prepend_scheme: PrependScheme::Never,
            split: true,
        }
    );
}

#[test]
fn split_all_five_behaviors() {
    for (name, behavior) in [
        ("Removed", SplitBehavior::Removed),
        ("Isolated", SplitBehavior::Isolated),
        ("MergedWithPrevious", SplitBehavior::MergedWithPrevious),
        ("MergedWithNext", SplitBehavior::MergedWithNext),
        ("Contiguous", SplitBehavior::Contiguous),
    ] {
        let got = pre(&format!(
            r#"{{"type":"Split","pattern":{{"Regex":"\\s+"}},"behavior":"{name}","invert":true}}"#
        ));
        assert_eq!(
            got,
            PreTokenizerConfig::Split {
                pattern: PatternConfig::Regex("\\s+".to_string()),
                behavior,
                invert: true,
            },
            "{name}"
        );
    }
}

#[test]
fn split_unknown_behavior_is_loud() {
    let doc = with_section(
        "pre_tokenizer",
        r#"{"type":"Split","pattern":{"String":" "},"behavior":"Sideways","invert":false}"#,
    );
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "pre_tokenizer.behavior");
    assert!(what.contains("Sideways"), "{what}");
    assert!(what.contains("Contiguous"), "{what}"); // lists the options
}

#[test]
fn pre_tokenizer_sequence_uses_the_pretokenizers_field() {
    let got = pre(r#"{"type":"Sequence","pretokenizers":[
            {"type":"WhitespaceSplit"},
            {"type":"ByteLevel","add_prefix_space":false}]}"#);
    let PreTokenizerConfig::Sequence(inner) = got else {
        panic!("must be Sequence");
    };
    assert_eq!(inner.len(), 2);
    assert_eq!(inner[0], PreTokenizerConfig::WhitespaceSplit);
}

#[test]
fn pre_tokenizer_unknown_tag() {
    let doc = with_section("pre_tokenizer", r#"{"type":"Mecab"}"#);
    assert!(matches!(
        load(&doc),
        Err(TokenizerError::UnknownTag {
            family: "pre_tokenizer",
            ..
        })
    ));
}

// ── Post-processors ─────────────────────────────────────────────────────

const TEMPLATE_BODY: &str = r#"{"type":"TemplateProcessing",
    "single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},
              {"Sequence":{"id":"A","type_id":0}},
              {"SpecialToken":{"id":"[SEP]","type_id":0}}],
    "pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},
            {"Sequence":{"id":"A","type_id":0}},
            {"SpecialToken":{"id":"[SEP]","type_id":0}},
            {"Sequence":{"id":"B","type_id":1}},
            {"SpecialToken":{"id":"[SEP]","type_id":1}}],
    "special_tokens":{
        "[CLS]":{"id":"[CLS]","ids":[101],"tokens":["[CLS]"]},
        "[SEP]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}}}"#;

#[test]
fn template_processing_loads_bert_shape() {
    let PostProcessorConfig::Template {
        single,
        pair,
        special_tokens,
    } = post(TEMPLATE_BODY)
    else {
        panic!("must be Template");
    };
    assert_eq!(single.len(), 3);
    assert_eq!(pair.len(), 5);
    assert_eq!(
        single[1],
        TemplatePiece::Sequence {
            id: SequenceId::A,
            type_id: 0
        }
    );
    assert_eq!(
        pair[3],
        TemplatePiece::Sequence {
            id: SequenceId::B,
            type_id: 1
        }
    );
    assert_eq!(
        pair[4],
        TemplatePiece::SpecialToken {
            id: "[SEP]".to_string(),
            type_id: 1
        }
    );
    assert_eq!(special_tokens.len(), 2);
    assert_eq!(special_tokens[0].ids, vec![101]);
    assert_eq!(special_tokens[0].tokens, vec!["[CLS]".to_string()]);
}

#[test]
fn template_referencing_undeclared_special_is_loud() {
    let body = r#"{"type":"TemplateProcessing",
        "single":[{"SpecialToken":{"id":"[MASK]","type_id":0}}],
        "pair":[],
        "special_tokens":{}}"#;
    let doc = with_section("post_processor", body);
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "post_processor.single[0]");
    assert!(what.contains("[MASK]"), "{what}");
}

#[test]
fn template_ids_tokens_length_mismatch_is_loud() {
    let body = r#"{"type":"TemplateProcessing","single":[],"pair":[],
        "special_tokens":{"[CLS]":{"id":"[CLS]","ids":[101,102],"tokens":["[CLS]"]}}}"#;
    let Err(TokenizerError::Schema { what, .. }) = load(&with_section("post_processor", body))
    else {
        panic!("must fail with Schema");
    };
    assert!(what.contains("same length"), "{what}");
}

#[test]
fn template_map_key_must_match_entry_id() {
    let body = r#"{"type":"TemplateProcessing","single":[],"pair":[],
        "special_tokens":{"[CLS]":{"id":"[SEP]","ids":[102],"tokens":["[SEP]"]}}}"#;
    let Err(TokenizerError::Schema { what, .. }) = load(&with_section("post_processor", body))
    else {
        panic!("must fail with Schema");
    };
    assert!(what.contains("disagrees"), "{what}");
}

#[test]
fn bert_and_roberta_processing() {
    assert_eq!(
        post(r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#),
        PostProcessorConfig::Bert {
            sep: ("[SEP]".to_string(), 102),
            cls: ("[CLS]".to_string(), 101),
        }
    );
    assert_eq!(
        post(
            r#"{"type":"RobertaProcessing","sep":["</s>",2],"cls":["<s>",0],
                "trim_offsets":true,"add_prefix_space":false}"#
        ),
        PostProcessorConfig::Roberta {
            sep: ("</s>".to_string(), 2),
            cls: ("<s>".to_string(), 0),
            trim_offsets: true,
            add_prefix_space: false,
        }
    );
}

#[test]
fn byte_level_and_sequence_post_processors() {
    assert_eq!(
        post(r#"{"type":"ByteLevel","trim_offsets":false}"#),
        PostProcessorConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: false,
            use_regex: true,
        }
    );
    let PostProcessorConfig::Sequence(inner) = post(
        r#"{"type":"Sequence","processors":[
            {"type":"ByteLevel","trim_offsets":false},
            {"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}]}"#,
    ) else {
        panic!("must be Sequence");
    };
    assert_eq!(inner.len(), 2);
}

#[test]
fn post_processor_unknown_tag() {
    let doc = with_section("post_processor", r#"{"type":"Chat"}"#);
    assert!(matches!(
        load(&doc),
        Err(TokenizerError::UnknownTag {
            family: "post_processor",
            ..
        })
    ));
}

// ── Decoders ────────────────────────────────────────────────────────────

#[test]
fn every_decoder_variant_loads() {
    assert_eq!(
        dec(r#"{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#),
        DecoderConfig::ByteLevel {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    );
    assert_eq!(
        dec(r###"{"type":"WordPiece","prefix":"##","cleanup":true}"###),
        DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true
        }
    );
    // Absent fields → reference defaults.
    assert_eq!(
        dec(r#"{"type":"WordPiece"}"#),
        DecoderConfig::WordPiece {
            prefix: "##".to_string(),
            cleanup: true
        }
    );
    assert_eq!(
        dec(r#"{"type":"BPEDecoder","suffix":"</w>"}"#),
        DecoderConfig::BpeDecoder {
            suffix: "</w>".to_string()
        }
    );
    assert_eq!(
        dec(r#"{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}"#),
        DecoderConfig::Metaspace {
            replacement: '▁',
            prepend_scheme: PrependScheme::Always,
            split: true,
        }
    );
    assert_eq!(
        dec(r#"{"type":"ByteFallback"}"#),
        DecoderConfig::ByteFallback
    );
    assert_eq!(dec(r#"{"type":"Fuse"}"#), DecoderConfig::Fuse);
    // Llama's strip-one-leading-space decoder.
    assert_eq!(
        dec(r#"{"type":"Strip","content":" ","start":1,"stop":0}"#),
        DecoderConfig::Strip {
            content: ' ',
            start: 1,
            stop: 0
        }
    );
    assert_eq!(
        dec(r#"{"type":"Replace","pattern":{"String":"▁"},"content":" "}"#),
        DecoderConfig::Replace {
            pattern: PatternConfig::String("▁".to_string()),
            content: " ".to_string()
        }
    );
    assert_eq!(
        dec(r#"{"type":"CTC","pad_token":"<pad>","word_delimiter_token":"|","cleanup":true}"#),
        DecoderConfig::Ctc {
            pad_token: "<pad>".to_string(),
            word_delimiter_token: "|".to_string(),
            cleanup: true,
        }
    );
    // Llama's actual decoder chain shape.
    let DecoderConfig::Sequence(inner) = dec(r#"{"type":"Sequence","decoders":[
            {"type":"Replace","pattern":{"String":"▁"},"content":" "},
            {"type":"ByteFallback"},
            {"type":"Fuse"},
            {"type":"Strip","content":" ","start":1,"stop":0}]}"#)
    else {
        panic!("must be Sequence");
    };
    assert_eq!(inner.len(), 4);
    assert_eq!(inner[1], DecoderConfig::ByteFallback);
}

#[test]
fn decoder_unknown_tag() {
    let doc = with_section("decoder", r#"{"type":"Reverse"}"#);
    assert!(matches!(
        load(&doc),
        Err(TokenizerError::UnknownTag {
            family: "decoder",
            ..
        })
    ));
}

// ── Added tokens ────────────────────────────────────────────────────────

#[test]
fn added_tokens_full_flag_set() {
    let doc = with_section(
        "added_tokens",
        r#"[{"id":50256,"content":"<|endoftext|>","single_word":false,"lstrip":false,
             "rstrip":false,"normalized":true,"special":true},
            {"id":50257,"content":"<pad>","single_word":true,"lstrip":true,
             "rstrip":true,"normalized":false,"special":false}]"#,
    );
    let a = load_ok(&doc);
    assert_eq!(
        a.added_tokens[0],
        AddedTokenConfig {
            id: 50256,
            content: "<|endoftext|>".to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: true,
            special: true,
        }
    );
    assert!(a.added_tokens[1].single_word);
}

#[test]
fn added_token_normalized_defaults_to_not_special() {
    let doc = with_section(
        "added_tokens",
        r#"[{"id":0,"content":"<s>","special":true},{"id":1,"content":"w"}]"#,
    );
    let a = load_ok(&doc);
    assert!(!a.added_tokens[0].normalized); // special → raw-text match
    assert!(a.added_tokens[1].normalized); // non-special → normalized
    assert!(!a.added_tokens[1].special);
}

#[test]
fn added_token_missing_content_is_loud() {
    let doc = with_section("added_tokens", r#"[{"id":0}]"#);
    let Err(TokenizerError::Schema { path, .. }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "added_tokens[0].content");
}

#[test]
fn added_token_id_beyond_u32_is_loud() {
    let doc = with_section("added_tokens", r#"[{"id":4294967296,"content":"x"}]"#);
    let Err(TokenizerError::Vocab { what }) = load(&doc) else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("u32"), "{what}");
}

// ── Truncation / padding ────────────────────────────────────────────────

#[test]
fn truncation_full_params_and_defaults() {
    let doc = with_section(
        "truncation",
        r#"{"direction":"Left","max_length":384,"strategy":"OnlySecond","stride":128}"#,
    );
    let t = load_ok(&doc).truncation.unwrap();
    assert_eq!(t.direction, Direction::Left);
    assert_eq!(t.max_length, 384);
    assert_eq!(t.strategy, TruncationStrategy::OnlySecond);
    assert_eq!(t.stride, 128);

    let t = load_ok(&with_section("truncation", "{}"))
        .truncation
        .unwrap();
    assert_eq!(t.direction, Direction::Right);
    assert_eq!(t.max_length, 512);
    assert_eq!(t.strategy, TruncationStrategy::LongestFirst);
    assert_eq!(t.stride, 0);
}

#[test]
fn truncation_unknown_strategy_is_loud() {
    let doc = with_section("truncation", r#"{"strategy":"Shortest"}"#);
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "truncation.strategy");
    assert!(what.contains("Shortest"), "{what}");
}

#[test]
fn padding_both_strategy_spellings() {
    let doc = with_section(
        "padding",
        r#"{"strategy":"BatchLongest","direction":"Right","pad_to_multiple_of":null,
            "pad_id":0,"pad_type_id":0,"pad_token":"[PAD]"}"#,
    );
    let p = load_ok(&doc).padding.unwrap();
    assert_eq!(p.strategy, PaddingStrategy::BatchLongest);
    assert_eq!(p.pad_to_multiple_of, None);
    assert_eq!(p.pad_token, "[PAD]");

    let doc = with_section(
        "padding",
        r#"{"strategy":{"Fixed":512},"direction":"Left","pad_to_multiple_of":8,
            "pad_id":3,"pad_type_id":1,"pad_token":"<pad>"}"#,
    );
    let p = load_ok(&doc).padding.unwrap();
    assert_eq!(p.strategy, PaddingStrategy::Fixed(512));
    assert_eq!(p.direction, Direction::Left);
    assert_eq!(p.pad_to_multiple_of, Some(8));
    assert_eq!(p.pad_id, 3);
    assert_eq!(p.pad_type_id, 1);
    assert_eq!(p.pad_token, "<pad>");
}

#[test]
fn padding_unknown_strategy_is_loud() {
    let doc = with_section("padding", r#"{"strategy":"Longest"}"#);
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "padding.strategy");
    assert!(what.contains("Longest"), "{what}");
}

#[test]
fn bad_direction_is_loud() {
    let doc = with_section("padding", r#"{"direction":"Up"}"#);
    let Err(TokenizerError::Schema { path, what }) = load(&doc) else {
        panic!("must fail with Schema");
    };
    assert_eq!(path, "padding.direction");
    assert!(what.contains("Up"), "{what}");
}

// ── Whole-artifact shapes ───────────────────────────────────────────────

#[test]
fn gpt2_shaped_artifact_loads_whole() {
    let doc = r#"{
        "version":"1.0",
        "truncation":null,
        "padding":null,
        "added_tokens":[{"id":3,"content":"<|endoftext|>","single_word":false,
            "lstrip":false,"rstrip":false,"normalized":true,"special":true}],
        "normalizer":null,
        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true},
        "model":{"type":"BPE","dropout":null,"unk_token":null,
            "continuing_subword_prefix":"","end_of_word_suffix":"",
            "fuse_unk":false,
            "vocab":{"a":0,"b":1,"ab":2,"<|endoftext|>":3},
            "merges":["a b"]},
        "post_processor":{"type":"ByteLevel","trim_offsets":false},
        "decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true}
    }"#;
    let a = load_ok(doc);
    assert!(matches!(a.model, ModelConfig::Bpe { .. }));
    assert!(matches!(
        a.pre_tokenizer,
        Some(PreTokenizerConfig::ByteLevel {
            add_prefix_space: false,
            ..
        })
    ));
    assert!(matches!(a.decoder, Some(DecoderConfig::ByteLevel { .. })));
    assert_eq!(a.added_tokens[0].content, "<|endoftext|>");
    // Empty-string prefix/suffix load as Some("") — held verbatim.
    let ModelConfig::Bpe {
        continuing_subword_prefix,
        end_of_word_suffix,
        ..
    } = &a.model
    else {
        unreachable!()
    };
    assert_eq!(continuing_subword_prefix.as_deref(), Some(""));
    assert_eq!(end_of_word_suffix.as_deref(), Some(""));
}

#[test]
fn bert_shaped_artifact_loads_whole() {
    let doc = format!(
        r###"{{
        "version":"1.0",
        "truncation":{{"direction":"Right","max_length":512,"strategy":"LongestFirst","stride":0}},
        "padding":{{"strategy":"BatchLongest","direction":"Right","pad_to_multiple_of":null,
            "pad_id":0,"pad_type_id":0,"pad_token":"[PAD]"}},
        "added_tokens":[
            {{"id":0,"content":"[PAD]","single_word":false,"lstrip":false,"rstrip":false,
              "normalized":false,"special":true}},
            {{"id":101,"content":"[CLS]","single_word":false,"lstrip":false,"rstrip":false,
              "normalized":false,"special":true}},
            {{"id":102,"content":"[SEP]","single_word":false,"lstrip":false,"rstrip":false,
              "normalized":false,"special":true}}],
        "normalizer":{{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,
            "strip_accents":null,"lowercase":true}},
        "pre_tokenizer":{{"type":"BertPreTokenizer"}},
        "model":{{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##",
            "max_input_chars_per_word":100,
            "vocab":{{"[PAD]":0,"[UNK]":100,"[CLS]":101,"[SEP]":102,"hello":103,"##s":104}}}},
        "post_processor":{TEMPLATE_BODY},
        "decoder":{{"type":"WordPiece","prefix":"##","cleanup":true}}
    }}"###
    );
    let a = load_ok(&doc);
    assert!(matches!(a.model, ModelConfig::WordPiece { .. }));
    assert!(matches!(a.normalizer, Some(NormalizerConfig::Bert { .. })));
    assert!(matches!(
        a.post_processor,
        Some(PostProcessorConfig::Template { .. })
    ));
    assert_eq!(a.truncation.unwrap().max_length, 512);
    assert_eq!(a.padding.unwrap().strategy, PaddingStrategy::BatchLongest);
}

// ── The §8 message contract, all nine rows ──────────────────────────────

#[test]
fn every_error_row_names_its_promised_context() {
    // Json: byte offset.
    let e = TokenizerError::Json {
        at: 17,
        what: "duplicate key \"a\" in object".to_string(),
    };
    assert!(e.to_string().contains("at byte 17"), "{e}");

    // Schema: JSON path.
    let e = TokenizerError::Schema {
        path: "added_tokens[3].content".to_string(),
        what: "expected a string, found null".to_string(),
    };
    assert!(e.to_string().contains("added_tokens[3].content"), "{e}");

    // UnknownTag: family, tag, pinned reference.
    let e = TokenizerError::UnknownTag {
        family: "decoder",
        tag: "Quantum".to_string(),
    };
    let msg = e.to_string();
    assert!(msg.contains("decoder"), "{msg}");
    assert!(msg.contains("Quantum"), "{msg}");
    assert!(msg.contains("0.21"), "{msg}");

    // UnsupportedField: path and reason.
    let e = TokenizerError::UnsupportedField {
        path: "model.dropout".to_string(),
        why: "BPE dropout is train-time".to_string(),
    };
    let msg = e.to_string();
    assert!(
        msg.contains("model.dropout") && msg.contains("train-time"),
        "{msg}"
    );

    // Vocab: the offending token/id.
    let e = TokenizerError::Vocab {
        what: "token \"x\" reuses id 7".to_string(),
    };
    assert!(e.to_string().contains("\"x\""), "{e}");

    // Charsmap: blob offset.
    let e = TokenizerError::Charsmap {
        at: 40,
        what: "out-of-bounds trie transition".to_string(),
    };
    assert!(e.to_string().contains("at offset 40"), "{e}");

    // RegexConstruct: pattern and construct, both named.
    let e = TokenizerError::RegexConstruct {
        pattern: "(?<=a)b".to_string(),
        construct: "lookbehind (?<=…)".to_string(),
    };
    let msg = e.to_string();
    assert!(
        msg.contains("(?<=a)b") && msg.contains("lookbehind"),
        "{msg}"
    );

    // Encode: the stage and fault text travel through.
    let e = TokenizerError::Encode {
        what: "Split pattern failed on segment 3".to_string(),
    };
    assert!(e.to_string().contains("segment 3"), "{e}");

    // Decode: id and vocab size.
    let e = TokenizerError::Decode {
        id: 99999,
        vocab_size: 50257,
    };
    let msg = e.to_string();
    assert!(msg.contains("99999") && msg.contains("50257"), "{msg}");
}

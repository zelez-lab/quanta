//! The BPE runtime against hand-built minimal artifacts loaded through
//! the REAL artifact loader: merge-rank determinism (including the
//! leftmost tie rule), `ignore_merges`, `byte_fallback` round-trips,
//! unk fusion, prefix/suffix mechanics, pre-token-local offsets, and
//! the load-time/tokenize-time fault split the pinned reference draws.

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::artifact::TokenizerArtifact;
use quanta_tokenizers::bpe::Bpe;
use quanta_tokenizers::model::{Model, ModelToken};

/// Load a whole artifact around the given model body (the real
/// loader validates it) and build the BPE runtime from its config.
fn bpe(model: &str) -> Bpe {
    let doc = format!(r#"{{"version":"1.0","model":{model}}}"#);
    let artifact = TokenizerArtifact::from_bytes(doc.as_bytes())
        .unwrap_or_else(|e| panic!("artifact must load, got: {e}"));
    Bpe::from_config(artifact.model)
}

fn load_model(model: &str) -> Result<TokenizerArtifact, TokenizerError> {
    let doc = format!(r#"{{"version":"1.0","model":{model}}}"#);
    TokenizerArtifact::from_bytes(doc.as_bytes())
}

fn t(id: u32, value: &str, offsets: (usize, usize)) -> ModelToken {
    ModelToken {
        id,
        value: value.to_string(),
        offsets,
    }
}

fn tokens(model: &Bpe, pretoken: &str) -> Vec<ModelToken> {
    model
        .tokenize(pretoken)
        .unwrap_or_else(|e| panic!("tokenize({pretoken:?}) must succeed, got: {e}"))
}

/// Every token's offsets are contiguous byte ranges that tile the
/// whole pre-token (valid only for configs that drop nothing).
fn assert_offsets_tile(model: &Bpe, pretoken: &str) {
    let toks = tokens(model, pretoken);
    let mut pos = 0;
    for tok in &toks {
        assert_eq!(
            tok.offsets.0, pos,
            "token {tok:?} is not contiguous in {pretoken:?}"
        );
        pos = tok.offsets.1;
    }
    assert_eq!(pos, pretoken.len(), "offsets must tile {pretoken:?}");
}

// ── Merge order ─────────────────────────────────────────────────────────

#[test]
fn merge_rank_decides_the_segmentation() {
    // Same vocab, same pairs — only the list ORDER (= rank) differs,
    // and the segmentation flips with it.
    let bc_first = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4},
            "merges":[["b","c"],["a","b"]]}"#);
    assert_eq!(
        tokens(&bc_first, "abc"),
        [t(0, "a", (0, 1)), t(4, "bc", (1, 3))]
    );

    let ab_first = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4},
            "merges":[["a","b"],["b","c"]]}"#);
    // The (b, c) entry is stale once "b" is absorbed — the runtime
    // must detect the expired pair, not apply it.
    assert_eq!(
        tokens(&ab_first, "abc"),
        [t(3, "ab", (0, 2)), t(2, "c", (2, 3))]
    );
}

#[test]
fn merges_cascade_through_freshly_merged_symbols() {
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},
            "merges":[["a","b"],["ab","c"]]}"#);
    assert_eq!(tokens(&model, "abc"), [t(4, "abc", (0, 3))]);

    let pairs = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"abab":3},
            "merges":[["a","b"],["ab","ab"]]}"#);
    assert_eq!(tokens(&pairs, "abab"), [t(3, "abab", (0, 4))]);
}

#[test]
fn equal_ranks_resolve_leftmost_first() {
    // "aaa" has the (a, a) pair pending at two positions with ONE
    // rank; the reference merges the leftmost, giving "aa a", never
    // "a aa".
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0,"aa":1},"merges":[["a","a"]]}"#);
    assert_eq!(
        tokens(&model, "aaa"),
        [t(1, "aa", (0, 2)), t(0, "a", (2, 3))]
    );
    assert_eq!(
        tokens(&model, "aaaaa"),
        [t(1, "aa", (0, 2)), t(1, "aa", (2, 4)), t(0, "a", (4, 5))]
    );
}

#[test]
fn legacy_and_pair_merge_spellings_build_the_same_runtime() {
    let legacy = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},
            "merges":["a b","ab c"]}"#);
    let pairs = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"abc":4},
            "merges":[["a","b"],["ab","c"]]}"#);
    assert_eq!(tokens(&legacy, "abc"), tokens(&pairs, "abc"));
    assert_eq!(tokens(&legacy, "abc"), [t(4, "abc", (0, 3))]);
}

// ── ignore_merges ───────────────────────────────────────────────────────

#[test]
fn ignore_merges_short_circuits_on_a_whole_word_vocab_hit() {
    let with =
        bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[],"ignore_merges":true}"#);
    assert_eq!(tokens(&with, "ab"), [t(2, "ab", (0, 2))]);
    // A miss falls through to the normal per-character path.
    assert_eq!(tokens(&with, "ba"), [t(1, "b", (0, 1)), t(0, "a", (1, 2))]);

    // Without the flag the same word splits (no merges apply).
    let without = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[]}"#);
    assert_eq!(
        tokens(&without, "ab"),
        [t(0, "a", (0, 1)), t(1, "b", (1, 2))]
    );
}

#[test]
fn empty_pretoken_tokenizes_to_nothing_even_under_ignore_merges() {
    // The reference checks emptiness BEFORE the ignore_merges vocab
    // hit — an empty vocab entry never surfaces.
    let model = bpe(r#"{"type":"BPE","vocab":{"":9,"a":0},"merges":[],"ignore_merges":true}"#);
    assert!(tokens(&model, "").is_empty());
}

// ── byte_fallback ───────────────────────────────────────────────────────

#[test]
fn byte_fallback_emits_one_token_per_byte_and_round_trips() {
    let model = bpe(r#"{"type":"BPE","vocab":{"<0xC3>":0,"<0xA9>":1,"a":2},
            "merges":[],"byte_fallback":true}"#);
    // "é" is not in the vocab; its UTF-8 bytes C3 A9 are.
    let toks = tokens(&model, "aé");
    assert_eq!(
        toks,
        [
            t(2, "a", (0, 1)),
            t(0, "<0xC3>", (1, 2)),
            t(1, "<0xA9>", (2, 3))
        ]
    );
    // Round-trip: the byte tokens decode back to the exact bytes.
    let bytes: Vec<u8> = toks[1..]
        .iter()
        .map(|tok| {
            let hex = tok
                .value
                .strip_prefix("<0x")
                .and_then(|v| v.strip_suffix('>'))
                .unwrap_or_else(|| panic!("{:?} is not a byte token", tok.value));
            u8::from_str_radix(hex, 16).unwrap()
        })
        .collect();
    assert_eq!(String::from_utf8(bytes).unwrap(), "é");
}

#[test]
fn byte_fallback_spelling_is_two_digit_uppercase_hex() {
    // The reference writes `<0xXX>` with uppercase hex, zero-padded to
    // two digits — "<0x0A>", never "<0xa>" or "<0x0a>".
    let model = bpe(r#"{"type":"BPE","vocab":{"<0x0A>":0},"merges":[],"byte_fallback":true}"#);
    assert_eq!(tokens(&model, "\n"), [t(0, "<0x0A>", (0, 1))]);
}

#[test]
fn byte_fallback_needs_every_byte_or_falls_through_to_unk() {
    let model = bpe(r#"{"type":"BPE","vocab":{"<unk>":0,"<0x61>":1},
            "merges":[],"byte_fallback":true,"unk_token":"<unk>"}"#);
    // 'a' falls back to its byte token; 'b' has no byte token, so the
    // whole piece becomes the unk token.
    assert_eq!(
        tokens(&model, "ab"),
        [t(1, "<0x61>", (0, 1)), t(0, "<unk>", (1, 2))]
    );
}

#[test]
fn byte_fallback_does_not_flush_a_pending_unk_run() {
    // Reference fidelity (merge_word's flush discipline): a pending
    // unk run is flushed only by a vocab hit or end-of-word. Byte
    // tokens therefore land AHEAD of a pending unk, and offsets stay
    // cumulative rather than text-positional.
    let model = bpe(r#"{"type":"BPE","vocab":{"<unk>":9,"<0xC3>":0,"<0xA9>":1},
            "merges":[],"byte_fallback":true,"unk_token":"<unk>","fuse_unk":true}"#);
    assert_eq!(
        tokens(&model, "zé"),
        [
            t(0, "<0xC3>", (0, 1)),
            t(1, "<0xA9>", (1, 2)),
            t(9, "<unk>", (2, 3))
        ]
    );
}

// ── unk_token / fuse_unk ────────────────────────────────────────────────

#[test]
fn unknown_runs_fuse_into_one_token_under_fuse_unk() {
    let fused = bpe(r#"{"type":"BPE","vocab":{"a":0,"<unk>":1},
            "merges":[],"unk_token":"<unk>","fuse_unk":true}"#);
    assert_eq!(
        tokens(&fused, "xya"),
        [t(1, "<unk>", (0, 2)), t(0, "a", (2, 3))]
    );
    assert_eq!(
        tokens(&fused, "axy"),
        [t(0, "a", (0, 1)), t(1, "<unk>", (1, 3))]
    );
    // Byte lengths accumulate: "é" is 2 bytes, so "éx" fuses to one
    // unk spanning 3 bytes.
    assert_eq!(tokens(&fused, "éx"), [t(1, "<unk>", (0, 3))]);
}

#[test]
fn unknown_runs_stay_separate_without_fuse_unk() {
    // fuse_unk defaults to false in the reference builder (and the
    // loader): one unk token per unknown character.
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0,"<unk>":1},"merges":[],"unk_token":"<unk>"}"#);
    assert_eq!(
        tokens(&model, "xya"),
        [
            t(1, "<unk>", (0, 1)),
            t(1, "<unk>", (1, 2)),
            t(0, "a", (2, 3))
        ]
    );
}

#[test]
fn no_unk_token_drops_unknown_input_and_shifts_offsets() {
    // Reference fidelity: with no unk_token, unknown characters vanish
    // and later offsets shift left (offsets are cumulative byte
    // lengths of EMITTED symbols, not text positions).
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0},"merges":[]}"#);
    assert_eq!(
        tokens(&model, "axa"),
        [t(0, "a", (0, 1)), t(0, "a", (1, 2))]
    );
}

#[test]
fn unk_token_missing_from_the_vocab_faults_at_tokenize_time() {
    // The loader accepts an unk_token the vocab lacks (so does the
    // reference); the fault fires the first time an unknown piece
    // actually needs it — reference `UnkTokenOutOfVocabulary`.
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0},"merges":[],"unk_token":"<unk>"}"#);
    assert_eq!(tokens(&model, "aa"), [t(0, "a", (0, 1)), t(0, "a", (1, 2))]);
    let Err(TokenizerError::Encode { what }) = model.tokenize("ax") else {
        panic!("must fail with Encode");
    };
    assert!(what.contains("<unk>"), "{what}");
}

// ── continuing_subword_prefix / end_of_word_suffix ──────────────────────

#[test]
fn continuing_subword_prefix_shapes_lookups_merges_and_values() {
    let model = bpe(
        r###"{"type":"BPE","vocab":{"h":0,"##e":1,"##l":2,"he":3,"##ll":4,"hell":5},
            "merges":[["h","##e"],["##l","##l"],["he","##ll"]],
            "continuing_subword_prefix":"##"}"###,
    );
    // The merged spelling strips one prefix off the right side:
    // (he, ##ll) -> "hell".
    assert_eq!(tokens(&model, "hell"), [t(5, "hell", (0, 4))]);
    assert_eq!(tokens(&model, "he"), [t(3, "he", (0, 2))]);
    // Values are VOCAB spellings (prefix included); offsets keep
    // counting the unmodified pre-token bytes.
    assert_eq!(
        tokens(&model, "hel"),
        [t(3, "he", (0, 2)), t(2, "##l", (2, 3))]
    );
    // The first character is looked up bare; later ones only with the
    // prefix ("##h" is not in the vocab, so the second 'h' drops).
    assert_eq!(tokens(&model, "hh"), [t(0, "h", (0, 1))]);
}

#[test]
fn end_of_word_suffix_applies_to_the_last_piece_only() {
    let model = bpe(
        r#"{"type":"BPE","vocab":{"a":0,"b</w>":1,"ab</w>":2,"a</w>":3},
            "merges":[["a","b</w>"]],"end_of_word_suffix":"</w>"}"#,
    );
    assert_eq!(tokens(&model, "ab"), [t(2, "ab</w>", (0, 2))]);
    // A single character is first AND last: suffixed, never prefixed.
    assert_eq!(tokens(&model, "a"), [t(3, "a</w>", (0, 1))]);
    assert_eq!(
        tokens(&model, "aa"),
        [t(0, "a", (0, 1)), t(3, "a</w>", (1, 2))]
    );
}

// ── Offsets within the pre-token ────────────────────────────────────────

#[test]
fn offsets_tile_the_pretoken_across_configurations() {
    // Non-dropping configurations: every token stream is contiguous
    // and covers the pre-token byte range exactly.
    let plain = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4},
            "merges":[["a","b"],["b","c"]]}"#);
    for input in ["abc", "cba", "aabbcc", "a"] {
        assert_offsets_tile(&plain, input);
    }
    // Pure config, all-known input: the values also concatenate back
    // to the pre-token (no prefix/suffix/unk respelling in play).
    let concat: String = tokens(&plain, "abc")
        .iter()
        .map(|t| t.value.as_str())
        .collect();
    assert_eq!(concat, "abc");

    let fused = bpe(r#"{"type":"BPE","vocab":{"a":0,"<unk>":1},
            "merges":[],"unk_token":"<unk>","fuse_unk":true}"#);
    for input in ["xya", "axy", "éxé", "xyz"] {
        assert_offsets_tile(&fused, input);
    }

    let bytes = bpe(r#"{"type":"BPE","vocab":{"<0xC3>":0,"<0xA9>":1,"a":2},
            "merges":[],"byte_fallback":true}"#);
    for input in ["aé", "éa", "ééé"] {
        assert_offsets_tile(&bytes, input);
    }

    let prefixed = bpe(
        r###"{"type":"BPE","vocab":{"h":0,"##e":1,"##l":2,"he":3,"##ll":4,"hell":5},
            "merges":[["h","##e"],["##l","##l"],["he","##ll"]],
            "continuing_subword_prefix":"##"}"###,
    );
    for input in ["hell", "hel", "he"] {
        assert_offsets_tile(&prefixed, input);
    }
}

// ── Hostile configs stop at the loader ──────────────────────────────────

#[test]
fn merge_naming_an_absent_token_errors_at_load_not_here() {
    // The model layer trusts validated config: these artifacts never
    // reach `Bpe::from_config` — the loader's §8 Vocab row stops them.
    let Err(TokenizerError::Vocab { what }) =
        load_model(r#"{"type":"BPE","vocab":{"a":0},"merges":[["a","b"]]}"#)
    else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("\"b\""), "{what}");

    // The merged RESULT must be in the vocab too.
    let Err(TokenizerError::Vocab { what }) =
        load_model(r#"{"type":"BPE","vocab":{"a":0,"b":1},"merges":[["a","b"]]}"#)
    else {
        panic!("must fail with Vocab");
    };
    assert!(what.contains("\"ab\""), "{what}");
}

// ── The trait surface ───────────────────────────────────────────────────

#[test]
fn vocab_lookups_round_trip_through_the_model_trait() {
    let model = bpe(r#"{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}"#);
    // Object-safe: the pipeline drives models through `dyn Model`.
    let model: &dyn Model = &model;
    assert_eq!(model.token_to_id("ab"), Some(2));
    assert_eq!(model.id_to_token(2), Some("ab"));
    assert_eq!(model.token_to_id("missing"), None);
    assert_eq!(model.id_to_token(9), None);
}

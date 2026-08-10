//! The WordPiece model against hand-built artifacts through the real
//! loader: longest-match-first, `##` continuation values vs raw byte
//! offsets, the `max_input_chars_per_word` boundary (exactly at /
//! over), the all-or-nothing unk collapse, and the load-time
//! `unk_token ∈ vocab` guard with its message contract.
//!
//! Every expected token list below was produced by the pinned
//! reference (HF `tokenizers` 0.21.2, model-level `tokenize`) on the
//! same vocab/parameters — reference-produced, not hand-derived.

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::artifact::{ModelConfig, TokenizerArtifact};
use quanta_tokenizers::model::{Model, ModelToken};
use quanta_tokenizers::wordpiece::WordPiece;

/// Load a `{"model": …}` artifact through the real loader and build
/// the model from the parsed config — the seam the orchestrator uses.
fn load(model_json: &str) -> WordPiece {
    let doc = format!(r#"{{"version":"1.0","model":{model_json}}}"#);
    let artifact = TokenizerArtifact::from_bytes(doc.as_bytes())
        .unwrap_or_else(|e| panic!("artifact must load, got: {e}"));
    let ModelConfig::WordPiece {
        vocab,
        unk_token,
        continuing_subword_prefix,
        max_input_chars_per_word,
    } = artifact.model
    else {
        panic!("must parse as WordPiece");
    };
    WordPiece::new(
        vocab,
        unk_token,
        continuing_subword_prefix,
        max_input_chars_per_word,
    )
    .unwrap_or_else(|e| panic!("model must build, got: {e}"))
}

fn tok(id: u32, value: &str, offsets: (usize, usize)) -> ModelToken {
    ModelToken {
        id,
        value: value.to_string(),
        offsets,
    }
}

/// The BERT-class vocab used across the greedy-match tests.
const BERT_ISH: &str = r###"{"type":"WordPiece",
    "vocab":{"[UNK]":0,"un":1,"##aff":2,"##able":3,"a":4,"##ff":5},
    "unk_token":"[UNK]","continuing_subword_prefix":"##",
    "max_input_chars_per_word":100}"###;

// ── Greedy longest-match-first ──────────────────────────────────────────

#[test]
fn longest_match_wins_and_continuations_carry_the_prefix() {
    // The classic: "unaffable" → un ##aff ##able. "##a" and "##ff"
    // never appear even though both are viable shorter matches;
    // values carry "##", offsets stay raw byte positions.
    let wp = load(BERT_ISH);
    assert_eq!(
        wp.tokenize("unaffable").unwrap(),
        [
            tok(1, "un", (0, 2)),
            tok(2, "##aff", (2, 5)),
            tok(3, "##able", (5, 9)),
        ]
    );
}

#[test]
fn whole_word_beats_any_split() {
    // "abc" is in the vocab, so "ab" + "##c" never gets a chance.
    let wp = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"ab":1,"abc":2,"##c":3},
            "unk_token":"[UNK]"}"###,
    );
    assert_eq!(wp.tokenize("abc").unwrap(), [tok(2, "abc", (0, 3))]);
}

#[test]
fn continuation_positions_are_longest_first_too() {
    let wp = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"x":1,"##yz":2,"##y":3,"##z":4},
            "unk_token":"[UNK]"}"###,
    );
    assert_eq!(
        wp.tokenize("xyz").unwrap(),
        [tok(1, "x", (0, 1)), tok(2, "##yz", (1, 3))]
    );
}

// ── The max_input_chars_per_word boundary ───────────────────────────────

/// A 3-char cap over a multibyte vocab: the guard counts CHARACTERS
/// while offsets count bytes.
const ACCENTED: &str = r###"{"type":"WordPiece","vocab":{"[UNK]":0,"é":1,"##é":2},
    "unk_token":"[UNK]","max_input_chars_per_word":3}"###;

#[test]
fn exactly_at_max_input_chars_tokenizes_normally() {
    // "ééé" is 3 chars / 6 bytes — at the cap, not over it.
    let wp = load(ACCENTED);
    assert_eq!(
        wp.tokenize("ééé").unwrap(),
        [
            tok(1, "é", (0, 2)),
            tok(2, "##é", (2, 4)),
            tok(2, "##é", (4, 6)),
        ]
    );
}

#[test]
fn over_max_input_chars_is_one_unk_spanning_the_bytes() {
    // 4 chars > 3: whole-word unk, offsets over the 8 BYTES, vocab
    // never consulted.
    let wp = load(ACCENTED);
    assert_eq!(wp.tokenize("éééé").unwrap(), [tok(0, "[UNK]", (0, 8))]);
}

#[test]
fn ascii_cap_boundary_at_and_over() {
    let at = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"ab":1},"unk_token":"[UNK]",
            "max_input_chars_per_word":2}"###,
    );
    assert_eq!(at.tokenize("ab").unwrap(), [tok(1, "ab", (0, 2))]);
    let over = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"ab":1},"unk_token":"[UNK]",
            "max_input_chars_per_word":1}"###,
    );
    assert_eq!(over.tokenize("ab").unwrap(), [tok(0, "[UNK]", (0, 2))]);
}

// ── The all-or-nothing unk collapse ─────────────────────────────────────

#[test]
fn any_unmatched_position_collapses_the_whole_word() {
    // "a" matches, then "x" matches nothing: the reference's is_bad
    // path discards the "a" token and emits ONE unk over the word.
    let wp = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"a":1,"##b":2},
            "unk_token":"[UNK]"}"###,
    );
    assert_eq!(wp.tokenize("axb").unwrap(), [tok(0, "[UNK]", (0, 3))]);
    // Unmatched at the FIRST position takes the same path.
    assert_eq!(wp.tokenize("za").unwrap(), [tok(0, "[UNK]", (0, 2))]);
}

#[test]
fn empty_pretoken_is_no_tokens() {
    let wp = load(BERT_ISH);
    assert_eq!(wp.tokenize("").unwrap(), []);
}

// ── The prefix is config, not a constant ────────────────────────────────

#[test]
fn custom_continuing_prefix_and_custom_unk() {
    let wp = load(
        r###"{"type":"WordPiece","vocab":{"<u>":0,"fo":1,"@@o":2},"unk_token":"<u>",
            "continuing_subword_prefix":"@@"}"###,
    );
    assert_eq!(
        wp.tokenize("foo").unwrap(),
        [tok(1, "fo", (0, 2)), tok(2, "@@o", (2, 3))]
    );
    // The unk value is the configured token.
    assert_eq!(wp.tokenize("zz").unwrap(), [tok(0, "<u>", (0, 2))]);
}

#[test]
fn empty_continuing_prefix_matches_plain_pieces_everywhere() {
    let wp = load(
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"a":1,"b":2},"unk_token":"[UNK]",
            "continuing_subword_prefix":""}"###,
    );
    assert_eq!(
        wp.tokenize("ab").unwrap(),
        [tok(1, "a", (0, 1)), tok(2, "b", (1, 2))]
    );
}

#[test]
fn loader_defaults_ride_through_to_behavior() {
    // No unk_token / prefix / cap in the artifact: the reference
    // builder defaults ([UNK], ##, 100) drive tokenization.
    let wp = load(r###"{"type":"WordPiece","vocab":{"[UNK]":0,"un":1,"##aff":2,"##able":3}}"###);
    assert_eq!(
        wp.tokenize("unaffable").unwrap(),
        [
            tok(1, "un", (0, 2)),
            tok(2, "##aff", (2, 5)),
            tok(3, "##able", (5, 9)),
        ]
    );
    assert_eq!(wp.tokenize("zz").unwrap(), [tok(0, "[UNK]", (0, 2))]);
}

// ── Lookups ─────────────────────────────────────────────────────────────

#[test]
fn token_and_id_lookups_round_trip() {
    let wp = load(BERT_ISH);
    assert_eq!(wp.token_to_id("##able"), Some(3));
    assert_eq!(wp.token_to_id("able"), None);
    assert_eq!(wp.id_to_token(5), Some("##ff"));
    assert_eq!(wp.id_to_token(99), None);
}

// ── Constructor guards (message contract) ───────────────────────────────

#[test]
fn constructor_requires_unk_token_in_vocab() {
    let err = WordPiece::new(
        vec![("a".to_string(), 0)],
        "[UNK]".to_string(),
        "##".to_string(),
        100,
    )
    .unwrap_err();
    let TokenizerError::Vocab { what } = err else {
        panic!("must fail with Vocab, got: {err}");
    };
    assert!(what.contains("[UNK]"), "{what}");
}

#[test]
fn constructor_rejects_duplicate_tokens_and_id_collisions() {
    let dup = WordPiece::new(
        vec![("a".to_string(), 0), ("a".to_string(), 1)],
        "a".to_string(),
        "##".to_string(),
        100,
    )
    .unwrap_err();
    let TokenizerError::Vocab { what } = dup else {
        panic!("must fail with Vocab, got: {dup}");
    };
    assert!(what.contains("\"a\""), "{what}");

    let collide = WordPiece::new(
        vec![("a".to_string(), 0), ("b".to_string(), 0)],
        "a".to_string(),
        "##".to_string(),
        100,
    )
    .unwrap_err();
    let TokenizerError::Vocab { what } = collide else {
        panic!("must fail with Vocab, got: {collide}");
    };
    assert!(what.contains("id 0"), "{what}");
}

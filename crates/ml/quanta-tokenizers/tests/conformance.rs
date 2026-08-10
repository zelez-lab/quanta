//! Fixture-differential conformance against the pinned reference —
//! THE correctness contract (scope §9/§10).
//!
//! Every committed artifact under `tests/fixtures/tok/` loads through
//! the public API and replays its committed vector file: for each
//! curated input (single and pair, `add_special_tokens` both ways)
//! the FULL encoding record is asserted byte-exact against what the
//! pinned HF `tokenizers` produced — ids, tokens, offsets, type_ids,
//! special_tokens_mask, attention_mask, word_ids — plus both decode
//! round-trips (`skip_special_tokens` true and false).
//!
//! Offsets in the vector files are BYTE offsets into the original
//! input: the Python binding reports char offsets, and the generator
//! converts them deterministically against the known input text (a
//! cumulative UTF-8 byte-length map, per sequence for pairs; special
//! tokens keep the reference's `(0, 0)` placeholder span). See
//! `tests/fixtures/tok/gen_fixtures.py` — the committed bytes are the
//! contract; CI never runs Python.
//!
//! One test per hand-built model family (BPE with byte_fallback,
//! WordPiece, Unigram, WordLevel) and one per real full-size anchor
//! (gpt2, bert-base-uncased). Failure context names the vector file,
//! case index, input, and field.

use quanta_tokenizers::json::{self, Value};
use quanta_tokenizers::{Encoding, Tokenizer};

/// Read a committed fixture. The fixtures ship with this harness; an
/// absent file is a broken checkout, not a pending maintainer step.
fn fixture(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/tok")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "fixture {name} unreadable ({e}) — regenerate with \
             tests/fixtures/tok/gen_fixtures.py (pinned tokenizers) and commit"
        )
    })
}

// ── Vector-file field extraction (via the crate's own JSON parser) ──────

fn field<'a>(case: &'a Value, key: &str, ctx: &str) -> &'a Value {
    case.get(key)
        .unwrap_or_else(|| panic!("{ctx}: vector field `{key}` missing"))
}

fn str_field<'a>(case: &'a Value, key: &str, ctx: &str) -> &'a str {
    field(case, key, ctx)
        .as_str()
        .unwrap_or_else(|| panic!("{ctx}: vector field `{key}` is not a string"))
}

fn bool_field(case: &Value, key: &str, ctx: &str) -> bool {
    field(case, key, ctx)
        .as_bool()
        .unwrap_or_else(|| panic!("{ctx}: vector field `{key}` is not a bool"))
}

fn elems<'a>(case: &'a Value, key: &str, ctx: &str) -> &'a [Value] {
    field(case, key, ctx)
        .as_array()
        .unwrap_or_else(|| panic!("{ctx}: vector field `{key}` is not an array"))
}

fn u32_list(case: &Value, key: &str, ctx: &str) -> Vec<u32> {
    elems(case, key, ctx)
        .iter()
        .map(|v| {
            v.as_number()
                .and_then(|n| n.as_u32())
                .unwrap_or_else(|| panic!("{ctx}: `{key}` element is not a u32"))
        })
        .collect()
}

fn string_list(case: &Value, key: &str, ctx: &str) -> Vec<String> {
    elems(case, key, ctx)
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("{ctx}: `{key}` element is not a string"))
                .to_string()
        })
        .collect()
}

fn offset_list(case: &Value, key: &str, ctx: &str) -> Vec<(usize, usize)> {
    elems(case, key, ctx)
        .iter()
        .map(|v| {
            let pair: Vec<usize> = v
                .as_array()
                .map(|span| {
                    span.iter()
                        .filter_map(|end| end.as_number().and_then(|n| n.as_usize()))
                        .collect()
                })
                .unwrap_or_default();
            match pair.as_slice() {
                [start, end] => (*start, *end),
                _ => panic!("{ctx}: `{key}` element is not a [start, end] span"),
            }
        })
        .collect()
}

fn word_id_list(case: &Value, key: &str, ctx: &str) -> Vec<Option<u32>> {
    elems(case, key, ctx)
        .iter()
        .map(|v| {
            if v.is_null() {
                None
            } else {
                Some(
                    v.as_number()
                        .and_then(|n| n.as_u32())
                        .unwrap_or_else(|| panic!("{ctx}: `{key}` element is not null/u32")),
                )
            }
        })
        .collect()
}

// ── The differential driver ─────────────────────────────────────────────

fn check_case(tok: &Tokenizer, case: &Value, index: usize, file: &str) {
    let kind = str_field(case, "kind", file);
    let text = str_field(case, "text", file);
    let ctx = format!("{file} case {index} ({kind} {text:?})");

    let add_special = bool_field(case, "add_special_tokens", &ctx);
    let encoding: Encoding = match kind {
        "single" => tok.encode(text, add_special),
        "pair" => tok.encode_pair(text, str_field(case, "text_pair", &ctx), add_special),
        other => panic!("{ctx}: unknown case kind {other:?}"),
    }
    .unwrap_or_else(|e| panic!("{ctx}: encode failed: {e}"));

    let ids = u32_list(case, "ids", &ctx);
    assert_eq!(encoding.ids(), ids.as_slice(), "{ctx}: ids");
    assert_eq!(
        encoding.tokens(),
        string_list(case, "tokens", &ctx).as_slice(),
        "{ctx}: tokens"
    );
    assert_eq!(
        encoding.offsets(),
        offset_list(case, "offsets", &ctx).as_slice(),
        "{ctx}: offsets (byte spans into the original input)"
    );
    assert_eq!(
        encoding.type_ids(),
        u32_list(case, "type_ids", &ctx).as_slice(),
        "{ctx}: type_ids"
    );
    assert_eq!(
        encoding.special_tokens_mask(),
        u32_list(case, "special_tokens_mask", &ctx).as_slice(),
        "{ctx}: special_tokens_mask"
    );
    assert_eq!(
        encoding.attention_mask(),
        u32_list(case, "attention_mask", &ctx).as_slice(),
        "{ctx}: attention_mask"
    );
    assert_eq!(
        encoding.word_ids(),
        word_id_list(case, "word_ids", &ctx).as_slice(),
        "{ctx}: word_ids"
    );

    let decoded = tok
        .decode(&ids, true)
        .unwrap_or_else(|e| panic!("{ctx}: decode(skip_special_tokens=true) failed: {e}"));
    assert_eq!(
        decoded,
        str_field(case, "decoded", &ctx),
        "{ctx}: decode(skip_special_tokens=true)"
    );
    let decoded_raw = tok
        .decode(&ids, false)
        .unwrap_or_else(|e| panic!("{ctx}: decode(skip_special_tokens=false) failed: {e}"));
    assert_eq!(
        decoded_raw,
        str_field(case, "decoded_raw", &ctx),
        "{ctx}: decode(skip_special_tokens=false)"
    );
}

fn run_conformance(artifact: &str, vectors: &str) {
    let tok = Tokenizer::from_bytes(&fixture(artifact))
        .unwrap_or_else(|e| panic!("{artifact} must load (loads-means-runs, scope §7): {e}"));
    let doc = json::parse(&fixture(vectors))
        .unwrap_or_else(|e| panic!("{vectors} must parse as JSON: {e}"));
    let cases = doc
        .get("cases")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{vectors}: no `cases` array"));
    assert!(!cases.is_empty(), "{vectors}: empty case list");
    for (index, case) in cases.iter().enumerate() {
        check_case(&tok, case, index, vectors);
    }
}

// ── One test per hand-built family, one per real anchor ─────────────────

#[test]
fn bpe_minimal_artifact_matches_reference() {
    run_conformance("tiny_bpe.tokenizer.json", "vectors/tiny_bpe.vectors.json");
}

#[test]
fn wordpiece_minimal_artifact_matches_reference() {
    run_conformance(
        "tiny_wordpiece.tokenizer.json",
        "vectors/tiny_wordpiece.vectors.json",
    );
}

#[test]
fn unigram_minimal_artifact_matches_reference() {
    run_conformance(
        "tiny_unigram.tokenizer.json",
        "vectors/tiny_unigram.vectors.json",
    );
}

#[test]
fn wordlevel_minimal_artifact_matches_reference() {
    run_conformance(
        "tiny_wordlevel.tokenizer.json",
        "vectors/tiny_wordlevel.vectors.json",
    );
}

#[test]
fn gpt2_real_anchor_matches_reference() {
    run_conformance("gpt2.tokenizer.json", "vectors/gpt2.vectors.json");
}

#[test]
fn bert_base_uncased_real_anchor_matches_reference() {
    run_conformance(
        "bert-base-uncased.tokenizer.json",
        "vectors/bert-base-uncased.vectors.json",
    );
}

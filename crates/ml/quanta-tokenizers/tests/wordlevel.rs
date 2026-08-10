//! The WordLevel runtime against hand-built minimal artifacts loaded
//! through the REAL artifact loader: exact lookup, the unk path with
//! whole-pre-token offsets, the reference's tokenize-time
//! missing-unk fault, and the loader-default unk spelling.

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::artifact::TokenizerArtifact;
use quanta_tokenizers::model::{Model, ModelToken};
use quanta_tokenizers::wordlevel::WordLevel;

/// Load a whole artifact around the given model body (the real
/// loader validates it) and build the WordLevel runtime.
fn wordlevel(model: &str) -> WordLevel {
    let doc = format!(r#"{{"version":"1.0","model":{model}}}"#);
    let artifact = TokenizerArtifact::from_bytes(doc.as_bytes())
        .unwrap_or_else(|e| panic!("artifact must load, got: {e}"));
    WordLevel::from_config(artifact.model)
}

fn t(id: u32, value: &str, offsets: (usize, usize)) -> ModelToken {
    ModelToken {
        id,
        value: value.to_string(),
        offsets,
    }
}

fn tokens(model: &WordLevel, pretoken: &str) -> Vec<ModelToken> {
    model
        .tokenize(pretoken)
        .unwrap_or_else(|e| panic!("tokenize({pretoken:?}) must succeed, got: {e}"))
}

#[test]
fn exact_lookup_yields_the_single_vocab_token() {
    let model = wordlevel(
        r#"{"type":"WordLevel","vocab":{"hello":0,"world":1,"<unk>":2},"unk_token":"<unk>"}"#,
    );
    assert_eq!(tokens(&model, "hello"), [t(0, "hello", (0, 5))]);
    assert_eq!(tokens(&model, "world"), [t(1, "world", (0, 5))]);
}

#[test]
fn offsets_are_byte_ranges_into_the_pretoken() {
    // "héllo" is 6 bytes — offsets count bytes, not characters, and
    // always span the whole pre-token.
    let model =
        wordlevel(r#"{"type":"WordLevel","vocab":{"héllo":0,"<unk>":1},"unk_token":"<unk>"}"#);
    assert_eq!(tokens(&model, "héllo"), [t(0, "héllo", (0, 6))]);
    assert_eq!(tokens(&model, "wörld"), [t(1, "<unk>", (0, 6))]);
}

#[test]
fn unknown_pretoken_becomes_the_unk_spelling_spanning_it() {
    let model =
        wordlevel(r#"{"type":"WordLevel","vocab":{"hello":0,"<unk>":2},"unk_token":"<unk>"}"#);
    // The value is the unk VOCAB spelling; the offsets still cover the
    // whole unknown pre-token.
    assert_eq!(tokens(&model, "goodbye"), [t(2, "<unk>", (0, 7))]);
}

#[test]
fn absent_unk_token_field_takes_the_reference_default() {
    // The loader fills the reference builder's default "<unk>" when
    // the artifact omits the field; the runtime then resolves it.
    let model = wordlevel(r#"{"type":"WordLevel","vocab":{"a":0,"<unk>":7}}"#);
    assert_eq!(tokens(&model, "b"), [t(7, "<unk>", (0, 1))]);
}

#[test]
fn empty_pretoken_follows_the_unk_path() {
    // The reference does not special-case the empty pre-token: "" is
    // just a word the vocab lacks.
    let model = wordlevel(r#"{"type":"WordLevel","vocab":{"a":0,"<unk>":1},"unk_token":"<unk>"}"#);
    assert_eq!(tokens(&model, ""), [t(1, "<unk>", (0, 0))]);
}

#[test]
fn unk_token_missing_from_the_vocab_faults_only_when_needed() {
    // The loader accepts an unk_token the vocab lacks (so does the
    // reference); known words still tokenize, and the fault fires the
    // first time an unknown word needs the unk — the reference's
    // `MissingUnkToken`, at tokenize time.
    let model = wordlevel(r#"{"type":"WordLevel","vocab":{"a":0},"unk_token":"<unk>"}"#);
    assert_eq!(tokens(&model, "a"), [t(0, "a", (0, 1))]);
    let Err(TokenizerError::Encode { what }) = model.tokenize("b") else {
        panic!("must fail with Encode");
    };
    assert!(what.contains("<unk>"), "{what}");
}

#[test]
fn vocab_lookups_round_trip_through_the_model_trait() {
    let model =
        wordlevel(r#"{"type":"WordLevel","vocab":{"hello":0,"<unk>":2},"unk_token":"<unk>"}"#);
    // Object-safe: the pipeline drives models through `dyn Model`.
    let model: &dyn Model = &model;
    assert_eq!(model.token_to_id("hello"), Some(0));
    assert_eq!(model.id_to_token(0), Some("hello"));
    assert_eq!(model.token_to_id("missing"), None);
    assert_eq!(model.id_to_token(9), None);
}

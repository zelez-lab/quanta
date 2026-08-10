//! The Unigram model against hand-built artifacts through the real
//! loader: multi-path lattices with known best segmentations, exact
//! and one-ulp tie determinism, unknown-character handling (with and
//! without `unk_id`), the `byte_fallback` interplay, and multibyte
//! pieces keeping offsets byte-correct.
//!
//! Every expected token list below was produced by the pinned
//! reference (HF `tokenizers` 0.21.2, model-level `tokenize`) on the
//! same table — reference-produced, not hand-derived. The two tie
//! tests pin the analysis documented on [`quanta_tokenizers::unigram`]:
//! path scores are per-path left folds in `f64`, so the port inherits
//! the reference's arithmetic (bit-identical sums) and its
//! strictly-greater update rule (first writer survives exact ties).

use quanta_tokenizers::TokenizerError;
use quanta_tokenizers::artifact::{ModelConfig, TokenizerArtifact};
use quanta_tokenizers::model::{Model, ModelToken};
use quanta_tokenizers::unigram::Unigram;

/// Load a `{"model": …}` artifact through the real loader and build
/// the model from the parsed config — the seam the orchestrator uses.
fn load(model_json: &str) -> Unigram {
    let doc = format!(r#"{{"version":"1.0","model":{model_json}}}"#);
    let artifact = TokenizerArtifact::from_bytes(doc.as_bytes())
        .unwrap_or_else(|e| panic!("artifact must load, got: {e}"));
    let ModelConfig::Unigram {
        vocab,
        unk_id,
        byte_fallback,
    } = artifact.model
    else {
        panic!("must parse as Unigram");
    };
    Unigram::new(vocab, unk_id, byte_fallback)
        .unwrap_or_else(|e| panic!("model must build, got: {e}"))
}

fn tok(id: u32, value: &str, offsets: (usize, usize)) -> ModelToken {
    ModelToken {
        id,
        value: value.to_string(),
        offsets,
    }
}

// ── The reference's own lattice vectors ─────────────────────────────────

/// The pinned reference's `encode` doctest table.
const DOCTEST_TABLE: &str = r#"{"type":"Unigram","unk_id":0,"vocab":[
    ["<unk>",0.0],["a",0.0],["b",0.0],["c",0.0],["d",0.0],
    ["cd",1.0],["ab",2.0],["abc",5.0],["abcd",10.0]]}"#;

#[test]
fn reference_doctest_lattice() {
    let u = load(DOCTEST_TABLE);
    assert_eq!(u.tokenize("abcd").unwrap(), [tok(8, "abcd", (0, 4))]);
    // Mixed known pieces and a trailing fused-unknown span.
    assert_eq!(
        u.tokenize("abcdacdxx").unwrap(),
        [
            tok(8, "abcd", (0, 4)),
            tok(1, "a", (4, 5)),
            tok(5, "cd", (5, 7)),
            tok(0, "xx", (7, 9)),
        ]
    );
}

#[test]
fn reference_test_encode2_matrix() {
    // The reference's `test_encode2` table (fuse_unk=true, optimized —
    // the deserialized-model posture), every input it pins for that
    // posture.
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",0.0],["ab",0.0],["cd",-0.1],["abc",-0.2],["a",-0.3],
            ["b",-0.4],["c",-0.5],["ABC",-0.5],["abcdabcd",20.0],
            ["q",20.5],["r",20.5],["qr",-0.5]]}"#,
    );
    assert_eq!(u.tokenize("abc").unwrap(), [tok(3, "abc", (0, 3))]);
    assert_eq!(u.tokenize("AB").unwrap(), [tok(0, "AB", (0, 2))]);
    assert_eq!(
        u.tokenize("abcd").unwrap(),
        [tok(1, "ab", (0, 2)), tok(2, "cd", (2, 4))]
    );
    assert_eq!(
        u.tokenize("abcc").unwrap(),
        [tok(3, "abc", (0, 3)), tok(6, "c", (3, 4))]
    );
    assert_eq!(
        u.tokenize("xabcabaabcdd").unwrap(),
        [
            tok(0, "x", (0, 1)),
            tok(3, "abc", (1, 4)),
            tok(1, "ab", (4, 6)),
            tok(4, "a", (6, 7)),
            tok(1, "ab", (7, 9)),
            tok(2, "cd", (9, 11)),
            tok(0, "d", (11, 12)),
        ]
    );
    // A whole run of unknown characters (ASCII then CJK) fuses into
    // ONE span — 3 chars + 2 three-byte chars = 9 bytes.
    assert_eq!(u.tokenize("xyz東京").unwrap(), [tok(0, "xyz東京", (0, 9))]);
    assert_eq!(
        u.tokenize("ababcdabcdcd").unwrap(),
        [
            tok(1, "ab", (0, 2)),
            tok(8, "abcdabcd", (2, 10)),
            tok(2, "cd", (10, 12)),
        ]
    );
    // Positive scores are scores too: q/r singles beat the qr piece.
    assert_eq!(
        u.tokenize("abqrcd").unwrap(),
        [
            tok(1, "ab", (0, 2)),
            tok(9, "q", (2, 3)),
            tok(10, "r", (3, 4)),
            tok(2, "cd", (4, 6)),
        ]
    );
}

// ── Best-path selection and tie behavior ────────────────────────────────

#[test]
fn higher_scoring_split_beats_a_longer_piece() {
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",-5.0],["a",-1.0],["b",-1.0],["ab",-3.0]]}"#,
    );
    // a+b sums to -2.0 > -3.0: the split wins over the single piece.
    assert_eq!(
        u.tokenize("ab").unwrap(),
        [tok(1, "a", (0, 1)), tok(2, "b", (1, 2))]
    );
}

#[test]
fn exact_tie_keeps_the_first_written_path() {
    // Dyadic scores make both path sums EXACTLY -3.0 in f64:
    //   a(-2.0) + bc(-1.0)  ==  ab(-1.5) + c(-1.5)
    // The node at byte 3 is written first from starts_at=1 ("bc",
    // the earliest-starting final piece); the later "c" proposal
    // compares strictly-greater against an equal score and loses.
    // A >= comparison would flip this to ["ab","c"] — the reference
    // (verified on 0.21.2) keeps ["a","bc"].
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",-5.0],["a",-2.0],["ab",-1.5],["bc",-1.0],["c",-1.5]]}"#,
    );
    assert_eq!(
        u.tokenize("abc").unwrap(),
        [tok(1, "a", (0, 1)), tok(3, "bc", (1, 3))]
    );
}

#[test]
fn near_tie_is_decided_in_f64_not_in_reals() {
    // Real sums tie (0.1 + 0.2 = 0.3) but f64 sums do not:
    //   (-0.1) + (-0.2) = -0.30000000000000004 < -0.3
    // so the single piece wins on the negative table…
    let neg = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",-5.0],["a",-0.1],["b",-0.2],["ab",-0.3]]}"#,
    );
    assert_eq!(neg.tokenize("ab").unwrap(), [tok(3, "ab", (0, 2))]);
    // …and the SPLIT wins on the positive twin, where the same
    // one-ulp rounding lands above 0.3. Both verified on 0.21.2 —
    // decimal intuition would call both ties.
    let pos = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",-5.0],["a",0.1],["b",0.2],["ab",0.3]]}"#,
    );
    assert_eq!(
        pos.tokenize("ab").unwrap(),
        [tok(1, "a", (0, 1)), tok(2, "b", (1, 2))]
    );
}

// ── Unknown characters and unk_id ───────────────────────────────────────

#[test]
fn unknown_run_fuses_into_one_span_with_the_declared_unk_id() {
    // unk_id deliberately NOT 0, pinning the id plumbing.
    let u = load(r#"{"type":"Unigram","unk_id":1,"vocab":[["a",-1.0],["<unk>",-10.0]]}"#);
    assert_eq!(
        u.tokenize("aXYa").unwrap(),
        [
            tok(0, "a", (0, 1)),
            tok(1, "XY", (1, 3)),
            tok(0, "a", (3, 4)),
        ]
    );
}

#[test]
fn missing_unk_id_errors_only_when_an_unknown_node_is_stored() {
    // No unk_id and nothing covers byte 1: the reference errors
    // (MissingUnkId) — so do we, as Encode.
    let bare = load(r#"{"type":"Unigram","vocab":[["ab",-1.0]]}"#);
    let err = bare.tokenize("ab").unwrap_err();
    let TokenizerError::Encode { what } = err else {
        panic!("must fail with Encode, got: {err}");
    };
    assert!(what.contains("unk_id"), "{what}");

    // Same absence of unk_id, but every would-be unknown node LOSES
    // to a stored piece path — the reference encodes fine (the error
    // sits inside the accepted-update branch, not on every unmatched
    // character). Verified on 0.21.2.
    let covered = load(
        r#"{"type":"Unigram","vocab":[
            ["a",-1.0],["ab",-1.0],["bc",-1.0],["abc",-1.0]]}"#,
    );
    assert_eq!(covered.tokenize("abc").unwrap(), [tok(3, "abc", (0, 3))]);
    assert_eq!(covered.tokenize("ab").unwrap(), [tok(1, "ab", (0, 2))]);
}

#[test]
fn unknown_fuses_by_id_even_for_real_matches_of_the_unk_piece() {
    // unk_id points at the ordinary piece "a". Backtracking fuses by
    // node id, so the genuine "a" match at byte 0 fuses with the
    // unknown "x" into one span "ax" — a reference quirk, kept and
    // verified on 0.21.2.
    let u = load(r#"{"type":"Unigram","unk_id":0,"vocab":[["a",-1.0],["b",-1.0]]}"#);
    assert_eq!(
        u.tokenize("axb").unwrap(),
        [tok(0, "ax", (0, 2)), tok(1, "b", (2, 3))]
    );
}

// ── byte_fallback ───────────────────────────────────────────────────────

const BYTE_TABLE: &str = r#"{"type":"Unigram","unk_id":0,"byte_fallback":true,
    "vocab":[["<unk>",0.0],["<0xC3>",-0.01],["<0xA9>",-0.03]]}"#;

#[test]
fn byte_fallback_emits_byte_pieces_sharing_the_span_offsets() {
    // "é" is unknown; its two bytes exist as <0xNN> pieces. BOTH
    // tokens carry the whole span's offsets (0, 2) — the reference
    // does not split offsets per byte.
    let u = load(BYTE_TABLE);
    assert_eq!(
        u.tokenize("é").unwrap(),
        [tok(1, "<0xC3>", (0, 2)), tok(2, "<0xA9>", (0, 2))]
    );
}

#[test]
fn byte_fallback_needs_every_byte_piece_or_falls_back_to_unk() {
    // "?" and "é" fuse into one unknown span; byte 0x3F has no piece,
    // so the WHOLE span becomes one unk-id token with the raw text.
    let u = load(BYTE_TABLE);
    assert_eq!(u.tokenize("?é").unwrap(), [tok(0, "?é", (0, 3))]);
}

#[test]
fn byte_fallback_off_keeps_the_raw_text_under_unk_id() {
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"byte_fallback":false,
            "vocab":[["<unk>",0.0],["<0xC3>",-0.01],["<0xA9>",-0.03]]}"#,
    );
    assert_eq!(u.tokenize("é").unwrap(), [tok(0, "é", (0, 2))]);
}

// ── Multibyte pieces and edges ──────────────────────────────────────────

#[test]
fn multibyte_pieces_keep_offsets_byte_correct() {
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",0.0],["東",-1.0],["京",-1.0],["東京",-1.5]]}"#,
    );
    // 東京 (-1.5) beats 東+京 (-2.0); 都 is unknown. Three-byte chars
    // throughout.
    assert_eq!(
        u.tokenize("東京都").unwrap(),
        [tok(3, "東京", (0, 6)), tok(0, "都", (6, 9))]
    );
}

#[test]
fn empty_piece_loads_and_never_matches() {
    // The reference trie holds the empty piece but can never yield a
    // zero-length match; ours shares the property (no infinite loop,
    // no zero-width token).
    let u = load(
        r#"{"type":"Unigram","unk_id":2,"vocab":[
            ["",0.0],["a",-1.0],["<unk>",-2.0]]}"#,
    );
    assert_eq!(u.tokenize("a").unwrap(), [tok(1, "a", (0, 1))]);
}

#[test]
fn empty_pretoken_is_no_tokens() {
    let u = load(DOCTEST_TABLE);
    assert_eq!(u.tokenize("").unwrap(), []);
}

// ── Lookups ─────────────────────────────────────────────────────────────

#[test]
fn ids_are_vocab_row_indices() {
    let u = load(
        r#"{"type":"Unigram","unk_id":0,"vocab":[
            ["<unk>",0.0],["東",-1.0],["京",-1.0],["東京",-1.5]]}"#,
    );
    assert_eq!(u.token_to_id("東京"), Some(3));
    assert_eq!(u.token_to_id("都"), None);
    assert_eq!(u.id_to_token(1), Some("東"));
    assert_eq!(u.id_to_token(4), None);
}

// ── Constructor guards (message contract) ───────────────────────────────

#[test]
fn constructor_rejects_duplicate_pieces() {
    let err = Unigram::new(
        vec![("a".to_string(), -1.0), ("a".to_string(), -2.0)],
        None,
        false,
    )
    .unwrap_err();
    let TokenizerError::Vocab { what } = err else {
        panic!("must fail with Vocab, got: {err}");
    };
    assert!(what.contains("\"a\""), "{what}");
}

#[test]
fn constructor_rejects_unk_id_out_of_range() {
    let err = Unigram::new(vec![("a".to_string(), -1.0)], Some(1), false).unwrap_err();
    let TokenizerError::Vocab { what } = err else {
        panic!("must fail with Vocab, got: {err}");
    };
    assert!(what.contains("unk_id 1"), "{what}");

    let empty = Unigram::new(Vec::new(), Some(0), false).unwrap_err();
    let TokenizerError::Vocab { what } = empty else {
        panic!("must fail with Vocab, got: {empty}");
    };
    assert!(what.contains("empty"), "{what}");
}

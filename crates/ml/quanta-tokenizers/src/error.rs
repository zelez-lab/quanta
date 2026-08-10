//! The tokenizer error taxonomy — §8 of the crate's declared scope.
//!
//! The crate has no quanta dependencies, so the enum stands alone (no
//! `ArrayError` wrapping). The message contract is the `NpyError` one:
//! every message is self-contained, names the offender, and states the
//! workaround where one exists. Artifact files are untrusted input, so
//! every variant that a file can trigger carries enough context (byte
//! offset, JSON path, tag, id) to locate the fault without re-parsing.

use std::fmt;

/// The conformance reference this crate is pinned against (§2 of the
/// scope): every `type` tag the pinned HF `tokenizers` version
/// deserializes either ships or is a named exclusion; anything outside
/// that inventory is a loud [`TokenizerError::UnknownTag`]. The exact
/// patch version is stamped by `gen_fixtures.py` when the conformance
/// layer lands; the claim boundary is the 0.21 format inventory.
pub const PINNED_REFERENCE: &str = "HF tokenizers 0.21.x";

/// Everything that can go wrong loading or running a tokenizer.
///
/// The variants are the scope's §8 taxonomy, one row each. Load-time
/// rows (`Json` / `Schema` / `UnknownTag` / `UnsupportedField` /
/// `Vocab` / `Charsmap`) fire from [`crate::artifact`] and its JSON
/// substrate; `RegexConstruct` / `Encode` / `Decode` are the pipeline
/// rows the later layers raise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// An RFC 8259 violation in the artifact bytes: bad grammar,
    /// truncation, the nesting-depth cap, lone surrogates, duplicate
    /// object keys. `at` is a byte offset into the input.
    Json { at: usize, what: String },
    /// Structurally valid JSON that is not a valid artifact at a known
    /// location: a missing required field, a wrong type, a malformed
    /// sub-structure. `path` is the JSON path (`"model.vocab"`,
    /// `"added_tokens[3].content"`).
    Schema { path: String, what: String },
    /// A pipeline-stage `type` tag outside the pinned reference
    /// inventory — the format-growth claim boundary (§2 of the scope).
    UnknownTag { family: &'static str, tag: String },
    /// A field that is IN the pinned inventory but whose semantics are
    /// a named exclusion (e.g. BPE `dropout` non-null).
    UnsupportedField { path: String, why: String },
    /// A vocabulary-consistency fault caught at load: an id collision,
    /// an id that does not fit in `u32`, a merge naming an absent
    /// token, a Unigram `unk_id` out of range, a duplicate piece.
    Vocab { what: String },
    /// A malformed `Precompiled` charsmap: bad base64 in the artifact,
    /// or (at run time) an out-of-bounds trie transition. `at` is an
    /// offset into the base64 text or the decoded blob respectively.
    Charsmap { at: usize, what: String },
    /// A pattern using a regex construct outside the engine's closed
    /// set (§6 of the scope), or a backtracking-budget trip on a
    /// hostile pattern. Names both the pattern and the construct.
    RegexConstruct { pattern: String, construct: String },
    /// A pipeline-time encode fault: a stage failing on its input, a
    /// template referencing an id the vocab lacks.
    Encode { what: String },
    /// A decode id outside the vocabulary.
    Decode { id: u32, vocab_size: usize },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Json { at, what } => {
                write!(f, "tokenizer.json: {what} at byte {at}")
            }
            TokenizerError::Schema { path, what } => {
                write!(f, "tokenizer.json at {path}: {what}")
            }
            TokenizerError::UnknownTag { family, tag } => write!(
                f,
                "unknown {family} type {tag:?}: not in the pinned reference inventory \
                 ({PINNED_REFERENCE}) — the artifact was written by a newer tokenizers \
                 format; supporting the tag is additive work, refusing it here prevents \
                 a silent misparse"
            ),
            TokenizerError::UnsupportedField { path, why } => {
                write!(f, "tokenizer.json at {path}: {why}")
            }
            TokenizerError::Vocab { what } => write!(f, "tokenizer vocabulary: {what}"),
            TokenizerError::Charsmap { at, what } => {
                write!(f, "Precompiled charsmap: {what} at offset {at}")
            }
            TokenizerError::RegexConstruct { pattern, construct } => write!(
                f,
                "regex pattern {pattern:?}: {construct} is outside the engine's construct \
                 set (the model-corpus profile, §6 of the crate scope) — a new construct \
                 is additive engine work, refusing it here prevents a silent mis-split"
            ),
            TokenizerError::Encode { what } => write!(f, "encode: {what}"),
            TokenizerError::Decode { id, vocab_size } => write!(
                f,
                "decode: id {id} is outside the vocabulary (vocab size {vocab_size})"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {}

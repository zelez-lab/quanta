//! The model seam: one pre-token in, its tokens out.
//!
//! A [`Model`] is the vocabulary-owning stage of the pipeline — the
//! piece that turns each pre-token string into ids. The four artifact
//! families (`BPE`, `WordPiece`, `Unigram`, `WordLevel`) all implement
//! this trait; the pipeline drives whichever one the artifact declares,
//! through `dyn Model` (the trait is object-safe by construction).
//!
//! ## The offsets contract
//!
//! [`ModelToken::offsets`] are **byte ranges into the pre-token
//! string** the model was handed — never into the original text. The
//! pipeline rebases them into original-text coordinates when it
//! assembles the `Encoding`; keeping the model layer pre-token-local is
//! what lets every family stay ignorant of normalization and
//! pre-tokenization. Offsets follow the pinned reference's cumulative
//! byte-length accounting, which means two documented reference
//! behaviors carry over: a token's `value` is the *vocabulary* spelling
//! (it may carry a `continuing_subword_prefix` / `end_of_word_suffix`
//! or be an `unk_token`, and then differs from the pre-token slice at
//! its offsets), and a model that *drops* input (BPE with no
//! `unk_token` meeting an unknown character) shifts the offsets of
//! everything after the dropped span — byte-for-byte reference
//! fidelity, not an invariant of this trait.

use crate::error::TokenizerError;

/// One pre-token in, its tokens out. Offsets are byte ranges INTO THE
/// PRE-TOKEN string (the pipeline rebases them into the original text).
pub trait Model: Send + Sync {
    /// Tokenize one pre-token. An `Err` is a tokenize-time fault the
    /// pinned reference also raises at this point (e.g. an `unk_token`
    /// the vocabulary does not actually carry, first needed here).
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError>;
    /// The vocabulary spelling of `id`, if the model vocabulary has it.
    fn id_to_token(&self, id: u32) -> Option<&str>;
    /// The id of `token`, if the model vocabulary has it.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}

/// One token produced by a [`Model`]: the vocabulary id, the
/// vocabulary spelling, and the byte range of the pre-token it covers.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelToken {
    pub id: u32,
    pub value: String,
    pub offsets: (usize, usize),
}

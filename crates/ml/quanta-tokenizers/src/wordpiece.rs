//! The WordPiece model — greedy longest-match-first subword lookup
//! (BERT-class artifacts).
//!
//! A faithful port of the pinned reference's tokenize loop
//! (`models/wordpiece/mod.rs` in [`crate::PINNED_REFERENCE`]), fed by
//! the fields [`crate::artifact::ModelConfig::WordPiece`] holds:
//!
//! - **Greedy longest-match-first.** From each position the longest
//!   vocab entry wins: try the whole remainder, shrink one *character*
//!   off the end until the candidate is in the vocab. Positions after
//!   the first prepend `continuing_subword_prefix` (`##` for BERT) to
//!   the candidate before lookup, and the matched token's *value*
//!   carries the prefix while its *offsets* stay raw byte positions
//!   into the pre-token.
//! - **`max_input_chars_per_word` is a whole-word guard** (the
//!   reference's own giant-token bound, §8 of the crate scope): a
//!   pre-token with strictly more *characters* than the limit becomes
//!   one `unk_token` spanning the whole pre-token, vocab never
//!   consulted. Exactly at the limit tokenizes normally.
//! - **Unknown is all-or-nothing.** If any position matches nothing
//!   (not even a single character), the whole pre-token collapses to
//!   one `unk_token` over `(0, pretoken.len())` — matches found before
//!   the failure are discarded, as the reference's `is_bad` flag does.
//!
//! One deliberate deviation, load-time only: the reference resolves
//! the unk token's id lazily and fails *mid-encode*
//! (`MissingUnkToken`) the first time an unknown word appears;
//! [`WordPiece::new`] checks `unk_token ∈ vocab` up front instead —
//! the scope's loads-means-runs rule (§7). Every artifact the
//! reference can fully run loads here; one that would fault on its
//! first unknown word is refused at load with the same fault named.

use crate::error::TokenizerError;
use crate::model::{Model, ModelToken};
use std::collections::HashMap;

/// A WordPiece vocabulary with its lookup parameters, ready to
/// tokenize pre-tokens. Construct with [`WordPiece::new`] from the
/// fields of a loaded [`crate::artifact::ModelConfig::WordPiece`].
#[derive(Clone)]
pub struct WordPiece {
    vocab: HashMap<String, u32>,
    vocab_r: HashMap<u32, String>,
    unk_token: String,
    unk_id: u32,
    continuing_subword_prefix: String,
    max_input_chars_per_word: usize,
}

impl std::fmt::Debug for WordPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The vocab is thousands of entries; print its size, like the
        // reference's own Debug impl.
        f.debug_struct("WordPiece")
            .field("vocab", &self.vocab.len())
            .field("unk_token", &self.unk_token)
            .field("continuing_subword_prefix", &self.continuing_subword_prefix)
            .field("max_input_chars_per_word", &self.max_input_chars_per_word)
            .finish()
    }
}

impl WordPiece {
    /// Build a WordPiece model from the loaded config fields.
    ///
    /// Revalidates what the schema layer already guarantees for
    /// artifact-loaded vocabs (no duplicate tokens, no id collisions —
    /// this constructor is public and hand-built vocabs reach it too)
    /// and enforces the loads-means-runs deviation documented on the
    /// module: `unk_token` must be in the vocab.
    pub fn new(
        vocab: Vec<(String, u32)>,
        unk_token: String,
        continuing_subword_prefix: String,
        max_input_chars_per_word: usize,
    ) -> Result<Self, TokenizerError> {
        let mut map = HashMap::with_capacity(vocab.len());
        let mut map_r = HashMap::with_capacity(vocab.len());
        for (token, id) in vocab {
            if map_r.insert(id, token.clone()).is_some() {
                return Err(TokenizerError::Vocab {
                    what: format!(
                        "id collision: token {token:?} reuses id {id}, already assigned \
                         to another token"
                    ),
                });
            }
            if map.insert(token.clone(), id).is_some() {
                return Err(TokenizerError::Vocab {
                    what: format!("duplicate token {token:?} in WordPiece vocab"),
                });
            }
        }
        let unk_id = *map.get(&unk_token).ok_or_else(|| TokenizerError::Vocab {
            what: format!(
                "WordPiece unk_token {unk_token:?} is not in the vocab — the model \
                 cannot represent unknown words (the pinned reference faults on the \
                 first unknown word instead; refusing at load keeps loads-means-runs)"
            ),
        })?;
        Ok(WordPiece {
            vocab: map,
            vocab_r: map_r,
            unk_token,
            unk_id,
            continuing_subword_prefix,
            max_input_chars_per_word,
        })
    }

    /// The whole-pre-token unk: the reference emits `unk_token` over
    /// `(0, pretoken.len())` both for over-long words and for the
    /// `is_bad` no-match collapse.
    fn unk(&self, pretoken: &str) -> ModelToken {
        ModelToken {
            id: self.unk_id,
            value: self.unk_token.clone(),
            offsets: (0, pretoken.len()),
        }
    }
}

impl Model for WordPiece {
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError> {
        // The giant-token guard counts CHARACTERS; offsets stay bytes.
        if pretoken.chars().count() > self.max_input_chars_per_word {
            return Ok(vec![self.unk(pretoken)]);
        }

        let mut sub_tokens = Vec::new();
        let mut candidate = String::new();
        let mut start = 0;
        while start < pretoken.len() {
            // Longest match first: try `pretoken[start..end]` with end
            // shrinking one character at a time.
            let mut end = pretoken.len();
            let matched = loop {
                if start >= end {
                    break None;
                }
                candidate.clear();
                if start > 0 {
                    candidate.push_str(&self.continuing_subword_prefix);
                }
                candidate.push_str(&pretoken[start..end]);
                if let Some(&id) = self.vocab.get(candidate.as_str()) {
                    break Some(ModelToken {
                        id,
                        value: candidate.clone(),
                        offsets: (start, end),
                    });
                }
                // `start < end` keeps the slice non-empty, so a last
                // character always exists.
                end -= pretoken[start..end]
                    .chars()
                    .next_back()
                    .map_or(1, char::len_utf8);
            };
            let Some(token) = matched else {
                // Nothing matched at this position: the WHOLE
                // pre-token collapses to unk (the reference's
                // `is_bad` path), discarding earlier matches.
                return Ok(vec![self.unk(pretoken)]);
            };
            sub_tokens.push(token);
            start = end;
        }
        Ok(sub_tokens)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab_r.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.vocab.get(token).copied()
    }
}

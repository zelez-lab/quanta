//! The BPE runtime: merge-ranked pair agglomeration over the
//! artifact's vocab and merges list.
//!
//! Built from the validated [`ModelConfig::Bpe`] the artifact loader
//! produces, and pinned line-for-line against the reference
//! ([`crate::PINNED_REFERENCE`], `models/bpe/{model,word}.rs`):
//!
//! - **Merge order.** Each merge's rank is its index in the merges
//!   list. Agglomeration is heap-driven: the pending pair with the
//!   lowest rank merges first, and equal ranks (the same pair pending
//!   at several positions) resolve leftmost-first — the reference's
//!   `Merge` ordering, so the output is deterministic.
//! - **`ignore_merges`.** A pre-token that is itself a vocabulary
//!   entry short-circuits to that single token before any splitting
//!   (after the empty-input check, which still wins).
//! - **`byte_fallback`.** A piece missing from the vocabulary is
//!   retried as its bytes spelled `<0xXX>` (two uppercase hex digits);
//!   all bytes present emits one length-1 token per byte, any byte
//!   missing falls through to `unk_token` handling.
//! - **`unk_token` / `fuse_unk`.** Unknown pieces become the
//!   `unk_token`; with `fuse_unk` a run of unknowns fuses into one
//!   token spanning the run. No `unk_token` means unknown pieces are
//!   dropped (and the offsets of later tokens shift left — reference
//!   behavior, see [`crate::model`]). An `unk_token` the vocabulary
//!   does not carry faults at tokenize time, exactly where the
//!   reference raises `UnkTokenOutOfVocabulary`.
//! - **`continuing_subword_prefix` / `end_of_word_suffix`.** Every
//!   piece after the first is looked up with the prefix prepended, the
//!   last piece with the suffix appended; token offsets keep counting
//!   the *unmodified* bytes. The merged spelling of a pair `(a, b)` is
//!   `a` + `b` with one prefix length stripped off `b` — the
//!   reference's merge-map arithmetic.
//!
//! The reference wraps this in a per-word cache; the merge is
//! deterministic, so the cache is pure perf and none ships here — it
//! is addable without reopening anything (the scope's never-reopen
//! test).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::artifact::ModelConfig;
use crate::error::TokenizerError;
use crate::model::{Model, ModelToken};

/// The BPE model runtime — see the module docs for the semantics.
#[derive(Debug, Clone)]
pub struct Bpe {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    /// `(left id, right id)` → `(rank, merged id)`; rank is the pair's
    /// index in the artifact's merges list.
    merges: HashMap<(u32, u32), (u32, u32)>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
}

impl Bpe {
    /// Build the runtime from the artifact's `BPE` model config.
    ///
    /// The config is trusted: `TokenizerArtifact::from_bytes` has
    /// already cross-checked every merge against the vocab (both
    /// sides and the merged spelling, with the
    /// `continuing_subword_prefix` arithmetic), so nothing is
    /// re-validated here.
    ///
    /// # Panics
    ///
    /// If `config` is not [`ModelConfig::Bpe`] (the pipeline matches
    /// the model family before constructing a runtime), or if a
    /// hand-built config violates the loader's merge invariants.
    pub fn from_config(config: ModelConfig) -> Self {
        let ModelConfig::Bpe {
            vocab,
            merges,
            unk_token,
            continuing_subword_prefix,
            end_of_word_suffix,
            fuse_unk,
            byte_fallback,
            ignore_merges,
        } = config
        else {
            panic!("Bpe::from_config requires a ModelConfig::Bpe");
        };
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut id_to_token = HashMap::with_capacity(vocab.len());
        for (token, id) in vocab {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }
        // The reference's merge map: rank = list index, merged id =
        // the vocab id of `a` + `b` with one prefix stripped off `b`
        // (stripped blindly by length, as the reference does).
        let prefix_len = continuing_subword_prefix.as_deref().map_or(0, str::len);
        let mut merge_map = HashMap::with_capacity(merges.len());
        for (rank, (a, b)) in merges.iter().enumerate() {
            let lookup = |token: &str| {
                token_to_id
                    .get(token)
                    .copied()
                    .expect("merge tokens are cross-checked against the vocab at artifact load")
            };
            let merged = format!("{a}{}", &b[prefix_len..]);
            merge_map.insert((lookup(a), lookup(b)), (rank as u32, lookup(&merged)));
        }
        Bpe {
            token_to_id,
            id_to_token,
            merges: merge_map,
            unk_token,
            continuing_subword_prefix,
            end_of_word_suffix,
            fuse_unk,
            byte_fallback,
            ignore_merges,
        }
    }

    /// Split the pre-token into per-character symbols (prefix/suffix
    /// applied, byte fallback and unk handling as configured), then run
    /// the ranked agglomeration. The reference's `merge_word`.
    fn merge_word(&self, word: &str) -> Result<Vec<Symbol>, TokenizerError> {
        let mut symbols: Vec<Symbol> = Vec::with_capacity(word.len());
        // A pending run of unknown input: (unk id, byte length so far).
        let mut unk: Option<(u32, usize)> = None;
        let mut chars = word.char_indices().peekable();
        while let Some((start, _)) = chars.next() {
            let end = chars.peek().map_or(word.len(), |&(next, _)| next);
            let byte_len = end - start;
            let is_first = start == 0;
            let is_last = end == word.len();
            let mut piece: Cow<'_, str> = Cow::Borrowed(&word[start..end]);
            if !is_first && let Some(prefix) = &self.continuing_subword_prefix {
                piece = Cow::Owned(format!("{prefix}{piece}"));
            }
            if is_last && let Some(suffix) = &self.end_of_word_suffix {
                piece = Cow::Owned(format!("{piece}{suffix}"));
            }
            if let Some(&id) = self.token_to_id.get(piece.as_ref()) {
                if let Some((unk_id, unk_len)) = unk.take() {
                    push_symbol(&mut symbols, unk_id, unk_len);
                }
                push_symbol(&mut symbols, id, byte_len);
                continue;
            }
            // Vocab miss: byte fallback first, on the *modified* piece
            // (prefix/suffix included), each byte a length-1 symbol.
            // Faithfully to the reference, a pending unk run is NOT
            // flushed here — only a vocab hit or end-of-word flushes
            // it, so byte tokens land ahead of a pending unk.
            if self.byte_fallback
                && let Some(ids) = piece
                    .bytes()
                    .map(|b| self.token_to_id.get(&byte_token(b)).copied())
                    .collect::<Option<Vec<u32>>>()
            {
                for id in ids {
                    push_symbol(&mut symbols, id, 1);
                }
                continue;
            }
            // Unknown input: fuse into / start a pending unk run, or —
            // with no unk_token declared — drop the character.
            if let Some(unk_token) = &self.unk_token {
                unk = Some(match (unk.take(), self.fuse_unk) {
                    (Some((unk_id, unk_len)), true) => (unk_id, unk_len + byte_len),
                    (Some((unk_id, unk_len)), false) => {
                        push_symbol(&mut symbols, unk_id, unk_len);
                        (self.unk_id(unk_token)?, byte_len)
                    }
                    (None, _) => (self.unk_id(unk_token)?, byte_len),
                });
            }
        }
        if let Some((unk_id, unk_len)) = unk {
            push_symbol(&mut symbols, unk_id, unk_len);
        }
        self.merge_all(&mut symbols);
        Ok(symbols)
    }

    /// Heap-driven agglomeration: lowest rank first, ties leftmost.
    /// The reference's `Word::merge_all` (dropout is excluded at load,
    /// so its skip machinery has no counterpart here).
    fn merge_all(&self, symbols: &mut Vec<Symbol>) {
        let mut queue = BinaryHeap::with_capacity(symbols.len());
        for (pos, window) in symbols.windows(2).enumerate() {
            if let Some(&(rank, new_id)) = self.merges.get(&(window[0].id, window[1].id)) {
                queue.push(Merge { pos, rank, new_id });
            }
        }
        while let Some(top) = queue.pop() {
            let symbol = symbols[top.pos];
            // Skip entries expired by earlier merges: the left symbol
            // may be gone (len 0), may have become the last symbol, or
            // the pair at this position may no longer produce this
            // merged id.
            if symbol.len == 0 || symbol.next < 0 {
                continue;
            }
            let next_pos = symbol.next as usize;
            let right = symbols[next_pos];
            let still_current = self
                .merges
                .get(&(symbol.id, right.id))
                .is_some_and(|&(_, new_id)| new_id == top.new_id);
            if !still_current {
                continue;
            }
            // Merge: the left symbol absorbs the right one.
            symbols[top.pos].id = top.new_id;
            symbols[top.pos].len += right.len;
            symbols[top.pos].next = right.next;
            symbols[next_pos].len = 0;
            if right.next >= 0 && (right.next as usize) < symbols.len() {
                symbols[right.next as usize].prev = top.pos as isize;
            }
            // Queue the pairs the merged symbol now forms.
            let current = symbols[top.pos];
            if current.prev >= 0 {
                let prev_pos = current.prev as usize;
                let prev = symbols[prev_pos];
                if let Some(&(rank, new_id)) = self.merges.get(&(prev.id, current.id)) {
                    queue.push(Merge {
                        pos: prev_pos,
                        rank,
                        new_id,
                    });
                }
            }
            if current.next >= 0 && (current.next as usize) < symbols.len() {
                let next = symbols[current.next as usize];
                if let Some(&(rank, new_id)) = self.merges.get(&(current.id, next.id)) {
                    queue.push(Merge {
                        pos: top.pos,
                        rank,
                        new_id,
                    });
                }
            }
        }
        symbols.retain(|s| s.len != 0);
    }

    /// Surviving symbols → tokens: values are the vocabulary
    /// spellings, offsets the cumulative byte lengths (the reference's
    /// `word_to_tokens` over `get_offsets_iter`).
    fn tokens(&self, symbols: &[Symbol]) -> Vec<ModelToken> {
        let mut end = 0;
        symbols
            .iter()
            .map(|s| {
                let start = end;
                end += s.len;
                let value = self
                    .id_to_token
                    .get(&s.id)
                    .expect("every symbol id comes from the vocab")
                    .clone();
                ModelToken {
                    id: s.id,
                    value,
                    offsets: (start, end),
                }
            })
            .collect()
    }

    /// The unk token's id, resolved lazily: the reference faults at
    /// tokenize time (`UnkTokenOutOfVocabulary`), first time an
    /// unknown piece actually needs it — never at load.
    fn unk_id(&self, unk_token: &str) -> Result<u32, TokenizerError> {
        self.token_to_id
            .get(unk_token)
            .copied()
            .ok_or_else(|| TokenizerError::Encode {
                what: format!(
                    "BPE unk_token {unk_token:?} is not in the vocabulary — the artifact \
                     names an unknown-token spelling its own vocab lacks"
                ),
            })
    }
}

impl Model for Bpe {
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError> {
        // Reference order: empty wins over everything, then the
        // ignore_merges whole-word short-circuit.
        if pretoken.is_empty() {
            return Ok(Vec::new());
        }
        if self.ignore_merges
            && let Some(&id) = self.token_to_id.get(pretoken)
        {
            return Ok(vec![ModelToken {
                id,
                value: pretoken.to_string(),
                offsets: (0, pretoken.len()),
            }]);
        }
        let symbols = self.merge_word(pretoken)?;
        Ok(self.tokens(&symbols))
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

/// The byte-fallback spelling of one byte: `<0xXX>`, two uppercase hex
/// digits — the reference's `format!("<{b:#04X}>")`.
fn byte_token(b: u8) -> String {
    format!("<{b:#04X}>")
}

/// One agglomeration symbol: a doubly-linked-list node over the word
/// (`prev`/`next` are indices, `-1` = none; `len == 0` marks a symbol
/// absorbed by a merge). `len` counts bytes of the ORIGINAL pre-token,
/// so cumulative lengths are the token offsets.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    id: u32,
    prev: isize,
    next: isize,
    len: usize,
}

/// Append a symbol, linking it to the previous one — the reference's
/// `Word::add`.
fn push_symbol(symbols: &mut Vec<Symbol>, id: u32, len: usize) {
    let pos = symbols.len() as isize;
    if let Some(last) = symbols.last_mut() {
        last.next = pos;
    }
    symbols.push(Symbol {
        id,
        prev: pos - 1,
        next: -1,
        len,
    });
}

/// A pending merge in the agglomeration heap.
#[derive(Debug, Eq)]
struct Merge {
    pos: usize,
    rank: u32,
    new_id: u32,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        // Consistent with `Ord`: `new_id` is payload, not identity.
        self.rank == other.rank && self.pos == other.pos
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so the max-heap pops the LOWEST rank first, ties
        // broken by the LEFTMOST position — the reference's ordering,
        // and the source of merge determinism.
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

//! The Unigram model — Viterbi best-segmentation over a piece/logprob
//! table (sentencepiece-class artifacts: T5, ALBERT, XLNet).
//!
//! A faithful port of the pinned reference's optimized encode path
//! (`models/unigram/model.rs::encode_optimized` in
//! [`crate::PINNED_REFERENCE`] — deserialized models always take it:
//! `Unigram::from` fixes `is_optimized = true` and `fuse_unk = true`),
//! fed by the fields [`crate::artifact::ModelConfig::Unigram`] holds:
//!
//! - **The lattice.** Nodes are byte positions `0..=len`. Walking
//!   character starts left to right, every vocab piece matching at a
//!   start (found by a byte-trie prefix walk, shortest first) proposes
//!   `score(piece) + best(start)` to the node at its end. When no
//!   piece of exactly one character matches at a start, an *unknown*
//!   node spanning that character is proposed at
//!   `min_score - 10.0` (the reference's `kUnkPenalty`).
//! - **Ties.** A proposal replaces the stored one only when *strictly*
//!   greater, so on an exact `f64` tie the first writer survives —
//!   see "Tie behavior" below for why this is reproduced bit-exactly.
//! - **Unknown fusing.** Backtracking fuses runs of adjacent unknown
//!   nodes into one span (`fuse_unk` is always true for deserialized
//!   models). The fuse test is `node.id == unk_id`, so a genuine
//!   vocab match of the piece *at* the unk index fuses too — a
//!   reference quirk kept on purpose.
//! - **`byte_fallback`.** A produced span that is not itself a vocab
//!   piece (an unknown span) is re-emitted as its bytes' `<0xNN>`
//!   pieces when the config declares `byte_fallback` and EVERY byte
//!   piece exists — all of them carrying the *whole span's* offsets
//!   (the reference does not split offsets per byte). Otherwise the
//!   span becomes one token with `unk_id` and the raw text as value.
//! - **No `unk_id` is a claim boundary, mid-encode.** The reference
//!   errors (`MissingUnkId`) only when an unknown node would actually
//!   be *stored* — an input fully covered by multi-character pieces
//!   encodes fine without an `unk_id`, even though single characters
//!   of it match nothing. The error placement is ported exactly and
//!   surfaces as [`TokenizerError::Encode`] naming the position.
//!
//! # Tie behavior — why argmax matches the reference bit-for-bit
//!
//! `f64` addition is not associative, so two segmentations whose
//! *real* logprob sums are equal can round to different `f64` totals —
//! each path's total is a left fold along its own piece boundaries —
//! and on real tables the argmax between near-tied segmentations is
//! decided by exactly those roundings. Accumulation order therefore
//! CAN affect the argmax; parity is achieved not by avoiding the
//! issue but by inheriting the reference's arithmetic wholesale: one
//! `f64` addition per lattice edge, `score + best(start)` in the
//! reference's operand order, over the identical proposal set. Every
//! stored score is then bit-identical to the reference's, so every
//! strictly-greater comparison resolves identically. Exact `f64` ties
//! (which survive rounding, e.g. dyadic scores) fall to update order:
//! proposals reach a node in ascending start order (the outer loop),
//! the unknown proposal last (no piece can start between a node's
//! last character boundary and the node), so the surviving path is
//! the one whose final piece starts earliest — the longest final
//! piece — recursively. Both cases are pinned by crafted-table tests:
//! an exact dyadic tie and a one-ulp near-tie.
//!
//! The reference's encode cache (`utils/cache.rs`) is a perf artifact
//! with no observable semantics and is deliberately not ported (§7 of
//! the crate scope: correctness parity, not throughput racing).

use crate::error::TokenizerError;
use crate::model::{Model, ModelToken};
use std::collections::HashMap;

/// The reference's `kUnkPenalty` (inherited from sentencepiece):
/// unknown nodes score `min_score - K_UNK_PENALTY`.
const K_UNK_PENALTY: f64 = 10.0;

/// A Unigram piece table ready to tokenize pre-tokens. Construct with
/// [`Unigram::new`] from the fields of a loaded
/// [`crate::artifact::ModelConfig::Unigram`].
#[derive(Clone)]
pub struct Unigram {
    /// `[piece, logprob]` rows; a piece's id is its row index.
    vocab: Vec<(String, f64)>,
    token_ids: HashMap<String, u32>,
    trie: PieceTrie,
    min_score: f64,
    unk_id: Option<u32>,
    byte_fallback: bool,
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

impl Unigram {
    /// Build a Unigram model from the loaded config fields.
    ///
    /// Mirrors the reference constructor's checks (empty vocab and
    /// `unk_id` range, both gated on `unk_id` being declared) and
    /// revalidates what the schema layer already guarantees for
    /// artifact-loaded vocabs (no duplicate pieces — this constructor
    /// is public and hand-built tables reach it too). An empty piece
    /// is accepted and never matched, exactly like the reference trie
    /// (its zero-length match is unreachable by construction).
    pub fn new(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self, TokenizerError> {
        if let Some(u) = unk_id {
            if vocab.is_empty() {
                return Err(TokenizerError::Vocab {
                    what: "Unigram vocab is empty but unk_id is declared — at least the \
                           unk piece is needed"
                        .to_string(),
                });
            }
            if u >= vocab.len() {
                return Err(TokenizerError::Vocab {
                    what: format!(
                        "Unigram unk_id {u} is out of range (the vocab holds {} pieces)",
                        vocab.len()
                    ),
                });
            }
        }
        if u32::try_from(vocab.len()).is_err() {
            return Err(TokenizerError::Vocab {
                what: format!(
                    "Unigram vocab holds {} pieces; ids are u32 by format contract",
                    vocab.len()
                ),
            });
        }
        let mut token_ids = HashMap::with_capacity(vocab.len());
        let mut trie = PieceTrie::new();
        let mut min_score = f64::INFINITY;
        for (i, (piece, score)) in vocab.iter().enumerate() {
            // Fits by the length check above.
            let id = i as u32;
            if token_ids.insert(piece.clone(), id).is_some() {
                return Err(TokenizerError::Vocab {
                    what: format!("duplicate piece {piece:?} in Unigram vocab"),
                });
            }
            trie.insert(piece.as_bytes(), id);
            if *score < min_score {
                min_score = *score;
            }
        }
        Ok(Unigram {
            vocab,
            token_ids,
            trie,
            min_score,
            // In range by the checks above.
            unk_id: unk_id.map(|u| u as u32),
            byte_fallback,
        })
    }

    /// The Viterbi pass: best segmentation of `pretoken` as byte
    /// spans, adjacent unknown nodes already fused. Spans tile the
    /// pre-token exactly, in order.
    fn viterbi(&self, pretoken: &str) -> Result<Vec<(usize, usize)>, TokenizerError> {
        /// The best path ending at a byte position (the reference's
        /// `BestPathNode`): the last piece's vocab id, the path's
        /// accumulated score, and where that piece starts
        /// (`None` = position not reached yet).
        #[derive(Clone)]
        struct BestPathNode {
            id: u32,
            score: f64,
            starts_at: Option<usize>,
        }
        let size = pretoken.len();
        let bytes = pretoken.as_bytes();
        let unk_score = self.min_score - K_UNK_PENALTY;
        let mut best = vec![
            BestPathNode {
                id: 0,
                score: 0.0,
                starts_at: None,
            };
            size + 1
        ];

        let mut starts_at = 0;
        while starts_at < size {
            let till_here = best[starts_at].score;
            // `starts_at` only ever advances by whole characters, so
            // it is always a character boundary.
            let mblen = pretoken[starts_at..]
                .chars()
                .next()
                .map_or(1, char::len_utf8);
            let mut has_single_node = false;
            // Every vocab piece matching at `starts_at`, shortest
            // first (the trie yields a match each time the walk
            // passes a piece end — the reference's
            // `common_prefix_search` order).
            let mut node = PieceTrie::ROOT;
            for (i, &b) in bytes[starts_at..].iter().enumerate() {
                let Some(next) = self.trie.step(node, b) else {
                    break;
                };
                node = next;
                let Some(id) = self.trie.piece(node) else {
                    continue;
                };
                let length = i + 1;
                let score = self.vocab[id as usize].1;
                let candidate = score + till_here;
                let target = &mut best[starts_at + length];
                if target.starts_at.is_none() || candidate > target.score {
                    *target = BestPathNode {
                        id,
                        score: candidate,
                        starts_at: Some(starts_at),
                    };
                }
                if length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let candidate = unk_score + till_here;
                let target = &mut best[starts_at + mblen];
                if target.starts_at.is_none() || candidate > target.score {
                    // The reference errors HERE, not on every
                    // unmatched character: an unknown node that loses
                    // to a stored piece path needs no unk_id.
                    let Some(unk) = self.unk_id else {
                        return Err(TokenizerError::Encode {
                            what: format!(
                                "Unigram: no piece covers {:?} at byte {starts_at} of \
                                 the pre-token and the model declares no unk_id — the \
                                 pinned reference fails identically (MissingUnkId)",
                                &pretoken[starts_at..starts_at + mblen]
                            ),
                        });
                    };
                    *target = BestPathNode {
                        id: unk,
                        score: candidate,
                        starts_at: Some(starts_at),
                    };
                }
            }
            starts_at += mblen;
        }

        // Backtrack, fusing runs of adjacent unknown nodes into one
        // span. Spans come out right-to-left and are reversed once.
        let mut spans: Vec<(usize, usize)> = Vec::new();
        let mut fused: Option<(usize, usize)> = None;
        let mut ends_at = size;
        while ends_at > 0 {
            let node = &best[ends_at];
            let starts_at = node
                .starts_at
                .expect("every character boundary receives a node or the pass errored");
            if self.unk_id == Some(node.id) {
                // Keep the run's right edge, extend its left edge.
                let right = fused.map_or(ends_at, |(_, e)| e);
                fused = Some((starts_at, right));
            } else {
                if let Some(span) = fused.take() {
                    spans.push(span);
                }
                spans.push((starts_at, ends_at));
            }
            ends_at = starts_at;
        }
        if let Some(span) = fused {
            spans.push(span);
        }
        spans.reverse();
        Ok(spans)
    }
}

impl Model for Unigram {
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError> {
        let spans = self.viterbi(pretoken)?;
        let mut tokens = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            let piece = &pretoken[start..end];
            // The reference re-looks the produced STRING up — that
            // lookup, not the lattice path, decides the id.
            if let Some(&id) = self.token_ids.get(piece) {
                tokens.push(ModelToken {
                    id,
                    value: piece.to_string(),
                    offsets: (start, end),
                });
                continue;
            }
            // An unknown span. With byte_fallback and every byte
            // piece present, emit `<0xNN>` pieces, ALL carrying the
            // whole span's offsets (reference semantics).
            if self.byte_fallback {
                let byte_ids: Option<Vec<u32>> = piece
                    .bytes()
                    .map(|b| self.token_ids.get(&format!("<0x{b:02X}>")).copied())
                    .collect();
                if let Some(byte_ids) = byte_ids {
                    tokens.extend(piece.bytes().zip(byte_ids).map(|(b, id)| ModelToken {
                        id,
                        value: format!("<0x{b:02X}>"),
                        offsets: (start, end),
                    }));
                    continue;
                }
            }
            // Fall back to unk_id with the raw text as the value. A
            // span can only be absent from the vocab by having
            // travelled the unknown path, so unk_id is present here;
            // the guard mirrors the reference's own.
            let Some(unk) = self.unk_id else {
                return Err(TokenizerError::Encode {
                    what: format!(
                        "Unigram: span {piece:?} is not in the vocab and the model \
                         declares no unk_id — the pinned reference fails identically \
                         (MissingUnkId)"
                    ),
                });
            };
            tokens.push(ModelToken {
                id: unk,
                value: piece.to_string(),
                offsets: (start, end),
            });
        }
        Ok(tokens)
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        // A piece's id is its vocab row index (reference semantics).
        self.vocab.get(id as usize).map(|(piece, _)| piece.as_str())
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_ids.get(token).copied()
    }
}

// ── The piece trie ──────────────────────────────────────────────────────

/// A byte trie over the vocab pieces, walked once per lattice start
/// position. Nodes live in one arena; children are sorted for binary
/// search. Matches surface strictly shortest-first, and a zero-length
/// piece is never surfaced (the walk yields only after consuming a
/// byte) — both properties the reference trie shares.
#[derive(Clone)]
struct PieceTrie {
    nodes: Vec<TrieNode>,
}

#[derive(Clone, Default)]
struct TrieNode {
    /// `(byte, child index)`, sorted by byte.
    children: Vec<(u8, usize)>,
    /// The vocab id of the piece ending at this node, if any.
    piece: Option<u32>,
}

impl PieceTrie {
    const ROOT: usize = 0;

    fn new() -> Self {
        PieceTrie {
            nodes: vec![TrieNode::default()],
        }
    }

    fn insert(&mut self, bytes: &[u8], id: u32) {
        let mut node = Self::ROOT;
        for &b in bytes {
            node = match self.nodes[node]
                .children
                .binary_search_by_key(&b, |&(c, _)| c)
            {
                Ok(i) => self.nodes[node].children[i].1,
                Err(i) => {
                    let next = self.nodes.len();
                    self.nodes.push(TrieNode::default());
                    self.nodes[node].children.insert(i, (b, next));
                    next
                }
            };
        }
        self.nodes[node].piece = Some(id);
    }

    fn step(&self, node: usize, b: u8) -> Option<usize> {
        let children = &self.nodes[node].children;
        children
            .binary_search_by_key(&b, |&(c, _)| c)
            .ok()
            .map(|i| children[i].1)
    }

    fn piece(&self, node: usize) -> Option<u32> {
        self.nodes[node].piece
    }
}

//! The `Encoding` record (§7's full reference shape) plus truncation
//! and padding — ports of HF tokenizers 0.21.x `tokenizer/encoding.rs`,
//! `utils/truncation.rs`, and `utils/padding.rs`.
//!
//! Offsets are byte offsets into the ORIGINAL input text (converted
//! through the alignment layer before tokens reach this record);
//! post-processor specials carry `(0, 0)`. `word_ids` restart per
//! sequence, `overflowing` holds the stride windows truncation cuts
//! off, and merging two encodings takes the cartesian product of their
//! overflow lists (the reference's `merge_with`), which is what makes a
//! pair template apply to every overflow window.
//!
//! Deviations from the reference, both on the §8 hostile-input posture
//! (artifact-supplied truncation params must not panic the process):
//! `stride >= max_len` is a loud [`TokenizerError::Encode`] instead of
//! an `assert!`, and the special-token headroom subtraction saturates.

use crate::artifact::{
    Direction, PaddingConfig, PaddingStrategy, TruncationConfig, TruncationStrategy,
};
use crate::error::TokenizerError;

/// The output of an encode: one row per token, in the reference's full
/// record shape. Field-for-field getters; `overflowing` holds the
/// truncation windows beyond the first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Encoding {
    pub(crate) ids: Vec<u32>,
    pub(crate) type_ids: Vec<u32>,
    pub(crate) tokens: Vec<String>,
    pub(crate) words: Vec<Option<u32>>,
    pub(crate) offsets: Vec<(usize, usize)>,
    pub(crate) special_tokens_mask: Vec<u32>,
    pub(crate) attention_mask: Vec<u32>,
    pub(crate) overflowing: Vec<Encoding>,
}

impl Encoding {
    /// Builds a row-parallel encoding from its columns (the
    /// post-processing layer's constructor).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        ids: Vec<u32>,
        type_ids: Vec<u32>,
        tokens: Vec<String>,
        words: Vec<Option<u32>>,
        offsets: Vec<(usize, usize)>,
        special_tokens_mask: Vec<u32>,
        attention_mask: Vec<u32>,
        overflowing: Vec<Encoding>,
    ) -> Self {
        Encoding {
            ids,
            type_ids,
            tokens,
            words,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
        }
    }

    /// Appends one model token row (attention 1, not special).
    pub(crate) fn push_token(
        &mut self,
        id: u32,
        token: String,
        offsets: (usize, usize),
        word: Option<u32>,
        type_id: u32,
    ) {
        self.ids.push(id);
        self.type_ids.push(type_id);
        self.tokens.push(token);
        self.words.push(word);
        self.offsets.push(offsets);
        self.special_tokens_mask.push(0);
        self.attention_mask.push(1);
    }

    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the encoding holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Token ids — the model-input vector.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Segment type ids (pair encodes assign 0/1, templates override).
    pub fn type_ids(&self) -> &[u32] {
        &self.type_ids
    }

    /// Token strings.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Word index per token (restarting per sequence); `None` for
    /// special and padding tokens.
    pub fn word_ids(&self) -> &[Option<u32>] {
        &self.words
    }

    /// Byte offsets into the ORIGINAL text; `(0, 0)` for specials and
    /// padding.
    pub fn offsets(&self) -> &[(usize, usize)] {
        &self.offsets
    }

    /// 1 where the token was inserted by the post-processor or padding.
    pub fn special_tokens_mask(&self) -> &[u32] {
        &self.special_tokens_mask
    }

    /// 1 for real tokens, 0 for padding.
    pub fn attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// The truncation windows beyond the first (with stride overlap).
    pub fn overflowing(&self) -> &[Encoding] {
        &self.overflowing
    }

    // ── Alignment helpers (§7 — byte space throughout) ──────────────────

    /// The original-text byte span of token `token`.
    pub fn token_to_chars(&self, token: usize) -> Option<(usize, usize)> {
        self.offsets.get(token).copied()
    }

    /// The word index of token `token`.
    pub fn token_to_word(&self, token: usize) -> Option<u32> {
        self.words.get(token).copied().flatten()
    }

    /// The token span `(start, one_past_end)` of word `word`.
    pub fn word_to_tokens(&self, word: u32) -> Option<(usize, usize)> {
        let mut bounds = None;
        for (i, w) in self.words.iter().enumerate() {
            if *w == Some(word) {
                let (start, _) = bounds.unwrap_or((i, i + 1));
                bounds = Some((start, i + 1));
            }
        }
        bounds
    }

    /// The original-text byte span of word `word`.
    pub fn word_to_chars(&self, word: u32) -> Option<(usize, usize)> {
        self.word_to_tokens(word)
            .map(|(start, end)| (self.offsets[start].0, self.offsets[end - 1].1))
    }

    /// The token containing original-text byte `pos`.
    pub fn char_to_token(&self, pos: usize) -> Option<usize> {
        self.offsets
            .iter()
            .position(|(start, end)| pos >= *start && pos < *end)
    }

    /// The word containing original-text byte `pos`.
    pub fn char_to_word(&self, pos: usize) -> Option<u32> {
        self.char_to_token(pos).and_then(|t| self.token_to_word(t))
    }

    // ── Truncation ──────────────────────────────────────────────────────

    /// Truncates to `max_len` tokens, moving the cut-off windows (with
    /// `stride` tokens of overlap) into `overflowing`. `Left` keeps the
    /// end of the sequence. Reference `Encoding::truncate`, with the
    /// `stride >= max_len` panic turned into a loud error.
    pub fn truncate(
        &mut self,
        max_len: usize,
        stride: usize,
        direction: Direction,
    ) -> Result<(), TokenizerError> {
        let encoding_len = self.ids.len();
        if max_len >= encoding_len {
            return Ok(());
        }
        if max_len == 0 {
            let cut = std::mem::take(self);
            self.overflowing.push(cut);
            return Ok(());
        }
        if stride >= max_len {
            return Err(TokenizerError::Encode {
                what: format!(
                    "truncation stride {stride} must be strictly less than the effective \
                     max_length {max_len} (max_length may be shorter than requested once \
                     special tokens are budgeted)"
                ),
            });
        }

        let offset = max_len - stride;
        let mut done = false;
        let parts_ranges: Vec<(usize, usize)> = match direction {
            Direction::Right => (0..encoding_len)
                .step_by(offset)
                .filter_map(|start| {
                    if done {
                        return None;
                    }
                    let stop = std::cmp::min(start + max_len, encoding_len);
                    done = stop == encoding_len;
                    Some((start, stop))
                })
                .collect(),
            Direction::Left => (0..encoding_len)
                .rev()
                .step_by(offset)
                .filter_map(|stop| {
                    let stop = stop + 1;
                    let start = stop.saturating_sub(max_len);
                    if start < stop && !done {
                        done = start == 0;
                        Some((start, stop))
                    } else {
                        None
                    }
                })
                .collect(),
        };

        let window = |&(start, stop): &(usize, usize)| Encoding {
            ids: self.ids[start..stop].to_vec(),
            type_ids: self.type_ids[start..stop].to_vec(),
            tokens: self.tokens[start..stop].to_vec(),
            words: self.words[start..stop].to_vec(),
            offsets: self.offsets[start..stop].to_vec(),
            special_tokens_mask: self.special_tokens_mask[start..stop].to_vec(),
            attention_mask: self.attention_mask[start..stop].to_vec(),
            overflowing: Vec::new(),
        };
        let mut new_encoding = window(&parts_ranges[0]);
        new_encoding.overflowing = parts_ranges[1..].iter().map(window).collect();
        *self = new_encoding;
        Ok(())
    }

    // ── Merge ───────────────────────────────────────────────────────────

    /// Appends `pair` to `self`; the overflow lists combine as their
    /// cartesian product (reference `merge_with`). Offsets are kept
    /// as-is (both parts index the same conceptual original per the
    /// post-processing contract).
    pub(crate) fn merge_with(&mut self, pair: Encoding) {
        let mut overflowings = Vec::new();
        for self_o in &self.overflowing {
            let mut merged = self_o.clone();
            merged.merge_with(pair.clone());
            overflowings.push(merged);
            for other_o in &pair.overflowing {
                let mut merged = self_o.clone();
                merged.merge_with(other_o.clone());
                overflowings.push(merged);
            }
        }
        for other_o in &pair.overflowing {
            let mut merged = self.clone();
            merged.merge_with(other_o.clone());
            overflowings.push(merged);
        }

        self.ids.extend(pair.ids);
        self.type_ids.extend(pair.type_ids);
        self.tokens.extend(pair.tokens);
        self.words.extend(pair.words);
        self.offsets.extend(pair.offsets);
        self.special_tokens_mask.extend(pair.special_tokens_mask);
        self.attention_mask.extend(pair.attention_mask);
        self.overflowing = overflowings;
    }

    /// Folds a run of encodings into one via [`Encoding::merge_with`].
    pub(crate) fn merge(encodings: impl IntoIterator<Item = Encoding>) -> Encoding {
        let mut merged = Encoding::default();
        for encoding in encodings {
            merged.merge_with(encoding);
        }
        merged
    }

    // ── Padding ─────────────────────────────────────────────────────────

    /// Pads to `target_length` (overflow windows included), inserting
    /// `pad_id`/`pad_type_id`/`pad_token` rows with attention 0 and
    /// special mask 1 on the given side. Longer encodings are left
    /// untouched.
    pub fn pad(
        &mut self,
        target_length: usize,
        pad_id: u32,
        pad_type_id: u32,
        pad_token: &str,
        direction: Direction,
    ) {
        for overflow in &mut self.overflowing {
            overflow.pad(target_length, pad_id, pad_type_id, pad_token, direction);
        }
        if self.ids.len() >= target_length {
            return;
        }
        let pad_length = target_length - self.ids.len();
        match direction {
            Direction::Left => {
                self.ids
                    .splice(0..0, std::iter::repeat_n(pad_id, pad_length));
                self.type_ids
                    .splice(0..0, std::iter::repeat_n(pad_type_id, pad_length));
                self.tokens
                    .splice(0..0, std::iter::repeat_n(pad_token.to_string(), pad_length));
                self.words
                    .splice(0..0, std::iter::repeat_n(None, pad_length));
                self.offsets
                    .splice(0..0, std::iter::repeat_n((0, 0), pad_length));
                self.special_tokens_mask
                    .splice(0..0, std::iter::repeat_n(1, pad_length));
                self.attention_mask
                    .splice(0..0, std::iter::repeat_n(0, pad_length));
            }
            Direction::Right => {
                self.ids.extend(std::iter::repeat_n(pad_id, pad_length));
                self.type_ids
                    .extend(std::iter::repeat_n(pad_type_id, pad_length));
                self.tokens
                    .extend(std::iter::repeat_n(pad_token.to_string(), pad_length));
                self.words.extend(std::iter::repeat_n(None, pad_length));
                self.offsets.extend(std::iter::repeat_n((0, 0), pad_length));
                self.special_tokens_mask
                    .extend(std::iter::repeat_n(1, pad_length));
                self.attention_mask
                    .extend(std::iter::repeat_n(0, pad_length));
            }
        }
    }
}

// ── Truncation over one or a pair (reference `utils/truncation.rs`) ─────

/// Applies the truncation params to a single encoding or a pair —
/// `LongestFirst` solves the two-budget split with the reference's
/// case arithmetic; `OnlyFirst`/`OnlySecond` cut one side and error
/// loudly when it cannot absorb the excess.
pub(crate) fn truncate_encodings(
    mut encoding: Encoding,
    mut pair_encoding: Option<Encoding>,
    params: &TruncationConfig,
) -> Result<(Encoding, Option<Encoding>), TokenizerError> {
    if params.max_length == 0 {
        encoding.truncate(0, params.stride, params.direction)?;
        if let Some(other) = pair_encoding.as_mut() {
            other.truncate(0, params.stride, params.direction)?;
        }
        return Ok((encoding, pair_encoding));
    }

    let total_length = encoding.len() + pair_encoding.as_ref().map_or(0, Encoding::len);
    if total_length <= params.max_length {
        return Ok((encoding, pair_encoding));
    }
    let to_remove = total_length - params.max_length;

    match params.strategy {
        TruncationStrategy::LongestFirst => {
            if let Some(other) = pair_encoding.as_mut() {
                // With n1 <= n2: either only the longer input truncates
                // (n2 = max - n1), or both truncate to a half each.
                let mut n1 = encoding.len();
                let mut n2 = other.len();
                let mut swap = false;
                if n1 > n2 {
                    swap = true;
                    std::mem::swap(&mut n1, &mut n2);
                }
                if n1 > params.max_length {
                    n2 = n1;
                } else {
                    n2 = std::cmp::max(n1, params.max_length - n1);
                }
                if n1 + n2 > params.max_length {
                    n1 = params.max_length / 2;
                    n2 = n1 + params.max_length % 2;
                }
                if swap {
                    std::mem::swap(&mut n1, &mut n2);
                }
                encoding.truncate(n1, params.stride, params.direction)?;
                other.truncate(n2, params.stride, params.direction)?;
            } else {
                encoding.truncate(total_length - to_remove, params.stride, params.direction)?;
            }
        }
        TruncationStrategy::OnlyFirst | TruncationStrategy::OnlySecond => {
            let target = if params.strategy == TruncationStrategy::OnlyFirst {
                &mut encoding
            } else if let Some(other) = pair_encoding.as_mut() {
                other
            } else {
                return Err(TokenizerError::Encode {
                    what: "truncation strategy OnlySecond needs a pair input".to_string(),
                });
            };
            let target_len = target.len();
            if target_len > to_remove {
                target.truncate(target_len - to_remove, params.stride, params.direction)?;
            } else {
                return Err(TokenizerError::Encode {
                    what: format!(
                        "the {} sequence ({target_len} tokens) is too short to absorb the \
                         {to_remove} tokens over max_length {}",
                        if params.strategy == TruncationStrategy::OnlyFirst {
                            "first"
                        } else {
                            "second"
                        },
                        params.max_length
                    ),
                });
            }
        }
    }
    Ok((encoding, pair_encoding))
}

// ── Batch padding (reference `utils/padding.rs`) ────────────────────────

/// Pads a batch in place: `BatchLongest` measures the batch, `Fixed`
/// uses the configured size, and `pad_to_multiple_of` rounds the target
/// up.
pub fn pad_encodings(encodings: &mut [Encoding], params: &PaddingConfig) {
    if encodings.is_empty() {
        return;
    }
    let mut pad_length = match params.strategy {
        PaddingStrategy::Fixed(size) => size,
        PaddingStrategy::BatchLongest => encodings
            .iter()
            .map(Encoding::len)
            .max()
            .expect("non-empty batch"),
    };
    if let Some(multiple) = params.pad_to_multiple_of
        && multiple > 0
        && !pad_length.is_multiple_of(multiple)
    {
        pad_length += multiple - pad_length % multiple;
    }
    for encoding in encodings.iter_mut() {
        encoding.pad(
            pad_length,
            params.pad_id,
            params.pad_type_id,
            &params.pad_token,
            params.direction,
        );
    }
}

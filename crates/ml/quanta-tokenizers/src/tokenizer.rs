//! The facade: artifact bytes in, running pipeline out.
//!
//! [`Tokenizer::from_bytes`] parses and validates the whole artifact
//! (the schema layer), compiles every stage executor — normalizer,
//! pre-tokenizer, post-processor, decoder, added-token matchers, with
//! regexes parsed and the charsmap decoded — and wires the model
//! runtime behind the [`Model`] seam, so a tokenizer that loads, runs
//! (§7). Encode follows the reference flow exactly (HF tokenizers
//! 0.21.x, `tokenizer/mod.rs` + `added_vocabulary.rs`):
//!
//! 1. **Added-token extraction** — the two-pass split: non-normalized
//!    tokens match the RAW text first; each remaining piece is then
//!    normalized and the normalized tokens match after (their contents
//!    are pre-normalized through the same normalizer at load).
//!    Matching is leftmost-longest; `single_word` discards matches
//!    embedded in word characters, `lstrip`/`rstrip` extend a match
//!    over adjacent whitespace. Matched spans arrive at the model
//!    pre-tokenized (one token, the span's text), so the pipeline
//!    never re-splits them.
//! 2. **Normalize** (inside pass 2) and **pre-tokenize** the remaining
//!    pieces, **tokenize** them through the model, and assemble the
//!    [`Encoding`] with offsets into the ORIGINAL text via the
//!    alignment layer.
//! 3. **Post-process**: truncation first (the saved artifact params
//!    are the active defaults; special-token headroom is subtracted
//!    when `add_special_tokens`), then the post-processor (or the
//!    default pair merge), then padding.
//!
//! `encode_batch` is a sequential host loop plus batch padding —
//! deliberately not internally threaded (zero-dep; §7). Callers who
//! want parallelism split the batch across `std::thread::scope`
//! threads and pad afterwards.
//!
//! Decode runs the artifact's decoder chain; an out-of-range id is a
//! loud [`TokenizerError::Decode`] naming the id (the scope's decode
//! row — the reference silently skips unknown ids; silence loses data).

use crate::artifact::{
    AddedTokenConfig, ModelConfig, PaddingConfig, TokenizerArtifact, TruncationConfig,
};
use crate::decode::{DecodeStream, Decoder};
use crate::encoding::{Encoding, pad_encodings, truncate_encodings};
use crate::error::TokenizerError;
use crate::model::{Model, ModelToken};
use crate::normalize::Normalizer;
use crate::normalized::NormalizedString;
use crate::postprocess::PostProcessor;
use crate::pretokenize::{PreTokenizedString, PreTokenizer, Split};
use crate::props::TABLES;
use crate::regex::{PropClass, PropertyLookup};
use std::collections::HashMap;

// ── INTEGRATOR SEAM ─────────────────────────────────────────────────────
// The four model families are built by the model lanes. This ONE
// function is the swap point: replace its body with the dispatch into
// bpe/wordpiece/unigram/wordlevel constructors. Everything else in the
// pipeline codes against `Box<dyn Model>` and needs no change.

/// Builds the model runtime for a validated model config.
///
/// The constructor conventions differ deliberately: BPE and WordLevel
/// are pure destructuring over loader-validated config (infallible);
/// WordPiece and Unigram perform genuine load-time validation of
/// their own (unk membership, unk_id range) and stay fallible.
fn build_model(config: &ModelConfig) -> Result<Box<dyn Model>, TokenizerError> {
    match config {
        cfg @ ModelConfig::Bpe { .. } => Ok(Box::new(crate::bpe::Bpe::from_config(cfg.clone()))),
        ModelConfig::WordPiece {
            vocab,
            unk_token,
            continuing_subword_prefix,
            max_input_chars_per_word,
        } => Ok(Box::new(crate::wordpiece::WordPiece::new(
            vocab.clone(),
            unk_token.clone(),
            continuing_subword_prefix.clone(),
            *max_input_chars_per_word,
        )?)),
        ModelConfig::Unigram {
            vocab,
            unk_id,
            byte_fallback,
        } => Ok(Box::new(crate::unigram::Unigram::new(
            vocab.clone(),
            *unk_id,
            *byte_fallback,
        )?)),
        cfg @ ModelConfig::WordLevel { .. } => Ok(Box::new(
            crate::wordlevel::WordLevel::from_config(cfg.clone()),
        )),
    }
}

// ── Added-token machinery (reference `added_vocabulary.rs`) ─────────────

/// The added vocabulary: the full §5 flag set, split into the
/// raw-matching (non-normalized) and normalized matcher sets.
struct AddedVocabulary {
    tokens: Vec<AddedTokenConfig>,
    by_content: HashMap<String, u32>,
    /// id → index into `tokens`.
    by_id: HashMap<u32, usize>,
    /// `(token index, match text)` for `normalized == false` tokens —
    /// matched against the RAW input.
    raw_patterns: Vec<(usize, String)>,
    /// `(token index, match text)` for `normalized == true` tokens —
    /// the content run through the loaded normalizer, matched after
    /// normalization.
    normalized_patterns: Vec<(usize, String)>,
}

fn is_word_char(c: char) -> bool {
    TABLES.is_class(c, PropClass::Word)
}

/// Byte index where the trailing-whitespace run of `s` begins
/// (`s.len()` when there is none) — the reference's `\s*$` match start.
fn trailing_ws_start(s: &str) -> usize {
    s.rfind(|c: char| !c.is_whitespace())
        .map_or(0, |i| i + s[i..].chars().next().map_or(1, char::len_utf8))
}

/// Length of the leading-whitespace run of `s` — the reference's
/// `^\s*` match end.
fn leading_ws_len(s: &str) -> usize {
    s.find(|c: char| !c.is_whitespace()).unwrap_or(s.len())
}

impl AddedVocabulary {
    fn build(
        tokens: &[AddedTokenConfig],
        normalizer: Option<&Normalizer>,
    ) -> Result<AddedVocabulary, TokenizerError> {
        let mut by_content = HashMap::with_capacity(tokens.len());
        let mut by_id = HashMap::with_capacity(tokens.len());
        let mut raw_patterns = Vec::new();
        let mut normalized_patterns = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            if token.content.is_empty() {
                continue;
            }
            by_content.insert(token.content.clone(), token.id);
            by_id.insert(token.id, i);
            if token.normalized {
                let text = match normalizer {
                    Some(n) => {
                        let mut content = NormalizedString::from(token.content.as_str());
                        n.apply(&mut content)?;
                        content.get().to_string()
                    }
                    None => token.content.clone(),
                };
                normalized_patterns.push((i, text));
            } else {
                raw_patterns.push((i, token.content.clone()));
            }
        }
        Ok(AddedVocabulary {
            tokens: tokens.to_vec(),
            by_content,
            by_id,
            raw_patterns,
            normalized_patterns,
        })
    }

    fn is_special(&self, content: &str) -> bool {
        self.by_content.contains_key(content)
            && self
                .by_id
                .get(&self.by_content[content])
                .is_some_and(|&i| self.tokens[i].special)
    }

    /// The `(matched token, span)` partition of `sentence`, reference
    /// `find_matches`: leftmost-longest raw matches, then the
    /// single_word / lstrip / rstrip flag pass.
    fn find_matches(
        &self,
        sentence: &str,
        patterns: &[(usize, String)],
    ) -> Vec<(Option<usize>, (usize, usize))> {
        if sentence.is_empty() {
            return vec![(None, (0, 0))];
        }
        let mut raw: Vec<(usize, usize, usize)> = Vec::new();
        if !patterns.is_empty() {
            let mut pos = 0;
            while pos < sentence.len() {
                let rest = &sentence[pos..];
                let mut best: Option<(usize, usize)> = None;
                for (idx, pattern) in patterns {
                    if !pattern.is_empty()
                        && rest.starts_with(pattern.as_str())
                        && best.is_none_or(|(_, len)| pattern.len() > len)
                    {
                        best = Some((*idx, pattern.len()));
                    }
                }
                match best {
                    Some((idx, len)) => {
                        raw.push((idx, pos, pos + len));
                        pos += len;
                    }
                    None => pos += rest.chars().next().map_or(1, char::len_utf8),
                }
            }
        }

        let mut splits = Vec::new();
        let mut start_offset = 0;
        for (idx, mut start, mut stop) in raw {
            let token = &self.tokens[idx];
            if token.single_word {
                let start_free = start == 0
                    || !sentence[..start]
                        .chars()
                        .next_back()
                        .is_some_and(is_word_char);
                let stop_free = stop == sentence.len()
                    || !sentence[stop..].chars().next().is_some_and(is_word_char);
                if !start_free || !stop_free {
                    continue;
                }
            }
            if token.lstrip {
                // The previous match may already own those spaces.
                start = std::cmp::max(trailing_ws_start(&sentence[..start]), start_offset);
            }
            if token.rstrip {
                stop += leading_ws_len(&sentence[stop..]);
            }
            if start_offset < start {
                splits.push((None, (start_offset, start)));
            }
            splits.push((Some(idx), (start, stop)));
            start_offset = stop;
        }
        if start_offset != sentence.len() {
            splits.push((None, (start_offset, sentence.len())));
        }
        splits
    }

    /// Reference `split_with_indices`: matched spans become one-token
    /// splits (the token text is the SPAN's text — strip-extended
    /// whitespace included), the rest stay open for the pipeline.
    fn split_with_indices(
        &self,
        sentence: NormalizedString,
        patterns: &[(usize, String)],
    ) -> Vec<Split> {
        self.find_matches(sentence.get(), patterns)
            .into_iter()
            .filter_map(|(idx, (start, stop))| {
                let slice = sentence.slice(start..stop)?;
                let tokens = idx.map(|i| {
                    let value = slice.get().to_string();
                    let len = value.len();
                    vec![ModelToken {
                        id: self.tokens[i].id,
                        value,
                        offsets: (0, len),
                    }]
                });
                Some(Split {
                    normalized: slice,
                    tokens,
                })
            })
            .collect()
    }

    /// Reference `extract_and_normalize`: the two-pass added-token
    /// split around normalization.
    fn extract_and_normalize(
        &self,
        normalizer: Option<&Normalizer>,
        sequence: &str,
    ) -> Result<PreTokenizedString, TokenizerError> {
        let mut pretokenized = PreTokenizedString::from(sequence);
        // 1. Non-normalized tokens split the raw text.
        pretokenized
            .split(|_, sequence| Ok(self.split_with_indices(sequence, &self.raw_patterns)))?;
        // 2. Each remaining piece is normalized, then the normalized
        //    tokens split it.
        pretokenized.split(|_, mut sequence| {
            if let Some(n) = normalizer {
                n.apply(&mut sequence)?;
            }
            Ok(self.split_with_indices(sequence, &self.normalized_patterns))
        })?;
        Ok(pretokenized)
    }
}

// ── The tokenizer ───────────────────────────────────────────────────────

/// A loaded, runnable tokenizer: the five pipeline stages, the added
/// vocabulary, and the artifact's saved truncation/padding as the
/// active defaults.
pub struct Tokenizer {
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    model: Box<dyn Model>,
    post_processor: Option<PostProcessor>,
    decoder: Option<Decoder>,
    added: AddedVocabulary,
    truncation: Option<TruncationConfig>,
    padding: Option<PaddingConfig>,
    /// The model vocabulary as declared by the artifact (Unigram rows
    /// index in order) — the `get_vocab` / `vocab_size` surface, which
    /// the [`Model`] seam does not carry.
    base_vocab: HashMap<String, u32>,
    /// Added tokens whose content is NOT in the base vocab — the
    /// `vocab_size(true)` increment.
    extra_added: usize,
}

/// The model vocabulary declared by an artifact model config.
fn base_vocab(config: &ModelConfig) -> HashMap<String, u32> {
    match config {
        ModelConfig::Bpe { vocab, .. }
        | ModelConfig::WordPiece { vocab, .. }
        | ModelConfig::WordLevel { vocab, .. } => vocab.iter().cloned().collect(),
        ModelConfig::Unigram { vocab, .. } => vocab
            .iter()
            .enumerate()
            .map(|(i, (piece, _))| (piece.clone(), u32::try_from(i).unwrap_or(u32::MAX)))
            .collect(),
    }
}

impl Tokenizer {
    /// Loads a `tokenizer.json` artifact from bytes — the house bytes
    /// rule (`std::fs::read` is the caller's one-liner). Validates the
    /// WHOLE artifact eagerly: schema, vocab cross-checks, regex
    /// compilation, charsmap decode.
    pub fn from_bytes(bytes: &[u8]) -> Result<Tokenizer, TokenizerError> {
        let artifact = TokenizerArtifact::from_bytes(bytes)?;
        let model = build_model(&artifact.model)?;
        Tokenizer::from_artifact(&artifact, model)
    }

    /// Assembles a tokenizer from a parsed artifact and a model
    /// runtime. This is [`Tokenizer::from_bytes`] minus the model
    /// construction — the constructor for callers (and tests) that
    /// bring their own [`Model`].
    pub fn from_artifact(
        artifact: &TokenizerArtifact,
        model: Box<dyn Model>,
    ) -> Result<Tokenizer, TokenizerError> {
        let normalizer = artifact
            .normalizer
            .as_ref()
            .map(Normalizer::compile)
            .transpose()?;
        let pre_tokenizer = artifact
            .pre_tokenizer
            .as_ref()
            .map(PreTokenizer::compile)
            .transpose()?;
        let post_processor = artifact
            .post_processor
            .as_ref()
            .map(PostProcessor::compile)
            .transpose()?;
        let decoder = artifact
            .decoder
            .as_ref()
            .map(Decoder::compile)
            .transpose()?;
        let added = AddedVocabulary::build(&artifact.added_tokens, normalizer.as_ref())?;
        let base_vocab = base_vocab(&artifact.model);
        let extra_added = added
            .tokens
            .iter()
            .filter(|t| !t.content.is_empty() && !base_vocab.contains_key(&t.content))
            .count();
        Ok(Tokenizer {
            normalizer,
            pre_tokenizer,
            model,
            post_processor,
            decoder,
            added,
            truncation: artifact.truncation,
            padding: artifact.padding.clone(),
            base_vocab,
            extra_added,
        })
    }

    // ── Encode ──────────────────────────────────────────────────────────

    fn encode_single(&self, text: &str, type_id: u32) -> Result<Encoding, TokenizerError> {
        let mut pretokenized = self
            .added
            .extract_and_normalize(self.normalizer.as_ref(), text)?;
        if let Some(pre_tokenizer) = &self.pre_tokenizer {
            pre_tokenizer.apply(&mut pretokenized)?;
        }
        pretokenized.tokenize(|normalized| self.model.tokenize(normalized.get()))?;
        pretokenized.into_encoding(None, type_id)
    }

    /// Encodes one text through the full pipeline, post-processing
    /// included.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding, TokenizerError> {
        let encoding = self.encode_single(text, 0)?;
        self.post_process(encoding, None, add_special_tokens)
    }

    /// Encodes a pair — the BERT-class story: pair templates, type
    /// ids, pair truncation strategies.
    pub fn encode_pair(
        &self,
        first: &str,
        second: &str,
        add_special_tokens: bool,
    ) -> Result<Encoding, TokenizerError> {
        let encoding = self.encode_single(first, 0)?;
        let pair = self.encode_single(second, 1)?;
        self.post_process(encoding, Some(pair), add_special_tokens)
    }

    /// Encodes a batch: a sequential host loop, then batch padding
    /// (`BatchLongest` needs the whole batch). Deliberately not
    /// internally threaded — callers who want parallelism run chunks
    /// on `std::thread::scope` threads and apply
    /// [`crate::encoding::pad_encodings`] themselves.
    pub fn encode_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, TokenizerError> {
        let mut encodings = texts
            .iter()
            .map(|text| self.encode(text.as_ref(), add_special_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(params) = &self.padding {
            pad_encodings(&mut encodings, params);
        }
        Ok(encodings)
    }

    /// The reference `post_process` order: truncate (with special-token
    /// headroom), process, pad.
    fn post_process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding, TokenizerError> {
        let (encoding, pair_encoding) = match &self.truncation {
            Some(trunc) => {
                let n_added = self.n_added_tokens(pair_encoding.is_some());
                let params = if add_special_tokens && n_added > 0 {
                    TruncationConfig {
                        max_length: trunc.max_length.saturating_sub(n_added),
                        ..*trunc
                    }
                } else {
                    *trunc
                };
                truncate_encodings(encoding, pair_encoding, &params)?
            }
            None => (encoding, pair_encoding),
        };
        let mut final_encoding = match &self.post_processor {
            Some(processor) => processor.process(encoding, pair_encoding, add_special_tokens)?,
            None => match pair_encoding {
                Some(pair) => {
                    let mut merged = encoding;
                    merged.merge_with(pair);
                    merged
                }
                None => encoding,
            },
        };
        if let Some(params) = &self.padding {
            let mut batch = [final_encoding];
            pad_encodings(&mut batch, params);
            [final_encoding] = batch;
        }
        Ok(final_encoding)
    }

    fn n_added_tokens(&self, is_pair: bool) -> usize {
        self.post_processor
            .as_ref()
            .map_or(0, |p| p.added_tokens(is_pair))
    }

    // ── Decode ──────────────────────────────────────────────────────────

    /// Decodes ids through the artifact's decoder chain. An id outside
    /// the vocabulary is a loud error naming the id (§7).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self
                .added
                .by_id
                .get(&id)
                .map(|&i| self.added.tokens[i].content.as_str())
                .or_else(|| self.model.id_to_token(id))
                .ok_or(TokenizerError::Decode {
                    id,
                    vocab_size: self.vocab_size(true),
                })?;
            if skip_special_tokens && self.added.is_special(token) {
                continue;
            }
            tokens.push(token.to_string());
        }
        match &self.decoder {
            Some(decoder) => decoder.decode(tokens),
            None => Ok(tokens.join(" ")),
        }
    }

    /// Decodes a batch sequentially.
    pub fn decode_batch(
        &self,
        sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>, TokenizerError> {
        sequences
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Incremental detokenization for generation loops
    /// ([`DecodeStream`]).
    pub fn decode_stream(&self, skip_special_tokens: bool) -> DecodeStream<'_> {
        DecodeStream::new(self, skip_special_tokens)
    }

    // ── Lookups (§7's quartet) ──────────────────────────────────────────

    /// Token → id, added tokens first.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.added
            .by_content
            .get(token)
            .copied()
            .or_else(|| self.model.token_to_id(token))
    }

    /// Id → token, added tokens first.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.added
            .by_id
            .get(&id)
            .map(|&i| self.added.tokens[i].content.as_str())
            .or_else(|| self.model.id_to_token(id))
    }

    /// Vocabulary size; `with_added` counts the union of model and
    /// added token strings (the reference's `get_vocab(true).len()`).
    pub fn vocab_size(&self, with_added: bool) -> usize {
        self.base_vocab.len() + if with_added { self.extra_added } else { 0 }
    }

    /// The vocabulary as token → id; added tokens overlay the model's.
    pub fn get_vocab(&self, with_added: bool) -> HashMap<String, u32> {
        let mut vocab = self.base_vocab.clone();
        if with_added {
            for token in &self.added.tokens {
                if !token.content.is_empty() {
                    vocab.insert(token.content.clone(), token.id);
                }
            }
        }
        vocab
    }

    // ── Truncation / padding overrides (§7) ─────────────────────────────

    /// Overrides (or with `None` disables) the active truncation. The
    /// artifact's saved section loaded as the initial value.
    pub fn set_truncation(&mut self, params: Option<TruncationConfig>) {
        self.truncation = params;
    }

    /// The active truncation params.
    pub fn truncation(&self) -> Option<&TruncationConfig> {
        self.truncation.as_ref()
    }

    /// Overrides (or disables) the active padding.
    pub fn set_padding(&mut self, params: Option<PaddingConfig>) {
        self.padding = params;
    }

    /// The active padding params.
    pub fn padding(&self) -> Option<&PaddingConfig> {
        self.padding.as_ref()
    }
}

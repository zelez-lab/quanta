//! Execution of the post-processor configs — ports of HF tokenizers
//! 0.21.x `processors/*.rs` plus the `PostProcessor` trait's default
//! `process` wrapper:
//!
//! - Before any variant runs, single/pair encodings get type ids 0/1
//!   (top level only — overflow windows keep theirs), exactly like the
//!   reference's default `process`; the variant outputs then merge with
//!   the overflow-cartesian rules, which is how templates reach every
//!   stride window.
//! - `TemplateProcessing` — single and pair templates over `Sequence` /
//!   `SpecialToken` pieces; sequence pieces overwrite type ids with the
//!   piece's, special pieces materialize their id runs with offsets
//!   `(0, 0)`, special mask 1, word id `None`, and vanish when
//!   `add_special_tokens` is false.
//! - `BertProcessing` (`[CLS] A [SEP]` / `+ B [SEP]`) and
//!   `RobertaProcessing` (`<s> A </s>` / `</s> B </s>`, all type ids 0,
//!   optional ByteLevel offset trim) — the fixed-form ancestors, each
//!   rewrapping its overflow windows the way the reference does.
//! - `ByteLevel` — `trim_offsets` only: leading `Ġ`/whitespace chars of
//!   a token shrink its offset span (the first token keeps ONE space
//!   when `add_prefix_space` added it), trailing ones shrink the end.
//! - `Sequence` — ordered composition over the encodings list.

use crate::artifact::{PostProcessorConfig, SequenceId, TemplatePiece};
use crate::encoding::Encoding;
use crate::error::TokenizerError;
use crate::pretokenize::byte_to_char;
use std::collections::HashMap;

/// A compiled, executable post-processor.
#[derive(Debug)]
pub enum PostProcessor {
    Template {
        single: Vec<TemplatePiece>,
        pair: Vec<TemplatePiece>,
        /// Special-token name → (ids, tokens), lengths equal (validated
        /// at the artifact layer).
        special_tokens: HashMap<String, (Vec<u32>, Vec<String>)>,
    },
    Bert {
        sep: (String, u32),
        cls: (String, u32),
    },
    Roberta {
        sep: (String, u32),
        cls: (String, u32),
        trim_offsets: bool,
        add_prefix_space: bool,
    },
    ByteLevel {
        add_prefix_space: bool,
        trim_offsets: bool,
    },
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    /// Compiles an artifact post-processor config into its executor.
    pub fn compile(config: &PostProcessorConfig) -> Result<PostProcessor, TokenizerError> {
        Ok(match config {
            PostProcessorConfig::Template {
                single,
                pair,
                special_tokens,
            } => PostProcessor::Template {
                single: single.clone(),
                pair: pair.clone(),
                special_tokens: special_tokens
                    .iter()
                    .map(|s| (s.id.clone(), (s.ids.clone(), s.tokens.clone())))
                    .collect(),
            },
            PostProcessorConfig::Bert { sep, cls } => PostProcessor::Bert {
                sep: sep.clone(),
                cls: cls.clone(),
            },
            PostProcessorConfig::Roberta {
                sep,
                cls,
                trim_offsets,
                add_prefix_space,
            } => PostProcessor::Roberta {
                sep: sep.clone(),
                cls: cls.clone(),
                trim_offsets: *trim_offsets,
                add_prefix_space: *add_prefix_space,
            },
            PostProcessorConfig::ByteLevel {
                add_prefix_space,
                trim_offsets,
                use_regex: _,
            } => PostProcessor::ByteLevel {
                add_prefix_space: *add_prefix_space,
                trim_offsets: *trim_offsets,
            },
            PostProcessorConfig::Sequence(inner) => PostProcessor::Sequence(
                inner
                    .iter()
                    .map(PostProcessor::compile)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// How many special tokens processing will add — the truncation
    /// headroom (reference `added_tokens`).
    pub fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            PostProcessor::Template {
                single,
                pair,
                special_tokens,
            } => {
                let template = if is_pair { pair } else { single };
                template
                    .iter()
                    .map(|piece| match piece {
                        TemplatePiece::Sequence { .. } => 0,
                        TemplatePiece::SpecialToken { id, .. } => {
                            special_tokens.get(id).map_or(0, |(ids, _)| ids.len())
                        }
                    })
                    .sum()
            }
            PostProcessor::Bert { .. } => {
                if is_pair {
                    3
                } else {
                    2
                }
            }
            PostProcessor::Roberta { .. } => {
                if is_pair {
                    4
                } else {
                    2
                }
            }
            PostProcessor::ByteLevel { .. } => 0,
            PostProcessor::Sequence(inner) => inner.iter().map(|p| p.added_tokens(is_pair)).sum(),
        }
    }

    /// The reference trait's default `process`: stamp segment type ids,
    /// run the variant, merge.
    pub fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding, TokenizerError> {
        let mut encodings = match pair_encoding {
            Some(pair) => vec![encoding, pair],
            None => vec![encoding],
        };
        for (i, encoding) in encodings.iter_mut().enumerate() {
            let len = encoding.len();
            encoding.type_ids = vec![u32::try_from(i).expect("two sequences"); len];
        }
        let encodings = self.process_encodings(encodings, add_special_tokens)?;
        Ok(Encoding::merge(encodings))
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Vec<Encoding>, TokenizerError> {
        match self {
            PostProcessor::Template {
                single,
                pair,
                special_tokens,
            } => {
                let template = if encodings.len() == 2 { pair } else { single };
                apply_template(template, special_tokens, encodings, add_special_tokens)
            }
            PostProcessor::Bert { sep, cls } => {
                if !add_special_tokens {
                    return Ok(encodings);
                }
                Ok(encodings
                    .into_iter()
                    .enumerate()
                    .map(|(i, encoding)| {
                        if i == 0 {
                            wrap(encoding, Some((&cls.0, cls.1, 0)), Some((&sep.0, sep.1, 0)))
                        } else {
                            // The pair keeps its stamped type ids (1) and
                            // gains a type-1 [SEP].
                            wrap(encoding, None, Some((&sep.0, sep.1, 1)))
                        }
                    })
                    .collect())
            }
            PostProcessor::Roberta {
                sep,
                cls,
                trim_offsets,
                add_prefix_space,
            } => {
                if *trim_offsets {
                    for encoding in &mut encodings {
                        process_offsets(encoding, *add_prefix_space);
                        for overflow in &mut encoding.overflowing {
                            process_offsets(overflow, *add_prefix_space);
                        }
                    }
                }
                // Roberta assigns type id 0 everywhere, pair included.
                for encoding in &mut encodings {
                    let len = encoding.len();
                    encoding.type_ids = vec![0; len];
                }
                if !add_special_tokens {
                    return Ok(encodings);
                }
                Ok(encodings
                    .into_iter()
                    .enumerate()
                    .map(|(i, encoding)| {
                        if i == 0 {
                            wrap(encoding, Some((&cls.0, cls.1, 0)), Some((&sep.0, sep.1, 0)))
                        } else {
                            wrap(encoding, Some((&sep.0, sep.1, 0)), Some((&sep.0, sep.1, 0)))
                        }
                    })
                    .collect())
            }
            PostProcessor::ByteLevel {
                add_prefix_space,
                trim_offsets,
            } => {
                if *trim_offsets {
                    for encoding in &mut encodings {
                        process_offsets(encoding, *add_prefix_space);
                        for overflow in &mut encoding.overflowing {
                            process_offsets(overflow, *add_prefix_space);
                        }
                    }
                }
                Ok(encodings)
            }
            PostProcessor::Sequence(inner) => {
                for processor in inner {
                    encodings = processor.process_encodings(encodings, add_special_tokens)?;
                }
                Ok(encodings)
            }
        }
    }
}

/// Wraps `encoding` (and each of its overflow windows, one level, the
/// reference's recursion depth) in optional head/tail special tokens
/// given as `(token, id, type_id)`.
fn wrap(
    mut encoding: Encoding,
    head: Option<(&str, u32, u32)>,
    tail: Option<(&str, u32, u32)>,
) -> Encoding {
    let overflowing = std::mem::take(&mut encoding.overflowing)
        .into_iter()
        .map(|o| wrap(o, head, tail))
        .collect();
    let mut out = Encoding::default();
    if let Some((token, id, type_id)) = head {
        push_special(&mut out, token, id, type_id);
    }
    out.ids.extend(encoding.ids);
    out.type_ids.extend(encoding.type_ids);
    out.tokens.extend(encoding.tokens);
    out.words.extend(encoding.words);
    out.offsets.extend(encoding.offsets);
    out.special_tokens_mask
        .extend(encoding.special_tokens_mask.iter().map(|_| 0));
    out.attention_mask
        .extend(encoding.attention_mask.iter().map(|_| 1));
    if let Some((token, id, type_id)) = tail {
        push_special(&mut out, token, id, type_id);
    }
    out.overflowing = overflowing;
    out
}

fn push_special(encoding: &mut Encoding, token: &str, id: u32, type_id: u32) {
    encoding.ids.push(id);
    encoding.type_ids.push(type_id);
    encoding.tokens.push(token.to_string());
    encoding.words.push(None);
    encoding.offsets.push((0, 0));
    encoding.special_tokens_mask.push(1);
    encoding.attention_mask.push(1);
}

/// Reference `apply_template`: pieces expand in order; sequence pieces
/// pull (and re-type) the matching input encoding, special pieces
/// materialize when `add_special_tokens`.
fn apply_template(
    template: &[TemplatePiece],
    special_tokens: &HashMap<String, (Vec<u32>, Vec<String>)>,
    mut encodings: Vec<Encoding>,
    add_special_tokens: bool,
) -> Result<Vec<Encoding>, TokenizerError> {
    let mut out = Vec::with_capacity(template.len());
    for piece in template {
        match piece {
            TemplatePiece::Sequence { id, type_id } => {
                let i = usize::from(*id != SequenceId::A);
                let encoding = encodings.get_mut(i).ok_or_else(|| TokenizerError::Encode {
                    what: format!(
                        "the template references sequence {} but the input has no pair",
                        if i == 0 { "A" } else { "B" }
                    ),
                })?;
                let len = encoding.len();
                encoding.type_ids = vec![*type_id; len];
                out.push(encoding.clone());
            }
            TemplatePiece::SpecialToken { id, type_id } => {
                if !add_special_tokens {
                    continue;
                }
                let (ids, tokens) =
                    special_tokens
                        .get(id)
                        .ok_or_else(|| TokenizerError::Encode {
                            what: format!(
                                "the template references special token {id:?}, which is not \
                             declared in special_tokens"
                            ),
                        })?;
                let len = ids.len();
                out.push(Encoding::from_parts(
                    ids.clone(),
                    vec![*type_id; len],
                    tokens.clone(),
                    vec![None; len],
                    vec![(0, 0); len],
                    vec![1; len],
                    vec![1; len],
                    Vec::new(),
                ));
            }
        }
    }
    Ok(out)
}

/// Reference `byte_level::process_offsets` — the `trim_offsets` offset
/// surgery: leading/trailing `Ġ` (the byte-level space stand-in) and
/// whitespace chars shrink the token's offset span; with
/// `add_prefix_space`, a single leading space on the first token is the
/// one the pipeline added and stays.
pub(crate) fn process_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    let g = byte_to_char(b' ');
    for (i, (token, offsets)) in encoding
        .tokens
        .iter()
        .zip(encoding.offsets.iter_mut())
        .enumerate()
    {
        let mut leading = token
            .chars()
            .take_while(|c| *c == g || c.is_whitespace())
            .count();
        let trailing = token
            .chars()
            .rev()
            .take_while(|c| *c == g || c.is_whitespace())
            .count();
        if leading == 0 && trailing == 0 {
            continue;
        }
        if leading > 0 {
            let is_first = i == 0 || offsets.0 == 0;
            if is_first && add_prefix_space && leading == 1 {
                leading = 0;
            }
            offsets.0 = std::cmp::min(offsets.0 + leading, offsets.1);
        }
        if trailing > 0 && offsets.1 >= trailing {
            offsets.1 = std::cmp::max(offsets.1 - trailing, offsets.0);
        }
    }
}

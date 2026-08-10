//! Execution of the decoder configs plus the streaming decode helper —
//! ports of HF tokenizers 0.21.x `decoders/*.rs` and the
//! `DecodeStream` state machine in `tokenizer/mod.rs`.
//!
//! Decoders run as a chain over the token list (`decode_chain`); the
//! final string is the chain output joined with nothing. The rows:
//!
//! - `ByteLevel` — every token maps through the inverse byte bijection
//!   into ONE byte stream, then lossy UTF-8 (a token with any
//!   non-table char passes through as raw bytes). Collecting across
//!   tokens first is what makes multibyte chars split across ids
//!   decode correctly.
//! - `WordPiece` (`##` continuation + the reference's cleanup list),
//!   `BPEDecoder` (suffix → space, none on the last token),
//!   `Metaspace` (replacement → space; the first token's replacements
//!   vanish unless the scheme is `never`), `ByteFallback` (`<0xAB>`
//!   token runs → bytes → string, `�` per byte on invalid runs),
//!   `Fuse`, `Strip` (up to `start`/`stop` copies of `content` off the
//!   ends), `Replace` (pattern → content per token), `CTC`
//!   (consecutive-duplicate collapse, pad strip, cleanup), `Sequence`.
//!
//! [`DecodeStream`] is the §7 prefix-diff streamer: each step decodes
//! the held ids, emits the new valid suffix only when it grew and does
//! not end in a replacement char (bytes split across tokens stay held),
//! then re-anchors its prefix window.

use crate::artifact::DecoderConfig;
use crate::error::TokenizerError;
use crate::normalized::Matcher;
use crate::pretokenize::char_to_byte;
use crate::tokenizer::Tokenizer;

/// A compiled, executable decoder chain element.
#[derive(Debug)]
pub enum Decoder {
    ByteLevel,
    WordPiece {
        prefix: String,
        cleanup: bool,
    },
    BpeDecoder {
        suffix: String,
    },
    Metaspace {
        replacement: char,
        /// `true` when the prepend scheme is `always` or `first` — the
        /// decoder only distinguishes `never`.
        prepend: bool,
    },
    ByteFallback,
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Replace {
        matcher: Matcher,
        content: String,
    },
    Ctc {
        pad_token: String,
        word_delimiter_token: String,
        cleanup: bool,
    },
    Sequence(Vec<Decoder>),
}

impl Decoder {
    /// Compiles an artifact decoder config into its executor.
    pub fn compile(config: &DecoderConfig) -> Result<Decoder, TokenizerError> {
        Ok(match config {
            DecoderConfig::ByteLevel { .. } => Decoder::ByteLevel,
            DecoderConfig::WordPiece { prefix, cleanup } => Decoder::WordPiece {
                prefix: prefix.clone(),
                cleanup: *cleanup,
            },
            DecoderConfig::BpeDecoder { suffix } => Decoder::BpeDecoder {
                suffix: suffix.clone(),
            },
            DecoderConfig::Metaspace {
                replacement,
                prepend_scheme,
                split: _,
            } => Decoder::Metaspace {
                replacement: *replacement,
                prepend: *prepend_scheme != crate::artifact::PrependScheme::Never,
            },
            DecoderConfig::ByteFallback => Decoder::ByteFallback,
            DecoderConfig::Fuse => Decoder::Fuse,
            DecoderConfig::Strip {
                content,
                start,
                stop,
            } => Decoder::Strip {
                content: *content,
                start: *start,
                stop: *stop,
            },
            DecoderConfig::Replace { pattern, content } => Decoder::Replace {
                matcher: Matcher::compile(pattern)?,
                content: content.clone(),
            },
            DecoderConfig::Ctc {
                pad_token,
                word_delimiter_token,
                cleanup,
            } => Decoder::Ctc {
                pad_token: pad_token.clone(),
                word_delimiter_token: word_delimiter_token.clone(),
                cleanup: *cleanup,
            },
            DecoderConfig::Sequence(inner) => Decoder::Sequence(
                inner
                    .iter()
                    .map(Decoder::compile)
                    .collect::<Result<_, _>>()?,
            ),
        })
    }

    /// Runs the chain and joins the result.
    pub fn decode(&self, tokens: Vec<String>) -> Result<String, TokenizerError> {
        Ok(self.decode_chain(tokens)?.join(""))
    }

    /// One chain step (reference `decode_chain` per variant).
    pub fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>, TokenizerError> {
        match self {
            Decoder::ByteLevel => {
                let bytes = tokens
                    .into_iter()
                    .flat_map(|t| {
                        t.chars()
                            .map(char_to_byte)
                            .collect::<Option<Vec<u8>>>()
                            .unwrap_or_else(|| t.as_bytes().to_vec())
                    })
                    .collect::<Vec<u8>>();
                Ok(vec![String::from_utf8_lossy(&bytes).into_owned()])
            }
            Decoder::WordPiece { prefix, cleanup } => Ok(tokens
                .into_iter()
                .enumerate()
                .map(|(i, mut token)| {
                    if i != 0 {
                        if token.starts_with(prefix.as_str()) {
                            token = token.replacen(prefix.as_str(), "", 1);
                        } else {
                            token = format!(" {token}");
                        }
                    }
                    if *cleanup {
                        token = wordpiece_cleanup(&token);
                    }
                    token
                })
                .collect()),
            Decoder::BpeDecoder { suffix } => {
                let Some(last) = tokens.len().checked_sub(1) else {
                    return Ok(Vec::new());
                };
                Ok(tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, token)| {
                        let replacement = if i == last { "" } else { " " };
                        token.replace(suffix.as_str(), replacement)
                    })
                    .collect())
            }
            Decoder::Metaspace {
                replacement,
                prepend,
            } => Ok(tokens
                .iter()
                .enumerate()
                .map(|(i, token)| {
                    token
                        .chars()
                        .filter_map(|c| {
                            if c == *replacement {
                                if i == 0 && *prepend { None } else { Some(' ') }
                            } else {
                                Some(c)
                            }
                        })
                        .collect::<String>()
                })
                .collect()),
            Decoder::ByteFallback => {
                let mut new_tokens: Vec<String> = Vec::new();
                let mut pending: Vec<u8> = Vec::new();
                let flush = |pending: &mut Vec<u8>, out: &mut Vec<String>| {
                    if pending.is_empty() {
                        return;
                    }
                    match std::str::from_utf8(pending) {
                        Ok(s) => out.push(s.to_string()),
                        Err(_) => {
                            out.extend(std::iter::repeat_n("\u{FFFD}".to_string(), pending.len()))
                        }
                    }
                    pending.clear();
                };
                for token in tokens {
                    let byte =
                        (token.len() == 6 && token.starts_with("<0x") && token.ends_with('>'))
                            .then(|| u8::from_str_radix(&token[3..5], 16).ok())
                            .flatten();
                    match byte {
                        Some(b) => pending.push(b),
                        None => {
                            flush(&mut pending, &mut new_tokens);
                            new_tokens.push(token);
                        }
                    }
                }
                flush(&mut pending, &mut new_tokens);
                Ok(new_tokens)
            }
            Decoder::Fuse => Ok(vec![tokens.concat()]),
            Decoder::Strip {
                content,
                start,
                stop,
            } => Ok(tokens
                .into_iter()
                .map(|token| {
                    let chars: Vec<char> = token.chars().collect();
                    let mut start_cut = 0;
                    for (i, &c) in chars.iter().enumerate().take(*start) {
                        if c == *content {
                            start_cut = i + 1;
                        } else {
                            break;
                        }
                    }
                    let mut stop_cut = chars.len();
                    for i in 0..std::cmp::min(*stop, chars.len()) {
                        let index = chars.len() - i - 1;
                        if index < start_cut || chars[index] != *content {
                            break;
                        }
                        stop_cut = index;
                    }
                    chars[start_cut..stop_cut].iter().collect::<String>()
                })
                .collect()),
            Decoder::Replace { matcher, content } => tokens
                .into_iter()
                .map(|token| {
                    let mut out = String::with_capacity(token.len());
                    for ((start, end), is_match) in matcher.spans(&token)? {
                        if is_match {
                            out.push_str(content);
                        } else {
                            out.push_str(&token[start..end]);
                        }
                    }
                    Ok(out)
                })
                .collect(),
            Decoder::Ctc {
                pad_token,
                word_delimiter_token,
                cleanup,
            } => {
                let mut deduped: Vec<String> = Vec::new();
                for token in tokens {
                    if deduped.last() != Some(&token) {
                        deduped.push(token);
                    }
                }
                Ok(deduped
                    .into_iter()
                    .filter_map(|token| {
                        let mut replaced = token.replace(pad_token.as_str(), "");
                        if *cleanup {
                            replaced = wordpiece_cleanup(&replaced)
                                .replace(word_delimiter_token.as_str(), " ");
                        }
                        if replaced.is_empty() {
                            None
                        } else {
                            Some(replaced)
                        }
                    })
                    .collect())
            }
            Decoder::Sequence(inner) => {
                let mut tokens = tokens;
                for decoder in inner {
                    tokens = decoder.decode_chain(tokens)?;
                }
                Ok(tokens)
            }
        }
    }
}

/// The reference WordPiece cleanup list (shared with `CTC`), including
/// its `" do not"` contraction.
fn wordpiece_cleanup(dirty: &str) -> String {
    dirty
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" do not", " don't")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

// ── Streaming decode (reference `DecodeStream`) ─────────────────────────

/// Incremental detokenization for generation loops: feed ids one at a
/// time, get back the newly-valid piece of text (or `None` while bytes
/// split across tokens are still incomplete). Prefix-diff semantics —
/// concatenating every emitted piece equals decoding the whole id run.
///
/// Mechanics: the stream holds a window of `[context | active]` ids
/// where `context` is the previous emit's ids. Each step decodes the
/// whole window and emits the suffix beyond the last decode — sharing
/// the window head between both decodes cancels every position-
/// dependent decoder effect (first-token metaspace drops, WordPiece
/// spacing) exactly. A decode ending in U+FFFD means a multibyte char
/// is still split across ids: the step holds (`None`) until the
/// completing id arrives. On emit the context generation is dropped,
/// so the window stays bounded at two emit generations.
///
/// This is the pinned reference's documented `DecodeStream` semantics
/// with a sound window: the 0.21.0 implementation's index algebra
/// underflows (a panic) after seven consecutive emitting steps and
/// mis-emits just before — probed empirically against the real crate —
/// so the port keeps the contract, not the bug. The conformance anchor
/// is the property its docs state: concatenated emits equal the
/// whole-sequence decode.
pub struct DecodeStream<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    /// The held window: context ids (previous emit) + active ids.
    ids: Vec<u32>,
    /// `decode(ids)` as of the last emit — trimmed off future decodes.
    prefix: String,
    /// Where the active generation starts inside `ids`.
    anchor: usize,
}

impl<'a> DecodeStream<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, skip_special_tokens: bool) -> Self {
        DecodeStream {
            tokenizer,
            skip_special_tokens,
            ids: Vec::new(),
            prefix: String::new(),
            anchor: 0,
        }
    }

    /// Feeds one id; returns the newly decoded suffix once it is valid
    /// UTF-8 and strictly grows the text.
    pub fn step(&mut self, id: u32) -> Result<Option<String>, TokenizerError> {
        self.ids.push(id);
        let string = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
        if string.len() > self.prefix.len() && !string.ends_with('\u{FFFD}') {
            if !string.starts_with(&self.prefix) {
                return Err(TokenizerError::Encode {
                    what: "decode stream lost its prefix: the decoder chain rewrote \
                           already-emitted text"
                        .to_string(),
                });
            }
            let new_text = string[self.prefix.len()..].to_string();
            // Drop the spent context generation; everything held
            // becomes the context for the next emit.
            self.ids.drain(..self.anchor);
            self.anchor = self.ids.len();
            self.prefix = self.tokenizer.decode(&self.ids, self.skip_special_tokens)?;
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
}

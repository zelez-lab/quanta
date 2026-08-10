//! # quanta-tokenizers — run pretrained tokenizers, dependency-free
//!
//! Loading and running the tokenizers that ship with pretrained models:
//! token ids in (`Embedding` heads the quanta-nn chain on
//! `ids: Array<u32>`), strings out (the generation loop decodes).
//! **`tokenizer.json` — the HF `tokenizers` single-file artifact — is
//! THE interchange format**: it is what every model on the Hub ships
//! (GPT-class, BERT-class, Llama-class, T5, …), it is self-contained
//! (vocab, merges, the whole pipeline configuration, special tokens,
//! saved truncation/padding), and it is the target every other format
//! converts *into*. Training tokenizers is permanently out of scope —
//! a trainer *produces* the artifact this crate consumes.
//!
//! ## Placement
//!
//! A tokenizer's subject is strings and `u32` ids — no `Gpu`, no
//! `Field`, no backend features — so the crate carries **zero
//! dependencies, quanta crates included**: pure host string
//! processing, `std`-only, and it compiles for `wasm32` (dija's
//! browser lane tokenizes client-side with the same crate). The
//! bridge to the GPU stack is one documented line on quanta-nn's side
//! of the seam, not a dependency:
//! `Array::from_slice(gpu, &encoding.ids(), &[n])?`.
//!
//! ## I/O and claim boundaries
//!
//! Artifacts load from `&[u8]` (`std::fs::read` is the caller's
//! one-liner); there are no file-path wrappers. The conformance
//! reference is pinned ([`PINNED_REFERENCE`]): every `type` tag that
//! version deserializes either ships or is a named exclusion, and a
//! tag outside the inventory is a loud [`TokenizerError::UnknownTag`]
//! — format growth is additive work against a detected marker, never
//! a misparse. Artifact files are untrusted: every length is
//! text-bounded, every index bounds-checked, recursion capped.
//!
//! ## Layout (the arc lands in layers)
//!
//! - [`json`] — the complete in-crate RFC 8259 parser (the artifact
//!   substrate).
//! - [`artifact`] — the typed `tokenizer.json` schema layer: every
//!   pipeline family parsed, validated, and held as config structs.
//! - [`error`] — the taxonomy ([`TokenizerError`]), loud and
//!   offset/context-carrying.
//!
//! The pipeline stages that *execute* these configs (normalizers over
//! vendored Unicode tables, the closed-construct split-regex engine,
//! the four model families, encode/decode and the `Encoding` record)
//! build on this layer; the crate announces when the arc completes —
//! no intermediate state is presented as shipped.

#![forbid(unsafe_code)]

pub mod artifact;
pub mod bpe;
pub mod decode;
pub mod encoding;
pub mod error;
pub mod json;
pub mod model;
pub mod normalize;
pub mod normalized;
pub mod postprocess;
pub mod pretokenize;
pub mod props;
pub mod regex;
pub mod tokenizer;
pub mod unicode;
pub mod unigram;
pub mod wordlevel;
pub mod wordpiece;

pub use encoding::Encoding;
pub use error::{PINNED_REFERENCE, TokenizerError};
pub use model::{Model, ModelToken};
pub use tokenizer::Tokenizer;

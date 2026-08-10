//! The WordLevel runtime: exact vocabulary lookup, or the unk token.
//!
//! The whole model, per the pinned reference
//! ([`crate::PINNED_REFERENCE`], `models/wordlevel/mod.rs`): a
//! pre-token that is a vocabulary entry becomes that single token; any
//! other pre-token becomes the `unk_token` (its *vocabulary* spelling,
//! with offsets still spanning the whole pre-token); an `unk_token`
//! the vocabulary does not carry faults at tokenize time — the
//! reference's `MissingUnkToken`, raised only when an unknown
//! pre-token actually needs it.

use std::collections::HashMap;

use crate::artifact::ModelConfig;
use crate::error::TokenizerError;
use crate::model::{Model, ModelToken};

/// The WordLevel model runtime.
#[derive(Debug, Clone)]
pub struct WordLevel {
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
    unk_token: String,
}

impl WordLevel {
    /// Build the runtime from the artifact's `WordLevel` model config.
    /// The config is trusted — `TokenizerArtifact::from_bytes` has
    /// already validated it.
    ///
    /// # Panics
    ///
    /// If `config` is not [`ModelConfig::WordLevel`] (the pipeline
    /// matches the model family before constructing a runtime).
    pub fn from_config(config: ModelConfig) -> Self {
        let ModelConfig::WordLevel { vocab, unk_token } = config else {
            panic!("WordLevel::from_config requires a ModelConfig::WordLevel");
        };
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let mut id_to_token = HashMap::with_capacity(vocab.len());
        for (token, id) in vocab {
            token_to_id.insert(token.clone(), id);
            id_to_token.insert(id, token);
        }
        WordLevel {
            token_to_id,
            id_to_token,
            unk_token,
        }
    }
}

impl Model for WordLevel {
    fn tokenize(&self, pretoken: &str) -> Result<Vec<ModelToken>, TokenizerError> {
        if let Some(&id) = self.token_to_id.get(pretoken) {
            Ok(vec![ModelToken {
                id,
                value: pretoken.to_string(),
                offsets: (0, pretoken.len()),
            }])
        } else if let Some(&unk_id) = self.token_to_id.get(&self.unk_token) {
            Ok(vec![ModelToken {
                id: unk_id,
                value: self.unk_token.clone(),
                offsets: (0, pretoken.len()),
            }])
        } else {
            Err(TokenizerError::Encode {
                what: format!(
                    "WordLevel unk_token {:?} is not in the vocabulary — the artifact \
                     names an unknown-token spelling its own vocab lacks",
                    self.unk_token
                ),
            })
        }
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

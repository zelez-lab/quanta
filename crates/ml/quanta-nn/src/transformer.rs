//! TransformerEncoderLayer — the summit block, pure composition.
//!
//! Pre-LN (the modern default — this crate ships it as THE form rather
//! than a `norm_first` flag; PARITY is capabilities, not torch's API):
//!
//! ```text
//! h = x + Dropout(MHA(LN₁(x)))                 — attention sub-block
//! y = h + Dropout(W₂·SwiGLU(W₁·LN₂(h)))        — SwiGLU feed-forward
//! ```
//!
//! Every piece is a shipped, tested citizen: the fused streaming
//! attention (T9200–T9209, optionally causal/rotary), the fused
//! LayerNorm (T9210), fused SwiGLU (T9226), key-based dropout
//! (T9231–T9233), residuals through ordinary VJPs. The block is a
//! [`Layer`]: `apply` is the eval forward (dropout = identity),
//! `apply_train` threads the key and drops — so it stacks in tuples,
//! checkpoints through the named-state machinery, and steps through the
//! fused optimizers with zero new machinery.
//!
//! 2-D `[T, E]` sequence core, like MHA (batch = host loop).

use quanta_autograd::{AutogradError, DiffScalar, Tape, Var};
use quanta_core::Gpu;

use crate::activation::swiglu_var;
use crate::attention::{MhaParams, MultiheadAttention};
use crate::dropout::dropout_var;
use crate::layer::{Key, Layer, Linear, LinearParams, NormParams};
use crate::norm::layer_norm_var;

/// The block configuration. `ffn_hidden` is the SwiGLU output width `H`
/// (the first projection widens to `2H` for the gate).
#[derive(Debug, Clone, Copy)]
pub struct TransformerEncoderLayer {
    pub attn: MultiheadAttention,
    pub ffn_hidden: usize,
    /// Dropout rate after the attention and feed-forward sub-blocks
    /// (training only; eval never rescales — inverted dropout).
    pub dropout: f32,
    /// Epsilon for both LayerNorms.
    pub eps: f32,
}

impl TransformerEncoderLayer {
    /// The bidirectional encoder default: biased MHA, no rope,
    /// `ffn_hidden = 4·embed`, dropout 0.1.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        TransformerEncoderLayer {
            attn: MultiheadAttention::new(embed_dim, num_heads),
            ffn_hidden: 4 * embed_dim,
            dropout: 0.1,
            eps: 1e-5,
        }
    }

    fn embed(&self) -> usize {
        self.attn.embed_dim
    }

    fn norm(&self) -> crate::layer::LayerNorm {
        crate::layer::LayerNorm {
            dim: self.embed(),
            eps: self.eps,
        }
    }

    fn ffn1(&self) -> Linear {
        Linear {
            in_dim: self.embed(),
            out_dim: 2 * self.ffn_hidden,
            bias: true,
        }
    }

    fn ffn2(&self) -> Linear {
        Linear {
            in_dim: self.ffn_hidden,
            out_dim: self.embed(),
            bias: true,
        }
    }

    /// The shared forward; `keyed` = `Some(key)` is the training path
    /// (dropout on both sub-block outputs), `None` is eval.
    fn forward<T: DiffScalar + quanta_array::ToF64>(
        &self,
        tape: &Tape<T>,
        p: &EncoderLayerParamsVars<T>,
        x: &Var<T>,
        keyed: Option<Key>,
    ) -> Result<(Var<T>, Option<Key>), AutogradError> {
        let mut key = keyed;
        let mut drop = |tape: &Tape<T>, v: Var<T>| -> Result<Var<T>, AutogradError> {
            match key.take() {
                Some(k) if self.dropout > 0.0 => {
                    let (k_use, k_rest) = k.split();
                    key = Some(k_rest);
                    dropout_var(tape, &v, self.dropout, k_use)
                }
                other => {
                    key = other;
                    Ok(v)
                }
            }
        };

        // Attention sub-block: x + Dropout(MHA(LN₁(x))).
        let n1 = layer_norm_var(
            tape,
            x,
            &p.norm1.gamma,
            p.norm1.beta.as_ref().ok_or_else(beta_missing)?,
            self.eps,
        )?;
        let a = self.attn.apply(tape, &p.attn, &n1)?;
        let a = drop(tape, a)?;
        let h = x.add(&a)?;

        // Feed-forward sub-block: h + Dropout(W₂·SwiGLU(W₁·LN₂(h))).
        let n2 = layer_norm_var(
            tape,
            &h,
            &p.norm2.gamma,
            p.norm2.beta.as_ref().ok_or_else(beta_missing)?,
            self.eps,
        )?;
        let f = self.ffn1().apply(tape, &p.ffn1, &n2)?;
        let f = swiglu_var(tape, &f)?;
        let f = self.ffn2().apply(tape, &p.ffn2, &f)?;
        let f = drop(tape, f)?;
        let y = h.add(&f)?;
        Ok((y, key))
    }
}

fn beta_missing() -> AutogradError {
    AutogradError::from(quanta_array::ArrayError::Gpu(
        quanta_core::QuantaError::invalid_param("TransformerEncoderLayer: norm beta missing"),
    ))
}

/// The block's parameter tree — five shipped subtrees, derived.
#[derive(crate::layer::ParamTree)]
#[param_tree(crate = crate)]
pub struct EncoderLayerParams<T: DiffScalar> {
    pub norm1: NormParams<T>,
    pub attn: MhaParams<T>,
    pub norm2: NormParams<T>,
    pub ffn1: LinearParams<T>,
    pub ffn2: LinearParams<T>,
}

impl<T: DiffScalar + quanta_array::ToF64> Layer<T> for TransformerEncoderLayer {
    type Params = EncoderLayerParams<T>;

    fn in_dim(&self) -> Option<usize> {
        Some(self.embed())
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.embed()
    }

    fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError> {
        let (k1, rest) = key.split();
        let (k2, rest) = rest.split();
        let (k3, k4) = rest.split();
        Ok(EncoderLayerParams {
            norm1: self.norm().init(gpu, k1)?,
            attn: self.attn.init(gpu, k2)?,
            norm2: self.norm().init(gpu, k1)?,
            ffn1: self.ffn1().init(gpu, k3)?,
            ffn2: self.ffn2().init(gpu, k4)?,
        })
    }

    fn apply(
        &self,
        tape: &Tape<T>,
        p: &EncoderLayerParamsVars<T>,
        x: &Var<T>,
    ) -> Result<Var<T>, AutogradError> {
        Ok(self.forward(tape, p, x, None)?.0)
    }

    fn apply_train(
        &self,
        tape: &Tape<T>,
        p: &EncoderLayerParamsVars<T>,
        x: &Var<T>,
        key: Key,
    ) -> Result<(Var<T>, Key), AutogradError> {
        let (y, rest) = self.forward(tape, p, x, Some(key))?;
        // `forward` always hands the remainder back on the keyed path.
        let rest = rest.unwrap_or_else(|| Key::new(0));
        Ok((y, rest))
    }
}

//! `MultiheadAttention` — the module form over the fused SDPA.
//!
//! Four `Linear` projections around H independent fused attention heads:
//! project once at full width, slice per-head column blocks (differentiable
//! views), run each head through the fused streaming kernel
//! ([`crate::functional::sdpa_var`] — T9200–T9209, no score matrix on
//! either pass), merge, and project out. Optional rotary embeddings
//! (T9216–T9218) rotate every head's queries and keys before the scores.
//!
//! Params are a derived tree ([`MhaParams`], the first consumer of
//! `#[derive(ParamTree)]`) of four `LinearParams` — so the whole module
//! flattens, clips, checkpoints, and steps through the fused optimizers
//! like everything else.
//!
//! The 2-D `[T, E]` sequence is the core (batch = host loop, the crate
//! convention). `Layer::apply` is self-attention; cross-attention is the
//! inherent [`MultiheadAttention::attend`] with distinct query and
//! key/value sources.

use crate::functional::{Sdpa, sdpa_var};
use crate::layer::{Key, Layer, Linear, LinearParams, ParamTree};
use crate::rope::rope_var;
use quanta_array::{ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, RopeCache, Tape, Var};
use quanta_core::Gpu;

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        msg,
    )))
}

/// Multi-head attention configuration. `embed_dim` must divide evenly by
/// `num_heads`; the head width is `embed_dim / num_heads`.
#[derive(Debug, Clone, Copy)]
pub struct MultiheadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    /// Bias on all four projections.
    pub bias: bool,
    /// Causal (lower-triangular) masking inside every head.
    pub causal: bool,
    /// Rotate queries and keys per head (rotary embeddings) before the
    /// scores; positions are `0..T` of each source.
    pub rope: bool,
    /// Rotary frequency base (the usual `10_000`).
    pub rope_base: f64,
}

impl MultiheadAttention {
    /// Bidirectional, biased, no rope — the transformer-encoder default.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        MultiheadAttention {
            embed_dim,
            num_heads,
            bias: true,
            causal: false,
            rope: false,
            rope_base: 10_000.0,
        }
    }

    /// Causal + rope — the decoder-block default.
    pub fn decoder(embed_dim: usize, num_heads: usize) -> Self {
        MultiheadAttention {
            causal: true,
            rope: true,
            ..MultiheadAttention::new(embed_dim, num_heads)
        }
    }

    fn proj(&self) -> Linear {
        Linear {
            in_dim: self.embed_dim,
            out_dim: self.embed_dim,
            bias: self.bias,
        }
    }

    fn head_dim(&self) -> Result<usize, AutogradError> {
        if self.num_heads == 0 || !self.embed_dim.is_multiple_of(self.num_heads) {
            return Err(bad("mha: embed_dim must divide evenly by num_heads"));
        }
        Ok(self.embed_dim / self.num_heads)
    }

    /// Cross-attention: queries from `q_src` `[Tq, E]`, keys/values from
    /// `kv_src` `[Tk, E]` → `[Tq, E]`. Self-attention is
    /// `attend(tape, params, x, x)` (what [`Layer::apply`] does).
    pub fn attend<T: DiffScalar + ToF64>(
        &self,
        tape: &Tape<T>,
        params: &MhaParamsVars<T>,
        q_src: &Var<T>,
        kv_src: &Var<T>,
    ) -> Result<Var<T>, AutogradError> {
        let hd = self.head_dim()?;
        let (qs, ks) = (
            q_src.value().shape().to_vec(),
            kv_src.value().shape().to_vec(),
        );
        if qs.len() != 2 || ks.len() != 2 || qs[1] != self.embed_dim || ks[1] != self.embed_dim {
            return Err(bad("mha: sources must be 2-D [T, embed_dim]"));
        }
        let (tq, tk) = (qs[0], ks[0]);
        let proj = self.proj();

        let q = proj.apply(tape, &params.wq, q_src)?;
        let k = proj.apply(tape, &params.wk, kv_src)?;
        let v = proj.apply(tape, &params.wv, kv_src)?;

        // Rotary caches cover each source's positions at head width.
        let gpu = q_src.value().gpu().clone();
        let cache_q = if self.rope {
            Some(RopeCache::<T>::new(&gpu, tq.max(1), hd, self.rope_base)?)
        } else {
            None
        };
        let cache_k = if self.rope {
            Some(RopeCache::<T>::new(&gpu, tk.max(1), hd, self.rope_base)?)
        } else {
            None
        };

        // Column block `[h·hd, (h+1)·hd)` as a differentiable view.
        let col_slice = |m: &Var<T>, start: usize| -> Result<Var<T>, AutogradError> {
            m.transpose(0, 1)?.narrow(start, hd)?.transpose(0, 1)
        };

        let opts = Sdpa {
            scale: None,
            causal: self.causal,
            kv_len: None,
        };
        let mut heads = Vec::with_capacity(self.num_heads); // each [hd, Tq]
        for h in 0..self.num_heads {
            let mut qh = col_slice(&q, h * hd)?;
            let mut kh = col_slice(&k, h * hd)?;
            let vh = col_slice(&v, h * hd)?;
            if let (Some(cq), Some(ck)) = (&cache_q, &cache_k) {
                qh = rope_var(tape, &qh, cq)?;
                kh = rope_var(tape, &kh, ck)?;
            }
            let ctx = sdpa_var(tape, &qh, &kh, &vh, opts)?; // [Tq, hd]
            heads.push(ctx.transpose(0, 1)?); // [hd, Tq]
        }
        let head_refs: Vec<&Var<T>> = heads.iter().collect();
        let ctx = Var::concat_axis0(&head_refs)?.transpose(0, 1)?; // [Tq, E]

        proj.apply(tape, &params.wo, &ctx)
    }
}

/// The four projection trees — a derived [`ParamTree`].
#[derive(ParamTree)]
#[param_tree(crate = crate)]
pub struct MhaParams<T: DiffScalar> {
    pub wq: LinearParams<T>,
    pub wk: LinearParams<T>,
    pub wv: LinearParams<T>,
    pub wo: LinearParams<T>,
}

impl<T: DiffScalar + ToF64> Layer<T> for MultiheadAttention {
    type Params = MhaParams<T>;

    fn in_dim(&self) -> Option<usize> {
        Some(self.embed_dim)
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.embed_dim
    }

    fn init(&self, gpu: &Gpu, key: Key) -> Result<Self::Params, AutogradError> {
        self.head_dim()?; // the divisibility contract fails at build time
        let proj = self.proj();
        let (kq, rest) = key.split();
        let (kk, rest) = rest.split();
        let (kv, ko) = rest.split();
        Ok(MhaParams {
            wq: proj.init(gpu, kq)?,
            wk: proj.init(gpu, kk)?,
            wv: proj.init(gpu, kv)?,
            wo: proj.init(gpu, ko)?,
        })
    }

    fn apply(
        &self,
        tape: &Tape<T>,
        params: &MhaParamsVars<T>,
        x: &Var<T>,
    ) -> Result<Var<T>, AutogradError> {
        self.attend(tape, params, x, x)
    }
}

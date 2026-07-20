//! Embedding — the token-table module (the chain head).
//!
//! Rides the existing autograd op: [`Var::embedding`] gathers rows of the
//! `[V, E]` table by `ids [B]` and its VJP scatter-adds each cotangent row
//! back to its source row (repeated ids accumulate — the sparse embedding
//! update). No new kernel, no new proof obligation: the module form is
//! configuration + init around a proven-in-tests op.
//!
//! **Deliberately NOT a [`Layer`](crate::layer::Layer)**: the shipped
//! Layer trait is `Var → Var`, and an embedding's input is `ids:
//! Array<u32>` — it heads the chain rather than sitting inside it (the
//! same shape as `MultiheadAttention`'s inherent `attend`). Compose
//! manually: `let x = emb.apply(&table_var, &ids)?;` then feed `x` into
//! the tuple stack.
//!
//! `ids` must be `< vocab`; an out-of-range id is a contract violation
//! (the gather kernel indexes the table unchecked).

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar, Var};
use quanta_core::Gpu;

use crate::layer::Key;

fn bad(msg: &'static str) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        msg,
    )))
}

/// Token embedding: a `[vocab, dim]` table, looked up by `u32` ids.
/// Params = the table itself (an [`Array`] — the `ParamTree` leaf), so it
/// binds/flattens/optimizes like every other tree.
pub struct Embedding {
    pub vocab: usize,
    pub dim: usize,
}

impl Embedding {
    /// Output width — for wiring the manual composition contract.
    pub fn out_dim(&self) -> usize {
        self.dim
    }

    /// Allocate + initialize the table: uniform on `(−√3, √3)`, which has
    /// unit standard deviation — the capability torch's `N(0,1)` default
    /// serves, from the key's uniform primitive (PARITY is capabilities,
    /// not API).
    pub fn init<T: DiffScalar + ToF64>(
        &self,
        gpu: &Gpu,
        key: Key,
    ) -> Result<Array<T>, AutogradError> {
        if self.vocab == 0 || self.dim == 0 {
            return Err(bad("Embedding: vocab and dim must be nonzero"));
        }
        let bound = 3.0f32.sqrt();
        let host: Vec<T> = key
            .uniform(self.vocab * self.dim, -bound, bound)
            .iter()
            .map(|&v| T::from_f64(v as f64))
            .collect();
        Array::from_slice(gpu, &host, &[self.vocab, self.dim]).map_err(AutogradError::from)
    }

    /// Look up `ids [B]` in the bound table `[V, E]` → `[B, E]`. The
    /// gradient scatter-adds into the table rows (repeated ids sum).
    pub fn apply<T: DiffScalar>(
        &self,
        table: &Var<T>,
        ids: &Array<u32>,
    ) -> Result<Var<T>, AutogradError> {
        if table.value().shape() != [self.vocab, self.dim] {
            return Err(bad("Embedding: table shape must be [vocab, dim]"));
        }
        table.embedding(ids)
    }
}

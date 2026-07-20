//! State save/load — the `state_dict` capability, on named trees.
//!
//! [`save_state`] serializes a [`ParamTree`] to bytes using the tree's
//! hierarchical names ([`ParamTree::named_flatten`]); [`load_state`]
//! rebuilds a tree of the witness's shape, **matching entries by NAME,
//! not order** — a reordered byte stream loads identically, and a
//! missing, extra, or wrong-shape leaf is a loud error naming the path.
//! (That name-keying is the whole point over `flatten`/`unflatten`,
//! which are positional.)
//!
//! The format is self-contained and dependency-free (the no-wrapper-
//! crates policy): magic `QNNS`, a version word, then per leaf its
//! name, an element-type tag, the shape, and the data. Elements travel
//! as **f64 little-endian** through the existing `ToF64`/`from_f64`
//! lane — exact for both `f32` and `f64` trees (`f32 → f64` is
//! lossless), no new trait obligations on the scalar. A compact
//! native-width encoding is a future increment; file-format interop
//! (safetensors / npy) is step 084.8's scope and will consume this
//! same traversal.
//!
//! Optimizer state trees are `ParamTree`s of the same shape as their
//! params, so checkpointing an optimizer is the same two calls.

use quanta_array::{Array, ArrayError, ToF64};
use quanta_autograd::{AutogradError, DiffScalar};
use quanta_core::Gpu;
use std::collections::HashMap;

use crate::layer::ParamTree;

const MAGIC: &[u8; 4] = b"QNNS";
const VERSION: u32 = 1;

/// Element-type tags. Keyed by the scalar's byte width — the two tape
/// scalars are `f32`/`f64`, and the width is the property the decoder
/// checks (values travel as f64 either way).
const TAG_F32: u8 = 0;
const TAG_F64: u8 = 1;

fn bad(msg: String) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        &msg,
    )))
}

fn scalar_tag<T>() -> Result<u8, AutogradError> {
    match std::mem::size_of::<T>() {
        4 => Ok(TAG_F32),
        8 => Ok(TAG_F64),
        w => Err(bad(format!(
            "state: unsupported scalar width {w} (f32/f64 trees only)"
        ))),
    }
}

/// Serialize a tree to bytes: every leaf under its hierarchical name.
pub fn save_state<T: DiffScalar + ToF64, P: ParamTree<T>>(
    tree: &P,
) -> Result<Vec<u8>, AutogradError> {
    let tag = scalar_tag::<T>()?;
    let leaves = tree.named_flatten();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(leaves.len() as u32).to_le_bytes());
    for (name, arr) in &leaves {
        let bytes = name.as_bytes();
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(bytes);
        out.push(tag);
        let shape = arr.shape();
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
        let host = arr.to_vec().map_err(AutogradError::from)?;
        for v in host {
            out.extend_from_slice(&v.to_f64().to_le_bytes());
        }
    }
    Ok(out)
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], AutogradError> {
        if self.pos + n > self.b.len() {
            return Err(bad("state: truncated byte stream".to_string()));
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, AutogradError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, AutogradError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Rebuild a tree of `witness`'s shape from `bytes`, matching leaves by
/// name. Loud on: version/type mismatch, a leaf the witness has that the
/// bytes lack, a leaf the bytes have that the witness lacks, and any
/// per-leaf shape mismatch — each error names the path.
pub fn load_state<T: DiffScalar + ToF64, P: ParamTree<T>>(
    witness: &P,
    gpu: &Gpu,
    bytes: &[u8],
) -> Result<P, AutogradError> {
    let tag = scalar_tag::<T>()?;
    let mut r = Reader { b: bytes, pos: 0 };
    if r.take(4)? != MAGIC {
        return Err(bad("state: not a QNNS byte stream".to_string()));
    }
    let version = r.u32()?;
    if version != VERSION {
        return Err(bad(format!(
            "state: version {version} (this build reads {VERSION})"
        )));
    }
    let count = r.u32()? as usize;

    let mut entries: HashMap<String, (Vec<usize>, Vec<f64>)> = HashMap::with_capacity(count);
    for _ in 0..count {
        let name_len = r.u32()? as usize;
        let name = std::str::from_utf8(r.take(name_len)?)
            .map_err(|_| bad("state: non-utf8 leaf name".to_string()))?
            .to_string();
        let etag = r.take(1)?[0];
        if etag != tag {
            return Err(bad(format!(
                "state: leaf `{name}` has element tag {etag}, tree wants {tag}"
            )));
        }
        let rank = r.u32()? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(r.u64()? as usize);
        }
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            data.push(f64::from_le_bytes(r.take(8)?.try_into().unwrap()));
        }
        if entries.insert(name.clone(), (shape, data)).is_some() {
            return Err(bad(format!("state: duplicate leaf `{name}`")));
        }
    }

    // Walk the witness's named order, pulling each leaf by name; then
    // anything left over is an extra the tree has no home for.
    let named = witness.named_flatten();
    let mut leaves: Vec<Array<T>> = Vec::with_capacity(named.len());
    for (name, shape_witness) in &named {
        let Some((shape, data)) = entries.remove(name) else {
            return Err(bad(format!("state: missing leaf `{name}`")));
        };
        if shape != shape_witness.shape() {
            return Err(bad(format!(
                "state: leaf `{name}` has shape {:?}, tree wants {:?}",
                shape,
                shape_witness.shape()
            )));
        }
        let host: Vec<T> = data.iter().map(|&v| T::from_f64(v)).collect();
        leaves.push(Array::from_slice(gpu, &host, &shape).map_err(AutogradError::from)?);
    }
    if let Some(name) = entries.keys().next() {
        return Err(bad(format!(
            "state: extra leaf `{name}` ({} extra total) — not in this tree",
            entries.len()
        )));
    }

    witness.unflatten(&mut leaves.into_iter())
}

//! Quantized checkpoints — the quanta quantized-safetensors convention,
//! the named/tree loaders over it, and [`QuantizedLinear`].
//!
//! Weight-only, symmetric integer quantization with f32 activations:
//! weights travel as int8 or packed-int4 codes plus f32 tile scales; at
//! inference the codes dequantize (`w = scale · q`, exact symmetric
//! algebra) and the model otherwise runs the existing f32 stack. The
//! array-level primitive is [`quanta_array::quant::QuantizedMatrix`];
//! this module is the checkpoint + layer face (the same split that put
//! safetensors here and npy in quanta-array).
//!
//! # The format (normative)
//!
//! A quantized checkpoint IS a valid safetensors file — 8 bytes
//! little-endian header length, JSON header, data section; any
//! third-party inspector can open it. One convention on top:
//!
//! * A quantized leaf `x` is stored as TWO tensors plus one metadata
//!   entry — and NO tensor named `x` (ambiguity is structurally
//!   impossible):
//!   * `x.q` — the codes. int8: dtype `I8`, shape `[R, C]`, one byte
//!     per code (two's complement), row-major. int4: dtype `U32`,
//!     shape `[R, ⌈C/8⌉]` — 8 codes per little-endian u32 word, LOW
//!     nibble first, every row starting a fresh word, the final word
//!     of each row zero-padded.
//!   * `x.qs` — the scales: dtype `F32`, shape `[⌈R/gr⌉, ⌈C/gc⌉]`. The
//!     scale of element `(r, c)` is `scales[(r/gr)·⌈C/gc⌉ + (c/gc)]`.
//!   * `__metadata__["quant:x"] = "<v>;gr=<gr>;gc=<gc>;rows=<R>;cols=<C>"`
//!     with `<v>` ∈ {`q8s`, `q4s`} — the self-describing scheme.
//!     `rows`/`cols` recover the int4 logical width and cross-check the
//!     int8 shapes.
//! * `__metadata__["quanta.quant"] = "1"` — the format version, present
//!   whenever the file holds a quantized leaf. An unknown version is a
//!   loud error naming both versions.
//! * Plain leaves keep their names and dtypes (`F32` — the existing
//!   writer; `F16`/`BF16` upconvert on load). Mixed files are the NORM:
//!   norms, biases, and embeddings ride beside quantized matrices.
//! * **Reserved, additive** (loud error today): an `x.qz` zero-points
//!   tensor plus `;mode=affine` in the scheme string — the affine
//!   extension will change no v1 symmetric file or reader.
//!
//! Loader validation is TOTAL before anything uploads: every §-rule
//! above is checked for every leaf first (orphan `quant:x`, missing
//! `x.q`/`x.qs`, name collisions, shape/grid/metadata disagreement,
//! non-finite scales), each a loud per-leaf error. Save is
//! deterministic (entries in given order, metadata keys sorted).
//!
//! # The two load modes
//!
//! * **Mode A — dequantize on load** ([`load_named_f32`], tree form
//!   [`load`]): every quantized leaf is dequantized on the HOST before
//!   upload; the model is an ordinary f32 model afterwards. Universal —
//!   every backend, never touches narrow storage. The tree form matches
//!   by NAME with the `load_state` missing/extra/shape contract, so an
//!   existing f32 model definition loads a quantized checkpoint with
//!   zero code changes.
//! * **Mode B — resident-quantized** ([`load_named`] +
//!   [`QuantizedLinear`]): codes live on-device at 1 byte (int8) or a
//!   half byte (int4) per element; each forward dequantizes. int8
//!   residency is capability-gated on [`Gpu::supports_narrow_int`]
//!   (WebGPU says no — Mode A is its complete answer); packed int4 is
//!   core u32 storage and rides every backend.
//!
//! Everything after quantization is exact and bit-pinned: dequantize is
//! `scale · f32(q)`, so `QuantizedLinear` output equals `Linear` fed
//! the dequantized weight BITWISE, and the quantized forward is
//! bit-reproducible cross-backend. Quantization itself carries the only
//! tolerance: `|s·round_te(x/s) − x| ≤ s/2` per element.
//!
//! External quantized formats (GPTQ / AWQ / GGUF / bitsandbytes) are
//! claim boundaries, not gaps: this reader reads OUR convention; a
//! foreign importer is a future reader constructing the same
//! [`QuantizedMatrix`].

use quanta_array::autograd::{AutogradError, Tape, Var};
use quanta_array::quant::{HostCodes, dequantize_host};
use quanta_array::{Array, ArrayError};
use quanta_core::Gpu;
use std::collections::{HashMap, HashSet};

use crate::layer::{Key, Layer, ParamTree};
use crate::safetensors::{
    LoadedSafetensors, RawEntry, RawFile, RawSaveEntry, entry_to_f32, parse_raw, save_raw,
};

pub use quanta_array::quant::{Granularity, QuantCodes, QuantDtype, QuantError, QuantizedMatrix};

/// The quantized-safetensors format version this build reads and writes.
pub const FORMAT_VERSION: &str = "1";

const VERSION_KEY: &str = "quanta.quant";
const META_PREFIX: &str = "quant:";

fn quant_err(e: QuantError) -> AutogradError {
    AutogradError::from(ArrayError::Quant(e))
}

fn format_err(leaf: &str, what: String) -> AutogradError {
    quant_err(QuantError::Format {
        leaf: leaf.to_string(),
        what,
    })
}

fn bad(msg: String) -> AutogradError {
    AutogradError::from(ArrayError::Gpu(quanta_core::QuantaError::invalid_param(
        &msg,
    )))
}

fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// The canonical granularity spelling for a stored tile grid, `None`
/// when no v1 writer produces that grid. Precedence where spellings
/// coincide: PerTensor, PerChannel, Group — reading back a canonical
/// spelling never changes the scale of any element.
fn granularity_from_grid(rows: usize, cols: usize, gr: usize, gc: usize) -> Option<Granularity> {
    if gr == 0 || gc == 0 || gr > rows || gc > cols {
        return None;
    }
    if gr == rows && gc == cols {
        return Some(Granularity::PerTensor);
    }
    if gr == 1 && gc == cols {
        return Some(Granularity::PerChannel { axis: 0 });
    }
    if gr == rows && gc == 1 {
        return Some(Granularity::PerChannel { axis: 1 });
    }
    if gc == 1 {
        return Some(Granularity::Group {
            axis: 0,
            size: gr as u32,
        });
    }
    if gr == 1 {
        return Some(Granularity::Group {
            axis: 1,
            size: gc as u32,
        });
    }
    None
}

// ── The mixed-checkpoint vocabulary ─────────────────────────────────────

/// One named checkpoint leaf: plain f32, or a quantized matrix. Mixed
/// trees are the norm — norms and biases stay f32 beside quantized
/// weights.
pub enum QuantLeaf {
    F32(Array<f32>),
    Quantized(QuantizedMatrix),
}

/// Quantize named leaves under a caller-owned policy: `Some((dtype,
/// granularity))` quantizes the leaf (rank-2 only — loud otherwise),
/// `None` keeps it f32. No magic name-matching: the caller decides
/// which leaves carry the weight mass.
pub fn quantize_named<F>(
    leaves: &[(String, Array<f32>)],
    policy: F,
) -> Result<Vec<(String, QuantLeaf)>, AutogradError>
where
    F: Fn(&str, &Array<f32>) -> Option<(QuantDtype, Granularity)>,
{
    leaves
        .iter()
        .map(|(name, arr)| {
            Ok((
                name.clone(),
                match policy(name, arr) {
                    Some((dtype, gran)) => QuantLeaf::Quantized(
                        QuantizedMatrix::quantize(arr, dtype, gran).map_err(AutogradError::from)?,
                    ),
                    None => QuantLeaf::F32(arr.shallow_clone()),
                },
            ))
        })
        .collect()
}

// ── The metadata grammar ────────────────────────────────────────────────

fn scheme_tag(dtype: QuantDtype) -> &'static str {
    match dtype {
        QuantDtype::Int8 => "q8s",
        QuantDtype::Int4 => "q4s",
    }
}

fn write_quant_meta(m: &QuantizedMatrix) -> String {
    let [rows, cols] = m.shape();
    let (gr, gc) = m.tile();
    format!(
        "{};gr={gr};gc={gc};rows={rows};cols={cols}",
        scheme_tag(m.dtype())
    )
}

#[derive(Clone, Copy, Debug)]
struct QuantMeta {
    dtype: QuantDtype,
    gr: usize,
    gc: usize,
    rows: usize,
    cols: usize,
}

/// Parse one `quant:x` value. Loud on: unknown scheme tag, malformed or
/// duplicated fields, zero values, the reserved `mode=affine`, and any
/// missing field.
fn parse_quant_meta(leaf: &str, s: &str) -> Result<QuantMeta, AutogradError> {
    let mut parts = s.split(';');
    let tag = parts.next().unwrap_or("");
    let dtype = match tag {
        "q8s" => QuantDtype::Int8,
        "q4s" => QuantDtype::Int4,
        other => {
            return Err(format_err(
                leaf,
                format!("unknown scheme tag `{other}` (this build reads q8s / q4s)"),
            ));
        }
    };
    let mut fields: [Option<usize>; 4] = [None; 4];
    const NAMES: [&str; 4] = ["gr", "gc", "rows", "cols"];
    for part in parts {
        let Some((k, v)) = part.split_once('=') else {
            return Err(format_err(
                leaf,
                format!("malformed metadata field `{part}` (want key=value)"),
            ));
        };
        if k == "mode" {
            return Err(format_err(
                leaf,
                format!(
                    "mode={v} is the reserved affine extension — this build reads v1 \
                     symmetric files only"
                ),
            ));
        }
        let Some(idx) = NAMES.iter().position(|n| *n == k) else {
            return Err(format_err(leaf, format!("unknown metadata field `{k}`")));
        };
        if fields[idx].is_some() {
            return Err(format_err(leaf, format!("duplicate metadata field `{k}`")));
        }
        let n: usize = v
            .parse()
            .map_err(|_| format_err(leaf, format!("field {k}={v} is not a positive integer")))?;
        if n == 0 {
            return Err(format_err(leaf, format!("field {k}=0 (must be positive)")));
        }
        fields[idx] = Some(n);
    }
    for (idx, name) in NAMES.iter().enumerate() {
        if fields[idx].is_none() {
            return Err(format_err(leaf, format!("missing metadata field `{name}`")));
        }
    }
    Ok(QuantMeta {
        dtype,
        gr: fields[0].unwrap(),
        gc: fields[1].unwrap(),
        rows: fields[2].unwrap(),
        cols: fields[3].unwrap(),
    })
}

// ── Save ────────────────────────────────────────────────────────────────

/// Serialize named leaves (plain and quantized, in the given order)
/// plus optional user metadata into quantized-safetensors bytes. The
/// version key and per-leaf `quant:x` entries are synthesized here —
/// user metadata keys that collide with the convention are loud errors.
pub fn save_named(
    entries: &[(String, QuantLeaf)],
    metadata: Option<&HashMap<String, String>>,
) -> Result<Vec<u8>, AutogradError> {
    if let Some(meta) = metadata {
        for k in meta.keys() {
            if k == VERSION_KEY || k.starts_with(META_PREFIX) {
                return Err(format_err(
                    k,
                    format!(
                        "user metadata key `{k}` collides with the quantized-safetensors \
                         convention (`{VERSION_KEY}` and `{META_PREFIX}*` are synthesized \
                         by the writer)"
                    ),
                ));
            }
        }
    }

    // Name discipline first, before any readback: leaf names unique,
    // and no plain leaf may occupy a quantized leaf's `x.q` / `x.qs`
    // tensor slots.
    let mut tensor_names: HashSet<String> = HashSet::new();
    let mut leaf_names: HashSet<&str> = HashSet::new();
    for (name, leaf) in entries {
        if !leaf_names.insert(name.as_str()) {
            return Err(format_err(name, "duplicate leaf name".to_string()));
        }
        let slots: Vec<String> = match leaf {
            QuantLeaf::F32(_) => vec![name.clone()],
            QuantLeaf::Quantized(_) => vec![format!("{name}.q"), format!("{name}.qs")],
        };
        for slot in slots {
            if !tensor_names.insert(slot.clone()) {
                return Err(format_err(
                    name,
                    format!("tensor name `{slot}` collides with another leaf's"),
                ));
            }
        }
    }

    let mut meta: HashMap<String, String> = metadata.cloned().unwrap_or_default();
    let mut raw: Vec<RawSaveEntry> = Vec::new();
    let mut any_quantized = false;
    for (name, leaf) in entries {
        match leaf {
            QuantLeaf::F32(arr) => {
                let host = arr
                    .contiguous()
                    .map_err(AutogradError::from)?
                    .to_vec()
                    .map_err(AutogradError::from)?;
                let mut bytes = Vec::with_capacity(host.len() * 4);
                for v in &host {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                raw.push(RawSaveEntry {
                    name: name.clone(),
                    dtype: "F32",
                    shape: arr.shape().to_vec(),
                    bytes,
                });
            }
            QuantLeaf::Quantized(m) => {
                any_quantized = true;
                meta.insert(format!("{META_PREFIX}{name}"), write_quant_meta(m));
                let (dtype, shape, bytes) = match m.codes() {
                    QuantCodes::Int8(a) => {
                        let host = a.to_vec().map_err(AutogradError::from)?;
                        let bytes: Vec<u8> = host.iter().map(|&v| v as u8).collect();
                        ("I8", a.shape().to_vec(), bytes)
                    }
                    QuantCodes::Int4Packed(a) => {
                        let host = a.to_vec().map_err(AutogradError::from)?;
                        let mut bytes = Vec::with_capacity(host.len() * 4);
                        for w in &host {
                            bytes.extend_from_slice(&w.to_le_bytes());
                        }
                        ("U32", a.shape().to_vec(), bytes)
                    }
                };
                raw.push(RawSaveEntry {
                    name: format!("{name}.q"),
                    dtype,
                    shape,
                    bytes,
                });
                let scales = m.scales().to_vec().map_err(AutogradError::from)?;
                let mut sbytes = Vec::with_capacity(scales.len() * 4);
                for s in &scales {
                    sbytes.extend_from_slice(&s.to_le_bytes());
                }
                raw.push(RawSaveEntry {
                    name: format!("{name}.qs"),
                    dtype: "F32",
                    shape: m.scales().shape().to_vec(),
                    bytes: sbytes,
                });
            }
        }
    }
    if any_quantized {
        meta.insert(VERSION_KEY.to_string(), FORMAT_VERSION.to_string());
    }
    let meta_opt = if meta.is_empty() && metadata.is_none() {
        None
    } else {
        Some(&meta)
    };
    Ok(save_raw(&raw, meta_opt))
}

// ── Load: total validation, then the two modes ──────────────────────────

/// One validated leaf, in header order.
enum Item {
    /// Index into the raw entries — an ordinary F32/F16/BF16 tensor.
    Plain(usize),
    /// A validated quantized leaf: its metadata, the canonical
    /// granularity of its grid, the entry index of `x.q`, and the
    /// (finite-checked) decoded scales.
    Quantized {
        name: String,
        meta: QuantMeta,
        gran: Granularity,
        q: usize,
        scales: Vec<f32>,
    },
}

/// Everything a quantized-safetensors byte string holds, validated
/// whole: the raw container, the leaves in header order, and the USER
/// metadata (the `quant:x` entries and the version key are format
/// machinery — stripped here, re-synthesized by [`save_named`]).
struct Validated<'a> {
    file: RawFile<'a>,
    items: Vec<Item>,
    metadata: HashMap<String, String>,
}

fn validate(bytes: &[u8]) -> Result<Validated<'_>, AutogradError> {
    let file = parse_raw(bytes)?;

    // The version gate first: an unknown version must fail before any
    // structural interpretation.
    let mut quant_keys: Vec<&str> = file
        .metadata
        .keys()
        .filter_map(|k| k.strip_prefix(META_PREFIX))
        .collect();
    quant_keys.sort_unstable();
    if let Some(v) = file.metadata.get(VERSION_KEY)
        && v.as_str() != FORMAT_VERSION
    {
        return Err(format_err(
            VERSION_KEY,
            format!("format version {v} (this build reads {FORMAT_VERSION})"),
        ));
    }
    if !quant_keys.is_empty() && !file.metadata.contains_key(VERSION_KEY) {
        return Err(format_err(
            VERSION_KEY,
            format!(
                "missing — a file with quant:* metadata must carry \
                 {VERSION_KEY} = \"{FORMAT_VERSION}\""
            ),
        ));
    }

    // Index the entries by name; the JSON grammar cannot forbid
    // duplicate keys, so the reader must.
    let mut by_name: HashMap<&str, usize> = HashMap::with_capacity(file.entries.len());
    for (i, e) in file.entries.iter().enumerate() {
        if by_name.insert(e.name.as_str(), i).is_some() {
            return Err(format_err(&e.name, "duplicate tensor name".to_string()));
        }
    }

    // Validate every quantized leaf completely before building items.
    enum Claim {
        /// The `x.q` entry — emits the quantized item at its position.
        Codes(usize),
        /// The `x.qs` entry — consumed, emits nothing.
        Scales,
    }
    let mut claimed: HashMap<usize, Claim> = HashMap::new();
    let mut quantized: Vec<(String, QuantMeta, Granularity, usize, Vec<f32>)> = Vec::new();
    for leaf in quant_keys {
        if leaf.is_empty() {
            return Err(format_err(
                META_PREFIX,
                "empty leaf name in quant:* metadata".to_string(),
            ));
        }
        let meta_str = &file.metadata[&format!("{META_PREFIX}{leaf}")];
        let meta = parse_quant_meta(leaf, meta_str)?;
        if by_name.contains_key(leaf) {
            return Err(format_err(
                leaf,
                format!(
                    "both a tensor `{leaf}` and quant:{leaf} metadata are present (a \
                         quantized leaf stores NO tensor under its own name)"
                ),
            ));
        }
        if by_name.contains_key(format!("{leaf}.qz").as_str()) {
            return Err(format_err(
                leaf,
                format!(
                    "`{leaf}.qz` zero-points are the reserved affine extension — this \
                     build reads v1 symmetric files only"
                ),
            ));
        }
        let q_name = format!("{leaf}.q");
        let qs_name = format!("{leaf}.qs");
        let Some(&qi) = by_name.get(q_name.as_str()) else {
            return Err(format_err(
                leaf,
                format!("quant:{leaf} present but tensor `{q_name}` is missing"),
            ));
        };
        let Some(&si) = by_name.get(qs_name.as_str()) else {
            return Err(format_err(
                leaf,
                format!("quant:{leaf} present but tensor `{qs_name}` is missing"),
            ));
        };
        let (rows, cols, gr, gc) = (meta.rows, meta.cols, meta.gr, meta.gc);
        let Some(gran) = granularity_from_grid(rows, cols, gr, gc) else {
            return Err(format_err(
                leaf,
                format!("grid (gr={gr}, gc={gc}) over [{rows}, {cols}] is not a v1 tile grid"),
            ));
        };

        // The codes tensor: dtype and shape must agree with the scheme.
        let qe = &file.entries[qi];
        let (want_dtype, want_shape, want_bytes) = match meta.dtype {
            QuantDtype::Int8 => ("I8", vec![rows, cols], rows * cols),
            QuantDtype::Int4 => (
                "U32",
                vec![rows, ceil_div(cols, 8)],
                rows * ceil_div(cols, 8) * 4,
            ),
        };
        if qe.dtype != want_dtype {
            return Err(format_err(
                leaf,
                format!(
                    "`{q_name}` has dtype {}, scheme {} wants {want_dtype}",
                    qe.dtype,
                    scheme_tag(meta.dtype)
                ),
            ));
        }
        if qe.shape != want_shape {
            return Err(format_err(
                leaf,
                format!(
                    "`{q_name}` has shape {:?}, metadata [rows={rows}, cols={cols}] wants \
                     {want_shape:?}",
                    qe.shape
                ),
            ));
        }
        if qe.end - qe.start != want_bytes {
            return Err(format_err(
                leaf,
                format!(
                    "`{q_name}` holds {} data bytes, shape {want_shape:?} wants {want_bytes}",
                    qe.end - qe.start
                ),
            ));
        }

        // The scales tensor: F32 on the ⌈·⌉ grid, every value finite.
        let se = &file.entries[si];
        let want_scales = vec![ceil_div(rows, gr), ceil_div(cols, gc)];
        if se.dtype != "F32" {
            return Err(format_err(
                leaf,
                format!("`{qs_name}` has dtype {}, scales are F32", se.dtype),
            ));
        }
        if se.shape != want_scales {
            return Err(format_err(
                leaf,
                format!(
                    "`{qs_name}` has shape {:?}, grid (gr={gr}, gc={gc}) wants {want_scales:?}",
                    se.shape
                ),
            ));
        }
        let scales = entry_to_f32(se, file.data)?;
        for (tile, s) in scales.iter().enumerate() {
            // Zero is LEGAL (an all-zero tile stores s = 0 and
            // dequantizes exactly); non-finite is a corrupt file.
            if !s.is_finite() {
                return Err(quant_err(QuantError::Scale {
                    leaf: leaf.to_string(),
                    tile,
                    value: *s,
                }));
            }
        }

        claimed.insert(qi, Claim::Codes(quantized.len()));
        claimed.insert(si, Claim::Scales);
        quantized.push((leaf.to_string(), meta, gran, qi, scales));
    }

    // Header-order items; a quantized leaf sits where its `x.q` does.
    // Unclaimed I8/U32 tensors are orphans — loud, not silently plain.
    let mut items = Vec::new();
    for (i, e) in file.entries.iter().enumerate() {
        match claimed.get(&i) {
            Some(Claim::Scales) => {}
            Some(Claim::Codes(qidx)) => {
                let (name, meta, gran, q, scales) = &quantized[*qidx];
                items.push(Item::Quantized {
                    name: name.clone(),
                    meta: *meta,
                    gran: *gran,
                    q: *q,
                    scales: scales.clone(),
                });
            }
            None => {
                if e.dtype == "I8" || e.dtype == "U32" {
                    return Err(format_err(
                        &e.name,
                        format!(
                            "dtype {} tensor carries no quant:<leaf> metadata — not a v1 \
                             quantized leaf, and not a plain F32/F16/BF16 tensor",
                            e.dtype
                        ),
                    ));
                }
                items.push(Item::Plain(i));
            }
        }
    }

    let metadata: HashMap<String, String> = file
        .metadata
        .iter()
        .filter(|(k, _)| *k != VERSION_KEY && !k.starts_with(META_PREFIX))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    Ok(Validated {
        file,
        items,
        metadata,
    })
}

fn decode_codes_i8(e: &RawEntry, data: &[u8]) -> Vec<i8> {
    data[e.start..e.end].iter().map(|&b| b as i8).collect()
}

fn decode_codes_u32(e: &RawEntry, data: &[u8]) -> Vec<u32> {
    data[e.start..e.end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// The Mode B gate for int8 residency, split out so the refusal path is
/// unit-testable on backends that support narrow storage.
fn gate_resident_int8(supports: bool, backend: &str, leaf: &str) -> Result<(), AutogradError> {
    if supports {
        Ok(())
    } else {
        Err(quant_err(QuantError::NotSupported {
            leaf: leaf.to_string(),
            backend: backend.to_string(),
        }))
    }
}

/// Everything a quantized checkpoint holds under Mode B: the leaves
/// (quantized ones device-resident as [`QuantizedMatrix`]) and the user
/// metadata.
pub struct LoadedQuant {
    pub leaves: Vec<(String, QuantLeaf)>,
    pub metadata: HashMap<String, String>,
}

/// **Mode B** load: quantized leaves arrive as device-resident
/// [`QuantizedMatrix`] (codes at storage width), plain leaves as
/// `Array<f32>`. int8 leaves on a backend without
/// [`Gpu::supports_narrow_int`] are a loud error naming the leaf and
/// pointing at [`load_named_f32`]; packed int4 rides every backend.
pub fn load_named(gpu: &Gpu, bytes: &[u8]) -> Result<LoadedQuant, AutogradError> {
    let v = validate(bytes)?;
    let mut leaves = Vec::with_capacity(v.items.len());
    for item in &v.items {
        match item {
            Item::Plain(i) => {
                let e = &v.file.entries[*i];
                let host = entry_to_f32(e, v.file.data)?;
                let arr = Array::from_slice(gpu, &host, &e.shape).map_err(AutogradError::from)?;
                leaves.push((e.name.clone(), QuantLeaf::F32(arr)));
            }
            Item::Quantized {
                name,
                meta,
                gran,
                q,
                scales,
            } => {
                let qe = &v.file.entries[*q];
                let codes = match meta.dtype {
                    QuantDtype::Int8 => {
                        gate_resident_int8(gpu.supports_narrow_int(), gpu.name(), name)?;
                        let host = decode_codes_i8(qe, v.file.data);
                        QuantCodes::Int8(
                            Array::from_slice(gpu, &host, &qe.shape)
                                .map_err(AutogradError::from)?,
                        )
                    }
                    QuantDtype::Int4 => {
                        let host = decode_codes_u32(qe, v.file.data);
                        QuantCodes::Int4Packed(
                            Array::from_slice(gpu, &host, &qe.shape)
                                .map_err(AutogradError::from)?,
                        )
                    }
                };
                let scales_arr = Array::from_slice(
                    gpu,
                    scales,
                    &[ceil_div(meta.rows, meta.gr), ceil_div(meta.cols, meta.gc)],
                )
                .map_err(AutogradError::from)?;
                let m =
                    QuantizedMatrix::from_parts(codes, [meta.rows, meta.cols], scales_arr, *gran)
                        .map_err(AutogradError::from)?;
                leaves.push((name.clone(), QuantLeaf::Quantized(m)));
            }
        }
    }
    Ok(LoadedQuant {
        leaves,
        metadata: v.metadata,
    })
}

/// **Mode A** load: every quantized leaf is dequantized on the HOST
/// (exact `scale · q` — bit-identical to the device kernel) before
/// upload, under its own name `x` at its logical `[rows, cols]` shape.
/// Universal: never touches narrow storage, WebGPU included.
pub fn load_named_f32(gpu: &Gpu, bytes: &[u8]) -> Result<LoadedSafetensors, AutogradError> {
    let v = validate(bytes)?;
    let mut tensors = Vec::with_capacity(v.items.len());
    for item in &v.items {
        match item {
            Item::Plain(i) => {
                let e = &v.file.entries[*i];
                let host = entry_to_f32(e, v.file.data)?;
                let arr = Array::from_slice(gpu, &host, &e.shape).map_err(AutogradError::from)?;
                tensors.push((e.name.clone(), arr));
            }
            Item::Quantized {
                name,
                meta,
                gran,
                q,
                scales,
            } => {
                let qe = &v.file.entries[*q];
                let codes = match meta.dtype {
                    QuantDtype::Int8 => HostCodes::Int8(decode_codes_i8(qe, v.file.data)),
                    QuantDtype::Int4 => HostCodes::Int4Packed(decode_codes_u32(qe, v.file.data)),
                };
                // The validation pass above pinned every length and
                // the grid, so this cannot fail for a file that passed
                // it — propagated rather than asserted (the seam is
                // fallible; the loader's checks make the Err arm
                // unreachable).
                let host = dequantize_host(&codes, scales, meta.rows, meta.cols, *gran)
                    .map_err(AutogradError::from)?;
                let arr = Array::from_slice(gpu, &host, &[meta.rows, meta.cols])
                    .map_err(AutogradError::from)?;
                tensors.push((name.clone(), arr));
            }
        }
    }
    Ok(LoadedSafetensors {
        tensors,
        metadata: v.metadata,
    })
}

/// The Mode A tree form: rebuild a tree of `witness`'s shape from
/// quantized-safetensors bytes, matching by NAME (the
/// [`crate::state::load_state`] contract) — an existing f32 model
/// definition loads a quantized checkpoint with zero code changes. A
/// missing, extra, or wrong-shape tensor is a loud error naming the
/// path.
pub fn load<P: ParamTree<f32>>(gpu: &Gpu, witness: &P, bytes: &[u8]) -> Result<P, AutogradError> {
    let loaded = load_named_f32(gpu, bytes)?;
    let mut by_name: HashMap<String, Array<f32>> = loaded.tensors.into_iter().collect();
    let named = witness.named_flatten();
    let mut leaves = Vec::with_capacity(named.len());
    for (name, want) in &named {
        let got = by_name.remove(name).ok_or_else(|| {
            bad(format!(
                "quantized-safetensors load: missing tensor {name:?}"
            ))
        })?;
        if got.shape() != want.shape() {
            return Err(bad(format!(
                "quantized-safetensors load: {name:?}: shape {:?} in file, tree wants {:?}",
                got.shape(),
                want.shape()
            )));
        }
        leaves.push(got);
    }
    if let Some(extra) = by_name.keys().next() {
        return Err(bad(format!(
            "quantized-safetensors load: tensor {extra:?} has no home in the tree"
        )));
    }
    witness.unflatten(&mut leaves.into_iter())
}

// ── QuantizedLinear ─────────────────────────────────────────────────────

enum QWeight {
    /// Mode A: the weight was dequantized once and lives as f32.
    Held(Array<f32>),
    /// Mode B: the codes stay resident; each forward dequantizes.
    Resident(QuantizedMatrix),
}

/// A frozen quantized dense layer `[N, in] → [N, out]`, weight stored
/// `[in, out]` like [`crate::layer::Linear`]. The weights ARE
/// configuration (a layer is configuration — decision D1), so `Params
/// = ()` like the activations: it stacks in tuples for free, its `init`
/// returns `()`, and no optimizer ever sees the codes.
///
/// Output equals `Linear` fed the dequantized weight BITWISE — the same
/// tape ops over the same values. Gradients flow through `x` (harmless;
/// none reach the codes — quantization-aware training is a different
/// product).
pub struct QuantizedLinear {
    w: QWeight,
    b: Option<Array<f32>>,
    in_dim: usize,
    out_dim: usize,
}

impl QuantizedLinear {
    fn check_bias(b: &Option<Array<f32>>, out_dim: usize) -> Result<(), AutogradError> {
        if let Some(b) = b
            && b.shape() != [out_dim]
        {
            return Err(bad(format!(
                "QuantizedLinear: bias shape {:?}, weight [in, {out_dim}] wants [{out_dim}]",
                b.shape()
            )));
        }
        Ok(())
    }

    /// **Mode B** — hold the codes resident (1 byte / half a byte per
    /// element between steps); every forward dequantizes into a scratch
    /// f32 the tape holds for the step.
    pub fn new(w: QuantizedMatrix, b: Option<Array<f32>>) -> Result<Self, AutogradError> {
        let [rows, cols] = w.shape();
        Self::check_bias(&b, cols)?;
        Ok(QuantizedLinear {
            w: QWeight::Resident(w),
            b,
            in_dim: rows,
            out_dim: cols,
        })
    }

    /// **Mode A** — dequantize ONCE here and hold the f32 weight; each
    /// forward then costs exactly what [`crate::layer::Linear`] costs.
    pub fn dequantized(w: &QuantizedMatrix, b: Option<Array<f32>>) -> Result<Self, AutogradError> {
        let [rows, cols] = w.shape();
        Self::check_bias(&b, cols)?;
        let held = w.dequantize().map_err(AutogradError::from)?;
        Ok(QuantizedLinear {
            w: QWeight::Held(held),
            b,
            in_dim: rows,
            out_dim: cols,
        })
    }
}

impl Layer<f32> for QuantizedLinear {
    type Params = ();

    fn in_dim(&self) -> Option<usize> {
        Some(self.in_dim)
    }
    fn out_dim(&self, _in: usize) -> usize {
        self.out_dim
    }
    fn init(&self, _gpu: &Gpu, _key: Key) -> Result<(), AutogradError> {
        Ok(())
    }

    // ── INTEGRATOR SEAM ─────────────────────────────────────────────
    // This body is the fused dequant-GEMM swap point. Today it spells
    // dequantize → tape.var → matmul (+ broadcast bias) — every step an
    // existing proven citizen, and the reason the ≡-Linear claim is
    // bitwise. The eventual target is quanta-blas's `mixed_quant` entry
    // (`gemm_quant` / `gemm_quant4`) once its per-channel / grouped
    // scale forms land: `QuantizedMatrix` is the operand that kernel
    // consumes, and swapping the spelling here buys the peak-memory win
    // (never materialising the f32 weight) with zero API, format, or
    // semantics change. Nothing is wired to blas today — the seam is
    // this one method body.
    fn apply(&self, tape: &Tape<f32>, _p: &(), x: &Var<f32>) -> Result<Var<f32>, AutogradError> {
        let w = match &self.w {
            QWeight::Held(w) => w.shallow_clone(),
            QWeight::Resident(q) => q.dequantize().map_err(AutogradError::from)?,
        };
        let wv = tape.var(w);
        let y = x.matmul(&wv)?;
        match &self.b {
            Some(b) => {
                let bv = tape.var(b.shallow_clone());
                let b2 = bv.reshape(&[1, self.out_dim])?;
                y.add(&b2)
            }
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Mode B refusal path cannot run on any backend these suites
    /// compile (software, Metal, and Vulkan all report
    /// `supports_narrow_int() == true`; WebGPU — the backend that says
    /// no — has no lane here). The gate takes the capability as data,
    /// so the refusal is pinned without a narrow-less device.
    #[test]
    fn resident_int8_gate_refusal_names_leaf_and_workaround() {
        let err = gate_resident_int8(false, "webgpu", "blk.0.ffn.w").unwrap_err();
        // The message contract lives on QuantError's Display: the leaf,
        // the caps reason, and the Mode A workaround.
        let AutogradError::Array(ArrayError::Quant(q)) = err else {
            panic!("expected a QuantError");
        };
        let text = format!("{q}");
        assert!(text.contains("blk.0.ffn.w"), "{text}");
        assert!(text.contains("webgpu"), "{text}");
        // The Display lives in quanta-array, which cannot name this
        // crate's APIs — the contract is leaf + backend + the Mode A
        // workaround stated in prose.
        assert!(text.contains("does not support"), "{text}");
        assert!(text.contains("dequantize-on-load"), "{text}");
        assert!(gate_resident_int8(true, "cpu", "w").is_ok());
    }

    #[test]
    fn quant_meta_grammar_round_trips_and_rejects() {
        let m = parse_quant_meta("w", "q8s;gr=1;gc=4;rows=2;cols=4").unwrap();
        assert!(matches!(m.dtype, QuantDtype::Int8));
        assert_eq!((m.gr, m.gc, m.rows, m.cols), (1, 4, 2, 4));
        // Field order is free; the writer's order is just canonical.
        assert!(parse_quant_meta("w", "q4s;cols=4;rows=2;gc=4;gr=1").is_ok());

        for (s, needle) in [
            ("q5s;gr=1;gc=1;rows=1;cols=1", "q5s"),
            ("q8s;gr=1;gc=1;rows=1", "cols"),
            ("q8s;gr=1;gc=1;rows=1;cols=1;mode=affine", "affine"),
            ("q8s;gr=0;gc=1;rows=1;cols=1", "gr=0"),
            ("q8s;gr=1;gr=2;gc=1;rows=1;cols=1", "duplicate"),
            ("q8s;gr=1;gc=1;rows=1;cols=1;zp=3", "zp"),
            ("q8s;gr;gc=1;rows=1;cols=1", "key=value"),
        ] {
            let text = format!("{:?}", parse_quant_meta("w", s).unwrap_err());
            assert!(text.contains(needle), "{s} → {text}");
        }
    }
}

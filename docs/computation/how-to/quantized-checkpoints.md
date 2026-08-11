# Run a quantized checkpoint

Weight-only quantization stores a trained model's weight matrices as
int8 or packed-int4 codes plus f32 scales — a checkpoint ÷4 or ÷8 on
disk — and dequantizes them back (`w = scale · q`, exact symmetric
algebra) so the model runs the same proven f32 stack as before:
matmul, fused attention, norms, all unchanged. The accuracy cost is a
theorem, not a hope: each weight moves by at most **half a scale
step** (`|s·round(x/s) − x| ≤ s/2`, T9234–T9235), and everything
*after* quantization is bit-exact on every backend.

The row-by-row completeness contract (formats, both modes, every
exclusion) is `QUANT_CONTRACT.md` at the quanta-nn crate root. The
runnable twin of this page is `examples/cookbook_quantized_inference.rs`
(`cargo run --release --example cookbook_quantized_inference
--features "nn,jit"`) — it trains a small MLP, quantizes it, and
reloads it both ways, asserting the two modes agree bitwise.

## Quantize a trained model's named tree

Any `ParamTree` hands out its leaves with hierarchical names;
`quantize_named` walks them under a policy **you** write — there is no
magic name-matching. The house policy: rank-2 weight matrices carry
the weight mass and get quantized; norms, biases, and everything else
stays f32 (mixed checkpoints are the norm, not a special case).

```rust,ignore
use quanta::nn::quant::{self, Granularity, QuantDtype};

let leaves = params.named_flatten();   // ("blocks.0.attn.wq.w", Array) pairs

let entries = quant::quantize_named(&leaves, |name, arr| {
    if arr.shape().len() != 2 || !name.ends_with(".w") {
        return None;                   // keep f32
    }
    if name.contains("ffn") {
        // int4, grouped along the input axis — the accuracy-per-bit
        // sweet spot for the FFN mass (g = 32/64/128 in real models).
        Some((QuantDtype::Int4, Granularity::Group { axis: 0, size: 32 }))
    } else {
        // int8 per-out-channel — the accuracy default for projections.
        Some((QuantDtype::Int8, Granularity::PerChannel { axis: 1 }))
    }
})?;
```

## Save

`save_named` writes the **quanta quantized-safetensors** convention: a
quantized leaf `x` becomes two ordinary tensors (`x.q` codes, `x.qs`
scales) plus a `quant:x` metadata line, versioned under
`quanta.quant`. The file stays a *valid safetensors* — any third-party
inspector opens it; plain leaves keep their names and dtypes.

```rust,ignore
let bytes = quant::save_named(&entries, None)?;   // deterministic bytes
std::fs::write("model.q.safetensors", &bytes)?;   // your file IO
```

## Reload — two modes, one file

**Mode A — dequantize on load** is universal (every backend, WebGPU
included): each quantized leaf is dequantized on the host —
bit-identical to the device kernel — and uploaded as f32. The tree
form means your *unmodified* f32 model definition loads the quantized
checkpoint, matching by name with the usual loud
missing/extra/shape errors:

```rust,ignore
let bytes = std::fs::read("model.q.safetensors")?;

// The witness is your existing params tree — zero code changes.
let restored: LmParams<f32> = quant::load(&gpu, &params, &bytes)?;
```

(`quant::load_named_f32` is the same load as a raw named list, if you
are not using a tree.)

**Mode B — resident codes** keeps the quantized form on the device: 1
byte (int8) or half a byte (int4) per element between steps. Quantized
leaves arrive as `QuantizedMatrix`; feed them to `QuantizedLinear`:

```rust,ignore
use quanta::nn::quant::{QuantLeaf, QuantizedLinear};

let loaded = quant::load_named(&gpu, &bytes)?;    // LoadedQuant
let mut weight = None;
let mut bias = None;
for (name, leaf) in loaded.leaves {
    match (name.as_str(), leaf) {
        ("w", QuantLeaf::Quantized(m)) => weight = Some(m),
        ("b", QuantLeaf::F32(a)) => bias = Some(a),
        _ => { /* the rest of the tree */ }
    }
}
let ql = QuantizedLinear::new(weight.unwrap(), bias)?;
```

## Forward through `QuantizedLinear`

`QuantizedLinear` is a `Layer<f32>` with `Params = ()` — frozen
weights ARE configuration, so it stacks in tuples like any activation
and no optimizer ever sees the codes. Each forward dequantizes the
codes (one dispatch) and runs the ordinary proven matmul:

```rust,ignore
let tape: Tape<f32> = Tape::new();
let xv = tape.var(x);
let y = ql.apply(&tape, &(), &xv)?;               // [N, in] → [N, out]

// Tuple-stackable — Params = () occupies its slot for free:
let stack = (ql, Relu);
let y = stack.apply(&tape, &((), ()), &xv)?;
```

Two constructions, one layer: `QuantizedLinear::new(w, b)` holds the
codes resident and dequantizes per forward (Mode B);
`QuantizedLinear::dequantized(&w, b)` dequantizes **once** and holds
the f32 weight, so each forward costs exactly what `Linear` costs
(Mode A). Both produce the same values — bitwise.

## What accuracy you get — the s/2 statement

The only place information is lost is quantization itself, and its
error is proven (Lean, T9234–T9235): with max-abs scales every code is
in range (the clamp never fires) and every weight element moves by at
most `s/2`, exactly zero when the value already is a code multiple.
Everything downstream is exact: device dequantize equals the host
reference **bitwise**, `QuantizedLinear` equals `Linear` fed the
dequantized weight **bitwise**, Mode A equals Mode B **bitwise**, and
the quantized forward is bit-reproducible across backends. So the
output deviation from the f32 model is bounded by the propagated `s/2`
per weight — the end-to-end test derives its gate from exactly that
bound — and *which* backend you run on changes nothing.

## The WebGPU note

Resident **int8** codes need native 8-bit storage, which WebGPU does
not expose — `load_named` refuses such a leaf loudly, naming the leaf,
the backend, and the workaround. Mode A (`load` / `load_named_f32`) is
WebGPU's complete answer: same file, same values, ordinary f32 model
afterwards. Resident **int4** works everywhere, WebGPU included — the
packed words are core u32 storage.

## See also

- [API reference — quantized inference](../../reference/api.md#quantized-inference--sciquant--nnquant)
- [From PyTorch — quantization mapping](../../migration/from-torch.md#quantization)
- [A tiny transformer, end to end](../tutorials/transformer-lm.md) — the
  model this workflow quantizes
- `crates/ml/quanta-nn/QUANT_CONTRACT.md` — the declared surface,
  including what is excluded (GPTQ/GGUF import, affine, QAT) and why

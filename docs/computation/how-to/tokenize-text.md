# Tokenize text (run a pretrained tokenizer)

`quanta-tokenizers` runs the `tokenizer.json` artifact every model on
the Hub ships — GPT-class, BERT-class, Llama-class, T5 — through the
full pipeline the HF `tokenizers` reference defines (pinned at
0.21.x): normalize → pre-tokenize → model → post-process, plus decode,
special tokens, truncation and padding. It is a standalone crate with
**zero dependencies** (no quanta crates either — pure `std`,
wasm32-clean), so add it next to `quanta` rather than as a feature:

```toml
[dependencies]
quanta = { version = "0.1", features = ["sci", "autograd", "nn", "metal"] }
quanta-tokenizers = "0.1"
```

The row-by-row completeness contract (what runs, what is excluded and
why) is `TOKENIZER_CONTRACT.md` at the crate root.

## Load a real tokenizer.json

Grab the artifact from any Hub model (e.g.
`huggingface.co/openai-community/gpt2` → `tokenizer.json`). Loading is
bytes-level — `std::fs::read` is the one-liner, and validation is
eager: every pipeline stage is constructed and every vocab/merge
cross-checked up front, so a `Tokenizer` that loads, runs.

```rust,ignore
use quanta_tokenizers::Tokenizer;

let bytes = std::fs::read("tokenizer.json")?;
let mut tok = Tokenizer::from_bytes(&bytes)?;
```

An artifact written by a *newer* tokenizers version than the pin fails
loudly with the unknown tag named — never a misparse.

## Encode

```rust,ignore
let enc = tok.encode("The GPU ate my homework.", true)?; // add_special_tokens
enc.ids();                 // &[u32] — what the model eats
enc.tokens();              // the token strings
enc.attention_mask();      // 1 = real token, 0 = padding
enc.offsets();             // byte spans into the ORIGINAL input (span tasks)
```

Pairs (the BERT-class story — pair templates, `type_ids`) go through
`encode_pair(a, b, true)`.

## Encode a batch, padded

The artifact's saved truncation/padding load as the active defaults.
Override (or disable with `None`) explicitly — here, pad to the
longest sequence in the batch:

```rust,ignore
use quanta_tokenizers::artifact::{Direction, PaddingConfig, PaddingStrategy};

let pad_id = tok.token_to_id("<pad>").unwrap_or(0); // the artifact names its specials
tok.set_padding(Some(PaddingConfig {
    strategy: PaddingStrategy::BatchLongest,
    direction: Direction::Right,
    pad_to_multiple_of: None,
    pad_id,
    pad_type_id: 0,
    pad_token: "<pad>".into(),
}));

let batch = tok.encode_batch(&["a bird in the hand", "two in the bush"], true)?;
assert_eq!(batch[0].ids().len(), batch[1].ids().len()); // rectangular
```

`encode_batch` is a sequential host loop by design (the crate has zero
deps, so no thread pool); callers who want parallelism run chunks on
`std::thread::scope` threads and apply
`quanta_tokenizers::encoding::pad_encodings` afterwards. Host
tokenization is µs-scale against ms-scale model steps.

## Feed the ids to the model

The bridge to the GPU stack is one line — `Encoding` hands out plain
`&[u32]`, and `Embedding` heads the quanta-nn chain on
`ids: Array<u32>`:

```rust,ignore
use quanta::sci::Array;

let ids = Array::from_slice(&gpu, enc.ids(), &[enc.ids().len()])?;
let x = embedding.apply(&table_var, &ids)?;   // nn::embedding::Embedding — [B, E]
// … the rest of the stack: transformer blocks, loss, optimizer …
```

For rectangular batches, concatenate the padded id rows and shape
`[batch, seq]`; `attention_mask()` gives you the padding mask the
attention options take.

## Decode a generation stream

Whole-sequence decode is `tok.decode(&ids, true)?`. In a generation
loop you want tokens printed as they arrive — and naive per-token
decode is WRONG for byte-level artifacts, which split multibyte
characters across ids. `DecodeStream` holds incomplete bytes and emits
only finished text:

```rust,ignore
let mut stream = tok.decode_stream(true); // skip_special_tokens
for id in generated_ids {                 // your greedy/sampling loop
    if let Some(piece) = stream.step(id)? {
        print!("{piece}");                // new text only; held bytes stay held
    }
}
```

Concatenating every emitted piece equals decoding the whole id run —
that property is pinned by the conformance suite, split-multibyte
cases included. Out-of-range ids error loudly, naming the id and the
vocab size.

## See also

- [API reference — quanta-tokenizers](../../reference/api.md#quanta-tokenizers--run-pretrained-tokenizers)
- [From PyTorch — tokenizer mapping](../../migration/from-torch.md#tokenizers)
- [Train a model with quanta::nn](../tutorials/nn-training.md) — where the ids go next

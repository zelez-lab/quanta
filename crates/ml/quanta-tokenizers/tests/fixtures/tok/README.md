# Tokenizer conformance fixtures

Ground truth for the fixture-differential conformance layer (scope
§9/§10 — THE correctness contract of quanta-tokenizers). Everything in
this directory is either downloaded immutable bytes or emitted by the
pinned reference implementation; **CI never runs Python — the
committed bytes are the contract.**

## Provenance

Generated ONCE by a maintainer with:

    python3 -m pip install tokenizers==0.21.4
    python3 gen_fixtures.py

- reference: HF `tokenizers` **0.21.4** (the 084.8 pin; asserted at
  runtime by the script)
- generating interpreter: Python 3.14.3 (any >= 3.9 reproduces the
  outputs byte-identically — emission is `ensure_ascii` JSON with
  fixed key order, minimal artifacts are written by the reference
  library's own serializer, and regeneration has been verified
  byte-identical across runs)

## Inventory

Real full-size anchors (immutable, hash-pinned in `gen_fixtures.py`;
the ratified ~2.3 MB budget, scope §9):

| file | bytes | sha256 |
|---|---|---|
| `gpt2.tokenizer.json` | 1,355,256 | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `bert-base-uncased.tokenizer.json` | 466,062 | `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98` |

Sources: `huggingface.co/openai-community/gpt2` and
`huggingface.co/google-bert/bert-base-uncased`, `resolve/main/
tokenizer.json`. An artifact is immutable: the script re-verifies the
hashes on every run and refuses a mismatch.

Hand-built minimal artifacts, one per model family — constructed with
the pinned reference library's own builders and saved by ITS
serializer, so even the minimal artifacts are reference-written bytes,
not our guess at them:

| file | bytes | exercises |
|---|---|---|
| `tiny_bpe.tokenizer.json` | 8,429 | crafted merge chains + a late-rank merge-order tie (`lo` vs the `hel/hell` chain), `byte_fallback` over a full `<0x00>`..`<0xFF>` byte-token block, `TemplateProcessing` single/pair templates with type_ids, `<s>`/`</s>`/`<unk>` specials plus one non-special added token with `lstrip`/`rstrip` (`[area]`), `Whitespace` pre-tokenizer, `Sequence[ByteFallback, Fuse]` decoder |
| `tiny_wordpiece.tokenizer.json` | 2,279 | `BertNormalizer` (clean_text, handle_chinese_chars, `strip_accents: null` following lowercase), `BertPreTokenizer`, `BertProcessing` (the fixed-form ancestor tag; the real anchor covers `TemplateProcessing`), `WordPiece` decoder, `max_input_chars_per_word: 12` — a non-default so the long-word inputs cross the boundary both ways |
| `tiny_unigram.tokenizer.json` | 2,030 | Viterbi best-path over ▁-convention pieces, `unk_id` fallback, `Nmt` normalizer, `Metaspace` pre-tokenizer + decoder |
| `tiny_wordlevel.tokenizer.json` | 814 | plain lookup + `unk_token`, `Lowercase` normalizer, `Whitespace` pre-tokenizer, a non-ASCII vocab entry (`café`), no decoder (the join-with-space default path) |

Conformance vectors, one file per artifact (64 cases each):

| file | bytes |
|---|---|
| `vectors/tiny_bpe.vectors.json` | 42,884 |
| `vectors/tiny_wordpiece.vectors.json` | 25,763 |
| `vectors/tiny_unigram.vectors.json` | 35,816 |
| `vectors/tiny_wordlevel.vectors.json` | 22,546 |
| `vectors/gpt2.vectors.json` | 31,958 |
| `vectors/bert-base-uncased.vectors.json` | 28,778 |

Total committed fixture bytes: **2,022,615** (~1.93 MB, inside the
ratified budget; 1.82 MB is the two immutable anchors).

## Vector-file format

One JSON document per artifact, one case per line (diffable), pure
ASCII (`ensure_ascii` — exact Unicode inputs survive as escapes).
Header records the artifact name, generator, pinned reference version,
and the offsets convention. Each case:

    kind                  "single" | "pair"
    text, text_pair       the input(s); text_pair on pair cases only
    add_special_tokens    every input is emitted with true AND false
    ids, tokens, offsets, type_ids,
    special_tokens_mask, attention_mask, word_ids
                          the reference's full encoding record
    decoded               reference decode(ids, skip_special_tokens=true)
    decoded_raw           reference decode(ids, skip_special_tokens=false)

**Offsets are byte offsets into the original input.** The Python
binding reports char offsets; `gen_fixtures.py` converts them
deterministically against the known input via a cumulative UTF-8
byte-length map — per sequence for pair cases (`sequence_ids` selects
which text a token's span indexes), and a generation-time guard
asserts every reported offset is a valid char index (it would trip
loudly if the binding's offset unit ever changed). Special tokens
carry the reference's `(0, 0)` placeholder span, asserted and kept
verbatim. `word_ids` uses JSON `null` for None (special tokens).

## Curated input set — rationale

A shared adversarial set runs against every artifact (scope §9): plain
ASCII and merge-chain words plus the tie word `lol`; accents and
NFC/NFD-sensitive text (`café`, `naïve déjà vu`, `Ça va?`); CJK
(`handle_chinese_chars` isolation); emoji with skin-tone modifier and
a ZWJ family sequence (astral codepoints splitting into 4 byte-level
tokens sharing one char span — the offset-conversion trap); Devanagari
digits; contractions (`I'm`, `don't`); whitespace runs, tabs/newlines,
leading/trailing and only-whitespace inputs; the empty string; long
words crossing WordPiece's `max_input_chars_per_word=12` from both
sides plus a natural monster; special tokens literally in text, spaced
(`<s> hello </s>`) and glued (`a<s>b`); the `lstrip`/`rstrip`
added-token edge (`x [area] y`); byte-fallback bait (`ζ 🦀`); control
characters (`\x00\x07` — `clean_text` strips them, the alignment
trap); RTL Arabic; astral-plane letters and a hieroglyph. Pair cases
cover pair templates/type_ids, an empty first member, and mixed
astral/CJK pairs.

## Regeneration

Only ever needed to extend the set (a reference re-pin is a deliberate,
diffable commit):

    cd crates/ml/quanta-tokenizers/tests/fixtures/tok
    python3 -m pip install tokenizers==0.21.4
    python3 gen_fixtures.py

The script verifies the anchor hashes, rebuilds the minimal artifacts,
and re-emits every vector file; with the pinned version the result is
byte-identical to what is committed. `tests/conformance.rs` replays
these files through the public API (`Tokenizer::from_bytes` /
`encode` / `encode_pair` / `decode`) and asserts every field.

# Tokenizer support — declared surface (084.8, SHIPPED)

This file is the crate's completeness contract, in the
`quanta-nn/PARITY.md` shape: every row is either **shipped** or a
**documented deferral / exclusion with its reasoning**. Nothing on the
reference surface (what a saved HF `tokenizers` artifact can ask a
runtime to do) is silently absent. It began as the ratified scope
document; the statuses below are the as-built truth, and §11 records
the places where implementation corrected the scope's wording.

**RATIFICATION RECORD (owner decision).** Tokenizers ship
**COMPANION-GRADE, not verified-track Tier-A**: pure host-side data
plumbing with executable ground truth (HF tokenizer fixtures) — the
safetensors/npy profile. Spec-byte-exact and fixture-driven tests are
the contract; no Lean/Verus obligations. Both flagged rows were
ratified IN: (1) offsets/alignment — excluding the format's span
tracking would be a scheduled reopen; (2) the ~2.3 MB real-anchor
fixture budget (gpt2 + bert-base-uncased) — committed ground truth
from real artifacts is the correctness story.

**Governing principle (binding, from the npy ratification record):** a
feature is implemented so it is never REOPENED to extend or complete
it. A deferral is legitimate only when the exclusion or a prepared
seam IS the finished answer (a different product, a claim boundary, a
security stance), never when it schedules a known return visit. The
two places this principle bit hardest — Unicode normalization (§4)
and the split-regex engine (§6) — were decided under it explicitly.

Binding house rules, applied throughout:

- **Zero external dependencies** — extends to wasm32, and here to
  quanta crates too (§3). The JSON parser, the Unicode normalization
  tables, the regex subset engine, and the sentencepiece `Precompiled`
  charsmap interpreter are all in-crate (the `safetensors.rs` /
  `npy_codec.rs` hand-rolled precedent). "Zero dep" means no crates.io
  entries; committed *generated* tables with a pinned generator script
  are data, not dependencies (§4).
- **Community-complete.** The full interchange surface is declared
  here with explicit exclusions — the quanta-blas general-eig pattern:
  named, reasoned gaps, never silent ones.
- **Open-source generality.** The API is plain text-in / ids-out for
  any consumer. `ai_project`'s LLM-plugin path and dija's browser lane
  appear only as motivation; nothing in the shape is specific to them.
- **Bytes-level I/O.** Artifacts load from `&[u8]` (`std::fs::read`
  is the caller's one-liner); no file-path wrappers.

## 1. Scope of "tokenizer support" — run, never train

What a GPU-compute stack needs is **loading and running pretrained
tokenizers**: the consumer story is quanta-nn language models — token
ids in (`Embedding` heads the chain on `ids: Array<u32>`), strings out
(the generation loop decodes). The shipped surface is therefore **what
HF `tokenizers`' saved artifact provides at inference time**: the full
encode pipeline (normalize → pre-tokenize → model → post-process),
decode, special tokens, truncation/padding.

**Training new tokenizers is excluded, permanently.** It is a
different product (corpus statistics, trainers, progress reporting),
used once per model lineage, always offline, with Python available by
definition — while *running* a tokenizer is in every
inference/training step and is exactly the "no Python in the loop" gap
084.8 exists to close. The exclusion is the finished answer: nothing
in this crate's shape would be reworked by a future trainer (a trainer
*produces* the artifact this crate consumes), and `save`/serialization
goes with it (§12) — this crate never mutates a tokenizer, so the
artifact on disk already IS the serialized form.

## 2. The interchange format and the reference pin

**`tokenizer.json` — the HF `tokenizers` single-file artifact — is THE
interchange format.** It is what every model on the Hub ships
(GPT-class, BERT-class, Llama-class, Qwen, T5, …), it is
self-contained (vocab, merges, the whole pipeline configuration,
special tokens, saved truncation/padding), and it is the target every
other format converts *into* (transformers converts sentencepiece
`.model` and tiktoken vocab files to `tokenizer.json` on save). Raw
sentencepiece protobufs and tiktoken `.tiktoken` files as *input
formats* are excluded — the capability ("run a Llama / GPT-4-class
tokenizer") is served because those models' artifacts exist as
`tokenizer.json`; a future reader for another container would
construct the same pipeline structs and is additive, not a reopening
(§12).

The artifact's top level: `version`, `truncation`, `padding`,
`added_tokens`, and five tagged-enum pipeline stages — `normalizer`,
`pre_tokenizer`, `model`, `post_processor`, `decoder`. **The
conformance reference is pinned**: HF `tokenizers` **0.21.x**
(`tokenizers==0.21.4` exact-pinned in
`tests/fixtures/tok/gen_fixtures.py`, the numpy==2.3.2 precedent). The
binding rule held: every type tag the pinned reference deserializes
either ships or is a named exclusion in this file. Any tag outside the
pinned inventory (i.e. written by a *future* tokenizers version) is a
loud `UnknownTag` error naming the tag and the pinned reference — the
ZIP64-style claim boundary: "runs what the pinned reference format
expresses"; format growth is additive work against a loudly-detected
marker, never a misparse. NOTE the tagging correction the anchors
taught (§11 b): the reference also accepts the model object UNTAGGED,
and real artifacts use that spelling — so does this crate.

### Model families — all four ship

| Model `type` | Class | Status | Notes |
|---|---|---|---|
| `BPE` | GPT-2 / RoBERTa / Llama / GPT-4-class | **shipped — complete** | vocab + merges (BOTH serialized spellings: legacy `"a b"` strings and the current `["a","b"]` pairs), `unk_token`, `continuing_subword_prefix`, `end_of_word_suffix`, `fuse_unk`, `byte_fallback` (Llama), `ignore_merges` (Llama-3/GPT-4o-class). `cache_capacity` accepted and ignored (a perf hint, not semantics). `dropout` non-null → loud `UnsupportedField` (§12: BPE-dropout is stochastic train-time augmentation; inference artifacts ship `null`; a key-threaded stochastic encode would be a new capability, not a reopen). The merge loop replays the reference's ranked agglomeration verbatim: lowest rank first, ties leftmost, stale-entry re-checks. |
| `WordPiece` | BERT-class | **shipped — complete** | vocab, `unk_token`, `continuing_subword_prefix`, `max_input_chars_per_word` (the whole-word guard, honored). Greedy longest-match with the char-counted length guard; unknown is all-or-nothing, as the reference's `is_bad` flag. One loads-means-runs deviation at load time (§11 f). |
| `Unigram` | sentencepiece-class (T5, ALBERT, XLNet) | **shipped — complete** | `[[piece, logprob]]` vocab, `unk_id`, `byte_fallback`. Optimized Viterbi over an in-memory trie built at load, one f64 addition per lattice edge in the reference's operand order — ties break bit-identically (pinned by a dyadic exact tie and a one-ulp pair). |
| `WordLevel` | lookup-table models | **shipped — complete** | vocab + `unk_token`; the lookup floor; excluding a trivial family from "the format's model enum ships" would have been a silent gap for no savings. |

## 3. Placement — `crates/ml/quanta-tokenizers`

**A companion crate, not a module in quanta-nn.** The placement rule
that put safetensors in quanta-nn and npy in quanta-array is *subject
vocabulary*: a tokenizer's subject is **strings and `u32` ids —
neither crate's vocabulary**. Concretely:

- **Zero quanta dependencies.** The crate needs no `Gpu`, no `Field`,
  no backend features — it is pure host string processing, `std`-only,
  and it compiles for wasm32 (dija's browser lane tokenizes
  client-side with the same crate).
- **New audience** — the companion-layout test: text-pipeline
  consumers who never touch the training stack (corpus preprocessing,
  dataset tooling, browser-side tokenization).
- The bridge to the GPU stack is one documented line, not a
  dependency: `Array::from_slice(&gpu, encoding.ids(), &[n])?` feeds
  `Embedding` directly (`ids: Array<u32>` — the chain head). The
  doc-example lives on quanta-nn's side of the seam.

### JSON parsing — a complete hand-rolled RFC 8259 parser

`src/json.rs`: one complete recursive-descent RFC 8259 parser,
in-crate — the closed spec, not a subset, which is itself a
never-reopen line: there is no JSON document it will meet later that
the grammar doesn't already cover. Full number grammar lexed by hand
with value conversion delegated to std's correctly-rounded
`str::parse::<f64>()`; surrogate pairs decoded to the astral
codepoint, lone surrogates rejected loudly; byte-offset error messages
("at byte N" — the house contract); recursion depth capped (64 — real
artifacts nest ≤ ~8) so hostile nesting cannot blow the stack;
duplicate keys in one object rejected loudly (a vocab with duplicate
tokens is corrupt, not a last-wins guess).

## 4. Unicode normalization — vendored generated tables

The dependency-weight decision, resolved by the governing principle:
scoping to table-free variants or feature-gating the tables were both
rejected (each a guaranteed return visit or a two-flavor "complete").
**Vendored generated tables shipped**: `src/unicode/tables.rs`,
produced by the checked-in, pinned generator
(`src/unicode/gen_unicode_tables.py`, UCD **16.0.0**, sources pinned
by SHA-256, self-checked against CPython's `unicodedata` on every
codepoint in all four forms before emitting — run once by a
maintainer, CI never runs Python, the committed artifact is the
truth). Scoped to exactly the properties the shipped surface consumes:
canonical + compatibility decompositions, canonical combining classes,
composition pairs + exclusions (Hangul stays algorithmic, no table),
and range-tables for the general categories and scripts the
normalizers and regex classes reference. NFD/NFC/NFKD/NFKC, canonical
ordering, and Hangul composition are pinned by 399 sampled
`NormalizationTest.txt` vectors plus the reference-generated
conformance layer (§9) — per the owner ruling, fixtures, not proofs.
Grapheme clustering (UAX #29) and full unconditional lowercase ride
the same tables.

## 5. Pipeline stages — the never-reopen line per family

Posture, as shipped: **the pinned reference version's full variant
inventory runs**, except the named exclusions. "Complete" per variant
means: every serialized field either affects behavior as the reference
implements it, or is named here as accepted-and-ignored. Unknown
*fields* inside a known variant are ignored — the pinned reference's
own serde posture (how `cache_capacity` and the legacy `str_rep` are
accepted-and-ignored); absent fields take the reference builders'
defaults.

### Normalizers (14)

| Variant | Status | Notes |
|---|---|---|
| `NFC` / `NFD` / `NFKC` / `NFKD` | **shipped** | over §4's tables |
| `BertNormalizer` | **shipped** | clean_text (control-char strip), handle_chinese_chars (CJK spacing ranges, the reference's exact range list including its `0x2B920` quirk), strip_accents (NFD + drop `Mn`; `null` follows `lowercase` — the reference's documented default), lowercase |
| `Lowercase` / `Strip` / `Prepend` / `Replace` | **shipped** | `Replace` with a `Regex` pattern routes through §6's engine (String patterns are plain find/replace) |
| `StripAccents` | **shipped** | drops ALL marks (`M*`), NO decomposition — the reference's actual standalone semantics; the scope's "NFD + drop Mn" wording was BertNormalizer-internal only (§11 a) |
| `Precompiled` | **shipped** | the sentencepiece charsmap: base64 blob **inside the artifact** (no vendored data needed) interpreted by an in-crate double-array-trie walker mirroring the reference's semantics quirks, every transition bounds-checked (§8 — this blob is the most hostile-input-shaped structure in the format); hostile blobs degrade to misses where the reference would panic |
| `Nmt` | **shipped** | fixed codepoint filter list, in code |
| `ByteLevel` (normalizer form) | **shipped** | Llama-3-style artifacts; same byte↔char bijection as the pre-tokenizer |
| `Sequence` | **shipped** | ordered composition, arbitrary nesting |

Every normalizer runs over the alignment-tracked `NormalizedString`
(the reference's change-stream transform algebra, `src/normalized.rs`)
so byte offsets into the ORIGINAL input survive any normalizer stack.

### Pre-tokenizers (12)

| Variant | Status | Notes |
|---|---|---|
| `ByteLevel` | **shipped** | the GPT-2 256-byte↔printable-char bijection (a hardcoded table, not UCD), `add_prefix_space`, `trim_offsets`, `use_regex` (the GPT-2 pattern via §6) |
| `BertPreTokenizer` | **shipped** | whitespace split + punctuation isolation (category tables) |
| `Whitespace` / `WhitespaceSplit` / `Punctuation` / `Digits` / `CharDelimiterSplit` | **shipped** | category-table driven where applicable; `Whitespace`'s `\w` follows Oniguruma exactly, Join_Control included (§11 c) |
| `Metaspace` | **shipped** | `replacement` (`▁`), `prepend_scheme` (Always/Never/First), `split` — **and** the legacy `add_prefix_space` serialization; both spellings load, deliberately wider than the pinned deserializer (§11 e) |
| `Split` | **shipped** | `pattern` (String or Regex — Regex via §6), all five `behavior` modes (Removed / Isolated / MergedWithPrevious / MergedWithNext / Contiguous), `invert` |
| `UnicodeScripts` | **shipped** | script ranges from §4's tables |
| `FixedLength` | **shipped** | fixed char-count chunks (surfaced by the §2 pinned-inventory audit, ships under the every-tag-ships rule) |
| `Sequence` | **shipped** | |

### Post-processors (5)

| Variant | Status | Notes |
|---|---|---|
| `TemplateProcessing` | **shipped** | the general one: single/pair templates, `type_ids`, special-token id maps — chat-model artifacts live here |
| `BertProcessing` / `RobertaProcessing` | **shipped** | the fixed-form ancestors |
| `ByteLevel` (post-processor form) | **shipped** | `trim_offsets` — offset surgery only |
| `Sequence` | **shipped** | |

### Decoders (10)

| Variant | Status | Notes |
|---|---|---|
| `ByteLevel` | **shipped** | inverse bijection → raw bytes → lossy-utf8 (reference semantics for partial sequences) |
| `WordPiece` / `BPEDecoder` / `Metaspace` / `ByteFallback` / `Fuse` / `Strip` / `Replace` / `CTC` | **shipped** | each small string surgery; the family ships whole — a decoder gap strands its model family's decode path |
| `Sequence` | **shipped** | |

### Added tokens / special tokens

| Item | Status | Notes |
|---|---|---|
| `added_tokens` — the AddedVocabulary | **shipped — complete** | the full flag set: `special`, `single_word`, `lstrip`, `rstrip`, `normalized` (non-normalized tokens split the RAW text before the normalizer; normalized ones match after — the reference's two-pass split, leftmost-longest), `content` → direct id mapping. This is the machinery every chat model's `<\|im_start\|>`-class tokens ride; it is not optional plumbing. |
| Special-token id lookups | **shipped** | via `token_to_id` — no bespoke accessors; the artifact names its own specials |

## 6. The split-regex engine — a construct-set claim boundary

`Split`, `ByteLevel(use_regex)`, and `Replace(Regex)` carry regex
patterns; the reference runs them on Oniguruma. Reimplementing a
general Oniguruma is not honest zero-dep work, and hardcoding the
known model patterns would be reopened by every new model lineage.
**Shipped: an in-crate backtracking engine over the closed construct
set the artifact corpus uses** (`src/regex.rs`) — literals,
alternation, character classes including `\p{…}`/`\P{…}` over §4's
category tables, `\s`/`\d`/`\w` and negations, `?` `+` `*` `{m,n}`
(greedy + lazy), non-capturing groups, inline case-insensitive groups
`(?i:…)`, and negative lookahead `(?!…)` — the full construct
inventory of the GPT-2 pattern, the cl100k/Llama-3 pattern, and every
whitespace/punctuation pattern in the mainstream corpus, each a named
engine test pinned against the REAL tables. A pattern using a
construct outside the set is a loud `RegexConstruct` error naming the
construct and the pattern — the same claim shape as npz/ZIP64: "runs
the patterns the model corpus writes", detected loudly at the
boundary; a new construct is additive engine work, not a redesign.
Input-linear-bounded backtracking budget with a loud error on blowup
(patterns are artifact-supplied, i.e. hostile); nullable-unbounded
quantifiers are rejected at parse so the budget error always means
blowup. `\w`'s exact membership was arbitrated by the conformance
layer (§11 c).

## 7. API surface

| Item | Status | Notes |
|---|---|---|
| `Tokenizer::from_bytes(&[u8]) -> Result<Tokenizer, TokenizerError>` | **shipped** | the house bytes rule; `std::fs::read` is the caller's one-liner. Validates the WHOLE artifact eagerly — every stage constructed, every vocab/merge cross-checked (§8) — so a `Tokenizer` that loads, runs. `from_artifact(&TokenizerArtifact, Box<dyn Model>)` is the layered constructor for callers that bring their own model. |
| `encode(&str, add_special_tokens: bool) -> Result<Encoding>` | **shipped** | full pipeline incl. the added-token two-pass split and post-processor; `&str` in (text is text; bytes-level applies to artifacts, not input) |
| `encode_pair(&str, &str, add_special_tokens: bool) -> Result<Encoding>` | **shipped** | the BERT-class pair story: pair templates, `type_ids`, OnlyFirst/OnlySecond truncation — WordPiece without pairs is half a tokenizer |
| `encode_batch(&[impl AsRef<str>], add_special_tokens: bool) -> Result<Vec<Encoding>>` | **shipped** | sequential host loop + batch padding (`BatchLongest` needs the batch). Deliberately not internally threaded (zero-dep: no rayon); the documented spelling for callers who care is `std::thread::scope` over chunks + `encoding::pad_encodings`. Perf posture: host tokenization is µs-scale against ms-scale model steps — a non-goal to race HF (§12). |
| `Encoding` — `ids: Vec<u32>`, `type_ids`, `tokens`, `offsets: Vec<(usize, usize)>`, `special_tokens_mask`, `attention_mask`, `word_ids: Vec<Option<u32>>`, `overflowing: Vec<Encoding>` | **shipped** | the reference's full encoding record. **Offsets are IN** (ratified): byte offsets into the ORIGINAL input, carried through normalization by the alignment-tracked string (`src/normalized.rs`) — span tasks (QA/NER extraction) are squarely "running pretrained models". Alignment helpers ship with it: `token_to_chars`, `token_to_word`, `word_to_tokens`, `word_to_chars`, `char_to_token`, `char_to_word`. |
| `decode(&[u32], skip_special_tokens: bool) -> Result<String>` + `decode_batch` | **shipped** | runs the artifact's decoder chain; an out-of-range id is a loud error naming the id and the vocab size — a documented deviation from the reference's silent skip (§11 f) |
| `DecodeStream` (`tokenizer.decode_stream(skip_special_tokens)`, `stream.step(id) -> Result<Option<String>>`) | **shipped** | incremental detokenization for generation loops — the OUTPUT-side twin of padding's "universally needed" argument. Prefix-diff semantics: emit the new valid-UTF-8 suffix, hold bytes split across tokens (byte-level artifacts split multibyte chars across ids — naive per-token decode is WRONG there). Ships the documented contract over a sound window; the pinned reference's own implementation panics (§11 d). |
| `token_to_id(&str) -> Option<u32>` / `id_to_token(u32) -> Option<&str>` / `vocab_size(with_added: bool)` / `get_vocab(with_added: bool)` | **shipped** | the lookup quartet every pipeline builds against |
| Truncation / padding: artifact values load as the ACTIVE defaults; `set_truncation(Option<TruncationConfig>)` / `set_padding(Option<PaddingConfig>)` override or disable; `truncation()` / `padding()` read back | **shipped** | reference semantics (the saved sections are the saved behavior — ignoring them would mis-run the artifact as shipped). Full params: truncation `max_length` / `strategy` (LongestFirst / OnlyFirst / OnlySecond) / `direction` / `stride` **with overflow** (`Encoding::overflowing`); padding `strategy` (BatchLongest / Fixed) / `direction` / `pad_id` / `pad_type_id` / `pad_token` / `pad_to_multiple_of`. The helpers are pure `Encoding` transforms a caller can also disable (`set_*(None)`) and apply manually. |
| File-path wrappers, `save()` / serialization | **not planned** | bytes-level house rule; save excluded with training (§1, §12) |

## 8. Error taxonomy + hostile-input posture

Own error enum — the crate has no quanta deps, so no `ArrayError`
wrapping; same message contract as `NpyError` (self-contained, names
the offender, states the workaround where one exists):

| `TokenizerError` variant | Covers | Message contract |
|---|---|---|
| `Json { at, what }` | RFC 8259 violations, truncation, depth cap, lone surrogates, duplicate keys | byte offset, the safetensors "at byte N" style |
| `Schema { path, what }` | wrong type / missing field / malformed structure at a known location | JSON-path (`"model.vocab"`, `"added_tokens[3].content"`) |
| `UnknownTag { family, tag }` | a `type` tag outside the pinned inventory (§2's claim boundary) | family, the tag, the pinned reference version (`PINNED_REFERENCE`) |
| `UnsupportedField { path, why }` | in-inventory but excluded semantics (BPE `dropout` non-null) | the exclusion's reason from this file |
| `Vocab { what }` | duplicate tokens, id collisions, ids ≥ 2³², a merge naming an absent token, Unigram `unk_id` out of range, WordPiece `unk_token` absent (§11 f) | the offending token/id — load-time, per §7's loads-means-runs rule |
| `Charsmap { at, what }` | malformed `Precompiled` blob: bad base64, out-of-bounds trie transition | blob offset |
| `RegexConstruct { pattern, construct }` | §6's claim boundary; also the backtracking-budget trip | names both |
| `Encode { what }` | pipeline-time faults (a `Split` pattern failure, template referencing an id the vocab lacks) | the stage and the fault |
| `Decode { id, vocab_size }` | out-of-range id | the id and the vocab size |

Hostile-input posture — **same grade as npy** (artifact files are
untrusted): every JSON length is text-bounded (no unvalidated size
drives an allocation); recursion depth capped; the base64 charsmap
length-checked before decode and its trie walker bounds-checks every
transition (a crafted blob must not OOB or spin — hostile blobs
degrade to misses where the reference would panic); §6's backtracker
carries an input-proportional budget (a crafted *pattern* + input must
not go exponential); merge ranks / template ids validated at load.
Encode-time complexity is stated, not hidden: the BPE merge loop is
rank-driven per pre-token, and `WordPiece::max_input_chars_per_word`
is honored (the reference's own giant-token guard).

## 9. Correctness contract — fixture-differential against the reference

The owner ruling's executable ground truth, npy-pattern verbatim: a
pinned `gen_fixtures.py` (`tokenizers==0.21.4`) is run ONCE by a
maintainer; **CI never runs Python — the committed bytes are the
contract.** Full inventory, hashes, and the vector-file format:
`tests/fixtures/tok/README.md`.

| Fixture class | As landed |
|---|---|
| **Hand-built minimal artifacts, one per model family** | `tiny_bpe` (merge chains + a late-rank merge-order tie, full `byte_fallback` block, `TemplateProcessing`, added-token `lstrip`/`rstrip`), `tiny_wordpiece` (BertNormalizer, `BertProcessing`, non-default `max_input_chars_per_word`), `tiny_unigram` (Viterbi over ▁-pieces, `Nmt`, Metaspace both roles), `tiny_wordlevel` — each built *by the pinned reference library's own builders* and saved by ITS serializer, so even the minimal artifacts are reference-written bytes, not our guess at them. |
| **Two real full-size anchors** | `gpt2` (1.36 MB — byte-level BPE, the most-trapped encode path) and `bert-base-uncased` (466 KB — WordPiece + BertNormalizer + pairs, the normalization-trapped path), immutable and sha256-pinned; the script re-verifies the hashes and refuses a mismatch. Total committed fixture bytes ~1.93 MB, inside the ratified budget. The anchors earned their keep on first contact (§11 b, §11 c). |
| **Conformance vectors** | 64 cases per artifact × 6 artifacts = **384 vectors**, every case carrying the reference's full record — ids, tokens, BYTE offsets (converted deterministically from the binding's char offsets), type_ids, both masks, word_ids, decode round-trips both `skip_special_tokens` ways — over the shared adversarial input set (merge ties, byte-fallback bait, NFC/NFD-sensitive accents, CJK, emoji ZWJ families, RTL, astral plane, control chars, whitespace runs, empty string, specials glued mid-text, added-token edges, pair cases, WordPiece length-guard boundaries). All 384 replay bit-exact through the public API. |
| **Error fixtures** | in-tree hand-built cases per §8 row a file can trigger: truncated JSON, unknown tags (future-version stubs), non-null `dropout`, corrupt charsmap, duplicate vocab entries. |

As-built delta from the scope's fixture plan, recorded honestly: the
planned *sentencepiece-converted real `Precompiled` fixture* (a
`transformers`-converted tiny sentencepiece model) did not land in the
committed set. The `Precompiled` path is instead pinned by the
walker's line-for-line mirror of the reference source (its double-array
semantics quirks included), hand-built charsmap blobs at unit and
pipeline level, and hostile-blob degradation tests. A
reference-produced charsmap artifact remains the natural additive
hardening if a sentencepiece-heavy consumer arrives; nothing in the
walker's shape would change.

## 10. Test coverage + CI lane

Home: `tests/` + `tests/fixtures/tok/`.

| Layer | Status | Notes |
|---|---|---|
| JSON parser unit suite | **shipped** | RFC 8259 vectors incl. surrogate pairs, exponent floats, depth cap, duplicate keys, truncation fuzz at structural boundaries (the npy truncation-fuzz pattern) |
| Unicode layer | **shipped** | 399 sampled `NormalizationTest.txt` vectors across all four forms, Hangul round-trips, category/script lookups against pinned UCD facts, grapheme clustering, charsmap walker (valid + hostile blobs) |
| Regex engine | **shipped** | each corpus pattern (GPT-2, cl100k-class, whitespace/punct family) against reference behavior over the REAL tables; construct-boundary loud errors; backtracking-budget trip |
| Per-variant component tests | **shipped** | every §5 row exercised (`tests/artifact.rs` schema forms, `tests/pipeline.rs` execution with alignment) |
| Model families | **shipped** | `tests/{bpe,wordpiece,unigram,wordlevel}.rs` — every expected token list reference-produced, not hand-derived; Unigram tie-breaking pinned bit-exact |
| Fixture-differential conformance | **shipped** | the §9 vectors, bit-exact on ids/tokens/masks/offsets/word_ids/decodes — THE contract (`tests/conformance.rs`) |
| Real-anchor conformance | **shipped** | gpt2 + bert-base-uncased over the full vector set |
| Error-path coverage | **shipped** | per §8 row, asserting the message carries the promised context |
| Round-trip properties | **shipped** | `DecodeStream` concatenation ≡ whole-sequence decode, split-multibyte cases included |
| Bridge doc-example | **shipped** | ids → `Array<u32>` → `Embedding` — documented on quanta-nn's side (`docs/computation/how-to/tokenize-text.md`); the crate itself stays quanta-free |
| CI lane | **shipped** | **companion-tests** (`cargo test -p quanta-tokenizers` — no backend features exist to choose). Pure host, backend-invariant by construction; no new lane. wasm32 honored by the cross-target check posture. |

## 11. Corrections the implementation taught

The scope was written against the reference's documentation; the
implementation was written against its behavior. Where they differed,
behavior won, and the difference is recorded here — each entry is
verifiable in the named source file.

**(a) Standalone `StripAccents` drops ALL marks, with NO
decomposition.** The scope's "drop `Mn` after decomposition (reference
semantics)" wording is not what the reference's standalone
`StripAccents` does: it filters `is_combining_mark` — General_Category
= Mark, i.e. all of `Mn`/`Mc`/`Me` — over the text AS IS, decomposing
nothing. The "NFD + drop `Mn`" spelling exists only *inside*
`BertNormalizer`'s `strip_accents` flag. As shipped
(`src/normalize.rs`): `StripAccents` drops every `M*` codepoint,
no decomposition; an artifact wanting the composed form stripped
spells `Sequence[NFD, StripAccents]`, exactly as the reference
requires.

**(b) Real artifacts serialize the model UNTAGGED.** The reference's
`ModelWrapper` deserializes tagged OR untagged, and real-world
artifacts — gpt2 and bert-base-uncased among them — carry the legacy
untagged model object. The conformance anchors caught this on first
contact. The schema layer (`src/artifact.rs::model`) infers the family
from field shape when `type` is absent, deterministic on distinctive
required fields: `merges` → BPE, `max_input_chars_per_word` →
WordPiece, an array-of-pairs vocab → Unigram, a map vocab with
`unk_token` → WordLevel. An object matching none of these is a loud
schema error naming the shapes tried.

**(c) Oniguruma's `\w` includes Join_Control.** The engine's word
class is letters, marks, decimal digits, connector punctuation — AND
Join_Control, exactly U+200C (ZWNJ) and U+200D (ZWJ)
(`src/props.rs`). The emoji-ZWJ conformance vectors arbitrated the
last member: the reference's `Whitespace` pre-tokenizer splits emoji
ZWJ families into alternating single-char matches because ZWJ is `\w`
there — probed live against tokenizers 0.21.4. Without Join_Control,
384-vector conformance is unreachable.

**(d) The pinned reference's own `DecodeStream` panics.** The 0.21.x
implementation's index algebra underflows — a panic — after seven
consecutive emitting steps, and mis-emits just before; probed
empirically against the real crate. This crate ships the *documented*
contract over a sound window (`src/decode.rs`): a `[context | active]`
id window re-anchored on every emit, so concatenated emits equal the
whole-sequence decode indefinitely. The port keeps the contract, not
the bug; the conformance anchor is the property the reference's docs
state, not the behavior its code exhibits.

**(e) Metaspace's legacy spelling loads wider than the pinned
deserializer.** The scope said the reference "still accepts" the
legacy `add_prefix_space` serialization; the 0.21 line's deserializer
in fact refuses it (the field left the format in the 0.20 line). Wave
one deliberately keeps the mapping
(`src/artifact.rs::metaspace_fields`): legacy `add_prefix_space:
true/false` → `prepend_scheme: Always/Never` when `prepend_scheme` is
absent, the explicit scheme winning when both appear, `str_rep`
accepted-and-ignored — so artifacts serialized by tokenizers ≤ 0.19
load here that 0.21.x refuses. A deliberate claim widening, not drift:
it cannot change the meaning of any artifact the pinned reference
accepts.

**(f) Loads-means-runs beats bug-for-bug, twice.** (1) WordPiece
resolves `unk_token ∈ vocab` at LOAD (`src/wordpiece.rs`); the
reference resolves it lazily and faults *mid-encode*
(`MissingUnkToken`) the first time an unknown word appears. Every
artifact the reference can fully run loads here; one that would fault
on its first unknown word is refused at load with the same fault
named. (2) `decode` on an out-of-range id is a loud
`TokenizerError::Decode` naming the id and the vocab size
(`src/tokenizer.rs`); the reference silently skips unknown ids —
silence loses data, and a generation loop feeding garbage ids deserves
to hear about it.

## 12. Explicitly out of scope — consolidated

| Item | Status | Reasoning |
|---|---|---|
| Training tokenizers (BPE/WordPiece/Unigram trainers) | **excluded (permanent)** | §1 — a different product; the artifact is the interchange; nothing here is reworked by a future trainer |
| `save()` / writing `tokenizer.json` | **excluded** | nothing originates here (no trainer); the loaded artifact IS its own serialization; a future trainer owns save |
| sentencepiece `.model` / tiktoken files as input formats | **excluded (seam named)** | `tokenizer.json` is THE interchange (§2); those corpora ship tokenizer.json on the Hub; a future reader is an additive constructor over the same pipeline structs |
| Chat templates (`tokenizer_config.json`'s Jinja) | **excluded** | a *different artifact* (transformers-level, not tokenizer.json) and a template-engine product; the mechanical need (special tokens) ships in §5; the caller formats the prompt string |
| `tokenizer_config.json` / `special_tokens_map.json` sidecars | **excluded** | tokenizer.json is self-contained for encode/decode (`added_tokens` carries the specials) |
| BPE `dropout` (stochastic encode) | **excluded (loud at load)** | train-time augmentation entangled with an RNG-policy API; inference artifacts ship `null`; a key-threaded stochastic encode (the `Dropout`-layer Key precedent) would be a new additive capability |
| Per-language segmenters beyond the format (MeCab/Jieba-class) | **excluded** | not expressible in the artifact; the reference doesn't run them either — outside the reference surface entirely |
| Byte-parallel / multithreaded internal encode | **excluded** | zero-dep (no rayon); sequential + the `std::thread::scope` spelling documented; µs-vs-ms perf posture stated in §7 — racing HF throughput is a non-goal, correctness parity is the product |
| GPU-side tokenization | **excluded (seam named)** | host-side is right today: branchy per-string control flow with dictionary lookups, a vanishing fraction of pipeline time, and the format's semantics are string-algorithmic. The seam a future GPU increment would use is already the API boundary: `Encoding`'s id/mask vectors are the device-upload contract, so a batch GPU tokenizer (cuDF-subword-class) would be a new producer of the same `Encoding` record — no API rework |
| Future tokenizers-format versions / unknown tags | **claim boundary** | §2 — loud `UnknownTag` naming tag + pinned version; growth is additive against a detected marker |
| Regex constructs outside §6's set | **claim boundary** | loud `RegexConstruct`; additive engine work if the corpus ever moves |
| Perf benchmarks / quanta-bench lane | **not planned** | no perf claim is made (§7 posture); nothing to gate |

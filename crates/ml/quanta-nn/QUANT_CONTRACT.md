# Quantized checkpoint inference — declared surface (084.8 quantized-inference slice, SHIPPED)

This file is the sub-feature's completeness contract, in the
`quanta-nn/PARITY.md` shape: every row is either **shipped** or a
**documented deferral / exclusion with its reasoning**. Nothing on the
reference surface (what a quantized checkpoint can ask an inference
runtime to do) is silently absent. It began as the ratified scope
document; the statuses below are the as-built truth, and §12 records
the places where integration corrected the scope's wording. The
surface spans two crates by the subject-vocabulary rule (§4–§5):
`quanta_array::quant` holds the array-level primitive, this crate's
`quant` module holds the checkpoint + layer face.

**RATIFICATION RECORD (owner decision).** (1) Verified-track grade =
**COMPANION plus the Lean supplement** (the s/2 round-trip bound and
clamp-free max-abs scales, T9234–T9235) — full Tier-A rejected, the
op-lowering proofs shipped with 084.1; (2) **GGUF Q8_0 read-in
EXCLUDED** (external formats are claim boundaries; ours is
safetensors-valid); (3) **int8 resident codes on WebGPU EXCLUDED**,
inheriting the narrow-dtype ruling — the universal host-dequant mode
covers WebGPU whole.

**Governing principle (binding, from the npy ratification record):** a
feature is implemented so it is never REOPENED to extend or complete
it. A deferral is legitimate only when the exclusion or a prepared
seam IS the finished answer (a different product, a claim boundary, a
reserved enum arm that makes the extension a value change), never when
it schedules a known return visit.

Binding house rules, applied throughout:

- **Zero external dependencies** — the format work extends the
  hand-rolled `safetensors.rs` machinery; the kernels ride the shipped
  IR ops.
- **Community-complete** — the full inference surface is declared here
  with named exclusions (the quanta-blas general-eig pattern).
- **Open-source generality** — the API is quantize / store / load /
  run for any consumer; nothing in the shape is specific to any
  in-stack consumer.
- **Bytes-level I/O** — checkpoints load from `&[u8]`, no file-path
  wrappers (the safetensors/npy house pattern).
- **Lower tier first** — every kernel primitive composes shipped
  quanta-ir arms (`Load(I8/I4)` + `Dequantize` + `Store(F32)`); the
  slice composes, it does not reimplement.

## 1. What "quantized checkpoint inference" means here

**Weight-only, symmetric integer quantization with f32 activations.**
Weights are stored as int8 or packed-int4 codes plus f32 scales; at
inference the codes dequantize (`w = scale · q`, exact symmetric
algebra) and the model otherwise runs the existing proven f32 stack —
gemm, fused SDPA, norms, losses, all unchanged. Two load modes ship
(§6): dequantize at load (universal, every backend) and
resident-quantized with per-forward dequantization (the steady-state
memory win, capability-gated for int8).

**The claim** (the ZIP64-style boundary): *quantizes any checkpoint
the stack can load, stores it in the quanta quantized-safetensors
convention, and runs it — bit-reproducibly across backends.* It does
NOT claim to read GPTQ/AWQ/GGUF artifacts; those are loud-error claim
boundaries with named seams (§2).

Half precision is not this surface: f16/bf16 checkpoints already load
(`safetensors.rs` upconverts exactly) — that is dtype interchange, not
quantization.

## 2. External quantized formats — where the line is

| Format family | Status | Reasoning |
|---|---|---|
| **quanta quantized-safetensors** (§8) | **shipped — THE format** | Self-defined, self-describing, additive-by-construction, and a *valid safetensors file* any third-party tool can open (pinned by a test that reads the container with the plain loader). The in-stack consumer path is closed end-to-end: load any f32/f16/bf16 checkpoint → quantize in-stack → save → load → run. No external artifact is required for the capability to be real. |
| Plain int8 + scale safetensors (torchao / compressed-tensors W8A16-class) | **excluded (claim boundary, seam named)** | The tensors are expressible in our vocabulary (symmetric per-channel int8 — exactly §3), but the scheme is carried in a transformers-level sidecar (config.json) this stack does not parse — a *different artifact*. A future importer is a *reader* that constructs the same `QuantizedMatrix` — additive, not a reopen. Unknown-dtype tensors in a plain `safetensors::load` already error loudly naming the tensor. |
| **GPTQ / AWQ** safetensors | **excluded (blocked on Affine — a reserved arm, not a gap)** | The IR carries the `zero_point` register and the `Affine` mode arm precisely so this is a value change later. Shipping affine now would mean shipping it without a consumer contract (each format packs zeros differently) — format importers and the affine lowering land together, as their own slice, constructing the same `QuantizedMatrix` + an additive zero-points field the file format already reserves (§8). |
| **GGUF** (llama.cpp) | **excluded (own future slice; seam = the same constructors)** | A different container AND a family of block codecs, most affine. Its simplest member (Q8_0: per-32-block symmetric int8 with f16 scales) maps onto §3's scale grid exactly — it was the one defensible pull-in, and the owner ratified the exclusion: it drags in a second container's alignment/metadata surface for one codec of a zoo whose center of mass (k-quants) is affine and stays out regardless — a partial GGUF reader would hollow the "runs GGUF" claim the way a partial descr table would have hollowed npy's. The seam (constructing `QuantizedMatrix` from a foreign reader) is identical either way. |
| **bitsandbytes NF4/FP4** | **excluded (permanent as-is)** | A different value axis (code-book lookup, not affine ints). If ever wanted, it is a new `QuantValue` arm — the orthogonal-axis construction absorbs it additively. |
| **FP8-scaled checkpoints** (F8_E4M3 + block scale-inv) | **deferred (named)** | The FP8 dtypes are shipped in the IR, but scaled-fp8 is a different dequant identity over a different payload; a codes variant is additive when a consumer exists. |
| f16 / bf16 checkpoints | **already shipped elsewhere** | `safetensors.rs` upconverts on load — not quantization (§1). |

## 3. The quantization vocabulary

**Values** — `QuantDtype::{Int8, Int4}`, both symmetric, mapping to
the IR's `Q8S`/`Q4S` (IR types never appear in the public API). Int4
is stored packed: 8 signed nibbles per `u32` word, **low nibble
first** (the GPTQ / llama.cpp layout), rows packed independently,
final word of each row zero-padded.

**Granularity — the 2-D scale grid, one formulation (the never-reopen
shape).** Quantized leaves are rank-2 `[R, C]`; rank ≠ 2 is a loud
error (§13). Scales form a tile grid: tile size `(gr, gc)`, scales
tensor shape `[⌈R/gr⌉, ⌈C/gc⌉]`, and the scale for element `(r, c)` is
`scales[(r/gr) · ⌈C/gc⌉ + (c/gc)]`. Every granularity anyone ships is
a choice of `(gr, gc)`:

| Host spelling | `(gr, gc)` | Covers |
|---|---|---|
| `Granularity::PerTensor` | `(R, C)` | one scale |
| `Granularity::PerChannel { axis: 0 }` | `(1, C)` | scale per row |
| `Granularity::PerChannel { axis: 1 }` | `(R, 1)` | scale per column — Linear's per-out-channel, the accuracy default |
| `Granularity::Group { axis: 0, size: g }` | `(g, 1)` | g input rows per output column — the int4 grouped form (g = 32/64/128) |
| `Granularity::Group { axis: 1, size: g }` | `(1, g)` | the transposed twin |

One dequant kernel per codes kind, parameterised by `(R, C, gr, gc)` —
adding a granularity later is a host-enum arm over the same kernel,
never a kernel change. Int4 shipped WITH `Group`: per-tensor int4 is
an accuracy wrong-answer generator for the type's primary population,
and shipping int4 without groups would have been the textbook
scheduled reopen.

**Mode: symmetric only** — enforced at the host API (no affine
constructor exists to call). The IR op's `zero_point` register and
`Affine` arm stay reserved; the file format reserves the zero-points
slot (§8), so affine is additive on all three layers when its slice
arrives.

**Rounding**: `round_ties_even` + clamp, inherited from
`quanta_ir::dtype` (`quantize_sym`), already pinned by the op matrix
and mirrored by every emitter. The quantizer calls the `dtype.rs`
functions directly — the reference arithmetic is never re-spelled.

**Scale selection**: max-abs per tile — `s = max|w| / hi` (`hi` =
127 / 7). A pure function of the weight tensor; deterministic;
requires no data, so it is NOT calibration (§13 draws that line). With
max-abs scales the clamp provably never fires and code −128 (−8) is
never produced — T9235. An all-zero tile stores `s = 0` with all-zero
codes and dequantizes exactly (documented, not an error).

## 4. API surface — `quanta_array::quant` (the array-level primitive)

Subject: "integer codes + scales ↔ `Array<f32>`" — quanta-array's
vocabulary (it is the raw-`KernelDef` + `wave_jit` idiom, the only one
with access to `ScalarType::I8/I4` and `KernelOp::Dequantize`). This
crate re-exports the types.

| Item | Status | Notes |
|---|---|---|
| `enum QuantDtype { Int8, Int4 }` + `bits()` | **shipped** | maps internally to IR `Q8S`/`Q4S` |
| `enum Granularity` + `tile(rows, cols) -> Result<(usize, usize)>` | **shipped** | §3's table; invalid axis / size 0 / size > extent are loud `QuantError::Granularity` |
| `struct QuantizedMatrix` | **shipped** | device-resident record: codes + scale grid + logical shape. Accessors: `shape()`, `dtype()`, `granularity()`, `tile()`, `codes()`, `scales()` — the last three beyond the scope's list, so a checkpoint writer and the future fused consumer read the record without friends-access (§12 c) |
| `QuantizedMatrix::quantize(a, dtype, gran) -> Result<Self>` | **shipped** | host-side scan over a readback (max-abs per tile + `quantize_sym` + pack) then upload; rank-2 with nonzero extents only, loud otherwise |
| `QuantizedMatrix::from_parts(codes, [R, C], scales, gran) -> Result<Self>` | **shipped (delta — §12 c)** | the seam a checkpoint reader uses to assemble the record from loaded tensors; validation total and loud (`QuantError::Grid` on any codes/scales/grid disagreement). Also the constructor a future foreign-format importer targets (§2). |
| `QuantizedMatrix::dequantize() -> Result<Array<f32>>` | **shipped** | ONE device dispatch: code load at native stride (int8) or its nibble (int4) → scale load at the grid index → `KernelOp::Dequantize` (true scheme) → f32 store. Bitwise-equal to `dequantize_host` on every backend. |
| `enum QuantCodes { Int8(Array<i8>), Int4Packed(Array<u32>) }` | **shipped (delta — §12 c)** | the device codes payload, public so `from_parts`/`codes()` speak a real type |
| `enum HostCodes { Int8(Vec<i8>), Int4Packed(Vec<u32>) }` | **shipped (delta — §12 c)** | the host twins' typed payload (the scope said "over slices"; one type per packing beats two parallel slice conventions) |
| `quantize_host(w, rows, cols, dtype, gran) -> Result<(HostCodes, Vec<f32>)>` | **shipped** | the reference the device path is pinned against, and the engine `QuantizedMatrix::quantize` runs; fallible (§12 a) |
| `dequantize_host(codes, scales, rows, cols, gran) -> Result<Vec<f32>>` | **shipped** | the bitwise twin of the device kernel, and the engine of Mode A's universal load; fallible (§12 a) |
| `ArrayError::Quant(QuantError)` | **shipped** | the `ArrayError::Npy(NpyError)` wrapping pattern; taxonomy in §10 |

## 5. API surface — `quanta_nn::quant` (the checkpoint + layer face)

Subject: named checkpoint leaves and `Layer` composition (the
safetensors precedent). The module docs in `src/quant.rs` are the
format's **normative text** (§8 summarises).

| Item | Status | Notes |
|---|---|---|
| `enum QuantLeaf { F32(Array<f32>), Quantized(QuantizedMatrix) }` | **shipped** | the mixed-checkpoint vocabulary — norms/biases stay f32 beside quantized matrices; mixed files are the NORM |
| `quantize_named(leaves, policy) -> Result<Vec<(String, QuantLeaf)>>` with `policy: Fn(&str, &Array<f32>) -> Option<(QuantDtype, Granularity)>` | **shipped** | `None` keeps the leaf f32. No magic name-matching: the caller decides which leaves carry the weight mass |
| `save_named(&[(String, QuantLeaf)], metadata: Option<&HashMap<String, String>>) -> Result<Vec<u8>>` | **shipped** | writes §8's convention through the extended safetensors writer; user metadata rides along, and keys colliding with the convention (`quanta.quant`, `quant:*`) are loud errors (§12 c). Deterministic bytes (sorted metadata keys, entries in given order). |
| `load_named(gpu, &[u8]) -> Result<LoadedQuant>` | **shipped** | **Mode B**: quantized leaves arrive device-resident as `QuantizedMatrix`, plain leaves as `Array<f32>`; `LoadedQuant { leaves, metadata }` carries the USER metadata (format machinery stripped). int8 leaves on a backend without `Gpu::supports_narrow_int()` are a loud per-leaf refusal (§7); packed int4 rides every backend. |
| `load_named_f32(gpu, &[u8]) -> Result<LoadedSafetensors>` | **shipped** | **Mode A**: every quantized leaf host-dequantized (exact `scale · q`, bit-identical to the device kernel) before upload, under its own name at its logical shape — universal, WebGPU included. Returns the same record plain `safetensors::load_named` returns. |
| `load::<P: ParamTree<f32>>(gpu, &witness, &[u8]) -> Result<P>` | **shipped** | the Mode A tree form: matches by NAME with the `load_state` missing/extra/shape contract, so an existing f32 model definition loads a quantized checkpoint with zero code changes. Takes a `&P` witness like every tree loader in this crate (§12 c). |
| `FORMAT_VERSION: &str` | **shipped (delta — §12 c)** | the quantized-safetensors version this build reads and writes (`"1"`) |
| `QuantizedLinear::new(w: QuantizedMatrix, b: Option<Array<f32>>)` | **shipped** | **Mode B construction**: codes stay resident (1 byte / ½ byte per element between steps); each forward dequantizes into a scratch f32 the tape holds for the step |
| `QuantizedLinear::dequantized(&QuantizedMatrix, b) -> Result<Self>` | **shipped (delta — §12 c)** | **Mode A construction**: dequantize ONCE, hold the f32 weight — each forward then costs exactly what `Linear` costs |
| `QuantizedLinear: Layer<f32>` with `Params = ()` | **shipped** | frozen weights ARE configuration (decision D1; the established zero-param activation shape). `apply` = dequantize → `tape.var` → `matmul` (+ broadcast bias) — every step an existing proven citizen; the method body is the declared fused dequant-GEMM swap seam (§6). Gradients flow through `x` (harmless; none reach the codes). Tuple-stackable; `init` returns `()`. |
| Quantized attention / transformer blocks | **not planned — assembly, by design** | `MultiheadAttention`/`TransformerEncoderLayer` keep their f32 `Linear`s. The quantized-attention capability is served by ASSEMBLY: `QuantizedLinear` projections around `functional::sdpa_var` — all public pieces. PARITY = capabilities, not API mirroring; a generic-projection constructor later is additive ergonomics, not a reopen. |
| `examples/cookbook_quantized_inference.rs` | **shipped** | the runnable twin of the how-to (`docs/computation/how-to/quantized-checkpoints.md`): train → quantize by name (int8 per-channel + int4 grouped) → save → reload BOTH modes → forward; asserts the two modes agree bitwise and the output stays inside the propagated s/2 envelope |

## 6. The two execution modes — and the honest memory claim

| Mode | What happens | Backends | What it buys |
|---|---|---|---|
| **A — dequantize on load** (`load_named_f32` / tree `load`; or `QuantizedLinear::dequantized`) | codes → host dequant → f32 arrays; the model is an ordinary f32 model afterwards | **all four** (host-side; never touches narrow storage) | checkpoint size ÷4 / ÷8 (disk, distribution, load bandwidth); zero new runtime machinery; works under every existing layer including `Embedding` |
| **B — resident-quantized** (`load_named` + `QuantizedLinear::new`) | codes live on-device at 1 byte (int8) / ½ byte (int4) per element; each forward dequantizes per layer into a scratch f32 the tape holds for the step | CPU / Metal / Vulkan for int8 (`supports_narrow_int`); **all four incl. WebGPU for int4** (PackedU32 is core u32 storage) | steady-state resident memory between steps = codes only |

**The peak-memory truth, stated so the claim is never overreached:**
under the tape-based forward, a dequantized weight lifted via
`tape.var` lives until the tape drops — so DURING a forward, Mode B's
peak approaches the f32 model. Mode B's honest wins are checkpoint
size, upload volume, and steady-state residency; the *peak*-memory win
(never materialising f32 weights) belongs to the fused dequant-GEMM,
which the 084.1 quantization design assigns to quanta-blas. **The seam
is finished here**: `QuantizedLinear::apply` is the single swap site
(marked `INTEGRATOR SEAM` in source), and `QuantizedMatrix` is the
operand the fused kernel will consume — swapping the two-dispatch
spelling for the fused GEMM later is an internal perf substitution
with zero API/format/semantics change. That is what makes the deferral
legitimate under the governing principle.

## 7. Kernel story per backend

| Backend | int8 codes | int4 codes | Notes |
|---|---|---|---|
| CPU (software) | **full** — native 1-byte stride | **full** — PackedU32 | interpreter is the differential oracle; host reference is the bitwise twin |
| Metal | **full** — native stride | **full** | dequant kernel + existing f32 gemm |
| Vulkan (incl. lavapipe) | **full where 8-bit storage features advertise** (`supports_narrow_int`) | **full** (core SPIR-V) | |
| WebGPU | **NotSupported for resident codes** (the narrow-dtype ruling, not relitigated; Mode A is the complete answer there) | **full** — PackedU32 + the shipped WGSL `Dequantize` arm | the asymmetry is the honest capability truth, stated in the refusal and the docs |

No new IR ops, no emitter changes, no new capability query: the slice
composes `Load(I8/I4)` + `Dequantize` + `Store(F32)` — every arm
already shipped and op-matrix-pinned.

## 8. The checkpoint format — quanta quantized-safetensors

The **normative text is the module documentation of `src/quant.rs`**;
in brief: an ordinary safetensors file (valid for any third-party
inspector), one convention on top. A quantized leaf `x` is stored as
TWO tensors plus one metadata entry — and NO tensor named `x`
(ambiguity is structurally impossible): `x.q` (codes: `I8 [R, C]`, or
`U32 [R, ⌈C/8⌉]` packed), `x.qs` (scales: `F32 [⌈R/gr⌉, ⌈C/gc⌉]`), and
`__metadata__["quant:x"] = "<v>;gr=<gr>;gc=<gc>;rows=<R>;cols=<C>"`
with `<v>` ∈ {`q8s`, `q4s`}. `__metadata__["quanta.quant"] = "1"` is
the format version, required whenever a `quant:*` entry is present;
unknown versions are loud. Plain leaves keep their names and dtypes;
mixed files are the norm.

**Reserved, additive** (loud error today): an `x.qz` zero-points
tensor plus `;mode=affine` in the scheme string — the affine extension
will change no v1 symmetric file or reader.

Loader validation is TOTAL before anything uploads: orphan `quant:x`,
missing `x.q`/`x.qs`, name collisions, a tensor named `x` beside
`quant:x`, shape/grid/metadata disagreement, non-finite scales, and
orphan `I8`/`U32` tensors carrying no `quant:*` metadata — each a loud
per-leaf error. Save is deterministic. QNNS (`state.rs`) is untouched:
quantized checkpoints are safetensors-only — one format, no dialect
drift.

## 9. Correctness contract

The methodology splits at the one place information is actually lost:

1. **Everything after quantization is EXACT and bit-pinned.**
   `dequantize(q, s) = s · q` is one f32 multiply on an
   exactly-converted int — op-matrix-pinned bit-exact per backend.
   Therefore: device dequant ≡ host reference **bitwise** (per lane,
   every granularity, non-divisible edges, the long-ramp stride
   guard); `QuantizedLinear` output ≡ `Linear` fed the dequantized
   weight **bitwise** (same tape ops); Mode A ≡ Mode B forward
   **bitwise**; and the quantized forward is **bit-reproducible
   cross-backend** because dequant is pinned and the f32 stack already
   is.
2. **Quantization itself carries the only tolerance, and it is a
   THEOREM, not a fudge factor.** T9234 (Lean,
   `specs/verify/lean/Quanta/Dtype/QuantRoundTrip.lean`): for `s > 0`,
   `|s · round_te(x/s) − x| ≤ s/2`, exact on code multiples. T9235:
   max-abs scales keep every code in `[−hi, hi]` — the clamp never
   fires, code `−(hi+1)` is never produced — and the composed theorem
   carries the bound through the full clamped quantize. The f32
   statement is the empirical twin with the half-ulp slack stated
   in-code; the end-to-end gate derives its tolerance from the bound's
   propagation and is a *quality sanity gate*, not the oracle — the
   oracle is row 1's bitwise chain.
3. **Fixture story — no Python, and why that is honest here**: this
   format is OURS; the external formats are excluded claim boundaries
   (§2). Ground truth is hand-built spec-byte-exact
   quantized-safetensors bytes (the `reference_bytes` house pattern),
   `dtype.rs` (Lean/op-matrix-pinned) as the arithmetic reference, and
   a plain-safetensors read of the container proving third-party
   validity. If a foreign importer ever ships, ITS scope brings the
   pinned external fixtures.

## 10. Error taxonomy

`QuantError`, wrapped as `ArrayError::Quant(...)` at the array level
and surfaced through `AutogradError` at the nn level; the `NpyError`
message contract (self-contained, names the offender, states the
workaround). The whole taxonomy is declared in quanta-array so it
lives in one place; this crate's loader constructs the
checkpoint-facing variants with real leaf names.

| Variant | Covers |
|---|---|
| `Rank { shape }` | quantize/load of a leaf that is not rank-2 with nonzero extents |
| `Granularity { what }` | axis > 1, group size 0, group size > extent |
| `Grid { what }` | codes / scales / grid disagreement assembling a `QuantizedMatrix` from parts (the reader seam) — an as-built addition (§12 c) |
| `Format { leaf, what }` | §8 violations: orphan `quant:x`, missing `x.q`/`x.qs`, name collision, shape/grid/metadata mismatch, bad scheme string, unknown format version, reserved affine spellings, orphan code-dtype tensors, metadata-key collisions on save |
| `NotSupported { leaf, backend }` | Mode B int8 on a backend without narrow storage — names the leaf, the backend, and the dequantize-on-load workaround in prose (§12 b) |
| `Scale { leaf, tile, value }` | a non-finite scale read from a checkpoint (corrupt or hostile). Zero is LEGAL: an all-zero tile stores `s = 0` and dequantizes exactly |

Hostile-input posture inherits the safetensors parser's grade: every
length text-bounded, no allocation driven by an unvalidated size,
total validation before upload.

## 11. Test coverage + CI lanes

Homes: `crates/sci/quanta-array/tests/quant.rs`,
`crates/ml/quanta-nn/tests/quant.rs`; fixture bytes built in-test (no
committed binaries for a format we define).

| Layer | Status | Notes |
|---|---|---|
| Host arithmetic | **shipped** | quantize/pack against `dtype.rs` directly (shared fns — no re-spelling to drift); ties-to-even literals pinned; the low-nibble-first word contract; independent row packing with final-word padding; the all-zero-tile row |
| Device-vs-host bitwise | **shipped** | dequant kernel ≡ host reference per lane: both dtypes × all five granularity spellings × non-divisible `(R, C)` edges × multi-word int4 rows × a 512-element ramp (the recorded narrow-stride trap) |
| Round-trip bound | **shipped** | the s/2 bound per element across granularities (T9234's f32 twin), exactness on code multiples, `Group` of full extent ≡ its `PerChannel` twin |
| Format | **shipped** | spec-byte-exact save against hand-built reference bytes; save→load bitwise round-trip (codes, scales, user metadata); **a plain safetensors reader opens the container** (third-party validity pinned); one loud-error test per §10 row asserting the message carries the promised context |
| Layer + mode equivalence | **shipped** | `QuantizedLinear` ≡ `Linear`(dequantized w) bitwise in BOTH construction modes; Mode A ≡ Mode B forward bitwise; tuple-stack composition; bias-shape refusal |
| Capability gate | **shipped** | the refusal path is pinned as data (the gate takes the capability as an argument — software/Metal/Vulkan all report `true`, so no compiled suite can reach it live); the open side asserts the int8 Mode B load succeeds wherever the suites run |
| End to end | **shipped** | a TRAINED linear (short SGD run) → `quantize_named` → save → reload both modes → forwards bitwise-equal to each other and within the s/2-derived bound of the f32 original |
| CI | **shipped** | **companion-tests** (software lane) per-PR for both suites — no new lanes (the narrow-dtype placement verbatim); the backend suites re-run them on Metal/Vulkan as they do the rest of the crates |
| Lean | **shipped** | `Quanta/Dtype/QuantRoundTrip.lean` imported from `Quanta.lean` (the lake-import check rule); T9234–T9235, 0 sorries, 0 new axioms |

## 12. Corrections the integration taught

The scope was written against the design; the implementation was
integrated against the real crates. Where they differed, the layering
won, and the difference is recorded here.

**(a) The host twins are fallible.** `quantize_host` /
`dequantize_host` return `Result` — dimension, length, and grid checks
are part of their own contract, not their callers' (they are public
reference functions, not private helpers). On the loader path the
validation pass pins every length and the grid before decode, so the
`Err` arms are unreachable for a file that passed it — but they
PROPAGATE (`map_err`, never `unwrap`): the seam stays honest for
callers that are not the loader, and no proof obligation is silently
converted into a panic.

**(b) The capability-refusal wording lives in quanta-array and states
the workaround in prose.** The scope asked the Mode B int8 refusal to
"point at `load_f32`". The message is `QuantError::NotSupported`'s
`Display`, which lives in quanta-array — a crate BELOW this one that
cannot name `quant::load_named_f32` (layering, not oversight). As
shipped, the contract is: the leaf, the backend, the caps reason, and
the Mode A workaround in prose — *"load via the f32 path
(dequantize-on-load) instead"*. The tests pin the real,
correctly-layered message on both sides of the gate.

**(c) As-built deltas from the scope's API tables.** The array half's
additions were recorded in the array landing and are folded in here:

- `QuantizedMatrix::from_parts` — the reader seam: a checkpoint (or
  future foreign-format importer) assembles the record from
  already-resident tensors, validated loudly. The scope's table had no
  constructor besides `quantize`; without this seam the loader would
  need private access.
- `HostCodes` / `QuantCodes` — typed payload enums for the host twins
  and the device record (the scope said "over slices"); one type per
  packing beats parallel slice conventions, and it is what lets the
  twins be reused as Mode A's engine unchanged.
- Accessor additions: `tile()`, `codes()`, `scales()` on
  `QuantizedMatrix` (plus `bits()`/`dtype()` on the small enums) — a
  checkpoint writer and the future fused-GEMM consumer read the record
  through the public surface.
- `QuantError::Grid` — a sixth variant for from-parts disagreement;
  the scope's five-variant table had no home for "the parts don't
  agree" that wasn't a checkpoint-format error.

And on this crate's half:

- `QuantizedLinear::dequantized` — the Mode A construction (dequantize
  once, hold f32). The scope's table only spelled Mode B's `new`; the
  layer serves both modes, so both constructions ship.
- The tree loader takes a witness: `load(gpu, &witness, bytes)` — the
  `load_state`/`safetensors::load` house signature (a tree cannot be
  rebuilt from names alone).
- `save_named` accepts user metadata (`Option<&HashMap>`), with loud
  refusal of keys colliding with the convention's (`quanta.quant`,
  `quant:*`); `load_named` returns `LoadedQuant { leaves, metadata }`
  and `load_named_f32` returns the plain `LoadedSafetensors` record —
  user metadata round-trips, format machinery is stripped and
  re-synthesized on save.
- Loader hardening beyond the scope's list: the version key is
  REQUIRED whenever `quant:*` metadata is present; orphan `I8`/`U32`
  tensors with no `quant:*` metadata are loud (never silently treated
  as plain); a stored tile grid reads back as its canonical
  `Granularity` spelling (PerTensor ≻ PerChannel ≻ Group where they
  coincide — never changing any element's scale).
- `FORMAT_VERSION` is a public const.
- `examples/cookbook_quantized_inference.rs` landed with the docs
  closer (one step behind the code commits, same arc): the runnable
  twin of the how-to, exercising the full path — train, quantize by
  name, save, reload both modes, forward — and asserting the two
  modes agree bitwise. §11's end-to-end row remains the CI-side proof
  of the same path.

## 13. Explicitly out of scope — consolidated

| Item | Status | Reasoning |
|---|---|---|
| Quantization-aware training (fake-quant, STE) | **excluded (permanent)** | a different product entangled with the optimizer/tape; inference artifacts are produced elsewhere. The seam a future QAT would use already exists (`Tape::custom_vjp`); nothing here is reworked by it. |
| Calibration / static activation quant | **excluded** | requires activation statistics over data — a tooling product. The line: max-abs weight scaling is a pure function of the tensor and ships (§3); anything needing data does not. |
| Dynamic / per-token activation quant, W8A8 | **excluded (owned elsewhere)** | needs the int8×int8→i32 GEMM, which the 084.1 quantization design assigns to quanta-blas. When that pillar lands, W8A8 consumes the same `QuantizedMatrix` — additive capability, new slice. |
| Fused dequant-GEMM (the peak-memory win) | **deferred (seam finished — §6)** | quanta-blas pillar per the ratified 084.1 design; `QuantizedLinear::apply` is the declared swap site; API/format/semantics final here. |
| Affine / zero-points | **excluded (reserved on all three layers)** | IR carries the operand + mode arm; format reserves `x.qz` + `mode=affine`; host API simply lacks a constructor. Lands with the first affine consumer (the GPTQ-class importer). |
| GPTQ / AWQ / GGUF / bitsandbytes / compressed-tensors import | **excluded (claim boundaries, seams named — §2)** | each a future *reader* constructing the same records; loud errors today name the boundary. |
| FP8-scaled checkpoints | **deferred (named — §2)** | additive codes variant when a consumer exists. |
| int8 resident codes on WebGPU | **excluded (ratified)** | inherits the narrow-dtype ruling verbatim — u32-slot inflation (4×) betrays the density contract; Mode A is WebGPU's complete answer, and int4-resident works there natively. The refusal is loud and names the workaround. |
| Quantized `Embedding` / `Conv2d` resident forms | **not planned (additive)** | Mode A already RUNS checkpoints with quantized embedding tables (any rank-2 leaf dequantizes on load); resident forms are new module forms over the same `QuantizedMatrix` — additive, no format/API change. `QuantizedLinear` covers the weight mass that matters (projections + FFN + lm-head). |
| Rank > 2 quantized leaves | **excluded (loud error)** | the weight population is matrices; a rank-4 conv form arrives with the conv resident form above, reusing the same grid over reshaped views. |
| KV-cache quantization | **excluded (owned elsewhere)** | inference-serving lane — PARITY already defers paged-KV-class serving optimizations. |
| bf16/f16 as "quantization" | **already shipped elsewhere** | §1; resident bf16 compute is 084.1's remaining scope. |
| Perf benchmarks / quanta-bench lane | **not planned** | no throughput claim is made (§6 states the memory truth instead); nothing to gate. The bit-reproducibility claim IS gated (§11). |

# npy / npz interop — declared surface (084.8, SHIPPED)

This file is the interop sub-feature's completeness contract, in the
`quanta-nn/PARITY.md` shape: every row is either **shipped** or a
**documented deferral / exclusion with its reasoning**. Nothing on the
reference surface (what `np.save` / `np.load` / `np.savez` do) is
silently absent. It began as the ratified scope document; the statuses
below are the as-built truth.

**RATIFICATION RECORD (owner decision).** The governing principle: a
feature is implemented so it never needs to be REOPENED to extend or
complete it — a deferral is legitimate only when the exclusion or a
prepared seam IS the finished answer (blocked on a missing model, a
security stance, a genuinely separate feature), never when it schedules
a known return visit. Applied here:

1. **Two sequenced, individually complete increments.** Increment 1 =
   the narrow-dtype `ArrayScalar` extension (`u8/u16/i8/i16` — its own
   scope doc, its own gates: ufuncs, reduces, astype, op-matrix
   differential coverage, the Metal narrow-stride ground). Increment 2
   = npy/npz, implemented ONCE over the full ten-type surface —
   never reopened. The original draft's "narrow ints deferred on
   missing `ArrayScalar` impls" was a guaranteed return visit, not a
   deferral.
2. **Big-endian load pulled IN** (byteswap-on-load is trivial;
   deferring it was a scheduled return visit).
3. **Deflate READ pulled IN** (`savez_compressed` files are
   numpy-written files; a reader claiming "reads what numpy writes"
   that refuses them would be reopened). Deflate WRITE stays out on
   different grounds: stored output is fully valid npz everything
   reads — there is nothing to come back for.
4. **ZIP64 stays out, with the claim boundary stated**: this reader's
   claim is "reads what numpy writes *at the scales this stack
   exchanges*"; numpy emits ZIP64 only above 4 GiB, safetensors is the
   declared big-weights lane, and the markers are detected loudly.
5. The remaining exclusions stand because exclusion is the complete
   answer: pickle/objects (permanent security stance), bf16 (the
   format has no descriptor), complex (stack-wide model gap), empty
   arrays (shape-model boundary), mmap-load (separate feature behind
   an already-prepared seam).

Binding house rules, applied throughout:

- **Zero external dependencies.** The npy header parser and the ZIP container
  reader/writer are hand-rolled for exactly the grammar they need — the
  `quanta-nn/src/safetensors.rs` precedent (minimal recursive-descent parser,
  strict about keys, correct about escapes, byte-offset error messages).
- **Community-complete.** The full interchange surface is declared here with
  explicit deferrals — the `quanta-blas` pattern, where general (non-symmetric)
  eig is a *named, reasoned* gap, not a silent one.
- **Open-source generality.** The API is plain-array I/O for any consumer.
  dija / thiaba / ai_project appear only as motivation; nothing in the shape
  is specific to them.

## 1. Placement — a module inside `quanta-array`

`quanta_array::npy` + `quanta_array::npz`, two public modules in
`crates/sci/quanta-array/src/` (shared private header codec in
`npy_codec.rs`, private `zip` module with `zip/inflate.rs`). Not a new
crate, not in quanta-nn.

Reasoning from precedent: safetensors lives inside quanta-nn because its
subject is quanta-nn's vocabulary — `ParamTree`s, named checkpoint leaves.
npy's subject is the plain `Array<T>`: dtype, shape, row-major bytes —
quanta-array's vocabulary exactly (`from_slice` / `to_vec` are the whole
runtime story). quanta-nn's own PARITY row already points here: *"npy / npz
interop — deferred to step 084.8 — plain-array IO belongs beside
quanta-array."* A separate `quanta-io` crate would ship two small modules and
a version number with no independent audience — the companion-crate layout
(`crates/sci/`) adds crates for new *audiences* (blas, fft, rand), not new
modules. No cargo feature gate: zero deps, small, always-on, pure host code
(no kernels, no backend variance).

## 2. Format ground truth

- **npy v1.0**: magic `\x93NUMPY`, version bytes `\x01\x00`, u16-LE header
  length, then an ASCII Python-dict-literal header
  `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }` — exactly
  those three keys, space-padded and `\n`-terminated so the data section
  starts 64-byte aligned (modern numpy; readers must trust the length field,
  not the alignment). Raw element bytes follow, C-order unless
  `fortran_order: True`.
- **npy v2.0**: identical except a u32-LE header length (headers ≥ 64 KiB).
  **v3.0**: header is utf-8 instead of latin-1 — only observable with
  structured field names, which we reject anyway.
- **npz**: an ordinary ZIP archive, one `<name>.npy` entry per array.
  `np.savez` writes **method 0 (stored — no compression)**;
  `np.savez_compressed` writes **method 8 (deflate)**. This asymmetry is the
  scope line for the hand-rolled ZIP layer (§5).

## 3. API surface

| Item | Status | Notes |
|---|---|---|
| `npy::save<T: NpyScalar>(&Array<T>) -> Result<Vec<u8>, ArrayError>` | **shipped** | One array → npy v1.0 bytes (auto v2.0 only if the header exceeds u16, numpy's own rule). Writes `<`-endian element bytes, `fortran_order: False`, always. Accepts ANY view — strided / transposed / broadcast / narrowed views serialize their **logical row-major content** (the existing `to_vec` gather; same route safetensors takes). `NpyScalar` is a sealed marker over the ten `ArrayScalar` types (§4). |
| `npy::load<T: NpyScalar>(gpu, &[u8]) -> Result<Array<T>, ArrayError>` | **shipped** | Typed load, the common path (you know what you saved). File descr must match `T` exactly, with one documented widening: `load::<f32>` also accepts `<f2` (f16 upconverts exactly — the safetensors precedent, real files ship f16). Any other descr is a loud `Dtype` / `DtypeMismatch` error naming both sides. |
| `npy::load_dyn(gpu, &[u8]) -> Result<NpyArray, ArrayError>` | **shipped** | Dtype-preserving load for "inspect what Python wrote": returns the `NpyArray` enum (§4) matching the file's descr. `<f2` upconverts to the `F32` variant (no f16 Array exists to preserve into). |
| `pub enum NpyArray { F32(Array<f32>), F64(Array<f64>), I32(Array<i32>), U32(Array<u32>), I64(Array<i64>), U64(Array<u64>), U8(Array<u8>), I8(Array<i8>), U16(Array<u16>), I16(Array<i16>) }` | **shipped** | The dynamic-dtype vocabulary, used in both directions (npz entries are mixed-dtype in the wild — labels u64 next to weights f32). Ten variants, the narrow types first-class from day one. `From<Array<T>>` impls for the ten types; `dtype()` (canonical descr), `shape()` accessors; `TryFrom<NpyArray> for Array<T>` with a loud dtype error. Building one is cheap (`Array` is Arc-backed). |
| `npy::header(&[u8]) -> Result<NpyHeader, ArrayError>` | **shipped** | Header introspection without touching the data section: `{ descr: String, fortran_order: bool, shape: Vec<usize>, version: (u8, u8), data_offset: usize }`. Lets a caller answer "what is this file?" before committing to a load — and is the natural probe point for the future mmap path (§9). |
| `npz::save_named(&[(String, NpyArray)]) -> Result<Vec<u8>, ArrayError>` | **shipped** | Multi-array archive, ZIP **stored** (method 0) — byte-for-byte the container class `np.savez` writes. Entry names get `.npy` appended (numpy's convention); duplicate names are a loud error; caller order is preserved. |
| `npz::load_named(gpu, &[u8]) -> Result<Vec<(String, NpyArray)>, ArrayError>` | **shipped** | Reads stored and deflate entries via the central directory (§5), strips the `.npy` suffix, returns archive order. A non-`.npy` entry means the ZIP wasn't written as an npz — loud error naming the entry, rather than a silent skip. An entry whose npy payload fails to decode is reported with the entry name (the inner fault's message follows it). |
| File-path wrappers (`save_file` / `load_file`) | **not planned** | The house pattern is bytes-level (safetensors shipped bytes-only and it cost nobody anything): `std::fs::write(path, npy::save(&a)?)` is the documented one-liner. Streams likewise — `&[u8]` in, `Vec<u8>` out. |

## 4. Dtype matrix

`ArrayScalar` = `f32 / f64 / i32 / u32 / i64 / u64` plus
`u8 / i8 / u16 / i16` (`crates/sci/quanta-array/src/scalar.rs`). The
matrix follows one principle: **typed loads are exact-width, with float
upconversion as the only widening** — the line safetensors drew (F32
exact; F16/BF16 upconvert; everything else a loud error naming the
tensor).

| descr | maps to | save | load | Notes |
|---|---|---|---|---|
| `<f4` | `f32` | **shipped** | **shipped** (exact) | |
| `<f8` | `f64` | **shipped** | **shipped** (exact) | |
| `<i4` | `i32` | **shipped** | **shipped** (exact) | |
| `<u4` | `u32` | **shipped** | **shipped** (exact) | |
| `<i8` | `i64` | **shipped** | **shipped** (exact) | |
| `<u8` | `u64` | **shipped** | **shipped** (exact) | |
| `<f2` (f16) | → `f32` | — | **shipped** (upconvert) | Exact embedding (f16 ⊂ f32), signs / subnormals / inf / NaN preserved — the decoder mirrors `f16_to_f32` in safetensors.rs. Never written: we have no f16 `Array` to save. |
| `\|u1` `\|i1` `<u2` `<i2` (narrow ints) | `u8/i8/u16/i16` | **shipped** | **shipped** (exact) | Exact from day one (increment 1, the narrow-dtype `ArrayScalar` extension, landed first). Deliberately never widened to u32/i32: a silent 4× memory inflation with a dtype-changing round-trip would have been worse than sequencing the increments. |
| `\|b1` (bool) | → `Array<u8>` | — | **shipped** | Loads as `Array<u8>` with every byte validated 0/1 (a loud error otherwise — numpy never writes other values; a file that does is corrupt). Never written: quanta-array has no bool array type, and saving a `u8` array writes `\|u1` honestly rather than guessing boolness. |
| `>f4` `>i4` … (big-endian) | same as `<` | — | **shipped** (byteswap-on-load) | Element-width byteswap into native order at load, all types. numpy writes `<` on every mainstream platform, so this path is rare — but deferring a trivial path is a scheduled return visit (ratification record #2). Saves always write `<`. `=` never appears in files numpy writes and is rejected as malformed. |
| bf16 | — | — | **excluded** | No standard npy descr exists for bf16 (numpy proper cannot represent it without third-party dtype packages). **safetensors is the bf16 interchange** and already ships in quanta-nn — pointing users there is the honest answer, not inventing a descr. |
| `<c8` `<c16` (complex) | — | — | **deferred** | No complex element type exists anywhere in the stack — the same departure-from-the-real-surface reasoning that made quanta-blas defer general eig. Revisit together with any quanta-wide complex story. |
| `\|O` (object / pickle), `S*` / `U*` (strings), structured descrs | — | — | **excluded (permanent)** | Object arrays embed pickle — arbitrary code execution in a file parser that must treat input as hostile. Strings and record dtypes have no Array representation and no consumer. These are exclusions, not deferrals: loud `Dtype` errors, never revisited as-is. |

## 5. npz container — how far the hand-rolled ZIP goes

| Item | Status | Notes |
|---|---|---|
| Write: stored entries (method 0) + central directory + EOCD | **shipped** | Exactly what `np.savez` produces. Local headers carry real sizes and CRC-32 (we buffer each entry — no data descriptors). CRC-32 (IEEE) is hand-rolled, table-based, tested against the classic `"123456789"` → `0xCBF43926` vector. |
| Write: deterministic bytes | **shipped** | Fixed DOS timestamp (1980-01-01), zeroed external attrs, entry order = caller order → identical input yields identical archive bytes. (numpy itself stamps wall-clock mtimes; we choose reproducibility — the same instinct as safetensors' sorted `__metadata__` keys.) |
| Read: central-directory-driven | **shipped** | The EOCD → central directory walk is authoritative for names, offsets, sizes, and CRCs; local headers are validated against it. This also sidesteps streaming-mode (bit-3 data-descriptor) entries for free — the CD always has real sizes. CRC verified on every read; mismatch is a loud per-entry error. |
| Read: deflate entries (method 8) | **shipped** (hand-rolled inflate) | `np.savez_compressed` files are numpy-written files — refusing them would get reopened (ratification record #3). RFC 1951 inflate, hand-rolled, zero-dep: fixed+dynamic Huffman, stored blocks, window copies; hostile-input-grade (every length/distance bounds-checked, output capped at the CD's declared uncompressed size — the zip-bomb guard), CRC-verified after inflation like every stored entry. Tested against RFC edge vectors and truncation fuzz; numpy-generated `savez_compressed` fixture tests arm when the fixtures land (§8). |
| Write: deflate | **not planned** | numpy's own default (`np.savez`) is stored; every reader accepts stored. Compressing our writes buys nothing but the inflate liability on our own files. |
| ZIP64 (≥ 4 GiB archives / entries, > 65 535 entries) | **deferred (loud error, claim boundary)** | The reader's CLAIM is "reads what numpy writes at the scales this stack exchanges" (ratification record #4): numpy emits ZIP64 only above 4 GiB, and safetensors is the declared big-weights lane. ZIP64 markers (`0xFFFFFFFF` sentinels, ZIP64 EOCD locator, the 0x0001 extra field) are detected and refused loudly rather than misparsed. |
| Foreign-ZIP tolerance (extra fields, comments) | **shipped (read)** | Extra fields and archive comments are skipped per the spec (numpy's zipfile emits them in some versions); anything structurally required is validated. We parse the container numpy writes, strictly, and tolerate only what the ZIP spec forces us to. |

## 6. Shape and layout semantics

| Item | Status | Notes |
|---|---|---|
| Row-major mapping | **shipped** | npy C-order ≡ `Layout::row_major` — a contiguous load is `from_slice(gpu, data, &shape)` verbatim; a save of any view is its `to_vec()` logical row-major gather. No stride metadata exists in npy; none is needed. |
| `fortran_order: True` on load | **shipped** | **Load-with-transpose, not reject.** F-order bytes of shape `(a, b, c)` are host-permuted into logical row-major before upload; the caller always receives a row-major contiguous `Array` of shape `(a, b, c)`. One host-side pass at load time; scipy and Fortran-adjacent tools really do emit these, and "valid file, we refuse" is a wall the workaround for which (transpose in Python) is exactly the work we can do ourselves. Saves always write `fortran_order: False`. |
| 0-d arrays (shape `()`) | **shipped** | Rank-0 shapes are first-class in `quanta-tensor` (empty extent list, linear size 1). Save writes `'shape': ()`, load of `()` produces a rank-0 `Array`. Round-trips exactly. 1-tuples write numpy's `(3,)` trailing-comma form. |
| Empty arrays (any 0 extent, e.g. `(0, 4)`) | **excluded (shape-model)** | `quanta-tensor` rejects zero extents by design (`ShapeError::ZeroExtent` — "shapes describe data that exists"). Loading one is a loud, *specific* npy error naming the shape and the rule — not a bare `Shape` passthrough. Saving one is unrepresentable (no such `Array` can be constructed). Revisit only if quanta-array ever admits zero-size axes; that is an array-model decision, not an interop one. |
| Header versions | **shipped** | Read: v1.0, v2.0, v3.0 accepted (one strict dict grammar covers all three — v3's utf-8 delta is invisible without structured names, which we reject). Anything else is a loud `Version` error. Write: v1.0, auto-upgrading to v2.0 only on u16 header overflow — numpy's own rule. |
| Alignment on write | **shipped** | Header space-padded so the data section starts 64-byte aligned, matching modern numpy — and deliberately mmap-friendly for §9's future hook. Read never assumes alignment; the length field is the truth. |

## 7. Error taxonomy

House style: `ArrayError` is one flat enum that wraps sub-enums from below
(`Layout(LayoutError)`, `Shape(ShapeError)`, `Gpu(QuantaError)`). Interop
follows the same wrapping shape — a new variant, not stringly errors:

```
ArrayError::Npy(NpyError)
```

| `NpyError` variant | Covers | Message contract |
|---|---|---|
| `Magic` | not an npy file / truncated before the version bytes | first bytes shown |
| `Version { major, minor }` | unrecognized version pair | versions we accept |
| `Header { at, what }` | dict-grammar violation, header length overrunning the buffer, unknown / missing / duplicate keys | byte offset, like the safetensors parser's "at byte N" |
| `Dtype { descr }` | every unsupported-descr row of §4 | the descr, the supported list, and the specific reason (safetensors-for-bf16, pickle exclusion, complex model gap) |
| `ByteOrder { descr }` | malformed `=` byte-order marks only (`>` files byteswap-load per §4) | names the malformed descr |
| `BoolValue { at }` | a `\|b1` byte outside {0, 1} | byte offset — numpy never writes these; the file is corrupt |
| `DtypeMismatch { file, requested }` | typed `load::<T>` against a different (but supported) descr; the `TryFrom<NpyArray>` unwrap against a different variant | both dtypes; "use `load_dyn` to take the file's dtype" |
| `EmptyShape { shape }` | any 0 extent (§6) | the shape and the shape-model rule |
| `DataLength { expected, got }` | data section ≠ element-count × width | both byte counts |
| `Zip { entry, what }` | container faults: bad EOCD/CD, CRC mismatch, local/CD disagreement, duplicate or non-`.npy` names, ZIP64 markers, corrupt deflate streams (bad Huffman tables, out-of-window copies, output-size overrun); also an npz entry whose npy payload fails to decode | entry name always present where one exists |

Every error is loud, names the offending entry/offset, and where a workaround
exists the message states it — the "loud error naming the tensor" contract
safetensors set. Parsing is hostile-input-grade throughout: every length field
is bounds-checked before use (the safetensors 100 MB-header-cap instinct), no
allocation is driven by an unvalidated size.

## 8. Test coverage

Home: `crates/sci/quanta-array/tests/npy_format.rs` (format substrate),
`tests/npy_typed.rs` + `tests/npz_typed.rs` (typed layer), plus
`tests/fixtures/npy/` for checked-in binaries.

| Layer | Status | Notes |
|---|---|---|
| Spec-byte-exact, hand-built | **shipped** | Tests construct reference npy bytes by hand from the spec — magic, version, padded header, LE data — and assert `save()` output equals them byte-for-byte, and that loading them reproduces the values (f32 / u8 / i16 saves; f16 / big-endian / Fortran / bool loads through unpadded headers, proving the length field is trusted over alignment). Same for a minimal hand-built stored ZIP and hand-assembled RFC 1951 streams. |
| numpy-generated fixtures, checked in | **pending (maintainer step)** | Small real files written by actual numpy, committed as the interchange ground truth — the inventory and pinned provenance live in `tests/fixtures/npy/gen_fixtures.py` (CI never runs Python; the committed bytes are the contract). The consuming tests are WRITTEN and skip loudly (`SKIPPED: numpy fixture … absent` on stderr) until the fixtures are committed, then arm automatically. |
| Round-trip properties | **shipped** | save→load bit-identical for all ten dtypes (float specials at the bit level); a permuted/strided/broadcast view saves identically to its contiguous copy; rank-0; npz mixed-dtype entry-order and name round-trip (unicode names included); deterministic bytes (two saves of the same input are equal — the fixed-timestamp guarantee). |
| Error-path coverage | **shipped** | One test per §7 row, asserting the *message* carries the promised context (entry name, descr, workaround). Truncation fuzz at every length boundary (magic, header len, header, data, EOCD, CD, deflate streams). CRC vector test. |
| CI lane | **shipped** | **companion-tests** (`.github/workflows/ci.yml`, `cargo test -p quanta-array --features software`) — the code is pure host parsing plus `from_slice`/`to_vec`, backend-invariant by construction, so the GPU-less software-executor lane is the right and sufficient gate; the nightly full lanes re-run the suite on real backends as they already do for the rest of the crate. No new lane. |

## 9. Deferrals and exclusions — consolidated

Rows above, gathered so the gaps are checkable in one place:

| Item | Status | Unlock / reasoning |
|---|---|---|
| Deflate read (`savez_compressed`) | **shipped** (ratified in) | hand-rolled RFC 1951 inflate, §5 |
| Deflate write | **not planned** | numpy's own default is stored; our stored output is fully valid npz — nothing to come back for |
| ZIP64 | **deferred (claim boundary)** | the claim is "what numpy writes at the scales this stack exchanges"; loud on markers; multi-GiB archives are safetensors' lane |
| Big-endian load | **shipped** (ratified in) | byteswap-on-load, §4 |
| Narrow ints + bool | **shipped** | increment 1 (the narrow-dtype `ArrayScalar` extension) landed first; the rows ship exact |
| Complex dtypes | **deferred** | no complex element type stack-wide (the quanta-blas general-eig reasoning) |
| bf16 | **excluded** | no standard npy descr; safetensors covers bf16 interchange |
| Object / pickled / string / structured dtypes | **excluded (permanent)** | pickle = code execution; hostile-input posture |
| Empty (0-extent) arrays | **excluded (shape-model)** | `quanta-tensor` rejects zero extents by design |
| Memory-mapped load (`np.load(mmap_mode=…)`) | **deferred (hook named)** | The zero-copy import machinery already exists: `HostField` / `Gpu::field_from_host` (step 094) adopts a host allocation as a device-visible field without copying. An mmap'd npy is precisely such an allocation, and §6's 64-byte data alignment on write keeps our own files eligible. The future shape: `npy::header` finds the data offset, the mapped data section becomes a `HostField`, `Array::from_field` wraps it. Deferred because lifetime/borrow design for a file-backed mapping deserves its own slice — but the seam is fully prepared, nothing here will need rework. |
| Appending / in-place npz update | **not planned** | numpy has none either; archives are rewrite-whole |
| File-path / stream wrappers | **not planned** | bytes-level API, the safetensors house pattern; `std::fs` one-liners documented |

/-
Mixed-precision GEMM — Lean formalisation of the `quanta-blas` mixed-dtype
`gemm_mixed` numerical contract (this increment: bf16 inputs).

A mixed-precision GEMM stores its A and B operands in a narrow type (here
bf16), loads each element converting **to f32**, and accumulates the inner
product in f32. So per output entry it computes

  C'[m,n] = α · dotRounded(bf16 A_row, bf16 B_col) + β · C[m,n]

where `bf16 ·` is the round-to-nearest-bf16 quantisation applied elementwise
on load. This is the standard f32-accumulate mixed-precision path.

The error therefore splits into two independent pieces:

  1. **The f32 GEMM error** over the bf16-rounded inputs — already proven,
     `Quanta.Blas.gemmEntry_error_decomp` applied to the quantised lists.
  2. **The input-quantisation error** — how far `α·dot(bf16 a, bf16 b)` is
     from the intended `α·dot(a, b)`, from rounding each input to bf16.

This file models bf16 input rounding with the same relative-error model the
f32 path uses (Higham §2.2), with bf16's own unit roundoff `bf16Unit`
(`2⁻⁸` for bf16's 7-bit mantissa, vs `2⁻²⁴` for f32). The bit-exact storage
round-trip itself is `Quanta.Dtype.Bf16.pack_unpack`; here we capture the
numeric effect of the f32→bf16 rounding step that precedes storage. The sole
new trust assumption is `bf16_rounding_model`, the bf16 analogue of the f32
`rounding_model`.
-/

import Quanta.Blas.Gemm
import Quanta.Dtype.Bf16

namespace Quanta.Blas

open scoped BigOperators

/-- bf16 unit roundoff `u_bf16` (abstract; `2⁻⁸` for bf16's 7-bit mantissa).
    Non-negative — a declaration, not a trust obligation, exactly as
    `unitRoundoff`. -/
axiom bf16Unit : ℝ

/-- Round a real to the nearest bf16-representable value. Opaque symbol;
    behaviour pinned by `bf16_rounding_model`. -/
axiom bf16Round (v : ℝ) : ℝ

/-- **The bf16 rounding model — this file's sole new trust assumption.** The
    bf16 analogue of the f32 `rounding_model`: `u_bf16 ≥ 0`, and rounding a
    value to bf16 returns the exact value times `(1 + δ)` with `|δ| ≤ u_bf16`.
    The f32→bf16 conversion in `quanta_ir::dtype::f32_to_bf16`
    (round-to-nearest-even) realises this; the bit-level round-trip is
    `Quanta.Dtype.Bf16.pack_unpack`. -/
axiom bf16_rounding_model :
    0 ≤ bf16Unit ∧
    ∀ v : ℝ, ∃ δ : ℝ, |δ| ≤ bf16Unit ∧ bf16Round v = v * (1 + δ)

/-- The bf16 unit roundoff is non-negative. -/
theorem bf16Unit_nonneg : 0 ≤ bf16Unit := bf16_rounding_model.1

/-- Per-value bf16 rounding spec. -/
theorem bf16Round_spec (v : ℝ) :
    ∃ δ : ℝ, |δ| ≤ bf16Unit ∧ bf16Round v = v * (1 + δ) :=
  bf16_rounding_model.2 v

/-- A single bf16 rounding has absolute forward error at most `u_bf16·|v|`. -/
theorem bf16Round_error (v : ℝ) :
    |bf16Round v - v| ≤ bf16Unit * |v| := by
  obtain ⟨δ, hδ, hv⟩ := bf16Round_spec v
  rw [hv]
  have : v * (1 + δ) - v = v * δ := by ring
  rw [this, abs_mul, mul_comm]
  exact mul_le_mul_of_nonneg_right hδ (abs_nonneg v)

/-- Elementwise bf16 quantisation of a list — what the kernel applies on
    load before the f32 inner product. -/
noncomputable def bf16List (xs : List ℝ) : List ℝ :=
  xs.map bf16Round

/-- bf16 quantisation preserves length. -/
theorem bf16List_length (xs : List ℝ) : (bf16List xs).length = xs.length := by
  simp [bf16List]

/-- Exact intended mixed-bf16 gemm entry: the same real-arithmetic
    `gemmEntry` the f32 path targets — bf16 is an implementation detail of
    *how* the entry is computed, not what it means. -/
def gemmEntryMixedBf16 (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntry α β a b c

/-- The computed mixed-bf16 gemm entry: the f32 rounded gemm entry, but over
    the **bf16-quantised** input lists (each A/B element rounded to bf16 on
    load, then accumulated in f32). -/
noncomputable def gemmEntryMixedBf16Rounded (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β (bf16List a) (bf16List b) c

/-- **Generic narrow-input entry error split.** For *any* elementwise input
    rounding `qa : ℝ → ℝ` applied to A and `qb` to B (here both the same
    narrow quantiser), the forward error of the narrow-input, f32-accumulate
    gemm entry against the intended real entry is at most the f32 gemm-entry
    error (over the quantised inputs, supplied verbatim by
    `gemmEntry_error_decomp`) plus the input-quantisation error
    `|α · (dot(qa, qb) − dot a b)|`. The proof is the triangle split at the
    quantised inner product; it never touches the rounding model, so it holds
    for bf16, f16, fp8 — any narrow input dtype. -/
theorem gemmEntry_narrow_error_split (α β : ℝ) (a' b' a b : List ℝ) (c : ℝ) :
    |gemmEntryRounded α β a' b' c - gemmEntry α β a b c|
      ≤ |gemmEntryRounded α β a' b' c - gemmEntry α β a' b' c|
        + |α * dot a' b' - α * dot a b| := by
  -- pivot on the exact entry over the quantised inputs (Eb).
  set Eb := gemmEntry α β a' b' c with hEb
  have hsplit :
      |gemmEntryRounded α β a' b' c - gemmEntry α β a b c|
        ≤ |gemmEntryRounded α β a' b' c - Eb| + |Eb - gemmEntry α β a b c| := by
    have := abs_add (gemmEntryRounded α β a' b' c - Eb) (Eb - gemmEntry α β a b c)
    have he : (gemmEntryRounded α β a' b' c - Eb) + (Eb - gemmEntry α β a b c)
        = gemmEntryRounded α β a' b' c - gemmEntry α β a b c := by ring
    rwa [he] at this
  have hbeta : Eb - gemmEntry α β a b c = α * dot a' b' - α * dot a b := by
    rw [hEb]; unfold gemmEntry; ring
  rw [hbeta] at hsplit
  exact hsplit

/-- **Mixed-bf16 entry error splits into f32-GEMM error + input-quantisation
    error.** The bf16 instance of `gemmEntry_narrow_error_split`: isolates the
    *already-proven* f32 numerics from the single new bf16-quantisation term,
    so adding a narrow dtype reuses the GEMM proof rather than redoing it. -/
theorem gemmEntryMixedBf16_error_split (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    |gemmEntryMixedBf16Rounded α β a b c - gemmEntryMixedBf16 α β a b c|
      ≤ |gemmEntryRounded α β (bf16List a) (bf16List b) c
            - gemmEntry α β (bf16List a) (bf16List b) c|
        + |α * dot (bf16List a) (bf16List b) - α * dot a b| := by
  unfold gemmEntryMixedBf16Rounded gemmEntryMixedBf16
  exact gemmEntry_narrow_error_split α β (bf16List a) (bf16List b) a b c

-- ── f16 (IEEE half) — the same split, its own unit roundoff ──────────────

/-- f16 unit roundoff `u_f16` (abstract; `2⁻¹¹` for f16's 10-bit mantissa). -/
axiom f16Unit : ℝ

/-- Round a real to the nearest f16-representable value. -/
axiom f16Round (v : ℝ) : ℝ

/-- **The f16 rounding model.** The f16 analogue of the f32 `rounding_model`,
    realised by `quanta_ir::dtype::f32_to_f16`. -/
axiom f16_rounding_model :
    0 ≤ f16Unit ∧
    ∀ v : ℝ, ∃ δ : ℝ, |δ| ≤ f16Unit ∧ f16Round v = v * (1 + δ)

/-- The f16 unit roundoff is non-negative. -/
theorem f16Unit_nonneg : 0 ≤ f16Unit := f16_rounding_model.1

/-- A single f16 rounding has absolute forward error at most `u_f16·|v|`. -/
theorem f16Round_error (v : ℝ) : |f16Round v - v| ≤ f16Unit * |v| := by
  obtain ⟨δ, hδ, hv⟩ := f16_rounding_model.2 v
  rw [hv]
  have : v * (1 + δ) - v = v * δ := by ring
  rw [this, abs_mul, mul_comm]
  exact mul_le_mul_of_nonneg_right hδ (abs_nonneg v)

/-- Elementwise f16 quantisation of a list. -/
noncomputable def f16List (xs : List ℝ) : List ℝ :=
  xs.map f16Round

/-- f16 quantisation preserves length. -/
theorem f16List_length (xs : List ℝ) : (f16List xs).length = xs.length := by
  simp [f16List]

/-- Exact intended mixed-f16 gemm entry (the real-arithmetic `gemmEntry`). -/
def gemmEntryMixedF16 (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntry α β a b c

/-- The computed mixed-f16 gemm entry: f32 rounded entry over f16-quantised
    inputs. -/
noncomputable def gemmEntryMixedF16Rounded (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β (f16List a) (f16List b) c

/-- **Mixed-f16 entry error split** — the f16 instance of
    `gemmEntry_narrow_error_split`. -/
theorem gemmEntryMixedF16_error_split (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    |gemmEntryMixedF16Rounded α β a b c - gemmEntryMixedF16 α β a b c|
      ≤ |gemmEntryRounded α β (f16List a) (f16List b) c
            - gemmEntry α β (f16List a) (f16List b) c|
        + |α * dot (f16List a) (f16List b) - α * dot a b| := by
  unfold gemmEntryMixedF16Rounded gemmEntryMixedF16
  exact gemmEntry_narrow_error_split α β (f16List a) (f16List b) a b c

-- ── fp8 E5M2 — the same split, its own unit roundoff ─────────────────────

/-- fp8 E5M2 unit roundoff (abstract; `2⁻³` for the 2-bit mantissa). -/
axiom fp8e5m2Unit : ℝ

/-- Round a real to the nearest fp8 E5M2 value. -/
axiom fp8e5m2Round (v : ℝ) : ℝ

/-- **The fp8 E5M2 rounding model**, realised by
    `quanta_ir::dtype::f32_to_fp8 _ 5 2`. -/
axiom fp8e5m2_rounding_model :
    0 ≤ fp8e5m2Unit ∧
    ∀ v : ℝ, ∃ δ : ℝ, |δ| ≤ fp8e5m2Unit ∧ fp8e5m2Round v = v * (1 + δ)

/-- The fp8 E5M2 unit roundoff is non-negative. -/
theorem fp8e5m2Unit_nonneg : 0 ≤ fp8e5m2Unit := fp8e5m2_rounding_model.1

/-- Elementwise fp8 E5M2 quantisation of a list. -/
noncomputable def fp8e5m2List (xs : List ℝ) : List ℝ :=
  xs.map fp8e5m2Round

/-- Computed mixed-fp8-E5M2 gemm entry: f32 rounded entry over the quantised
    inputs. -/
noncomputable def gemmEntryMixedFp8E5M2Rounded (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β (fp8e5m2List a) (fp8e5m2List b) c

/-- **Mixed-fp8-E5M2 entry error split** — instance of
    `gemmEntry_narrow_error_split`. -/
theorem gemmEntryMixedFp8E5M2_error_split (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    |gemmEntryMixedFp8E5M2Rounded α β a b c - gemmEntry α β a b c|
      ≤ |gemmEntryRounded α β (fp8e5m2List a) (fp8e5m2List b) c
            - gemmEntry α β (fp8e5m2List a) (fp8e5m2List b) c|
        + |α * dot (fp8e5m2List a) (fp8e5m2List b) - α * dot a b| := by
  unfold gemmEntryMixedFp8E5M2Rounded
  exact gemmEntry_narrow_error_split α β (fp8e5m2List a) (fp8e5m2List b) a b c

-- ── fp8 E4M3 — the same split, its own unit roundoff ─────────────────────

/-- fp8 E4M3 unit roundoff (abstract; `2⁻⁴` for the 3-bit mantissa). -/
axiom fp8e4m3Unit : ℝ

/-- Round a real to the nearest fp8 E4M3 value. -/
axiom fp8e4m3Round (v : ℝ) : ℝ

/-- **The fp8 E4M3 rounding model**, realised by
    `quanta_ir::dtype::f32_to_fp8 _ 4 3`. -/
axiom fp8e4m3_rounding_model :
    0 ≤ fp8e4m3Unit ∧
    ∀ v : ℝ, ∃ δ : ℝ, |δ| ≤ fp8e4m3Unit ∧ fp8e4m3Round v = v * (1 + δ)

/-- The fp8 E4M3 unit roundoff is non-negative. -/
theorem fp8e4m3Unit_nonneg : 0 ≤ fp8e4m3Unit := fp8e4m3_rounding_model.1

/-- Elementwise fp8 E4M3 quantisation of a list. -/
noncomputable def fp8e4m3List (xs : List ℝ) : List ℝ :=
  xs.map fp8e4m3Round

/-- Computed mixed-fp8-E4M3 gemm entry. -/
noncomputable def gemmEntryMixedFp8E4M3Rounded (α β : ℝ) (a b : List ℝ) (c : ℝ) : ℝ :=
  gemmEntryRounded α β (fp8e4m3List a) (fp8e4m3List b) c

/-- **Mixed-fp8-E4M3 entry error split** — instance of
    `gemmEntry_narrow_error_split`. -/
theorem gemmEntryMixedFp8E4M3_error_split (α β : ℝ) (a b : List ℝ) (c : ℝ) :
    |gemmEntryMixedFp8E4M3Rounded α β a b c - gemmEntry α β a b c|
      ≤ |gemmEntryRounded α β (fp8e4m3List a) (fp8e4m3List b) c
            - gemmEntry α β (fp8e4m3List a) (fp8e4m3List b) c|
        + |α * dot (fp8e4m3List a) (fp8e4m3List b) - α * dot a b| := by
  unfold gemmEntryMixedFp8E4M3Rounded
  exact gemmEntry_narrow_error_split α β (fp8e4m3List a) (fp8e4m3List b) a b c

end Quanta.Blas

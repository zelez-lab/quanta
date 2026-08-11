/-
Quantization round-trip bound + clamp-free max-abs scales — the Lean
supplement of the quantized-inference slice, extending the 084.1
foundation in `Quanta.Dtype.Quant`.

The runtime (`dtype::quantize_sym` / `dtype::dequantize_sym`, reused
verbatim by the quantizer) computes

  quantize(x)   = clamp(round_ties_even(x / s), -(hi+1), hi)
  dequantize(q) = s * q

with the max-abs scale construction `s = max|w| / hi` (hi = 127 for
int8, 7 for int4 — `quantRange`). `Quant.lean` proved the integer
content (clamp range, int4 pack/unpack); the op-matrix pins the float
arithmetic per backend bit-exactly. THIS file proves the two ℝ-level
claims the scope's correctness contract makes about the composition —
the only place in the pipeline where information is lost:

* **T9234 — the round-trip bound.** For `s > 0`,
  `|s · round_te(x/s) − x| ≤ s/2`, with EXACTNESS when `x` is an exact
  code multiple `s·k`. This is the single tolerance the whole quantized
  path carries; the round-trip property tests assert its f32 twin per
  element (with the half-ulp slack stated in-code), and the end-to-end
  logits gate derives its tolerance from this bound's propagation.

* **T9235 — clamp-free max-abs scales.** With `s = maxabs/hi` and
  `|x| ≤ maxabs`, every code lands in `[−hi, hi]`: the clamp in
  `quantize_sym` never fires and code `−(hi+1)` (−128 / −8) is never
  produced. Composed (`t9234_quantize_dequantize_roundtrip`), the FULL
  runtime quantize — clamp included, modeled over `Quant.clampCode` —
  satisfies the T9234 bound for every in-range input.

ℝ-level, no new axioms (0 TCB growth): `roundTiesEven` is modeled
directly over `Int.floor` / `Int.fract`. Only the nearest-integer
property feeds the bound, but the ties-even choice at exact halves is
carried so the model mirrors the pinned rounding mode — both tie arms
sit at exactly 1/2, so the bound is tight there.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Archimedean
import Mathlib.Algebra.Order.Floor
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Quanta.Dtype.Quant

namespace Quanta.Dtype.QuantRoundTrip

/-- Round-half-to-even over ℝ: the nearest integer, choosing the even
    one at exact halves. Mirrors `f32::round_ties_even` as used by
    `dtype::quantize_sym`. -/
noncomputable def roundTiesEven (x : ℝ) : ℤ :=
  if Int.fract x < 1 / 2 then ⌊x⌋
  else if 1 / 2 < Int.fract x then ⌊x⌋ + 1
  else if Even ⌊x⌋ then ⌊x⌋ else ⌊x⌋ + 1

/-- `roundTiesEven` is a nearest-integer rounding: it never moves a
    point by more than 1/2. The engine of T9234. -/
theorem abs_roundTiesEven_sub_le_half (x : ℝ) :
    |(roundTiesEven x : ℝ) - x| ≤ 1 / 2 := by
  have hfl := Int.self_sub_fract x
  have h0 : (0 : ℝ) ≤ Int.fract x := Int.fract_nonneg x
  have h1 : Int.fract x < 1 := Int.fract_lt_one x
  unfold roundTiesEven
  split_ifs with hlt hgt hev
  · rw [abs_le]; constructor <;> linarith
  · rw [abs_le]; push_cast; constructor <;> linarith
  · have htie : Int.fract x = 1 / 2 :=
      le_antisymm (not_lt.1 hgt) (not_lt.1 hlt)
    rw [abs_le]; constructor <;> linarith
  · have htie : Int.fract x = 1 / 2 :=
      le_antisymm (not_lt.1 hgt) (not_lt.1 hlt)
    rw [abs_le]; push_cast; constructor <;> linarith

/-- Integers round to themselves — the exactness kernel. -/
theorem roundTiesEven_intCast (k : ℤ) : roundTiesEven (k : ℝ) = k := by
  norm_num [roundTiesEven, Int.fract_intCast, Int.floor_intCast]

/-- Rounding never falls below an integer lower bound. -/
theorem le_roundTiesEven {x : ℝ} {B : ℤ} (h : (B : ℝ) ≤ x) :
    B ≤ roundTiesEven x := by
  have hfl : B ≤ ⌊x⌋ := Int.le_floor.2 h
  unfold roundTiesEven
  split_ifs <;> omega

/-- Rounding never exceeds an integer upper bound: the round-up arm
    only fires when `x` sits strictly above its floor, which pushes the
    floor strictly below the bound. -/
theorem roundTiesEven_le {x : ℝ} {B : ℤ} (h : x ≤ (B : ℝ)) :
    roundTiesEven x ≤ B := by
  have hfl : ⌊x⌋ ≤ B := by
    have hb := Int.floor_le_floor h
    rwa [Int.floor_intCast] at hb
  have hkey : (1 : ℝ) / 2 ≤ Int.fract x → ⌊x⌋ + 1 ≤ B := by
    intro hhalf
    have hsf := Int.self_sub_fract x
    have hxlt : (⌊x⌋ : ℝ) < (B : ℝ) := by linarith
    have : ⌊x⌋ < B := by exact_mod_cast hxlt
    omega
  unfold roundTiesEven
  split_ifs with h1 h2 h3
  · exact hfl
  · exact hkey (le_of_lt h2)
  · exact hfl
  · exact hkey (not_lt.1 h1)

/-- The clamp is the identity on in-range codes (over `Quant.clampCode`,
    the shipped 084.1 model). -/
theorem clampCode_eq_of_in_range (c lo hi : ℤ) (h1 : lo ≤ c) (h2 : c ≤ hi) :
    Quant.clampCode c lo hi = c := by
  unfold Quant.clampCode
  split_ifs <;> omega

/-- The FULL runtime quantize, clamp included — `dtype::quantize_sym`
    over ℝ, with the `quantRange` clamp bounds `[-(hi+1), hi]`
    (int8: `[-128, 127]`, int4: `[-8, 7]` — `range_int8`/`range_int4`). -/
noncomputable def quantizeSym (s : ℝ) (hi : ℤ) (x : ℝ) : ℤ :=
  Quant.clampCode (roundTiesEven (x / s)) (-(hi + 1)) hi

-- ── T9234 — the round-trip bound ──────────────────────────────────────

/-- **T9234** — symmetric round-trip bound: dequantize ∘ quantize moves
    any element by at most half a scale step,
    `|s · round_te(x/s) − x| ≤ s/2`. The per-element tolerance the
    round-trip property tests assert, and the seed of the end-to-end
    logits gate's derived tolerance. -/
theorem t9234_roundtrip_half_scale (s x : ℝ) (hs : 0 < s) :
    |s * (roundTiesEven (x / s) : ℝ) - x| ≤ s / 2 := by
  have h := abs_roundTiesEven_sub_le_half (x / s)
  have hkey : s * ((roundTiesEven (x / s) : ℝ) - x / s)
      = s * (roundTiesEven (x / s) : ℝ) - x := by
    rw [mul_sub, ← mul_div_assoc, mul_div_cancel_left₀ _ hs.ne']
  calc |s * (roundTiesEven (x / s) : ℝ) - x|
      = s * |(roundTiesEven (x / s) : ℝ) - x / s| := by
        rw [← hkey, abs_mul, abs_of_pos hs]
    _ ≤ s * (1 / 2) := mul_le_mul_of_nonneg_left h hs.le
    _ = s / 2 := by ring

/-- **T9234 (corollary)** — exactness on code multiples: an input that
    IS a code (`x = s·k`) round-trips with zero error. With `|k| ≤ hi`
    this extends through the clamp (`clampCode_eq_of_in_range`), so
    stored checkpoints re-quantize to themselves. -/
theorem t9234_exact_at_codes (s : ℝ) (k : ℤ) (hs : 0 < s) :
    s * (roundTiesEven (s * k / s) : ℝ) = s * k := by
  rw [mul_div_cancel_left₀ _ hs.ne', roundTiesEven_intCast]

-- ── T9235 — clamp-free max-abs scales ─────────────────────────────────

/-- **T9235** — the max-abs scale construction `s = maxabs/hi` keeps
    every rounded code inside `[−hi, hi]`: the `quantize_sym` clamp
    never fires, and code `−(hi+1)` (−128 int8 / −8 int4) is never
    produced by quantization. -/
theorem t9235_maxabs_clamp_free (maxabs x : ℝ) (hi : ℤ)
    (hhi : 0 < hi) (hmax : 0 < maxabs) (hx : |x| ≤ maxabs) :
    -hi ≤ roundTiesEven (x / (maxabs / hi)) ∧
      roundTiesEven (x / (maxabs / hi)) ≤ hi := by
  have hhiR : (0 : ℝ) < (hi : ℝ) := by exact_mod_cast hhi
  have hs : 0 < maxabs / (hi : ℝ) := div_pos hmax hhiR
  have hsm : maxabs / (hi : ℝ) * (hi : ℝ) = maxabs :=
    div_mul_cancel₀ _ hhiR.ne'
  obtain ⟨hxlo, hxhi⟩ := abs_le.1 hx
  constructor
  · refine le_roundTiesEven ?_
    push_cast
    rw [le_div_iff₀ hs]
    linarith
  · refine roundTiesEven_le ?_
    rw [div_le_iff₀ hs]
    linarith

/-- **T9234 ∘ T9235** — the composed runtime guarantee: under max-abs
    scales the FULL quantize (clamp included) satisfies the s/2
    round-trip bound for every in-range element. This is the statement
    the quantizer's tests exercise end to end. -/
theorem t9234_quantize_dequantize_roundtrip (maxabs x : ℝ) (hi : ℤ)
    (hhi : 0 < hi) (hmax : 0 < maxabs) (hx : |x| ≤ maxabs) :
    |maxabs / hi * (quantizeSym (maxabs / hi) hi x : ℝ) - x|
      ≤ maxabs / hi / 2 := by
  obtain ⟨hlo, hup⟩ := t9235_maxabs_clamp_free maxabs x hi hhi hmax hx
  have hhiR : (0 : ℝ) < (hi : ℝ) := by exact_mod_cast hhi
  have hs : 0 < maxabs / (hi : ℝ) := div_pos hmax hhiR
  unfold quantizeSym
  rw [clampCode_eq_of_in_range _ _ _ (by omega) hup]
  exact t9234_roundtrip_half_scale _ x hs

end Quanta.Dtype.QuantRoundTrip

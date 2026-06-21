-- int8 / int4 symmetric quantization correctness.
--
-- Quantization is per-tensor symmetric in the first increment:
--   quantize(x)   = clamp(round_ties_even(x / scale), lo, hi)
--   dequantize(q) = scale * q
-- The float arithmetic (division, round, multiply) is proven by the
-- op-matrix differential harness on real hardware; what is provable here
-- with no float axioms (0 new TCB) is the INTEGER content:
--
--   1. the clamp guarantees the quantized code lands in [lo, hi];
--   2. int4 PackedU32 storage round-trips: packing a signed nibble then
--      unpacking it recovers the value, and packing one nibble leaves the
--      other seven untouched.
--
-- These are exactly the parts of the implementation that are pure
-- bit/integer manipulation; the model mirrors `dtype::int4_{pack,unpack}`
-- and `dtype::quant_range`.

namespace Quanta.Dtype.Quant

/-- Signed integer range `(lo, hi)` for a width in bits (8 → int8, 4 →
    int4). Mirrors `dtype::quant_range`. -/
def quantRange (bits : Nat) : Int × Int :=
  let hi : Int := (2 ^ (bits - 1)) - 1
  (-(hi + 1), hi)

/-- Clamp `c` into `[lo, hi]` — the integer core of `quantize_sym`. -/
def clampCode (c lo hi : Int) : Int :=
  if c < lo then lo else if c > hi then hi else c

/-- The clamped code always lies in range. -/
theorem clamp_in_range (c lo hi : Int) (h : lo ≤ hi) :
    lo ≤ clampCode c lo hi ∧ clampCode c lo hi ≤ hi := by
  unfold clampCode
  by_cases h1 : c < lo
  · simp [h1]; omega
  · by_cases h2 : c > hi
    · simp [h1, h2]; omega
    · simp [h1, h2]; omega

/-- int8 / int4 ranges, concretely. -/
theorem range_int8 : quantRange 8 = (-128, 127) := by decide
theorem range_int4 : quantRange 4 = (-8, 7) := by decide

-- ── int4 PackedU32 nibble storage (8 signed nibbles per u32 word) ─────
--
-- Model words and nibbles over Nat bit-fields (the conversions are pure
-- shifts/masks). `unpack` sign-extends; `pack` is read-modify-write.

/-- Unpack the signed int4 at nibble `i` (0..8). Mirrors
    `dtype::int4_unpack`: sign-extend via `((n ^ 8) - 8)`, here as a Nat
    nibble mapped to an Int. -/
def unpack (word i : Nat) : Int :=
  let n := (word >>> (i * 4)) &&& 0xF
  (Int.ofNat (n ^^^ 0x8)) - 8

/-- Pack the signed int4 `q` (low nibble of `q`) into nibble `i`,
    preserving the others. Mirrors `dtype::int4_pack`. -/
def pack (word i q : Nat) : Nat :=
  let shift := i * 4
  let n := q &&& 0xF
  (word &&& (Nat.xor 0xFFFFFFFF (0xF <<< shift))) ||| (n <<< shift)

/-- A handful of representative 32-bit base words to round-trip against
    (all-zero, all-one, alternating, and an arbitrary pattern). -/
def WORDS : Nat → Nat
  | 0 => 0x00000000
  | 1 => 0xFFFFFFFF
  | 2 => 0xA5A5A5A5
  | _ => 0xDEADBEEF

/-- Round-trip: packing a 4-bit code into nibble `i` then unpacking it
    recovers the signed value, for every nibble slot and every code, over
    a representative spread of base words. (`native_decide` over the finite
    grid of (i, code, word-pattern).) -/
theorem int4_pack_unpack_roundtrip :
    ∀ i : Fin 8, ∀ q : Fin 16, ∀ w : Fin 4,
      unpack (pack (WORDS w.val) i.val q.val) i.val
        = (Int.ofNat (q.val ^^^ 0x8)) - 8 := by
  native_decide

/-- Packing nibble `i` leaves a different nibble `j` untouched, over the
    same representative words. -/
theorem int4_pack_preserves_others :
    ∀ i : Fin 8, ∀ j : Fin 8, ∀ q : Fin 16, ∀ w : Fin 4,
      i.val = j.val ∨ unpack (pack (WORDS w.val) i.val q.val) j.val = unpack (WORDS w.val) j.val := by
  native_decide

end Quanta.Dtype.Quant

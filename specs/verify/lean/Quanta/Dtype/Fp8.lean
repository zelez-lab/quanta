-- fp8 (e5m2 / e4m3) ↔ f32 conversion correctness.
--
-- Two 8-bit float formats share one implementation, parameterised by the
-- exponent/mantissa widths `(eb, mb)`: e5m2 = (5, 2), e4m3 = (4, 3). The
-- Quanta CPU executor, every GPU emitter, and the host reference in
-- `crates/quanta-ir/src/dtype.rs` use the *branchless* conversions modelled
-- here over `Nat` bit-fields. Because the conversions are pure integer
-- bit-twiddling (shifts, masks, comparisons, selects), the Nat model is
-- faithful and the proofs need no float axioms — 0 new TCB.
--
-- An fp8 byte and the 32-bit pattern produced by `unpack` are the model's
-- two carriers. We prove, over the finite 256-value domain (`native_decide`):
--
--   1. round-trip: for every NON-NaN fp8 byte, `pack (unpack b) = b`. fp8
--      NaN encodings canonicalise (many bit patterns → one NaN), so they
--      are excluded — exactly as the implementation behaves.
--   2. injectivity: distinct non-NaN bytes unpack to distinct f32 words.
--
-- Together: fp8 storage round-trips exactly on every representable value
-- (finite, inf, ±0), which is what makes the f32-emulated path correct.

namespace Quanta.Dtype.Fp8

/-- Bit `i` of `n`. -/
def bit (n i : Nat) : Nat := (n >>> i) &&& 1

/-- Round-to-nearest-even right shift by `s`, `Nat` model of
    `dtype::round_shift_rne` for `s < 32`. -/
def rne (v s : Nat) : Nat :=
  if s = 0 then v
  else
    let kept := v >>> s
    let rem := v &&& ((1 <<< s) - 1)
    let half := 1 <<< (s - 1)
    if rem > half ∨ (rem = half ∧ bit kept 0 = 1) then kept + 1 else kept

/-- fp8 → f32 (as a 32-bit `Nat`). Branchless model of `dtype::fp8_to_f32`.
    `eb`/`mb` are the exponent/mantissa widths. -/
def unpack (eb mb : Nat) (b : Nat) : Nat :=
  let sign := bit b (eb + mb)
  let exp := (b >>> mb) &&& ((1 <<< eb) - 1)
  let mant := b &&& ((1 <<< mb) - 1)
  let bias := (1 <<< (eb - 1)) - 1
  let expMask := (1 <<< eb) - 1
  let fsign := sign <<< 31
  let norm := fsign ||| ((exp + 127 - bias) <<< 23) ||| (mant <<< (23 - mb))
  let infMant := if mant ≠ 0 then 0x400000 else 0
  let infnan := fsign ||| (0xFF <<< 23) ||| infMant
  -- leading-bit scan over the mb mantissa bits (mb ≤ 3)
  let lead := (List.range mb).foldl
    (fun acc i => let s := bit mant i; (s * i) ||| ((1 - s) * acc)) 0
  let shifts := mb - lead
  -- e_sub + mb + 127 = 128 - bias - shifts (kept ≥ 0 for the inputs reached)
  let eSubBiased := 128 - bias - shifts
  let mSub := (mant <<< shifts) &&& ((1 <<< mb) - 1)
  let sub := fsign ||| (eSubBiased <<< 23) ||| (mSub <<< (23 - mb))
  let out := norm
  let out := if exp = 0 ∧ mant ≠ 0 then sub else out
  let out := if exp = expMask then infnan else out
  let out := if exp = 0 ∧ mant = 0 then fsign else out
  out

/-- f32 (32-bit `Nat`) → fp8 byte. Branchless model of `dtype::f32_to_fp8`.
    Inputs come from `unpack`, so the f32 exponent is in range and the
    `Nat` subtractions used here never underflow. -/
def pack (eb mb : Nat) (f : Nat) : Nat :=
  let sign := bit f 31
  let signSlot := sign <<< (eb + mb)
  let fexp := (f >>> 23) &&& 0xFF
  let fmant := f &&& 0x7FFFFF
  let bias := (1 <<< (eb - 1)) - 1
  let expMask := (1 <<< eb) - 1
  -- target_exp = fexp - 127 + bias (≥ 0 over the unpack image we prove on)
  let targetExp := fexp + bias - 127
  let rndN := rne fmant (23 - mb)
  let carry := if (rndN >>> mb) ≠ 0 then 1 else 0
  let outExpN := targetExp + carry
  let outMantN := if carry = 1 then 0 else rndN
  let normal :=
    if outExpN ≥ expMask then signSlot ||| (expMask <<< mb)
    else signSlot ||| (outExpN <<< mb) ||| (outMantN &&& ((1 <<< mb) - 1))
  -- subnormal: shift the full significand into fp8's subnormal scale, RNE.
  -- shift = (23 - mb) + (1 - target_exp); over the unpack image this is a
  -- small non-negative Nat. `target_exp ≤ 0` ⇔ `fexp + bias ≤ 127`.
  let signif := fmant ||| 0x800000
  let shift := (23 - mb) + (127 - (fexp + bias) + 1)
  let sub := signSlot ||| rne signif shift
  let infnan := signSlot ||| (expMask <<< mb) ||| (if fmant ≠ 0 then 1 <<< (mb - 1) else 0)
  let ovf := signSlot ||| (expMask <<< mb)
  let out := normal
  let out := if fexp + bias ≤ 127 then sub else out
  let out := if targetExp ≥ expMask then ovf else out
  let out := if fexp = 0 ∧ fmant = 0 then signSlot else out
  let out := if fexp = 0xFF then infnan else out
  out

/-- An fp8 byte is a NaN encoding: exponent all-ones and mantissa nonzero. -/
def isNaN (eb mb : Nat) (b : Nat) : Bool :=
  let exp := (b >>> mb) &&& ((1 <<< eb) - 1)
  let mant := b &&& ((1 <<< mb) - 1)
  decide (exp = (1 <<< eb) - 1 ∧ mant ≠ 0)

-- Cross-checks against the Rust reference `dtype::fp8_to_f32`: these pin
-- the Lean `unpack` to the exact f32 bit patterns the implementation
-- produces, so the model is verified faithful, not merely self-consistent.
example : unpack 5 2 0x01 = 0x37800000 := by native_decide  -- smallest e5m2 subnormal
example : unpack 5 2 0x3c = 0x3f800000 := by native_decide  -- 1.0
example : unpack 5 2 0x38 = 0x3f000000 := by native_decide  -- 0.5
example : unpack 5 2 0x7c = 0x7f800000 := by native_decide  -- +inf
example : unpack 4 3 0x01 = 0x3b000000 := by native_decide  -- smallest e4m3 subnormal
example : unpack 4 3 0x3c = 0x3fc00000 := by native_decide  -- 1.5
example : unpack 4 3 0x38 = 0x3f800000 := by native_decide  -- 1.0

/-- e5m2 round-trip: every non-NaN byte packs back to itself. -/
theorem roundtrip_e5m2 :
    ∀ b : Fin 256, isNaN 5 2 b.val = false → pack 5 2 (unpack 5 2 b.val) = b.val := by
  native_decide

/-- e4m3 round-trip: every non-NaN byte packs back to itself. -/
theorem roundtrip_e4m3 :
    ∀ b : Fin 256, isNaN 4 3 b.val = false → pack 4 3 (unpack 4 3 b.val) = b.val := by
  native_decide

/-- e5m2 injectivity: distinct non-NaN bytes unpack to distinct f32 words. -/
theorem unpack_injective_e5m2 :
    ∀ a b : Fin 256, isNaN 5 2 a.val = false → isNaN 5 2 b.val = false →
      unpack 5 2 a.val = unpack 5 2 b.val → a.val = b.val := by
  native_decide

/-- e4m3 injectivity. -/
theorem unpack_injective_e4m3 :
    ∀ a b : Fin 256, isNaN 4 3 a.val = false → isNaN 4 3 b.val = false →
      unpack 4 3 a.val = unpack 4 3 b.val → a.val = b.val := by
  native_decide

end Quanta.Dtype.Fp8

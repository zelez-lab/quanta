-- bfloat16 ↔ f32 conversion correctness.
--
-- bf16 is the top 16 bits of an f32: sign(1) | exponent(8) | mantissa(7).
-- The Quanta CPU executor and every emitter convert with:
--   unpack : bf16_bits → f32_bits   = bits <<< 16          (zero-extend low 16)
--   pack   : f32_bits  → bf16_bits  = bits >>> 16          (truncate; round-to-
--                                                            nearest-even adds a
--                                                            bias first)
--
-- This module proves the two facts the implementation relies on, modelled
-- over `Nat` bit-fields (the conversions are pure shifts, so the bit-level
-- model is faithful and the proofs need no float axioms — 0 new TCB):
--
--   1. pack ∘ unpack = id            on every 16-bit pattern        (lossless
--                                                                    storage)
--   2. unpack is injective           distinct bf16 → distinct f32   (no two
--                                                                    bf16 alias)
--
-- Together these say bf16 storage round-trips exactly, which is what makes
-- the f32-emulated path correct: a value written as bf16 and read back is
-- the same value, and the f32 produced on load is uniquely determined by
-- the stored bits.

namespace Quanta.Dtype.Bf16

/-- unpack: place the 16 bf16 bits into the high half of a 32-bit word. -/
def unpack (b : Nat) : Nat := b * 65536  -- b <<< 16

/-- pack (truncating): take the high 16 bits of a 32-bit word. -/
def pack (f : Nat) : Nat := f / 65536    -- f >>> 16

/-- Round-trip: packing an unpacked bf16 pattern recovers it exactly,
    for every 16-bit value. (`b * 65536 / 65536 = b`.) -/
theorem pack_unpack (b : Nat) : pack (unpack b) = b := by
  unfold pack unpack
  omega

/-- The unpacked f32 word never exceeds 32 bits for a 16-bit input:
    `unpack b < 2^32` when `b < 2^16`. Confirms unpack lands in the
    high half with the low 16 bits zero. -/
theorem unpack_lt_2pow32 (b : Nat) (h : b < 65536) : unpack b < 4294967296 := by
  unfold unpack
  omega

/-- The low 16 bits of an unpacked word are zero — i.e. unpack produces a
    bf16-representable f32 (mantissa tail clear). -/
theorem unpack_low_bits_zero (b : Nat) : unpack b % 65536 = 0 := by
  unfold unpack
  omega

/-- unpack is injective: distinct bf16 patterns map to distinct f32 words,
    so no two bf16 values alias the same f32. -/
theorem unpack_injective {a b : Nat} (h : unpack a = unpack b) : a = b := by
  unfold unpack at h
  omega

/-- pack truncates toward zero by exactly 16 bits: the recovered bf16 is the
    high half of the f32 word regardless of the low (mantissa-tail) bits.
    `pack (unpack b + r) = b` for any tail `r < 65536`. This is the
    round-to-zero contract; the implementation's round-to-nearest-even adds
    a bias before this truncation, handled in the executor and pinned by the
    differential op-matrix lane. -/
theorem pack_unpack_with_tail (b r : Nat) (hr : r < 65536) :
    pack (unpack b + r) = b := by
  unfold pack unpack
  omega

end Quanta.Dtype.Bf16

//! Narrow-scalar (bf16 / fp8) storage conversions, ported line-for-line
//! from the JIT emitter (`quanta-ir/src/emit_spirv/ops.rs`) so the AOT and
//! JIT SPIR-V emitters share one storage contract: bf16 buffers are 16-bit
//! elements, fp8 buffers are 8-bit elements (native stride, matching the
//! host upload and the CPU executor); push-constant members stay u32-slot.

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    // ── bf16 storage conversions ─────────────────────────────────────────
    //
    // bf16 is stored as a 16-bit pattern (the top half of an f32) but
    // computed in f32. These convert at the Load/Store boundary. The
    // packing rounds to nearest-even and must match the CPU executor's
    // `f32_to_bf16` bit-for-bit (the differential oracle).

    /// Convert a loaded bf16 storage value (`u16` buffer element, or a
    /// `u32` push-constant member carrying the bits in its low half) into
    /// an f32 register: `f32 = bitcast(bits << 16)`. `loaded_ty` is the
    /// SPIR-V type the value was loaded as. Returns the f32 value id.
    pub(crate) fn bf16_unpack_to_f32(&mut self, loaded: u32, loaded_ty: u32) -> u32 {
        let u32_ty = self.ensure_type_u32();
        let f32_ty = self.ensure_type_f32();
        // Widen to u32 if the storage element was narrow.
        let bits32 = if loaded_ty != u32_ty {
            let w = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_U_CONVERT, &[u32_ty, w, loaded]);
            w
        } else {
            loaded
        };
        let sixteen = self.emit_constant_u32(16);
        let shifted = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_LEFT_LOGICAL,
            &[u32_ty, shifted, bits32, sixteen],
        );
        let f = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[f32_ty, f, shifted]);
        f
    }

    /// Pack an f32 register into a bf16 storage value (`u16`),
    /// round-to-nearest-even:
    ///   bits = bitcast<u32>(f); bias = 0x7fff + ((bits >> 16) & 1);
    ///   out  = (bits + bias) >> 16
    /// (NaN handling: a NaN's exponent is all-ones so the bias never
    /// overflows it into ±inf, matching the CPU path for finite values; the
    /// op-matrix oracle uses the same formula and skips NaN cases.)
    /// Returns the u16-typed storage value id.
    pub(crate) fn bf16_pack_from_f32(&mut self, f32_val: u32) -> u32 {
        let u32_ty = self.ensure_type_u32();
        let bits = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[u32_ty, bits, f32_val]);
        // lsb = (bits >> 16) & 1
        let sixteen = self.emit_constant_u32(16);
        let one = self.emit_constant_u32(1);
        let hi = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_RIGHT_LOGICAL,
            &[u32_ty, hi, bits, sixteen],
        );
        let lsb = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_BITWISE_AND,
            &[u32_ty, lsb, hi, one],
        );
        // bias = 0x7fff + lsb
        let base_bias = self.emit_constant_u32(0x7fff);
        let bias = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[u32_ty, bias, base_bias, lsb],
        );
        // rounded = (bits + bias) >> 16
        let summed = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IADD,
            &[u32_ty, summed, bits, bias],
        );
        let rounded = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_SHIFT_RIGHT_LOGICAL,
            &[u32_ty, rounded, summed, sixteen],
        );
        // Narrow to u16 for the native storage element.
        let u16_ty = self.ensure_type_u16();
        let narrowed = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_U_CONVERT,
            &[u16_ty, narrowed, rounded],
        );
        narrowed
    }

    // ── fp8 storage conversions ──────────────────────────────────────────
    //
    // The branchless reference is `crate::dtype::{fp8_to_f32, f32_to_fp8}`;
    // these emit the identical arithmetic op-by-op. Storage is a `u8`
    // buffer element (native 1-byte stride; push-constant members stay
    // u32-slot). The conversion math itself runs in u32. `eb`/`mb` are the
    // exponent/mantissa widths. Helpers below are thin one-op wrappers so
    // the conversion reads close to the Rust source.

    fn spv_u32(&mut self) -> u32 {
        self.ensure_type_u32()
    }
    fn spv_const(&mut self, v: u32) -> u32 {
        self.emit_constant_u32(v)
    }
    /// Emit a 2-arg op producing `ty`, return the result id.
    fn spv_bin(&mut self, opcode: u16, ty: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_function, opcode, &[ty, id, a, b]);
        id
    }
    fn spv_and(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_BITWISE_AND, t, a, b)
    }
    fn spv_or(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_BITWISE_OR, t, a, b)
    }
    fn spv_add(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_IADD, t, a, b)
    }
    fn spv_sub(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_ISUB, t, a, b)
    }
    fn spv_mul(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_IMUL, t, a, b)
    }
    fn spv_shl(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_SHIFT_LEFT_LOGICAL, t, a, b)
    }
    fn spv_shr(&mut self, a: u32, b: u32) -> u32 {
        let t = self.spv_u32();
        self.spv_bin(OP_SHIFT_RIGHT_LOGICAL, t, a, b)
    }
    /// A bool-typed comparison.
    fn spv_cmp(&mut self, opcode: u16, a: u32, b: u32) -> u32 {
        let bt = self.ensure_type_bool();
        self.spv_bin(opcode, bt, a, b)
    }
    fn spv_logic(&mut self, opcode: u16, a: u32, b: u32) -> u32 {
        let bt = self.ensure_type_bool();
        self.spv_bin(opcode, bt, a, b)
    }
    /// `select(cond ? a : b)` over u32 (SPIR-V operand order is cond,a,b).
    fn spv_select(&mut self, ty: u32, cond: u32, a: u32, b: u32) -> u32 {
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_SELECT, &[ty, id, cond, a, b]);
        id
    }

    /// Round-to-nearest-even right shift by a constant `s` (mirrors
    /// `dtype::round_shift_rne` for the constant-shift cases used here:
    /// `s` is always a known small value, so no runtime clamp is needed).
    fn spv_rne_const(&mut self, v: u32, s: u32) -> u32 {
        if s == 0 {
            return v;
        }
        let u = self.spv_u32();
        let s_c = self.spv_const(s);
        let kept = self.spv_shr(v, s_c);
        let mask = self.spv_const((1u32 << s) - 1);
        let rem = self.spv_and(v, mask);
        let half = self.spv_const(1u32 << (s - 1));
        // roundup = rem > half || (rem == half && (kept&1)==1)
        let gt = self.spv_cmp(OP_UGREATER_THAN, rem, half);
        let eq = self.spv_cmp(OP_IEQUAL, rem, half);
        let one = self.spv_const(1);
        let kept_lsb = self.spv_and(kept, one);
        let lsb_set = self.spv_cmp(OP_IEQUAL, kept_lsb, one);
        let tie_up = self.spv_logic(OP_LOGICAL_AND, eq, lsb_set);
        let roundup = self.spv_logic(OP_LOGICAL_OR, gt, tie_up);
        let zero = self.spv_const(0);
        let inc = self.spv_select(u, roundup, one, zero);
        self.spv_add(kept, inc)
    }

    /// Unpack a loaded fp8 value (`u8` buffer element, or a `u32`
    /// push-constant member carrying the byte in its low bits) into an f32
    /// register. `loaded_ty` is the SPIR-V type the value was loaded as.
    /// Mirrors `dtype::fp8_to_f32`.
    pub(crate) fn fp8_unpack_to_f32(
        &mut self,
        loaded: u32,
        loaded_ty: u32,
        eb: u32,
        mb: u32,
    ) -> u32 {
        let u = self.spv_u32();
        let f32_ty = self.ensure_type_f32();
        // Widen the byte to u32 before the bit math.
        let loaded = if loaded_ty != u {
            let w = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_U_CONVERT, &[u, w, loaded]);
            w
        } else {
            loaded
        };
        let bias = (1u32 << (eb - 1)) - 1;
        let exp_mask = (1u32 << eb) - 1;
        let mant_mask = (1u32 << mb) - 1;

        let sm = self.spv_const(eb + mb);
        let sign = {
            let s = self.spv_shr(loaded, sm);
            let one = self.spv_const(1);
            self.spv_and(s, one)
        };
        let exp = {
            let mbc = self.spv_const(mb);
            let s = self.spv_shr(loaded, mbc);
            let m = self.spv_const(exp_mask);
            self.spv_and(s, m)
        };
        let mant = {
            let m = self.spv_const(mant_mask);
            self.spv_and(loaded, m)
        };
        let c31 = self.spv_const(31);
        let fsign = self.spv_shl(sign, c31);

        // norm = fsign | ((exp + 127 - bias) << 23) | (mant << (23-mb))
        let c127 = self.spv_const(127);
        let cbias = self.spv_const(bias);
        let e1 = self.spv_add(exp, c127);
        let e2 = self.spv_sub(e1, cbias);
        let c23 = self.spv_const(23);
        let es = self.spv_shl(e2, c23);
        let clo = self.spv_const(23 - mb);
        let ms = self.spv_shl(mant, clo);
        let norm = {
            let t = self.spv_or(fsign, es);
            self.spv_or(t, ms)
        };

        // infnan = fsign | (0xFF << 23) | (mant!=0 ? 0x400000 : 0)
        let c0xff = self.spv_const(0xFF);
        let ff_sh = self.spv_shl(c0xff, c23);
        let zero = self.spv_const(0);
        let cz = self.spv_const(0x0040_0000);
        let mant_nz = self.spv_cmp(OP_INOT_EQUAL, mant, zero);
        let inf_mant = self.spv_select(u, mant_nz, cz, zero);
        let infnan = {
            let t = self.spv_or(fsign, ff_sh);
            self.spv_or(t, inf_mant)
        };

        // subnormal: unrolled leading-bit scan over mb bits.
        let mut lead = self.spv_const(0);
        for i in 0..mb {
            let ci = self.spv_const(i);
            let one = self.spv_const(1);
            let sh = self.spv_shr(mant, ci);
            let bit = self.spv_and(sh, one);
            // lead = bit*i | (1-bit)*lead
            let bi = self.spv_mul(bit, ci);
            let inv = self.spv_sub(one, bit);
            let il = self.spv_mul(inv, lead);
            lead = self.spv_or(bi, il);
        }
        let cmb = self.spv_const(mb);
        let shifts = self.spv_sub(cmb, lead);
        // e_sub + mb + 127 = (1 - bias - mb - shifts) + mb + 127
        //                  = 128 - bias - shifts
        let c_base = self.spv_const(128 - bias);
        let e_sub_biased = self.spv_sub(c_base, shifts);
        let m_shifted = self.spv_shl(mant, shifts);
        let mm = self.spv_const(mant_mask);
        let m_sub = self.spv_and(m_shifted, mm);
        let sub = {
            let es2 = self.spv_shl(e_sub_biased, c23);
            let ms2 = self.spv_shl(m_sub, clo);
            let t = self.spv_or(fsign, es2);
            self.spv_or(t, ms2)
        };

        // predicates
        let cem = self.spv_const(exp_mask);
        let is_inf = self.spv_cmp(OP_IEQUAL, exp, cem);
        let exp_z = self.spv_cmp(OP_IEQUAL, exp, zero);
        let mant_z = self.spv_cmp(OP_IEQUAL, mant, zero);
        let is_zero = self.spv_logic(OP_LOGICAL_AND, exp_z, mant_z);
        let is_sub = self.spv_logic(OP_LOGICAL_AND, exp_z, mant_nz);

        // out = select chain: norm; sub; infnan; zero(=fsign)
        let mut out = norm;
        out = self.spv_select(u, is_sub, sub, out);
        out = self.spv_select(u, is_inf, infnan, out);
        out = self.spv_select(u, is_zero, fsign, out);

        let f = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[f32_ty, f, out]);
        f
    }

    /// Pack an f32 register into an fp8 storage value, narrowed to the
    /// `u8` buffer element type. Mirrors `dtype::f32_to_fp8`.
    pub(crate) fn fp8_pack_from_f32(&mut self, f32_val: u32, eb: u32, mb: u32) -> u32 {
        let u = self.spv_u32();
        let bias = (1u32 << (eb - 1)) - 1;
        let exp_mask = (1u32 << eb) - 1;
        let mant_mask = (1u32 << mb) - 1;

        let b = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[u, b, f32_val]);

        let c31 = self.spv_const(31);
        let one = self.spv_const(1);
        let zero = self.spv_const(0);
        let sign = {
            let s = self.spv_shr(b, c31);
            self.spv_and(s, one)
        };
        let sm = self.spv_const(eb + mb);
        let sign_slot = self.spv_shl(sign, sm);
        let c23 = self.spv_const(23);
        let c0xff = self.spv_const(0xFF);
        let fexp = {
            let s = self.spv_shr(b, c23);
            self.spv_and(s, c0xff)
        };
        let cmant = self.spv_const(0x007F_FFFF);
        let fmant = self.spv_and(b, cmant);

        // target_exp = (fexp - 127) + bias = fexp + (bias - 127), as i32.
        // Work signed: use signed ops. Represent constants as u32 bit
        // patterns; SPIR-V int ops are sign-agnostic, only comparisons
        // differ. We compare with signed ops where the reference does.
        let c127 = self.spv_const(127);
        let cbias = self.spv_const(bias);
        // target_exp = fexp - 127 + bias  (may be negative → wraps in u32)
        let te0 = self.spv_sub(fexp, c127);
        let target_exp = self.spv_add(te0, cbias);

        // normal: rnd_n = rne(fmant, 23-mb)
        let rnd_n = self.spv_rne_const(fmant, 23 - mb);
        let cmb = self.spv_const(mb);
        let rnd_hi = self.spv_shr(rnd_n, cmb);
        let carry = self.spv_cmp(OP_INOT_EQUAL, rnd_hi, zero);
        let carry_u = self.spv_select(u, carry, one, zero);
        let out_exp_n = self.spv_add(target_exp, carry_u);
        let out_mant_n = self.spv_select(u, carry, zero, rnd_n);
        let cem = self.spv_const(exp_mask);
        let em_sh = self.spv_shl(cem, cmb);
        let inf_val = self.spv_or(sign_slot, em_sh);
        let normal = {
            let oe = self.spv_shl(out_exp_n, cmb);
            let mm = self.spv_const(mant_mask);
            let om = self.spv_and(out_mant_n, mm);
            let t = self.spv_or(sign_slot, oe);
            let body = self.spv_or(t, om);
            // out_exp_n >= exp_mask ? inf : body  (unsigned compare ok:
            // a wrapped-negative target_exp is huge → ≥ exp_mask → inf,
            // but that case is overridden by is_sub below anyway)
            let oexp_big = self.spv_cmp(OP_UGREATER_THAN_EQUAL, out_exp_n, cem);
            self.spv_select(u, oexp_big, inf_val, body)
        };

        // subnormal: shift = (23-mb) + (1 - target_exp)
        //   = (24 - mb) - target_exp ; signed.
        let signif = {
            let c_imp = self.spv_const(0x0080_0000);
            self.spv_or(fmant, c_imp)
        };
        let c24mb = self.spv_const(24 - mb);
        let shift = self.spv_sub(c24mb, target_exp); // signed value in u32
        // rne with a *runtime* shift; reuse the variable-shift form.
        let sub_rne = self.spv_rne_var(signif, shift);
        let sub_body = self.spv_or(sign_slot, sub_rne);
        // shift > 31 (signed) → ±0
        let c31s = self.spv_const(31);
        let shift_big = self.spv_cmp(OP_SGREATER_THAN, shift, c31s);
        let sub = self.spv_select(u, shift_big, sign_slot, sub_body);

        // inf/nan, overflow, zero
        let nan_m = {
            let c = self.spv_const(1u32 << (mb - 1));
            let nz = self.spv_cmp(OP_INOT_EQUAL, fmant, zero);
            self.spv_select(u, nz, c, zero)
        };
        let infnan = {
            let t = self.spv_or(sign_slot, em_sh);
            self.spv_or(t, nan_m)
        };
        let ovf = self.spv_or(sign_slot, em_sh);

        // predicates (signed where the reference is signed)
        let is_infnan = self.spv_cmp(OP_IEQUAL, fexp, c0xff);
        let fexp_z = self.spv_cmp(OP_IEQUAL, fexp, zero);
        let fmant_z = self.spv_cmp(OP_IEQUAL, fmant, zero);
        let is_zero = self.spv_logic(OP_LOGICAL_AND, fexp_z, fmant_z);
        let is_ovf = self.spv_cmp(OP_SGREATER_THAN_EQUAL, target_exp, cem);
        let is_sub = self.spv_cmp(OP_SLESS_THAN_EQUAL, target_exp, zero);

        // out = normal; sub; ovf; zero; infnan  (priority bottom→top)
        let mut out = normal;
        out = self.spv_select(u, is_sub, sub, out);
        out = self.spv_select(u, is_ovf, ovf, out);
        out = self.spv_select(u, is_zero, sign_slot, out);
        out = self.spv_select(u, is_infnan, infnan, out);
        // Narrow to the u8 storage element.
        let u8_ty = self.ensure_type_u8();
        let narrowed = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_U_CONVERT,
            &[u8_ty, narrowed, out],
        );
        narrowed
    }

    /// Round-to-nearest-even right shift by a *runtime* shift amount,
    /// branchless over `s` (clamps `s>=32` → 0 like `dtype::round_shift_rne`).
    fn spv_rne_var(&mut self, v: u32, s: u32) -> u32 {
        let u = self.spv_u32();
        let c32 = self.spv_const(32);
        let big = self.spv_cmp(OP_UGREATER_THAN_EQUAL, s, c32);
        let c31 = self.spv_const(31);
        let s_c = self.spv_select(u, big, c31, s);
        let kept = self.spv_shr(v, s_c);
        // mask = (1 << s_c) - 1
        let one = self.spv_const(1);
        let sh1 = self.spv_shl(one, s_c);
        let mask = self.spv_sub(sh1, one);
        let rem = self.spv_and(v, mask);
        // half = 1 << (s_c==0 ? 0 : s_c-1)
        let zero = self.spv_const(0);
        let sc_z = self.spv_cmp(OP_IEQUAL, s_c, zero);
        let scm1 = self.spv_sub(s_c, one);
        let hshift = self.spv_select(u, sc_z, zero, scm1);
        let half = self.spv_shl(one, hshift);
        let gt = self.spv_cmp(OP_UGREATER_THAN, rem, half);
        let eq = self.spv_cmp(OP_IEQUAL, rem, half);
        let kept_lsb = self.spv_and(kept, one);
        let lsb_set = self.spv_cmp(OP_IEQUAL, kept_lsb, one);
        let tie_up = self.spv_logic(OP_LOGICAL_AND, eq, lsb_set);
        let roundup = self.spv_logic(OP_LOGICAL_OR, gt, tie_up);
        let inc = self.spv_select(u, roundup, one, zero);
        let r = self.spv_add(kept, inc);
        // s==0 ? v : r
        let s_z = self.spv_cmp(OP_IEQUAL, s, zero);
        let r0 = self.spv_select(u, s_z, v, r);
        // big ? 0 : r0
        self.spv_select(u, big, zero, r0)
    }
}

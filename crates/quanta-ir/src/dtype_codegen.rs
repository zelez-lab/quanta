//! GPU source for the fp8 ↔ f32 conversions.
//!
//! The CPU oracle and the Lean spec use the branchless reference in
//! [`crate::dtype`]; these emit the *identical* arithmetic as WGSL / MSL
//! source (and SPIR-V is emitted op-by-op in the SPIR-V backend). All
//! three must agree bit-for-bit with `dtype::{fp8_to_f32, f32_to_fp8}` —
//! the op-matrix differential harness enforces this on real hardware.
//!
//! The conversions are pure integer bit-twiddling in `u32` registers, so
//! they need no extension and run on every backend. Storage stride is the
//! caller's concern: MSL passes the byte from a `uchar` buffer element
//! (native 1-byte stride), WGSL passes the whole `u32` slot word (its only
//! legal narrow layout). `eb`/`mb` are the exponent and mantissa bit
//! widths (e5m2 = 5,2 — e4m3 = 4,3).

/// Function-name suffix for a given (eb, mb): `e5m2` / `e4m3`.
pub fn fp8_tag(eb: u32, mb: u32) -> String {
    format!("e{eb}m{mb}")
}

/// Which fp8 formats a kernel touches at a Load/Store boundary, as
/// `(eb, mb)` pairs. Drives emission of the conversion helpers in the
/// WGSL and MSL backends.
pub fn kernel_fp8_formats(kernel: &crate::KernelDef) -> Vec<(u32, u32)> {
    use crate::{KernelOp, ScalarType};
    let mut e5m2 = false;
    let mut e4m3 = false;
    fn scan(ops: &[crate::KernelOp], e5m2: &mut bool, e4m3: &mut bool) {
        for op in ops {
            match op {
                KernelOp::Load { ty, .. } | KernelOp::Store { ty, .. } => match ty {
                    ScalarType::FP8E5M2 => *e5m2 = true,
                    ScalarType::FP8E4M3 => *e4m3 = true,
                    _ => {}
                },
                KernelOp::Branch {
                    then_ops, else_ops, ..
                } => {
                    scan(then_ops, e5m2, e4m3);
                    scan(else_ops, e5m2, e4m3);
                }
                KernelOp::Loop { body, .. } => scan(body, e5m2, e4m3),
                _ => {}
            }
        }
    }
    scan(&kernel.body, &mut e5m2, &mut e4m3);
    let mut out = Vec::new();
    if e5m2 {
        out.push((5, 2));
    }
    if e4m3 {
        out.push((4, 3));
    }
    out
}

/// WGSL: an unpack `fn qa_fp8_<tag>_unpack(bits: u32) -> f32` and a pack
/// `fn qa_fp8_<tag>_pack(val: f32) -> u32`, mirroring the branchless
/// reference line for line (`if c {a} else {b}` → `select(b, a, c)`).
pub fn wgsl_fp8_helpers(eb: u32, mb: u32) -> String {
    let tag = fp8_tag(eb, mb);
    let bias = (1u32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;
    let mant_mask = (1u32 << mb) - 1;
    let mb_i = mb;
    let lo_shift = 23 - mb;

    // round-to-nearest-even right shift, branchless over s.
    // (s is always a small constant or a bounded value here, but keep the
    // clamp so far-underflow shifts stay defined.)
    let rne = format!(
        "fn qa_rne_{tag}(v: u32, s: u32) -> u32 {{\n  \
           let big = s >= 32u;\n  \
           let sc = select(s, 31u, big);\n  \
           let kept = v >> sc;\n  \
           let rem = v & ((1u << sc) - 1u);\n  \
           let half = 1u << select(sc - 1u, 0u, sc == 0u);\n  \
           let roundup = (rem > half) || ((rem == half) && ((kept & 1u) == 1u));\n  \
           let r = kept + select(0u, 1u, roundup);\n  \
           let r0 = select(r, v, s == 0u);\n  \
           return select(r0, 0u, big);\n\
         }}\n"
    );

    // unpack: see dtype::fp8_to_f32.
    let unpack = {
        // Unrolled leading-bit scan over the mb mantissa bits.
        let mut lead = String::from("  var lead: u32 = 0u;\n");
        for i in 0..mb_i {
            lead.push_str(&format!(
                "  let s{i} = (mant >> {i}u) & 1u; lead = (s{i} * {i}u) | ((1u - s{i}) * lead);\n"
            ));
        }
        format!(
            "fn qa_fp8_{tag}_unpack(bits: u32) -> f32 {{\n  \
               let sign = (bits >> {sm}u) & 1u;\n  \
               let exp = (bits >> {mb_i}u) & {exp_mask}u;\n  \
               let mant = bits & {mant_mask}u;\n  \
               let fsign = sign << 31u;\n  \
               let norm = fsign | ((exp + 127u - {bias}u) << 23u) | (mant << {lo_shift}u);\n  \
               let inf_mant = select(0u, 0x00400000u, mant != 0u);\n  \
               let infnan = fsign | (0xFFu << 23u) | inf_mant;\n\
             {lead}  \
               let shifts = {mb_i}u - lead;\n  \
               let e_sub = i32(1u) - i32({bias}u) - i32({mb_i}u) - i32(shifts);\n  \
               let m_sub = (mant << shifts) & {mant_mask}u;\n  \
               let sub = fsign | (u32(e_sub + i32({mb_i}u) + 127) << 23u) | (m_sub << {lo_shift}u);\n  \
               let is_inf = exp == {exp_mask}u;\n  \
               let is_zero = (exp == 0u) && (mant == 0u);\n  \
               let is_sub = (exp == 0u) && (mant != 0u);\n  \
               var out = norm;\n  \
               out = select(out, sub, is_sub);\n  \
               out = select(out, infnan, is_inf);\n  \
               out = select(out, fsign, is_zero);\n  \
               return bitcast<f32>(out);\n\
             }}\n",
            sm = eb + mb,
        )
    };

    // pack: see dtype::f32_to_fp8.
    let pack = format!(
        "fn qa_fp8_{tag}_pack(val: f32) -> u32 {{\n  \
           let b = bitcast<u32>(val);\n  \
           let sign = (b >> 31u) & 1u;\n  \
           let sign_slot = sign << {sm}u;\n  \
           let fexp = i32((b >> 23u) & 0xFFu);\n  \
           let fmant = b & 0x007FFFFFu;\n  \
           let target_exp = (fexp - 127) + i32({bias}u);\n  \
           let rnd_n = qa_rne_{tag}(fmant, {lo_shift}u);\n  \
           let carry = (rnd_n >> {mb_i}u) != 0u;\n  \
           let out_exp_n = u32(target_exp) + select(0u, 1u, carry);\n  \
           let out_mant_n = select(rnd_n, 0u, carry);\n  \
           let normal = select(\n    \
               sign_slot | (out_exp_n << {mb_i}u) | (out_mant_n & {mant_mask}u),\n    \
               sign_slot | ({exp_mask}u << {mb_i}u),\n    \
               out_exp_n >= {exp_mask}u);\n  \
           let signif = fmant | 0x00800000u;\n  \
           let shift = i32({lo_shift}u) + (1 - target_exp);\n  \
           let sub = select(\n    \
               sign_slot | qa_rne_{tag}(signif, u32(shift)),\n    \
               sign_slot,\n    \
               shift > 31);\n  \
           let nan_m = select(0u, 1u << {mb_minus1}u, fmant != 0u);\n  \
           let infnan = sign_slot | ({exp_mask}u << {mb_i}u) | nan_m;\n  \
           let ovf = sign_slot | ({exp_mask}u << {mb_i}u);\n  \
           let is_infnan = fexp == 0xFF;\n  \
           let is_zero = (fexp == 0) && (fmant == 0u);\n  \
           let is_ovf = target_exp >= i32({exp_mask}u);\n  \
           let is_sub = target_exp <= 0;\n  \
           var out = normal;\n  \
           out = select(out, sub, is_sub);\n  \
           out = select(out, ovf, is_ovf);\n  \
           out = select(out, sign_slot, is_zero);\n  \
           out = select(out, infnan, is_infnan);\n  \
           return out;\n\
         }}\n",
        sm = eb + mb,
        mb_minus1 = mb - 1,
    );

    format!("{rne}{unpack}{pack}")
}

/// MSL: same two helpers as `inline` device functions.
pub fn msl_fp8_helpers(eb: u32, mb: u32) -> String {
    let tag = fp8_tag(eb, mb);
    let bias = (1u32 << (eb - 1)) - 1;
    let exp_mask = (1u32 << eb) - 1;
    let mant_mask = (1u32 << mb) - 1;
    let lo_shift = 23 - mb;
    let sm = eb + mb;

    let rne = format!(
        "inline uint qa_rne_{tag}(uint v, uint s) {{\n  \
           bool big = s >= 32u;\n  \
           uint sc = big ? 31u : s;\n  \
           uint kept = v >> sc;\n  \
           uint rem = v & ((1u << sc) - 1u);\n  \
           uint hbit = 1u << (sc == 0u ? 0u : (sc - 1u));\n  \
           bool roundup = (rem > hbit) || ((rem == hbit) && ((kept & 1u) == 1u));\n  \
           uint r = kept + (roundup ? 1u : 0u);\n  \
           uint r0 = (s == 0u) ? v : r;\n  \
           return big ? 0u : r0;\n\
         }}\n"
    );

    let mut lead = String::from("  uint lead = 0u;\n");
    for i in 0..mb {
        lead.push_str(&format!(
            "  uint s{i} = (mant >> {i}u) & 1u; lead = (s{i} * {i}u) | ((1u - s{i}) * lead);\n"
        ));
    }
    let unpack = format!(
        "inline float qa_fp8_{tag}_unpack(uint bits) {{\n  \
           uint sign = (bits >> {sm}u) & 1u;\n  \
           uint exp = (bits >> {mb}u) & {exp_mask}u;\n  \
           uint mant = bits & {mant_mask}u;\n  \
           uint fsign = sign << 31u;\n  \
           uint norm = fsign | ((exp + 127u - {bias}u) << 23u) | (mant << {lo_shift}u);\n  \
           uint inf_mant = (mant != 0u) ? 0x00400000u : 0u;\n  \
           uint infnan = fsign | (0xFFu << 23u) | inf_mant;\n\
         {lead}  \
           uint shifts = {mb}u - lead;\n  \
           int e_sub = 1 - int({bias}u) - int({mb}u) - int(shifts);\n  \
           uint m_sub = (mant << shifts) & {mant_mask}u;\n  \
           uint sub = fsign | (uint(e_sub + int({mb}u) + 127) << 23u) | (m_sub << {lo_shift}u);\n  \
           bool is_inf = exp == {exp_mask}u;\n  \
           bool is_zero = (exp == 0u) && (mant == 0u);\n  \
           bool is_sub = (exp == 0u) && (mant != 0u);\n  \
           uint out = norm;\n  \
           out = is_sub ? sub : out;\n  \
           out = is_inf ? infnan : out;\n  \
           out = is_zero ? fsign : out;\n  \
           return as_type<float>(out);\n\
         }}\n"
    );

    let pack = format!(
        "inline uint qa_fp8_{tag}_pack(float val) {{\n  \
           uint b = as_type<uint>(val);\n  \
           uint sign = (b >> 31u) & 1u;\n  \
           uint sign_slot = sign << {sm}u;\n  \
           int fexp = int((b >> 23u) & 0xFFu);\n  \
           uint fmant = b & 0x007FFFFFu;\n  \
           int target_exp = (fexp - 127) + int({bias}u);\n  \
           uint rnd_n = qa_rne_{tag}(fmant, {lo_shift}u);\n  \
           bool carry = (rnd_n >> {mb}u) != 0u;\n  \
           uint out_exp_n = uint(target_exp) + (carry ? 1u : 0u);\n  \
           uint out_mant_n = carry ? 0u : rnd_n;\n  \
           uint normal = (out_exp_n >= {exp_mask}u)\n    \
               ? (sign_slot | ({exp_mask}u << {mb}u))\n    \
               : (sign_slot | (out_exp_n << {mb}u) | (out_mant_n & {mant_mask}u));\n  \
           uint signif = fmant | 0x00800000u;\n  \
           int shift = int({lo_shift}u) + (1 - target_exp);\n  \
           uint sub = (shift > 31) ? sign_slot : (sign_slot | qa_rne_{tag}(signif, uint(shift)));\n  \
           uint nan_m = (fmant != 0u) ? (1u << {mb_minus1}u) : 0u;\n  \
           uint infnan = sign_slot | ({exp_mask}u << {mb}u) | nan_m;\n  \
           uint ovf = sign_slot | ({exp_mask}u << {mb}u);\n  \
           bool is_infnan = fexp == 0xFF;\n  \
           bool is_zero = (fexp == 0) && (fmant == 0u);\n  \
           bool is_ovf = target_exp >= int({exp_mask}u);\n  \
           bool is_sub = target_exp <= 0;\n  \
           uint out = normal;\n  \
           out = is_sub ? sub : out;\n  \
           out = is_ovf ? ovf : out;\n  \
           out = is_zero ? sign_slot : out;\n  \
           out = is_infnan ? infnan : out;\n  \
           return out;\n\
         }}\n",
        mb_minus1 = mb - 1,
    );

    format!("{rne}{unpack}{pack}")
}

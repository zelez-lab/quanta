//! Radix-2 Cooley-Tukey FFT on the GPU (split re/im, power-of-2).
//!
//! Two hand-built kernels, orchestrated by the host:
//!
//! 1. **Bit-reversal** — thread `i` writes `out[i] = in[bitrev(i, log2n)]`,
//!    reordering the input so the in-place butterflies land in natural order.
//!    `bitrev` is computed in-kernel by a `log2n`-step shift loop.
//! 2. **Butterfly stage** — dispatched `log2n` times. Stage with group size
//!    `m` runs `N/2` threads (one per butterfly): thread `t` owns group
//!    `t/half`, position `t%half` (`half = m/2`), combines `x[i0]` and
//!    `x[i0+half]` with twiddle `w = exp(sign·2πi·j/m)` (`sin`/`cos` in-kernel).
//!    In place — each butterfly touches a disjoint index pair within the stage.
//!
//! Inverse FFT flips the twiddle sign and divides by N (a final scale kernel).
//! Sizes must be a power of 2; others return `NotSupported`.

use core::f32::consts::PI;

use quanta::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MathFn, Reg, ScalarType, serialize_kernel,
};

const F32: ScalarType = ScalarType::F32;
const U32: ScalarType = ScalarType::U32;

/// Forward FFT. `re`/`im` are split complex inputs of length `n` (a power of
/// 2); returns the transformed `(re, im)` as host vectors. Errors if `n` is
/// not a power of 2 or the lengths disagree.
pub fn fft(gpu: &Gpu, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run(gpu, re, im, -1.0, false)
}

/// Inverse FFT (`+` twiddle sign, `1/n` scale): `ifft(fft(x)) == x`.
pub fn ifft(gpu: &Gpu, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    run(gpu, re, im, 1.0, true)
}

fn run(
    gpu: &Gpu,
    re: &[f32],
    im: &[f32],
    sign: f32,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
    if re.len() != im.len() {
        return Err(QuantaError::invalid_param("fft: re/im length mismatch"));
    }
    let n = re.len();
    if n == 0 {
        return Ok((vec![], vec![]));
    }
    if !n.is_power_of_two() {
        return Err(QuantaError::not_supported(
            "fft: length must be a power of 2",
        ));
    }
    let log2n = n.trailing_zeros();

    // Buffers. Bit-reverse into the work buffers (out), then butterfly in place.
    let in_re = gpu.field::<f32>(n)?;
    let in_im = gpu.field::<f32>(n)?;
    let work_re = gpu.field::<f32>(n)?;
    let work_im = gpu.field::<f32>(n)?;
    in_re.write(re)?;
    in_im.write(im)?;

    // 1. Bit-reversal permutation: work[i] = in[bitrev(i)].
    bitrev(gpu, &in_re, &work_re, n, log2n)?;
    bitrev(gpu, &in_im, &work_im, n, log2n)?;

    // 2. log2n butterfly stages, in place on work.
    let mut m = 2u32;
    for _ in 0..log2n {
        butterfly_stage(gpu, &work_re, &work_im, n, m, sign)?;
        m *= 2;
    }

    // 3. Inverse: divide by n.
    if inverse {
        scale(gpu, &work_re, n, 1.0 / n as f32)?;
        scale(gpu, &work_im, n, 1.0 / n as f32)?;
    }

    Ok((work_re.read()?, work_im.read()?))
}

/// `out[i] = in[bitrev(i, log2n)]` — reverse the low `log2n` bits of `i`.
fn bitrev(
    gpu: &Gpu,
    input: &Field<f32>,
    out: &Field<f32>,
    n: usize,
    log2n: u32,
) -> Result<(), QuantaError> {
    // r0=i; r1=rev (acc=0); r2=x (running copy of i); loop log2n times:
    //   rev = (rev << 1) | (x & 1); x >>= 1.  then out[i] = in[rev].
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(0),
        }, // rev
        KernelOp::Copy {
            dst: Reg(2),
            src: Reg(0),
            ty: U32,
        }, // x = i
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::U32(1),
        }, // const 1
        KernelOp::Const {
            dst: Reg(4),
            value: ConstValue::U32(log2n),
        },
        KernelOp::Loop {
            count: Reg(4),
            iter_reg: Reg(5),
            body: vec![
                // rev = (rev << 1) | (x & 1)
                KernelOp::BinOp {
                    dst: Reg(6),
                    a: Reg(1),
                    b: Reg(3),
                    op: BinOp::Shl,
                    ty: U32,
                },
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(2),
                    b: Reg(3),
                    op: BinOp::BitAnd,
                    ty: U32,
                },
                KernelOp::BinOp {
                    dst: Reg(1),
                    a: Reg(6),
                    b: Reg(7),
                    op: BinOp::BitOr,
                    ty: U32,
                },
                // x >>= 1
                KernelOp::BinOp {
                    dst: Reg(2),
                    a: Reg(2),
                    b: Reg(3),
                    op: BinOp::Shr,
                    ty: U32,
                },
            ],
        },
        KernelOp::Load {
            dst: Reg(8),
            field: 0,
            index: Reg(1),
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(8),
            ty: F32,
        },
    ];
    let def = KernelDef {
        name: "fft_bitrev".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "in".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 1,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 9,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    dispatch2(gpu, &def, input, out, n as u32)
}

/// One radix-2 butterfly stage with group size `m`, in place on `(re, im)`.
/// Dispatched with `n/2` threads (one butterfly each).
fn butterfly_stage(
    gpu: &Gpu,
    re: &Field<f32>,
    im: &Field<f32>,
    n: usize,
    m: u32,
    sign: f32,
) -> Result<(), QuantaError> {
    let half = m / 2;
    // r0=t (butterfly id); r1=half; r2=m;
    // g = t / half; j = t % half; base = g*m; i0 = base + j; i1 = i0 + half;
    // theta = sign·2π·j/m; w = (cos θ, sin θ);
    // a = x[i0]; b = x[i1]; wb = w·b; x[i0]=a+wb; x[i1]=a−wb. (re & im fields.)
    let twiddle_c = sign * 2.0 * PI; // baked: sign·2π; divide by m, mul by j in-kernel
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(half),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(m),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Div,
            ty: U32,
        }, // g
        KernelOp::BinOp {
            dst: Reg(4),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Rem,
            ty: U32,
        }, // j
        KernelOp::BinOp {
            dst: Reg(5),
            a: Reg(3),
            b: Reg(2),
            op: BinOp::Mul,
            ty: U32,
        }, // base
        KernelOp::BinOp {
            dst: Reg(6),
            a: Reg(5),
            b: Reg(4),
            op: BinOp::Add,
            ty: U32,
        }, // i0
        KernelOp::BinOp {
            dst: Reg(7),
            a: Reg(6),
            b: Reg(1),
            op: BinOp::Add,
            ty: U32,
        }, // i1
        // theta = (sign·2π / m) · j  →  compute as (j_f * twiddle_c) / m_f
        KernelOp::Cast {
            dst: Reg(8),
            src: Reg(4),
            from: U32,
            to: F32,
        }, // j_f
        KernelOp::Cast {
            dst: Reg(9),
            src: Reg(2),
            from: U32,
            to: F32,
        }, // m_f
        KernelOp::Const {
            dst: Reg(10),
            value: ConstValue::F32(twiddle_c),
        },
        KernelOp::BinOp {
            dst: Reg(11),
            a: Reg(8),
            b: Reg(10),
            op: BinOp::Mul,
            ty: F32,
        }, // j·2πsign
        KernelOp::BinOp {
            dst: Reg(12),
            a: Reg(11),
            b: Reg(9),
            op: BinOp::Div,
            ty: F32,
        }, // /m = θ
        KernelOp::MathCall {
            dst: Reg(13),
            func: MathFn::Cos,
            args: vec![Reg(12)],
            ty: F32,
        }, // w_re
        KernelOp::MathCall {
            dst: Reg(14),
            func: MathFn::Sin,
            args: vec![Reg(12)],
            ty: F32,
        }, // w_im
        // a = (re[i0], im[i0]); b = (re[i1], im[i1])
        KernelOp::Load {
            dst: Reg(15),
            field: 0,
            index: Reg(6),
            ty: F32,
        }, // a_re
        KernelOp::Load {
            dst: Reg(16),
            field: 1,
            index: Reg(6),
            ty: F32,
        }, // a_im
        KernelOp::Load {
            dst: Reg(17),
            field: 0,
            index: Reg(7),
            ty: F32,
        }, // b_re
        KernelOp::Load {
            dst: Reg(18),
            field: 1,
            index: Reg(7),
            ty: F32,
        }, // b_im
        // wb = w·b: wb_re = w_re·b_re − w_im·b_im; wb_im = w_re·b_im + w_im·b_re
        KernelOp::BinOp {
            dst: Reg(19),
            a: Reg(13),
            b: Reg(17),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(20),
            a: Reg(14),
            b: Reg(18),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(21),
            a: Reg(19),
            b: Reg(20),
            op: BinOp::Sub,
            ty: F32,
        }, // wb_re
        KernelOp::BinOp {
            dst: Reg(22),
            a: Reg(13),
            b: Reg(18),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(23),
            a: Reg(14),
            b: Reg(17),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(24),
            a: Reg(22),
            b: Reg(23),
            op: BinOp::Add,
            ty: F32,
        }, // wb_im
        // x[i0] = a + wb ; x[i1] = a − wb
        KernelOp::BinOp {
            dst: Reg(25),
            a: Reg(15),
            b: Reg(21),
            op: BinOp::Add,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(26),
            a: Reg(16),
            b: Reg(24),
            op: BinOp::Add,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(27),
            a: Reg(15),
            b: Reg(21),
            op: BinOp::Sub,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(28),
            a: Reg(16),
            b: Reg(24),
            op: BinOp::Sub,
            ty: F32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(6),
            src: Reg(25),
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(6),
            src: Reg(26),
            ty: F32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(7),
            src: Reg(27),
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(7),
            src: Reg(28),
            ty: F32,
        },
    ];
    let def = KernelDef {
        name: "fft_butterfly".into(),
        params: vec![
            KernelParam::FieldWrite {
                name: "re".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "im".into(),
                slot: 1,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 29,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, re);
    wave.bind(1, im);
    gpu.dispatch(&wave, (n / 2) as u32)?.wait()?;
    Ok(())
}

/// In-place scale `x[i] *= s` (the inverse FFT's 1/n).
fn scale(gpu: &Gpu, x: &Field<f32>, n: usize, s: f32) -> Result<(), QuantaError> {
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Load {
            dst: Reg(1),
            field: 0,
            index: Reg(0),
            ty: F32,
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::F32(s),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(1),
            b: Reg(2),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(0),
            src: Reg(3),
            ty: F32,
        },
    ];
    let def = KernelDef {
        name: "fft_scale".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "xo".into(),
                slot: 1,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, x);
    wave.bind(1, x); // in place
    gpu.dispatch(&wave, n as u32)?.wait()?;
    Ok(())
}

/// Dispatch a 2-buffer (read in, write out) kernel over `n` threads.
fn dispatch2(
    gpu: &Gpu,
    def: &KernelDef,
    input: &Field<f32>,
    out: &Field<f32>,
    n: u32,
) -> Result<(), QuantaError> {
    let bytes = serialize_kernel(def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, input);
    wave.bind(1, out);
    gpu.dispatch(&wave, n)?.wait()?;
    Ok(())
}

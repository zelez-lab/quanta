//! Mixed-precision GEMM (narrow inputs, f32 accumulate).
//!
//! A,B are stored in a narrow dtype and converted to f32 on load; the inner
//! product accumulates in f32 and C is f32. This is the standard ML
//! mixed-precision path (16-bit weights/activations, f32 math). The output
//! contract is the same real-arithmetic `gemmEntry`; the narrow dtype is an
//! implementation detail of *how* the entry is computed. The forward-error
//! bound (`specs/verify/lean/Quanta/Blas/GemmMixed.lean`) splits the entry
//! error into the proven f32 GEMM error over the quantised inputs plus the
//! input-quantisation error — so each dtype reuses the GEMM proof.
//!
//! Narrow floats are stored **tightly packed at their native width**: bf16 in
//! a `Field<u16>` (2 bytes/elem), matching `Load { ty: BF16 }`'s 2-byte stride
//! on every backend. (fp8 will use `Field<u8>`.) Each load yields an f32-valued
//! register — the emitter auto-converts. This is the naive
//! one-thread-per-output-entry kernel — the same shape the f32 path started
//! from; the tiled / cooperative-matrix paths are a later perf fork.

use quanta::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};

/// Input element dtype for a mixed-precision GEMM. Output (C) is always f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmInputType {
    /// bfloat16 — 1 sign / 8 exponent / 7 mantissa (top 16 bits of an f32).
    Bf16,
    /// IEEE half — 1 sign / 5 exponent / 10 mantissa.
    F16,
}

impl GemmInputType {
    fn scalar_type(self) -> ScalarType {
        match self {
            GemmInputType::Bf16 => ScalarType::BF16,
            GemmInputType::F16 => ScalarType::F16,
        }
    }

    fn tag(self) -> &'static str {
        match self {
            GemmInputType::Bf16 => "bf16",
            GemmInputType::F16 => "f16",
        }
    }
}

/// `gemm_mixed`: `C ← α·A·B + β·C`, A `m×k` and B `k×n` row-major in the
/// narrow `dtype`, C `m×n` in f32. A and B are `Field<u16>` holding one bf16
/// bit pattern per (2-byte) element; C is `Field<f32>` (read for `β·C`,
/// overwritten with the result). Errors on a shape mismatch.
#[allow(clippy::too_many_arguments)]
pub fn gemm_mixed(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<u16>,
    b: &Field<u16>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    if a.len() != mu * ku {
        return Err(QuantaError::invalid_param(
            "gemm_mixed: A length must be m*k",
        ));
    }
    if b.len() != ku * nu {
        return Err(QuantaError::invalid_param(
            "gemm_mixed: B length must be k*n",
        ));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param(
            "gemm_mixed: C length must be m*n",
        ));
    }
    if mu * nu == 0 || ku == 0 {
        return Ok(());
    }

    let def = build_gemm_mixed_def(dtype, n, k, alpha, beta);
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is read (β·C) and written
    gpu.dispatch(&wave, m * n)?.wait()?;
    Ok(())
}

/// `gemv_mixed`: `y ← α·A·x + β·y`, A `m×n` row-major in `dtype`, x in
/// `dtype`, y in f32. A gemv is a gemm with one output column (N=1), so this
/// routes into `gemm_mixed(m, 1, n, …)` — same kernel, same proven bound
/// (`Quanta.Blas.gemvEntry_eq_gemmEntry`).
#[allow(clippy::too_many_arguments)]
pub fn gemv_mixed(
    gpu: &Gpu,
    dtype: GemmInputType,
    m: u32,
    n: u32,
    alpha: f32,
    a: &Field<u16>,
    x: &Field<u16>,
    beta: f32,
    y: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu) = (m as usize, n as usize);
    if a.len() != mu * nu {
        return Err(QuantaError::invalid_param(
            "gemv_mixed: A length must be m*n",
        ));
    }
    if x.len() != nu {
        return Err(QuantaError::invalid_param("gemv_mixed: x length must be n"));
    }
    if y.len() != mu {
        return Err(QuantaError::invalid_param("gemv_mixed: y length must be m"));
    }
    gemm_mixed(gpu, dtype, m, 1, n, alpha, a, x, beta, y)
}

/// Naive mixed-precision GEMM kernel: one thread per output entry `C[row,col]`
/// (`m·n` threads). A/B are loaded as `dtype` (auto-converted to f32 in
/// register); the inner product accumulates in f32; the result is the f32
/// `α·acc + β·C[i]`. `n`, `k`, `alpha`, `beta` are baked in as constants (the
/// def is built per dispatch, so they're known).
fn build_gemm_mixed_def(dtype: GemmInputType, n: u32, k: u32, alpha: f32, beta: f32) -> KernelDef {
    let in_ty = dtype.scalar_type();
    use ScalarType::{F32, U32};
    // Registers:
    //   r0 = i (flat output index, = quark_id)
    //   r1 = n, r2 = k (consts)
    //   r3 = row = i / n,  r4 = col = i % n
    //   r5 = acc (f32), r6 = p (loop iter, 0..k)
    //   r7 = a_idx = row*k + p,  r8 = b_idx = p*n + col
    //   r9 = a_val (f32 from bf16), r10 = b_val (f32), r11 = a_val*b_val
    //   r12 = alpha, r13 = beta (consts)
    //   r14 = c_val = C[i], r15 = alpha*acc, r16 = beta*c, r17 = result
    let body = vec![
        KernelOp::QuarkId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(n),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(k),
        },
        KernelOp::BinOp {
            dst: Reg(3),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Div,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(4),
            a: Reg(0),
            b: Reg(1),
            op: BinOp::Rem,
            ty: U32,
        },
        // acc = 0.0
        KernelOp::Const {
            dst: Reg(5),
            value: ConstValue::F32(0.0),
        },
        // for p in 0..k: acc += A[row*k+p] * B[p*n+col]
        KernelOp::Loop {
            count: Reg(2),
            iter_reg: Reg(6),
            body: vec![
                // a_idx = row*k + p
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(3),
                    b: Reg(2),
                    op: BinOp::Mul,
                    ty: U32,
                },
                KernelOp::BinOp {
                    dst: Reg(7),
                    a: Reg(7),
                    b: Reg(6),
                    op: BinOp::Add,
                    ty: U32,
                },
                // b_idx = p*n + col
                KernelOp::BinOp {
                    dst: Reg(8),
                    a: Reg(6),
                    b: Reg(1),
                    op: BinOp::Mul,
                    ty: U32,
                },
                KernelOp::BinOp {
                    dst: Reg(8),
                    a: Reg(8),
                    b: Reg(4),
                    op: BinOp::Add,
                    ty: U32,
                },
                // load narrow → register, then cast to f32. bf16/fp8 load
                // into an f32 register already (msl_name = float), so the cast
                // is a no-op there; f16 loads into a native `half` register, so
                // the cast forces the f32 promotion BEFORE the multiply — the
                // product must be f32 (the oracle accumulates in f32), not
                // half-precision.
                KernelOp::Load {
                    dst: Reg(9),
                    field: 0,
                    index: Reg(7),
                    ty: in_ty,
                },
                KernelOp::Load {
                    dst: Reg(10),
                    field: 1,
                    index: Reg(8),
                    ty: in_ty,
                },
                KernelOp::Cast {
                    dst: Reg(18),
                    src: Reg(9),
                    from: in_ty,
                    to: F32,
                },
                KernelOp::Cast {
                    dst: Reg(19),
                    src: Reg(10),
                    from: in_ty,
                    to: F32,
                },
                KernelOp::BinOp {
                    dst: Reg(11),
                    a: Reg(18),
                    b: Reg(19),
                    op: BinOp::Mul,
                    ty: F32,
                },
                KernelOp::BinOp {
                    dst: Reg(5),
                    a: Reg(5),
                    b: Reg(11),
                    op: BinOp::Add,
                    ty: F32,
                },
            ],
        },
        // result = alpha*acc + beta*C[i]
        KernelOp::Const {
            dst: Reg(12),
            value: ConstValue::F32(alpha),
        },
        KernelOp::Const {
            dst: Reg(13),
            value: ConstValue::F32(beta),
        },
        KernelOp::Load {
            dst: Reg(14),
            field: 2,
            index: Reg(0),
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(15),
            a: Reg(12),
            b: Reg(5),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(16),
            a: Reg(13),
            b: Reg(14),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(17),
            a: Reg(15),
            b: Reg(16),
            op: BinOp::Add,
            ty: F32,
        },
        KernelOp::Store {
            field: 2,
            index: Reg(0),
            src: Reg(17),
            ty: F32,
        },
    ];
    KernelDef {
        name: format!("blas_gemm_mixed_{}", dtype.tag()),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: in_ty,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: in_ty,
            },
            KernelParam::FieldWrite {
                name: "c".into(),
                slot: 2,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 20,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

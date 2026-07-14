//! Shared naive mixed-precision GEMM kernel builder + dispatch, used by both
//! the float (`mixed`) and quantized (`mixed_quant`) entries.
//!
//! One thread per output entry `C[row,col]` (`m·n` threads). A/B are loaded as
//! `in_ty` and cast to f32 (bf16/fp8 already load into an f32 register, so the
//! cast is a no-op; f16 loads a native `half` and i8 loads an `int`, so the
//! cast forces the f32 promotion BEFORE the multiply — the product must
//! accumulate in f32). The result is the f32 `α·acc + β·C[i]`. `n`, `k`,
//! `alpha`, `beta` are baked in as constants (the def is built per dispatch).
//!
//! Quantized inputs fold the dequantisation scales into `alpha`: dequantised
//! `A·B = (sa·qa)·(sb·qb)` summed `= sa·sb·Σ qa·qb`, so passing
//! `alpha_eff = α·sa·sb` over the raw integer codes computes the dequantised
//! result with no per-element scale op.

use quanta_core::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};

/// Build the naive mixed kernel for input element type `in_ty` (tagged for the
/// kernel name), with `n`/`k`/`alpha`/`beta` as baked constants.
pub(crate) fn build_def(
    in_ty: ScalarType,
    tag: &str,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
) -> KernelDef {
    use ScalarType::{F32, U32};
    // r0=i; r1=n; r2=k; r3=row=i/n; r4=col=i%n; r5=acc(f32); r6=p (0..k);
    // r7=a_idx; r8=b_idx; r9/r10=loaded; r18/r19=cast f32; r11=product;
    // r12=alpha; r13=beta; r14=C[i]; r15=α·acc; r16=β·C; r17=result.
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
        KernelOp::Const {
            dst: Reg(5),
            value: ConstValue::F32(0.0),
        },
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
        name: format!("blas_gemm_mixed_{tag}"),
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

/// Narrow storage elements that can be widened into the WGSL u32-slot
/// layout (one element per 32-bit word, raw bits zero-extended into the
/// low bits). Implemented for every `T` that rides `dispatch`; only the
/// genuinely narrow dtypes (bf16 / fp8) ever get widened.
pub(crate) trait NarrowStorageBits: quanta_core::GpuType + Copy {
    fn widen_to_word(self) -> u32;
}
impl NarrowStorageBits for u16 {
    fn widen_to_word(self) -> u32 {
        self as u32
    }
}
impl NarrowStorageBits for u8 {
    fn widen_to_word(self) -> u32 {
        self as u32
    }
}
impl NarrowStorageBits for i32 {
    // int8 codes ride a Field<i32> (word-sized already); never widened.
    fn widen_to_word(self) -> u32 {
        self as u32
    }
}

/// Shared dispatch for any narrow input storage type `T`. Validates shapes,
/// builds the per-dtype `KernelDef`, binds and dispatches `m·n` threads. The
/// kernel reads A/B at the dtype's native stride (`scalar_size`), so `T` must
/// match `in_ty`'s storage width — the public wrappers enforce that.
///
/// WGSL exception: WebGPU cannot hold 16-/8-bit storage elements, so its
/// kernels read bf16/fp8 one-element-per-u32-word. When the active backend
/// reports that layout (`Gpu::narrow_storage_u32_slot`), the tight A/B data
/// is re-uploaded expanded before binding — a documented host-side repack
/// that keeps the public tight-`Field` contract identical on every backend.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch<T: NarrowStorageBits>(
    gpu: &Gpu,
    in_ty: ScalarType,
    tag: &str,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<T>,
    b: &Field<T>,
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
    let def = build_def(in_ty, tag, n, k, alpha, beta);
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;

    // WGSL u32-slot repack (WebGPU only): expand tight narrow input to
    // one element per word. C is f32 and needs no repack.
    let is_narrow_float = matches!(
        in_ty,
        ScalarType::BF16 | ScalarType::FP8E5M2 | ScalarType::FP8E4M3
    );
    if is_narrow_float && gpu.narrow_storage_u32_slot() {
        let widen = |f: &Field<T>| -> Result<Field<u32>, QuantaError> {
            let words: Vec<u32> = f
                .read()?
                .into_iter()
                .map(NarrowStorageBits::widen_to_word)
                .collect();
            let wf = gpu.field::<u32>(words.len())?;
            wf.write(&words)?;
            Ok(wf)
        };
        let aw = widen(a)?;
        let bw = widen(b)?;
        wave.bind(0, &aw);
        wave.bind(1, &bw);
        wave.bind(2, c);
        gpu.dispatch(&wave, m * n)?.wait()?;
        return Ok(());
    }

    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is read (β·C) and written
    gpu.dispatch(&wave, m * n)?.wait()?;
    Ok(())
}

/// Dispatch for the **packed int4** path: A/B are `Field<u32>` holding 8 signed
/// nibbles per word, so their field length is `ceil(m·k / 8)` / `ceil(k·n / 8)`
/// words, not the logical element count. The kernel still indexes by the
/// logical `(m, n, k)` — `Load { ty: I4 }` resolves word `idx/8`, nibble
/// `idx%8` on every backend. C is f32.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_i4(
    gpu: &Gpu,
    tag: &str,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    a: &Field<u32>,
    b: &Field<u32>,
    beta: f32,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let words = |elems: usize| elems.div_ceil(8);
    if a.len() != words(mu * ku) {
        return Err(QuantaError::invalid_param(
            "gemm_quant4: A length must be ceil(m*k/8) packed words",
        ));
    }
    if b.len() != words(ku * nu) {
        return Err(QuantaError::invalid_param(
            "gemm_quant4: B length must be ceil(k*n/8) packed words",
        ));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param(
            "gemm_quant4: C length must be m*n",
        ));
    }
    if mu * nu == 0 || ku == 0 {
        return Ok(());
    }
    let def = build_def(ScalarType::I4, tag, n, k, alpha, beta);
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c);
    gpu.dispatch(&wave, m * n)?.wait()?;
    Ok(())
}

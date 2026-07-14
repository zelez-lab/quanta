//! Level-1 BLAS GPU ops (f32).
//!
//! Each op builds a `KernelDef` at runtime and dispatches it through the
//! Quanta JIT (`wave_jit`), mirroring the runtime-kernel style of
//! `quanta-array`'s ufuncs. The numerical contract each op satisfies is
//! proven in `specs/verify/lean/Quanta/Blas/Reference.lean`.
//!
//! - `scal` / `axpy` are **in-place** (bandwidth-bound: writing back into
//!   the input buffer saves an allocation + a buffer of write traffic, and
//!   matches classic BLAS semantics). They bind the input field to both the
//!   read and the write slot; each thread touches only its own index, so
//!   read-then-write at that index is hazard-free.
//! - `dot` / `nrm2` are **device-resident reductions**: multiply into a
//!   temp field on the GPU, then reduce with the `quanta-prims`
//!   `device_reduce_add_f32_field` path — the data never leaves the device.

use quanta_core::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};

const F32: ScalarType = ScalarType::F32;

/// `scal`: `x[i] ← alpha · x[i]`, in place on the device.
pub fn scal(gpu: &Gpu, alpha: f32, x: &Field<f32>) -> Result<(), QuantaError> {
    let n = x.len();
    if n == 0 {
        return Ok(());
    }
    // r0 = i; r1 = x[i]; r2 = alpha; r3 = alpha*x[i] → store to slot 1 (= x).
    let def = KernelDef {
        name: "blas_scal".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "xout".into(),
                slot: 1,
                scalar_type: F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(alpha),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(2),
                b: Reg(1),
                op: BinOp::Mul,
                ty: F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(3),
                ty: F32,
            },
        ],
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
    wave.bind(1, x); // in place: write back into x
    gpu.dispatch(&wave, n as u32)?.wait()?;
    Ok(())
}

/// `axpy`: `y[i] ← alpha · x[i] + y[i]`, in place on `y`. `x` and `y`
/// must have the same length.
pub fn axpy(gpu: &Gpu, alpha: f32, x: &Field<f32>, y: &Field<f32>) -> Result<(), QuantaError> {
    let n = x.len();
    if n != y.len() {
        return Err(QuantaError::invalid_param("axpy: x and y length mismatch"));
    }
    if n == 0 {
        return Ok(());
    }
    // r0=i; r1=x[i]; r2=y[i]; r3=alpha; r4=alpha*x[i]; r5=+y[i] → store to y.
    let def = KernelDef {
        name: "blas_axpy".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldRead {
                name: "y".into(),
                slot: 1,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "yout".into(),
                slot: 2,
                scalar_type: F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: F32,
            },
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::F32(alpha),
            },
            KernelOp::BinOp {
                dst: Reg(4),
                a: Reg(3),
                b: Reg(1),
                op: BinOp::Mul,
                ty: F32,
            },
            KernelOp::BinOp {
                dst: Reg(5),
                a: Reg(4),
                b: Reg(2),
                op: BinOp::Add,
                ty: F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(5),
                ty: F32,
            },
        ],
        body_source: None,
        next_reg: 6,
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
    wave.bind(1, y);
    wave.bind(2, y); // in place: write back into y
    gpu.dispatch(&wave, n as u32)?.wait()?;
    Ok(())
}

/// Elementwise product `prod[i] = x[i]·y[i]` into a fresh device field —
/// the first half of `dot` / `nrm2`, kept resident for the reduce.
fn hadamard(gpu: &Gpu, x: &Field<f32>, y: &Field<f32>) -> Result<Field<f32>, QuantaError> {
    let n = x.len();
    let prod = gpu.field::<f32>(n)?;
    let def = KernelDef {
        name: "blas_hadamard".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "x".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldRead {
                name: "y".into(),
                slot: 1,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "p".into(),
                slot: 2,
                scalar_type: F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Mul,
                ty: F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: F32,
            },
        ],
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
    wave.bind(1, y);
    wave.bind(2, &prod);
    gpu.dispatch(&wave, n as u32)?.wait()?;
    Ok(prod)
}

/// `dot`: inner product `Σ x[i]·y[i]`. Multiplies into a temp field on the
/// device, then reduces with the prims device-resident reduce — the data
/// never round-trips through host memory. `x` and `y` must match in length.
pub fn dot(gpu: &Gpu, x: &Field<f32>, y: &Field<f32>) -> Result<f32, QuantaError> {
    let n = x.len();
    if n != y.len() {
        return Err(QuantaError::invalid_param("dot: x and y length mismatch"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let prod = hadamard(gpu, x, y)?;
    quanta_prims::device_reduce_add_f32_field(gpu, &prod, n)
}

/// `nrm2`: Euclidean norm `sqrt(Σ x[i]²)` = `sqrt(dot(x, x))`.
pub fn nrm2(gpu: &Gpu, x: &Field<f32>) -> Result<f32, QuantaError> {
    let s = dot(gpu, x, x)?;
    Ok(s.sqrt())
}

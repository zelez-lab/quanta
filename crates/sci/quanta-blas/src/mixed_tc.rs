//! Tensor-core / cooperative-matrix GEMM, on the device's own shape.
//!
//! `C ← A·B + C`, row-major, with the fragment shape and the element types
//! taken from what the device enumerates — 8×8×8 f32 or f16 on Metal
//! (`simdgroup_matrix`), 16×16×16 (and non-square forms) with f16 inputs and
//! f32 accumulation on the `VK_KHR_cooperative_matrix` cards. Each subgroup
//! owns a `BR`×`BC` grid of accumulator fragments loaded from C — a
//! `(shape.m·BR)`×`(shape.n·BC)` output tile, so 32×32 on the 8×8×8 shape and
//! 64×64 on a 16×16×16 one. It sweeps K in `shape.k`-wide steps loading `BR` A
//! row-strip fragments + `BC` B col-strip fragments and issuing `BR·BC` MMAs —
//! `BR+BC` global fragment loads feed `BR·BC` MMAs (each fragment reused 4×
//! at 4×4), the arithmetic intensity that lets the tensor-core units beat the
//! SIMT tiled kernel (~1.5× at N=512 on M1 Pro, 553 vs 372 GFLOP/s). The
//! accumulators are stored back at the end.
//!
//! Reuses the proven `gemmEntry` contract — the differential test against the
//! f32 reference is the binding check, no new Lean. The kernel is hand-built
//! `KernelDef` IR: the cooperative-matrix ops (`CooperativeMatrixLoad` /
//! `CooperativeMMA` / `CooperativeMatrixStore`) have no `#[quanta::kernel]`
//! Rust surface and are subgroup-collective, so the WASM route can't express
//! them. A device that enumerates no shape with the requested element types —
//! or that does not fix a subgroup size — returns `NotSupported`, never a
//! silent fallback; the public `gemm` router is what takes the tiled path.
//!
//! Scope: `C += A·B` (α = β = 1), `m` a multiple of `shape.m·BR`, `n` of
//! `shape.n·BC`, `k` of `shape.k`. General α/β, tails, and threadgroup-shared
//! staging of the A/B tiles (which would cut the global fragment loads
//! further) are later increments.

use quanta_core::{CoopMatrixShape, Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MatrixFrag, Reg, ScalarType,
    serialize_kernel,
};

const BR: u32 = 4; // accumulator fragments down (rows) per subgroup
const BC: u32 = 4; // accumulator fragments across (cols) per subgroup

/// Fragment edge of the single-tile shared-staging probe below, which is
/// pinned to the 8×8×8 f32 shape (the one Metal enumerates).
const PROBE_FRAG: u32 = 8;

/// Host element types that fill a cooperative-matrix fragment.
///
/// The host type is the *storage* of the fragment element, not its
/// arithmetic: f16 has no host type in Rust, so a `u16` holds the IEEE
/// binary16 bit pattern — the same narrow-dtype storage contract
/// `Field<u16>` carries everywhere else in this crate.
pub trait TcElem: Copy + 'static {
    /// The fragment element type the kernel declares for this host type.
    const TY: ScalarType;
}

impl TcElem for f32 {
    const TY: ScalarType = ScalarType::F32;
}

impl TcElem for u16 {
    const TY: ScalarType = ScalarType::F16;
}

/// The shape this device would run a cooperative-matrix GEMM on with `A`/`B`
/// elements of `in_ty` and `C`/`D` of `acc_ty`: the **first** matching shape
/// in [`Gpu::cooperative_matrix_shapes`], the enumeration order being the
/// device's own preference. `None` when it lists none — ask before
/// allocating, since [`gemm_tc`] refuses rather than fall back.
pub fn tc_shape_for(gpu: &Gpu, in_ty: ScalarType, acc_ty: ScalarType) -> Option<CoopMatrixShape> {
    gpu.cooperative_matrix_shapes()
        .into_iter()
        .find(|s| s.ab_ty == in_ty && s.c_ty == acc_ty && s.result_ty == acc_ty)
}

/// `C ← A·B + C` on the cooperative-matrix path, on the first shape the
/// device enumerates with `A`/`B` elements of `In::TY` and `C`/`D` of
/// `Acc::TY` (see [`tc_shape_for`]). Requires `m` a multiple of
/// `shape.m·BR`, `n` of `shape.n·BC`, `k` of `shape.k`, and a device that
/// fixes its subgroup size; otherwise `NotSupported` — never a silent
/// fallback, which is what you want when you are measuring.
///
/// Register-blocked: each subgroup owns a `BR`×`BC` grid of accumulator
/// fragments, i.e. a 32×32 output tile on an 8×8×8 shape and 64×64 on a
/// 16×16×16 one. Per K-step it loads `BR` A row-strip fragments + `BC` B
/// col-strip fragments and issues `BR·BC` MMAs — 8 loads feed 16 MMAs at
/// 4×4 (each fragment reused 4×), the arithmetic intensity that lets the
/// tensor-core units beat the SIMT tiled path (~1.5× at N=512 on M1 Pro).
///
/// `In = Acc = f32` is the Metal-accelerated form; `In = u16` (binary16 bit
/// patterns) with `Acc = f32` is what the discrete cards enumerate first.
pub fn gemm_tc<In: TcElem, Acc: TcElem>(
    gpu: &Gpu,
    m: u32,
    n: u32,
    k: u32,
    a: &Field<In>,
    b: &Field<In>,
    c: &Field<Acc>,
) -> Result<(), QuantaError> {
    let Some(shape) = tc_shape_for(gpu, In::TY, Acc::TY) else {
        return Err(QuantaError::not_supported(format!(
            "no cooperative-matrix shape with {:?} inputs and {:?} accumulation on this device",
            In::TY,
            Acc::TY
        )));
    };
    let subgroup = gpu.subgroup_size();
    if subgroup == 0 {
        return Err(QuantaError::not_supported(
            "gemm_tc dispatches one subgroup per output tile, and this device does not fix its subgroup size",
        ));
    }
    let (tile_m, tile_n) = (u32::from(shape.m) * BR, u32::from(shape.n) * BC);
    let step_k = u32::from(shape.k);
    if !m.is_multiple_of(tile_m) || !n.is_multiple_of(tile_n) || !k.is_multiple_of(step_k) {
        return Err(QuantaError::not_supported(format!(
            "gemm_tc on this device's {}x{}x{} shape requires m multiple of {tile_m}, \
             n multiple of {tile_n}, k multiple of {step_k}",
            shape.m, shape.n, shape.k
        )));
    }
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    if a.len() != mu * ku {
        return Err(QuantaError::invalid_param("gemm_tc: A length must be m*k"));
    }
    if b.len() != ku * nu {
        return Err(QuantaError::invalid_param("gemm_tc: B length must be k*n"));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param("gemm_tc: C length must be m*n"));
    }
    if mu * nu == 0 || ku == 0 {
        return Ok(());
    }

    let mut def = build_tc_def(shape, n, k);
    // One subgroup per output tile: the workgroup is exactly the device's
    // subgroup, so the tile index is the workgroup index (`NucleusId`).
    def.workgroup_size = [subgroup, 1, 1];
    def.subgroup_size = Some(subgroup);
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is the accumulator (read + written)
    let tiles = (m / tile_m) * (n / tile_n);
    gpu.dispatch(&wave, tiles * subgroup)?.wait()?;
    Ok(())
}

/// `C ← A·B + C` on the all-f32 cooperative-matrix shape — the form Metal
/// accelerates (8×8×8, so `m`/`n` multiples of 32 and `k` a multiple of 8)
/// and the one the public [`gemm`](crate::gemm) router tries.
pub fn gemm_f32_tc(
    gpu: &Gpu,
    m: u32,
    n: u32,
    k: u32,
    a: &Field<f32>,
    b: &Field<f32>,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    gemm_tc::<f32, f32>(gpu, m, n, k, a, b, c)
}

/// Probe: single-tile (8×8×8, one subgroup) shared-staged GEMM `C = A·B`.
/// Stages the 8×8 A and B tiles into threadgroup memory cooperatively (32
/// threads copy 2 elements each), barriers, loads both fragments **from
/// shared**, one MMA into a zeroed accumulator, stores. Validates that
/// `simdgroup_load` works against a `shared_<id>` array before the full
/// shared-staged kernel is built on it. `#[doc(hidden)]` — test-only.
#[doc(hidden)]
pub fn gemm_f32_tc_shared_probe(
    gpu: &Gpu,
    a: &Field<f32>,
    b: &Field<f32>,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if !device_has_f32_8x8x8(gpu) {
        return Err(QuantaError::not_supported(
            "this kernel is built on 8x8x8 f32 cooperative matrices, which this device does not enumerate",
        ));
    }
    let def = build_tc_shared_probe_def();
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c);
    gpu.dispatch(&wave, 32)?.wait()?; // one subgroup
    Ok(())
}

fn build_tc_shared_probe_def() -> KernelDef {
    use ScalarType::{F32, U32};
    let f8: u8 = PROBE_FRAG as u8;
    // Shared tiles: id 0 = A (8×8), id 1 = B (8×8).
    let mut next = 50u32;
    let mut fresh = || {
        let r = next;
        next += 1;
        r
    };
    let tid = 0u32; // proton_id (in-workgroup thread index)
    let n8 = fresh(); // const 8
    let n64 = fresh(); // const 64
    let n32 = fresh(); // const 32
    let mut body = vec![
        KernelOp::SharedDecl {
            id: 0,
            ty: F32,
            count: 64,
        },
        KernelOp::SharedDecl {
            id: 1,
            ty: F32,
            count: 64,
        },
        KernelOp::ProtonId { dst: Reg(tid) },
        KernelOp::Const {
            dst: Reg(n8),
            value: ConstValue::U32(8),
        },
        KernelOp::Const {
            dst: Reg(n64),
            value: ConstValue::U32(64),
        },
        KernelOp::Const {
            dst: Reg(n32),
            value: ConstValue::U32(32),
        },
    ];
    // Cooperative copy: thread t copies element t and t+32 of each 8×8 tile.
    // The tiles are contiguous 8×8 row-major in both global (stride 8) and
    // shared, so the flat index is identical — a straight copy.
    for half in 0..2u32 {
        let e = fresh(); // element index = tid + half*32
        let off = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off),
            value: ConstValue::U32(half * 32),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(e),
            a: Reg(tid),
            b: Reg(off),
            op: BinOp::Add,
            ty: U32,
        });
        // a_sh[e] = a[e]; b_sh[e] = b[e]
        let av = fresh();
        body.push(KernelOp::Load {
            dst: Reg(av),
            field: 0,
            index: Reg(e),
            ty: F32,
        });
        body.push(KernelOp::SharedStore {
            id: 0,
            index: Reg(e),
            src: Reg(av),
            ty: F32,
        });
        let bv = fresh();
        body.push(KernelOp::Load {
            dst: Reg(bv),
            field: 1,
            index: Reg(e),
            ty: F32,
        });
        body.push(KernelOp::SharedStore {
            id: 1,
            index: Reg(e),
            src: Reg(bv),
            ty: F32,
        });
    }
    body.push(KernelOp::Barrier);
    // Load A,B fragments from shared (stride 8), zero accumulator, MMA, store.
    let zero_idx = fresh();
    body.push(KernelOp::Const {
        dst: Reg(zero_idx),
        value: ConstValue::U32(0),
    });
    let af = fresh();
    let bf = fresh();
    let acc = fresh();
    body.push(KernelOp::CooperativeMatrixLoad {
        dst: Reg(af),
        field: 0,
        index: Reg(zero_idx),
        stride: Reg(n8),
        frag: MatrixFrag::A,
        from_shared: true,
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
    });
    body.push(KernelOp::CooperativeMatrixLoad {
        dst: Reg(bf),
        field: 1,
        index: Reg(zero_idx),
        stride: Reg(n8),
        frag: MatrixFrag::B,
        from_shared: true,
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
    });
    // acc starts at the (zeroed) C tile loaded from global → C = A·B + C0.
    body.push(KernelOp::CooperativeMatrixLoad {
        dst: Reg(acc),
        field: 2,
        index: Reg(zero_idx),
        stride: Reg(n8),
        frag: MatrixFrag::Accumulator,
        from_shared: false,
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
    });
    body.push(KernelOp::CooperativeMMA {
        dst: Reg(acc),
        a: Reg(af),
        b: Reg(bf),
        c: Reg(acc),
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
    });
    body.push(KernelOp::CooperativeMatrixStore {
        field: 2,
        index: Reg(zero_idx),
        stride: Reg(n8),
        src: Reg(acc),
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
    });
    KernelDef {
        name: "blas_gemm_f32_tc_shared_probe".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: F32,
            },
            KernelParam::FieldWrite {
                name: "c".into(),
                slot: 2,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: next + 1,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [32, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 0,
    }
}

/// Whether the device enumerates the `8×8×8`, all-f32 shape these kernels
/// are built on.
fn device_has_f32_8x8x8(gpu: &Gpu) -> bool {
    use quanta_core::ScalarType::F32;
    gpu.cooperative_matrix_shapes().iter().any(|s| {
        (s.m, s.n, s.k) == (PROBE_FRAG as u8, PROBE_FRAG as u8, PROBE_FRAG as u8)
            && s.ab_ty == F32
            && s.c_ty == F32
            && s.result_ty == F32
    })
}

/// The hand-built tensor-core kernels, for emitter validation (spirv-val
/// on the SPIR-V lowering, which cannot run on this crate's GPU tests
/// without a `VK_KHR_cooperative_matrix` device). Not API.
///
/// One entry per shape family the cards enumerate, so the SPIR-V a real
/// device would be handed is validated on a host that has none: uniform
/// f32 and f16 at Metal's 8×8×8, mixed f16→f32 at the square 16×16×16, and
/// the non-square 16×8×16 (A 16×16, B 16×8, accumulator 16×8) that the
/// same drivers list next to it.
#[doc(hidden)]
pub fn tc_kernel_defs_for_validation() -> Vec<(&'static str, KernelDef)> {
    use ScalarType::{F16, F32};
    let shape = |m: u8, n: u8, k: u8, ab: ScalarType, acc: ScalarType| CoopMatrixShape {
        m,
        n,
        k,
        ab_ty: ab,
        c_ty: acc,
        result_ty: acc,
    };
    vec![
        (
            "gemm_tc[8x8x8 f32,n=64,k=16]",
            build_tc_def(shape(8, 8, 8, F32, F32), 64, 16),
        ),
        (
            "gemm_tc[8x8x8 f16,n=64,k=16]",
            build_tc_def(shape(8, 8, 8, F16, F16), 64, 16),
        ),
        (
            "gemm_tc[16x16x16 f16->f32,n=64,k=32]",
            build_tc_def(shape(16, 16, 16, F16, F32), 64, 32),
        ),
        (
            "gemm_tc[16x8x16 f16->f32,n=64,k=32]",
            build_tc_def(shape(16, 8, 16, F16, F32), 64, 32),
        ),
        ("gemm_f32_tc_shared_probe", build_tc_shared_probe_def()),
    ]
}

/// Build the register-blocked cooperative-matrix GEMM kernel (`C += A·B`)
/// for one device shape: A fragments are `shape.m × shape.k`, B fragments
/// `shape.k × shape.n`, the `BR×BC` accumulators `shape.m × shape.n`, so the
/// output tile one subgroup owns is `(shape.m·BR) × (shape.n·BC)` — 32×32 on
/// 8×8×8, 64×64 on 16×16×16 — and the K sweep steps by `shape.k`. A/B
/// fragments carry `shape.ab_ty`, the accumulator load, the MMA and the store
/// carry `shape.c_ty`; the driver matches exactly that against the shape it
/// enumerated. `n`, `k` are baked constants. The op sequence is generated
/// programmatically over the fragment grid so the blocking factor is just
/// `BR`/`BC`.
///
/// Each loaded A row-strip fragment feeds `BC` MMAs and each B col-strip feeds
/// `BR`, so a `BR×BC` tile does `BR·BC` MMAs from `BR+BC` global loads per
/// K-step — the arithmetic intensity that lets the MMA units run ahead of the
/// memory system.
///
/// The workgroup is one subgroup wide at the 32-lane default; `gemm_tc`
/// re-stamps it with the device's own width before the JIT, since the tile
/// index is the workgroup index.
///
/// Register map (the MSL JIT emitter declares `rN` per op, so a dst must never
/// alias an operand and every result takes a fresh register):
/// - r0..r11: tile coordinates + loop bounds (fixed).
/// - acc fragments: a fixed block starting at `ACC` (persist across the loop).
/// - rowbase[r] / colbase[c] / c_index[r][c]: fixed blocks the store reuses.
/// - everything else: `fresh()`.
fn build_tc_def(shape: CoopMatrixShape, n: u32, k: u32) -> KernelDef {
    use ScalarType::U32;
    let (fm, fn_, fk) = (shape.m, shape.n, shape.k);
    let (in_ty, acc_ty) = (shape.ab_ty, shape.c_ty);
    let tile_m = u32::from(fm) * BR;
    let tile_n = u32::from(fn_) * BC;
    let frag = |dst, field, index, stride, role| KernelOp::CooperativeMatrixLoad {
        dst,
        field,
        index,
        stride,
        frag: role,
        from_shared: false,
        m: fm,
        n: fn_,
        k: fk,
        ty: match role {
            MatrixFrag::Accumulator => acc_ty,
            MatrixFrag::A | MatrixFrag::B => in_ty,
        },
    };

    // Fixed register blocks.
    let n_acc = (BR * BC) as usize;
    const ACC: u32 = 100; // acc[r*BC + c]
    let rowbase = 40u32; // rowbase[r], r in 0..BR
    let colbase = 50u32; // colbase[c], c in 0..BC
    let cidx = 60u32; // c_index[r*BC + c], reused by the store
    let mut next = 200u32;
    let mut fresh = || {
        let r = next;
        next += 1;
        r
    };

    // r0=tile id; r1=shape.k (the K step); r2=tile_n; r3=n; r4=k;
    // r5=tiles_n=n/tile_n; r6=block_row; r7=block_col;
    // r8=row0=block_row*tile_m; r9=col0; r10=num_k_tiles=k/shape.k;
    // r11=tile_m.
    let mut body = vec![
        KernelOp::NucleusId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(u32::from(fk)),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(tile_n),
        },
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::U32(n),
        },
        KernelOp::Const {
            dst: Reg(4),
            value: ConstValue::U32(k),
        },
        KernelOp::Const {
            dst: Reg(11),
            value: ConstValue::U32(tile_m),
        },
        KernelOp::BinOp {
            dst: Reg(5),
            a: Reg(3),
            b: Reg(2),
            op: BinOp::Div,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(6),
            a: Reg(0),
            b: Reg(5),
            op: BinOp::Div,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(7),
            a: Reg(0),
            b: Reg(5),
            op: BinOp::Rem,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(8),
            a: Reg(6),
            b: Reg(11),
            op: BinOp::Mul,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(9),
            a: Reg(7),
            b: Reg(2),
            op: BinOp::Mul,
            ty: U32,
        },
        KernelOp::BinOp {
            dst: Reg(10),
            a: Reg(4),
            b: Reg(1),
            op: BinOp::Div,
            ty: U32,
        },
    ];

    // rowbase[r] = row0 + r*shape.m
    for r in 0..BR {
        let off = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off),
            value: ConstValue::U32(r * u32::from(fm)),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(rowbase + r),
            a: Reg(8),
            b: Reg(off),
            op: BinOp::Add,
            ty: U32,
        });
    }
    // colbase[c] = col0 + c*shape.n
    for c in 0..BC {
        let off = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off),
            value: ConstValue::U32(c * u32::from(fn_)),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(colbase + c),
            a: Reg(9),
            b: Reg(off),
            op: BinOp::Add,
            ty: U32,
        });
    }
    // c_index[r][c] = rowbase[r]*n + colbase[c]; load each accumulator from C.
    for r in 0..BR {
        for c in 0..BC {
            let idx = (r * BC + c) as usize;
            let mul = fresh();
            body.push(KernelOp::BinOp {
                dst: Reg(mul),
                a: Reg(rowbase + r),
                b: Reg(3),
                op: BinOp::Mul,
                ty: U32,
            });
            body.push(KernelOp::BinOp {
                dst: Reg(cidx + idx as u32),
                a: Reg(mul),
                b: Reg(colbase + c),
                op: BinOp::Add,
                ty: U32,
            });
            body.push(frag(
                Reg(ACC + idx as u32),
                2,
                Reg(cidx + idx as u32),
                Reg(3),
                MatrixFrag::Accumulator,
            ));
        }
    }

    // K-loop: load BR A row-strips + BC B col-strips, then BR·BC MMAs.
    let kt = fresh();
    let mut loop_body: Vec<KernelOp> = Vec::new();
    let k_off = fresh(); // kt * shape.k — the K position of this step
    loop_body.push(KernelOp::BinOp {
        dst: Reg(k_off),
        a: Reg(kt),
        b: Reg(1),
        op: BinOp::Mul,
        ty: U32,
    });
    // a_frag[r] from A at rowbase[r]*k + k_off (stride k).
    let a_frag: Vec<u32> = (0..BR).map(|_| fresh()).collect();
    for r in 0..BR {
        let mul = fresh();
        let aidx = fresh();
        loop_body.push(KernelOp::BinOp {
            dst: Reg(mul),
            a: Reg(rowbase + r),
            b: Reg(4),
            op: BinOp::Mul,
            ty: U32,
        });
        loop_body.push(KernelOp::BinOp {
            dst: Reg(aidx),
            a: Reg(mul),
            b: Reg(k_off),
            op: BinOp::Add,
            ty: U32,
        });
        loop_body.push(frag(
            Reg(a_frag[r as usize]),
            0,
            Reg(aidx),
            Reg(4),
            MatrixFrag::A,
        ));
    }
    // b_frag[c] from B at k_off*n + colbase[c] (stride n).
    let b_frag: Vec<u32> = (0..BC).map(|_| fresh()).collect();
    for c in 0..BC {
        let mul = fresh();
        let bidx = fresh();
        loop_body.push(KernelOp::BinOp {
            dst: Reg(mul),
            a: Reg(k_off),
            b: Reg(3),
            op: BinOp::Mul,
            ty: U32,
        });
        loop_body.push(KernelOp::BinOp {
            dst: Reg(bidx),
            a: Reg(mul),
            b: Reg(colbase + c),
            op: BinOp::Add,
            ty: U32,
        });
        loop_body.push(frag(
            Reg(b_frag[c as usize]),
            1,
            Reg(bidx),
            Reg(3),
            MatrixFrag::B,
        ));
    }
    // acc[r][c] += a_frag[r] · b_frag[c]
    for r in 0..BR {
        for c in 0..BC {
            let acc = ACC + (r * BC + c);
            loop_body.push(KernelOp::CooperativeMMA {
                dst: Reg(acc),
                a: Reg(a_frag[r as usize]),
                b: Reg(b_frag[c as usize]),
                c: Reg(acc),
                m: fm,
                n: fn_,
                k: fk,
                ty: acc_ty,
            });
        }
    }
    body.push(KernelOp::Loop {
        count: Reg(10),
        iter_reg: Reg(kt),
        body: loop_body,
    });

    // Store the accumulators back to C (each C index is in cidx[r*BC+c]).
    for idx in 0..n_acc as u32 {
        body.push(KernelOp::CooperativeMatrixStore {
            field: 2,
            index: Reg(cidx + idx),
            stride: Reg(3),
            src: Reg(ACC + idx),
            m: fm,
            n: fn_,
            k: fk,
            ty: acc_ty,
        });
    }
    let nr = next;

    KernelDef {
        name: format!("blas_gemm_tc_{fm}x{fn_}x{fk}_{in_ty:?}_{acc_ty:?}"),
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
                scalar_type: acc_ty,
            },
        ],
        body,
        body_source: None,
        next_reg: nr + 1,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [32, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 0,
    }
}

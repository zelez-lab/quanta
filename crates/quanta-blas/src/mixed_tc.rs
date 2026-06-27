//! Tensor-core / cooperative-matrix GEMM (f32) — Metal `simdgroup_matrix`.
//!
//! `C ← A·B + C`, row-major f32, with `m`/`n` multiples of 32 and `k` a
//! multiple of 8. Each subgroup (32 lanes) owns a **32×32 output tile** held as
//! a 4×4 grid of 8×8 `simdgroup_matrix` accumulators (loaded from C). It sweeps
//! K in 8-wide steps loading 4 A row-strip fragments + 4 B col-strip fragments
//! and issuing 16 `simdgroup_multiply_accumulate`s — so 8 global fragment loads
//! feed 16 MMAs (each A-frag feeds 4 MMAs, each B-frag 4), the arithmetic
//! intensity that lets the tensor-core units beat the SIMT tiled kernel
//! (~1.5× at N=512 on M1 Pro, 553 vs 372 GFLOP/s). The accumulators are stored
//! back at the end. The blocking factor is `BR`×`BC` (currently 4×4).
//!
//! Reuses the proven `gemmEntry` contract — the differential test against the
//! f32 reference is the binding check, no new Lean. The kernel is hand-built
//! `KernelDef` IR: the cooperative-matrix ops (`CooperativeMatrixLoad` /
//! `CooperativeMMA` / `CooperativeMatrixStore`) have no `#[quanta::kernel]`
//! Rust surface and are subgroup-collective, so the WASM route can't express
//! them. Only backends advertising `supports_cooperative_matrix()` run this;
//! others return `NotSupported`.
//!
//! Scope: `C += A·B` (α = β = 1), m/n multiple of 32, k multiple of 8. General
//! α/β, tails, and threadgroup-shared staging of the A/B tiles (which would cut
//! the global fragment loads further) are later increments; the public `gemm`
//! routes everything else to the tiled kernel.

use quanta::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MatrixFrag, Reg, ScalarType,
    serialize_kernel,
};

const FRAG: u32 = 8; // simdgroup_matrix fragment edge (8×8×8 MMA)
const BR: u32 = 4; // accumulator fragments down (rows) per subgroup
const BC: u32 = 4; // accumulator fragments across (cols) per subgroup
const TILE_M: u32 = FRAG * BR; // output tile rows per subgroup (32)
const TILE_N: u32 = FRAG * BC; // output tile cols per subgroup (32)

/// `gemm_f32_tc`: `C ← A·B + C` on the cooperative-matrix path. Requires
/// `supports_cooperative_matrix()`, `m`/`n` multiples of 32 and `k` a multiple
/// of 8; otherwise returns `NotSupported` so the caller can fall back to the
/// tiled kernel.
///
/// Register-blocked: each subgroup owns a 32×32 output tile = a 4×4 grid of 8×8
/// accumulator fragments. Per K-step it loads 4 A row-strip fragments + 4 B
/// col-strip fragments and issues 16 MMAs — so 8 loads feed 16 MMAs (each
/// fragment reused 4×), the arithmetic intensity that lets the tensor-core
/// units beat the SIMT tiled path (~1.5× at N=512 on M1 Pro).
pub fn gemm_f32_tc(
    gpu: &Gpu,
    m: u32,
    n: u32,
    k: u32,
    a: &Field<f32>,
    b: &Field<f32>,
    c: &Field<f32>,
) -> Result<(), QuantaError> {
    if !gpu.supports_cooperative_matrix() {
        return Err(QuantaError::not_supported(
            "cooperative matrix (tensor cores) not available on this backend",
        ));
    }
    if !m.is_multiple_of(TILE_M) || !n.is_multiple_of(TILE_N) || !k.is_multiple_of(FRAG) {
        return Err(QuantaError::not_supported(
            "gemm_f32_tc requires m multiple of 32, n multiple of 32, k multiple of 8",
        ));
    }
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    if a.len() != mu * ku {
        return Err(QuantaError::invalid_param(
            "gemm_f32_tc: A length must be m*k",
        ));
    }
    if b.len() != ku * nu {
        return Err(QuantaError::invalid_param(
            "gemm_f32_tc: B length must be k*n",
        ));
    }
    if c.len() != mu * nu {
        return Err(QuantaError::invalid_param(
            "gemm_f32_tc: C length must be m*n",
        ));
    }
    if mu * nu == 0 || ku == 0 {
        return Ok(());
    }

    let def = build_tc_def(n, k);
    let bytes = serialize_kernel(&def);
    let mut wave = gpu.wave_jit(&bytes)?;
    wave.bind(0, a);
    wave.bind(1, b);
    wave.bind(2, c); // in place: C is the accumulator (read + written)
    // One subgroup (32 lanes) per TILE_M×TILE_N output tile.
    let tiles = (m / TILE_M) * (n / TILE_N);
    gpu.dispatch(&wave, tiles * 32)?.wait()?;
    Ok(())
}

/// Build the register-blocked cooperative-matrix GEMM kernel (`C += A·B`,
/// `TILE_M×TILE_N` output tile per subgroup, `BR×BC` fragments). `n`, `k` are
/// baked constants. The op sequence is generated programmatically over the
/// fragment grid so the blocking factor is just `BR`/`BC`.
///
/// Each loaded A row-strip fragment feeds `BC` MMAs and each B col-strip feeds
/// `BR`, so a `BR×BC` tile does `BR·BC` MMAs from `BR+BC` global loads per
/// K-step — the arithmetic intensity that lets the MMA units run ahead of the
/// memory system.
///
/// Register map (the MSL JIT emitter declares `rN` per op, so a dst must never
/// alias an operand and every result takes a fresh register):
/// - r0..r10: tile coordinates + loop bounds (fixed).
/// - acc fragments: a fixed block starting at `ACC` (persist across the loop).
/// - rowbase[r] / colbase[c] / c_index[r][c]: fixed blocks the store reuses.
/// - everything else: `fresh()`.
fn build_tc_def(n: u32, k: u32) -> KernelDef {
    use ScalarType::{F32, U32};
    let f8: u8 = FRAG as u8;
    let frag = |dst, field, index, stride, role| KernelOp::CooperativeMatrixLoad {
        dst,
        field,
        index,
        stride,
        frag: role,
        m: f8,
        n: f8,
        k: f8,
        ty: F32,
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

    // r0=tile id; r1=FRAG(8); r2=TILE_N; r3=n; r4=k; r5=tiles_n=n/TILE_N;
    // r6=block_row; r7=block_col; r8=row0=block_row*TILE_M; r9=col0;
    // r10=num_k_tiles=k/8. (TILE_M baked as a const where needed.)
    let mut body = vec![
        KernelOp::NucleusId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(FRAG),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(TILE_N),
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
            value: ConstValue::U32(TILE_M),
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

    // rowbase[r] = row0 + r*8
    for r in 0..BR {
        let off = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off),
            value: ConstValue::U32(r * FRAG),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(rowbase + r),
            a: Reg(8),
            b: Reg(off),
            op: BinOp::Add,
            ty: U32,
        });
    }
    // colbase[c] = col0 + c*8
    for c in 0..BC {
        let off = fresh();
        body.push(KernelOp::Const {
            dst: Reg(off),
            value: ConstValue::U32(c * FRAG),
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
    let kt8 = fresh();
    loop_body.push(KernelOp::BinOp {
        dst: Reg(kt8),
        a: Reg(kt),
        b: Reg(1),
        op: BinOp::Mul,
        ty: U32,
    });
    // a_frag[r] from A at rowbase[r]*k + kt8 (stride k).
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
            b: Reg(kt8),
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
    // b_frag[c] from B at kt8*n + colbase[c] (stride n).
    let b_frag: Vec<u32> = (0..BC).map(|_| fresh()).collect();
    for c in 0..BC {
        let mul = fresh();
        let bidx = fresh();
        loop_body.push(KernelOp::BinOp {
            dst: Reg(mul),
            a: Reg(kt8),
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
                m: f8,
                n: f8,
                k: f8,
                ty: F32,
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
            m: f8,
            n: f8,
            k: f8,
            ty: F32,
        });
    }
    let nr = next;

    KernelDef {
        name: "blas_gemm_f32_tc".into(),
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
        next_reg: nr + 1,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [32, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 0,
    }
}

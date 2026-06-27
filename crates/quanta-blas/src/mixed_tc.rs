//! Tensor-core / cooperative-matrix GEMM (f32) — Metal `simdgroup_matrix`.
//!
//! `C ← A·B + C`, row-major f32, with `m`/`n` multiples of 16 and `k` a
//! multiple of 8. Each subgroup (32 lanes) owns a **16×16 output tile** held as
//! a 2×2 grid of 8×8 `simdgroup_matrix` accumulators (loaded from C). It sweeps
//! K in 8-wide steps loading 2 A row-strip fragments + 2 B col-strip fragments
//! and issuing 4 `simdgroup_multiply_accumulate`s — so each loaded fragment
//! feeds two MMAs (2× arithmetic intensity), which is what lets the tensor-core
//! units beat the SIMT tiled kernel (~1.33× at N=512 on M1 Pro). The
//! accumulators are stored back at the end.
//!
//! Reuses the proven `gemmEntry` contract — the differential test against the
//! f32 reference is the binding check, no new Lean. The kernel is hand-built
//! `KernelDef` IR: the cooperative-matrix ops (`CooperativeMatrixLoad` /
//! `CooperativeMMA` / `CooperativeMatrixStore`) have no `#[quanta::kernel]`
//! Rust surface and are subgroup-collective, so the WASM route can't express
//! them. Only backends advertising `supports_cooperative_matrix()` run this;
//! others return `NotSupported`.
//!
//! v1 scope: `C += A·B` (α = β = 1), m/n multiple of 16, k multiple of 8.
//! General α/β, tails, and deeper (4×4 + shared-staged) blocking are later
//! increments; the public `gemm` routes everything else to the tiled kernel.

use quanta::{Field, Gpu, QuantaError};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, MatrixFrag, Reg, ScalarType,
    serialize_kernel,
};

const FRAG: u32 = 8; // simdgroup_matrix fragment edge (8×8×8 MMA)
const TILE: u32 = 16; // output tile edge per subgroup (2×2 = 4 fragments)

/// `gemm_f32_tc`: `C ← A·B + C` on the cooperative-matrix path. Requires
/// `supports_cooperative_matrix()` and `m`/`n`/`k` all multiples of 16;
/// otherwise returns `NotSupported` so the caller can fall back to the tiled
/// kernel.
///
/// Register-blocked: each subgroup owns a 16×16 output tile = a 2×2 grid of
/// 8×8 accumulator fragments. Per K-step it loads 2 A-fragments (the tile's two
/// row-strips) and 2 B-fragments (two col-strips) and issues 4 MMAs — so each
/// loaded fragment feeds two MMAs (2× arithmetic intensity over the 1-fragment
/// kernel), the reuse that lets the tensor-core units beat the SIMT tiled path.
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
    if !m.is_multiple_of(TILE) || !n.is_multiple_of(TILE) || !k.is_multiple_of(FRAG) {
        return Err(QuantaError::not_supported(
            "gemm_f32_tc requires m, n multiples of 16 and k a multiple of 8 (v1)",
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
    // One subgroup (32 lanes) per 16×16 output tile.
    let tiles = (m / TILE) * (n / TILE);
    gpu.dispatch(&wave, tiles * 32)?.wait()?;
    Ok(())
}

/// Build the register-blocked cooperative-matrix GEMM kernel (`C += A·B`,
/// 16×16 output tile per subgroup, 2×2 fragments). `n`, `k` are baked
/// constants. The op sequence is generated programmatically to keep the 4
/// accumulators, their C/A/B indices, and the K-loop bookkeeping in one place.
///
/// Register map: a running counter (`nr`) hands out fresh registers — the MSL
/// JIT emitter declares `rN` per op, so a dst must never alias an operand, and
/// every BinOp result takes a new register.
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

    // Constants + tile coordinates.
    // r0=tile id; r1=FRAG(8); r2=TILE(16); r3=n; r4=k;
    // r5=tiles_n=n/16; r6=block_row=tile/tiles_n; r7=block_col=tile%tiles_n;
    // r8=row0=block_row*16; r9=col0=block_col*16; r10=num_k_tiles=k/8.
    let mut body = vec![
        KernelOp::NucleusId { dst: Reg(0) },
        KernelOp::Const {
            dst: Reg(1),
            value: ConstValue::U32(FRAG),
        },
        KernelOp::Const {
            dst: Reg(2),
            value: ConstValue::U32(TILE),
        },
        KernelOp::Const {
            dst: Reg(3),
            value: ConstValue::U32(n),
        },
        KernelOp::Const {
            dst: Reg(4),
            value: ConstValue::U32(k),
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
            b: Reg(2),
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

    // Fragment sub-offsets within the 16×16 tile (2×2 grid of 8×8 frags).
    let offsets = [(0u32, 0u32), (0, 8), (8, 0), (8, 8)];
    // Fixed accumulator registers (persist across the K-loop), one per frag.
    let acc_reg = [40u32, 41, 42, 43];
    let mut nr = 50u32; // fresh-register counter for everything else

    // Per-frag absolute row/col base (row0+fr, col0+fc) and the C index, then
    // load the accumulator tile from C (the β·C term, β = 1).
    // rowbase[i] in r(20+i), colbase[i] in r(24+i) — small fixed block so the
    // K-loop can recompute A/B indices from them.
    for (i, (fr, fc)) in offsets.iter().enumerate() {
        // rowbase = row0 + fr
        let fr_c = nr;
        nr += 1;
        body.push(KernelOp::Const {
            dst: Reg(fr_c),
            value: ConstValue::U32(*fr),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(20 + i as u32),
            a: Reg(8),
            b: Reg(fr_c),
            op: BinOp::Add,
            ty: U32,
        });
        // colbase = col0 + fc
        let fc_c = nr;
        nr += 1;
        body.push(KernelOp::Const {
            dst: Reg(fc_c),
            value: ConstValue::U32(*fc),
        });
        body.push(KernelOp::BinOp {
            dst: Reg(24 + i as u32),
            a: Reg(9),
            b: Reg(fc_c),
            op: BinOp::Add,
            ty: U32,
        });
        // c_index = rowbase*n + colbase, written straight into the fixed reg
        // r(32+i) the matching store reuses (no Copy — the MSL emitter's Copy
        // assigns without declaring, so the dst must already exist).
        let mul = nr;
        nr += 1;
        body.push(KernelOp::BinOp {
            dst: Reg(mul),
            a: Reg(20 + i as u32),
            b: Reg(3),
            op: BinOp::Mul,
            ty: U32,
        });
        body.push(KernelOp::BinOp {
            dst: Reg(32 + i as u32),
            a: Reg(mul),
            b: Reg(24 + i as u32),
            op: BinOp::Add,
            ty: U32,
        });
        body.push(frag(
            Reg(acc_reg[i]),
            2,
            Reg(32 + i as u32),
            Reg(3),
            MatrixFrag::Accumulator,
        ));
    }

    // K-loop: for each 8-wide K-tile, load the 2 A row-strips + 2 B col-strips,
    // then 4 MMAs (each loaded fragment feeds two accumulators).
    let kt = nr;
    nr += 1;
    let mut loop_body: Vec<KernelOp> = Vec::new();
    // kt8 = kt * 8
    let kt8 = nr;
    nr += 1;
    loop_body.push(KernelOp::BinOp {
        dst: Reg(kt8),
        a: Reg(kt),
        b: Reg(1),
        op: BinOp::Mul,
        ty: U32,
    });
    // A row-strips: a_index[r] = rowbase[r]*k + kt8, for the two distinct rows
    // (offsets 0 and 8 → accumulators {0,1} share row 0; {2,3} share row 8).
    // rowbase for row-group g is r(20 + g*2) (i=0 and i=2).
    let a_frag = [nr, nr + 1];
    nr += 2;
    for (g, &af) in a_frag.iter().enumerate() {
        let rb = 20 + (g as u32) * 2; // rowbase reg of the first frag in the row group
        let mul = nr;
        nr += 1;
        let aidx = nr;
        nr += 1;
        loop_body.push(KernelOp::BinOp {
            dst: Reg(mul),
            a: Reg(rb),
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
        loop_body.push(frag(Reg(af), 0, Reg(aidx), Reg(4), MatrixFrag::A));
    }
    // B col-strips: b_index[c] = kt8*n + colbase[c] (colbase of cols 0 and 8 →
    // accumulators {0,2} share col 0; {1,3} share col 8). colbase reg = r(24+i).
    let b_frag = [nr, nr + 1];
    nr += 2;
    for (g, &bf) in b_frag.iter().enumerate() {
        let cb = 24 + g as u32; // colbase reg for col group g (i=0 → col 0, i=1 → col 8)
        let mul = nr;
        nr += 1;
        let bidx = nr;
        nr += 1;
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
            b: Reg(cb),
            op: BinOp::Add,
            ty: U32,
        });
        loop_body.push(frag(Reg(bf), 1, Reg(bidx), Reg(3), MatrixFrag::B));
    }
    // 4 MMAs: acc += A[row group] · B[col group], each fragment feeding two.
    for (&(fr, fc), &acc) in offsets.iter().zip(acc_reg.iter()) {
        let ag = if fr == 0 { a_frag[0] } else { a_frag[1] };
        let bg = if fc == 0 { b_frag[0] } else { b_frag[1] };
        loop_body.push(KernelOp::CooperativeMMA {
            dst: Reg(acc),
            a: Reg(ag),
            b: Reg(bg),
            c: Reg(acc),
            m: f8,
            n: f8,
            k: f8,
            ty: F32,
        });
    }
    body.push(KernelOp::Loop {
        count: Reg(10),
        iter_reg: Reg(kt),
        body: loop_body,
    });

    // Store the 4 accumulators back to C (C index of frag i is in r(32+i)).
    for (i, &acc) in acc_reg.iter().enumerate() {
        body.push(KernelOp::CooperativeMatrixStore {
            field: 2,
            index: Reg(32 + i as u32),
            stride: Reg(3),
            src: Reg(acc),
            m: f8,
            n: f8,
            k: f8,
            ty: F32,
        });
    }

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

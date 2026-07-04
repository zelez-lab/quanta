//! Plan-based FFT dispatch — compile once, execute many (the VkFFT pattern).
//!
//! [`FftPlan`] fixes the transform size `n` (a power of 2) and direction at
//! construction and amortizes everything reusable across executes:
//!
//! - **Kernels are JIT-compiled once.** The bit-reversal wave, one butterfly
//!   wave per stage (`log₂N` of them, per-stage constants baked in), and the
//!   inverse `1/N` scale wave are built at [`FftPlan::new`] and cached.
//!   [`FftPlan::execute`] only rebinds fields and dispatches.
//! - **Twiddles are precomputed.** A host-computed table
//!   `tw[k] = (cos θ_k, sin θ_k)`, `θ_k = sign·2π·k/N` for `k < N/2`, lives in
//!   two plan-owned `Field<f32>`s; the butterfly kernel LOADS
//!   `w = tw[j·N/m]` instead of evaluating `sin`/`cos` per butterfly.
//!
//! Buffer ownership: the plan owns the twiddle table and the compiled waves
//! (that is the point of a plan); the I/O and work fields are allocated per
//! [`execute`](FftPlan::execute) so a plan holds no transform-sized scratch
//! between calls and executes are independent.

use core::f32::consts::PI;

use quanta::{Field, Gpu, QuantaError, Wave};
use quanta_ir::{
    BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType, serialize_kernel,
};

const F32: ScalarType = ScalarType::F32;
const U32: ScalarType = ScalarType::U32;

/// A compiled FFT plan for a fixed size and direction.
///
/// Construction JIT-compiles every kernel the transform needs and uploads the
/// precomputed twiddle table; [`execute`](Self::execute) reuses them, so
/// repeated transforms of the same size skip kernel rebuilds, re-JITs, and
/// per-butterfly `sin`/`cos`.
///
/// ```no_run
/// # fn main() -> Result<(), quanta::QuantaError> {
/// let gpu = quanta::init_cpu();
/// let mut plan = quanta_fft::FftPlan::new(&gpu, 1024, false)?; // forward
/// for frame in signal_frames() {
///     let (re, im) = plan.execute(&frame.re, &frame.im)?;
///     // ...
/// }
/// # Ok(()) }
/// # struct Frame { re: Vec<f32>, im: Vec<f32> }
/// # fn signal_frames() -> Vec<Frame> { vec![] }
/// ```
pub struct FftPlan {
    gpu: Gpu,
    n: usize,
    inverse: bool,
    /// Precomputed twiddle table, split complex: `tw_re[k] + i·tw_im[k]` =
    /// `exp(sign·2πi·k/n)` for `k < n/2` (one dummy entry when `n == 1`).
    tw_re: Field<f32>,
    tw_im: Field<f32>,
    /// Bit-reversal permutation wave (`out[i] = in[bitrev(i)]`).
    bitrev: Wave,
    /// One butterfly wave per stage; entry `s` has group size `m = 2^(s+1)`
    /// and its twiddle stride baked in. Twiddle fields are bound at slots
    /// 2/3 once, at construction.
    stages: Vec<Wave>,
    /// Inverse only: the in-place `x *= 1/n` normalisation wave.
    scale: Option<Wave>,
}

impl FftPlan {
    /// Build a plan for size `n` (must be a power of 2, `NotSupported`
    /// otherwise) and direction (`inverse = false` → forward).
    ///
    /// Compiles the kernels and uploads the twiddle table once; the plan can
    /// then [`execute`](Self::execute) any number of times.
    pub fn new(gpu: &Gpu, n: usize, inverse: bool) -> Result<Self, QuantaError> {
        if n == 0 || !n.is_power_of_two() {
            return Err(QuantaError::not_supported(
                "fft: length must be a power of 2",
            ));
        }
        let log2n = n.trailing_zeros();
        let sign: f32 = if inverse { 1.0 } else { -1.0 };

        // Twiddle table: tw[k] = exp(sign·2πi·k/n), k in 0..n/2. Stage with
        // group size m reads w_j = tw[j·(n/m)] = exp(sign·2πi·j/m).
        let twiddle_c = sign * 2.0 * PI;
        let tw_len = (n / 2).max(1); // n == 1: one dummy entry (no stage reads it)
        let mut host_re = Vec::with_capacity(tw_len);
        let mut host_im = Vec::with_capacity(tw_len);
        for k in 0..tw_len {
            let theta = (k as f32 * twiddle_c) / n as f32;
            host_re.push(theta.cos());
            host_im.push(theta.sin());
        }
        let tw_re = gpu.field::<f32>(tw_len)?;
        let tw_im = gpu.field::<f32>(tw_len)?;
        tw_re.write(&host_re)?;
        tw_im.write(&host_im)?;

        // Compile every wave once.
        let bitrev = gpu.wave_jit(&serialize_kernel(&bitrev_def(log2n)))?;
        let mut stages = Vec::with_capacity(log2n as usize);
        let mut m = 2u32;
        for _ in 0..log2n {
            let mut wave = gpu.wave_jit(&serialize_kernel(&butterfly_def(m, n as u32)))?;
            // The twiddle table never changes for the plan's lifetime.
            wave.bind(2, &tw_re);
            wave.bind(3, &tw_im);
            stages.push(wave);
            m *= 2;
        }
        let scale = if inverse {
            Some(gpu.wave_jit(&serialize_kernel(&scale_def(1.0 / n as f32)))?)
        } else {
            None
        };

        Ok(Self {
            gpu: gpu.clone(),
            n,
            inverse,
            tw_re,
            tw_im,
            bitrev,
            stages,
            scale,
        })
    }

    /// The transform size this plan was built for.
    pub fn len(&self) -> usize {
        self.n
    }

    /// `true` when the plan transforms zero points — never, since plans
    /// require a power-of-2 size (provided for `len`/`is_empty` pairing).
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether this is an inverse plan (`+` twiddle sign, `1/n` scale).
    pub fn inverse(&self) -> bool {
        self.inverse
    }

    /// Read back the precomputed twiddle table `(re, im)` —
    /// `tw[k] = exp(sign·2πi·k/n)` for `k < n/2`. Diagnostics/testing; the
    /// butterfly kernels load from the same device buffers.
    pub fn twiddles(&self) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
        Ok((self.tw_re.read()?, self.tw_im.read()?))
    }

    /// Run the transform on split-complex input of length [`len`](Self::len).
    ///
    /// Rebinds the cached waves and dispatches — no kernel rebuild, no
    /// re-JIT, no twiddle recompute. `&mut self` because rebinding mutates
    /// the cached waves' binding tables.
    pub fn execute(&mut self, re: &[f32], im: &[f32]) -> Result<(Vec<f32>, Vec<f32>), QuantaError> {
        if re.len() != im.len() {
            return Err(QuantaError::invalid_param("fft: re/im length mismatch"));
        }
        if re.len() != self.n {
            return Err(QuantaError::invalid_param(
                "fft: input length differs from the plan size",
            ));
        }
        let n = self.n;

        // Per-execute I/O: bit-reverse into the work buffers, then butterfly
        // in place.
        let in_re = self.gpu.field::<f32>(n)?;
        let in_im = self.gpu.field::<f32>(n)?;
        let work_re = self.gpu.field::<f32>(n)?;
        let work_im = self.gpu.field::<f32>(n)?;
        in_re.write(re)?;
        in_im.write(im)?;

        // 1. Bit-reversal permutation: work[i] = in[bitrev(i)].
        for (input, out) in [(&in_re, &work_re), (&in_im, &work_im)] {
            self.bitrev.bind(0, input);
            self.bitrev.bind(1, out);
            self.gpu.dispatch(&self.bitrev, n as u32)?.wait()?;
        }

        // 2. log2n butterfly stages, in place on work (twiddles at slots 2/3
        // stay bound from construction).
        for wave in &mut self.stages {
            wave.bind(0, &work_re);
            wave.bind(1, &work_im);
            self.gpu.dispatch(wave, (n / 2) as u32)?.wait()?;
        }

        // 3. Inverse: divide by n.
        if let Some(scale) = &mut self.scale {
            for field in [&work_re, &work_im] {
                scale.bind(0, field);
                scale.bind(1, field); // in place
                self.gpu.dispatch(scale, n as u32)?.wait()?;
            }
        }

        Ok((work_re.read()?, work_im.read()?))
    }
}

/// `out[i] = in[bitrev(i, log2n)]` — reverse the low `log2n` bits of `i`.
fn bitrev_def(log2n: u32) -> KernelDef {
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
    KernelDef {
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
    }
}

/// One radix-2 butterfly stage with group size `m`, in place on `(re, im)`,
/// twiddles loaded from the precomputed table at slots 2/3. Dispatched with
/// `n/2` threads (one butterfly each).
fn butterfly_def(m: u32, n: u32) -> KernelDef {
    let half = m / 2;
    let stride = n / m; // tw[j·stride] = exp(sign·2πi·j/m)
    // r0=t (butterfly id); r1=half; r2=m;
    // g = t / half; j = t % half; base = g*m; i0 = base + j; i1 = i0 + half;
    // w = (tw_re[j·stride], tw_im[j·stride]);
    // a = x[i0]; b = x[i1]; wb = w·b; x[i0]=a+wb; x[i1]=a−wb. (re & im fields.)
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
        // k = j·stride — the twiddle-table index for this butterfly
        KernelOp::Const {
            dst: Reg(8),
            value: ConstValue::U32(stride),
        },
        KernelOp::BinOp {
            dst: Reg(9),
            a: Reg(4),
            b: Reg(8),
            op: BinOp::Mul,
            ty: U32,
        }, // k
        KernelOp::Load {
            dst: Reg(10),
            field: 2,
            index: Reg(9),
            ty: F32,
        }, // w_re
        KernelOp::Load {
            dst: Reg(11),
            field: 3,
            index: Reg(9),
            ty: F32,
        }, // w_im
        // a = (re[i0], im[i0]); b = (re[i1], im[i1])
        KernelOp::Load {
            dst: Reg(12),
            field: 0,
            index: Reg(6),
            ty: F32,
        }, // a_re
        KernelOp::Load {
            dst: Reg(13),
            field: 1,
            index: Reg(6),
            ty: F32,
        }, // a_im
        KernelOp::Load {
            dst: Reg(14),
            field: 0,
            index: Reg(7),
            ty: F32,
        }, // b_re
        KernelOp::Load {
            dst: Reg(15),
            field: 1,
            index: Reg(7),
            ty: F32,
        }, // b_im
        // wb = w·b: wb_re = w_re·b_re − w_im·b_im; wb_im = w_re·b_im + w_im·b_re
        KernelOp::BinOp {
            dst: Reg(16),
            a: Reg(10),
            b: Reg(14),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(17),
            a: Reg(11),
            b: Reg(15),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(18),
            a: Reg(16),
            b: Reg(17),
            op: BinOp::Sub,
            ty: F32,
        }, // wb_re
        KernelOp::BinOp {
            dst: Reg(19),
            a: Reg(10),
            b: Reg(15),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(20),
            a: Reg(11),
            b: Reg(14),
            op: BinOp::Mul,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(21),
            a: Reg(19),
            b: Reg(20),
            op: BinOp::Add,
            ty: F32,
        }, // wb_im
        // x[i0] = a + wb ; x[i1] = a − wb
        KernelOp::BinOp {
            dst: Reg(22),
            a: Reg(12),
            b: Reg(18),
            op: BinOp::Add,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(23),
            a: Reg(13),
            b: Reg(21),
            op: BinOp::Add,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(24),
            a: Reg(12),
            b: Reg(18),
            op: BinOp::Sub,
            ty: F32,
        },
        KernelOp::BinOp {
            dst: Reg(25),
            a: Reg(13),
            b: Reg(21),
            op: BinOp::Sub,
            ty: F32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(6),
            src: Reg(22),
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(6),
            src: Reg(23),
            ty: F32,
        },
        KernelOp::Store {
            field: 0,
            index: Reg(7),
            src: Reg(24),
            ty: F32,
        },
        KernelOp::Store {
            field: 1,
            index: Reg(7),
            src: Reg(25),
            ty: F32,
        },
    ];
    KernelDef {
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
            KernelParam::FieldRead {
                name: "tw_re".into(),
                slot: 2,
                scalar_type: F32,
            },
            KernelParam::FieldRead {
                name: "tw_im".into(),
                slot: 3,
                scalar_type: F32,
            },
        ],
        body,
        body_source: None,
        next_reg: 26,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// In-place scale `x[i] *= s` (the inverse FFT's 1/n).
fn scale_def(s: f32) -> KernelDef {
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
    KernelDef {
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
    }
}

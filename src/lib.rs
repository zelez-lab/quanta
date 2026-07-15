//! # Quanta — GPU compute and rendering API
//!
//! Quarks, protons, nuclei, fields, waves, pulses.
//!
//! One API (~20 functions), any GPU. Drivers are thin translation layers
//! to platform-specific backends (Metal on macOS/iOS, Vulkan on Linux/Android/Windows,
//! WebGPU in browsers).
//!
//! ## Naming (subatomic physics)
//!
//! | Name     | GPU hardware       | Description                          |
//! |----------|--------------------|--------------------------------------|
//! | Quark    | Thread / lane      | Smallest unit of execution           |
//! | Proton   | Core               | Executes quarks                      |
//! | Nucleus  | Compute unit       | Group of protons                     |
//! | Field    | Buffer             | Data that quarks operate on          |
//! | Wave     | Compute dispatch   | A bound kernel ready to dispatch     |
//! | Pulse    | Fence / sync       | GPU completion signal                |
//! | Quanta   | The whole GPU      | All nuclei firing together           |
//!
//! ## Kernel language
//!
//! GPU kernels are written as annotated Rust functions:
//!
//! ```ignore
//! #[quanta::kernel]
//! fn scan_filter(input: &[f32], output: &mut [f32], threshold: f32) {
//!     let i = quark_id();
//!     if input[i] > threshold {
//!         output[i] = input[i];
//!     }
//! }
//! ```
//!
//! The proc macro compiles directly to GPU ISA at build time:
//! - AMD: LLVM → amdgcn binary
//! - NVIDIA: LLVM → nvptx64 PTX
//! - Apple: generates MSL source string
//! - Browser: generates WGSL source string
//!
//! No SPIR-V. No intermediate representation. Direct to target.
//!
//! ## Crate layout
//!
//! This crate is the facade over the split substrate:
//!
//! - `quanta-core` — the shared device line (sealed `GpuDevice`,
//!   drivers, fields/textures/sync). Re-exported wholesale here.
//! - the compute face — `#[quanta::kernel]` / waves / the scan
//!   library, compiled in behind the `compute` feature.
//! - `quanta-render` — the render face (`RenderGpu`, typed wrappers,
//!   render-stage macros), re-exported here behind the `render`
//!   feature. Render-only consumers (e.g. a UI toolkit) can depend on
//!   `quanta-render` directly and never pull the compute stack into
//!   their dependency graph.

#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "compute")]
pub mod kernel;
#[cfg(feature = "compute")]
pub mod scan;

/// GPU intrinsics — `extern "C"` imports surfaced as
/// `import "quanta" "<name>"` in the WASM emitted by
/// `#[quanta::kernel]`. cfg-gated to wasm32; the lowering pass on the
/// host side resolves them. See roadmap step 058.
#[cfg(feature = "compute")]
pub mod intrinsics;

// Host-side stubs for every GPU intrinsic, used by `_src!()` macros
// emitted by `#[quanta::device]` (which inject `use
// ::quanta::__device_host_stubs::*`). The stubs live in `quanta-core`
// so companion crates reach them without depending on this facade;
// core exposes both `device_host_stubs` and the `__device_host_stubs`
// alias, and the wholesale `pub use quanta_core::*` above re-exports
// the latter under the historical facade name.

/// Spec enum tables for the WebGPU IDL (B′ track of FFI TCB shrink).
/// Generated from `web/webgpu.idl` by `quanta codegen webgpu`; the
/// `tests` block inside checks that every enum string Quanta hands
/// the JS side is a member of the spec table. Top-level so native
/// `cargo test` runs the subset check without needing a wasm32 build.
pub mod webgpu_generated_codes;

// The shared substrate: device discovery (`init` / `devices` /
// `init_cpu`), the `Gpu` handle, fields, textures, sync, and — under
// the matching features — the compute / render data models.
pub use quanta_core::*;

// Re-export kernel types. (`ShaderBinary` / `ShaderStage` are *render*
// types — they live on the render surface, re-exported at the root
// only when the `render` feature is on.)
//
// `ScalarType` is a `quanta-ir` kernel-language type (the scalar tag
// carried by `GpuType::scalar_type()`); it is re-exported at the root
// under the `compute` feature because `#[quanta::kernel]`-generated
// code names it through `quanta::ScalarType`.
#[cfg(feature = "compute")]
pub use kernel::{GpuType, KernelBinary, ScalarType};

// Compute-face proc macros — only when the `compute` feature is on
// (symmetric to the render-stage macros below).
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::__kernel_inner;
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::device;
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::gpu_type;
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::import_devices;
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::kernel;

// Derive macros (compute face)
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::Fields;
#[cfg(feature = "compute")]
pub use quanta_compute_dsl::Uniforms;

// Render face — the `quanta-render` crate, re-exported wholesale so
// `use quanta::*` covers render consumers that come through the
// facade. This brings in the render-stage shader macros, the typed
// render wrappers, and the `RenderGpu` extension trait alongside the
// render data model re-exported from `quanta-core` above.
#[cfg(feature = "render")]
pub use quanta_render::*;

// ── Companion-crate umbrella (tokio model) ──────────────────────────
//
// The companion crates stay independent packages (each with its own
// tests, proofs, and version line), but the facade is the single
// user-facing namespace: enabling `sci` / `prims` / `autograd` mounts
// them as modules here, feature-gated exactly like tokio's `net` /
// `fs` / `time`. Backend selection is inherited from the facade's
// backend features (`software` / `metal` / `vulkan`), which forward
// weakly to every companion — no companion-specific feature spelling
// leaks into user code. `quanta::nn` (the neural stack) is the newest
// member; its completeness contract lives in its crate's PARITY.md.

/// Scientific computing on the GPU — the NumPy/SciPy face of Quanta.
///
/// Enabled by the `sci` feature. The centrepiece is [`Array`](crate::sci::Array),
/// a host-side N-dimensional array backed by GPU memory: NumPy-style
/// construction, broadcasting ufuncs, reductions, and zero-copy shape
/// manipulation, with every kernel JIT-dispatched through the same
/// Quanta runtime as hand-written `#[quanta::kernel]` code. If you are
/// coming from Python, `quanta::sci` ≈ `numpy` and the submodules map
/// onto the packages you already know:
///
/// | Quanta                | Python                          |
/// |-----------------------|---------------------------------|
/// | `sci::Array`          | `numpy.ndarray`                 |
/// | `sci::linalg`         | `numpy.linalg` / `scipy.linalg` |
/// | `sci::fft`            | `numpy.fft` / `scipy.fft`       |
/// | `sci::random`         | `numpy.random`                  |
/// | `sci::layout`         | (CuTe-style layout algebra)     |
///
/// The high-level path is `Array`'s own methods (`add`, `sum`,
/// `matmul`, …); the submodules expose the underlying verified
/// libraries directly for when you need raw control over device
/// buffers. Everything here is compute-only — it composes with the
/// render face but never requires it.
#[cfg(feature = "sci")]
pub mod sci {
    pub use quanta_array::*;

    /// Raw verified BLAS + factorisations over device [`Field`](crate::Field)s
    /// (re-export of `quanta-blas`).
    ///
    /// Level-1/2/3 BLAS (`axpy`, `dot`, `gemm`, `trsm`, …), the exact
    /// factorisations (`cholesky`, `lu`, `qr`) with their solvers, and
    /// the iterative symmetric decompositions (`eigh`, `svd`) — every
    /// op with a mechanically-proven forward-error bound. This is the
    /// low-level surface (`numpy.linalg` / `scipy.linalg` territory,
    /// but operating on raw `Field<f32>` buffers you allocate and
    /// write yourself); the high-level path remains the methods on
    /// [`Array`], which manage buffers for you.
    pub mod linalg {
        pub use quanta_blas::*;
    }

    /// Fast Fourier transforms on the GPU (re-export of `quanta-fft`).
    ///
    /// The `numpy.fft` / `scipy.fft` counterpart: forward + inverse
    /// complex FFT of any length (radix-2 Cooley-Tukey for powers of
    /// two, Bluestein chirp-z otherwise), 2-D transforms, real-input
    /// `rfft`/`irfft`, and plan-based dispatch (`FftPlan`, the VkFFT
    /// pattern) for repeated same-size transforms. Complex data
    /// travels as split re/im `f32` slices. The pure-Rust direct-DFT
    /// oracle lives in [`reference`](crate::sci::fft::reference) and is always
    /// available, even without a backend.
    pub mod fft {
        pub use quanta_fft::*;
    }

    /// Random number generation, host-side and in-kernel (re-export of
    /// `quanta-rand`).
    ///
    /// The `numpy.random` counterpart: a host [`Rng`](crate::sci::random::Rng)
    /// (xoshiro128++ with constant-time jump-ahead for independent
    /// streams) plus counter-based GPU fills (Philox4×32-10,
    /// Threefry4×32-20) for uniform / normal / exponential /
    /// lognormal / Bernoulli / Poisson draws. Every GPU fill is
    /// bit-exact reproducible from `(seed, len)` across all backends,
    /// and the in-kernel generators are callable from your own
    /// `#[quanta::kernel]` functions by qualified path.
    pub mod random {
        pub use quanta_rand::*;
    }

    /// Tensor layout algebra (re-export of `quanta-tensor`).
    ///
    /// The shape/stride substrate under [`Array`]:
    /// [`Shape`], [`Layout`], the local ops
    /// (`transpose`, `permute`, `slice`, `broadcast`) and the CuTe-style
    /// global algebra (`compose`, `logical_divide`, …) that GEMM-, FFT-
    /// and sort-style tilings are expressed in. Pure host-side types —
    /// no GPU runtime — with the structural theorems proven in Lean.
    /// NumPy has no direct analogue; this is the layer NumPy hides
    /// inside `ndarray.strides`.
    pub mod layout {
        pub use quanta_tensor::*;
    }
}

/// Block-cooperative and device-wide GPU primitives (re-export of
/// `quanta-prims`).
///
/// Enabled by the `prims` feature. The CUB / rocPRIM / moderngpu
/// surface for Quanta: block-level reduce / scan / sort / histogram /
/// compact / top-k as `#[quanta::device]` functions your kernels call
/// cooperatively, plus the Tier-3 device-wide conveniences
/// (`device_reduce_*`, `device_sort_u32`) that take a host slice and
/// return the result. Pair with a backend feature (`software`, `metal`,
/// `vulkan`) for the device entry points; without one, the crate is a
/// pure-Rust reference library ([`reference`](crate::prims::reference), the
/// differential-testing oracle). Unlike CUB, the same Rust source runs
/// on Metal, Vulkan, and the CPU JIT.
#[cfg(feature = "prims")]
pub mod prims {
    pub use quanta_prims::*;
}

/// Reverse-mode automatic differentiation over `sci::Array`
/// (re-export of `quanta-autograd`).
///
/// Enabled by the `autograd` feature. A tape-based autodiff engine in
/// the PyTorch/`autograd` tradition: wrap arrays as
/// [`Var`](crate::autograd::Var)s on a [`Tape`](crate::autograd::Tape), compose
/// elementwise ops, activations, reductions, `matmul`, `conv2d`, and
/// pooling, then pull gradients back with `grad`. Each op's
/// vector-Jacobian product is the analytic derivative, proven in Lean.
/// It is a differentiation *primitive*, not an ML framework — layers
/// and optimisers beyond the basics belong to the future `quanta::nn`.
/// Enable `sci` alongside to name the `Array` type the API speaks.
#[cfg(feature = "autograd")]
pub mod autograd {
    pub use quanta_autograd::*;
}

/// The neural network stack (feature `nn`) — layers, fused kernels
/// (attention, norms, rotary), losses, optimizers, initialization, and
/// the training loop, built over the `autograd` tape and the `sci`
/// array. Single-node pure compute by design: the
/// distributed, actor-aware mirrors live upstream in dija-nn and wrap
/// this. The crate's completeness contract is `PARITY.md` at its root —
/// every declared item ships or carries a documented deferral.
#[cfg(feature = "nn")]
pub mod nn {
    pub use quanta_nn::*;
}

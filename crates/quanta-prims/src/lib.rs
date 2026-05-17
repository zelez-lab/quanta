//! Block-cooperative GPU primitives for Quanta kernels.
//!
//! v0.1 ships three load-bearing primitives plus the warp-shuffle
//! utilities they share:
//!
//! - **Block reduce** (`block_reduce_add_u32`): cooperative sum
//!   reduction across the threads of a workgroup, producing one
//!   value per workgroup.
//! - **Block scan** (`block_scan_add_u32`, planned): inclusive
//!   prefix sum across the workgroup.
//! - **Block radix sort** (`block_radix_sort_u32`, planned):
//!   cooperative key-only sort of a workgroup-sized tile.
//!
//! These primitives are the building blocks every downstream GPU
//! algorithm reduces to. They mirror CUB / rocPRIM / moderngpu's
//! block-level surface, with one important difference: Quanta's
//! primitives run on Metal, Vulkan, WebGPU, and the software CPU
//! backend from the same Rust source.
//!
//! ## Why a "block-cooperative" library
//!
//! Standalone GPU sort / scan / reduce as a top-level
//! "process this buffer" API has limited users — by the time you
//! have GPU data and want it sorted, you're inside a larger
//! pipeline. The valuable shape is *device functions your kernel
//! can call cooperatively*, just like CUB's `BlockReduceT::Sum`.
//!
//! Each primitive in this crate exposes:
//!
//! 1. A `#[quanta::device]` device-callable function — the
//!    cooperative kernel-body fragment.
//! 2. A reference single-thread CPU implementation — correctness
//!    oracle for differential testing.
//! 3. (Where applicable) a device-level convenience wrapper that
//!    builds a top-level kernel calling the device function.
//!
//! ## Usage — CPU reference
//!
//! ```
//! use quanta_prims::reference::reduce_add_u32;
//!
//! let xs = [1u32, 2, 3, 4, 5];
//! assert_eq!(reduce_add_u32(&xs), 15);
//! ```
//!
//! ## Usage — inside a kernel (with `gpu` feature)
//!
//! ```ignore
//! use quanta::*;
//! use quanta_prims::block_reduce_add_u32_kernel;
//!
//! #[quanta::kernel(workgroup_size = [256, 1, 1])]
//! fn my_reduce(data: &[u32], out: &mut [u32]) {
//!     let i = quark_id();
//!     let block = nucleus_id();
//!
//!     // Each quark loads its element and contributes to a
//!     // block-wide sum. Only the first thread in the block
//!     // writes the result.
//!     let value = data[i as usize];
//!     let block_sum = block_reduce_add_u32_kernel(value);
//!     if proton_id() == 0 {
//!         out[block as usize] = block_sum;
//!     }
//! }
//! ```
//!
//! ## Algorithm overview
//!
//! **Block reduce** uses a two-stage pattern:
//!
//! 1. **Warp-level reduction** via the `reduce_add_u32` subgroup
//!    intrinsic — each lane contributes, warp leaders end up
//!    holding the warp sum.
//! 2. **Cross-warp reduction** via workgroup-shared memory — warp
//!    leaders write their partial sum to shared, then the first
//!    warp re-reduces over those partials.
//!
//! The result is the workgroup-wide sum, replicated in every
//! lane that participated.
//!
//! ## v0.1 scope
//!
//! Tier 1 (load-bearing core):
//! - [`block_reduce_add_u32_kernel`]   — block sum reduce
//! - `block_scan_add_u32_kernel`       — block prefix sum (planned)
//! - `block_radix_sort_u32_kernel`     — block key-only sort (planned)
//!
//! Tier 2 (extensions, point releases): histogram, top-k,
//! compact, segmented variants.

// Subgroup intrinsics are FFI imports — the `unsafe` is unavoidable
// at the call site. The reference module is pure safe Rust; only
// the gpu_kernel module uses unsafe.
//
// `missing_docs` is intentionally not denied because the
// `#[quanta::kernel]` macro emits statics + dispatch fns without
// doc strings. We rely on clippy + cargo doc warnings to catch
// undocumented public items in author-written code.
#![deny(rustdoc::broken_intra_doc_links)]

pub mod reference;

#[cfg(feature = "gpu")]
mod gpu_kernel;
#[cfg(feature = "gpu")]
pub use gpu_kernel::*;

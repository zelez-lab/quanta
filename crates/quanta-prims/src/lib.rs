//! Block-cooperative GPU primitives for Quanta kernels.
//!
//! v0.1 ships the load-bearing Tier-1 trio plus the warp
//! intrinsics they share:
//!
//! - **Block reduce**: cooperative add / min / max reduction
//!   across the threads of a workgroup, producing one value per
//!   workgroup. `block_reduce_add_X` / `min_X` / `max_X` for
//!   `X ∈ {u32, i32, f32}`.
//! - **Block scan**: inclusive prefix sum across the workgroup.
//!   `block_scan_add_X` for the same three types.
//! - **Block sort**: cooperative ascending sort of a 256-key
//!   tile. `block_radix_sort_u32_buffer` (bitonic algorithm).
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
//!     // Required: [u32; 32] scratch at slot 0 for the
//!     // cross-warp aggregation stage.
//!     #[quanta::shared] let scratch: [u32; 32];
//!
//!     let i = quark_id();
//!     let block = nucleus_id();
//!     let lane = proton_id();
//!
//!     if lane < 32u32 { scratch[lane] = 0u32; }
//!     barrier();
//!
//!     let value = data[i as usize];
//!     let block_sum = block_reduce_add_u32_kernel(value);
//!     if lane == 0u32 {
//!         out[block as usize] = block_sum;
//!     }
//! }
//! ```
//!
//! ## Algorithm overview
//!
//! **Block reduce** uses a two-stage pattern:
//!
//! 1. **Warp-level reduction** via the matching `reduce_*_X`
//!    subgroup intrinsic — every lane in a subgroup gets the
//!    warp-wide result.
//! 2. **Cross-warp reduction** via workgroup-shared memory.
//!    Lane 0 of each warp publishes its partial; warp 0
//!    re-reduces over the partials. After the second
//!    warp-reduce, lane 0 of the workgroup holds the
//!    block-wide total.
//!
//! Constraint: `workgroup_size ≤ subgroup_size²`.
//!
//! **Block scan** uses a three-stage variant: warp-local scan,
//! cross-warp totals via shared memory + an exclusive scan over
//! those totals, then apply the per-warp prefix offset.
//!
//! **Block sort** (`block_radix_sort_u32_buffer`) is a stable
//! multi-bit LSD radix sort: 16 passes of 2-bit digits, each pass
//! ranking via a packed Hillis-Steele prefix sum and scattering
//! stably. The key-value variant (`block_sort_kv_u32_buffer`)
//! uses the bitonic compare-exchange network instead (unstable,
//! but moves payloads with one shared array per stream).
//!
//! ## v0.1 scope
//!
//! Tier 1 (load-bearing core, all shipped):
//! - `block_reduce_add` / `min` / `max` × `{u32, i32, f32}`
//! - `block_scan_add` × `{u32, i32, f32}`
//! - `block_radix_sort_u32` (stable LSD radix, 256 keys per
//!   workgroup)
//!
//! Tier 2 (all shipped):
//! - `block_compact_u32_buffer` — per-block stream compaction
//!   with explicit predicate array
//! - `block_histogram_u32_buffer` — per-block 256-bucket
//!   histogram via shared-memory atomics (Metal only today)
//! - `block_top_k_u32_buffer` — per-block top-K selection
//!   (sort-based, K up to 256)
//! - `block_segmented_scan_add_u32_buffer` /
//!   `block_segmented_reduce_add_u32_buffer` — head-flag
//!   segmented prefix sum and per-segment totals
//! - `block_sort_kv_u32_buffer` — key-value sort (bitonic,
//!   payload permuted alongside the keys)
//!
//! Tier 3 (device-wide convenience wrappers, all shipped):
//! - `device_reduce_{add,min,max}_{u32,i32,f32}` — host slice in,
//!   scalar out; multi-pass block reduce with identity padding
//! - `device_sort_u32` — host slice in, sorted copy out;
//!   device-wide bitonic network (one launch per pass)
//!
//! Each block primitive ships as a `#[quanta::device]` callable
//! function (e.g. `block_reduce_add_u32_kernel`) plus a top-level
//! `*_buffer` convenience kernel. See `gpu_kernel.rs` for the
//! full list; `device_wide.rs` holds the Tier-3 host wrappers.
//!
//! Still queued: key-value LSD radix (stable kv sort), segmented
//! sort.

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
mod device_wide;
#[cfg(feature = "gpu")]
mod gpu_kernel;
#[cfg(feature = "gpu")]
pub use device_wide::*;
#[cfg(feature = "gpu")]
pub use gpu_kernel::*;

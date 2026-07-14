//! Large 1D dispatches — every output element must be written.
//!
//! quanta-array kernels are LocalSize [1,1,1], so `dispatch(n)` issues
//! `n` workgroups. Above `maxComputeWorkGroupCount[0]` (65535 on V3D
//! and lavapipe) the Vulkan driver folds the dispatch into a 2D grid
//! with linearized `QuarkId` (see `quanta_ir::dispatch_fold`). This
//! suite pins the *semantics* on the software lane — an n well above
//! the fold threshold produces exactly the same values as a small-n
//! reference tiled up, with no dropped or duplicated elements. The
//! folded path itself is exercised on real Vulkan hardware; here we
//! guarantee the linearization change didn't disturb 1D dispatch
//! behavior anywhere else.

use quanta_array::Array;

/// The device these tests run on: the real GPU under a hardware backend
/// feature (metal / vulkan), else the CPU JIT (portable, no GPU needed).
fn gpu() -> quanta::Gpu {
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        quanta::init().expect("a GPU device")
    }
    #[cfg(not(any(feature = "metal", feature = "vulkan")))]
    {
        quanta::init_cpu()
    }
}

/// 401_408 = the full-batch activation shape [64, 8, 28, 28] that exceeded
/// the V3D grid cap. Kept moderate so the software lane stays fast.
const BIG_N: usize = 401_408;

#[test]
fn large_unary_writes_every_output() {
    let g = gpu();
    let data: Vec<f32> = (0..BIG_N).map(|i| ((i % 1013) as f32) - 500.0).collect();
    let a = Array::from_slice(&g, &data, &[BIG_N]).unwrap();
    let out = a.abs().unwrap().to_vec().unwrap();
    assert_eq!(out.len(), BIG_N);
    for (i, (got, x)) in out.iter().zip(&data).enumerate() {
        let want = x.abs();
        assert!(
            (got - want).abs() <= f32::EPSILON,
            "elem {i}: {got} vs {want}"
        );
    }
}

#[test]
fn large_binary_matches_small_reference_tiled() {
    let g = gpu();
    // The same 256-element tile repeated: results for the big dispatch
    // must equal the small dispatch's results tiled up. Any dropped
    // tail, duplicated write, or index skew shows up as a mismatch.
    let tile: Vec<f32> = (0..256).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let reps = BIG_N / tile.len();
    let big: Vec<f32> = tile.iter().cycle().take(BIG_N).copied().collect();

    let ta = Array::from_slice(&g, &tile, &[tile.len()]).unwrap();
    let small = ta.mul(&ta).unwrap().to_vec().unwrap();

    let ba = Array::from_slice(&g, &big, &[BIG_N]).unwrap();
    let out = ba.mul(&ba).unwrap().to_vec().unwrap();

    assert_eq!(out.len(), BIG_N);
    for r in 0..reps {
        let chunk = &out[r * tile.len()..(r + 1) * tile.len()];
        assert_eq!(chunk, &small[..], "tile {r} diverged from reference");
    }
}

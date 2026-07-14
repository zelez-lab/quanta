//! Folded 1D dispatch — shared constants + group-splitting math.
//!
//! Devices cap the per-dimension workgroup count (Broadcom V3D and
//! llvmpipe: `maxComputeWorkGroupCount[0]` = 65535). A 1D
//! thread-count dispatch whose group count exceeds the cap must
//! spread groups across the grid's y dimension. To keep 1D semantics
//! exact — no waste threads that would read/write out of bounds in
//! unguarded elementwise kernels — the fold uses a **fixed row width**
//! plus a remainder row at a base offset:
//!
//! * full-rows rectangle: `groups / FOLD_ROW_GROUPS` rows of exactly
//!   [`FOLD_ROW_GROUPS`] groups, dispatched at base (0, 0);
//! * remainder row: `groups % FOLD_ROW_GROUPS` groups, dispatched at
//!   base (0, `full_rows`) via `vkCmdDispatchBase` so its workgroup
//!   IDs continue where the rectangle stopped.
//!
//! The SPIR-V emitters bake the matching linearization into every
//! kernel:
//!
//! * `QuarkId`  = `gid.x + gid.y * (FOLD_ROW_GROUPS * workgroup_size_x)`
//! * `NucleusId` = `wg_id.x + wg_id.y * FOLD_ROW_GROUPS`
//!
//! For ordinary 1D dispatches (`gid.y == 0`) both formulas reduce to
//! the plain `.x` read, so behavior is unchanged unless a dispatch
//! actually folds. Because the row width is a compile-time constant
//! (not `gl_NumWorkGroups.x`), the formula holds across both the
//! rectangle and the remainder dispatch.
//!
//! The constant must agree between the emitters (this crate and the
//! forked emitter in `quanta-compiler`) and the Vulkan dispatch path
//! in the `quanta` crate — all three reference [`FOLD_ROW_GROUPS`]
//! (quanta-compiler through its quanta-ir dependency).

/// Fixed row width (in workgroups) of a folded 1D dispatch. Must not
/// exceed any target device's `maxComputeWorkGroupCount[0]` (65535 on
/// V3D and llvmpipe). A power of two keeps the linearization multiply
/// cheap and the arithmetic overflow-safe: with the largest supported
/// workgroup size (1024), `FOLD_ROW_GROUPS * wg_x` = 2^25, well inside
/// u32.
pub const FOLD_ROW_GROUPS: u32 = 32768;

/// Split a 1D group count into `(full_rows, remainder)` for the folded
/// dispatch: `full_rows` rows of exactly [`FOLD_ROW_GROUPS`] groups
/// plus one row of `remainder` groups at base y = `full_rows`.
/// `groups == full_rows * FOLD_ROW_GROUPS + remainder`, with
/// `remainder < FOLD_ROW_GROUPS`.
pub fn fold_groups(groups: u32) -> (u32, u32) {
    (groups / FOLD_ROW_GROUPS, groups % FOLD_ROW_GROUPS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_reconstruction() {
        for groups in [
            0u32,
            1,
            65535,
            65536,
            FOLD_ROW_GROUPS - 1,
            FOLD_ROW_GROUPS,
            FOLD_ROW_GROUPS + 1,
            401_408, // relu on [64, 8, 28, 28]
            451_584, // im2col intermediate
            2_147_483_647,
            u32::MAX,
        ] {
            let (rows, rem) = fold_groups(groups);
            assert_eq!(
                rows as u64 * FOLD_ROW_GROUPS as u64 + rem as u64,
                groups as u64,
                "groups={groups}"
            );
            assert!(rem < FOLD_ROW_GROUPS);
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)] // deliberate compile-time invariant checks
    fn row_width_fits_v3d_and_lavapipe_limits() {
        // maxComputeWorkGroupCount[0] is 65535 on both targets.
        assert!(FOLD_ROW_GROUPS <= 65535);
        // Power of two (documented invariant).
        assert_eq!(FOLD_ROW_GROUPS.count_ones(), 1);
        // Linearization constant can't overflow u32 for the largest
        // workgroup size Vulkan devices report (1024).
        assert!((FOLD_ROW_GROUPS as u64) * 1024 < u32::MAX as u64);
    }

    #[test]
    fn cnn_shapes_fold_within_grid_limits() {
        // The full-batch MNIST CNN shapes that exceeded the V3D cap
        // (LocalSize [1,1,1] → groups == thread count).
        for n in [401_408u32, 451_584] {
            let (rows, rem) = fold_groups(n);
            assert!((1..=65535).contains(&rows));
            assert!(rem <= 65535);
        }
    }
}

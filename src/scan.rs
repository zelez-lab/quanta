//! Parallel prefix sum (exclusive scan) library.
//!
//! Provides work-efficient scan primitives built on the Blelloch algorithm.
//! V1 ships as JIT kernels (KernelDef embedded and compiled at first use).
//!
//! ## Algorithm
//!
//! Blelloch two-pass exclusive scan:
//! 1. **Up-sweep (reduce)**: Build partial sums in a binary tree pattern
//! 2. **Down-sweep (distribute)**: Propagate prefix sums back down
//!
//! Complexity: O(n) work, O(log n) steps per workgroup.
//!
//! ## Limitations (V1)
//!
//! - Single workgroup: handles up to `workgroup_size` elements (1024)
//! - For larger arrays: call `exclusive_scan_chunked` which dispatches
//!   multiple workgroups and combines results
//! - Only f32 and u32 types supported

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use quanta_ir::{BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

/// Maximum elements handled by a single workgroup scan.
pub const SCAN_WORKGROUP_SIZE: u32 = 256;

/// Build a KernelDef that performs an exclusive prefix sum on f32 data.
///
/// The kernel operates on a single workgroup. Each thread loads one element
/// into shared memory, performs the Blelloch up-sweep and down-sweep, then
/// writes the result back.
///
/// Parameters:
/// - slot 0: `data` (read-write f32 buffer)
/// - slot 1: `count` (push constant u32 - number of elements)
pub fn build_exclusive_scan_f32() -> KernelDef {
    build_exclusive_scan(ScalarType::F32, ConstValue::F32(0.0))
}

/// Build a KernelDef that performs an exclusive prefix sum on u32 data.
///
/// Same structure as the f32 variant but operates on unsigned integers.
pub fn build_exclusive_scan_u32() -> KernelDef {
    build_exclusive_scan(ScalarType::U32, ConstValue::U32(0))
}

#[allow(clippy::vec_init_then_push)]
fn build_exclusive_scan(ty: ScalarType, zero: ConstValue) -> KernelDef {
    // Register allocator
    let mut next_reg = 0u32;
    let mut alloc_reg = || {
        let r = Reg(next_reg);
        next_reg += 1;
        r
    };

    // --- Thread indexing ---
    let r_lid = alloc_reg(); // proton_id
    let r_gid = alloc_reg(); // group_id
    let r_gs = alloc_reg(); // group_size
    let r_quark = alloc_reg(); // quark_id (global)

    // --- Constants ---
    let r_zero = alloc_reg(); // identity element (0)
    let r_one = alloc_reg(); // constant 1
    let r_two = alloc_reg(); // constant 2

    // --- Working registers ---
    let r_val = alloc_reg(); // loaded value
    let r_count_reg = alloc_reg(); // count from push constant (loaded via field slot 1)
    let r_in_bounds = alloc_reg(); // quark_id < count
    let r_shared_id = 0u32; // shared memory declaration id

    // --- Up-sweep registers ---
    let r_stride = alloc_reg();
    let r_stride2 = alloc_reg();
    let r_ai = alloc_reg();
    let r_bi = alloc_reg();
    let r_va = alloc_reg();
    let r_vb = alloc_reg();
    let r_sum = alloc_reg();
    let r_active = alloc_reg();
    let r_idx_check = alloc_reg();

    // --- Down-sweep registers ---
    let r_ds_stride = alloc_reg();
    let r_ds_stride2 = alloc_reg();
    let r_ds_ai = alloc_reg();
    let r_ds_bi = alloc_reg();
    let r_ds_va = alloc_reg();
    let r_ds_vb = alloc_reg();
    let r_ds_sum = alloc_reg();
    let r_ds_active = alloc_reg();
    let r_ds_idx_check = alloc_reg();
    let _r_half_gs = alloc_reg();

    // --- Loop registers ---
    let r_up_iter = alloc_reg();
    let r_up_count = alloc_reg();
    let r_down_iter = alloc_reg();
    let r_down_count = alloc_reg();

    // --- Additional ---
    let r_result = alloc_reg();
    let r_last_idx = alloc_reg();
    let r_is_last = alloc_reg();

    let mut body = Vec::new();

    // Get thread indices
    body.push(KernelOp::ProtonId { dst: r_lid });
    body.push(KernelOp::NucleusId { dst: r_gid });
    body.push(KernelOp::ProtonSize { dst: r_gs });
    body.push(KernelOp::QuarkId { dst: r_quark });

    // Constants
    body.push(KernelOp::Const {
        dst: r_zero,
        value: zero,
    });
    body.push(KernelOp::Const {
        dst: r_one,
        value: ConstValue::U32(1),
    });
    body.push(KernelOp::Const {
        dst: r_two,
        value: ConstValue::U32(2),
    });

    // Declare shared memory
    body.push(KernelOp::SharedDecl {
        id: r_shared_id,
        ty,
        count: SCAN_WORKGROUP_SIZE,
    });

    // Load element: if in bounds, load from data[quark_id]; else load 0
    body.push(KernelOp::Load {
        dst: r_count_reg,
        field: 1,
        index: r_zero, // push constant at index 0 of slot 1
        ty: ScalarType::U32,
    });
    body.push(KernelOp::Cmp {
        dst: r_in_bounds,
        a: r_quark,
        b: r_count_reg,
        op: quanta_ir::CmpOp::Lt,
        ty: ScalarType::U32,
    });
    body.push(KernelOp::Branch {
        cond: r_in_bounds,
        then_ops: vec![KernelOp::Load {
            dst: r_val,
            field: 0,
            index: r_quark,
            ty,
        }],
        else_ops: vec![KernelOp::Const {
            dst: r_val,
            value: zero,
        }],
    });

    // Store to shared memory
    body.push(KernelOp::SharedStore {
        id: r_shared_id,
        index: r_lid,
        src: r_val,
        ty,
    });
    body.push(KernelOp::Barrier);

    // --- Up-sweep phase ---
    // for d = 0 to log2(n)-1:
    //   stride = 1 << (d+1)
    //   if lid % stride == stride - 1:
    //     shared[lid] += shared[lid - (stride >> 1)]
    //
    // We compute log2(group_size) iterations via a Loop with group_size/2 as count
    // but actually we use a fixed 8 iterations (log2(256) = 8)

    body.push(KernelOp::Const {
        dst: r_up_count,
        value: ConstValue::U32(8), // log2(256) = 8
    });
    body.push(KernelOp::Const {
        dst: r_stride,
        value: ConstValue::U32(1),
    });

    body.push(KernelOp::Loop {
        count: r_up_count,
        iter_reg: r_up_iter,
        body: vec![
            // stride = stride * 2
            KernelOp::BinOp {
                dst: r_stride,
                a: r_stride,
                b: r_two,
                op: BinOp::Mul,
                ty: ScalarType::U32,
            },
            // stride2 = stride >> 1 (half stride)
            KernelOp::BinOp {
                dst: r_stride2,
                a: r_stride,
                b: r_one,
                op: BinOp::Shr,
                ty: ScalarType::U32,
            },
            // Check if this thread is active: (lid + 1) % stride == 0
            KernelOp::BinOp {
                dst: r_bi,
                a: r_lid,
                b: r_one,
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::BinOp {
                dst: r_idx_check,
                a: r_bi,
                b: r_stride,
                op: BinOp::Rem,
                ty: ScalarType::U32,
            },
            KernelOp::Cmp {
                dst: r_active,
                a: r_idx_check,
                b: r_zero,
                op: quanta_ir::CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: r_active,
                then_ops: vec![
                    // ai = lid - stride2
                    KernelOp::BinOp {
                        dst: r_ai,
                        a: r_lid,
                        b: r_stride2,
                        op: BinOp::Sub,
                        ty: ScalarType::U32,
                    },
                    // va = shared[ai]
                    KernelOp::SharedLoad {
                        dst: r_va,
                        id: r_shared_id,
                        index: r_ai,
                        ty,
                    },
                    // vb = shared[lid]
                    KernelOp::SharedLoad {
                        dst: r_vb,
                        id: r_shared_id,
                        index: r_lid,
                        ty,
                    },
                    // sum = va + vb
                    KernelOp::BinOp {
                        dst: r_sum,
                        a: r_va,
                        b: r_vb,
                        op: BinOp::Add,
                        ty,
                    },
                    // shared[lid] = sum
                    KernelOp::SharedStore {
                        id: r_shared_id,
                        index: r_lid,
                        src: r_sum,
                        ty,
                    },
                ],
                else_ops: vec![],
            },
            KernelOp::Barrier,
        ],
    });

    // Set last element to zero (exclusive scan identity)
    body.push(KernelOp::BinOp {
        dst: r_last_idx,
        a: r_gs,
        b: r_one,
        op: BinOp::Sub,
        ty: ScalarType::U32,
    });
    body.push(KernelOp::Cmp {
        dst: r_is_last,
        a: r_lid,
        b: r_last_idx,
        op: quanta_ir::CmpOp::Eq,
        ty: ScalarType::U32,
    });
    body.push(KernelOp::Branch {
        cond: r_is_last,
        then_ops: vec![KernelOp::SharedStore {
            id: r_shared_id,
            index: r_lid,
            src: r_zero,
            ty,
        }],
        else_ops: vec![],
    });
    body.push(KernelOp::Barrier);

    // --- Down-sweep phase ---
    // for d = log2(n)-1 downto 0:
    //   stride = 1 << (d+1)
    //   if lid % stride == stride - 1:
    //     temp = shared[lid - stride/2]
    //     shared[lid - stride/2] = shared[lid]
    //     shared[lid] += temp
    //
    // We reverse the loop: start with full stride and halve each iteration

    body.push(KernelOp::Const {
        dst: r_down_count,
        value: ConstValue::U32(8),
    });
    body.push(KernelOp::Const {
        dst: r_ds_stride,
        value: ConstValue::U32(256), // start at group_size
    });

    body.push(KernelOp::Loop {
        count: r_down_count,
        iter_reg: r_down_iter,
        body: vec![
            // stride2 = stride >> 1
            KernelOp::BinOp {
                dst: r_ds_stride2,
                a: r_ds_stride,
                b: r_one,
                op: BinOp::Shr,
                ty: ScalarType::U32,
            },
            // Check if active: (lid + 1) % stride == 0
            KernelOp::BinOp {
                dst: r_ds_bi,
                a: r_lid,
                b: r_one,
                op: BinOp::Add,
                ty: ScalarType::U32,
            },
            KernelOp::BinOp {
                dst: r_ds_idx_check,
                a: r_ds_bi,
                b: r_ds_stride,
                op: BinOp::Rem,
                ty: ScalarType::U32,
            },
            KernelOp::Cmp {
                dst: r_ds_active,
                a: r_ds_idx_check,
                b: r_zero,
                op: quanta_ir::CmpOp::Eq,
                ty: ScalarType::U32,
            },
            KernelOp::Branch {
                cond: r_ds_active,
                then_ops: vec![
                    // ai = lid - stride2
                    KernelOp::BinOp {
                        dst: r_ds_ai,
                        a: r_lid,
                        b: r_ds_stride2,
                        op: BinOp::Sub,
                        ty: ScalarType::U32,
                    },
                    // temp = shared[ai]
                    KernelOp::SharedLoad {
                        dst: r_ds_va,
                        id: r_shared_id,
                        index: r_ds_ai,
                        ty,
                    },
                    // shared[ai] = shared[lid]
                    KernelOp::SharedLoad {
                        dst: r_ds_vb,
                        id: r_shared_id,
                        index: r_lid,
                        ty,
                    },
                    KernelOp::SharedStore {
                        id: r_shared_id,
                        index: r_ds_ai,
                        src: r_ds_vb,
                        ty,
                    },
                    // shared[lid] = shared[lid] + temp
                    KernelOp::BinOp {
                        dst: r_ds_sum,
                        a: r_ds_vb,
                        b: r_ds_va,
                        op: BinOp::Add,
                        ty,
                    },
                    KernelOp::SharedStore {
                        id: r_shared_id,
                        index: r_lid,
                        src: r_ds_sum,
                        ty,
                    },
                ],
                else_ops: vec![],
            },
            KernelOp::Barrier,
            // stride = stride >> 1 (halve for next iteration)
            KernelOp::BinOp {
                dst: r_ds_stride,
                a: r_ds_stride,
                b: r_one,
                op: BinOp::Shr,
                ty: ScalarType::U32,
            },
        ],
    });

    // Write result back to global memory if in bounds
    body.push(KernelOp::SharedLoad {
        dst: r_result,
        id: r_shared_id,
        index: r_lid,
        ty,
    });
    body.push(KernelOp::Branch {
        cond: r_in_bounds,
        then_ops: vec![KernelOp::Store {
            field: 0,
            index: r_quark,
            src: r_result,
            ty,
        }],
        else_ops: vec![],
    });

    KernelDef {
        name: String::from("exclusive_scan"),
        params: vec![
            KernelParam::FieldWrite {
                name: String::from("data"),
                slot: 0,
                scalar_type: ty,
            },
            KernelParam::FieldRead {
                name: String::from("count_buf"),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
        ],
        body,
        body_source: None,
        next_reg,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [SCAN_WORKGROUP_SIZE, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    }
}

/// Serialize the exclusive scan f32 kernel to wire bytes.
///
/// The returned bytes can be passed to `gpu.wave_jit()` to compile and
/// dispatch the scan kernel.
pub fn exclusive_scan_f32_bytes() -> Vec<u8> {
    quanta_ir::serialize_kernel(&build_exclusive_scan_f32())
}

/// Serialize the exclusive scan u32 kernel to wire bytes.
pub fn exclusive_scan_u32_bytes() -> Vec<u8> {
    quanta_ir::serialize_kernel(&build_exclusive_scan_u32())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_kernel_def_f32_is_valid() {
        let def = build_exclusive_scan_f32();
        assert_eq!(def.name, "exclusive_scan");
        assert_eq!(def.workgroup_size, [SCAN_WORKGROUP_SIZE, 1, 1]);
        assert_eq!(def.params.len(), 2);
        assert!(!def.body.is_empty());
    }

    #[test]
    fn scan_kernel_def_u32_is_valid() {
        let def = build_exclusive_scan_u32();
        assert_eq!(def.name, "exclusive_scan");
        assert!(!def.body.is_empty());
    }

    #[test]
    fn scan_kernel_serializes_roundtrip() {
        let def = build_exclusive_scan_f32();
        let bytes = quanta_ir::serialize_kernel(&def);
        let back = quanta_ir::deserialize_kernel(&bytes).unwrap();
        assert_eq!(back.name, def.name);
        assert_eq!(back.workgroup_size, def.workgroup_size);
        assert_eq!(back.body.len(), def.body.len());
    }

    #[test]
    fn scan_bytes_helpers_produce_valid_wire() {
        let f32_bytes = exclusive_scan_f32_bytes();
        assert!(!f32_bytes.is_empty());
        let def = quanta_ir::deserialize_kernel(&f32_bytes).unwrap();
        assert_eq!(def.name, "exclusive_scan");

        let u32_bytes = exclusive_scan_u32_bytes();
        assert!(!u32_bytes.is_empty());
        let def = quanta_ir::deserialize_kernel(&u32_bytes).unwrap();
        assert_eq!(def.name, "exclusive_scan");
    }
}

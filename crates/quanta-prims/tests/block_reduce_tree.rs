//! Differential tests for the subgroup-free tree-reduce family:
//! `block_reduce_{add,min,max}_{u32,i32,f32}_tree_buffer`.
//!
//! The tree kernels are the fallback for devices whose driver can't
//! lower subgroup arithmetic (Broadcom V3D). On subgroup-capable
//! backends (Metal / software CPU, where this suite runs) they must
//! produce the same per-block results as BOTH the reference oracle
//! and the subgroup kernels — exact for u32/i32 and for f32 min/max,
//! within a small tolerance for f32 add (different tree orders).
//!
//! A final structural test pins the "subgroup-free" property itself:
//! the embedded KernelDef IR of every tree kernel must contain no
//! Subgroup* op, so the SPIR-V emitters never emit
//! `OpGroupNonUniform*` (the exact opcode family V3D aborts on).
//!
//! Skips gracefully when no GPU backend is available.

#![cfg(feature = "gpu")]

use quanta_prims::{
    block_reduce_add_f32_buffer, block_reduce_add_f32_tree_buffer, block_reduce_add_i32_buffer,
    block_reduce_add_i32_tree_buffer, block_reduce_add_u32_buffer,
    block_reduce_add_u32_tree_buffer, block_reduce_max_f32_buffer,
    block_reduce_max_f32_tree_buffer, block_reduce_max_i32_buffer,
    block_reduce_max_i32_tree_buffer, block_reduce_max_u32_buffer,
    block_reduce_max_u32_tree_buffer, block_reduce_min_f32_buffer,
    block_reduce_min_f32_tree_buffer, block_reduce_min_i32_buffer,
    block_reduce_min_i32_tree_buffer, block_reduce_min_u32_buffer,
    block_reduce_min_u32_tree_buffer, reference,
};

const BLOCK: usize = 256;
const NUM_BLOCKS: usize = 4;

fn try_gpu() -> Option<quanta::Gpu> {
    // Real device when available (Metal / Vulkan); otherwise the
    // software CPU lane — unlike the sibling suites we don't skip,
    // because the tree kernels must also run correctly through the
    // CPU JIT route (barrier-segmented cooperative execution).
    Some(quanta::init().unwrap_or_else(|_| quanta::init_cpu()))
}

type Builder = fn(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>;

fn run<T: Copy + Default>(gpu: &quanta::Gpu, builder: Builder, data: &[T]) -> Vec<T> {
    let num_blocks = data.len() / BLOCK;
    let data_field = gpu.field::<T>(data.len()).unwrap();
    let out_field = gpu.field::<T>(num_blocks).unwrap();
    data_field.write(data).unwrap();
    out_field.write(&vec![T::default(); num_blocks]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &data_field);
    wave.bind(1, &out_field);
    let mut pulse = gpu.dispatch(&wave, data.len() as u32).unwrap();
    pulse.wait().unwrap();
    out_field.read().unwrap()
}

fn per_block<T: Copy, R>(data: &[T], f: impl Fn(&[T]) -> R) -> Vec<R> {
    data.chunks(BLOCK).map(|c| f(c)).collect()
}

/// Run the SUBGROUP reference kernel, but only on devices that support
/// subgroup arithmetic. On a device without it (e.g. Broadcom V3D, which
/// *aborts the process* on `OpGroupNonUniformFAdd`), the subgroup kernel
/// can't run at all — so the tree-vs-subgroup cross-check is skipped there.
/// The tree-vs-reference check (the correctness that matters) always runs.
fn run_sub<T: Copy + Default>(gpu: &quanta::Gpu, builder: Builder, data: &[T]) -> Option<Vec<T>> {
    gpu.supports_subgroups().then(|| run(gpu, builder, data))
}

// ── u32 / i32: exact agreement with reference AND subgroup kernel ──

#[test]
fn tree_add_u32_matches_reference_and_subgroup() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| (i * 37 + 11) as u32 % 1000)
        .collect();
    let tree = run(&gpu, block_reduce_add_u32_tree_buffer, &data);
    let want = per_block(&data, reference::reduce_add_u32);
    assert_eq!(tree, want, "tree vs reference");
    if let Some(sub) = run_sub(&gpu, block_reduce_add_u32_buffer, &data) {
        assert_eq!(tree, sub, "tree vs subgroup");
    }
}

#[test]
fn tree_add_i32_matches_reference_and_subgroup() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<i32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| ((i * 31 + 7) % 999) as i32 - 500)
        .collect();
    let tree = run(&gpu, block_reduce_add_i32_tree_buffer, &data);
    let want = per_block(&data, reference::reduce_add_i32);
    assert_eq!(tree, want, "tree vs reference");
    if let Some(sub) = run_sub(&gpu, block_reduce_add_i32_buffer, &data) {
        assert_eq!(tree, sub, "tree vs subgroup");
    }
}

#[test]
fn tree_min_max_u32_match_reference_and_subgroup() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<u32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| ((i as u64 * 2654435761) % 4_000_000_000) as u32)
        .collect();
    let tree_min = run(&gpu, block_reduce_min_u32_tree_buffer, &data);
    let tree_max = run(&gpu, block_reduce_max_u32_tree_buffer, &data);
    assert_eq!(
        tree_min,
        per_block(&data, reference::reduce_min_u32),
        "min vs reference"
    );
    assert_eq!(
        tree_max,
        per_block(&data, reference::reduce_max_u32),
        "max vs reference"
    );
    if let Some(sub) = run_sub(&gpu, block_reduce_min_u32_buffer, &data) {
        assert_eq!(tree_min, sub, "min vs subgroup");
    }
    if let Some(sub) = run_sub(&gpu, block_reduce_max_u32_buffer, &data) {
        assert_eq!(tree_max, sub, "max vs subgroup");
    }
}

#[test]
fn tree_min_max_i32_match_reference_and_subgroup() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<i32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| (((i as u64 * 2654435761) % 4_000_000) as i64 - 2_000_000) as i32)
        .collect();
    let tree_min = run(&gpu, block_reduce_min_i32_tree_buffer, &data);
    let tree_max = run(&gpu, block_reduce_max_i32_tree_buffer, &data);
    assert_eq!(
        tree_min,
        per_block(&data, reference::reduce_min_i32),
        "min vs reference"
    );
    assert_eq!(
        tree_max,
        per_block(&data, reference::reduce_max_i32),
        "max vs reference"
    );
    if let Some(sub) = run_sub(&gpu, block_reduce_min_i32_buffer, &data) {
        assert_eq!(tree_min, sub, "min vs subgroup");
    }
    if let Some(sub) = run_sub(&gpu, block_reduce_max_i32_buffer, &data) {
        assert_eq!(tree_max, sub, "max vs subgroup");
    }
}

// ── f32: min/max exact; add within tolerance of both oracles ───────

#[test]
fn tree_add_f32_matches_reference_and_subgroup_within_ulps() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<f32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| ((i * 13 + 5) % 201) as f32 / 7.0 - 14.0)
        .collect();
    let tree = run(&gpu, block_reduce_add_f32_tree_buffer, &data);
    let sub = run_sub(&gpu, block_reduce_add_f32_buffer, &data);
    let want = per_block(&data, reference::reduce_add_f32);
    for b in 0..NUM_BLOCKS {
        let scale = 1.0 + want[b].abs();
        assert!(
            (tree[b] - want[b]).abs() <= 1e-4 * scale,
            "block {b}: tree {} vs reference {}",
            tree[b],
            want[b]
        );
        if let Some(sub) = &sub {
            assert!(
                (tree[b] - sub[b]).abs() <= 1e-4 * scale,
                "block {b}: tree {} vs subgroup {}",
                tree[b],
                sub[b]
            );
        }
    }
}

#[test]
fn tree_min_max_f32_match_reference_and_subgroup() {
    let Some(gpu) = try_gpu() else { return };
    // Comparison-based reduce is order-independent: exact equality.
    let data: Vec<f32> = (0..BLOCK * NUM_BLOCKS)
        .map(|i| (((i as u64 * 2654435761) % 100_000) as f32 - 50_000.0) / 3.0)
        .collect();
    let tree_min = run(&gpu, block_reduce_min_f32_tree_buffer, &data);
    let tree_max = run(&gpu, block_reduce_max_f32_tree_buffer, &data);
    assert_eq!(
        tree_min,
        per_block(&data, reference::reduce_min_f32),
        "min vs reference"
    );
    assert_eq!(
        tree_max,
        per_block(&data, reference::reduce_max_f32),
        "max vs reference"
    );
    if let Some(sub) = run_sub(&gpu, block_reduce_min_f32_buffer, &data) {
        assert_eq!(tree_min, sub, "min vs subgroup");
    }
    if let Some(sub) = run_sub(&gpu, block_reduce_max_f32_buffer, &data) {
        assert_eq!(tree_max, sub, "max vs subgroup");
    }
}

// ── device_wide wrappers stay correct through the runtime gate ─────

#[test]
fn device_reduce_wrappers_still_correct() {
    let Some(gpu) = try_gpu() else { return };
    let data: Vec<f32> = (0..1000).map(|i| (i % 97) as f32 * 0.25).collect();
    let want: f32 = data.iter().sum();
    let got = quanta_prims::device_reduce_add_f32(&gpu, &data).unwrap();
    assert!(
        (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
        "{got} vs {want}"
    );
    let ints: Vec<i32> = (0..777).map(|i| (i % 51) - 25).collect();
    assert_eq!(
        quanta_prims::device_reduce_min_i32(&gpu, &ints).unwrap(),
        *ints.iter().min().unwrap()
    );
    assert_eq!(
        quanta_prims::device_reduce_max_i32(&gpu, &ints).unwrap(),
        *ints.iter().max().unwrap()
    );
}

// ── structural: tree kernels carry NO subgroup ops in their IR ──────

#[test]
fn tree_kernels_are_subgroup_free_in_ir() {
    use quanta_ir::KernelOp;

    fn has_subgroup(ops: &[KernelOp]) -> bool {
        ops.iter().any(|op| match op {
            KernelOp::SubgroupReduceAdd { .. }
            | KernelOp::SubgroupReduceMin { .. }
            | KernelOp::SubgroupReduceMax { .. }
            | KernelOp::SubgroupExclusiveAdd { .. }
            | KernelOp::SubgroupInclusiveAdd { .. }
            | KernelOp::SubgroupSize { .. }
            | KernelOp::WaveShuffle { .. }
            | KernelOp::WaveBallot { .. } => true,
            KernelOp::Branch {
                then_ops, else_ops, ..
            } => has_subgroup(then_ops) || has_subgroup(else_ops),
            KernelOp::Loop { body, .. } => has_subgroup(body),
            _ => false,
        })
    }

    // (name, embedded KernelDef IR emitted by #[quanta::kernel])
    let kernels: [(&str, &[u8]); 9] = [
        (
            "add_u32",
            quanta_prims::__BLOCK_REDUCE_ADD_U32_TREE_BUFFER_IR,
        ),
        (
            "add_i32",
            quanta_prims::__BLOCK_REDUCE_ADD_I32_TREE_BUFFER_IR,
        ),
        (
            "add_f32",
            quanta_prims::__BLOCK_REDUCE_ADD_F32_TREE_BUFFER_IR,
        ),
        (
            "min_u32",
            quanta_prims::__BLOCK_REDUCE_MIN_U32_TREE_BUFFER_IR,
        ),
        (
            "min_i32",
            quanta_prims::__BLOCK_REDUCE_MIN_I32_TREE_BUFFER_IR,
        ),
        (
            "min_f32",
            quanta_prims::__BLOCK_REDUCE_MIN_F32_TREE_BUFFER_IR,
        ),
        (
            "max_u32",
            quanta_prims::__BLOCK_REDUCE_MAX_U32_TREE_BUFFER_IR,
        ),
        (
            "max_i32",
            quanta_prims::__BLOCK_REDUCE_MAX_I32_TREE_BUFFER_IR,
        ),
        (
            "max_f32",
            quanta_prims::__BLOCK_REDUCE_MAX_F32_TREE_BUFFER_IR,
        ),
    ];
    for (name, ir) in kernels {
        let def = quanta_ir::deserialize_kernel(ir).unwrap();
        assert!(
            !has_subgroup(&def.body),
            "tree kernel {name} contains subgroup ops — would abort on V3D"
        );
        for f in &def.device_functions {
            assert!(
                !has_subgroup(&f.body),
                "tree kernel {name} device fn {} contains subgroup ops",
                f.name
            );
        }
    }
}

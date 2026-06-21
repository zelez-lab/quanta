#![cfg(feature = "render")]
//! Integration tests for `IndirectCommandBuffer` (steps 032 + 033).
//!
//! Exercises the typed ICB API on the CPU software device:
//! - record N dispatches into an ICB,
//! - execute, observe the cumulative effect,
//! - verify the lifecycle invariants proven in
//!   `Quanta.Icb` (Lean) and `quanta-api/icb_safety.rs` (Verus).
//!
//! The CPU driver refines the abstract `Icb.execute` semantics:
//! `execute(count)` runs the first `count` recorded dispatches in
//! order, which is exactly the `take count |> foldl exec` shape
//! proven in `t7002_partial_execute_eq_take_foldl`.
//!
//! Run: cargo test --test icb_basic --features software

#![cfg(feature = "software")]

use quanta::kernel::*;

fn build_add_one_kernel() -> Vec<u8> {
    // data[quark_id()] += 1.0
    let def = KernelDef {
        name: "icb_add_one".into(),
        params: vec![KernelParam::FieldWrite {
            name: "data".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(1.0),
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [4, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    quanta_ir::serialize_kernel(&def)
}

// ─── Lifecycle invariants (refines T7050–T7054) ───────────────────────────

#[test]
fn icb_create_returns_empty_buffer_with_capacity() {
    let gpu = quanta::init_cpu();
    let icb = gpu.indirect_command_buffer(8).unwrap();
    assert_eq!(icb.capacity(), 8);
    assert_eq!(icb.len(), 0);
    assert!(icb.is_empty());
}

#[test]
fn icb_record_extends_length_by_one() {
    // Refinement of T7051: each successful record increments the
    // recorded length by exactly 1.
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0; 4]).unwrap();
    let kernel = build_add_one_kernel();
    let mut wave = gpu.wave_jit(&kernel).unwrap();
    wave.bind(0, &field);

    let mut icb = gpu.indirect_command_buffer(4).unwrap();
    assert_eq!(icb.len(), 0);
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    assert_eq!(icb.len(), 1);
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    assert_eq!(icb.len(), 2);
}

#[test]
fn icb_record_fails_when_full() {
    // Refinement of T7052: record fails once recorded == cap.
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0; 4]).unwrap();
    let kernel = build_add_one_kernel();
    let mut wave = gpu.wave_jit(&kernel).unwrap();
    wave.bind(0, &field);

    let mut icb = gpu.indirect_command_buffer(2).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    let err = icb.record_dispatch(&wave, [1, 1, 1]).unwrap_err();
    assert!(err.to_string().contains("full"), "got: {err}");
}

#[test]
fn icb_execute_count_too_large_fails() {
    // Refinement of the can_execute precondition: count must be
    // ≤ recorded.
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0; 4]).unwrap();
    let kernel = build_add_one_kernel();
    let mut wave = gpu.wave_jit(&kernel).unwrap();
    wave.bind(0, &field);

    let mut icb = gpu.indirect_command_buffer(4).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    let err = icb.execute(2).unwrap_err();
    assert!(err.to_string().contains("exceeds"), "got: {err}");
}

// ─── Equivalence (refines T7000 / T7001) ───────────────────────────────────

#[test]
fn icb_execute_two_dispatches_matches_direct() {
    // The Lean theorem T7000 states:
    //   icb.execute(N) == foldl exec s [c1, c2, ..., cN]
    // This test exercises the refinement on the CPU device:
    // recording two `add_one` dispatches and executing the ICB has
    // the same effect as dispatching `add_one` twice directly.
    let gpu = quanta::init_cpu();
    let kernel = build_add_one_kernel();

    // ── Path A: two direct dispatches on field_a ──
    let field_a = gpu.field::<f32>(4).unwrap();
    field_a.write(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    let mut wave_a = gpu.wave_jit(&kernel).unwrap();
    wave_a.bind(0, &field_a);
    gpu.dispatch(&wave_a, 4).unwrap().wait().unwrap();
    gpu.dispatch(&wave_a, 4).unwrap().wait().unwrap();
    let direct = field_a.read().unwrap();

    // ── Path B: two recorded ICB dispatches on field_b ──
    let field_b = gpu.field::<f32>(4).unwrap();
    field_b.write(&[10.0, 20.0, 30.0, 40.0]).unwrap();
    let mut wave_b = gpu.wave_jit(&kernel).unwrap();
    wave_b.bind(0, &field_b);
    let mut icb = gpu.indirect_command_buffer(4).unwrap();
    icb.record_dispatch(&wave_b, [1, 1, 1]).unwrap();
    icb.record_dispatch(&wave_b, [1, 1, 1]).unwrap();
    icb.execute_all().unwrap();
    let via_icb = field_b.read().unwrap();

    assert_eq!(via_icb, direct);
    // Each element +2 over baseline.
    assert_eq!(via_icb, vec![12.0, 22.0, 32.0, 42.0]);
}

#[test]
fn icb_partial_execute_runs_only_first_n() {
    // Refines T7002: execute(count) runs the first `count` recorded
    // dispatches. Recording 3 dispatches and executing only 2 must
    // match recording-and-running 2.
    let gpu = quanta::init_cpu();
    let kernel = build_add_one_kernel();

    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0; 4]).unwrap();
    let mut wave = gpu.wave_jit(&kernel).unwrap();
    wave.bind(0, &field);

    let mut icb = gpu.indirect_command_buffer(8).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    icb.execute(2).unwrap(); // run only the first 2
    let result = field.read().unwrap();
    assert_eq!(result, vec![2.0; 4], "expected +2 (executed first 2 of 3)");
}

#[test]
fn icb_re_execute_repeats_dispatches() {
    // Refines T7053: execute is read-only on the ICB ghost state —
    // the recorded sequence is unchanged, so calling execute twice
    // runs the recorded dispatches twice.
    let gpu = quanta::init_cpu();
    let kernel = build_add_one_kernel();

    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[0.0; 4]).unwrap();
    let mut wave = gpu.wave_jit(&kernel).unwrap();
    wave.bind(0, &field);

    let mut icb = gpu.indirect_command_buffer(4).unwrap();
    icb.record_dispatch(&wave, [1, 1, 1]).unwrap();
    icb.execute_all().unwrap();
    icb.execute_all().unwrap();
    let result = field.read().unwrap();
    assert_eq!(result, vec![2.0; 4], "expected +2 (executed twice)");
}

// ─── Render-path recording (refines T7006) ───────────────────────────────

#[test]
fn icb_record_draw_extends_length_by_one() {
    // T7006 refinement on the CPU device: record_draw appends the
    // recorded sequence by exactly one Draw command. The CPU has
    // no rasterizer, so execute replays the Draw arm as a no-op
    // (still proof-correct: the recording shape is what T7006
    // states, not the visible side-effect).
    let gpu = quanta::init_cpu();
    let pipeline = gpu.pipeline(&quanta::PipelineDesc::default()).unwrap();
    let mut icb = gpu.indirect_command_buffer(4).unwrap();
    assert_eq!(icb.len(), 0);
    icb.record_draw(&pipeline, 6, 1).unwrap();
    assert_eq!(icb.len(), 1);
    icb.record_draw(&pipeline, 12, 2).unwrap();
    assert_eq!(icb.len(), 2);
    icb.execute_all().unwrap(); // CPU: Draw arm no-ops
}

#[test]
fn icb_record_draw_fails_when_full() {
    let gpu = quanta::init_cpu();
    let pipeline = gpu.pipeline(&quanta::PipelineDesc::default()).unwrap();
    let mut icb = gpu.indirect_command_buffer(1).unwrap();
    icb.record_draw(&pipeline, 6, 1).unwrap();
    let err = icb.record_draw(&pipeline, 6, 1).unwrap_err();
    assert!(err.to_string().contains("full"), "got: {err}");
}

#[test]
fn icb_drop_destroys_handle() {
    // Drop must release the underlying handle; subsequent
    // operations on the raw handle should be rejected.
    let gpu = quanta::init_cpu();
    let handle = {
        let icb = gpu.indirect_command_buffer(4).unwrap();
        icb.handle()
        // icb dropped here
    };
    // The raw-handle execute path should now fail because the
    // handle was destroyed.
    let err = gpu.indirect_buffer_execute(handle, 0).unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

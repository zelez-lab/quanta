#![cfg(feature = "render")]
//! Compile-pin for the seven *stub* shader-stage macros — tessellation,
//! mesh, and ray tracing.
//!
//! Each stub attribute (`tess_control`, `tess_eval`, `task`, `mesh`,
//! `ray_gen`, `closest_hit`, `miss`) must expand to a `ShaderBinary` with
//! the correct `ShaderStage` and no WGSL. A regression once dropped the
//! `wgsl` field from these expansions, so the generated code failed to
//! compile ("missing field `wgsl`"). This test *is* the guard: it applies
//! each attribute and asserts on the generated static, so a dropped field
//! breaks the build here.
//!
//! Deliberately no `#[quanta::vertex]` / `#[quanta::fragment]` /
//! `#[quanta::kernel]` — those invoke the shader compiler binary at build
//! time, which need not be present to run this test.
//!
//! Run: cargo test --test shader_stage_macros

#[quanta::tess_control]
fn stub_tess_control() {}

#[quanta::tess_eval]
fn stub_tess_eval() {}

#[quanta::task]
fn stub_task() {}

#[quanta::mesh]
fn stub_mesh() {}

#[quanta::ray_gen]
fn stub_ray_gen() {}

#[quanta::closest_hit]
fn stub_closest_hit() {}

#[quanta::miss]
fn stub_miss() {}

#[test]
fn stub_stages_carry_correct_stage_and_no_wgsl() {
    // Each accessor returns the generated static; assert its stage tag and
    // that WGSL is absent (stubs embed no source).
    let cases: [(&'static quanta::ShaderBinary, quanta::ShaderStage); 7] = [
        (stub_tess_control(), quanta::ShaderStage::TessControl),
        (stub_tess_eval(), quanta::ShaderStage::TessEval),
        (stub_task(), quanta::ShaderStage::Task),
        (stub_mesh(), quanta::ShaderStage::Mesh),
        (stub_ray_gen(), quanta::ShaderStage::RayGen),
        (stub_closest_hit(), quanta::ShaderStage::ClosestHit),
        (stub_miss(), quanta::ShaderStage::Miss),
    ];

    for (binary, expected_stage) in cases {
        assert_eq!(binary.stage, expected_stage);
        assert!(
            binary.wgsl.is_none(),
            "stub stage {:?} should embed no WGSL",
            expected_stage
        );
    }
}

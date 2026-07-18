//! Unit tests for the AST-driven MSL body emitter.
//!
//! These assert on the emitted MSL *text* — no `xcrun` needed. The three
//! previously-failing shapes (line-wrapped constructor path, statement-position
//! `if`, `&T` uniform deref) each get a fixture; the full-shader metallib
//! round-trip lives in `shader.rs`'s xcrun-gated tests.

use super::{BodyTail, emit_body};

/// Bodies are token-stringified from the proc macro, so the tests use the
/// spaced form (`Vec4 :: new`, `pos . x`) that `to_token_stream()` produces —
/// exactly what miscompiled before.
#[test]
fn line_wrapped_vec_constructor_lowers_to_float4() {
    // `Vec4 :: new` (proc-macro path split) must still become `float4(...)`.
    let msl = emit_body(
        "{ Vec4 :: new (1.0, 2.0, 3.0, 4.0) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap();
    assert!(
        msl.contains("return float4(1.0, 2.0, 3.0, 4.0);"),
        "got: {msl}"
    );
    assert!(!msl.contains("Vec4"), "Vec4 leaked into MSL: {msl}");
}

#[test]
fn vec2_vec3_constructors() {
    let msl2 = emit_body("{ Vec2 :: new (1.0, 2.0) }", BodyTail::Return, &[], None).unwrap();
    assert!(msl2.contains("float2(1.0, 2.0)"), "got: {msl2}");
    let msl3 = emit_body(
        "{ Vec3 :: new (1.0, 2.0, 3.0) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap();
    assert!(msl3.contains("float3(1.0, 2.0, 3.0)"), "got: {msl3}");
}

#[test]
fn statement_position_if_lowers_to_msl_if_else() {
    // A tail `if/else` returning a value — the old emitter passed the raw
    // `if cond { .. } else { .. }` to MSL, which rejects statement-position
    // `if` as an expression. Now each arm gets its own `return`.
    let body = "{ let x = 1.0 ; if x > 0.0 { Vec4 :: new (x, x, x, x) } else { Vec4 :: new (0.0, 0.0, 0.0, 0.0) } }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("if (x > 0.0) {"), "got: {msl}");
    assert!(msl.contains("} else {"), "got: {msl}");
    assert!(msl.contains("return float4(x, x, x, x);"), "got: {msl}");
    assert!(
        msl.contains("return float4(0.0, 0.0, 0.0, 0.0);"),
        "got: {msl}"
    );
}

#[test]
fn let_bound_if_expression_flows_through_a_local() {
    // `let d = if c { a } else { b };` — MSL has no if-expression, so declare
    // `auto d;` and assign in each branch.
    let body = "{ let d = if 1.0 > 0.0 { 2.0 } else { 3.0 } ; Vec4 :: new (d, d, d, d) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("float d;"), "got: {msl}");
    assert!(msl.contains("d = 2.0;"), "got: {msl}");
    assert!(msl.contains("d = 3.0;"), "got: {msl}");
    assert!(msl.contains("return float4(d, d, d, d);"), "got: {msl}");
}

#[test]
fn uniform_deref_drops_the_star() {
    // `*viewport` (a `&Vec2` uniform deref) must NOT reach MSL — the uniform is
    // bound as `constant float2&`, so the deref is implicit.
    let body = "{ let x = * viewport ; Vec4 :: new (x, x, x, x) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(!msl.contains('*'), "deref star leaked into MSL: {msl}");
    assert!(msl.contains("auto x = viewport;"), "got: {msl}");
}

#[test]
fn field_access_and_swizzles() {
    let body = "{ let a = pos . x ; let b = uv_rect . zw ; Vec4 :: new (a, b, a, b) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("auto a = pos.x;"), "got: {msl}");
    assert!(msl.contains("auto b = uv_rect.zw;"), "got: {msl}");
}

#[test]
fn casts_are_stripped() {
    let body = "{ let a = 3.0 as f32 ; Vec4 :: new (a, a, a, a) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("auto a = 3.0;"), "got: {msl}");
    assert!(!msl.contains(" as "), "cast leaked: {msl}");
}

#[test]
fn intrinsics_map_to_msl_builtins() {
    let body = "{ let a = min (1.0, 2.0) ; let b = smoothstep (- 0.75, 0.75, a) ; let c = length (Vec2 :: new (a, b)) ; Vec4 :: new (a, b, c, c) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("min(1.0, 2.0)"), "got: {msl}");
    assert!(msl.contains("smoothstep(-0.75, 0.75, a)"), "got: {msl}");
    assert!(msl.contains("length(float2(a, b))"), "got: {msl}");
}

#[test]
fn inverse_sqrt_aliases_to_rsqrt() {
    let msl = emit_body(
        "{ let a = inverse_sqrt (4.0) ; Vec4 :: new (a, a, a, a) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap();
    assert!(msl.contains("rsqrt(4.0)"), "got: {msl}");
}

#[test]
fn sample_call_kept_for_downstream_rewrite() {
    // The AST walker leaves `sample(0, uv)` as `sample(0.0, ...)`; the fragment
    // emitter rewrites it to `tex_0.sample(smp_0, ...)` afterward.
    let body = "{ let c = sample (0, Vec2 :: new (0.0, 0.0)) ; c }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("sample(0.0, float2(0.0, 0.0))"), "got: {msl}");
}

#[test]
fn swizzle_on_sample_result() {
    // `.x` directly on a sample() call — the glyph fragment shape. The
    // field access must survive the walk so the downstream texture
    // rewrite sees `sample(...)`.x intact.
    let body = "{ let coverage = sample (0, Vec2 :: new (0.0, 0.0)) . x ; coverage }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(
        msl.contains("sample(0.0, float2(0.0, 0.0)).x"),
        "got: {msl}"
    );
}

#[test]
fn assign_result_to_var_for_vertex() {
    let msl = emit_body(
        "{ Vec4 :: new (1.0, 2.0, 3.0, 4.0) }",
        BodyTail::Assign("pos_result"),
        &[],
        None,
    )
    .unwrap();
    assert!(
        msl.contains("pos_result = float4(1.0, 2.0, 3.0, 4.0);"),
        "got: {msl}"
    );
    assert!(
        !msl.contains("return"),
        "vertex must assign, not return: {msl}"
    );
}

#[test]
fn nested_lets_inside_if_branch() {
    // `if c { let a = ..; a } else { .. }` — the branch block has its own lets.
    let body =
        "{ let d = if 1.0 > 0.0 { let a = 5.0 ; a } else { 6.0 } ; Vec4 :: new (d, d, d, d) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(msl.contains("auto a = 5.0;"), "got: {msl}");
    assert!(msl.contains("d = a;"), "got: {msl}");
}

#[test]
fn rejects_unknown_intrinsic() {
    let err = emit_body(
        "{ let a = frobnicate (1.0) ; Vec4 :: new (a, a, a, a) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(
        err.contains("frobnicate"),
        "err should name the construct: {err}"
    );
}

#[test]
fn rejects_if_without_else() {
    let err = emit_body(
        "{ if 1.0 > 0.0 { Vec4 :: new (1.0, 1.0, 1.0, 1.0) } }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(
        err.to_lowercase().contains("else"),
        "err should mention else: {err}"
    );
}

#[test]
fn rejects_method_call() {
    let err = emit_body(
        "{ let a = foo . bar (1.0) ; a }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(err.contains("method call"), "err: {err}");
}

#[test]
fn for_loop_lowers_to_native_msl_for() {
    // Bounded `for i in 0 .. 4` over a `let mut` accumulator — the loop
    // becomes a native MSL `for` with a `uint` counter, and the counter
    // participates in float arithmetic inside the body.
    let body =
        "{ let mut a = 0.0 ; for i in 0 .. 4 { a = a + i * 0.5 ; } Vec4 :: new (a, a, a, 1.0) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(
        msl.contains("for (uint i = 0u; i < 4u; ++i) {"),
        "got: {msl}"
    );
    assert!(msl.contains("a = a + i * 0.5;"), "got: {msl}");
}

#[test]
fn for_loop_suffixed_and_nonzero_start_bounds() {
    // `2u32 .. 8u32` — the suffixed spelling and a non-zero start.
    let body =
        "{ let mut a = 0.0 ; for k in 2u32 .. 8u32 { a = a + 1.0 ; } Vec4 :: new (a, a, a, 1.0) }";
    let msl = emit_body(body, BodyTail::Return, &[], None).unwrap();
    assert!(
        msl.contains("for (uint k = 2u; k < 8u; ++k) {"),
        "got: {msl}"
    );
}

#[test]
fn rejects_for_loop_nonconst_bound() {
    let err = emit_body(
        "{ let mut a = 0.0 ; for i in 0 .. n { a = a + 1.0 ; } Vec4 :: new (a, a, a, 1.0) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(
        err.contains("compile-time constant"),
        "err should demand a constant bound: {err}"
    );
}

#[test]
fn rejects_for_loop_inclusive_range() {
    let err = emit_body(
        "{ let mut a = 0.0 ; for i in 0 ..= 4 { a = a + 1.0 ; } Vec4 :: new (a, a, a, 1.0) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(err.contains("..="), "err should name ..=: {err}");
}

// ─── Shared-struct varyings: vertex struct-literal tail + fragment receiver ──

fn surface() -> quanta_ir::ShaderVaryings {
    quanta_ir::ShaderVaryings {
        struct_name: "Surface".to_string(),
        position: "clip".to_string(),
        fields: vec![
            quanta_ir::VaryingField {
                name: "uv".to_string(),
                ty: quanta_ir::ShaderType::Vec2,
            },
            quanta_ir::VaryingField {
                name: "kind".to_string(),
                ty: quanta_ir::ShaderType::U32,
            },
        ],
        binding: None,
    }
}

fn surface_recv() -> quanta_ir::ShaderVaryings {
    quanta_ir::ShaderVaryings {
        binding: Some("s".to_string()),
        ..surface()
    }
}

#[test]
fn vertex_struct_literal_lowers_to_member_assignments() {
    let v = surface();
    let body = "{ Surface { clip : Vec4 :: new (0.0, 0.0, 0.0, 1.0) , uv : Vec2 :: new (0.5, 0.5) , kind : 3u32 } }";
    let msl = emit_body(body, BodyTail::StructOut("out"), &[], Some(&v)).unwrap();
    assert!(
        msl.contains("out.clip = float4(0.0, 0.0, 0.0, 1.0);"),
        "got: {msl}"
    );
    assert!(msl.contains("out.uv = float2(0.5, 0.5);"), "got: {msl}");
    assert!(msl.contains("out.kind = 3u;"), "got: {msl}");
}

#[test]
fn vertex_struct_literal_accepts_shorthand_and_any_order() {
    let v = surface();
    let body = "{ let uv = Vec2 :: new (0.5, 0.5) ; \
                Surface { kind : 0u32 , uv , clip : Vec4 :: new (0.0, 0.0, 0.0, 1.0) } }";
    let msl = emit_body(body, BodyTail::StructOut("out"), &[], Some(&v)).unwrap();
    // Assignments come out in DECLARATION order (position first), whatever
    // the literal's order.
    let clip_at = msl.find("out.clip =").expect("clip assigned");
    let uv_at = msl.find("out.uv = uv;").expect("uv shorthand assigned");
    let kind_at = msl.find("out.kind = 0u;").expect("kind assigned");
    assert!(clip_at < uv_at && uv_at < kind_at, "got: {msl}");
}

#[test]
fn vertex_struct_literal_rejects_missing_field() {
    let v = surface();
    let body =
        "{ Surface { clip : Vec4 :: new (0.0, 0.0, 0.0, 1.0) , uv : Vec2 :: new (0.5, 0.5) } }";
    let err = emit_body(body, BodyTail::StructOut("out"), &[], Some(&v)).unwrap_err();
    assert!(err.contains("missing field `kind`"), "err: {err}");
}

#[test]
fn vertex_struct_literal_rejects_unknown_field() {
    let v = surface();
    let body = "{ Surface { clip : Vec4 :: new (0.0, 0.0, 0.0, 1.0) , uv : Vec2 :: new (0.5, 0.5) , kind : 0u32 , bogus : 1.0 } }";
    let err = emit_body(body, BodyTail::StructOut("out"), &[], Some(&v)).unwrap_err();
    assert!(err.contains("unknown field `bogus`"), "err: {err}");
}

#[test]
fn vertex_tail_must_be_the_struct_literal() {
    let v = surface();
    let err = emit_body(
        "{ Vec4 :: new (0.0, 0.0, 0.0, 1.0) }",
        BodyTail::StructOut("out"),
        &[],
        Some(&v),
    )
    .unwrap_err();
    assert!(err.contains("struct literal"), "err: {err}");
}

#[test]
fn vertex_struct_literal_rejects_field_type_mismatch() {
    let v = surface();
    // `uv` is declared Vec2 but the literal provides a Vec3.
    let body = "{ Surface { clip : Vec4 :: new (0.0, 0.0, 0.0, 1.0) , uv : Vec3 :: new (0.0, 0.0, 0.0) , kind : 0u32 } }";
    let err = emit_body(body, BodyTail::StructOut("out"), &[], Some(&v)).unwrap_err();
    assert!(err.contains("expects float2"), "err: {err}");
}

#[test]
fn fragment_receiver_field_access_passes_through() {
    let v = surface_recv();
    let body = "{ Vec4 :: new (s . uv . x , s . uv . y , 0.0 , 1.0 ) }";
    let msl = emit_body(body, BodyTail::Return, &[], Some(&v)).unwrap();
    assert!(
        msl.contains("return float4(s.uv.x, s.uv.y, 0.0, 1.0);"),
        "got: {msl}"
    );
}

#[test]
fn fragment_receiver_position_field_reads_window_position() {
    let v = surface_recv();
    let body = "{ Vec4 :: new (s . clip . x / 64.0 , 0.0 , 0.0 , 1.0 ) }";
    let msl = emit_body(body, BodyTail::Return, &[], Some(&v)).unwrap();
    assert!(msl.contains("s.clip.x / 64.0"), "got: {msl}");
}

#[test]
fn fragment_receiver_unknown_field_is_rejected() {
    let v = surface_recv();
    let err = emit_body(
        "{ Vec4 :: new (s . bogus , 0.0 , 0.0 , 1.0 ) }",
        BodyTail::Return,
        &[],
        Some(&v),
    )
    .unwrap_err();
    assert!(
        err.contains("no field `bogus` on the varyings struct `Surface`"),
        "err: {err}"
    );
}

#[test]
fn fragment_bare_receiver_is_rejected() {
    let v = surface_recv();
    let err = emit_body("{ s }", BodyTail::Return, &[], Some(&v)).unwrap_err();
    assert!(err.contains("field access"), "err: {err}");
}

#[test]
fn fragment_u32_receiver_field_compares_integer() {
    let v = surface_recv();
    let body =
        "{ let c = if s . kind == 3u32 { 1.0 } else { 0.2 } ; Vec4 :: new (c , c , c , 1.0 ) }";
    let msl = emit_body(body, BodyTail::Return, &[], Some(&v)).unwrap();
    assert!(msl.contains("s.kind == 3u"), "got: {msl}");
}

#[test]
fn rejects_for_loop_counter_shadowing() {
    let err = emit_body(
        "{ let i = 1.0 ; for i in 0 .. 4 { } Vec4 :: new (i, i, i, 1.0) }",
        BodyTail::Return,
        &[],
        None,
    )
    .unwrap_err();
    assert!(err.contains("shadows"), "err: {err}");
}

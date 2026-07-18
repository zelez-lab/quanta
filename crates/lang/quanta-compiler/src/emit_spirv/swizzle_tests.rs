//! Postfix component/swizzle access on shader-expression values.
//!
//! Pins the grammar accepting `.x`/`.zw`-style access on any
//! value-producing atom — `sample(...)` results, `Vec4::new(...)`,
//! parenthesized expressions — not just named params. The acceptance
//! body is the real glyph fragment shape that previously fell back to
//! a passthrough module ("unexpected token: Dot").

use quanta_ir::{ShaderDef, ShaderStage, ShaderType, ShaderVaryings, VaryingField};

use crate::emit_spirv::emitter::SpvEmitter;
use crate::emit_spirv::tokenizer::tokenize_shader_expr;

/// A fragment consuming the given varyings through receiver `s` (the
/// shared-struct model — bodies read `s . <field>`). No varying interface at
/// all when `fields` is empty.
fn frag(fields: &[(&str, ShaderType)], body: &str) -> ShaderDef {
    ShaderDef {
        name: "swizzle_test".to_string(),
        stage: ShaderStage::Fragment,
        params: vec![],
        return_type: ShaderType::Vec4,
        body_source: body.to_string(),
        varyings: if fields.is_empty() {
            None
        } else {
            Some(ShaderVaryings {
                struct_name: "V".to_string(),
                position: "pos".to_string(),
                fields: fields
                    .iter()
                    .map(|(name, ty)| VaryingField {
                        name: name.to_string(),
                        ty: *ty,
                    })
                    .collect(),
                binding: Some("s".to_string()),
            })
        },
    }
}

/// Opcodes present in a SPIR-V binary (header skipped).
fn opcodes(spirv: &[u8]) -> Vec<u16> {
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let mut ops = Vec::new();
    let mut i = 5; // header words
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        ops.push((words[i] & 0xFFFF) as u16);
        i += word_count.max(1);
    }
    ops
}

const OP_VECTOR_SHUFFLE: u16 = 79;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_IMAGE_SAMPLE_IMPLICIT_LOD: u16 = 87;

/// The real glyph fragment body (post-macro slot form): swizzle on a
/// `sample()` result. Passthrough modules contain no sample op, so its
/// presence proves the body genuinely translated.
#[test]
fn sample_swizzle_glyph_body_translates() {
    let def = frag(
        &[
            ("corner", ShaderType::Vec2),
            ("size", ShaderType::Vec2),
            ("uv_rect", ShaderType::Vec4),
            ("color", ShaderType::Vec4),
        ],
        "let u = mix ( s . uv_rect . x , s . uv_rect . z , s . corner . x ) ; \
         let v = mix ( s . uv_rect . y , s . uv_rect . w , s . corner . y ) ; \
         let coverage = sample(0 , Vec2 :: new ( u , v ) ) . x ; \
         Vec4 :: new ( s . color . x * coverage , s . color . y * coverage , \
         s . color . z * coverage , s . color . w * coverage )",
    );
    let spirv = crate::emit_spirv::emit_fragment(&def).expect("glyph body must translate");
    let ops = opcodes(&spirv);
    assert!(
        ops.contains(&OP_IMAGE_SAMPLE_IMPLICIT_LOD),
        "sample() must be emitted (a passthrough module means the body failed translation)"
    );
    assert!(
        ops.contains(&OP_COMPOSITE_EXTRACT),
        "the .x swizzle must extract"
    );
}

/// Multi-component swizzle lowers to OpVectorShuffle.
#[test]
fn multi_component_swizzle_shuffles() {
    let def = frag(
        &[("uv_rect", ShaderType::Vec4)],
        "let t = s . uv_rect . zw ; Vec4 :: new ( t . x , t . y , 0.0 , 1.0 )",
    );
    let spirv = crate::emit_spirv::emit_fragment(&def).expect("swizzle body must translate");
    assert!(opcodes(&spirv).contains(&OP_VECTOR_SHUFFLE));
}

/// Swizzle directly on a constructor result.
#[test]
fn swizzle_on_constructor_value() {
    let def = frag(&[], "Vec4 :: new ( 0.25 , 0.5 , 0.75 , 1.0 ) . y");
    let spirv = crate::emit_spirv::emit_fragment(&def).expect("constructor swizzle must translate");
    assert!(opcodes(&spirv).contains(&OP_COMPOSITE_EXTRACT));
}

/// Out-of-range components are rejected with a named error at the
/// parser level (`.z` on a Vec2).
#[test]
fn out_of_range_component_errors() {
    let mut e = SpvEmitter::new();
    let f32_ty = e.ensure_type_f32();
    let vec2_ty = e.ensure_type_vector(f32_ty, 2);
    let params = vec![("corner".to_string(), 1u32, vec2_ty, ShaderType::Vec2)];
    let tokens = tokenize_shader_expr("corner . z");
    let mut pos = 0;
    let mut locals = Vec::new();
    let err = e
        .parse_atom(&tokens, &mut pos, &params, &mut locals)
        .expect_err(".z on Vec2 must be rejected");
    assert!(err.contains("out of range"), "unexpected error: {err}");
}

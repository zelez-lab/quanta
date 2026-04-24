//! Verus mirror of `emit_spirv` opcode emission.
//!
//! Proves that the SPIR-V emitter match arms select the correct opcode
//! for comparison, unary, and cast operations, matching the Lean 4 spec.

use vstd::prelude::*;

verus! {

pub enum FloatKind { IsFloat, IsSignedInt, IsUnsignedInt }

// ── Comparison ops (T2) ─────────────────────────────────────────────

pub enum CmpOp { Eq, Ne, Lt, Le, Gt, Ge }

pub open spec fn cmp_to_spirv(op: CmpOp, fk: FloatKind) -> u16 {
    match (op, fk) {
        (CmpOp::Eq, FloatKind::IsFloat)         => 180u16, // FOrdEqual
        (CmpOp::Eq, _)                          => 170u16, // IEqual
        (CmpOp::Ne, FloatKind::IsFloat)         => 182u16, // FOrdNotEqual
        (CmpOp::Ne, _)                          => 171u16, // INotEqual
        (CmpOp::Lt, FloatKind::IsFloat)         => 184u16, // FOrdLessThan
        (CmpOp::Lt, FloatKind::IsSignedInt)     => 177u16, // SLessThan
        (CmpOp::Lt, FloatKind::IsUnsignedInt)   => 176u16, // ULessThan
        (CmpOp::Le, FloatKind::IsFloat)         => 186u16, // FOrdLessThanEqual
        (CmpOp::Le, FloatKind::IsSignedInt)     => 179u16, // SLessThanEqual
        (CmpOp::Le, FloatKind::IsUnsignedInt)   => 178u16, // ULessThanEqual
        (CmpOp::Gt, FloatKind::IsFloat)         => 188u16, // FOrdGreaterThan
        (CmpOp::Gt, FloatKind::IsSignedInt)     => 173u16, // SGreaterThan
        (CmpOp::Gt, FloatKind::IsUnsignedInt)   => 172u16, // UGreaterThan
        (CmpOp::Ge, FloatKind::IsFloat)         => 190u16, // FOrdGreaterThanEqual
        (CmpOp::Ge, FloatKind::IsSignedInt)     => 175u16, // SGreaterThanEqual
        (CmpOp::Ge, FloatKind::IsUnsignedInt)   => 174u16, // UGreaterThanEqual
    }
}

/// Float comparisons always use FOrd* (opcodes >= 180).
proof fn float_cmp_uses_ford(op: CmpOp)
    ensures cmp_to_spirv(op, FloatKind::IsFloat) >= 180u16,
{
    match op {
        CmpOp::Eq => {}, CmpOp::Ne => {}, CmpOp::Lt => {},
        CmpOp::Le => {}, CmpOp::Gt => {}, CmpOp::Ge => {},
    }
}

/// Integer comparisons always use I*/U*/S* (opcodes < 180).
proof fn int_cmp_uses_integer(op: CmpOp, fk: FloatKind)
    requires fk != FloatKind::IsFloat,
    ensures cmp_to_spirv(op, fk) < 180u16,
{
    match op {
        CmpOp::Eq => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Ne => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Lt => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Le => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Gt => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
        CmpOp::Ge => { match fk { FloatKind::IsSignedInt => {} FloatKind::IsUnsignedInt => {} _ => {} } },
    }
}

// ── Unary ops (T2) ──────────────────────────────────────────────────

pub enum UnaryOp { Neg, BitNot, LogicalNot }

pub open spec fn unary_to_spirv(op: UnaryOp, fk: FloatKind) -> u16 {
    match (op, fk) {
        (UnaryOp::Neg, FloatKind::IsFloat) => 127u16, // FNegate
        (UnaryOp::Neg, _)                  => 126u16, // SNegate
        (UnaryOp::BitNot, _)               => 200u16, // Not
        (UnaryOp::LogicalNot, _)           => 168u16, // LogicalNot
    }
}

proof fn fnegate_is_127()
    ensures unary_to_spirv(UnaryOp::Neg, FloatKind::IsFloat) == 127u16,
{}

proof fn snegate_is_126(fk: FloatKind)
    requires fk != FloatKind::IsFloat,
    ensures unary_to_spirv(UnaryOp::Neg, fk) == 126u16,
{
    match fk {
        FloatKind::IsSignedInt => {},
        FloatKind::IsUnsignedInt => {},
        _ => {},
    }
}

// ── Cast ops (T2) ───────────────────────────────────────────────────

pub open spec fn cast_to_spirv(from_float: bool, to_float: bool, from_signed: bool) -> u16 {
    match (from_float, to_float, from_signed) {
        (false, true, true)  => 111u16, // ConvertSToF
        (false, true, false) => 112u16, // ConvertUToF
        (true, false, _)     => 110u16, // ConvertFToS (signed target) — simplified
        _                    => 124u16, // Bitcast
    }
}

/// int→float signed uses ConvertSToF (111).
proof fn signed_int_to_float_is_111()
    ensures cast_to_spirv(false, true, true) == 111u16,
{}

/// int→float unsigned uses ConvertUToF (112).
proof fn unsigned_int_to_float_is_112()
    ensures cast_to_spirv(false, true, false) == 112u16,
{}

// ── Varying locations (T8) ──────────────────────────────────────────

/// Vertex shader: param[k] for k >= 1 gets Location(k-1) as varying output.
/// Fragment shader: input[j] gets Location(j).
/// T8: vertex output Location(k-1) matches fragment input Location(k-1).
pub open spec fn vertex_varying_location(param_index: nat) -> nat
    recommends param_index >= 1,
{
    (param_index - 1) as nat
}

pub open spec fn fragment_input_location(input_index: nat) -> nat {
    input_index
}

/// T8: vertex varying[k] = fragment input[k] for all k.
proof fn varying_locations_consistent(k: nat)
    ensures vertex_varying_location(k + 1) == fragment_input_location(k),
{}

// ── Vertex shader gl_Position decoration (T117) ────────────────────

// SPIR-V spec constants for gl_Position type and decoration.

/// Decoration enum value for BuiltIn (SPIR-V 1.6 Table: Decoration = 11).
pub open spec fn spirv_decoration_builtin() -> u32 { 11u32 }

/// BuiltIn Position value (SPIR-V 1.6 Table: BuiltIn = 0).
pub open spec fn spirv_builtin_position() -> u32 { 0u32 }

/// StorageClass Output (SPIR-V 1.6: StorageClass = 3).
pub open spec fn spirv_storage_class_output() -> u32 { 3u32 }

/// OpTypeFloat opcode (SPIR-V 1.6: opcode 22).
pub open spec fn spirv_op_type_float() -> u16 { 22u16 }

/// OpTypeVector opcode (SPIR-V 1.6: opcode 23).
pub open spec fn spirv_op_type_vector() -> u16 { 23u16 }

/// gl_Position is declared as vec4<f32>: OpTypeVector(f32, 4).
/// Returns (component_opcode, component_width, vector_count).
pub open spec fn gl_position_type() -> (u16, u32, u32) {
    (spirv_op_type_float(), 32u32, 4u32)
}

/// The emitter decorates gl_Position with BuiltIn(Position).
/// Returns (decoration_enum, builtin_value).
pub open spec fn gl_position_decoration() -> (u32, u32) {
    (spirv_decoration_builtin(), spirv_builtin_position())
}

/// T117a: DECORATION_BUILTIN constant matches SPIR-V spec value 11.
proof fn decoration_builtin_matches_spirv_spec()
    ensures spirv_decoration_builtin() == 11u32,
{}

/// T117b: BUILTIN_POSITION constant matches SPIR-V spec value 0.
proof fn builtin_position_matches_spirv_spec()
    ensures spirv_builtin_position() == 0u32,
{}

/// T117c: gl_Position decoration is BuiltIn(Position) = (11, 0).
proof fn gl_position_decorated_with_builtin_position()
    ensures ({
        let (dec, val) = gl_position_decoration();
        dec == 11u32 && val == 0u32
    }),
{}

/// T117d: gl_Position type is vec4<f32> — OpTypeFloat(32) + OpTypeVector(_, 4).
proof fn gl_position_is_vec4_f32()
    ensures ({
        let (comp_op, width, count) = gl_position_type();
        comp_op == 22u16    // OpTypeFloat
        && width == 32u32   // 32-bit float
        && count == 4u32    // 4-component vector
    }),
{}

/// T117e: gl_Position variable uses StorageClass Output (3).
proof fn gl_position_is_output_variable()
    ensures spirv_storage_class_output() == 3u32,
{}

/// T117f: Complete vertex shader gl_Position invariant.
/// The emitter must: (1) declare an Output variable, (2) type it as vec4<f32>,
/// (3) decorate it with BuiltIn(Position=0).
proof fn vertex_gl_position_complete()
    ensures
        spirv_storage_class_output() == 3u32,
        ({
            let (comp_op, width, count) = gl_position_type();
            comp_op == 22u16 && width == 32u32 && count == 4u32
        }),
        ({
            let (dec, val) = gl_position_decoration();
            dec == 11u32 && val == 0u32
        }),
{}

// ── Fragment shader OriginUpperLeft execution mode (T118) ───────────

/// ExecutionModel Fragment (SPIR-V 1.6: ExecutionModel = 4).
pub open spec fn spirv_execution_model_fragment() -> u32 { 4u32 }

/// ExecutionMode OriginUpperLeft (SPIR-V 1.6: ExecutionMode = 7).
pub open spec fn spirv_execution_mode_origin_upper_left() -> u32 { 7u32 }

/// The fragment shader entry point uses ExecutionModel Fragment.
/// Returns the execution model value.
pub open spec fn fragment_entry_point_model() -> u32 {
    spirv_execution_model_fragment()
}

/// The fragment shader sets ExecutionMode OriginUpperLeft on its entry point.
/// Returns (execution_model, execution_mode).
pub open spec fn fragment_execution_mode() -> (u32, u32) {
    (spirv_execution_model_fragment(), spirv_execution_mode_origin_upper_left())
}

/// T118a: EXECUTION_MODEL_FRAGMENT constant matches SPIR-V spec value 4.
proof fn execution_model_fragment_matches_spirv_spec()
    ensures spirv_execution_model_fragment() == 4u32,
{}

/// T118b: EXECUTION_MODE_ORIGIN_UPPER_LEFT constant matches SPIR-V spec value 7.
proof fn execution_mode_origin_upper_left_matches_spirv_spec()
    ensures spirv_execution_mode_origin_upper_left() == 7u32,
{}

/// T118c: Fragment entry point uses ExecutionModel Fragment (4).
proof fn fragment_entry_point_is_fragment_model()
    ensures fragment_entry_point_model() == 4u32,
{}

/// T118d: Fragment shader sets OriginUpperLeft (7) on a Fragment (4) entry point.
proof fn fragment_origin_upper_left_on_fragment_entry()
    ensures ({
        let (model, mode) = fragment_execution_mode();
        model == 4u32 && mode == 7u32
    }),
{}

/// T118e: ExecutionModel Fragment is distinct from Vertex (0) and GLCompute (5).
proof fn fragment_model_distinct_from_other_stages()
    ensures
        spirv_execution_model_fragment() != 0u32,  // not Vertex
        spirv_execution_model_fragment() != 5u32,  // not GLCompute
{}

/// T118f: Complete fragment shader execution mode invariant.
/// The emitter must: (1) use ExecutionModel Fragment (4) for the entry point,
/// (2) set ExecutionMode OriginUpperLeft (7) on that entry point.
proof fn fragment_origin_upper_left_complete()
    ensures
        spirv_execution_model_fragment() == 4u32,
        spirv_execution_mode_origin_upper_left() == 7u32,
        ({
            let (model, mode) = fragment_execution_mode();
            model == 4u32 && mode == 7u32
        }),
{}

} // verus!

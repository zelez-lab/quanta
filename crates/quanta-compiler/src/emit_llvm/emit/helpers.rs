//! Op-level helpers: binop, cmp, unary, cast, math intrinsics, vector type construction.

use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicType, BasicTypeEnum, VectorType};
use inkwell::values::{BasicValueEnum, FastMathFlags};
use inkwell::{FloatPredicate, IntPredicate};

use quanta_ir::*;

use super::super::{is_float_type, scalar_to_llvm_type};

/// All fast-math flags combined: enables FMA contraction, reassociation,
/// reciprocal approximation, and assumes no NaN/Inf/negative-zero.
fn all_fast_math_flags() -> FastMathFlags {
    FastMathFlags::AllowReassoc
        | FastMathFlags::NoNaNs
        | FastMathFlags::NoInfs
        | FastMathFlags::NoSignedZeros
        | FastMathFlags::AllowReciprocal
        | FastMathFlags::AllowContract
        | FastMathFlags::ApproxFunc
}

/// Set fast-math flags on a value if it is a float instruction.
fn set_fast_math(val: BasicValueEnum<'_>) {
    if let BasicValueEnum::FloatValue(fv) = val
        && let Some(inst) = fv.as_instruction()
    {
        let _ = inst.set_fast_math_flags(all_fast_math_flags());
    }
}

pub(super) fn emit_binop<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    op: &BinOp,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    if is_float_type(ty) {
        let a = lhs.into_float_value();
        let b = rhs.into_float_value();
        let r = match op {
            BinOp::Add => builder.build_float_add(a, b, ""),
            BinOp::Sub => builder.build_float_sub(a, b, ""),
            BinOp::Mul => builder.build_float_mul(a, b, ""),
            BinOp::Div => builder.build_float_div(a, b, ""),
            BinOp::Rem => builder.build_float_rem(a, b, ""),
            BinOp::SatAdd => builder.build_float_add(a, b, ""), // float doesn't overflow
            BinOp::SatSub => builder.build_float_sub(a, b, ""), // float doesn't overflow
            _ => return Err("bitwise ops not supported on floats".into()),
        }
        .map_err(|e| e.to_string())?;
        let result: BasicValueEnum = r.into();
        set_fast_math(result);
        Ok(result)
    } else {
        let a = lhs.into_int_value();
        let b = rhs.into_int_value();
        let r = match op {
            BinOp::Add => builder.build_int_add(a, b, ""),
            BinOp::Sub => builder.build_int_sub(a, b, ""),
            BinOp::Mul => builder.build_int_mul(a, b, ""),
            BinOp::Div => builder.build_int_unsigned_div(a, b, ""),
            BinOp::Rem => builder.build_int_unsigned_rem(a, b, ""),
            BinOp::BitAnd => builder.build_and(a, b, ""),
            BinOp::BitOr => builder.build_or(a, b, ""),
            BinOp::BitXor => builder.build_xor(a, b, ""),
            BinOp::Shl => builder.build_left_shift(a, b, ""),
            BinOp::Shr => builder.build_right_shift(a, b, false, ""),
            BinOp::SatAdd => {
                // Saturating add: add then clamp overflow
                let sum = builder.build_int_add(a, b, "").map_err(|e| e.to_string())?;
                // Unsigned overflow: sum < a
                let overflow = builder
                    .build_int_compare(inkwell::IntPredicate::ULT, sum, a, "")
                    .map_err(|e| e.to_string())?;
                let max_val = a.get_type().const_all_ones();
                return builder
                    .build_select(overflow, max_val, sum, "")
                    .map_err(|e| e.to_string());
            }
            BinOp::SatSub => {
                let diff = builder.build_int_sub(a, b, "").map_err(|e| e.to_string())?;
                let underflow = builder
                    .build_int_compare(inkwell::IntPredicate::ULT, a, b, "")
                    .map_err(|e| e.to_string())?;
                let zero = a.get_type().const_zero();
                return builder
                    .build_select(underflow, zero, diff, "")
                    .map_err(|e| e.to_string());
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(r.into())
    }
}

pub(super) fn emit_cmp<'ctx>(
    builder: &Builder<'ctx>,
    lhs: BasicValueEnum<'ctx>,
    rhs: BasicValueEnum<'ctx>,
    op: &CmpOp,
    ty: &ScalarType,
) -> Result<inkwell::values::IntValue<'ctx>, String> {
    if is_float_type(ty) {
        let a = lhs.into_float_value();
        let b = rhs.into_float_value();
        let pred = match op {
            CmpOp::Eq => FloatPredicate::OEQ,
            CmpOp::Ne => FloatPredicate::ONE,
            CmpOp::Lt => FloatPredicate::OLT,
            CmpOp::Le => FloatPredicate::OLE,
            CmpOp::Gt => FloatPredicate::OGT,
            CmpOp::Ge => FloatPredicate::OGE,
        };
        builder
            .build_float_compare(pred, a, b, "cmp")
            .map_err(|e| e.to_string())
    } else {
        let a = lhs.into_int_value();
        let b = rhs.into_int_value();
        let pred = match op {
            CmpOp::Eq => IntPredicate::EQ,
            CmpOp::Ne => IntPredicate::NE,
            CmpOp::Lt => IntPredicate::ULT,
            CmpOp::Le => IntPredicate::ULE,
            CmpOp::Gt => IntPredicate::UGT,
            CmpOp::Ge => IntPredicate::UGE,
        };
        builder
            .build_int_compare(pred, a, b, "cmp")
            .map_err(|e| e.to_string())
    }
}

pub(super) fn emit_unary<'ctx>(
    builder: &Builder<'ctx>,
    val: BasicValueEnum<'ctx>,
    op: &UnaryOp,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    match op {
        UnaryOp::Neg => {
            if is_float_type(ty) {
                let r = builder
                    .build_float_neg(val.into_float_value(), "neg")
                    .map_err(|e| e.to_string())?;
                let result: BasicValueEnum = r.into();
                set_fast_math(result);
                Ok(result)
            } else {
                Ok(builder
                    .build_int_neg(val.into_int_value(), "neg")
                    .map_err(|e| e.to_string())?
                    .into())
            }
        }
        UnaryOp::BitNot => Ok(builder
            .build_not(val.into_int_value(), "not")
            .map_err(|e| e.to_string())?
            .into()),
        UnaryOp::LogicalNot => Ok(builder
            .build_not(val.into_int_value(), "lnot")
            .map_err(|e| e.to_string())?
            .into()),
    }
}

pub(super) fn emit_cast<'ctx>(
    context: &'ctx Context,
    builder: &Builder<'ctx>,
    val: BasicValueEnum<'ctx>,
    from: &ScalarType,
    to: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    let target_ty = scalar_to_llvm_type(context, to);

    match (is_float_type(from), is_float_type(to)) {
        (true, true) => {
            // float -> float (extend or truncate)
            Ok(builder
                .build_float_cast(val.into_float_value(), target_ty.into_float_type(), "fcast")
                .map_err(|e| e.to_string())?
                .into())
        }
        (true, false) => {
            // float -> int
            Ok(builder
                .build_float_to_unsigned_int(
                    val.into_float_value(),
                    target_ty.into_int_type(),
                    "f2i",
                )
                .map_err(|e| e.to_string())?
                .into())
        }
        (false, true) => {
            // int -> float
            Ok(builder
                .build_unsigned_int_to_float(
                    val.into_int_value(),
                    target_ty.into_float_type(),
                    "i2f",
                )
                .map_err(|e| e.to_string())?
                .into())
        }
        (false, false) => {
            // int -> int (extend or truncate)
            Ok(builder
                .build_int_cast(val.into_int_value(), target_ty.into_int_type(), "icast")
                .map_err(|e| e.to_string())?
                .into())
        }
    }
}

pub(super) fn emit_math_direct<'ctx>(
    context: &'ctx Context,
    module: &Module<'ctx>,
    builder: &Builder<'ctx>,
    arg_vals: &[BasicValueEnum<'ctx>],
    func: &MathFn,
    ty: &ScalarType,
) -> Result<BasicValueEnum<'ctx>, String> {
    let llvm_ty = scalar_to_llvm_type(context, ty);
    let type_suffix = match ty {
        ScalarType::F32 => ".f32",
        ScalarType::F64 => ".f64",
        ScalarType::F16 => ".f16",
        _ => return Err("math functions require float type".into()),
    };

    let intrinsic_name = match func {
        MathFn::Sin => format!("llvm.sin{}", type_suffix),
        MathFn::Cos => format!("llvm.cos{}", type_suffix),
        MathFn::Sqrt => format!("llvm.sqrt{}", type_suffix),
        MathFn::Exp => format!("llvm.exp{}", type_suffix),
        MathFn::Exp2 => format!("llvm.exp2{}", type_suffix),
        MathFn::Log => format!("llvm.log{}", type_suffix),
        MathFn::Log2 => format!("llvm.log2{}", type_suffix),
        MathFn::Pow => format!("llvm.pow{}", type_suffix),
        MathFn::Abs => format!("llvm.fabs{}", type_suffix),
        MathFn::Floor => format!("llvm.floor{}", type_suffix),
        MathFn::Ceil => format!("llvm.ceil{}", type_suffix),
        MathFn::Round => format!("llvm.round{}", type_suffix),
        MathFn::Fma => format!("llvm.fma{}", type_suffix),
        MathFn::Min => format!("llvm.minnum{}", type_suffix),
        MathFn::Max => format!("llvm.maxnum{}", type_suffix),
        // Functions without LLVM intrinsics -- use libdevice or expand
        MathFn::Tan
        | MathFn::Asin
        | MathFn::Acos
        | MathFn::Atan
        | MathFn::Atan2
        | MathFn::Rsqrt
        | MathFn::Clamp => {
            // Fallback: emit as a regular function call (target libdevice provides these)
            format!(
                "__nv_{}{}",
                format!("{:?}", func).to_lowercase(),
                type_suffix
            )
        }
    };

    let fn_type = match arg_vals.len() {
        1 => llvm_ty.fn_type(&[llvm_ty.into()], false),
        2 => llvm_ty.fn_type(&[llvm_ty.into(), llvm_ty.into()], false),
        3 => llvm_ty.fn_type(&[llvm_ty.into(), llvm_ty.into(), llvm_ty.into()], false),
        _ => return Err("math function with unsupported arity".into()),
    };

    let func_val = module
        .get_function(&intrinsic_name)
        .unwrap_or_else(|| module.add_function(&intrinsic_name, fn_type, None));

    let call_args: Vec<inkwell::values::BasicMetadataValueEnum<'ctx>> =
        arg_vals.iter().map(|v| (*v).into()).collect();

    let result = builder
        .build_call(func_val, &call_args, "math")
        .map_err(|e| e.to_string())?
        .try_as_basic_value()
        .basic()
        .ok_or("math function returned void")?;

    set_fast_math(result);
    Ok(result)
}

/// Create a fixed-width LLVM vector type from a scalar BasicTypeEnum.
pub(super) fn make_vec_type<'ctx>(scalar: BasicTypeEnum<'ctx>, size: u32) -> VectorType<'ctx> {
    match scalar {
        BasicTypeEnum::FloatType(t) => t.vec_type(size),
        BasicTypeEnum::IntType(t) => t.vec_type(size),
        BasicTypeEnum::PointerType(t) => t.vec_type(size),
        // Structs/arrays/vectors cannot form vector elements in LLVM --
        // this arm should never be reached for valid GPU IR.
        _ => panic!("unsupported scalar type for vector construction"),
    }
}

//! Atom parser — literals, identifiers, Vec constructors, math calls,
//! texture sampling, field access.

use super::constants::*;
use super::emitter::SpvEmitter;
use super::tokenizer::{ShaderToken, glsl_func_id};

impl SpvEmitter {
    /// Parse one atom, then apply postfix component/swizzle access
    /// (`.x`, `.zw`, …) uniformly — the value-producing atom forms
    /// (`sample(...)`, `Vec4::new(...)`, math calls, parenthesized
    /// expressions) all accept it, matching the MSL emitter's surface.
    pub(crate) fn parse_atom(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let mut cur = self.parse_atom_inner(tokens, pos, params, locals)?;
        // Postfix loop: chained accesses (`v.zw.x`) reduce left to right.
        while tokens.get(*pos) == Some(&ShaderToken::Dot)
            && let Some(ShaderToken::Ident(field)) = tokens.get(*pos + 1)
        {
            if !is_swizzle(field) {
                break;
            }
            let field = field.clone();
            *pos += 2;
            cur = self.apply_swizzle(cur.0, cur.1, &field)?;
        }
        Ok(cur)
    }

    fn parse_atom_inner(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos >= tokens.len() {
            return Err("unexpected end of expression".to_string());
        }

        match &tokens[*pos] {
            ShaderToken::Float(val) => {
                *pos += 1;
                let id = self.emit_constant_f32(*val);
                Ok((id, quanta_ir::ShaderType::F32))
            }
            ShaderToken::Open => {
                *pos += 1;
                let result = self.parse_conditional(tokens, pos, params, locals)?;
                if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                    *pos += 1;
                }
                // `(expr).x` (the `(*uniform).x` shape) is handled by the
                // postfix loop in `parse_atom`.
                Ok(result)
            }
            ShaderToken::Ident(name) => {
                let name = name.clone();
                *pos += 1;

                // Vec{2,3,4}::new(args)
                if (name == "Vec2" || name == "Vec3" || name == "Vec4")
                    && *pos + 2 <= tokens.len()
                    && tokens.get(*pos) == Some(&ShaderToken::ColonColon)
                    && tokens
                        .get(*pos + 1)
                        .map(|t| matches!(t, ShaderToken::Ident(n) if n == "new"))
                        .unwrap_or(false)
                {
                    return self.parse_vec_constructor(&name, tokens, pos, params, locals);
                }

                // Texture sampling: sample(slot, uv)
                if name == "sample" && *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
                    return self.parse_texture_sample(tokens, pos, params, locals);
                }

                // Window-space position: frag_coord() → vec4 (x,y = pixel
                // coords, z = depth, w = 1/w). Loads the `BuiltIn FragCoord`
                // Input declared by `emit_fragment_shader`; outside a fragment
                // body the var is absent and the error routes the body to the
                // passthrough fallback, mirroring the MSL vertex rejection.
                if name == "frag_coord" && *pos < tokens.len() && tokens[*pos] == ShaderToken::Open
                {
                    *pos += 1; // '('
                    consume_call_close(tokens, pos);
                    let Some(var_id) = self.frag_coord_var else {
                        return Err(
                            "frag_coord() is only available in fragment shader bodies".to_string()
                        );
                    };
                    let f32_ty = self.ensure_type_f32();
                    let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                    let loaded = self.alloc_id();
                    Self::emit_op(&mut self.sec_function, OP_LOAD, &[vec4_ty, loaded, var_id]);
                    return Ok((loaded, quanta_ir::ShaderType::Vec4));
                }

                // Screen-space derivatives — core fragment-stage ops.
                if matches!(name.as_str(), "fwidth" | "dpdx" | "dpdy")
                    && *pos < tokens.len()
                    && tokens[*pos] == ShaderToken::Open
                {
                    *pos += 1; // '('
                    let (arg, ty) = self.parse_conditional(tokens, pos, params, locals)?;
                    consume_call_close(tokens, pos);
                    let opcode = match name.as_str() {
                        "fwidth" => OP_FWIDTH,
                        "dpdx" => OP_DPDX,
                        _ => OP_DPDY,
                    };
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(ty);
                    Self::emit_op(&mut self.sec_function, opcode, &[ty_id, result, arg]);
                    return Ok((result, ty));
                }

                // Math function calls: sin(x), sqrt(x), clamp(x, a, b), etc.
                if *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
                    if let Some(glsl_op) = glsl_func_id(&name) {
                        return self.parse_glsl_call(&name, glsl_op, tokens, pos, params, locals);
                    }

                    // dot() as SPIR-V OpDot (not GLSL ext)
                    if name == "dot" {
                        return self.parse_dot_call(tokens, pos, params, locals);
                    }
                }

                // Slice indexing: `name[index]` on a `&[T]` slice param.
                if *pos < tokens.len() && tokens[*pos] == ShaderToken::BracketOpen {
                    return self.parse_slice_index(&name, tokens, pos, params, locals);
                }

                // param.field (e.g. pos.x)
                if *pos + 1 < tokens.len()
                    && tokens[*pos] == ShaderToken::Dot
                    && let ShaderToken::Ident(field) = &tokens[*pos + 1]
                {
                    return self.parse_field_access(&name, field, pos, params, locals);
                }

                // Bare identifier — local, param, or boolean literal
                self.parse_bare_ident(&name, params, locals)
            }
            other => Err(format!("unexpected token: {other:?}")),
        }
    }

    /// Index a `&[T]` slice param: `name [ index ]`. The index is any scalar
    /// (f32-typed) expression the grammar accepts; it is truncated to u32 with
    /// `OpConvertFToU`, then an `OpAccessChain` into member 0 (the runtime
    /// array) at that index yields a pointer that is `OpLoad`ed as the element
    /// type. Bounds are UNCHECKED — the GPU storage-buffer contract. Indexing a
    /// non-slice param or an unknown name is an error (which sends the whole
    /// body to the SPIR-V passthrough fallback).
    fn parse_slice_index(
        &mut self,
        name: &str,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let Some(&(var_id, elem_ty, elem_shader_ty)) = self.slice_params.get(name) else {
            return Err(format!(
                "`{name}[..]` indexes a non-slice value; only `&[T]` slice params support indexing"
            ));
        };
        *pos += 1; // '['
        let (index_f, _) = self.parse_conditional(tokens, pos, params, locals)?;
        if tokens.get(*pos) == Some(&ShaderToken::BracketClose) {
            *pos += 1;
        } else {
            return Err(format!("expected `]` after index into `{name}`"));
        }

        // f32 index → u32 (truncating).
        let uint_ty = self.ensure_type_u32();
        let index_u = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_CONVERT_F_TO_U,
            &[uint_ty, index_u, index_f],
        );

        // OpAccessChain [member 0, index] into the runtime array, then load.
        let zero = self.emit_constant_u32(0);
        let ptr_elem = self.ensure_type_pointer(STORAGE_CLASS_STORAGE_BUFFER, elem_ty);
        let access = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_ACCESS_CHAIN,
            &[ptr_elem, access, var_id, zero, index_u],
        );
        let loaded = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_LOAD, &[elem_ty, loaded, access]);
        Ok((loaded, elem_shader_ty))
    }

    fn parse_vec_constructor(
        &mut self,
        name: &str,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 2; // skip :: new
        let count = match name {
            "Vec2" => 2u32,
            "Vec3" => 3,
            "Vec4" => 4,
            _ => unreachable!(),
        };
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Open {
            *pos += 1;
        }
        let mut components = Vec::new();
        for i in 0..count {
            if i > 0 && *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
                *pos += 1;
            }
            let (c, _) = self.parse_conditional(tokens, pos, params, locals)?;
            components.push(c);
        }
        consume_call_close(tokens, pos);
        let f32_ty = self.ensure_type_f32();
        let vec_ty = self.ensure_type_vector(f32_ty, count);
        let result = self.alloc_id();
        let mut ops = vec![vec_ty, result];
        ops.extend_from_slice(&components);
        Self::emit_op(&mut self.sec_function, OP_COMPOSITE_CONSTRUCT, &ops);
        let out_ty = match count {
            2 => quanta_ir::ShaderType::Vec2,
            3 => quanta_ir::ShaderType::Vec3,
            4 => quanta_ir::ShaderType::Vec4,
            _ => unreachable!(),
        };
        Ok((result, out_ty))
    }

    fn parse_texture_sample(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 1; // skip '('
        let slot = if let ShaderToken::Float(f) = &tokens[*pos] {
            let s = *f as u32;
            *pos += 1;
            s
        } else {
            return Err("sample() first arg must be a literal slot number".to_string());
        };
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
            *pos += 1;
        }
        let (uv_id, _) = self.parse_conditional(tokens, pos, params, locals)?;
        consume_call_close(tokens, pos);

        let Some(&(sampler_var, sampled_image_ty)) = self.texture_samplers.get(&slot) else {
            return Err(format!("texture slot {} not declared", slot));
        };
        let loaded_sampler = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_LOAD,
            &[sampled_image_ty, loaded_sampler, sampler_var],
        );
        let f32_ty = self.ensure_type_f32();
        let vec4_ty = self.ensure_type_vector(f32_ty, 4);
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_IMAGE_SAMPLE_IMPLICIT_LOD,
            &[vec4_ty, result, loaded_sampler, uv_id],
        );
        Ok((result, quanta_ir::ShaderType::Vec4))
    }

    fn parse_glsl_call(
        &mut self,
        name: &str,
        glsl_op: u32,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 1; // skip '('
        let mut args = Vec::new();
        let mut first_ty = quanta_ir::ShaderType::F32;
        loop {
            if tokens.get(*pos) == Some(&ShaderToken::Close) {
                break;
            }
            if !args.is_empty() && tokens.get(*pos) == Some(&ShaderToken::Comma) {
                *pos += 1;
                // A trailing comma leaves `)` next — stop before parsing a
                // phantom argument (rustfmt wraps calls with a trailing comma).
                if tokens.get(*pos) == Some(&ShaderToken::Close) {
                    break;
                }
            }
            let (a, t) = self.parse_conditional(tokens, pos, params, locals)?;
            if args.is_empty() {
                first_ty = t;
            }
            args.push(a);
        }
        consume_call_close(tokens, pos);

        let result_ty = if name == "dot" || name == "length" || name == "distance" {
            quanta_ir::ShaderType::F32
        } else {
            first_ty
        };

        let ext = self.ensure_glsl_ext();
        let result = self.alloc_id();
        let ty_id = self.shader_type_id(result_ty);
        let mut ops = vec![ty_id, result, ext, glsl_op];
        ops.extend_from_slice(&args);
        Self::emit_op(&mut self.sec_function, OP_EXT_INST, &ops);
        Ok((result, result_ty))
    }

    fn parse_dot_call(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 1; // skip '('
        let (a, _) = self.parse_conditional(tokens, pos, params, locals)?;
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
            *pos += 1;
        }
        let (b, _) = self.parse_conditional(tokens, pos, params, locals)?;
        consume_call_close(tokens, pos);
        let f32_ty = self.ensure_type_f32();
        let result = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_DOT, &[f32_ty, result, a, b]);
        Ok((result, quanta_ir::ShaderType::F32))
    }

    /// Extract a single component (`.x`/`.r`, …) from a composite
    /// VALUE id.
    fn extract_component(
        &mut self,
        value: u32,
        field: &str,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let index = component_index(
            field
                .chars()
                .next()
                .ok_or_else(|| "empty field".to_string())?,
        )
        .ok_or_else(|| format!("unknown field: {field}"))?;
        let f32_ty = self.ensure_type_f32();
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[f32_ty, result, value, index],
        );
        Ok((result, quanta_ir::ShaderType::F32))
    }

    /// Apply a component or multi-component swizzle (`.x`, `.zw`, `.rgb`,
    /// …) to a vector VALUE, validating each component against the source
    /// arity. Single components lower to `OpCompositeExtract`; runs of
    /// 2–4 lower to `OpVectorShuffle` (source vector on both operands).
    fn apply_swizzle(
        &mut self,
        value: u32,
        ty: quanta_ir::ShaderType,
        field: &str,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let Some(arity) = vector_arity(ty) else {
            return Err(format!(
                "cannot swizzle `.{field}` on a non-vector value ({ty:?})"
            ));
        };
        let indices: Vec<u32> = field
            .chars()
            .map(|c| component_index(c).ok_or_else(|| format!("unknown field: {field}")))
            .collect::<Result<_, _>>()?;
        if indices.is_empty() || indices.len() > 4 {
            return Err(format!("unsupported swizzle length: .{field}"));
        }
        if let Some(bad) = indices.iter().find(|&&i| i >= arity) {
            let name = ['x', 'y', 'z', 'w'][*bad as usize];
            return Err(format!(
                "swizzle component `{name}` out of range for {ty:?}"
            ));
        }
        if indices.len() == 1 {
            return self.extract_component(value, &field[..1]);
        }
        let f32_ty = self.ensure_type_f32();
        let out_len = indices.len() as u32;
        let vec_ty = self.ensure_type_vector(f32_ty, out_len);
        let result = self.alloc_id();
        let mut ops = vec![vec_ty, result, value, value];
        ops.extend_from_slice(&indices);
        Self::emit_op(&mut self.sec_function, OP_VECTOR_SHUFFLE, &ops);
        let out_ty = match out_len {
            2 => quanta_ir::ShaderType::Vec2,
            3 => quanta_ir::ShaderType::Vec3,
            _ => quanta_ir::ShaderType::Vec4,
        };
        Ok((result, out_ty))
    }

    fn parse_field_access(
        &mut self,
        name: &str,
        field: &str,
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let field = field.to_string();
        *pos += 2;

        if let Some((_, var_id, type_id, sty)) = params.iter().find(|(n, _, _, _)| *n == name) {
            let sty = *sty;
            let loaded = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[*type_id, loaded, *var_id],
            );
            return self.apply_swizzle(loaded, sty, &field);
        }
        if let Some((_, val_id, val_ty)) = locals.iter().find(|(n, _, _)| *n == name) {
            let (val_id, val_ty) = (*val_id, *val_ty);
            return self.apply_swizzle(val_id, val_ty, &field);
        }
        Err(format!("unknown variable: {name}"))
    }

    fn parse_bare_ident(
        &mut self,
        name: &str,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if name == "true" {
            let id = self.emit_constant_f32(1.0);
            return Ok((id, quanta_ir::ShaderType::F32));
        }
        if name == "false" {
            let id = self.emit_constant_f32(0.0);
            return Ok((id, quanta_ir::ShaderType::F32));
        }
        if let Some((_, val_id, val_ty)) = locals.iter().find(|(n, _, _)| *n == name) {
            return Ok((*val_id, *val_ty));
        }
        if let Some((_, var_id, type_id, sty)) = params.iter().find(|(n, _, _, _)| *n == name) {
            let loaded = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[*type_id, loaded, *var_id],
            );
            return Ok((loaded, *sty));
        }
        Err(format!("unknown identifier: {name}"))
    }
}

/// Close a call's argument list: skip an optional trailing comma, then the
/// `)`. rustfmt appends a trailing comma to every wrapped multi-line call and
/// the macro's token printer preserves it, so `f(a, b,)` is an ordinary shape
/// on the wire — every call-shaped form (constructors, intrinsics, `sample`,
/// derivatives, `dot`) routes its closing paren through here so the comma is
/// tolerated uniformly rather than left to defeat the caller's parse.
fn consume_call_close(tokens: &[ShaderToken], pos: &mut usize) {
    if tokens.get(*pos) == Some(&ShaderToken::Comma) {
        *pos += 1;
    }
    if tokens.get(*pos) == Some(&ShaderToken::Close) {
        *pos += 1;
    }
}

/// True when `field` is a pure component/swizzle run (`x`, `zw`, `rgba`, …)
/// — the only postfix accesses the shader grammar accepts on values.
fn is_swizzle(field: &str) -> bool {
    !field.is_empty() && field.len() <= 4 && field.chars().all(|c| component_index(c).is_some())
}

/// Component letter → vector index (`x`/`r` → 0 … `w`/`a` → 3).
fn component_index(c: char) -> Option<u32> {
    match c {
        'x' | 'r' => Some(0),
        'y' | 'g' => Some(1),
        'z' | 'b' => Some(2),
        'w' | 'a' => Some(3),
        _ => None,
    }
}

/// Component count of a vector shader type; `None` for scalars/matrices.
fn vector_arity(ty: quanta_ir::ShaderType) -> Option<u32> {
    match ty {
        quanta_ir::ShaderType::Vec2 => Some(2),
        quanta_ir::ShaderType::Vec3 => Some(3),
        quanta_ir::ShaderType::Vec4 => Some(4),
        _ => None,
    }
}

#[cfg(test)]
#[path = "swizzle_tests.rs"]
mod tests;

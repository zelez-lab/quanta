//! Atom parser — literals, identifiers, Vec constructors, math calls,
//! texture sampling, field access.

use super::constants::*;
use super::emitter::SpvEmitter;
use super::tokenizer::{ShaderToken, glsl_func_id};

impl SpvEmitter {
    pub(crate) fn parse_atom(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
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
                // `(expr).x` — component access on a parenthesized
                // value (the `(*uniform).x` shape).
                if tokens.get(*pos) == Some(&ShaderToken::Dot)
                    && let Some(ShaderToken::Ident(field)) = tokens.get(*pos + 1)
                {
                    let field = field.clone();
                    *pos += 2;
                    return self.extract_component(result.0, &field);
                }
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

                // Screen-space derivatives — core fragment-stage ops.
                if matches!(name.as_str(), "fwidth" | "dpdx" | "dpdy")
                    && *pos < tokens.len()
                    && tokens[*pos] == ShaderToken::Open
                {
                    *pos += 1; // '('
                    let (arg, ty) = self.parse_conditional(tokens, pos, params, locals)?;
                    if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                        *pos += 1;
                    }
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

    fn parse_vec_constructor(
        &mut self,
        name: &str,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
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
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
            *pos += 1;
        }
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
        locals: &[(String, u32, quanta_ir::ShaderType)],
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
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
            *pos += 1;
        }

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
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 1; // skip '('
        let mut args = Vec::new();
        let mut first_ty = quanta_ir::ShaderType::F32;
        loop {
            if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
                break;
            }
            if !args.is_empty() && *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
                *pos += 1;
            }
            let (a, t) = self.parse_conditional(tokens, pos, params, locals)?;
            if args.is_empty() {
                first_ty = t;
            }
            args.push(a);
        }
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
            *pos += 1;
        }

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
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        *pos += 1; // skip '('
        let (a, _) = self.parse_conditional(tokens, pos, params, locals)?;
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Comma {
            *pos += 1;
        }
        let (b, _) = self.parse_conditional(tokens, pos, params, locals)?;
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Close {
            *pos += 1;
        }
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
        let index = match field {
            "x" | "r" => 0u32,
            "y" | "g" => 1,
            "z" | "b" => 2,
            "w" | "a" => 3,
            _ => return Err(format!("unknown field: {field}")),
        };
        let f32_ty = self.ensure_type_f32();
        let result = self.alloc_id();
        Self::emit_op(
            &mut self.sec_function,
            OP_COMPOSITE_EXTRACT,
            &[f32_ty, result, value, index],
        );
        Ok((result, quanta_ir::ShaderType::F32))
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

        if let Some((_, var_id, type_id, _)) = params.iter().find(|(n, _, _, _)| *n == name) {
            let loaded = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_LOAD,
                &[*type_id, loaded, *var_id],
            );
            return self.extract_component(loaded, &field);
        }
        if let Some((_, val_id, _)) = locals.iter().find(|(n, _, _)| *n == name) {
            let val_id = *val_id;
            return self.extract_component(val_id, &field);
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

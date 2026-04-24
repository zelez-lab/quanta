//! Shader body expression parser — recursive descent into SPIR-V.
//!
//! Parses tokenized Rust shader body into SPIR-V instructions.
//! Supports: Vec constructors, field access, arithmetic, float literals,
//! let bindings, math functions (GLSL.std.450), matrix-vector multiply,
//! if/else, comparisons, and uniform parameter access via push constants.

use super::constants::*;
use super::emitter::SpvEmitter;
use super::tokenizer::{ShaderCmpOp, ShaderToken, tokenize_shader_expr};

impl SpvEmitter {
    /// Evaluate a shader body_source and emit SPIR-V instructions.
    /// Returns the SPIR-V result ID of the final expression and its type.
    pub(crate) fn eval_shader_body(
        &mut self,
        body_source: &str,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let src = body_source.trim();
        let src = if src.starts_with('{') && src.ends_with('}') {
            &src[1..src.len() - 1]
        } else {
            src
        };

        let mut locals: Vec<(String, u32, quanta_ir::ShaderType)> = Vec::new();
        let mut remaining = src.trim();

        // Process let-bindings
        while remaining.starts_with("let ") {
            let semi = remaining.find(';').ok_or("missing ; after let binding")?;
            let binding = &remaining[..semi];
            remaining = remaining[semi + 1..].trim();

            let binding = binding.trim_start_matches("let ").trim();
            let binding = binding.trim_start_matches("mut ").trim();
            let eq_pos = binding.find('=').ok_or("missing = in let binding")?;
            let var_name = binding[..eq_pos].trim().to_string();
            let expr_str = binding[eq_pos + 1..].trim();

            let (val_id, val_ty) = self.eval_expr(expr_str, params, &locals)?;
            locals.push((var_name, val_id, val_ty));
        }

        let remaining = remaining.trim().trim_end_matches(';').trim();
        if remaining.is_empty() {
            return Err("empty shader body".to_string());
        }
        self.eval_expr(remaining, params, &locals)
    }

    pub(crate) fn eval_expr(
        &mut self,
        src: &str,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let tokens = tokenize_shader_expr(src);
        let mut pos = 0;
        self.parse_conditional(&tokens, &mut pos, params, locals)
    }

    pub(crate) fn parse_conditional(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Ident("if".to_string()) {
            *pos += 1;
            let (cond, _) = self.parse_comparison(tokens, pos, params, locals)?;

            // Skip '{'
            if *pos < tokens.len() && tokens[*pos] == ShaderToken::BraceOpen {
                *pos += 1;
            }
            // Find matching '}' for then-branch
            let then_start = *pos;
            let mut depth = 1i32;
            while *pos < tokens.len() && depth > 0 {
                match &tokens[*pos] {
                    ShaderToken::BraceOpen => depth += 1,
                    ShaderToken::BraceClose => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    *pos += 1;
                }
            }
            let then_tokens: Vec<ShaderToken> = tokens[then_start..*pos].to_vec();
            if *pos < tokens.len() {
                *pos += 1; // skip '}'
            }

            // Parse else branch
            let has_else =
                *pos < tokens.len() && tokens[*pos] == ShaderToken::Ident("else".to_string());

            if !has_else {
                return Err("if without else not supported in shader expressions".to_string());
            }
            *pos += 1; // skip 'else'
            if *pos < tokens.len() && tokens[*pos] == ShaderToken::BraceOpen {
                *pos += 1;
            }
            let else_start = *pos;
            depth = 1;
            while *pos < tokens.len() && depth > 0 {
                match &tokens[*pos] {
                    ShaderToken::BraceOpen => depth += 1,
                    ShaderToken::BraceClose => depth -= 1,
                    _ => {}
                }
                if depth > 0 {
                    *pos += 1;
                }
            }
            let else_tokens: Vec<ShaderToken> = tokens[else_start..*pos].to_vec();
            if *pos < tokens.len() {
                *pos += 1; // skip '}'
            }

            // Emit SPIR-V structured control flow
            let then_label = self.alloc_id();
            let else_label = self.alloc_id();
            let merge_label = self.alloc_id();

            Self::emit_op(
                &mut self.sec_function,
                OP_SELECTION_MERGE,
                &[merge_label, 0],
            );
            Self::emit_op(
                &mut self.sec_function,
                OP_BRANCH_CONDITIONAL,
                &[cond, then_label, else_label],
            );

            // Then block
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
            let mut then_pos = 0;
            let (then_id, then_ty) =
                self.parse_conditional(&then_tokens, &mut then_pos, params, locals)?;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Else block
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
            let mut else_pos = 0;
            let (else_id, _) =
                self.parse_conditional(&else_tokens, &mut else_pos, params, locals)?;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Merge block with OpPhi
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(then_ty);
            Self::emit_op(
                &mut self.sec_function,
                OP_PHI,
                &[ty_id, result, then_id, then_label, else_id, else_label],
            );

            return Ok((result, then_ty));
        }
        self.parse_comparison(tokens, pos, params, locals)
    }

    pub(crate) fn parse_comparison(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (left, ty) = self.parse_additive(tokens, pos, params, locals)?;
        if *pos < tokens.len() {
            let cmp_op = match &tokens[*pos] {
                ShaderToken::Cmp(c) => Some(*c),
                _ => None,
            };
            if let Some(op) = cmp_op {
                *pos += 1;
                let (right, _) = self.parse_additive(tokens, pos, params, locals)?;
                let bool_ty = self.ensure_type_bool();
                let result = self.alloc_id();
                let opcode = match op {
                    ShaderCmpOp::Lt => OP_FORD_LESS_THAN,
                    ShaderCmpOp::Gt => OP_FORD_GREATER_THAN,
                    ShaderCmpOp::Le => OP_FORD_LESS_THAN_EQUAL,
                    ShaderCmpOp::Ge => OP_FORD_GREATER_THAN_EQUAL,
                    ShaderCmpOp::Eq => OP_FORD_EQUAL,
                    ShaderCmpOp::Ne => OP_FORD_NOT_EQUAL,
                };
                Self::emit_op(
                    &mut self.sec_function,
                    opcode,
                    &[bool_ty, result, left, right],
                );
                return Ok((result, quanta_ir::ShaderType::F32));
            }
        }
        Ok((left, ty))
    }

    pub(crate) fn parse_additive(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, ty) = self.parse_multiplicative(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('+') => {
                    *pos += 1;
                    let (right, _) = self.parse_multiplicative(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FADD,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                ShaderToken::Op('-') => {
                    *pos += 1;
                    let (right, _) = self.parse_multiplicative(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FSUB,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                _ => break,
            }
        }
        Ok((left, ty))
    }

    /// Matrix-vector multiplication detection
    pub(crate) fn parse_multiplicative(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, mut left_ty) = self.parse_unary(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('*') => {
                    *pos += 1;
                    let (right, right_ty) = self.parse_unary(tokens, pos, params, locals)?;
                    let result = self.alloc_id();

                    let is_left_mat = matches!(
                        left_ty,
                        quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
                    );
                    let is_right_vec = matches!(
                        right_ty,
                        quanta_ir::ShaderType::Vec4 | quanta_ir::ShaderType::Vec3
                    );

                    if is_left_mat && is_right_vec {
                        let result_ty_id = self.shader_type_id(right_ty);
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_MATRIX_TIMES_VECTOR,
                            &[result_ty_id, result, left, right],
                        );
                        left = result;
                        left_ty = right_ty;
                    } else {
                        let ty_id = self.shader_type_id(left_ty);
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_FMUL,
                            &[ty_id, result, left, right],
                        );
                        left = result;
                    }
                }
                ShaderToken::Op('/') => {
                    *pos += 1;
                    let (right, _) = self.parse_unary(tokens, pos, params, locals)?;
                    let result = self.alloc_id();
                    let ty_id = self.shader_type_id(left_ty);
                    Self::emit_op(
                        &mut self.sec_function,
                        OP_FDIV,
                        &[ty_id, result, left, right],
                    );
                    left = result;
                }
                _ => break,
            }
        }
        Ok((left, left_ty))
    }

    pub(crate) fn parse_unary(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Op('-') {
            *pos += 1;
            let (val, ty) = self.parse_unary(tokens, pos, params, locals)?;
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(ty);
            Self::emit_op(&mut self.sec_function, OP_F_NEGATE, &[ty_id, result, val]);
            return Ok((result, ty));
        }
        self.parse_atom(tokens, pos, params, locals)
    }
}

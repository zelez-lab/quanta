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
        let tokens = tokenize_shader_expr(src);
        let mut pos = 0;
        let mut locals: Vec<(String, u32, quanta_ir::ShaderType)> = Vec::new();
        match self.parse_statements(&tokens, &mut pos, params, &mut locals)? {
            Some(v) => Ok(v),
            None => Err("shader body has no trailing result expression".to_string()),
        }
    }

    /// Statement walker: `let [mut]` bindings, assignments to locals,
    /// statement-`if`/`else` (locals assigned in branches are merged
    /// with OpPhi — SSA construction, no Function-storage variables),
    /// and a trailing result expression.
    fn parse_statements(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<Option<(u32, quanta_ir::ShaderType)>, String> {
        let mut last_value: Option<(u32, quanta_ir::ShaderType)> = None;
        while *pos < tokens.len() {
            // `let [mut] name = expr ;`
            if tokens[*pos] == ShaderToken::Ident("let".to_string()) {
                *pos += 1;
                if tokens.get(*pos) == Some(&ShaderToken::Ident("mut".to_string())) {
                    *pos += 1;
                }
                let name = match tokens.get(*pos) {
                    Some(ShaderToken::Ident(n)) => n.clone(),
                    _ => return Err("expected identifier after `let`".to_string()),
                };
                *pos += 1;
                if tokens.get(*pos) != Some(&ShaderToken::Eq) {
                    return Err("expected `=` in let binding".to_string());
                }
                *pos += 1;
                let (id, ty) = self.parse_conditional(tokens, pos, params, locals)?;
                if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                    *pos += 1;
                }
                locals.push((name, id, ty));
                last_value = None;
                continue;
            }
            // `name = expr ;` — assignment to an existing local
            if let Some(ShaderToken::Ident(name)) = tokens.get(*pos)
                && tokens.get(*pos + 1) == Some(&ShaderToken::Eq)
            {
                if !locals.iter().any(|(n, _, _)| n == name) {
                    return Err(format!("assignment to unknown local `{name}`"));
                }
                let name = name.clone();
                *pos += 2;
                let (id, ty) = self.parse_conditional(tokens, pos, params, locals)?;
                if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                    *pos += 1;
                }
                if let Some(slot) = locals.iter_mut().find(|(n, _, _)| *n == name) {
                    slot.1 = id;
                    slot.2 = ty;
                }
                last_value = None;
                continue;
            }
            // statement-level `if` (may still yield a value when both
            // branches end in a trailing expression)
            if tokens.get(*pos) == Some(&ShaderToken::Ident("if".to_string())) {
                last_value = self.parse_if_statement(tokens, pos, params, locals)?;
                if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                    *pos += 1;
                    last_value = None;
                }
                continue;
            }
            // plain expression: trailing result, or discarded if `;`
            let v = self.parse_conditional(tokens, pos, params, locals)?;
            if tokens.get(*pos) == Some(&ShaderToken::Semi) {
                *pos += 1;
                last_value = None;
            } else {
                last_value = Some(v);
            }
        }
        Ok(last_value)
    }

    /// Consume a `{ … }` group starting at `pos`, returning the inner
    /// token slice and advancing past the matching close brace.
    fn take_braced(tokens: &[ShaderToken], pos: &mut usize) -> Result<Vec<ShaderToken>, String> {
        if tokens.get(*pos) != Some(&ShaderToken::BraceOpen) {
            return Err("expected `{`".to_string());
        }
        *pos += 1;
        let start = *pos;
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
        let inner = tokens[start..*pos].to_vec();
        if *pos < tokens.len() {
            *pos += 1; // matching `}`
        }
        Ok(inner)
    }

    /// `if cond { … } [else { … }]` at statement level. Branches run
    /// the statement walker over CLONED locals; at the merge block
    /// every local the branches diverged on gets an OpPhi. When both
    /// branches end in a trailing expression of the same type, the
    /// `if` itself yields a phi-merged value.
    fn parse_if_statement(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<Option<(u32, quanta_ir::ShaderType)>, String> {
        *pos += 1; // `if`
        let (cond, _) = self.parse_comparison(tokens, pos, params, locals)?;
        let then_tokens = Self::take_braced(tokens, pos)?;
        let else_tokens = if tokens.get(*pos) == Some(&ShaderToken::Ident("else".to_string())) {
            *pos += 1;
            Some(Self::take_braced(tokens, pos)?)
        } else {
            None
        };

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

        let base_len = locals.len();

        // Then branch
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
        self.current_block = then_label;
        let mut then_locals = locals.clone();
        let mut tp = 0;
        let then_val = self.parse_statements(&then_tokens, &mut tp, params, &mut then_locals)?;
        let then_end = self.current_block;
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

        // Else branch (empty when absent — just a jump to the merge)
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
        self.current_block = else_label;
        let mut else_locals = locals.clone();
        let else_val = if let Some(et) = &else_tokens {
            let mut ep = 0;
            self.parse_statements(et, &mut ep, params, &mut else_locals)?
        } else {
            None
        };
        let else_end = self.current_block;
        Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

        // Merge block: phi every diverged local
        Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
        self.current_block = merge_label;
        for i in 0..base_len {
            let (ref name, t_id, t_ty) = then_locals[i];
            let (_, e_id, e_ty) = else_locals[i];
            if t_id == e_id {
                locals[i].1 = t_id;
                continue;
            }
            if t_ty != e_ty {
                return Err(format!(
                    "local `{name}` assigned different types in if/else branches"
                ));
            }
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(t_ty);
            Self::emit_op(
                &mut self.sec_function,
                OP_PHI,
                &[ty_id, result, t_id, then_end, e_id, else_end],
            );
            locals[i].1 = result;
            locals[i].2 = t_ty;
        }

        // Value form: both branches produced a trailing expression
        match (then_val, else_val) {
            (Some((t_id, t_ty)), Some((e_id, e_ty))) if t_ty == e_ty => {
                let result = self.alloc_id();
                let ty_id = self.shader_type_id(t_ty);
                Self::emit_op(
                    &mut self.sec_function,
                    OP_PHI,
                    &[ty_id, result, t_id, then_end, e_id, else_end],
                );
                Ok(Some((result, t_ty)))
            }
            _ => Ok(None),
        }
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

    /// A branch of an if-EXPRESSION walks the statement grammar over a
    /// CLONE of the caller's locals, so an assignment that rebinds one of
    /// those outer locals would mutate only the clone and silently vanish
    /// at the merge — while the MSL emitter honors the write. Until the
    /// expression-if merge phis mutated outer locals the way the
    /// statement-level `if` already does, reject the shape instead of
    /// miscompiling it. New bindings (`let`, shadowing) only APPEND to the
    /// clone, so a changed id inside the original prefix is exactly an
    /// outer mutation — direct, or via a nested statement-`if`'s phi.
    fn reject_outer_mutation(
        original: &[(String, u32, quanta_ir::ShaderType)],
        walked: &[(String, u32, quanta_ir::ShaderType)],
    ) -> Result<(), String> {
        for (orig, after) in original.iter().zip(walked.iter()) {
            if orig.1 != after.1 {
                return Err(format!(
                    "assignment to outer local `{}` inside an if-expression branch \
                     is not supported; use a statement-level `if`",
                    orig.0
                ));
            }
        }
        Ok(())
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

            // Then block. A branch body is a statement block: zero or more
            // branch-local `let`/assignments followed by a mandatory tail
            // expression (`{ let nx = …; expr }`), so it runs through the
            // statement walker over CLONED locals — the branch-local bindings
            // stay scoped to the branch and never leak outward.
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
            self.current_block = then_label;
            let mut then_locals = locals.to_vec();
            let mut then_pos = 0;
            let (then_id, then_ty) = self
                .parse_statements(&then_tokens, &mut then_pos, params, &mut then_locals)?
                .ok_or_else(|| "if-expression then-branch has no result value".to_string())?;
            Self::reject_outer_mutation(locals, &then_locals)?;
            // A nested if inside the branch moves us to its merge
            // block — the phi below must name the ACTUAL predecessor.
            let then_end = self.current_block;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Else block (same block grammar as the then-branch).
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
            self.current_block = else_label;
            let mut else_locals = locals.to_vec();
            let mut else_pos = 0;
            let (else_id, _) = self
                .parse_statements(&else_tokens, &mut else_pos, params, &mut else_locals)?
                .ok_or_else(|| "if-expression else-branch has no result value".to_string())?;
            Self::reject_outer_mutation(locals, &else_locals)?;
            let else_end = self.current_block;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Merge block with OpPhi
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
            self.current_block = merge_label;
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(then_ty);
            Self::emit_op(
                &mut self.sec_function,
                OP_PHI,
                &[ty_id, result, then_id, then_end, else_id, else_end],
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
        // Unary deref: uniform params are `&T` in the source, so
        // bodies write `*viewport` / `(*viewport).x` — value semantics
        // here make it a no-op.
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Op('*') {
            *pos += 1;
            return self.parse_unary(tokens, pos, params, locals);
        }
        self.parse_atom(tokens, pos, params, locals)
    }
}

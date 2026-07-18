//! Shader body expression parser — recursive descent into SPIR-V.
//!
//! Parses tokenized Rust shader body into SPIR-V instructions.
//! Supports: Vec constructors, field access, arithmetic, float literals,
//! let bindings, math functions (GLSL.std.450), matrix-vector multiply,
//! if/else, comparisons, uniform parameter access via storage buffers, and
//! `&[T]` slice indexing (`name[index]`).
//!
//! `locals` threads as `&mut Vec` through the whole descent so that BOTH the
//! statement-level `if` and the expression-`if` can rebind outer locals at
//! their merge with `OpPhi` (SSA construction) — the two forms are now
//! symmetric, and neither silently drops a branch assignment.

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

    /// Phi-merge the locals of two branches back into the caller's `locals`.
    ///
    /// For each original-prefix slot (`0..base_len`) whose id changed in EITHER
    /// branch, emit an `OpPhi` over (then-value @ `then_end`, else-value @
    /// `else_end`) — a slot changed in only one branch phis the changed value
    /// against the still-original id from the other. A slot whose type diverged
    /// across the branches is an error. Assumes the merge block is current.
    /// Shared by the statement-level `if` and the expression-`if`.
    fn phi_merge_locals(
        &mut self,
        locals: &mut [(String, u32, quanta_ir::ShaderType)],
        then_locals: &[(String, u32, quanta_ir::ShaderType)],
        else_locals: &[(String, u32, quanta_ir::ShaderType)],
        base_len: usize,
        then_end: u32,
        else_end: u32,
    ) -> Result<(), String> {
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
        Ok(())
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
        self.phi_merge_locals(
            locals,
            &then_locals,
            &else_locals,
            base_len,
            then_end,
            else_end,
        )?;

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

    /// An if-EXPRESSION. Its branches run the statement grammar over CLONED
    /// locals; at the merge, every outer local a branch rebound is phi-merged
    /// back into the caller's `locals` — exactly like the statement-level `if`
    /// — and the branch VALUES are phi-merged into the expression's result.
    /// Branch-local `let`s only APPEND to the clone and stay scoped to the
    /// branch.
    pub(crate) fn parse_conditional(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
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

            let base_len = locals.len();

            // Then block. A branch body is a statement block: zero or more
            // branch-local `let`/assignments followed by a mandatory tail
            // expression (`{ let nx = …; expr }`), so it runs through the
            // statement walker over CLONED locals — the branch-local bindings
            // stay scoped to the branch, and an assignment to an OUTER local is
            // phi-merged back below (parity with the statement-level `if`).
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[then_label]);
            self.current_block = then_label;
            let mut then_locals = locals.clone();
            let mut then_pos = 0;
            let (then_id, then_ty) = self
                .parse_statements(&then_tokens, &mut then_pos, params, &mut then_locals)?
                .ok_or_else(|| "if-expression then-branch has no result value".to_string())?;
            // A nested if inside the branch moves us to its merge
            // block — the phi below must name the ACTUAL predecessor.
            let then_end = self.current_block;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Else block (same block grammar as the then-branch).
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[else_label]);
            self.current_block = else_label;
            let mut else_locals = locals.clone();
            let mut else_pos = 0;
            let (else_id, _) = self
                .parse_statements(&else_tokens, &mut else_pos, params, &mut else_locals)?
                .ok_or_else(|| "if-expression else-branch has no result value".to_string())?;
            let else_end = self.current_block;
            Self::emit_op(&mut self.sec_function, OP_BRANCH, &[merge_label]);

            // Merge block: phi mutated outer locals (like the statement-`if`),
            // then phi the branch VALUES into the expression's result.
            Self::emit_op(&mut self.sec_function, OP_LABEL, &[merge_label]);
            self.current_block = merge_label;
            self.phi_merge_locals(
                locals,
                &then_locals,
                &else_locals,
                base_len,
                then_end,
                else_end,
            )?;
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

    /// Convert a scalar value to the target scalar family when exactly one of
    /// {`F32`, `U32`} meets the other; every other (type, target) pair passes
    /// through untouched. This is how a BARE integer literal — tokenized as
    /// f32, see `ShaderToken::UInt` — participates in u32 comparisons
    /// (`shape_type == 3` → `OpConvertFToU %float_3` → `OpIEqual`), and how a
    /// u32 value joins float arithmetic (`OpConvertUToF`). An f32→u32
    /// conversion truncates toward zero; negative values are undefined per
    /// SPIR-V, which only integer-literal comparisons ever exercise here.
    pub(crate) fn coerce_scalar(
        &mut self,
        id: u32,
        ty: quanta_ir::ShaderType,
        target: quanta_ir::ShaderType,
    ) -> (u32, quanta_ir::ShaderType) {
        use quanta_ir::ShaderType::{F32, U32};
        let (opcode, target_ty_id) = match (ty, target) {
            (F32, U32) => (OP_CONVERT_F_TO_U, self.ensure_type_u32()),
            (U32, F32) => (OP_CONVERT_U_TO_F, self.ensure_type_f32()),
            _ => return (id, ty),
        };
        let result = self.alloc_id();
        Self::emit_op(&mut self.sec_function, opcode, &[target_ty_id, result, id]);
        (result, target)
    }

    pub(crate) fn parse_comparison(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (left, ty) = self.parse_additive(tokens, pos, params, locals)?;
        if *pos < tokens.len() {
            let cmp_op = match &tokens[*pos] {
                ShaderToken::Cmp(c) => Some(*c),
                _ => None,
            };
            if let Some(op) = cmp_op {
                *pos += 1;
                let (right, right_ty) = self.parse_additive(tokens, pos, params, locals)?;
                // A comparison with a u32 on EITHER side is an integer
                // comparison: the other side coerces to u32 (a bare literal
                // RHS arrives f32-typed) and the UNSIGNED opcode family is
                // used — comparing a uint with the float ops is invalid
                // SPIR-V. Pure float comparisons keep the FOrd ops.
                let unsigned =
                    ty == quanta_ir::ShaderType::U32 || right_ty == quanta_ir::ShaderType::U32;
                let (left, right) = if unsigned {
                    let (l, _) = self.coerce_scalar(left, ty, quanta_ir::ShaderType::U32);
                    let (r, _) = self.coerce_scalar(right, right_ty, quanta_ir::ShaderType::U32);
                    (l, r)
                } else {
                    (left, right)
                };
                let bool_ty = self.ensure_type_bool();
                let result = self.alloc_id();
                let opcode = if unsigned {
                    match op {
                        ShaderCmpOp::Lt => OP_ULESS_THAN,
                        ShaderCmpOp::Gt => OP_UGREATER_THAN,
                        ShaderCmpOp::Le => OP_ULESS_THAN_EQ,
                        ShaderCmpOp::Ge => OP_UGREATER_THAN_EQUAL,
                        ShaderCmpOp::Eq => OP_IEQUAL,
                        ShaderCmpOp::Ne => OP_INOT_EQUAL,
                    }
                } else {
                    match op {
                        ShaderCmpOp::Lt => OP_FORD_LESS_THAN,
                        ShaderCmpOp::Gt => OP_FORD_GREATER_THAN,
                        ShaderCmpOp::Le => OP_FORD_LESS_THAN_EQUAL,
                        ShaderCmpOp::Ge => OP_FORD_GREATER_THAN_EQUAL,
                        ShaderCmpOp::Eq => OP_FORD_EQUAL,
                        ShaderCmpOp::Ne => OP_FORD_NOT_EQUAL,
                    }
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

    /// Emit one scalar-aware arithmetic op. Two u32 operands use the integer
    /// opcode at u32 type; a MIXED u32/f32 pair widens the u32 side to f32
    /// (float wins — silently truncating `x * 0.5` through u32 would be
    /// worse) and uses the float opcode; everything else is the pre-existing
    /// float path keyed on the left type. Returns the result value and type.
    fn emit_arith_op(
        &mut self,
        left: (u32, quanta_ir::ShaderType),
        right: (u32, quanta_ir::ShaderType),
        int_op: u16,
        float_op: u16,
    ) -> (u32, quanta_ir::ShaderType) {
        use quanta_ir::ShaderType::U32;
        let result = self.alloc_id();
        if left.1 == U32 && right.1 == U32 {
            let ty_id = self.ensure_type_u32();
            Self::emit_op(
                &mut self.sec_function,
                int_op,
                &[ty_id, result, left.0, right.0],
            );
            return (result, U32);
        }
        let (l, lty) = self.coerce_scalar(left.0, left.1, quanta_ir::ShaderType::F32);
        let (r, _) = self.coerce_scalar(right.0, right.1, quanta_ir::ShaderType::F32);
        let ty_id = self.shader_type_id(lty);
        Self::emit_op(&mut self.sec_function, float_op, &[ty_id, result, l, r]);
        (result, lty)
    }

    pub(crate) fn parse_additive(
        &mut self,
        tokens: &[ShaderToken],
        pos: &mut usize,
        params: &[(String, u32, u32, quanta_ir::ShaderType)],
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, mut ty) = self.parse_multiplicative(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('+') => {
                    *pos += 1;
                    let right = self.parse_multiplicative(tokens, pos, params, locals)?;
                    (left, ty) = self.emit_arith_op((left, ty), right, OP_IADD, OP_FADD);
                }
                ShaderToken::Op('-') => {
                    *pos += 1;
                    let right = self.parse_multiplicative(tokens, pos, params, locals)?;
                    (left, ty) = self.emit_arith_op((left, ty), right, OP_ISUB, OP_FSUB);
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
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        let (mut left, mut left_ty) = self.parse_unary(tokens, pos, params, locals)?;
        while *pos < tokens.len() {
            match &tokens[*pos] {
                ShaderToken::Op('*') => {
                    *pos += 1;
                    let (right, right_ty) = self.parse_unary(tokens, pos, params, locals)?;

                    let is_left_mat = matches!(
                        left_ty,
                        quanta_ir::ShaderType::Mat4 | quanta_ir::ShaderType::Mat3
                    );
                    let is_right_vec = matches!(
                        right_ty,
                        quanta_ir::ShaderType::Vec4 | quanta_ir::ShaderType::Vec3
                    );

                    if is_left_mat && is_right_vec {
                        let result = self.alloc_id();
                        let result_ty_id = self.shader_type_id(right_ty);
                        Self::emit_op(
                            &mut self.sec_function,
                            OP_MATRIX_TIMES_VECTOR,
                            &[result_ty_id, result, left, right],
                        );
                        left = result;
                        left_ty = right_ty;
                    } else {
                        (left, left_ty) = self.emit_arith_op(
                            (left, left_ty),
                            (right, right_ty),
                            OP_IMUL,
                            OP_FMUL,
                        );
                    }
                }
                ShaderToken::Op('/') => {
                    *pos += 1;
                    let right = self.parse_unary(tokens, pos, params, locals)?;
                    // u32 / u32 is the UNSIGNED integer division; anything
                    // mixed widens to float division (emit_arith_op).
                    (left, left_ty) = self.emit_arith_op((left, left_ty), right, OP_UDIV, OP_FDIV);
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
        locals: &mut Vec<(String, u32, quanta_ir::ShaderType)>,
    ) -> Result<(u32, quanta_ir::ShaderType), String> {
        if *pos < tokens.len() && tokens[*pos] == ShaderToken::Op('-') {
            *pos += 1;
            let (val, ty) = self.parse_unary(tokens, pos, params, locals)?;
            let result = self.alloc_id();
            let ty_id = self.shader_type_id(ty);
            // OpFNegate on an integer operand is invalid; a u32 negates with
            // the integer op (two's-complement wrap, like Rust `wrapping_neg`).
            let opcode = if ty == quanta_ir::ShaderType::U32 {
                OP_S_NEGATE
            } else {
                OP_F_NEGATE
            };
            Self::emit_op(&mut self.sec_function, opcode, &[ty_id, result, val]);
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

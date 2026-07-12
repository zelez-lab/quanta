//! SpvEmitter core — struct, constructor, ID allocation, instruction encoding,
//! register management, name/decoration helpers, finalize.

use std::collections::HashMap;

use super::constants::*;

/// SPIR-V requires instructions in a strict section order. We build each
/// section into its own word buffer, then concatenate at the end.
pub(crate) struct SpvEmitter {
    pub(crate) next_id: u32,

    // Section buffers (in required order)
    pub(crate) sec_capability: Vec<u32>,
    pub(crate) sec_extension: Vec<u32>,
    pub(crate) sec_ext_inst_import: Vec<u32>,
    pub(crate) sec_memory_model: Vec<u32>,
    pub(crate) sec_entry_point: Vec<u32>,
    pub(crate) sec_execution_mode: Vec<u32>,
    pub(crate) sec_debug: Vec<u32>,
    pub(crate) sec_annotation: Vec<u32>,
    pub(crate) sec_type_const: Vec<u32>,
    pub(crate) sec_global_var: Vec<u32>,
    pub(crate) sec_function: Vec<u32>,

    // Type caches to avoid duplicates
    pub(crate) type_void: Option<u32>,
    pub(crate) type_bool: Option<u32>,
    pub(crate) type_u32: Option<u32>,
    pub(crate) type_u16: Option<u32>,
    pub(crate) type_u8: Option<u32>,
    pub(crate) type_i32: Option<u32>,
    pub(crate) type_u64: Option<u32>,
    pub(crate) type_i64: Option<u32>,
    pub(crate) type_f16: Option<u32>,
    pub(crate) type_f32: Option<u32>,
    pub(crate) type_f64: Option<u32>,
    pub(crate) type_v3uint: Option<u32>,
    pub(crate) type_cache: HashMap<String, u32>,

    // Constant cache: key = "type_id:bit_pattern"
    pub(crate) const_cache: HashMap<String, u32>,

    // GLSL.std.450 extended instruction set ID
    pub(crate) glsl_ext_id: Option<u32>,

    // Stack of loop merge labels for Break support
    pub(crate) loop_merge_stack: Vec<u32>,

    // Register → SPIR-V ID mapping (function-scoped variables)
    pub(crate) reg_ids: HashMap<u32, u32>,
    // Register → type ID (so we know what type a register holds)
    pub(crate) reg_types: HashMap<u32, u32>,

    // Mutable registers demoted to `Function`-storage OpVariables:
    // reg → (variable_id, element_type_id). The KernelOp contract allows a
    // register to be written in a Branch arm / Loop body and read after the
    // merge (mutable-register semantics); pure SSA renames can't express
    // that, so those registers go through OpLoad/OpStore on a function-local
    // variable instead — mirroring the LLVM backend's `reg_slots` allocas.
    // Detected up front by `reg_mutability::collect_mutable_regs`;
    // single-def temporaries stay SSA renames in `reg_ids`.
    pub(crate) demoted_regs: HashMap<u32, (u32, u32)>,

    // Field slot → (variable_id, element_type_id, is_writable)
    pub(crate) field_vars: HashMap<u32, (u32, u32, bool)>,

    // Texture slot → (variable_id, loaded_type_id). For sampled (`&Texture2D`)
    // slots the type is OpTypeSampledImage; for storage (`&mut Texture2D`)
    // slots it is the plain OpTypeImage.
    pub(crate) texture_samplers: HashMap<u32, (u32, u32)>,
    // Sampled slots only: slot → underlying OpTypeImage id (OpImageFetch takes
    // a plain image, so a load unwraps the sampled image with OpImage).
    pub(crate) texture_image_types: HashMap<u32, u32>,
    // Slots declared `&mut Texture2D` — read_write storage images. A load
    // against such a slot emits OpImageRead rather than OpImageFetch.
    pub(crate) texture_storage_slots: std::collections::HashSet<u32>,

    // Track total push constant bytes needed
    pub(crate) push_constant_size: u32,
    // Which field slots are push constants (PushConstant storage class)
    pub(crate) push_constant_slots: std::collections::HashSet<u32>,
    // Slot → member index inside the single push-constant Block (Vulkan
    // allows only one push-constant interface per entry point).
    pub(crate) push_constant_member: HashMap<u32, u32>,

    // Shared memory: id → (variable_id, element_type_id)
    pub(crate) shared_vars: HashMap<u32, (u32, u32)>,

    // Track which types already have ArrayStride decoration applied
    pub(crate) decorated_stride: std::collections::HashSet<u32>,
    // Track which struct types already have Block decoration
    pub(crate) decorated_block: std::collections::HashSet<u32>,

    // Device function name → (function_id, return_type_id, param_type_ids)
    pub(crate) device_fn_ids: HashMap<String, (u32, u32, Vec<u32>)>,
    // Buffer for device function bodies (emitted after the main function)
    pub(crate) sec_device_fns: Vec<u32>,

    // Register → known constant value (when the register was defined
    // by `KernelOp::Const { value: U32/U64/I32/I64 }`). Used by the
    // Loop emitter to decide whether to apply LOOP_CONTROL_UNROLL for
    // small known iteration counts. Keyed by `Reg.0`, value is the
    // const sign-extended to i64. Float / bool consts are deliberately
    // NOT tracked here — only integers feed Loop.count.
    pub(crate) reg_const_int: HashMap<u32, i64>,

    // Workgroup size x of the kernel being emitted (set by
    // emit_kernel). Feeds the folded-dispatch linearization constant
    // in QuarkId (`FOLD_ROW_GROUPS * wg_x`).
    pub(crate) wg_x: u32,
}

impl SpvEmitter {
    pub(crate) fn new() -> Self {
        Self {
            next_id: 1,
            sec_capability: Vec::new(),
            sec_extension: Vec::new(),
            sec_ext_inst_import: Vec::new(),
            sec_memory_model: Vec::new(),
            sec_entry_point: Vec::new(),
            sec_execution_mode: Vec::new(),
            sec_debug: Vec::new(),
            sec_annotation: Vec::new(),
            sec_type_const: Vec::new(),
            sec_global_var: Vec::new(),
            sec_function: Vec::new(),
            type_void: None,
            type_bool: None,
            type_u32: None,
            type_u16: None,
            type_u8: None,
            type_i32: None,
            type_u64: None,
            type_i64: None,
            type_f16: None,
            type_f32: None,
            type_f64: None,
            type_v3uint: None,
            type_cache: HashMap::new(),
            const_cache: HashMap::new(),
            glsl_ext_id: None,
            loop_merge_stack: Vec::new(),
            reg_ids: HashMap::new(),
            reg_types: HashMap::new(),
            demoted_regs: HashMap::new(),
            field_vars: HashMap::new(),
            texture_samplers: HashMap::new(),
            texture_image_types: HashMap::new(),
            texture_storage_slots: std::collections::HashSet::new(),
            push_constant_size: 0,
            push_constant_slots: std::collections::HashSet::new(),
            push_constant_member: HashMap::new(),
            shared_vars: HashMap::new(),
            decorated_stride: std::collections::HashSet::new(),
            decorated_block: std::collections::HashSet::new(),
            device_fn_ids: HashMap::new(),
            sec_device_fns: Vec::new(),
            reg_const_int: HashMap::new(),
            wg_x: 1,
        }
    }

    // ── ID allocator ────────────────────────────────────────────────────────

    pub(crate) fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Instruction encoding ────────────────────────────────────────────────

    /// Encode an instruction into a section buffer.
    /// Format: (word_count << 16) | opcode, then operand words.
    pub(crate) fn emit_op(section: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u16;
        section.push(((word_count as u32) << 16) | (opcode as u32));
        section.extend_from_slice(operands);
    }

    /// Encode a string as SPIR-V literal words (null-terminated, padded to
    /// word boundary).
    pub(crate) fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        // +1 for null terminator, round up to multiple of 4
        let total_bytes = bytes.len() + 1;
        let word_count = total_bytes.div_ceil(4);
        let mut words = vec![0u32; word_count];
        for (i, &b) in bytes.iter().enumerate() {
            let word_idx = i / 4;
            let byte_idx = i % 4;
            words[word_idx] |= (b as u32) << (byte_idx * 8);
        }
        // Null terminator is already there since we initialized to 0
        words
    }

    // ── Register management ─────────────────────────────────────────────────

    pub(crate) fn reg_value_id(&mut self, reg: crate::Reg) -> Result<u32, String> {
        // Demoted (mutable) register: read its current value from the
        // function-local variable. Loads from Function storage are
        // dominance-valid anywhere in the function.
        if let Some(&(var_id, elem_ty)) = self.demoted_regs.get(&reg.0) {
            let out = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_LOAD, &[elem_ty, out, var_id]);
            return Ok(out);
        }
        self.reg_ids
            .get(&reg.0)
            .copied()
            .ok_or_else(|| format!("register r{} used before definition", reg.0))
    }

    /// Look up a register's known integer constant value, if any. Returns
    /// `Some(v)` only when the register was defined by `KernelOp::Const`
    /// carrying a U32/U64/I32/I64 payload. Used by the Loop emitter to
    /// pick LOOP_CONTROL_UNROLL for short known iteration counts.
    pub(crate) fn lookup_reg_const_int(&self, reg: crate::Reg) -> Option<i64> {
        self.reg_const_int.get(&reg.0).copied()
    }

    pub(crate) fn set_reg(&mut self, reg: crate::Reg, id: u32, type_id: u32) {
        // Demoted (mutable) register: writes become OpStore into its
        // function-local variable (coerced to the slot's element type).
        // `reg_ids` is deliberately NOT updated — every later read loads
        // the variable, so Branch/Loop need no reg-id reconciliation.
        if let Some(&(var_id, elem_ty)) = self.demoted_regs.get(&reg.0) {
            let val = self.coerce_to(id, type_id, elem_ty);
            Self::emit_op(&mut self.sec_function, OP_STORE, &[var_id, val]);
            self.reg_types.insert(reg.0, elem_ty);
            return;
        }
        self.reg_ids.insert(reg.0, id);
        self.reg_types.insert(reg.0, type_id);
    }

    /// Declare `Function`-storage OpVariables for the demoted (mutable)
    /// registers of a body. Must be called right after the function's entry
    /// `OpLabel` — SPIR-V requires all Function-storage variables in the
    /// first block of the function.
    pub(crate) fn declare_demoted_regs(
        &mut self,
        demoted: &std::collections::BTreeMap<u32, crate::ScalarType>,
    ) {
        for (&reg, &sty) in demoted {
            let elem_ty = self.scalar_type_id(sty);
            let ptr_ty = self.ensure_type_pointer(STORAGE_CLASS_FUNCTION, elem_ty);
            let var_id = self.alloc_id();
            Self::emit_op(
                &mut self.sec_function,
                OP_VARIABLE,
                &[ptr_ty, var_id, STORAGE_CLASS_FUNCTION],
            );
            self.emit_name(var_id, &format!("r{}_slot", reg));
            self.demoted_regs.insert(reg, (var_id, elem_ty));
            self.reg_types.insert(reg, elem_ty);
        }
    }

    /// Typed `(zero, one)` constants for a known *float* type id, or
    /// `None` if the id isn't one of the cached float types. Same
    /// no-new-types discipline as `int_zero_one_of`.
    fn float_zero_one_of(&mut self, ty: u32) -> Option<(u32, u32)> {
        if self.type_f32 == Some(ty) {
            Some((self.emit_constant_f32(0.0), self.emit_constant_f32(1.0)))
        } else if self.type_f64 == Some(ty) {
            Some((self.emit_constant_f64(0.0), self.emit_constant_f64(1.0)))
        } else if self.type_f16 == Some(ty) {
            Some((
                self.emit_constant_f16(0x0000),
                self.emit_constant_f16(0x3C00),
            ))
        } else {
            None
        }
    }

    /// Typed `(zero, one)` constants for a known *integer* type id, or
    /// `None` if the id isn't one of the cached int types. Reads the type
    /// caches without materializing new types (calling `ensure_type_u64`
    /// here would inject an unused 64-bit type — and its capability — into
    /// every module).
    fn int_zero_one_of(&mut self, ty: u32) -> Option<(u32, u32)> {
        if self.type_u32 == Some(ty) {
            Some((self.emit_constant_u32(0), self.emit_constant_u32(1)))
        } else if self.type_i32 == Some(ty) {
            Some((self.emit_constant_i32(0), self.emit_constant_i32(1)))
        } else if self.type_u64 == Some(ty) {
            Some((self.emit_constant_u64(0), self.emit_constant_u64(1)))
        } else if self.type_i64 == Some(ty) {
            Some((self.emit_constant_i64(0), self.emit_constant_i64(1)))
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn reg_type_id(&self, reg: crate::Reg) -> Result<u32, String> {
        self.reg_types
            .get(&reg.0)
            .copied()
            .ok_or_else(|| format!("register r{} type unknown", reg.0))
    }

    /// Coerce a value from `from_ty` to `to_ty`, inserting an `OpBitcast` if they
    /// differ. SPIR-V is strictly typed, so a value produced as `%int` (signed)
    /// cannot feed an op — or a phi — declared `%uint` (unsigned), even though
    /// the bits are identical. This bridges that gap: same-size int↔int (and the
    /// reverse) is a free reinterpret. When the types already match it is a
    /// no-op. Returns the (possibly new) value id.
    pub(crate) fn coerce_to(&mut self, val: u32, from_ty: u32, to_ty: u32) -> u32 {
        if from_ty == to_ty {
            return val;
        }
        // `%bool` has no bit representation in SPIR-V, so OpBitcast to or
        // from it is invalid (and the OpConvert* family doesn't take %bool
        // either). Bridge with a semantic conversion instead: numeric →
        // bool is a truthiness test (`val != 0`), bool → numeric
        // materializes 0/1 with OpSelect — for float targets too, so a
        // mask never round-trips through a driver's native bool encoding
        // (V3D uses all-ones, which read back as 2^32 in f32).
        if self.type_bool == Some(to_ty) {
            if let Some((zero, _)) = self.int_zero_one_of(from_ty) {
                let out = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_INOT_EQUAL,
                    &[to_ty, out, val, zero],
                );
                return out;
            }
            if let Some((zero, _)) = self.float_zero_one_of(from_ty) {
                let out = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_FORD_NOT_EQUAL,
                    &[to_ty, out, val, zero],
                );
                return out;
            }
        }
        if self.type_bool == Some(from_ty) {
            let zero_one = self
                .int_zero_one_of(to_ty)
                .or_else(|| self.float_zero_one_of(to_ty));
            if let Some((zero, one)) = zero_one {
                let out = self.alloc_id();
                Self::emit_op(
                    &mut self.sec_function,
                    OP_SELECT,
                    &[to_ty, out, val, one, zero],
                );
                return out;
            }
        }
        // Int↔int across widths: OpBitcast requires equal total bit width,
        // so bridge with OpUConvert (zero-extend / truncate) instead. This
        // catches operands whose tracked width is stale relative to the op
        // that consumes them (the wasm route reuses registers freely).
        let is_64 = |s: &Self, t: u32| s.type_u64 == Some(t) || s.type_i64 == Some(t);
        let is_32 = |s: &Self, t: u32| s.type_u32 == Some(t) || s.type_i32 == Some(t);
        if (is_64(self, from_ty) && is_32(self, to_ty))
            || (is_32(self, from_ty) && is_64(self, to_ty))
        {
            let out = self.alloc_id();
            Self::emit_op(&mut self.sec_function, OP_U_CONVERT, &[to_ty, out, val]);
            return out;
        }
        let out = self.alloc_id();
        Self::emit_op(&mut self.sec_function, OP_BITCAST, &[to_ty, out, val]);
        out
    }

    // ── Name / decoration helpers ───────────────────────────────────────────

    pub(crate) fn emit_name(&mut self, id: u32, name: &str) {
        let name_words = Self::string_words(name);
        let mut ops = vec![id];
        ops.extend_from_slice(&name_words);
        Self::emit_op(&mut self.sec_debug, OP_NAME, &ops);
    }

    pub(crate) fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(operands);
        Self::emit_op(&mut self.sec_annotation, OP_DECORATE, &ops);
    }

    pub(crate) fn member_decorate(
        &mut self,
        struct_type: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let mut ops = vec![struct_type, member, decoration];
        ops.extend_from_slice(operands);
        Self::emit_op(&mut self.sec_annotation, OP_MEMBER_DECORATE, &ops);
    }

    // ── Push constant helpers ───────────────────────────────────────────────

    pub(crate) fn is_push_constant_field(&self, slot: u32) -> bool {
        self.push_constant_slots.contains(&slot)
    }

    // ── Finalize: concatenate sections and emit header ──────────────────────

    pub(crate) fn finalize(self) -> Vec<u8> {
        let mut words = Vec::new();

        // All sections concatenated (to compute bound = max ID).
        // Device functions are emitted after the main function.
        let all_sections: Vec<&[u32]> = vec![
            &self.sec_capability,
            &self.sec_extension,
            &self.sec_ext_inst_import,
            &self.sec_memory_model,
            &self.sec_entry_point,
            &self.sec_execution_mode,
            &self.sec_debug,
            &self.sec_annotation,
            &self.sec_type_const,
            &self.sec_global_var,
            &self.sec_device_fns,
            &self.sec_function,
        ];

        // Header
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_3);
        words.push(SPIRV_GENERATOR);
        words.push(self.next_id); // Bound (max ID + 1)
        words.push(SPIRV_SCHEMA);

        // Sections in order
        for section in all_sections {
            words.extend_from_slice(section);
        }

        // Convert to bytes (little-endian)
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }
}

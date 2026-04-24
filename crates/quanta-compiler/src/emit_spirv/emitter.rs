//! SpvEmitter core — struct, constructor, ID allocation, instruction encoding,
//! register management, name/decoration helpers.

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
    pub(crate) type_i32: Option<u32>,
    pub(crate) type_f32: Option<u32>,
    pub(crate) type_f64: Option<u32>,
    pub(crate) type_v3uint: Option<u32>,
    pub(crate) type_cache: HashMap<String, u32>,

    // Constant cache: key = "type_id:bit_pattern"
    pub(crate) const_cache: HashMap<String, u32>,

    // GLSL.std.450 extended instruction set ID
    pub(crate) glsl_ext_id: Option<u32>,

    // Texture sampler variables: slot → (var_id, type_id)
    pub(crate) texture_samplers: HashMap<u32, (u32, u32)>,

    // Stack of loop merge labels for Break support
    pub(crate) loop_merge_stack: Vec<u32>,

    // Register → SPIR-V ID mapping (function-scoped variables)
    pub(crate) reg_ids: HashMap<u32, u32>,
    pub(crate) reg_types: HashMap<u32, u32>,

    // Field slot → (variable_id, element_type_id, is_writable)
    pub(crate) field_vars: HashMap<u32, (u32, u32, bool)>,

    // Push constant tracking
    pub(crate) push_constant_size: u32,
    pub(crate) push_constant_slots: std::collections::HashSet<u32>,

    // Shared memory: id → (variable_id, element_type_id)
    pub(crate) shared_vars: HashMap<u32, (u32, u32)>,

    // Decoration tracking
    pub(crate) decorated_stride: std::collections::HashSet<u32>,
    pub(crate) decorated_block: std::collections::HashSet<u32>,

    // Device functions
    pub(crate) device_fn_ids: HashMap<String, (u32, u32, Vec<u32>)>,
    pub(crate) sec_device_fns: Vec<u32>,
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
            type_i32: None,
            type_f32: None,
            type_f64: None,
            type_v3uint: None,
            type_cache: HashMap::new(),
            const_cache: HashMap::new(),
            glsl_ext_id: None,
            texture_samplers: HashMap::new(),
            loop_merge_stack: Vec::new(),
            reg_ids: HashMap::new(),
            reg_types: HashMap::new(),
            field_vars: HashMap::new(),
            push_constant_size: 0,
            push_constant_slots: std::collections::HashSet::new(),
            shared_vars: HashMap::new(),
            decorated_stride: std::collections::HashSet::new(),
            decorated_block: std::collections::HashSet::new(),
            device_fn_ids: HashMap::new(),
            sec_device_fns: Vec::new(),
        }
    }

    // ── ID allocator ────────────────────────────────────────────────────────

    pub(crate) fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // ── Instruction encoding ────────────────────────────────────────────────

    pub(crate) fn emit_op(section: &mut Vec<u32>, opcode: u16, operands: &[u32]) {
        let word_count = (1 + operands.len()) as u16;
        section.push(((word_count as u32) << 16) | (opcode as u32));
        section.extend_from_slice(operands);
    }

    pub(crate) fn string_words(s: &str) -> Vec<u32> {
        let bytes = s.as_bytes();
        let total_bytes = bytes.len() + 1;
        let word_count = total_bytes.div_ceil(4);
        let mut words = vec![0u32; word_count];
        for (i, &b) in bytes.iter().enumerate() {
            let word_idx = i / 4;
            let byte_idx = i % 4;
            words[word_idx] |= (b as u32) << (byte_idx * 8);
        }
        words
    }

    // ── Register management ─────────────────────────────────────────────────

    pub(crate) fn reg_value_id(&self, reg: quanta_ir::Reg) -> Result<u32, String> {
        self.reg_ids
            .get(&reg.0)
            .copied()
            .ok_or_else(|| format!("register r{} used before definition", reg.0))
    }

    pub(crate) fn set_reg(&mut self, reg: quanta_ir::Reg, id: u32, type_id: u32) {
        self.reg_ids.insert(reg.0, id);
        self.reg_types.insert(reg.0, type_id);
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

        let mut words = vec![
            SPIRV_MAGIC,
            SPIRV_VERSION_1_3,
            SPIRV_GENERATOR,
            self.next_id,
            SPIRV_SCHEMA,
        ];

        for section in all_sections {
            words.extend_from_slice(section);
        }

        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        bytes
    }
}

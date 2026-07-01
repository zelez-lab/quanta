//! SPIR-V type and constant emission helpers.
//!
//! ensure_type_* methods create OpType instructions with deduplication.
//! emit_constant_* methods create OpConstant instructions with caching.

use quanta_ir::ScalarType;

use super::constants::*;
use super::emitter::SpvEmitter;

impl SpvEmitter {
    // ── Primitive types ──────────────────────────────────────────────────────

    pub(crate) fn ensure_type_void(&mut self) -> u32 {
        if let Some(id) = self.type_void {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_VOID, &[id]);
        self.type_void = Some(id);
        id
    }

    pub(crate) fn ensure_type_bool(&mut self) -> u32 {
        if let Some(id) = self.type_bool {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_BOOL, &[id]);
        self.type_bool = Some(id);
        id
    }

    pub(crate) fn ensure_type_u32(&mut self) -> u32 {
        if let Some(id) = self.type_u32 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 0]);
        self.type_u32 = Some(id);
        id
    }

    pub(crate) fn ensure_type_i32(&mut self) -> u32 {
        if let Some(id) = self.type_i32 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 1]);
        self.type_i32 = Some(id);
        id
    }

    pub(crate) fn ensure_type_f16(&mut self) -> u32 {
        if let Some(id) = self.type_f16 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FLOAT, &[id, 16]);
        // Declare Float16 capability when f16 types are used
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_FLOAT16],
        );
        self.type_f16 = Some(id);
        id
    }

    pub(crate) fn ensure_type_f32(&mut self) -> u32 {
        if let Some(id) = self.type_f32 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FLOAT, &[id, 32]);
        self.type_f32 = Some(id);
        id
    }

    pub(crate) fn ensure_type_f64(&mut self) -> u32 {
        if let Some(id) = self.type_f64 {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FLOAT, &[id, 64]);
        // Declare Float64 capability when f64 types are used — referencing
        // OpTypeFloat 64 without it is invalid SPIR-V (rejected by
        // spirv-val and by drivers at pipeline-creation time).
        Self::emit_op(
            &mut self.sec_capability,
            OP_CAPABILITY,
            &[CAPABILITY_FLOAT64],
        );
        self.type_f64 = Some(id);
        id
    }

    // ── Composite types ─────────────────────────────────────────────────────

    pub(crate) fn ensure_type_vector(&mut self, elem_type: u32, count: u32) -> u32 {
        let key = format!("vec_{}_{}", elem_type, count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_VECTOR,
            &[id, elem_type, count],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_matrix(&mut self, col_type: u32, count: u32) -> u32 {
        let key = format!("mat_{}_{}", col_type, count);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_MATRIX,
            &[id, col_type, count],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_v3uint(&mut self) -> u32 {
        if let Some(id) = self.type_v3uint {
            return id;
        }
        let uint = self.ensure_type_u32();
        let id = self.ensure_type_vector(uint, 3);
        self.type_v3uint = Some(id);
        id
    }

    pub(crate) fn ensure_type_runtime_array(&mut self, elem_type: u32) -> u32 {
        let key = format!("rtarr_{}", elem_type);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_RUNTIME_ARRAY,
            &[id, elem_type],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_array(&mut self, elem_type: u32, length_id: u32) -> u32 {
        let key = format!("arr_{}_{}", elem_type, length_id);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_ARRAY,
            &[id, elem_type, length_id],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_struct(&mut self, members: &[u32]) -> u32 {
        let key = format!(
            "struct_{}",
            members
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id];
        ops.extend_from_slice(members);
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_STRUCT, &ops);
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_pointer(&mut self, storage_class: u32, pointee: u32) -> u32 {
        let key = format!("ptr_{}_{}", storage_class, pointee);
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(
            &mut self.sec_type_const,
            OP_TYPE_POINTER,
            &[id, storage_class, pointee],
        );
        self.type_cache.insert(key, id);
        id
    }

    pub(crate) fn ensure_type_function(&mut self, return_type: u32, params: &[u32]) -> u32 {
        let key = format!(
            "fn_{}_{}",
            return_type,
            params
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );
        if let Some(&id) = self.type_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let mut ops = vec![id, return_type];
        ops.extend_from_slice(params);
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_FUNCTION, &ops);
        self.type_cache.insert(key, id);
        id
    }

    // ── Scalar type mapping ─────────────────────────────────────────────────

    /// The *signed* SPIR-V int type to bitcast into for a genuinely-signed op
    /// (SDiv/SRem/SAR) whose canonical operands are the unsigned form. This
    /// emitter models all ints in 32-bit width, so `%int` suffices.
    pub(crate) fn ensure_type_i32_for(&mut self, _ty: ScalarType) -> u32 {
        self.ensure_type_i32()
    }

    pub(crate) fn scalar_type_id(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F32 => self.ensure_type_f32(),
            ScalarType::F64 => self.ensure_type_f64(),
            // ALL 32-bit ints (signed and unsigned) share ONE canonical SSA
            // type: unsigned. Signedness is a property of the *op* (SDiv vs
            // UDiv), not the value — the few signed ops bitcast to `%int`
            // locally and back. This keeps every int SSA value one type so phis
            // and bitwise ops can never mismatch (invalid SPIR-V Vulkan rejects).
            ScalarType::U8
            | ScalarType::U16
            | ScalarType::U32
            | ScalarType::I8
            | ScalarType::I16
            | ScalarType::I32
            | ScalarType::I4 => self.ensure_type_u32(),
            ScalarType::U64 | ScalarType::I64 => self.ensure_type_u32(),
            ScalarType::F16 => self.ensure_type_f16(),
            // bf16/fp8 compute in f32 in the body (emulated path).
            ScalarType::BF16 | ScalarType::FP8E5M2 | ScalarType::FP8E4M3 => self.ensure_type_f32(),
            ScalarType::Bool => self.ensure_type_bool(),
        }
    }

    pub(crate) fn scalar_byte_size(ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F16 => 2,
            ScalarType::BF16 => 2,
            ScalarType::FP8E5M2 | ScalarType::FP8E4M3 => 1,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
            ScalarType::U8 | ScalarType::I8 => 1,
            ScalarType::U16 | ScalarType::I16 => 2,
            ScalarType::U32 | ScalarType::I32 => 4,
            ScalarType::U64 | ScalarType::I64 => 8,
            ScalarType::I4 => 1, // logical nibble; PackedU32 at I/O
            ScalarType::Bool => 4,
        }
    }

    // ── Shader type mapping (for vertex/fragment) ───────────────────────────

    pub(crate) fn shader_type_id(&mut self, ty: quanta_ir::ShaderType) -> u32 {
        let f32_ty = self.ensure_type_f32();
        match ty {
            quanta_ir::ShaderType::F32 => f32_ty,
            quanta_ir::ShaderType::Vec2 => self.ensure_type_vector(f32_ty, 2),
            quanta_ir::ShaderType::Vec3 => self.ensure_type_vector(f32_ty, 3),
            quanta_ir::ShaderType::Vec4 => self.ensure_type_vector(f32_ty, 4),
            quanta_ir::ShaderType::Mat4 => {
                let vec4_ty = self.ensure_type_vector(f32_ty, 4);
                self.ensure_type_matrix(vec4_ty, 4)
            }
            quanta_ir::ShaderType::Mat3 => {
                let vec3_ty = self.ensure_type_vector(f32_ty, 3);
                self.ensure_type_matrix(vec3_ty, 3)
            }
        }
    }

    pub(crate) fn shader_type_components(ty: quanta_ir::ShaderType) -> u32 {
        match ty {
            quanta_ir::ShaderType::F32 => 1,
            quanta_ir::ShaderType::Vec2 => 2,
            quanta_ir::ShaderType::Vec3 => 3,
            quanta_ir::ShaderType::Vec4 | quanta_ir::ShaderType::Mat4 => 4,
            quanta_ir::ShaderType::Mat3 => 3,
        }
    }

    // ── Constant emission ───────────────────────────────────────────────────

    pub(crate) fn emit_constant_u32(&mut self, val: u32) -> u32 {
        let ty = self.ensure_type_u32();
        let key = format!("{}:{}", ty, val);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, val]);
        self.const_cache.insert(key, id);
        id
    }

    pub(crate) fn emit_constant_i32(&mut self, val: i32) -> u32 {
        let ty = self.ensure_type_i32();
        let key = format!("{}:{}", ty, val as u32);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, val as u32]);
        self.const_cache.insert(key, id);
        id
    }

    #[allow(dead_code)]
    pub(crate) fn emit_constant_f16(&mut self, val: u16) -> u32 {
        let ty = self.ensure_type_f16();
        let key = format!("{}:{}", ty, val);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        // F16 constant is stored as a 32-bit word with the f16 value in the low 16 bits
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, val as u32]);
        self.const_cache.insert(key, id);
        id
    }

    pub(crate) fn emit_constant_f32(&mut self, val: f32) -> u32 {
        let ty = self.ensure_type_f32();
        let bits = val.to_bits();
        let key = format!("{}:{}", ty, bits);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, bits]);
        self.const_cache.insert(key, id);
        id
    }

    pub(crate) fn emit_constant_f64(&mut self, val: f64) -> u32 {
        let ty = self.ensure_type_f64();
        let bits = val.to_bits();
        let lo = bits as u32;
        let hi = (bits >> 32) as u32;
        let key = format!("{}:{}:{}", ty, lo, hi);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, lo, hi]);
        self.const_cache.insert(key, id);
        id
    }

    pub(crate) fn emit_constant_bool(&mut self, val: bool) -> u32 {
        let ty = self.ensure_type_bool();
        let key = format!("bool:{}", val);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        let opcode = if val {
            OP_CONSTANT_TRUE
        } else {
            OP_CONSTANT_FALSE
        };
        Self::emit_op(&mut self.sec_type_const, opcode, &[ty, id]);
        self.const_cache.insert(key, id);
        id
    }

    // ── GLSL extension ──────────────────────────────────────────────────────

    pub(crate) fn ensure_glsl_ext(&mut self) -> u32 {
        if let Some(id) = self.glsl_ext_id {
            return id;
        }
        let id = self.alloc_id();
        let name_words = Self::string_words("GLSL.std.450");
        let mut ops = vec![id];
        ops.extend_from_slice(&name_words);
        Self::emit_op(&mut self.sec_ext_inst_import, OP_EXT_INST_IMPORT, &ops);
        self.glsl_ext_id = Some(id);
        id
    }
}

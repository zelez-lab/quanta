//! SPIR-V type and constant emission helpers.
//!
//! ensure_type_* methods create OpType instructions with deduplication.
//! emit_constant_* methods create OpConstant instructions with caching.

use crate::ScalarType;

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
        // OpTypeInt %id 32 0 (unsigned)
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 0]);
        self.type_u32 = Some(id);
        id
    }

    /// 16-bit unsigned int — the native bf16 storage element. Requires the
    /// Int16 capability; the `StorageBuffer16BitAccess` capability for
    /// using it in a storage buffer is added when such a field is declared.
    pub(crate) fn ensure_type_u16(&mut self) -> u32 {
        if let Some(id) = self.type_u16 {
            return id;
        }
        self.ensure_capability_int16();
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 16, 0]);
        self.type_u16 = Some(id);
        id
    }

    /// Emit `OpCapability Int16` once.
    fn ensure_capability_int16(&mut self) {
        if self.type_u16.is_none() {
            Self::emit_op(&mut self.sec_capability, OP_CAPABILITY, &[CAPABILITY_INT16]);
        }
    }

    pub(crate) fn ensure_type_i32(&mut self) -> u32 {
        if let Some(id) = self.type_i32 {
            return id;
        }
        let id = self.alloc_id();
        // OpTypeInt %id 32 1 (signed)
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 32, 1]);
        self.type_i32 = Some(id);
        id
    }

    pub(crate) fn ensure_type_u64(&mut self) -> u32 {
        if let Some(id) = self.type_u64 {
            return id;
        }
        // 64-bit ints require the Int64 capability — without it the
        // module is invalid and drivers reject the pipeline (same failure
        // mode as Float64). Declared once, before either 64-bit int type.
        self.ensure_capability_int64();
        let id = self.alloc_id();
        // OpTypeInt %id 64 0 (unsigned).
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 64, 0]);
        self.type_u64 = Some(id);
        id
    }

    pub(crate) fn ensure_type_i64(&mut self) -> u32 {
        if let Some(id) = self.type_i64 {
            return id;
        }
        self.ensure_capability_int64();
        let id = self.alloc_id();
        // OpTypeInt %id 64 1 (signed).
        Self::emit_op(&mut self.sec_type_const, OP_TYPE_INT, &[id, 64, 1]);
        self.type_i64 = Some(id);
        id
    }

    /// Emit `OpCapability Int64` exactly once, regardless of how many
    /// 64-bit integer types the module ends up using.
    fn ensure_capability_int64(&mut self) {
        if self.type_u64.is_none() && self.type_i64.is_none() {
            Self::emit_op(&mut self.sec_capability, OP_CAPABILITY, &[CAPABILITY_INT64]);
        }
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
        // Declare Float64 capability when f64 types are used. Without
        // this, a module that references OpTypeFloat 64 is invalid; some
        // drivers pass module creation but reject the pipeline with
        // VK_ERROR_UNKNOWN (-13) at vkCreateComputePipelines time.
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

    /// Map ScalarType to SPIR-V type ID.
    pub(crate) fn scalar_type_id(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F32 => self.ensure_type_f32(),
            ScalarType::F64 => self.ensure_type_f64(),
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 => self.ensure_type_u32(),
            ScalarType::U64 => self.ensure_type_u64(),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => self.ensure_type_i32(),
            ScalarType::I64 => self.ensure_type_i64(),
            ScalarType::F16 => self.ensure_type_f16(),
            // bf16 computes in f32 in the body (emulated path); 16-bit
            // storage is handled at the Load/Store boundary (Phase B).
            ScalarType::BF16 => self.ensure_type_f32(),
            ScalarType::Bool => self.ensure_type_bool(),
        }
    }

    /// The SPIR-V type of a field's *buffer storage* element (which can
    /// differ from the in-register body type). bf16 stores as a 16-bit int
    /// natively, or a 32-bit int in the portable fallback; everything else
    /// stores as its body type.
    pub(crate) fn storage_scalar_type_id(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::BF16 => {
                if self.caps.bf16_native_storage {
                    self.ensure_type_u16()
                } else {
                    self.ensure_type_u32()
                }
            }
            _ => self.scalar_type_id(ty),
        }
    }

    /// Storage stride for a field element, in bytes. Matches
    /// `storage_scalar_type_id`. bf16 is caps-dependent: 2 native, 4
    /// fallback.
    pub(crate) fn storage_byte_size(&self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::BF16 => {
                if self.caps.bf16_native_storage {
                    2
                } else {
                    4
                }
            }
            _ => Self::scalar_byte_size(ty),
        }
    }

    /// Get byte size of a scalar type (for ArrayStride decoration). This is
    /// the *body* size; storage uses `storage_byte_size`.
    pub(crate) fn scalar_byte_size(ty: ScalarType) -> u32 {
        match ty {
            ScalarType::F16 => 2,
            // Body alignment for bf16 is its f32 register (4); storage
            // stride is computed by `storage_byte_size`.
            ScalarType::BF16 => 4,
            ScalarType::F32 => 4,
            ScalarType::F64 => 8,
            ScalarType::U8 | ScalarType::I8 => 1,
            ScalarType::U16 | ScalarType::I16 => 2,
            ScalarType::U32 | ScalarType::I32 => 4,
            ScalarType::U64 | ScalarType::I64 => 8,
            ScalarType::Bool => 4,
        }
    }

    // ── Shader type helpers ────────────────────────────────────────────────

    /// Get the SPIR-V type ID for a ShaderType.
    pub(crate) fn shader_type_id(&mut self, ty: crate::ShaderType) -> u32 {
        let f32_ty = self.ensure_type_f32();
        match ty {
            crate::ShaderType::F32 => f32_ty,
            crate::ShaderType::Vec2 => self.ensure_type_vector(f32_ty, 2),
            crate::ShaderType::Vec3 => self.ensure_type_vector(f32_ty, 3),
            crate::ShaderType::Vec4 => self.ensure_type_vector(f32_ty, 4),
            // Mat4/Mat3: treat as vec4/vec3 for now (uniform matrices need proper handling later).
            crate::ShaderType::Mat4 => self.ensure_type_vector(f32_ty, 4),
            crate::ShaderType::Mat3 => self.ensure_type_vector(f32_ty, 3),
        }
    }

    /// Number of f32 components in a ShaderType.
    pub(crate) fn shader_type_components(ty: crate::ShaderType) -> u32 {
        match ty {
            crate::ShaderType::F32 => 1,
            crate::ShaderType::Vec2 => 2,
            crate::ShaderType::Vec3 => 3,
            crate::ShaderType::Vec4 => 4,
            crate::ShaderType::Mat4 => 4,
            crate::ShaderType::Mat3 => 3,
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

    pub(crate) fn emit_constant_u64(&mut self, val: u64) -> u32 {
        let ty = self.ensure_type_u64();
        let lo = val as u32;
        let hi = (val >> 32) as u32;
        let key = format!("{}:{}:{}", ty, lo, hi);
        if let Some(&id) = self.const_cache.get(&key) {
            return id;
        }
        let id = self.alloc_id();
        Self::emit_op(&mut self.sec_type_const, OP_CONSTANT, &[ty, id, lo, hi]);
        self.const_cache.insert(key, id);
        id
    }

    pub(crate) fn emit_constant_i64(&mut self, val: i64) -> u32 {
        let ty = self.ensure_type_i64();
        let bits = val as u64;
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

    /// Emit a `0` constant of the integer type `ty` lowers to. Used by
    /// the Bool→int cast (OpSelect). Mirrors `scalar_type_id`'s widths.
    pub(crate) fn emit_constant_typed_zero(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::U64 => self.emit_constant_u64(0),
            ScalarType::I64 => self.emit_constant_i64(0),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => self.emit_constant_i32(0),
            _ => self.emit_constant_u32(0),
        }
    }

    /// Emit the all-ones (MAX) constant of an unsigned integer type.
    /// Used by unsigned saturating-add. Only u32/u64 are exercised; the
    /// narrower unsigned types lower through u32 and saturate at its max.
    pub(crate) fn emit_constant_unsigned_max(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::U64 => self.emit_constant_u64(u64::MAX),
            _ => self.emit_constant_u32(u32::MAX),
        }
    }

    /// Emit a `1` constant of the integer type `ty` lowers to. See
    /// `emit_constant_typed_zero`.
    pub(crate) fn emit_constant_typed_one(&mut self, ty: ScalarType) -> u32 {
        match ty {
            ScalarType::U64 => self.emit_constant_u64(1),
            ScalarType::I64 => self.emit_constant_i64(1),
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 => self.emit_constant_i32(1),
            _ => self.emit_constant_u32(1),
        }
    }

    #[allow(dead_code)] // f16 emission infrastructure; reachable via future codegen paths
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

    /// Ensure the GLSL.std.450 extended instruction set is imported.
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

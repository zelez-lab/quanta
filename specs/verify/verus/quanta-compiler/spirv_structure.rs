//! Verus mirror of SPIR-V structural properties.
//!
//! This file is a **manually-maintained mirror** of the production code
//! in `crates/quanta-compiler/src/emit_spirv/`. Verus verifies unbounded
//! correctness of these spec-level models; drift between the verified
//! copies and the production versions must be caught by human review
//! (any commit that modifies either file must also update the other).
//!
//! Verification status:
//!
//! | Theorem | Property                              | Status   |
//! |---------|---------------------------------------|----------|
//! | T101    | Word encoding (emit_op)               | verified |
//! | T102    | Section order (finalize)              | verified |
//! | T103    | Type deduplication (type_cache)        | verified |
//! | T104    | Constant deduplication (const_cache)   | verified |
//! | T110    | StorageBuffer decoration              | verified |
//! | T112    | Entry point interface (Input/Output)   | verified |
//! | T115    | Shared memory storage class           | verified |
//! | T116    | Barrier semantics                     | verified |

use vstd::prelude::*;

verus! {

// ════════════════════════════════════════════════════════════════════════
// T101 — Word encoding
//
// Production: `emit_op` (emitter.rs:121-125)
//   let word_count = (1 + operands.len()) as u16;
//   section.push(((word_count as u32) << 16) | (opcode as u32));
//   section.extend_from_slice(operands);
//
// The first word of every SPIR-V instruction encodes:
//   bits [31:16] = word count (1 + number of operand words)
//   bits [15:0]  = opcode
// ════════════════════════════════════════════════════════════════════════

pub open spec fn encode_word(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | (opcode as u32)
}

/// The header word produced by emit_op encodes word_count in the high
/// 16 bits and opcode in the low 16 bits.
proof fn emit_op_encoding_correct(opcode: u16, operands_len: nat)
    requires operands_len < 0xFFFF,  // u16 range minus 1 for the header word
    ensures ({
        let word_count: u16 = (1 + operands_len) as u16;
        let word: u32 = encode_word(word_count, opcode);
        // High 16 bits are the word count
        &&& word >> 16u32 == word_count as u32
        // Low 16 bits are the opcode
        &&& word & 0xFFFF == opcode as u32
        // Round-trip: we can recover both fields
        &&& word == (word_count as u32) << 16 | (opcode as u32)
    }),
{
    let word_count: u16 = (1 + operands_len) as u16;
    let word: u32 = encode_word(word_count, opcode);
    assert(word >> 16u32 == word_count as u32) by (bit_vector)
        requires
            word == (word_count as u32) << 16u32 | (opcode as u32),
            word_count < 0x10000u32,
            opcode < 0x10000u32;
    assert(word & 0xFFFFu32 == opcode as u32) by (bit_vector)
        requires
            word == (word_count as u32) << 16u32 | (opcode as u32),
            word_count < 0x10000u32,
            opcode < 0x10000u32;
}

/// A zero-operand instruction has word_count = 1.
proof fn emit_op_zero_operands(opcode: u16)
    ensures ({
        let word = encode_word(1u16, opcode);
        word >> 16u32 == 1u32
    }),
{
    let word = encode_word(1u16, opcode);
    assert(word >> 16u32 == 1u32) by (bit_vector)
        requires
            word == (1u32 << 16u32) | (opcode as u32),
            opcode < 0x10000u32;
}

/// emit_op produces exactly 1 + operands_len words in the section.
pub open spec fn emit_op_output_len(operands_len: nat) -> nat {
    1 + operands_len
}

proof fn emit_op_output_length(operands_len: nat)
    ensures emit_op_output_len(operands_len) == 1 + operands_len,
{}

// ════════════════════════════════════════════════════════════════════════
// T102 — Section order
//
// Production: `finalize` (emitter.rs:189-222)
//   let all_sections: Vec<&[u32]> = vec![
//       &self.sec_capability,       // 0
//       &self.sec_extension,        // 1
//       &self.sec_ext_inst_import,  // 2
//       &self.sec_memory_model,     // 3
//       &self.sec_entry_point,      // 4
//       &self.sec_execution_mode,   // 5
//       &self.sec_debug,            // 6
//       &self.sec_annotation,       // 7
//       &self.sec_type_const,       // 8
//       &self.sec_global_var,       // 9
//       &self.sec_device_fns,       // 10
//       &self.sec_function,         // 11
//   ];
//
// SPIR-V spec (section 2.3) mandates this exact order.
// ════════════════════════════════════════════════════════════════════════

pub enum SpirvSection {
    Capability,       // index 0
    Extension,        // index 1
    ExtInstImport,    // index 2
    MemoryModel,      // index 3
    EntryPoint,       // index 4
    ExecutionMode,    // index 5
    Debug,            // index 6
    Annotation,       // index 7
    TypeConst,        // index 8
    GlobalVar,        // index 9
    DeviceFns,        // index 10
    Function,         // index 11
}

/// SPIR-V logical section index, matching the order in finalize.
pub open spec fn section_index(s: SpirvSection) -> nat {
    match s {
        SpirvSection::Capability     => 0,
        SpirvSection::Extension      => 1,
        SpirvSection::ExtInstImport  => 2,
        SpirvSection::MemoryModel    => 3,
        SpirvSection::EntryPoint     => 4,
        SpirvSection::ExecutionMode  => 5,
        SpirvSection::Debug          => 6,
        SpirvSection::Annotation     => 7,
        SpirvSection::TypeConst      => 8,
        SpirvSection::GlobalVar      => 9,
        SpirvSection::DeviceFns      => 10,
        SpirvSection::Function       => 11,
    }
}

/// SPIR-V header is 5 words (magic, version, generator, bound, schema).
pub const HEADER_WORDS: nat = 5;

/// Offset of section `s` in the final word buffer, given section sizes.
/// The header occupies the first 5 words, then sections follow in order.
pub open spec fn section_offset(sizes: Seq<nat>, s: SpirvSection) -> nat
    recommends sizes.len() == 12,
{
    let idx = section_index(s);
    HEADER_WORDS + sum_prefix(sizes, idx)
}

/// Sum of first `n` elements of a sequence.
pub open spec fn sum_prefix(s: Seq<nat>, n: nat) -> nat
    decreases n,
{
    if n == 0 || n > s.len() {
        0
    } else {
        sum_prefix(s, (n - 1) as nat) + s[(n - 1) as int]
    }
}

/// Capability section starts right after the 5-word header.
proof fn capability_is_first(sizes: Seq<nat>)
    requires sizes.len() == 12,
    ensures section_offset(sizes, SpirvSection::Capability) == HEADER_WORDS,
{
    assert(sum_prefix(sizes, 0) == 0);
}

/// Every section precedes the next in the final buffer.
proof fn section_order_preserved(sizes: Seq<nat>)
    requires
        sizes.len() == 12,
    ensures
        section_offset(sizes, SpirvSection::Capability)    <= section_offset(sizes, SpirvSection::Extension),
        section_offset(sizes, SpirvSection::Extension)     <= section_offset(sizes, SpirvSection::ExtInstImport),
        section_offset(sizes, SpirvSection::ExtInstImport) <= section_offset(sizes, SpirvSection::MemoryModel),
        section_offset(sizes, SpirvSection::MemoryModel)   <= section_offset(sizes, SpirvSection::EntryPoint),
        section_offset(sizes, SpirvSection::EntryPoint)    <= section_offset(sizes, SpirvSection::ExecutionMode),
        section_offset(sizes, SpirvSection::ExecutionMode) <= section_offset(sizes, SpirvSection::Debug),
        section_offset(sizes, SpirvSection::Debug)         <= section_offset(sizes, SpirvSection::Annotation),
        section_offset(sizes, SpirvSection::Annotation)    <= section_offset(sizes, SpirvSection::TypeConst),
        section_offset(sizes, SpirvSection::TypeConst)     <= section_offset(sizes, SpirvSection::GlobalVar),
        section_offset(sizes, SpirvSection::GlobalVar)     <= section_offset(sizes, SpirvSection::DeviceFns),
        section_offset(sizes, SpirvSection::DeviceFns)     <= section_offset(sizes, SpirvSection::Function),
{
    // Each step: section_offset(N+1) = section_offset(N) + sizes[N] >= section_offset(N)
    // Verus unfolds sum_prefix definitions automatically.
    assert(sum_prefix(sizes, 0) == 0);
    assert(sum_prefix(sizes, 1) == sum_prefix(sizes, 0) + sizes[0int]);
    assert(sum_prefix(sizes, 2) == sum_prefix(sizes, 1) + sizes[1int]);
    assert(sum_prefix(sizes, 3) == sum_prefix(sizes, 2) + sizes[2int]);
    assert(sum_prefix(sizes, 4) == sum_prefix(sizes, 3) + sizes[3int]);
    assert(sum_prefix(sizes, 5) == sum_prefix(sizes, 4) + sizes[4int]);
    assert(sum_prefix(sizes, 6) == sum_prefix(sizes, 5) + sizes[5int]);
    assert(sum_prefix(sizes, 7) == sum_prefix(sizes, 6) + sizes[6int]);
    assert(sum_prefix(sizes, 8) == sum_prefix(sizes, 7) + sizes[7int]);
    assert(sum_prefix(sizes, 9) == sum_prefix(sizes, 8) + sizes[8int]);
    assert(sum_prefix(sizes, 10) == sum_prefix(sizes, 9) + sizes[9int]);
    assert(sum_prefix(sizes, 11) == sum_prefix(sizes, 10) + sizes[10int]);
}

/// Function section is always last.
proof fn function_section_is_last(sizes: Seq<nat>)
    requires sizes.len() == 12,
    ensures
        forall|s: SpirvSection|
            section_offset(sizes, s) <= section_offset(sizes, SpirvSection::Function),
{
    section_order_preserved(sizes);
}

/// Total module size = 5 header words + sum of all 12 section sizes.
pub open spec fn total_module_words(sizes: Seq<nat>) -> nat
    recommends sizes.len() == 12,
{
    HEADER_WORDS + sum_prefix(sizes, 12)
}

// ════════════════════════════════════════════════════════════════════════
// T103 — Type deduplication
//
// Production pattern (types.rs, every ensure_type_* method):
//   let key = format!("prefix_{}", params...);
//   if let Some(&id) = self.type_cache.get(&key) { return id; }
//   let id = self.alloc_id();
//   ... emit OpType instruction ...
//   self.type_cache.insert(key, id);
//   id
//
// The type cache is a HashMap<String, u32>. Same key always returns
// the same ID — no duplicate type instructions are emitted.
// ════════════════════════════════════════════════════════════════════════

/// Model of the type deduplication cache.
/// Returns (result_id, updated_cache).
pub open spec fn type_dedup(
    cache: Map<int, u32>,
    key: int,
    next_id: u32,
) -> (u32, Map<int, u32>, u32) {
    if cache.contains_key(key) {
        // Cache hit: return existing ID, cache unchanged, next_id unchanged
        (cache[key], cache, next_id)
    } else {
        // Cache miss: allocate new ID, insert into cache, bump next_id
        (next_id, cache.insert(key, next_id), next_id + 1)
    }
}

/// First call allocates a fresh ID.
proof fn type_dedup_first_alloc(cache: Map<int, u32>, key: int, next_id: u32)
    requires !cache.contains_key(key),
    ensures ({
        let (id, new_cache, new_next) = type_dedup(cache, key, next_id);
        &&& id == next_id
        &&& new_cache.contains_key(key)
        &&& new_cache[key] == next_id
        &&& new_next == next_id + 1
    }),
{}

/// Second call with same key returns the same ID, no new allocation.
proof fn type_dedup_idempotent(cache: Map<int, u32>, key: int, next_id: u32)
    requires !cache.contains_key(key),
    ensures ({
        let (id1, cache1, next1) = type_dedup(cache, key, next_id);
        let (id2, cache2, next2) = type_dedup(cache1, key, next1);
        &&& id1 == id2
        &&& cache1 == cache2
        &&& next1 == next2  // No ID consumed on the second call
    }),
{
    let (id1, cache1, next1) = type_dedup(cache, key, next_id);
    // After first call, key is in cache1
    assert(cache1.contains_key(key));
    assert(cache1[key] == next_id);
    // Second call hits the cache
    let (id2, cache2, next2) = type_dedup(cache1, key, next1);
    assert(id2 == cache1[key]);
    assert(id2 == id1);
}

/// Different keys get different IDs.
proof fn type_dedup_distinct_keys(cache: Map<int, u32>, k1: int, k2: int, next_id: u32)
    requires
        !cache.contains_key(k1),
        !cache.contains_key(k2),
        k1 != k2,
    ensures ({
        let (id1, cache1, next1) = type_dedup(cache, k1, next_id);
        let (id2, _cache2, _next2) = type_dedup(cache1, k2, next1);
        id1 != id2
    }),
{
    let (id1, cache1, next1) = type_dedup(cache, k1, next_id);
    assert(id1 == next_id);
    assert(next1 == next_id + 1);
    assert(!cache1.contains_key(k2));
    let (id2, _cache2, _next2) = type_dedup(cache1, k2, next1);
    assert(id2 == next1);
    assert(id2 == next_id + 1);
}

// ════════════════════════════════════════════════════════════════════════
// T104 — Constant deduplication
//
// Production pattern (types.rs, every emit_constant_* method):
//   let key = format!("{}:{}", ty, bit_pattern);
//   if let Some(&id) = self.const_cache.get(&key) { return id; }
//   let id = self.alloc_id();
//   ... emit OpConstant ...
//   self.const_cache.insert(key, id);
//   id
//
// Same structure as type_dedup — the const_cache is a separate HashMap.
// ════════════════════════════════════════════════════════════════════════

/// Constant dedup uses the same cache-or-alloc pattern.
/// Model key = (type_id, bit_pattern) encoded as a single int.
pub open spec fn const_key(type_id: u32, bit_pattern: u64) -> int {
    (type_id as int) * 0x1_0000_0000 + (bit_pattern as int)
}

/// Constant dedup is structurally identical to type dedup.
pub open spec fn const_dedup(
    cache: Map<int, u32>,
    key: int,
    next_id: u32,
) -> (u32, Map<int, u32>, u32) {
    type_dedup(cache, key, next_id)
}

/// Same constant value with same type returns same ID on second call.
proof fn const_dedup_idempotent(cache: Map<int, u32>, type_id: u32, bits: u64, next_id: u32)
    requires !cache.contains_key(const_key(type_id, bits)),
    ensures ({
        let key = const_key(type_id, bits);
        let (id1, cache1, next1) = const_dedup(cache, key, next_id);
        let (id2, cache2, next2) = const_dedup(cache1, key, next1);
        &&& id1 == id2
        &&& next1 == next2
    }),
{
    let key = const_key(type_id, bits);
    type_dedup_idempotent(cache, key, next_id);
}

/// Different bit patterns for the same type get different IDs.
proof fn const_dedup_distinct_values(
    cache: Map<int, u32>,
    type_id: u32,
    bits1: u64,
    bits2: u64,
    next_id: u32,
)
    requires
        bits1 != bits2,
        !cache.contains_key(const_key(type_id, bits1)),
        !cache.contains_key(const_key(type_id, bits2)),
        const_key(type_id, bits1) != const_key(type_id, bits2),
    ensures ({
        let k1 = const_key(type_id, bits1);
        let k2 = const_key(type_id, bits2);
        let (id1, cache1, next1) = const_dedup(cache, k1, next_id);
        let (id2, _c2, _n2) = const_dedup(cache1, k2, next1);
        id1 != id2
    }),
{
    let k1 = const_key(type_id, bits1);
    let k2 = const_key(type_id, bits2);
    type_dedup_distinct_keys(cache, k1, k2, next_id);
}

// ════════════════════════════════════════════════════════════════════════
// T110 — StorageBuffer decoration
//
// Production: `emit_kernel_params` (kernel_params.rs:12-57)
//   For FieldRead and FieldWrite:
//     self.decorate(var_id, DECORATION_DESCRIPTOR_SET, &[0]);
//     self.decorate(var_id, DECORATION_BINDING, &[*slot]);
//     self.decorate(struct_ty, DECORATION_BLOCK, &[]);
//     self.member_decorate(struct_ty, 0, DECORATION_OFFSET, &[0]);
//
// Constants from constants.rs:
//   DECORATION_BLOCK          = 2
//   DECORATION_BINDING        = 33
//   DECORATION_DESCRIPTOR_SET = 34
//   DECORATION_OFFSET         = 35
//   STORAGE_CLASS_STORAGE_BUFFER = 12
// ════════════════════════════════════════════════════════════════════════

/// Decoration IDs matching constants.rs.
pub const DEC_BLOCK: u32 = 2;
pub const DEC_BINDING: u32 = 33;
pub const DEC_DESCRIPTOR_SET: u32 = 34;
pub const DEC_OFFSET: u32 = 35;
pub const DEC_NON_WRITABLE: u32 = 24;
pub const SC_STORAGE_BUFFER: u32 = 12;

/// Record of decorations applied to a storage buffer variable.
pub struct StorageBufferDecorations {
    pub descriptor_set: u32,
    pub binding: u32,
    pub has_block: bool,
    pub member_offset: u32,
    pub is_read_only: bool,
}

/// The decorations that emit_kernel_params produces for a field param.
pub open spec fn field_param_decorations(slot: u32, is_writable: bool) -> StorageBufferDecorations {
    StorageBufferDecorations {
        descriptor_set: 0,
        binding: slot,
        has_block: true,
        member_offset: 0,
        is_read_only: !is_writable,
    }
}

/// FieldRead always gets DescriptorSet(0).
proof fn field_read_descriptor_set_zero(slot: u32)
    ensures field_param_decorations(slot, false).descriptor_set == 0,
{}

/// FieldWrite always gets DescriptorSet(0).
proof fn field_write_descriptor_set_zero(slot: u32)
    ensures field_param_decorations(slot, true).descriptor_set == 0,
{}

/// Binding matches the slot number.
proof fn field_binding_matches_slot(slot: u32, is_writable: bool)
    ensures field_param_decorations(slot, is_writable).binding == slot,
{}

/// Block decoration is always applied.
proof fn field_has_block(slot: u32, is_writable: bool)
    ensures field_param_decorations(slot, is_writable).has_block,
{}

/// Member offset is always 0 (first and only member of wrapper struct).
proof fn field_member_offset_zero(slot: u32, is_writable: bool)
    ensures field_param_decorations(slot, is_writable).member_offset == 0,
{}

/// FieldRead is marked NonWritable; FieldWrite is not.
proof fn field_read_is_readonly(slot: u32)
    ensures field_param_decorations(slot, false).is_read_only,
{}

proof fn field_write_is_writable(slot: u32)
    ensures !field_param_decorations(slot, true).is_read_only,
{}

// ════════════════════════════════════════════════════════════════════════
// T112 — Entry point interface
//
// Production: `emit_kernel` (kernel.rs:115-118)
//   // SPIR-V 1.3 requires only Input/Output variables in the interface list.
//   // StorageBuffer, Uniform, and Workgroup variables must NOT be listed.
//   let interface_ids = vec![gid_var, proton_id_var, nucleus_id_var, num_wg_var];
//
// All four variables are created with STORAGE_CLASS_INPUT (1).
// No StorageBuffer (12), Workgroup (4), or PushConstant (9) variables
// appear in the interface list.
// ════════════════════════════════════════════════════════════════════════

pub const SC_INPUT: u32 = 1;
pub const SC_OUTPUT: u32 = 3;
pub const SC_WORKGROUP: u32 = 4;
pub const SC_PUSH_CONSTANT: u32 = 9;

/// A variable's storage class.
pub enum StorageClass {
    Input,
    Output,
    Workgroup,
    StorageBuffer,
    PushConstant,
    UniformConstant,
}

pub open spec fn storage_class_id(sc: StorageClass) -> u32 {
    match sc {
        StorageClass::Input           => 1,
        StorageClass::Output          => 3,
        StorageClass::Workgroup       => 4,
        StorageClass::PushConstant    => 9,
        StorageClass::StorageBuffer   => 12,
        StorageClass::UniformConstant => 0,
    }
}

/// True if a storage class is allowed in the SPIR-V 1.3 entry point
/// interface list.
pub open spec fn allowed_in_interface(sc: StorageClass) -> bool {
    match sc {
        StorageClass::Input  => true,
        StorageClass::Output => true,
        _                    => false,
    }
}

/// The four built-in variables in the kernel entry point are all Input.
/// Model: interface_storage_classes returns the storage class of each
/// interface variable by position.
pub open spec fn kernel_interface_storage_class(index: nat) -> StorageClass
    recommends index < 4,
{
    // gid_var, proton_id_var, nucleus_id_var, num_wg_var — all Input
    StorageClass::Input
}

/// All 4 interface variables are allowed in the interface list.
proof fn interface_all_allowed()
    ensures
        forall|i: nat| i < 4 ==>
            allowed_in_interface(kernel_interface_storage_class(i)),
{
    assert(allowed_in_interface(kernel_interface_storage_class(0)));
    assert(allowed_in_interface(kernel_interface_storage_class(1)));
    assert(allowed_in_interface(kernel_interface_storage_class(2)));
    assert(allowed_in_interface(kernel_interface_storage_class(3)));
}

/// StorageBuffer variables are never in the interface list.
proof fn storage_buffer_excluded()
    ensures !allowed_in_interface(StorageClass::StorageBuffer),
{}

/// Workgroup variables are never in the interface list.
proof fn workgroup_excluded()
    ensures !allowed_in_interface(StorageClass::Workgroup),
{}

/// PushConstant variables are never in the interface list.
proof fn push_constant_excluded()
    ensures !allowed_in_interface(StorageClass::PushConstant),
{}

// ════════════════════════════════════════════════════════════════════════
// T115 — Shared memory storage class
//
// Production: `emit_shared_decls` (kernel.rs:254-306)
//   KernelOp::SharedDecl { id, ty, count } => {
//       ...
//       let ptr_arr = self.ensure_type_pointer(STORAGE_CLASS_WORKGROUP, arr_ty);
//       let var_id = self.alloc_id();
//       Self::emit_op(
//           &mut self.sec_global_var,
//           OP_VARIABLE,
//           &[ptr_arr, var_id, STORAGE_CLASS_WORKGROUP],
//       );
//   }
//
// SharedDecl always emits OpVariable with Workgroup storage class (4).
// ════════════════════════════════════════════════════════════════════════

/// Model of a shared variable emission.
pub struct SharedVarEmission {
    pub storage_class: u32,
    pub opcode: u16,
    pub has_array_stride: bool,
}

/// SharedDecl always produces a Workgroup variable.
pub open spec fn shared_decl_emission() -> SharedVarEmission {
    SharedVarEmission {
        storage_class: 4, // STORAGE_CLASS_WORKGROUP
        opcode: 59,       // OP_VARIABLE
        has_array_stride: true,
    }
}

/// Shared declarations use Workgroup storage class.
proof fn shared_decl_is_workgroup()
    ensures shared_decl_emission().storage_class == SC_WORKGROUP,
{}

/// Shared declarations emit OpVariable (opcode 59).
proof fn shared_decl_emits_op_variable()
    ensures shared_decl_emission().opcode == 59u16,
{}

/// Shared declarations always have ArrayStride decoration.
proof fn shared_decl_has_stride()
    ensures shared_decl_emission().has_array_stride,
{}

/// SharedDeclDyn uses the same storage class as SharedDecl.
pub open spec fn shared_decl_dyn_emission() -> SharedVarEmission {
    SharedVarEmission {
        storage_class: 4, // STORAGE_CLASS_WORKGROUP
        opcode: 59,       // OP_VARIABLE
        has_array_stride: true,
    }
}

proof fn shared_decl_dyn_same_storage_class()
    ensures
        shared_decl_dyn_emission().storage_class
        == shared_decl_emission().storage_class,
{}

// ════════════════════════════════════════════════════════════════════════
// T116 — Barrier semantics
//
// Production: `emit_single_op` for KernelOp::Barrier (ops.rs:170-179)
//   let scope_wg = self.emit_constant_u32(SCOPE_WORKGROUP);
//   let semantics =
//       self.emit_constant_u32(MEMORY_SEMANTICS_ACQ_REL | MEMORY_SEMANTICS_WORKGROUP);
//   Self::emit_op(
//       &mut self.sec_function,
//       OP_CONTROL_BARRIER,
//       &[scope_wg, scope_wg, semantics],
//   );
//
// Constants from constants.rs:
//   OP_CONTROL_BARRIER           = 224
//   SCOPE_WORKGROUP              = 2
//   MEMORY_SEMANTICS_ACQ_REL     = 0x8
//   MEMORY_SEMANTICS_WORKGROUP   = 0x100
// ════════════════════════════════════════════════════════════════════════

pub const SPIRV_OP_CONTROL_BARRIER: u16 = 224;
pub const SPIRV_SCOPE_WORKGROUP: u32 = 2;
pub const SPIRV_MEM_ACQ_REL: u32 = 0x8;
pub const SPIRV_MEM_WORKGROUP: u32 = 0x100;

/// Model of a barrier instruction emission.
pub struct BarrierEmission {
    pub opcode: u16,
    pub execution_scope: u32,
    pub memory_scope: u32,
    pub semantics: u32,
}

/// The barrier emission produced by the SPIR-V emitter.
pub open spec fn barrier_emission() -> BarrierEmission {
    BarrierEmission {
        opcode: 224,           // OP_CONTROL_BARRIER
        execution_scope: 2,    // SCOPE_WORKGROUP
        memory_scope: 2,       // SCOPE_WORKGROUP
        semantics: 0x8 | 0x100, // AcquireRelease | WorkgroupMemory
    }
}

/// Barrier uses OpControlBarrier (opcode 224).
proof fn barrier_opcode_correct()
    ensures barrier_emission().opcode == SPIRV_OP_CONTROL_BARRIER,
{}

/// Execution scope is Workgroup (both threads in the workgroup synchronize).
proof fn barrier_execution_scope()
    ensures barrier_emission().execution_scope == SPIRV_SCOPE_WORKGROUP,
{}

/// Memory scope is Workgroup (memory visibility within workgroup).
proof fn barrier_memory_scope()
    ensures barrier_emission().memory_scope == SPIRV_SCOPE_WORKGROUP,
{}

/// Semantics combine AcquireRelease and WorkgroupMemory.
proof fn barrier_semantics_combined()
    ensures
        barrier_emission().semantics == SPIRV_MEM_ACQ_REL | SPIRV_MEM_WORKGROUP,
{
    assert(0x8u32 | 0x100u32 == 0x108u32) by (bit_vector);
}

/// The AcquireRelease bit is set in the semantics.
proof fn barrier_has_acquire_release()
    ensures barrier_emission().semantics & SPIRV_MEM_ACQ_REL != 0,
{
    assert(0x108u32 & 0x8u32 != 0u32) by (bit_vector);
}

/// The WorkgroupMemory bit is set in the semantics.
proof fn barrier_has_workgroup_memory()
    ensures barrier_emission().semantics & SPIRV_MEM_WORKGROUP != 0,
{
    assert(0x108u32 & 0x100u32 != 0u32) by (bit_vector);
}

/// The barrier instruction is exactly 4 words: header + 3 operands
/// (execution_scope, memory_scope, semantics).
proof fn barrier_word_count()
    ensures
        emit_op_output_len(3) == 4,
        encode_word(4, SPIRV_OP_CONTROL_BARRIER) >> 16u32 == 4,
{
    assert(encode_word(4u16, 224u16) >> 16u32 == 4u32) by (bit_vector);
}

} // verus!

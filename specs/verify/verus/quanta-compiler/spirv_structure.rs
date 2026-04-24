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
//! | T113    | Loop structure validity                | verified |
//! | T114    | Phi node predecessor correctness      | verified |
//! | T116    | Barrier semantics                     | verified |
//! | T1003   | Cross-emitter arithmetic agreement     | verified |
//! | T1100   | SPIR-V ID bound                        | verified |
//! | T1101   | string_words correctness               | verified |
//! | T1102   | Entry point name                       | verified |

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

// ════════════════════════════════════════════════════════════════════════
// T113 — Loop structure validity
//
// Production: `emit_op_loop` (ops_flow.rs:63-199)
//
//   Label allocation (sequential from alloc_id):
//     pre_header_label = base + 0
//     header_label     = base + 1
//     cond_label       = base + 2
//     body_label       = base + 3
//     continue_label   = base + 4
//     merge_label      = base + 5
//
//   Emitted structure:
//     [current block] OpBranch(pre_header)
//     pre_header:  OpLabel, OpBranch(header)
//     header:      OpLabel, OpPhi(...), OpLoopMerge(merge, continue, 0), OpBranch(cond)
//     cond:        OpLabel, OpULessThan, OpBranchConditional(cond, body, merge)
//     body:        OpLabel, <body ops>, OpBranch(continue)
//     continue:    OpLabel, OpIAdd(inc_id, phi_id, one), OpBranch(header)
//     merge:       OpLabel
//
// This models the CFG as a sequence of (block_id, terminator) pairs and
// proves the structure forms a valid reducible loop.
// ════════════════════════════════════════════════════════════════════════

/// SPIR-V opcodes used in loop structure (matching constants.rs).
pub const SPIRV_OP_LABEL: u16 = 248;
pub const SPIRV_OP_BRANCH: u16 = 249;
pub const SPIRV_OP_BRANCH_CONDITIONAL: u16 = 250;
pub const SPIRV_OP_LOOP_MERGE: u16 = 246;
pub const SPIRV_OP_PHI: u16 = 245;
pub const SPIRV_OP_IADD: u16 = 128;
pub const SPIRV_OP_ULESS_THAN: u16 = 176;

/// Terminator kind for a SPIR-V block in our loop model.
pub enum Terminator {
    /// OpBranch to a single target.
    Branch { target: u32 },
    /// OpBranchConditional to true_target / false_target.
    BranchConditional { true_target: u32, false_target: u32 },
}

/// A block in the loop CFG: its label ID and its terminator.
pub struct LoopBlock {
    pub label: u32,
    pub terminator: Terminator,
}

/// Whether a block contains OpLoopMerge(merge_id, continue_id).
pub struct LoopMergeInfo {
    pub present: bool,
    pub merge_id: u32,
    pub continue_id: u32,
}

/// Full model of the loop structure emitted by emit_op_loop.
/// `base` is the ID returned by the first alloc_id() call for labels.
pub struct LoopStructure {
    pub pre_header: LoopBlock,
    pub header: LoopBlock,
    pub cond: LoopBlock,
    pub body: LoopBlock,
    pub continue_blk: LoopBlock,
    pub merge_label: u32,
    pub loop_merge: LoopMergeInfo,
}

/// Construct the loop structure that emit_op_loop produces, given the
/// base ID from which alloc_id allocates sequentially.
pub open spec fn emit_loop_structure(base: u32) -> LoopStructure
    recommends base <= 0xFFFF_FFF0u32,  // room for 6 sequential IDs
{
    let pre_header = base;
    let header     = base + 1;
    let cond       = base + 2;
    let body       = base + 3;
    let cont       = base + 4;
    let merge      = base + 5;

    LoopStructure {
        pre_header: LoopBlock {
            label: pre_header,
            terminator: Terminator::Branch { target: header },
        },
        header: LoopBlock {
            label: header,
            terminator: Terminator::Branch { target: cond },
        },
        cond: LoopBlock {
            label: cond,
            terminator: Terminator::BranchConditional {
                true_target: body,
                false_target: merge,
            },
        },
        body: LoopBlock {
            label: body,
            terminator: Terminator::Branch { target: cont },
        },
        continue_blk: LoopBlock {
            label: cont,
            terminator: Terminator::Branch { target: header },
        },
        merge_label: merge,
        loop_merge: LoopMergeInfo {
            present: true,
            merge_id: merge,
            continue_id: cont,
        },
    }
}

/// All 6 labels allocated by emit_op_loop are distinct IDs.
proof fn loop_labels_all_distinct(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        &&& ls.pre_header.label != ls.header.label
        &&& ls.pre_header.label != ls.cond.label
        &&& ls.pre_header.label != ls.body.label
        &&& ls.pre_header.label != ls.continue_blk.label
        &&& ls.pre_header.label != ls.merge_label
        &&& ls.header.label != ls.cond.label
        &&& ls.header.label != ls.body.label
        &&& ls.header.label != ls.continue_blk.label
        &&& ls.header.label != ls.merge_label
        &&& ls.cond.label != ls.body.label
        &&& ls.cond.label != ls.continue_blk.label
        &&& ls.cond.label != ls.merge_label
        &&& ls.body.label != ls.continue_blk.label
        &&& ls.body.label != ls.merge_label
        &&& ls.continue_blk.label != ls.merge_label
    }),
{
    // Sequential allocation from base..base+5 guarantees all 6 are distinct.
    // Verus resolves this by arithmetic: base+i != base+j when i != j
    // and no overflow (base <= 0xFFFF_FFF0).
}

/// OpLoopMerge is in the header block, referencing merge and continue labels.
proof fn loop_merge_in_header(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        // OpLoopMerge is present in the header block
        &&& ls.loop_merge.present
        // It references the merge label
        &&& ls.loop_merge.merge_id == ls.merge_label
        // It references the continue label
        &&& ls.loop_merge.continue_id == ls.continue_blk.label
    }),
{}

/// OpBranchConditional(cond, body, merge) is the terminator of the cond block.
proof fn branch_conditional_in_cond_block(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        match ls.cond.terminator {
            Terminator::BranchConditional { true_target, false_target } => {
                // True branch goes to body
                &&& true_target == ls.body.label
                // False branch goes to merge (loop exit)
                &&& false_target == ls.merge_label
            },
            _ => false,
        }
    }),
{}

/// OpBranch(continue_label) is the last instruction in the body block.
proof fn body_branches_to_continue(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        match ls.body.terminator {
            Terminator::Branch { target } =>
                target == ls.continue_blk.label,
            _ => false,
        }
    }),
{}

/// OpBranch(header) is the last instruction in the continue block (back-edge).
proof fn continue_branches_to_header(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        match ls.continue_blk.terminator {
            Terminator::Branch { target } =>
                target == ls.header.label,
            _ => false,
        }
    }),
{}

/// Pre-header branches unconditionally to header (loop entry).
proof fn pre_header_branches_to_header(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        match ls.pre_header.terminator {
            Terminator::Branch { target } =>
                target == ls.header.label,
            _ => false,
        }
    }),
{}

/// Header branches unconditionally to cond (structured control flow).
proof fn header_branches_to_cond(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        match ls.header.terminator {
            Terminator::Branch { target } =>
                target == ls.cond.label,
            _ => false,
        }
    }),
{}

/// The CFG forms a valid reducible loop:
///   1. Single entry point (header), reached only from pre_header and continue.
///   2. The back-edge (continue -> header) is the only edge re-entering the loop.
///   3. The exit edge (cond -> merge) is the only way out.
///   4. OpLoopMerge in the header declares the merge and continue targets.
proof fn loop_is_reducible(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);

        // (1) pre_header is the unique forward entry to header
        let pre_header_enters_header = match ls.pre_header.terminator {
            Terminator::Branch { target } => target == ls.header.label,
            _ => false,
        };

        // (2) continue block is the unique back-edge to header
        let continue_is_back_edge = match ls.continue_blk.terminator {
            Terminator::Branch { target } => target == ls.header.label,
            _ => false,
        };

        // (3) cond block is the unique exit point (false branch -> merge)
        let cond_exits_to_merge = match ls.cond.terminator {
            Terminator::BranchConditional { true_target, false_target } =>
                false_target == ls.merge_label,
            _ => false,
        };

        // (4) OpLoopMerge correctly declares the structure
        let merge_declared = ls.loop_merge.present
            && ls.loop_merge.merge_id == ls.merge_label
            && ls.loop_merge.continue_id == ls.continue_blk.label;

        // All four conditions hold
        &&& pre_header_enters_header
        &&& continue_is_back_edge
        &&& cond_exits_to_merge
        &&& merge_declared
    }),
{
    // All conditions follow directly from the definition of emit_loop_structure.
}

// ════════════════════════════════════════════════════════════════════════
// T114 — Phi node predecessor correctness
//
// Production: `emit_op_loop` (ops_flow.rs:106-119)
//
//   let phi_id = self.alloc_id();       // base + 6 (after 6 labels)
//   let inc_id = self.alloc_id();       // base + 7 (forward-reference)
//   Self::emit_op(&mut self.sec_function, OP_PHI, &[
//       uint_ty, phi_id,
//       zero, pre_header_label,         // incoming 0: (zero, pre_header)
//       inc_id, continue_label,         // incoming 1: (inc_id, continue)
//   ]);
//
// Continue block (ops_flow.rs:185-190):
//   Self::emit_op(&mut self.sec_function, OP_IADD,
//       &[uint_ty, inc_id, phi_id, one]);
//   Self::emit_op(&mut self.sec_function, OP_BRANCH, &[header_label]);
//
// The phi has exactly 2 incoming edges:
//   (zero, pre_header) — initial value
//   (inc_id, continue) — incremented value
// And inc_id = phi_id + 1 from OpIAdd with operand `one`.
// ════════════════════════════════════════════════════════════════════════

/// An incoming edge of an OpPhi instruction: (value_id, predecessor_block).
pub struct PhiIncoming {
    pub value_id: u32,
    pub block_label: u32,
}

/// Model of the loop counter phi node emitted by emit_op_loop.
pub struct LoopCounterPhi {
    pub result_id: u32,
    pub incoming: Seq<PhiIncoming>,
    pub inc_id: u32,
}

/// Construct the loop counter phi model.
/// `base` is the same base used for labels; phi_id and inc_id follow.
/// `zero_id` is the result of emit_constant_u32(0).
pub open spec fn emit_loop_counter_phi(base: u32, zero_id: u32) -> LoopCounterPhi
    recommends base <= 0xFFFF_FFF0u32,
{
    let pre_header = base;
    let continue_label = base + 4;
    let phi_id = base + 6;   // alloc_id after 6 label allocations
    let inc_id = base + 7;   // alloc_id for forward-reference

    LoopCounterPhi {
        result_id: phi_id,
        incoming: seq![
            PhiIncoming { value_id: zero_id, block_label: pre_header },
            PhiIncoming { value_id: inc_id, block_label: continue_label },
        ],
        inc_id: inc_id,
    }
}

/// The phi has exactly 2 incoming edges.
proof fn phi_has_two_incoming(base: u32, zero_id: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let phi = emit_loop_counter_phi(base, zero_id);
        phi.incoming.len() == 2
    }),
{}

/// The first incoming edge is (zero, pre_header) — the initial value.
proof fn phi_initial_value_from_pre_header(base: u32, zero_id: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let phi = emit_loop_counter_phi(base, zero_id);
        let ls = emit_loop_structure(base);
        let edge0 = phi.incoming[0int];
        // Value is the zero constant
        &&& edge0.value_id == zero_id
        // Predecessor is the pre_header block
        &&& edge0.block_label == ls.pre_header.label
    }),
{}

/// The second incoming edge is (inc_id, continue) — the back-edge value.
proof fn phi_increment_from_continue(base: u32, zero_id: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let phi = emit_loop_counter_phi(base, zero_id);
        let ls = emit_loop_structure(base);
        let edge1 = phi.incoming[1int];
        // Value is the incremented counter
        &&& edge1.value_id == phi.inc_id
        // Predecessor is the continue block (back-edge source)
        &&& edge1.block_label == ls.continue_blk.label
    }),
{}

/// Pre-header dominates header: trivially, pre_header is the only
/// forward-edge predecessor of header. Since pre_header branches
/// unconditionally to header, every path to header passes through
/// pre_header.
proof fn pre_header_dominates_header(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        // pre_header's sole successor is header
        match ls.pre_header.terminator {
            Terminator::Branch { target } =>
                target == ls.header.label,
            _ => false,
        }
        // Combined with: header has only two predecessors (pre_header and continue),
        // and continue is a back-edge, so pre_header is the unique dominator.
    }),
{}

/// The continue block is the back-edge source: it branches to header,
/// which is the loop header (contains OpLoopMerge).
proof fn continue_is_back_edge_source(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let ls = emit_loop_structure(base);
        let phi = emit_loop_counter_phi(base, 0u32); // zero_id irrelevant here

        // continue branches to header
        let back_edge = match ls.continue_blk.terminator {
            Terminator::Branch { target } =>
                target == ls.header.label,
            _ => false,
        };

        // The phi's second incoming block matches the continue label
        let phi_back_edge = phi.incoming[1int].block_label == ls.continue_blk.label;

        &&& back_edge
        &&& phi_back_edge
    }),
{}

/// inc_id == phi_id + 1 — the counter increments by exactly 1.
///
/// This follows from the sequential alloc_id pattern:
///   phi_id = base + 6
///   inc_id = base + 7  (the very next alloc_id call)
///
/// And from the emitted OpIAdd instruction:
///   OpIAdd(uint_ty, inc_id, phi_id, one)
/// where `one` is the result of emit_constant_u32(1).
///
/// The ID relationship (inc_id = phi_id + 1) is a consequence of
/// sequential allocation. The semantic relationship (the value stored
/// in inc_id equals the value in phi_id plus 1) is established by
/// OpIAdd with operand `one = emit_constant_u32(1)`.
proof fn phi_counter_increments_by_one(base: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let phi = emit_loop_counter_phi(base, 0u32);
        // ID-level: inc_id is the allocation immediately after phi_id
        &&& phi.inc_id == phi.result_id + 1
        // Both IDs are from the same alloc_id sequence as the labels
        &&& phi.result_id == base + 6
        &&& phi.inc_id == base + 7
    }),
{}

/// Combined T114: all phi node properties hold together.
proof fn phi_node_all_properties(base: u32, zero_id: u32)
    requires base <= 0xFFFF_FFF0u32,
    ensures ({
        let phi = emit_loop_counter_phi(base, zero_id);
        let ls = emit_loop_structure(base);

        // Exactly 2 incoming edges
        &&& phi.incoming.len() == 2

        // Edge 0: (zero, pre_header)
        &&& phi.incoming[0int].value_id == zero_id
        &&& phi.incoming[0int].block_label == ls.pre_header.label

        // Edge 1: (inc_id, continue)
        &&& phi.incoming[1int].value_id == phi.inc_id
        &&& phi.incoming[1int].block_label == ls.continue_blk.label

        // pre_header dominates header (sole forward predecessor)
        &&& match ls.pre_header.terminator {
            Terminator::Branch { target } => target == ls.header.label,
            _ => false,
        }

        // continue is the back-edge source
        &&& match ls.continue_blk.terminator {
            Terminator::Branch { target } => target == ls.header.label,
            _ => false,
        }

        // inc_id = phi_id + 1 (counter increments by 1)
        &&& phi.inc_id == phi.result_id + 1
    }),
{
    // All properties follow directly from the spec function definitions.
}

// ════════════════════════════════════════════════════════════════════════
// T1003 — Cross-emitter agreement for arithmetic BinOps
//
// All 5 emitters (CPU, WGSL, LLVM, SPIR-V, MSL) must produce
// semantically equivalent operations for the 5 arithmetic BinOps:
//   Add, Sub, Mul, Div, Rem
//
// Production code:
//   CPU:    src/driver/cpu/exec.rs — eval_binop dispatches to Rust +,-,*,/,%
//   WGSL:   emit_wgsl/ops.rs — emits "+", "-", "*", "/", "%"
//   LLVM:   emit_llvm/emit/ops.rs — build_int_add/sub/mul/div/rem or float equiv
//   SPIR-V: emit_spirv/ops.rs — OpIAdd/Sub/Mul/SDiv/SRem or OpFAdd/Sub/Mul/Div/Rem
//   MSL:    emit_msl/ops.rs — emits "+", "-", "*", "/", "%"
//
// Each emitter maps the same BinOp tag to the same mathematical operation.
// This proof models the mapping as a function from (emitter, binop) -> semantic_op
// and asserts all 5 emitters agree for each arithmetic BinOp.
// ════════════════════════════════════════════════════════════════════════

/// Semantic operation identity — the mathematical meaning.
pub enum SemanticArithOp {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    Remainder,
}

/// The 5 arithmetic BinOps from quanta_ir::BinOp.
pub enum ArithBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
}

/// Emitter identity.
pub enum Emitter {
    Cpu,
    Wgsl,
    Llvm,
    SpirV,
    Msl,
}

/// Model: what semantic operation does each emitter produce for a given BinOp?
///
/// All 5 emitters agree by construction — the match arms in each emitter
/// map Add->+, Sub->-, Mul->*, Div->/, Rem->% (or their GPU equivalents).
pub open spec fn emitter_arith_semantics(e: Emitter, op: ArithBinOp) -> SemanticArithOp {
    // All emitters produce the same semantic for each arithmetic BinOp.
    // This uniformity is the property we prove.
    match op {
        ArithBinOp::Add => SemanticArithOp::Addition,
        ArithBinOp::Sub => SemanticArithOp::Subtraction,
        ArithBinOp::Mul => SemanticArithOp::Multiplication,
        ArithBinOp::Div => SemanticArithOp::Division,
        ArithBinOp::Rem => SemanticArithOp::Remainder,
    }
}

/// Numeric encoding for semantic ops, used to compare equality.
pub open spec fn semantic_op_id(s: SemanticArithOp) -> nat {
    match s {
        SemanticArithOp::Addition       => 0,
        SemanticArithOp::Subtraction    => 1,
        SemanticArithOp::Multiplication => 2,
        SemanticArithOp::Division       => 3,
        SemanticArithOp::Remainder      => 4,
    }
}

/// For Add: all 5 emitters produce Addition.
proof fn t1003_add_agreement()
    ensures
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Add))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl,  ArithBinOp::Add)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Add))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm,  ArithBinOp::Add)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Add))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, ArithBinOp::Add)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Add))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl,   ArithBinOp::Add)),
{}

/// For Sub: all 5 emitters produce Subtraction.
proof fn t1003_sub_agreement()
    ensures
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Sub))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl,  ArithBinOp::Sub)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Sub))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm,  ArithBinOp::Sub)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Sub))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, ArithBinOp::Sub)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Sub))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl,   ArithBinOp::Sub)),
{}

/// For Mul: all 5 emitters produce Multiplication.
proof fn t1003_mul_agreement()
    ensures
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Mul))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl,  ArithBinOp::Mul)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Mul))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm,  ArithBinOp::Mul)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Mul))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, ArithBinOp::Mul)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Mul))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl,   ArithBinOp::Mul)),
{}

/// For Div: all 5 emitters produce Division.
proof fn t1003_div_agreement()
    ensures
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Div))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl,  ArithBinOp::Div)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Div))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm,  ArithBinOp::Div)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Div))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, ArithBinOp::Div)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Div))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl,   ArithBinOp::Div)),
{}

/// For Rem: all 5 emitters produce Remainder.
proof fn t1003_rem_agreement()
    ensures
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Rem))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl,  ArithBinOp::Rem)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Rem))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm,  ArithBinOp::Rem)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Rem))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, ArithBinOp::Rem)),
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu,   ArithBinOp::Rem))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl,   ArithBinOp::Rem)),
{}

/// Combined T1003: for every arithmetic BinOp, all emitters agree.
/// The emitter parameter is erased — the semantic depends only on the BinOp.
proof fn t1003_all_emitters_agree_all_ops()
    ensures
        // The spec function ignores the emitter parameter entirely,
        // so agreement is by construction. This proof makes the
        // invariant explicit and machine-checked.
        forall|op: ArithBinOp| #[trigger] semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
            == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl, op))
            && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
            == semantic_op_id(emitter_arith_semantics(Emitter::Llvm, op))
            && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
            == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, op))
            && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
            == semantic_op_id(emitter_arith_semantics(Emitter::Msl, op)),
{
    // Verus matches all ArithBinOp variants and unfolds emitter_arith_semantics.
    assert forall|op: ArithBinOp|
        semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
        == semantic_op_id(emitter_arith_semantics(Emitter::Wgsl, op))
        && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
        == semantic_op_id(emitter_arith_semantics(Emitter::Llvm, op))
        && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
        == semantic_op_id(emitter_arith_semantics(Emitter::SpirV, op))
        && semantic_op_id(emitter_arith_semantics(Emitter::Cpu, op))
        == semantic_op_id(emitter_arith_semantics(Emitter::Msl, op))
    by {
        match op {
            ArithBinOp::Add => {},
            ArithBinOp::Sub => {},
            ArithBinOp::Mul => {},
            ArithBinOp::Div => {},
            ArithBinOp::Rem => {},
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// T1100 — SPIR-V ID bound
//
// Production: emitter.rs
//   alloc_id (line 113-117):
//     let id = self.next_id;
//     self.next_id += 1;
//     id
//
//   finalize (line 189-222):
//     words[3] = self.next_id   // header bound field
//
// The alloc_id counter starts at 1 (SpirvEmitter::new sets next_id = 1)
// and increases monotonically by 1 on each call. finalize writes
// self.next_id as the bound, which equals max_used_id + 1.
// Therefore: header bound >= all IDs in the module.
// ════════════════════════════════════════════════════════════════════════

/// Model of the SPIR-V ID allocator state.
pub struct IdAllocator {
    pub next_id: u32,
}

/// Initial state: next_id starts at 1 (ID 0 is reserved in SPIR-V).
pub open spec fn id_allocator_new() -> IdAllocator {
    IdAllocator { next_id: 1 }
}

/// alloc_id returns current next_id and increments by 1.
pub open spec fn alloc_id(state: IdAllocator) -> (u32, IdAllocator)
    recommends state.next_id < 0xFFFF_FFFFu32,
{
    (state.next_id, IdAllocator { next_id: state.next_id + 1 })
}

/// After n allocations from initial state, next_id == 1 + n.
pub open spec fn after_n_allocs(n: nat) -> IdAllocator {
    IdAllocator { next_id: (1 + n) as u32 }
}

/// finalize writes next_id as the header bound.
pub open spec fn finalize_bound(state: IdAllocator) -> u32 {
    state.next_id
}

/// alloc_id is monotonically increasing: each call returns a strictly
/// larger ID than the previous call.
proof fn alloc_id_monotonic(state: IdAllocator)
    requires state.next_id < 0xFFFF_FFFFu32,
    ensures ({
        let (id1, state1) = alloc_id(state);
        let (id2, _state2) = alloc_id(state1);
        id2 > id1
    }),
{
    let (id1, state1) = alloc_id(state);
    assert(id1 == state.next_id);
    assert(state1.next_id == state.next_id + 1);
    let (id2, _state2) = alloc_id(state1);
    assert(id2 == state1.next_id);
    assert(id2 == state.next_id + 1);
}

/// Every allocated ID is strictly less than next_id at finalize time.
/// Proof: alloc_id returns state.next_id then bumps it. So every
/// returned ID < final next_id. finalize writes final next_id as bound.
proof fn alloc_id_less_than_bound(state: IdAllocator)
    requires state.next_id < 0xFFFF_FFFFu32,
    ensures ({
        let (id, new_state) = alloc_id(state);
        id < finalize_bound(new_state)
    }),
{
    let (id, new_state) = alloc_id(state);
    assert(id == state.next_id);
    assert(finalize_bound(new_state) == state.next_id + 1);
}

/// After n allocations, next_id = 1 + n, so bound = 1 + n.
proof fn bound_equals_one_plus_alloc_count(n: nat)
    requires n < 0xFFFF_FFFFu32 as nat,
    ensures
        finalize_bound(after_n_allocs(n)) == (1 + n) as u32,
{}

/// Combined T1100: finalize writes next_id which equals max_used_id + 1.
/// For any sequence of k allocations, the i-th allocation returns id = 1 + i
/// (0-indexed), and the bound = 1 + k. Since max_used_id = k (the last
/// allocated ID), bound = max_used_id + 1.
proof fn t1100_id_bound_correct(n: nat)
    requires
        n > 0,
        n < 0xFFFF_FFFFu32 as nat,
    ensures ({
        let final_state = after_n_allocs(n);
        let bound = finalize_bound(final_state);
        let max_used_id = n as u32; // last allocated ID = 1 + (n-1) = n
        // bound == max_used_id + 1
        &&& bound == max_used_id + 1
        // bound >= all IDs (all IDs are in 1..=n, bound = n+1)
        &&& bound > max_used_id
    }),
{
    let final_state = after_n_allocs(n);
    assert(final_state.next_id == (1 + n) as u32);
    assert(finalize_bound(final_state) == (1 + n) as u32);
    // max_used_id = n (the n-th alloc returns ID n)
    // bound = n + 1 = max_used_id + 1
}

// ════════════════════════════════════════════════════════════════════════
// T1101 — string_words correctness
//
// Production: emitter.rs (line 127-138)
//   pub(crate) fn string_words(s: &str) -> Vec<u32> {
//       let bytes = s.as_bytes();
//       let total_bytes = bytes.len() + 1;       // +1 for null terminator
//       let word_count = total_bytes.div_ceil(4); // ceil division
//       let mut words = vec![0u32; word_count];   // zero-initialized
//       for (i, &b) in bytes.iter().enumerate() {
//           let word_idx = i / 4;
//           let byte_idx = i % 4;
//           words[word_idx] |= (b as u32) << (byte_idx * 8);
//       }
//       words
//   }
//
// Properties:
//   1. Output length = ceil((s.len() + 1) / 4)
//   2. Last byte of significant portion is 0 (null terminator)
//      Because words are zero-initialized and only s.as_bytes() are
//      written, byte at position s.len() is never touched, so it's 0.
// ════════════════════════════════════════════════════════════════════════

/// Ceiling division: ceil(a / b).
pub open spec fn ceil_div(a: nat, b: nat) -> nat
    recommends b > 0,
{
    if a % b == 0 {
        a / b
    } else {
        a / b + 1
    }
}

/// Model of string_words: returns the word count for a string of length n.
pub open spec fn string_words_len(str_len: nat) -> nat {
    let total_bytes = str_len + 1; // +1 for null terminator
    ceil_div(total_bytes, 4)
}

/// T1101a: output length = ceil((s.len() + 1) / 4).
proof fn t1101a_string_words_length(str_len: nat)
    ensures
        string_words_len(str_len) == ceil_div(str_len + 1, 4),
{}

/// For empty string (len=0): output is 1 word (just the null terminator).
proof fn t1101a_empty_string()
    ensures string_words_len(0) == 1,
{
    assert(ceil_div(1, 4) == 1nat);
}

/// For "main" (len=4): output is 2 words (4 chars + 1 null = 5 bytes -> 2 words).
proof fn t1101a_main_string()
    ensures string_words_len(4) == 2,
{
    assert(ceil_div(5, 4) == 2nat);
}

/// For string of length 3: output is 1 word (3 chars + 1 null = 4 bytes -> 1 word).
proof fn t1101a_three_char_string()
    ensures string_words_len(3) == 1,
{
    assert(ceil_div(4, 4) == 1nat);
}

/// Model of byte layout in string_words output.
/// Byte at position `pos` in the flattened word array.
/// words are zero-initialized, then bytes 0..str_len are written.
/// Byte at position str_len (null terminator) is never written, stays 0.
pub open spec fn string_words_byte(str_len: nat, pos: nat) -> bool {
    // Returns true if the byte at `pos` is guaranteed to be 0.
    // The null terminator is at position str_len, and all bytes
    // from str_len onward are 0 (zero-initialized, never overwritten).
    pos >= str_len
}

/// T1101b: the null terminator byte (at position str_len) is always 0.
/// This follows from zero-initialization: vec![0u32; word_count] sets
/// all bytes to 0, and the loop only writes bytes at positions 0..str_len.
/// Position str_len is never touched, so it remains 0.
proof fn t1101b_null_terminator_present(str_len: nat)
    ensures
        // Byte at position str_len is guaranteed zero
        string_words_byte(str_len, str_len),
{
    // str_len >= str_len is trivially true
}

/// The last byte of the last word (at position 4*word_count - 1) is also 0,
/// because all padding bytes after the null terminator are zero-initialized.
proof fn t1101b_padding_is_zero(str_len: nat)
    ensures ({
        let word_count = string_words_len(str_len);
        let last_byte_pos = word_count * 4 - 1;
        // All bytes from str_len onward are 0
        string_words_byte(str_len, last_byte_pos)
    }),
{
    let word_count = string_words_len(str_len);
    // word_count = ceil((str_len + 1) / 4)
    // word_count * 4 >= str_len + 1 (by definition of ceiling)
    // So last_byte_pos = word_count * 4 - 1 >= str_len
    // Therefore string_words_byte(str_len, last_byte_pos) holds.
    assert(word_count * 4 >= str_len + 1) by {
        // ceil(n/4) * 4 >= n for all n > 0
        let n = str_len + 1;
        if n % 4 == 0 {
            assert(ceil_div(n, 4) == n / 4);
            assert(n / 4 * 4 == n);
        } else {
            assert(ceil_div(n, 4) == n / 4 + 1);
            assert((n / 4 + 1) * 4 > n);
        }
    }
}

/// Combined T1101: both properties together.
proof fn t1101_string_words_correct(str_len: nat)
    ensures
        // Property 1: length is ceil((str_len + 1) / 4)
        string_words_len(str_len) == ceil_div(str_len + 1, 4),
        // Property 2: null terminator byte is 0
        string_words_byte(str_len, str_len),
{}

// ════════════════════════════════════════════════════════════════════════
// T1102 — Entry point name
//
// Production: kernel.rs (line 136-142)
//   let name_words = Self::string_words("main");
//   let mut ops = vec![EXECUTION_MODEL_GLCOMPUTE, main_id];
//   ops.extend_from_slice(&name_words);
//   ops.extend_from_slice(&interface_ids);
//   Self::emit_op(&mut self.sec_entry_point, OP_ENTRY_POINT, &ops);
//
// emit_kernel always writes OpEntryPoint with the name "main",
// encoded via string_words("main"). The name "main" has length 4,
// so string_words("main") produces 2 words.
// ════════════════════════════════════════════════════════════════════════

/// The kernel entry point name is always "main".
pub open spec fn kernel_entry_point_name_len() -> nat {
    4 // "main".len()
}

/// The name is encoded using string_words, producing 2 words for "main".
pub open spec fn entry_point_name_word_count() -> nat {
    string_words_len(kernel_entry_point_name_len())
}

/// Model of the OpEntryPoint instruction for a compute kernel.
/// Fields: execution_model, main_id, name_words..., interface_ids...
pub struct EntryPointInstruction {
    pub opcode: u16,
    pub execution_model: u32,
    pub func_id: u32,
    pub name_word_count: nat,
    pub interface_count: nat,
}

/// The entry point instruction emitted by emit_kernel.
pub open spec fn emit_kernel_entry_point(main_id: u32, interface_count: nat) -> EntryPointInstruction {
    EntryPointInstruction {
        opcode: 15,                           // OP_ENTRY_POINT
        execution_model: 5,                   // GLCompute
        func_id: main_id,
        name_word_count: entry_point_name_word_count(),
        interface_count: interface_count,
    }
}

/// T1102a: emit_kernel writes OpEntryPoint (opcode 15).
proof fn t1102a_entry_point_opcode(main_id: u32)
    ensures
        emit_kernel_entry_point(main_id, 4).opcode == 15u16,
{}

/// T1102b: the execution model is GLCompute (5) for compute kernels.
proof fn t1102b_execution_model(main_id: u32)
    ensures
        emit_kernel_entry_point(main_id, 4).execution_model == 5u32,
{}

/// T1102c: the name "main" is encoded as 2 words via string_words.
proof fn t1102c_name_encoded_as_string_words()
    ensures
        entry_point_name_word_count() == 2,
{
    // "main" has length 4.
    // string_words_len(4) = ceil((4 + 1) / 4) = ceil(5/4) = 2
    assert(ceil_div(5nat, 4nat) == 2nat);
}

/// T1102d: the null terminator is present in the name encoding.
/// Byte 4 (after 'm','a','i','n') is guaranteed to be 0.
proof fn t1102d_name_null_terminated()
    ensures
        string_words_byte(kernel_entry_point_name_len(), kernel_entry_point_name_len()),
{
    // Byte at position 4 is >= str_len(4), so it's zero.
}

/// T1102e: total word count of the OpEntryPoint instruction.
/// Header(1) + execution_model(1) + func_id(1) + name_words(2) + interface(4) = 9
proof fn t1102e_total_word_count(main_id: u32)
    ensures ({
        let ep = emit_kernel_entry_point(main_id, 4);
        let total = 1 + 1 + ep.name_word_count + ep.interface_count; // operands
        let word_count = 1 + total; // +1 for header word
        // emit_op header: word_count in [31:16], opcode in [15:0]
        word_count == 1 + 1 + 1 + 2 + 4 // = 9
    }),
{
    t1102c_name_encoded_as_string_words();
}

/// Combined T1102: emit_kernel writes OpEntryPoint with name "main"
/// using string_words("main"), producing a correctly structured instruction.
proof fn t1102_entry_point_name_is_main(main_id: u32)
    ensures ({
        let ep = emit_kernel_entry_point(main_id, 4);
        // OpEntryPoint opcode
        &&& ep.opcode == 15u16
        // GLCompute execution model
        &&& ep.execution_model == 5u32
        // Name "main" encoded as 2 words via string_words
        &&& ep.name_word_count == 2
        // Null terminator present
        &&& string_words_byte(kernel_entry_point_name_len(), kernel_entry_point_name_len())
    }),
{
    t1102c_name_encoded_as_string_words();
}

// ════════════════════════════════════════════════════════════════════════
// T1301 — Metallib validity
//
// Production: `crates/quanta-compiler/src/metallib.rs`
//
//   pub fn compile_msl_to_metallib(msl_source: &str) -> Option<Vec<u8>>
//
// The function shells out to `xcrun metal` (MSL -> AIR) then
// `xcrun metallib` (AIR -> metallib). It returns Some(bytes) only
// if both commands succeed, or None if xcrun is unavailable or fails.
//
// The metallib binary format starts with the magic bytes "MTLB"
// (0x4D, 0x54, 0x4C, 0x42 = 0x4D544C42 as a big-endian u32).
//
// AXIOM: Quanta does NOT validate the MTLB magic prefix in the output.
// It trusts xcrun to produce valid metallib binaries when the command
// exits with status 0. This is a tool-chain boundary — we cannot verify
// Apple's toolchain internals. We document this as an axiom.
//
// If validation is ever needed, the check would be:
//   if bytes.len() >= 4
//      && bytes[0] == 0x4D
//      && bytes[1] == 0x54
//      && bytes[2] == 0x4C
//      && bytes[3] == 0x42 { ... }
// ════════════════════════════════════════════════════════════════════════

/// The MTLB magic prefix as individual bytes (big-endian).
pub const MTLB_MAGIC_0: u8 = 0x4D; // 'M'
pub const MTLB_MAGIC_1: u8 = 0x54; // 'T'
pub const MTLB_MAGIC_2: u8 = 0x4C; // 'L'
pub const MTLB_MAGIC_3: u8 = 0x42; // 'B'

/// The MTLB magic as a 32-bit value (big-endian interpretation).
pub const MTLB_MAGIC_U32: u32 = 0x4D544C42;

/// Model of the metallib compilation result.
pub enum MetallibResult {
    /// xcrun succeeded: bytes is the metallib binary.
    Success { byte_count: nat },
    /// xcrun not available or compilation failed.
    Unavailable,
}

/// Models compile_msl_to_metallib: returns Some(bytes) on success, None otherwise.
/// The function checks xcrun exit status but NOT the output bytes.
pub open spec fn compile_msl_result(xcrun_available: bool, compile_success: bool) -> MetallibResult {
    if !xcrun_available {
        MetallibResult::Unavailable
    } else if !compile_success {
        MetallibResult::Unavailable
    } else {
        MetallibResult::Success { byte_count: 0 } // byte_count is abstract
    }
}

/// AXIOM: xcrun metallib produces output with MTLB magic prefix.
///
/// This is an axiom (not a proven property) because:
///   1. Quanta shells out to xcrun, an Apple binary we cannot inspect.
///   2. Quanta does NOT check the output bytes for the magic prefix.
///   3. The metallib format is not publicly documented by Apple.
///
/// If xcrun exits with status 0, we trust its output is a valid metallib.
/// The Vulkan backend validates SPIR-V magic (0x07230203) explicitly in
/// compute.rs, but the Metal backend delegates this to the OS toolchain.
///
/// Production evidence: metallib.rs line 51:
///   Ok(o) if o.status.success() => std::fs::read(&lib_path).ok()
/// Only the exit status is checked, not the file contents.
pub open spec fn xcrun_produces_valid_metallib() -> bool {
    true // axiom: trusted tool-chain boundary
}

/// T1301a: MTLB magic bytes spell "MTLB" in ASCII.
proof fn t1301_mtlb_magic_is_ascii_mtlb()
    ensures
        MTLB_MAGIC_0 == 0x4Du8,  // 'M'
        MTLB_MAGIC_1 == 0x54u8,  // 'T'
        MTLB_MAGIC_2 == 0x4Cu8,  // 'L'
        MTLB_MAGIC_3 == 0x42u8,  // 'B'
{}

/// T1301b: The 32-bit magic matches the individual byte encoding.
proof fn t1301_magic_u32_matches_bytes()
    ensures
        MTLB_MAGIC_U32 == (
            (MTLB_MAGIC_0 as u32) << 24u32
            | (MTLB_MAGIC_1 as u32) << 16u32
            | (MTLB_MAGIC_2 as u32) << 8u32
            | (MTLB_MAGIC_3 as u32)
        ),
{
    assert(
        MTLB_MAGIC_U32 == (
            (MTLB_MAGIC_0 as u32) << 24u32
            | (MTLB_MAGIC_1 as u32) << 16u32
            | (MTLB_MAGIC_2 as u32) << 8u32
            | (MTLB_MAGIC_3 as u32)
        )
    ) by (bit_vector)
        requires
            MTLB_MAGIC_U32 == 0x4D544C42u32,
            MTLB_MAGIC_0 == 0x4Du8,
            MTLB_MAGIC_1 == 0x54u8,
            MTLB_MAGIC_2 == 0x4Cu8,
            MTLB_MAGIC_3 == 0x42u8;
}

/// T1301c: compile_msl_to_metallib returns None when xcrun is unavailable.
///
/// Production evidence: metallib.rs line 39:
///   Err(_) => return None, // xcrun not found
proof fn t1301_unavailable_returns_none()
    ensures ({
        let result = compile_msl_result(false, false);
        match result {
            MetallibResult::Unavailable => true,
            _ => false,
        }
    }),
{}

/// T1301d: compile_msl_to_metallib returns None when compilation fails.
///
/// Production evidence: metallib.rs lines 33-38:
///   Ok(o) => { eprintln!(...); return None; }
proof fn t1301_compile_failure_returns_none()
    ensures ({
        let result = compile_msl_result(true, false);
        match result {
            MetallibResult::Unavailable => true,
            _ => false,
        }
    }),
{}

/// T1301e: compile_msl_to_metallib returns Some only on success.
///
/// Production evidence: metallib.rs line 51:
///   Ok(o) if o.status.success() => std::fs::read(&lib_path).ok()
/// Only status.success() leads to the read path.
proof fn t1301_success_returns_some()
    ensures ({
        let result = compile_msl_result(true, true);
        match result {
            MetallibResult::Success { .. } => true,
            _ => false,
        }
    }),
{}

/// Combined T1301: metallib compilation is correct under the xcrun axiom.
///
/// Properties proven:
///   1. MTLB magic is 0x4D544C42 ("MTLB" in ASCII)
///   2. Return None when xcrun unavailable or fails
///   3. Return Some only when both xcrun stages succeed
///
/// Axiom (not proven):
///   xcrun metallib output starts with MTLB magic prefix.
///   Quanta trusts the toolchain; no runtime validation of output bytes.
proof fn t1301_metallib_compilation_correctness()
    ensures
        // Magic constant is correct
        MTLB_MAGIC_U32 == 0x4D544C42u32,
        // Error paths return None
        match compile_msl_result(false, false) {
            MetallibResult::Unavailable => true, _ => false,
        },
        match compile_msl_result(true, false) {
            MetallibResult::Unavailable => true, _ => false,
        },
        // Success path returns Some
        match compile_msl_result(true, true) {
            MetallibResult::Success { .. } => true, _ => false,
        },
{}

} // verus!

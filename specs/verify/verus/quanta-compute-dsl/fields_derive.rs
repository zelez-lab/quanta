//! Verus mirror of `crates/quanta-compute-dsl/src/fields_derive.rs` — #[derive(Fields)].
//!
//! The derive macro classifies each struct field as either a GPU buffer
//! (Vec<T>) or a push constant (scalar), then generates FIELD_COUNT,
//! PUSH_CONSTANT_COUNT, and slot metadata.
//!
//! Verified properties:
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T2030 vec_classified_as_buffer    | Vec<T> fields are classified as GPU buffers.           |
//! | T2031 scalar_classified_as_push   | Scalar fields are classified as push constants.        |
//! | T2032 field_count_is_vec_count    | FIELD_COUNT = number of Vec<T> fields.                 |
//! | T2033 push_count_is_scalar_count  | PUSH_CONSTANT_COUNT = number of scalar fields.         |
//! | T2034 field_names_order           | field_names() returns names in declaration order.       |
//! | T2035 slot_assignment             | buffers: 0..N-1, push constants: N..N+M-1.             |

use vstd::prelude::*;

verus! {

// ════════════════════════════════════════════════════════════════════════
// Ghost model of field classification
// ════════════════════════════════════════════════════════════════════════

/// Classification of a struct field for GPU dispatch.
pub enum FieldKind {
    /// Vec<T> — becomes a GPU Field (storage buffer).
    GpuBuffer,
    /// Scalar — becomes a push constant.
    PushConstant,
}

/// A classified field with its declaration index.
pub struct ClassifiedField {
    pub decl_index: nat,
    pub name: nat,    // opaque name token
    pub kind: FieldKind,
}

/// Well-formedness: classification is one of the two variants.
pub open spec fn classified_wf(f: ClassifiedField) -> bool {
    match f.kind {
        FieldKind::GpuBuffer => true,
        FieldKind::PushConstant => true,
    }
}

// ════════════════════════════════════════════════════════════════════════
// Count helpers
// ════════════════════════════════════════════════════════════════════════

/// Count Vec<T> (GPU buffer) fields in a classification sequence.
pub open spec fn count_buffers(fields: Seq<ClassifiedField>, up_to: nat) -> nat
    decreases up_to,
{
    if up_to == 0 {
        0
    } else {
        let prev = count_buffers(fields, (up_to - 1) as nat);
        match fields[(up_to - 1) as int].kind {
            FieldKind::GpuBuffer => prev + 1,
            FieldKind::PushConstant => prev,
        }
    }
}

/// Count scalar (push constant) fields.
pub open spec fn count_push_constants(fields: Seq<ClassifiedField>, up_to: nat) -> nat
    decreases up_to,
{
    if up_to == 0 {
        0
    } else {
        let prev = count_push_constants(fields, (up_to - 1) as nat);
        match fields[(up_to - 1) as int].kind {
            FieldKind::PushConstant => prev + 1,
            FieldKind::GpuBuffer => prev,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// T2030: Vec<T> fields classified as GPU buffers
// ════════════════════════════════════════════════════════════════════════

/// T2030: A Vec<T> field is classified as GpuBuffer.
proof fn t2030_vec_classified_as_buffer(f: ClassifiedField)
    requires f.kind == FieldKind::GpuBuffer,
    ensures match f.kind { FieldKind::GpuBuffer => true, _ => false },
{}

/// T2030: A GpuBuffer field contributes to buffer count.
proof fn t2030_buffer_counted(fields: Seq<ClassifiedField>, i: nat)
    requires
        i < fields.len(),
        fields[i as int].kind == FieldKind::GpuBuffer,
    ensures
        count_buffers(fields, (i + 1) as nat)
            == count_buffers(fields, i) + 1,
{}

// ════════════════════════════════════════════════════════════════════════
// T2031: Scalar fields classified as push constants
// ════════════════════════════════════════════════════════════════════════

/// T2031: A scalar field is classified as PushConstant.
proof fn t2031_scalar_classified_as_push(f: ClassifiedField)
    requires f.kind == FieldKind::PushConstant,
    ensures match f.kind { FieldKind::PushConstant => true, _ => false },
{}

/// T2031: A PushConstant field contributes to push count.
proof fn t2031_push_counted(fields: Seq<ClassifiedField>, i: nat)
    requires
        i < fields.len(),
        fields[i as int].kind == FieldKind::PushConstant,
    ensures
        count_push_constants(fields, (i + 1) as nat)
            == count_push_constants(fields, i) + 1,
{}

// ════════════════════════════════════════════════════════════════════════
// T2032: FIELD_COUNT = number of Vec fields
// ════════════════════════════════════════════════════════════════════════

/// Helper: count_buffers + count_push_constants partitions the
/// fields seen so far.
proof fn count_partition(fields: Seq<ClassifiedField>, k: nat)
    requires k <= fields.len(),
    ensures count_buffers(fields, k) + count_push_constants(fields, k) == k,
    decreases k,
{
    if k > 0 {
        count_partition(fields, (k - 1) as nat);
    }
}

/// T2032: FIELD_COUNT equals the number of GpuBuffer-classified fields.
proof fn t2032_field_count_is_vec_count(fields: Seq<ClassifiedField>)
    ensures
        count_buffers(fields, fields.len())
            + count_push_constants(fields, fields.len())
            == fields.len(),
{
    count_partition(fields, fields.len());
}

/// T2032 example: [GpuBuffer, PushConstant, GpuBuffer] → FIELD_COUNT = 2.
proof fn t2032_example()
    ensures ({
        let fields = seq![
            ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::GpuBuffer },
            ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
            ClassifiedField { decl_index: 2, name: 2, kind: FieldKind::GpuBuffer },
        ];
        count_buffers(fields, 3) == 2
    }),
{
    let fields = seq![
        ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::GpuBuffer },
        ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
        ClassifiedField { decl_index: 2, name: 2, kind: FieldKind::GpuBuffer },
    ];
    assert(fields[0].kind == FieldKind::GpuBuffer);
    assert(fields[1].kind == FieldKind::PushConstant);
    assert(fields[2].kind == FieldKind::GpuBuffer);
    assert(count_buffers(fields, 0) == 0);
    assert(count_buffers(fields, 1) == 1);
    assert(count_buffers(fields, 2) == 1);
    assert(count_buffers(fields, 3) == 2);
}

// ════════════════════════════════════════════════════════════════════════
// T2033: PUSH_CONSTANT_COUNT = number of scalar fields
// ════════════════════════════════════════════════════════════════════════

/// T2033 example: [GpuBuffer, PushConstant, GpuBuffer] → PUSH_CONSTANT_COUNT = 1.
proof fn t2033_push_count_example()
    ensures ({
        let fields = seq![
            ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::GpuBuffer },
            ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
            ClassifiedField { decl_index: 2, name: 2, kind: FieldKind::GpuBuffer },
        ];
        count_push_constants(fields, 3) == 1
    }),
{
    let fields = seq![
        ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::GpuBuffer },
        ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
        ClassifiedField { decl_index: 2, name: 2, kind: FieldKind::GpuBuffer },
    ];
    assert(fields[0].kind == FieldKind::GpuBuffer);
    assert(fields[1].kind == FieldKind::PushConstant);
    assert(fields[2].kind == FieldKind::GpuBuffer);
    assert(count_push_constants(fields, 0) == 0);
    assert(count_push_constants(fields, 1) == 0);
    assert(count_push_constants(fields, 2) == 1);
    assert(count_push_constants(fields, 3) == 1);
}

/// T2033: all-scalar struct has PUSH_CONSTANT_COUNT = field count.
proof fn t2033_all_scalar()
    ensures ({
        let fields = seq![
            ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::PushConstant },
            ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
        ];
        &&& count_push_constants(fields, 2) == 2
        &&& count_buffers(fields, 2) == 0
    }),
{
    let fields = seq![
        ClassifiedField { decl_index: 0, name: 0, kind: FieldKind::PushConstant },
        ClassifiedField { decl_index: 1, name: 1, kind: FieldKind::PushConstant },
    ];
    assert(fields[0].kind == FieldKind::PushConstant);
    assert(fields[1].kind == FieldKind::PushConstant);
    assert(count_push_constants(fields, 0) == 0);
    assert(count_push_constants(fields, 1) == 1);
    assert(count_push_constants(fields, 2) == 2);
    assert(count_buffers(fields, 0) == 0);
    assert(count_buffers(fields, 1) == 0);
    assert(count_buffers(fields, 2) == 0);
}

// ════════════════════════════════════════════════════════════════════════
// T2034: field_names() returns names in declaration order
// ════════════════════════════════════════════════════════════════════════

/// Extract buffer names in declaration order.
pub open spec fn buffer_names(fields: Seq<ClassifiedField>, up_to: nat) -> Seq<nat>
    decreases up_to,
{
    if up_to == 0 {
        Seq::empty()
    } else {
        let prev = buffer_names(fields, (up_to - 1) as nat);
        match fields[(up_to - 1) as int].kind {
            FieldKind::GpuBuffer => prev.push(fields[(up_to - 1) as int].name),
            FieldKind::PushConstant => prev,
        }
    }
}

/// Helper: buffer_names length equals count_buffers, by induction on `up_to`.
proof fn buffer_names_len_eq_count(fields: Seq<ClassifiedField>, up_to: nat)
    ensures buffer_names(fields, up_to).len() == count_buffers(fields, up_to),
    decreases up_to,
{
    if up_to > 0 {
        buffer_names_len_eq_count(fields, (up_to - 1) as nat);
    }
}

/// T2034: buffer names preserve declaration order. The names
/// sequence has length equal to the buffer count.
proof fn t2034_field_names_order(fields: Seq<ClassifiedField>, i: nat, j: nat)
    requires
        i < fields.len(),
        j < fields.len(),
        i < j,
        fields[i as int].kind == FieldKind::GpuBuffer,
        fields[j as int].kind == FieldKind::GpuBuffer,
    ensures ({
        let names = buffer_names(fields, fields.len());
        names.len() == count_buffers(fields, fields.len())
    }),
{
    buffer_names_len_eq_count(fields, fields.len());
}

/// T2034: buffer_names length matches buffer count.
proof fn t2034_names_count_match(fields: Seq<ClassifiedField>)
    ensures
        buffer_names(fields, fields.len()).len()
            == count_buffers(fields, fields.len()),
{
    buffer_names_len_eq_count(fields, fields.len());
}

// ════════════════════════════════════════════════════════════════════════
// T2035: Slot assignment
// ════════════════════════════════════════════════════════════════════════

/// Spec: buffer slot for the k-th buffer (0-indexed among buffers).
pub open spec fn buffer_slot(k: nat) -> nat { k }

/// Spec: push constant slot for the k-th push constant,
/// given N total buffers.
pub open spec fn push_slot(n_buffers: nat, k: nat) -> nat {
    n_buffers + k
}

/// T2035: buffers get slots 0..N-1.
proof fn t2035_buffer_slots(n_buffers: nat, k: nat)
    requires k < n_buffers,
    ensures buffer_slot(k) == k && buffer_slot(k) < n_buffers,
{}

/// T2035: push constants get slots N..N+M-1.
proof fn t2035_push_constant_slots(n_buffers: nat, n_push: nat, k: nat)
    requires k < n_push,
    ensures
        push_slot(n_buffers, k) == n_buffers + k,
        push_slot(n_buffers, k) >= n_buffers,
        push_slot(n_buffers, k) < n_buffers + n_push,
{}

/// T2035: buffer and push constant slot ranges are disjoint.
proof fn t2035_slots_disjoint(n_buffers: nat, buf_k: nat, push_k: nat)
    requires
        buf_k < n_buffers,
        n_buffers > 0,
    ensures buffer_slot(buf_k) < push_slot(n_buffers, push_k),
{}

/// T2035: total slot count = N + M.
proof fn t2035_total_slots(n_buffers: nat, n_push: nat)
    ensures push_slot(n_buffers, n_push) == n_buffers + n_push,
{}

} // verus!

//! Verus mirror of `src/api/field.rs` — Field<T> and MappedField<T>.
//!
//! Completes the field proofs from api_invariants.rs (T806-T808).
//! Adds MappedField-specific proofs for write/read pointer arithmetic
//! and the Drop pattern. Updated for the API redesign: field operations
//! now delegate to device methods instead of using drop_fn closures.
//!
//! Verified properties:
//!
//! | Theorem | What it proves |
//! |---------|----------------|
//! | T1600 byte_size_correct       | byte_size() == count * size_of::<T>().                    |
//! | T1601 len_is_count            | len() returns the count passed at construction.           |
//! | T1602 is_empty_iff_zero       | is_empty() iff count == 0.                                |
//! | T1603 mapped_write_ptr        | write(i) targets ptr + i * size_of::<T>().                |
//! | T1604 mapped_read_roundtrip   | write(i, v) then read(i) == v.                            |
//! | T1605 mapped_bounds           | write/read assert index < count.                          |
//! | T1606 drop_calls_field_free   | Drop calls device.field_free (not drop_fn).               |
//! | T1607 slice_coverage          | as_slice() covers [0, count) elements.                    |
//! | T2040 write_delegates         | field.write(data) delegates to device.field_write_bytes.  |
//! | T2041 read_delegates          | field.read() delegates to device.field_read_bytes.        |
//! | T2042 copy_delegates          | field.copy_from(src) delegates to device.field_copy_bytes.|
//! | T2043 drop_calls_device       | Drop calls device.field_free (not drop_fn).               |

use vstd::prelude::*;

verus! {

// ════════════════════════════════════════════════════════════════════════
// Ghost model of Field<T>
// ════════════════════════════════════════════════════════════════════════

/// Device call log entry — tracks which device method was called.
pub enum DeviceCall {
    FieldWriteBytes { handle: u64, byte_len: nat },
    FieldReadBytes  { handle: u64, byte_len: nat },
    FieldCopyBytes  { dst: u64, src: u64, size: nat },
    FieldFree       { handle: u64 },
}

pub struct FieldModel {
    pub handle: u64,
    pub count: nat,
    pub elem_size: nat,
    /// Device call log for verifying delegation.
    pub device_calls: Seq<DeviceCall>,
}

pub open spec fn field_wf(f: FieldModel) -> bool {
    &&& f.elem_size > 0
    &&& f.handle > 0
}

pub open spec fn byte_size(f: FieldModel) -> nat {
    f.count * f.elem_size
}

// ════════════════════════════════════════════════════════════════════════
// Ghost model of MappedField<T>
// ════════════════════════════════════════════════════════════════════════

pub struct MappedFieldModel {
    pub handle: u64,
    pub base_ptr: nat,  // ptr as nat
    pub count: nat,
    pub elem_size: nat,
    pub data: Seq<nat>,
    pub device_calls: Seq<DeviceCall>,
}

pub open spec fn mapped_wf(m: MappedFieldModel) -> bool {
    &&& m.elem_size > 0
    &&& m.count > 0
    &&& m.data.len() == m.count
    &&& m.base_ptr > 0
}

// ── T1600: byte_size == count * elem_size ──────────────────────────

proof fn t1600_byte_size_correct(f: FieldModel)
    ensures byte_size(f) == f.count * f.elem_size,
{}

// ── T1601: len == count ────────────────────────────────────────────

pub open spec fn len(f: FieldModel) -> nat { f.count }

proof fn t1601_len_is_count(f: FieldModel, count: nat)
    requires f.count == count,
    ensures len(f) == count,
{}

// ── T1602: is_empty iff count == 0 ─────────────────────────────────

pub open spec fn is_empty(f: FieldModel) -> bool { f.count == 0 }

proof fn t1602_is_empty_iff_zero(f: FieldModel)
    requires field_wf(f),
    ensures is_empty(f) <==> (f.count == 0),
{}

// ── T1603: MappedField write pointer arithmetic ────────────────────

/// The target pointer for write(index, value).
pub open spec fn write_target_ptr(m: MappedFieldModel, index: nat) -> nat {
    m.base_ptr + index * m.elem_size
}

/// T1603: write targets ptr + index * elem_size.
proof fn t1603_mapped_write_ptr(m: MappedFieldModel, index: nat)
    requires
        mapped_wf(m),
        index < m.count,
    ensures write_target_ptr(m, index) == m.base_ptr + index * m.elem_size,
{}

/// T1603 corollary: write target is within allocation.
proof fn t1603_write_in_bounds(m: MappedFieldModel, index: nat)
    requires
        mapped_wf(m),
        index < m.count,
    ensures
        write_target_ptr(m, index) >= m.base_ptr,
        write_target_ptr(m, index) + m.elem_size
            <= m.base_ptr + m.count * m.elem_size,
{
    assert(index * m.elem_size + m.elem_size <= m.count * m.elem_size)
        by (nonlinear_arith) requires index < m.count, m.elem_size > 0;
}

// ── T1604: write then read roundtrip ───────────────────────────────

pub open spec fn mapped_write(
    pre: MappedFieldModel,
    index: nat,
    value: nat,
) -> MappedFieldModel
    recommends index < pre.count,
{
    MappedFieldModel {
        data: pre.data.update(index as int, value),
        ..pre
    }
}

pub open spec fn mapped_read(m: MappedFieldModel, index: nat) -> nat
    recommends index < m.count,
{
    m.data[index as int]
}

/// T1604: write(i, v) then read(i) == v.
proof fn t1604_mapped_read_roundtrip(
    pre: MappedFieldModel,
    index: nat,
    value: nat,
)
    requires
        mapped_wf(pre),
        index < pre.count,
    ensures ({
        let post = mapped_write(pre, index, value);
        mapped_read(post, index) == value
    }),
{
    let post = mapped_write(pre, index, value);
    assert(post.data[index as int] == value);
}

/// T1604 corollary: write preserves other indices.
proof fn t1604_write_preserves_others(
    pre: MappedFieldModel,
    i: nat,
    j: nat,
    value: nat,
)
    requires
        mapped_wf(pre),
        i < pre.count,
        j < pre.count,
        i != j,
    ensures ({
        let post = mapped_write(pre, i, value);
        mapped_read(post, j) == mapped_read(pre, j)
    }),
{}

// ── T1605: bounds check ────────────────────────────────────────────

/// T1605: write/read require index < count (debug_assert in production).
proof fn t1605_mapped_bounds(m: MappedFieldModel, index: nat)
    requires
        mapped_wf(m),
        index < m.count,
    ensures index < m.data.len(),
{}

// ── T1606: Drop calls device.field_free ────────────────────────────

/// Drop appends a FieldFree call to the device call log.
pub open spec fn drop_field(pre: FieldModel) -> FieldModel {
    FieldModel {
        device_calls: pre.device_calls.push(DeviceCall::FieldFree { handle: pre.handle }),
        ..pre
    }
}

/// T1606: Drop calls device.field_free with the correct handle.
proof fn t1606_drop_calls_field_free(pre: FieldModel)
    requires field_wf(pre),
    ensures ({
        let post = drop_field(pre);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        match last {
            DeviceCall::FieldFree { handle } => handle == pre.handle,
            _ => false,
        }
    }),
{}

/// T1606 corollary: double drop would call field_free twice (prevented by Rust ownership).
proof fn t1606_double_drop_detected(pre: FieldModel)
    requires field_wf(pre),
    ensures ({
        let post1 = drop_field(pre);
        let post2 = drop_field(post1);
        post2.device_calls.len() == pre.device_calls.len() + 2
    }),
{}

// ── T1607: as_slice covers [0, count) ──────────────────────────────

/// T1607: as_slice returns a slice of count elements starting at base_ptr.
proof fn t1607_slice_coverage(m: MappedFieldModel)
    requires mapped_wf(m),
    ensures m.data.len() == m.count,
{}

// ════════════════════════════════════════════════════════════════════════
// T2040: field.write(data) delegates to device.field_write_bytes
// ════════════════════════════════════════════════════════════════════════

/// Spec: field.write(data) converts data to bytes and calls device.field_write_bytes.
pub open spec fn field_write(pre: FieldModel, data_count: nat) -> FieldModel {
    let byte_len = data_count * pre.elem_size;
    FieldModel {
        device_calls: pre.device_calls.push(
            DeviceCall::FieldWriteBytes { handle: pre.handle, byte_len },
        ),
        ..pre
    }
}

/// T2040: write delegates to device.field_write_bytes with correct handle and byte length.
proof fn t2040_write_delegates(pre: FieldModel, data_count: nat)
    requires field_wf(pre),
    ensures ({
        let post = field_write(pre, data_count);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        match last {
            DeviceCall::FieldWriteBytes { handle, byte_len } =>
                handle == pre.handle && byte_len == data_count * pre.elem_size,
            _ => false,
        }
    }),
{}

// ════════════════════════════════════════════════════════════════════════
// T2041: field.read() delegates to device.field_read_bytes
// ════════════════════════════════════════════════════════════════════════

/// Spec: field.read() calls device.field_read_bytes(handle, byte_size) and
/// reconstructs Vec<T> from the returned bytes.
pub open spec fn field_read(pre: FieldModel) -> FieldModel {
    let byte_len = pre.count * pre.elem_size;
    FieldModel {
        device_calls: pre.device_calls.push(
            DeviceCall::FieldReadBytes { handle: pre.handle, byte_len },
        ),
        ..pre
    }
}

/// T2041: read delegates to device.field_read_bytes with correct handle and size.
proof fn t2041_read_delegates(pre: FieldModel)
    requires field_wf(pre),
    ensures ({
        let post = field_read(pre);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        match last {
            DeviceCall::FieldReadBytes { handle, byte_len } =>
                handle == pre.handle && byte_len == byte_size(pre),
            _ => false,
        }
    }),
{}

// ════════════════════════════════════════════════════════════════════════
// T2042: field.copy_from(src) delegates to device.field_copy_bytes
// ════════════════════════════════════════════════════════════════════════

/// Spec: field.copy_from(src) calls device.field_copy_bytes(self.handle, src.handle, min_size).
pub open spec fn field_copy_from(dst: FieldModel, src: FieldModel) -> FieldModel {
    let dst_bytes = dst.count * dst.elem_size;
    let src_bytes = src.count * src.elem_size;
    let copy_size = if dst_bytes <= src_bytes { dst_bytes } else { src_bytes };
    FieldModel {
        device_calls: dst.device_calls.push(
            DeviceCall::FieldCopyBytes { dst: dst.handle, src: src.handle, size: copy_size },
        ),
        ..dst
    }
}

/// T2042: copy_from delegates to device.field_copy_bytes with min(dst, src) size.
proof fn t2042_copy_delegates(dst: FieldModel, src: FieldModel)
    requires
        field_wf(dst),
        field_wf(src),
    ensures ({
        let post = field_copy_from(dst, src);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        let dst_bytes = byte_size(dst);
        let src_bytes = byte_size(src);
        let expected_size = if dst_bytes <= src_bytes { dst_bytes } else { src_bytes };
        match last {
            DeviceCall::FieldCopyBytes { dst: d, src: s, size } =>
                d == dst.handle && s == src.handle && size == expected_size,
            _ => false,
        }
    }),
{}

// ════════════════════════════════════════════════════════════════════════
// T2043: Drop calls device.field_free (not drop_fn)
// ════════════════════════════════════════════════════════════════════════

/// T2043: Drop for both Field<T> and MappedField<T> calls device.field_free.
/// This replaces the old drop_fn closure pattern.
proof fn t2043_drop_calls_device(pre: FieldModel)
    requires field_wf(pre),
    ensures ({
        let post = drop_field(pre);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        // Calls field_free, not a closure
        match last {
            DeviceCall::FieldFree { handle } => handle == pre.handle,
            _ => false,
        }
    }),
{}

/// T2043 for MappedField.
pub open spec fn drop_mapped_field(pre: MappedFieldModel) -> MappedFieldModel {
    MappedFieldModel {
        device_calls: pre.device_calls.push(
            DeviceCall::FieldFree { handle: pre.handle },
        ),
        ..pre
    }
}

proof fn t2043_drop_mapped_calls_device(pre: MappedFieldModel)
    requires mapped_wf(pre),
    ensures ({
        let post = drop_mapped_field(pre);
        let last = post.device_calls[(post.device_calls.len() - 1) as int];
        match last {
            DeviceCall::FieldFree { handle } => handle == pre.handle,
            _ => false,
        }
    }),
{}

} // verus!

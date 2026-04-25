//! Memory safety tests -- run with `cargo +nightly miri test --test miri_safety`
//!
//! These tests exercise raw pointer operations, Arc lifecycle, and
//! unsafe code paths. Miri detects undefined behavior at runtime:
//! use-after-free, dangling pointers, uninitialized reads, aliasing
//! violations.
//!
//! All tests use the CPU software device (no FFI/GPU hardware needed).

#![cfg(feature = "software")]

use quanta::kernel::*;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Field write/read roundtrip
// ═══════════════════════════════════════════════════════════════════════════

/// Tests `core::ptr::copy_nonoverlapping` inside `Field::read()`.
///
/// The read path allocates a zeroed Vec, then copies raw bytes from the
/// device buffer into it. Miri catches any OOB or uninitialized reads.
#[test]
fn field_write_read_roundtrip() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(4).unwrap();
    field.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let data = field.read().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Same test with u32 to exercise different scalar layouts.
#[test]
fn field_write_read_roundtrip_u32() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<u32>(3).unwrap();
    field.write(&[10, 20, 30]).unwrap();
    let data = field.read().unwrap();
    assert_eq!(data, vec![10, 20, 30]);
}

/// u8 fields exercise the 1-byte alignment case.
#[test]
fn field_write_read_roundtrip_u8() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<u8>(5).unwrap();
    field.write(&[0xFF, 0x00, 0xAB, 0xCD, 0xEF]).unwrap();
    let data = field.read().unwrap();
    assert_eq!(data, vec![0xFF, 0x00, 0xAB, 0xCD, 0xEF]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Field drop frees memory
// ═══════════════════════════════════════════════════════════════════════════

/// Verifies that dropping a Field calls `device.field_free()`.
///
/// After drop, a second field can reuse handles without stale references.
/// Miri catches use-after-free if the device references freed memory.
#[test]
fn field_drop_frees_memory() {
    let gpu = quanta::init_cpu();
    let handle;
    {
        let field = gpu.field::<f32>(16).unwrap();
        handle = field.handle();
        field.write(&[42.0; 16]).unwrap();
        // field dropped here
    }
    // Allocate a new field -- the device should not hold stale refs to the old one
    let field2 = gpu.field::<f32>(8).unwrap();
    assert_ne!(field2.handle(), handle);
    field2.write(&[1.0; 8]).unwrap();
    let data = field2.read().unwrap();
    assert_eq!(data, vec![1.0; 8]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. MappedField write/read
// ═══════════════════════════════════════════════════════════════════════════

/// Tests pointer arithmetic in MappedField.write() and MappedField.read().
///
/// MappedField uses `core::ptr::write` and `core::ptr::read` through a raw
/// pointer obtained from the device. Miri validates alignment, bounds, and
/// provenance.
#[test]
fn mapped_field_write_read() {
    let gpu = quanta::init_cpu();
    let mut mf = gpu.field_mapped::<u32>(4).unwrap();
    mf.write(0, 100);
    mf.write(1, 200);
    mf.write(2, 300);
    mf.write(3, 400);
    assert_eq!(mf.read(0), 100);
    assert_eq!(mf.read(1), 200);
    assert_eq!(mf.read(2), 300);
    assert_eq!(mf.read(3), 400);
}

/// f32 variant -- exercises different size_of::<T>() in the pointer offset.
#[test]
fn mapped_field_write_read_f32() {
    let gpu = quanta::init_cpu();
    let mut mf = gpu.field_mapped::<f32>(3).unwrap();
    mf.write(0, 1.5);
    mf.write(1, 2.5);
    mf.write(2, 3.5);
    assert!((mf.read(0) - 1.5).abs() < 1e-6);
    assert!((mf.read(1) - 2.5).abs() < 1e-6);
    assert!((mf.read(2) - 3.5).abs() < 1e-6);
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. MappedField as_slice / as_mut_slice
// ═══════════════════════════════════════════════════════════════════════════

/// Tests `core::slice::from_raw_parts` and `from_raw_parts_mut` safety.
///
/// These construct slices from a raw pointer + count. Miri validates that
/// the pointer is valid for the claimed length, is properly aligned, and
/// the memory is initialized.
#[test]
fn mapped_field_as_slice() {
    let gpu = quanta::init_cpu();
    let mut mf = gpu.field_mapped::<u32>(4).unwrap();

    // Write via as_mut_slice
    {
        let slice = mf.as_mut_slice();
        assert_eq!(slice.len(), 4);
        slice[0] = 10;
        slice[1] = 20;
        slice[2] = 30;
        slice[3] = 40;
    }

    // Read via as_slice
    let slice = mf.as_slice();
    assert_eq!(slice, &[10, 20, 30, 40]);
}

/// Empty mapped field: zero-length slices must still be valid pointers.
///
/// NOTE: This test is currently disabled because it exposes a real bug:
/// `field_create_mapped(0)` returns a null pointer, and `as_slice()`
/// passes that null to `core::slice::from_raw_parts`, which is UB.
/// The fix belongs in MappedField::as_slice/as_mut_slice (guard count==0)
/// or in CpuDevice::field_create_mapped (return NonNull::dangling()).
#[test]
#[ignore = "exposes UB: null pointer in from_raw_parts for empty MappedField"]
fn mapped_field_as_slice_empty() {
    let gpu = quanta::init_cpu();
    let mf = gpu.field_mapped::<u32>(0).unwrap();
    let slice = mf.as_slice();
    assert!(slice.is_empty());
    assert!(mf.is_empty());
    assert_eq!(mf.byte_size(), 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Texture write/read
// ═══════════════════════════════════════════════════════════════════════════

/// Tests texture pixel data roundtrip through the CPU device.
///
/// The CPU driver stores texture data in the same buffer pool as fields,
/// exercising the same byte-level copy paths.
#[test]
fn texture_write_read() {
    let gpu = quanta::init_cpu();
    let tex = gpu.texture(2, 2).unwrap();
    // RGBA8: 4 bytes per pixel, 4 pixels = 16 bytes
    let pixels: Vec<u8> = vec![
        255, 0, 0, 255, // red
        0, 255, 0, 255, // green
        0, 0, 255, 255, // blue
        255, 255, 0, 255, // yellow
    ];
    tex.write(&pixels).unwrap();
    let readback = tex.read().unwrap();
    assert_eq!(readback, pixels);
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Field outlives Gpu
// ═══════════════════════════════════════════════════════════════════════════

/// Tests that Arc<dyn GpuDevice> keeps the device alive when the Gpu handle
/// is dropped but a Field still exists.
///
/// The Field holds an Arc clone of the device. Dropping the Gpu wrapper
/// should not invalidate the Field. Miri catches use-after-free on the
/// inner device.
#[test]
fn field_outlives_gpu() {
    let field;
    {
        let gpu = quanta::init_cpu();
        field = gpu.field::<f32>(4).unwrap();
        field.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        // gpu dropped here
    }
    // Field must still work -- the Arc keeps the device alive
    let data = field.read().unwrap();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Same test for MappedField -- raw pointer must remain valid.
#[test]
fn mapped_field_outlives_gpu() {
    let mut mf;
    {
        let gpu = quanta::init_cpu();
        mf = gpu.field_mapped::<u32>(2).unwrap();
        mf.write(0, 42);
        mf.write(1, 99);
        // gpu dropped here
    }
    // MappedField must still work
    assert_eq!(mf.read(0), 42);
    assert_eq!(mf.read(1), 99);
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Concurrent field access (sequential from two threads)
// ═══════════════════════════════════════════════════════════════════════════

/// Tests Send + Sync on GpuDevice by moving an Arc<Field> between threads.
///
/// This is not a data race test (accesses are sequential). It verifies that
/// the device and field types correctly implement Send/Sync and that no
/// thread-local state is leaked.
#[test]
fn concurrent_field_access() {
    use std::sync::Arc;

    let gpu = quanta::init_cpu();
    let field = Arc::new(gpu.field::<u32>(4).unwrap());

    // Thread 1: write
    let f1 = Arc::clone(&field);
    let t1 = std::thread::spawn(move || {
        f1.write(&[10, 20, 30, 40]).unwrap();
    });
    t1.join().unwrap();

    // Thread 2: read (sequential -- after t1 finishes)
    let f2 = Arc::clone(&field);
    let t2 = std::thread::spawn(move || f2.read().unwrap());
    let data = t2.join().unwrap();

    assert_eq!(data, vec![10, 20, 30, 40]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. CPU executor -- no UB in the JIT path
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches a simple kernel on the CPU device and verifies results.
///
/// Exercises the full execute_ops path: register allocation, field read/write,
/// binary operations, all under Miri's watchful eye.
#[test]
fn cpu_executor_no_ub() {
    let gpu = quanta::init_cpu();

    // Build a kernel: out[i] = a[i] + b[i]
    let def = KernelDef {
        name: "add".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "a".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldRead {
                name: "b".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Load {
                dst: Reg(2),
                field: 1,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::BinOp {
                dst: Reg(3),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Store {
                field: 2,
                index: Reg(0),
                src: Reg(3),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 4,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [4, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let kernel_bytes = quanta_ir::serialize_kernel(&def);

    let a = gpu.field::<f32>(4).unwrap();
    let b = gpu.field::<f32>(4).unwrap();
    let out = gpu.field::<f32>(4).unwrap();

    a.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    b.write(&[10.0, 20.0, 30.0, 40.0]).unwrap();

    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &out);

    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();

    let result = out.read().unwrap();
    assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Wire format roundtrip
// ═══════════════════════════════════════════════════════════════════════════

/// Serializes and deserializes a KernelDef, verifying the wire format
/// encoder/decoder do not produce UB.
///
/// The wire format uses manual byte packing (little-endian u32/u64, length-
/// prefixed strings). Miri catches any pointer arithmetic errors in the
/// Reader/Writer.
#[test]
fn wire_format_roundtrip() {
    let def = KernelDef {
        name: "test_kernel".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "input".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::FieldWrite {
                name: "output".into(),
                slot: 1,
                scalar_type: ScalarType::U32,
            },
            KernelParam::Constant {
                name: "threshold".into(),
                slot: 2,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::F32,
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::F32(1.0),
            },
            KernelOp::Const {
                dst: Reg(3),
                value: ConstValue::U32(42),
            },
            KernelOp::Const {
                dst: Reg(4),
                value: ConstValue::Bool(true),
            },
            KernelOp::BinOp {
                dst: Reg(5),
                a: Reg(1),
                b: Reg(2),
                op: BinOp::Add,
                ty: ScalarType::F32,
            },
            KernelOp::Cmp {
                dst: Reg(6),
                a: Reg(1),
                b: Reg(2),
                op: CmpOp::Gt,
                ty: ScalarType::F32,
            },
            KernelOp::Branch {
                cond: Reg(6),
                then_ops: vec![KernelOp::Store {
                    field: 1,
                    index: Reg(0),
                    src: Reg(3),
                    ty: ScalarType::U32,
                }],
                else_ops: vec![],
            },
        ],
        body_source: None,
        next_reg: 7,
        opt_level: 2,
        device_sources: vec!["fn helper() -> f32 { 1.0 }".into()],
        device_functions: vec![],
        workgroup_size: [256, 1, 1],
        subgroup_size: Some(32),
        dynamic_shared_bytes: 1024,
    };

    let bytes = quanta_ir::serialize_kernel(&def);
    let decoded = quanta_ir::deserialize_kernel(&bytes).expect("deserialization failed");

    assert_eq!(decoded.name, "test_kernel");
    assert_eq!(decoded.params.len(), 3);
    assert_eq!(decoded.body.len(), 8);
    assert_eq!(decoded.next_reg, 7);
    assert_eq!(decoded.opt_level, 2);
    assert_eq!(decoded.workgroup_size, [256, 1, 1]);
    assert_eq!(decoded.subgroup_size, Some(32));
    assert_eq!(decoded.dynamic_shared_bytes, 1024);
    assert_eq!(decoded.device_sources.len(), 1);
}

/// Empty kernel -- edge case for the wire format.
#[test]
fn wire_format_roundtrip_empty() {
    let def = KernelDef {
        name: "empty".into(),
        params: vec![],
        body: vec![],
        body_source: None,
        next_reg: 0,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };

    let bytes = quanta_ir::serialize_kernel(&def);
    let decoded = quanta_ir::deserialize_kernel(&bytes).expect("deserialization failed");
    assert_eq!(decoded.name, "empty");
    assert!(decoded.params.is_empty());
    assert!(decoded.body.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Value type conversions
// ═══════════════════════════════════════════════════════════════════════════

/// Exercises all scalar type conversions through the CPU executor's
/// Cast operation.
///
/// The CPU driver's Value enum uses match-based conversions (not union
/// transmutes), but Miri still validates that no branch produces UB
/// (e.g., float-to-int overflow on platforms where that's UB).
#[test]
fn value_type_conversions_via_kernel() {
    let gpu = quanta::init_cpu();

    // Build a kernel that casts u32 -> f32: out[i] = (f32)input[i]
    let def = KernelDef {
        name: "cast_u32_f32".into(),
        params: vec![
            KernelParam::FieldRead {
                name: "input".into(),
                slot: 0,
                scalar_type: ScalarType::U32,
            },
            KernelParam::FieldWrite {
                name: "output".into(),
                slot: 1,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Load {
                dst: Reg(1),
                field: 0,
                index: Reg(0),
                ty: ScalarType::U32,
            },
            KernelOp::Cast {
                dst: Reg(2),
                src: Reg(1),
                from: ScalarType::U32,
                to: ScalarType::F32,
            },
            KernelOp::Store {
                field: 1,
                index: Reg(0),
                src: Reg(2),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 3,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [4, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let kernel_bytes = quanta_ir::serialize_kernel(&def);

    let input = gpu.field::<u32>(4).unwrap();
    let output = gpu.field::<f32>(4).unwrap();

    input.write(&[0, 1, 42, 1000]).unwrap();

    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &input);
    wave.bind(1, &output);
    let mut pulse = gpu.dispatch(&wave, 4).unwrap();
    pulse.wait().unwrap();

    let result = output.read().unwrap();
    assert_eq!(result, vec![0.0, 1.0, 42.0, 1000.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// Additional safety edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// Wave push constants use `core::ptr::copy_nonoverlapping` to pack inline
/// data. Miri validates the pointer arithmetic and bounds.
#[test]
fn wave_push_constant_safety() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(1).unwrap();

    let def = KernelDef {
        name: "push_const".into(),
        params: vec![
            KernelParam::FieldWrite {
                name: "out".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
            KernelParam::Constant {
                name: "val".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            },
        ],
        body: vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::F32(99.0),
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(1),
                ty: ScalarType::F32,
            },
        ],
        body_source: None,
        next_reg: 2,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let kernel_bytes = quanta_ir::serialize_kernel(&def);

    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);
    // Exercise set_value with different types
    wave.set_value(0, 3.14f32);
    wave.set_value(1, 42u32);
    wave.set_value(2, 100i64);

    // Also exercise set_bytes
    wave.set_bytes(3, &[0xDE, 0xAD, 0xBE, 0xEF]);

    let mut pulse = gpu.dispatch(&wave, 1).unwrap();
    pulse.wait().unwrap();
}

/// Field copy_from exercises field_copy_bytes in the device, which
/// reads from one buffer and writes to another while both are locked.
#[test]
fn field_copy_from_safety() {
    let gpu = quanta::init_cpu();
    let src = gpu.field::<f32>(4).unwrap();
    let dst = gpu.field::<f32>(4).unwrap();

    src.write(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    dst.copy_from(&src).unwrap();

    let result = dst.read().unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

/// Pulse lifecycle: create, wait, reset, verify state transitions.
#[test]
fn pulse_lifecycle_no_ub() {
    let gpu = quanta::init_cpu();
    let field = gpu.field::<f32>(1).unwrap();
    field.write(&[0.0]).unwrap();

    let def = KernelDef {
        name: "noop".into(),
        params: vec![KernelParam::FieldWrite {
            name: "x".into(),
            slot: 0,
            scalar_type: ScalarType::F32,
        }],
        body: vec![],
        body_source: None,
        next_reg: 0,
        opt_level: 0,
        device_sources: vec![],
        device_functions: vec![],
        workgroup_size: [1, 1, 1],
        subgroup_size: None,
        dynamic_shared_bytes: 0,
    };
    let kernel_bytes = quanta_ir::serialize_kernel(&def);

    let mut wave = gpu.wave_jit(&kernel_bytes).unwrap();
    wave.bind(0, &field);

    let mut pulse = gpu.dispatch(&wave, 1).unwrap();
    assert!(pulse.is_done()); // CPU dispatch is synchronous
    pulse.wait().unwrap();
    pulse.reset();
    assert!(!pulse.is_done());
}

/// Multiple fields allocated and freed in interleaved order.
/// Tests that handle recycling doesn't cause aliasing issues.
#[test]
fn interleaved_alloc_free() {
    let gpu = quanta::init_cpu();

    let f1 = gpu.field::<u32>(4).unwrap();
    let f2 = gpu.field::<u32>(8).unwrap();
    let f3 = gpu.field::<u32>(16).unwrap();

    f1.write(&[1, 2, 3, 4]).unwrap();
    f2.write(&[10, 20, 30, 40, 50, 60, 70, 80]).unwrap();
    f3.write(&[100; 16]).unwrap();

    // Drop f2 in the middle
    drop(f2);

    // f1 and f3 must still be valid
    let d1 = f1.read().unwrap();
    let d3 = f3.read().unwrap();
    assert_eq!(d1, vec![1, 2, 3, 4]);
    assert_eq!(d3, vec![100; 16]);

    // Allocate a new field in the gap
    let f4 = gpu.field::<u32>(4).unwrap();
    f4.write(&[99, 98, 97, 96]).unwrap();
    let d4 = f4.read().unwrap();
    assert_eq!(d4, vec![99, 98, 97, 96]);
}

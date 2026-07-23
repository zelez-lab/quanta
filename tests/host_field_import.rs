//! Zero-copy host-memory field import.
//!
//! `gpu.field_from_host(&data)` wraps caller-owned memory as a
//! read-only field — zero-copy where the backend has an import path
//! (`supports_host_import`), a queryable staged copy elsewhere.
//! Releasing the field releases the view, never the caller's pages.
//! The ghost model (T760–T766) proves the state machine; these tests
//! check the production side against it.
//!
//! Run: cargo test --test host_field_import --features software

use quanta::QuantaErrorKind;

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

/// Page-aligned host allocation standing in for an mmap'd region:
/// same alignment guarantees, no file needed.
struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new_f32(count: usize, align: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(count * 4, align).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Self { ptr, layout }
    }

    fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const f32, self.layout.size() / 4) }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut f32, self.layout.size() / 4) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// The software backend imports by pointer passthrough: any slice,
/// zero copies, `is_imported() == true`.
#[test]
#[cfg(feature = "software")]
fn cpu_import_is_passthrough() {
    let gpu = quanta::init_cpu();
    assert!(gpu.supports_host_import());
    assert_eq!(gpu.host_import_alignment(), Some(1));
    let data = vec![7.5f32; 1024];
    let hf = gpu.field_from_host(&data).unwrap();
    assert!(hf.is_imported());
    assert_eq!(hf.len(), 1024);
    drop(hf);
    assert!(
        data.iter().all(|&x| x == 7.5),
        "drop must not touch host data"
    );
}

/// A kernel reads the imported region directly; results prove the
/// device saw the caller's bytes. Runs on every backend `init()`
/// picks; asserts the zero-copy/staged answer matches the capability.
#[quanta::kernel]
fn double_it(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    output[i] = input[i] * 2.0f32;
}

#[test]
fn kernel_reads_imported_region() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let align = gpu.host_import_alignment().unwrap_or(1).max(4);
    let count = align / 4 * 4; // a few granules' worth of f32s
    let mut buf = AlignedBuf::new_f32(count, align);
    for (i, x) in buf.as_mut_slice().iter_mut().enumerate() {
        *x = i as f32;
    }

    let hf = gpu.field_from_host(buf.as_slice()).unwrap();
    assert_eq!(hf.is_imported(), gpu.supports_host_import());

    let out = gpu.field::<f32>(count).unwrap();
    let mut wave = double_it(&gpu).expect("create wave");
    wave.bind_host(0, &hf);
    wave.bind(1, &out);
    gpu.dispatch(&wave, count as u32).unwrap().wait().unwrap();

    let result = out.read().unwrap();
    for (i, &r) in result.iter().enumerate() {
        assert_eq!(r, i as f32 * 2.0, "element {i}");
    }
    drop(hf);
    // Releasing the view never touches the caller's pages.
    for (i, &x) in buf.as_slice().iter().enumerate() {
        assert_eq!(x, i as f32, "host data intact after drop");
    }
}

/// The imported view registers and releases exactly one registry
/// entry — same leak-check idiom as every wrapper before it.
#[test]
fn import_registers_and_frees_one_entry() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let align = gpu.host_import_alignment().unwrap_or(1).max(4);
    let buf = AlignedBuf::new_f32(align / 4 * 4, align);

    let before = gpu.debug_registry_counts();
    let hf = gpu.field_from_host(buf.as_slice()).unwrap();
    let during = gpu.debug_registry_counts();
    assert_ne!(before, during, "import must register a buffer entry");
    drop(hf);
    let after = gpu.debug_registry_counts();
    assert_eq!(before, after, "dropping a HostField must free its entry");
}

/// Alignment is a hard contract on import-capable backends: violations
/// are InvalidParam, never a silent staged copy (T765's production
/// side).
#[test]
fn unaligned_import_rejected() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let Some(align) = gpu.host_import_alignment() else {
        eprintln!("skipping: no import path");
        return;
    };
    if align < 8 {
        eprintln!("skipping: pointer-passthrough backend has no granularity");
        return;
    }
    let buf = AlignedBuf::new_f32(align / 4 * 2, align);

    // Misaligned base (aligned + one element), aligned length.
    let err = unsafe { gpu.field_from_host_ptr::<f32>(buf.as_slice().as_ptr().add(1), align / 4) }
        .unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));

    // Aligned base, misaligned length.
    let err = gpu
        .field_from_host(&buf.as_slice()[..align / 4 - 1])
        .unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
}

/// Empty imports are rejected up front on every backend.
#[test]
fn empty_import_rejected() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };
    let err = gpu.field_from_host::<f32>(&[]).unwrap_err();
    assert!(matches!(err.kind, QuantaErrorKind::InvalidParam(_)));
}

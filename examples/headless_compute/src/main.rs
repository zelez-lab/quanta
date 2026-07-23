//! Headless compute consumer — the Thiaba / ai_project shape.
//!
//! Depends on `quanta` alone with `default-features = false, features =
//! ["software"]`: render OFF. Proves the step-085 boundary at runtime —
//! a pure GPGPU app builds and dispatches a kernel with zero rendering
//! code compiled and no rendering type on its surface. The companion CI
//! check asserts `cargo tree` for this crate contains no `quanta-render`.

use quanta::{QuantaError, init_cpu};

#[derive(quanta::Fields)]
struct VecAdd {
    a: Vec<f32>,
    b: Vec<f32>,
    result: Vec<f32>,
}

#[quanta::kernel]
fn vector_add(d: &VecAdd) {
    let i = quark_id();
    d.result[i] = d.a[i] + d.b[i];
}

/// The database scan shape: squared L2 distance from every stored
/// vector to the query. `DIM` is fixed in the kernel; top-k happens
/// host-side over the small distance array.
#[quanta::kernel]
fn knn_distances(vectors: &[f32], query: &[f32], distances: &mut [f32]) {
    let j = quark_id();
    let mut acc = 0.0f32;
    for k in 0..8 {
        let d = vectors[j * 8 + k] - query[k];
        acc = acc + d * d;
    }
    distances[j] = acc;
}

const DIM: usize = 8;
const N: usize = 1000;

/// mmap'd file region — the primary use case of host import: the
/// on-disk format IS the in-memory format, and the GPU binds the
/// pages directly. Hand-declared POSIX externs: no dependency.
#[cfg(unix)]
mod region {
    use core::ffi::{c_int, c_void};

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }
    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;

    /// A read-only mmap of a whole file. Unmapped on drop.
    pub struct Mapped {
        ptr: *const u8,
        len: usize,
    }

    impl Mapped {
        pub fn open(path: &std::path::Path, len: usize) -> Mapped {
            use std::os::unix::io::AsRawFd;
            let file = std::fs::File::open(path).expect("open data file");
            let ptr = unsafe {
                mmap(
                    core::ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert!(ptr as isize != -1, "mmap failed");
            Mapped {
                ptr: ptr as *const u8,
                len,
            }
        }

        pub fn as_f32s(&self) -> &[f32] {
            unsafe { core::slice::from_raw_parts(self.ptr as *const f32, self.len / 4) }
        }
    }

    impl Drop for Mapped {
        fn drop(&mut self) {
            unsafe { munmap(self.ptr as *mut c_void, self.len) };
        }
    }
}

#[cfg(unix)]
fn knn_over_mmap(gpu: &quanta::Gpu) -> Result<(), QuantaError> {
    // Write [N, DIM] f32 vectors, padded to the import granularity so
    // the mapped slice satisfies the alignment contract as-is.
    let align = gpu.host_import_alignment().unwrap_or(1).max(4);
    let data_bytes = N * DIM * 4;
    let padded_bytes = data_bytes.div_ceil(align) * align;
    let mut bytes = vec![0u8; padded_bytes];
    for j in 0..N {
        for k in 0..DIM {
            let v = (j * DIM + k) as f32 * 0.001;
            bytes[(j * DIM + k) * 4..][..4].copy_from_slice(&v.to_le_bytes());
        }
    }
    let path = std::env::temp_dir().join("quanta_headless_knn.f32");
    std::fs::write(&path, &bytes).expect("write data file");

    // Bind the mapped pages directly — the database-scan shape.
    let mapped = region::Mapped::open(&path, padded_bytes);
    let vectors = gpu.field_from_host(mapped.as_f32s())?;

    let query_data: Vec<f32> = (0..DIM).map(|k| (2 * DIM + k) as f32 * 0.001).collect();
    let query = gpu.field::<f32>(DIM)?;
    query.write(&query_data)?;
    let distances = gpu.field::<f32>(N)?;

    let mut wave = knn_distances(gpu).expect("create wave");
    wave.bind_host(0, &vectors);
    wave.bind(1, &query);
    wave.bind(2, &distances);
    gpu.dispatch(&wave, N as u32)?.wait()?;

    // Top-k on the host over the small distance array.
    let d = distances.read()?;
    let mut idx: Vec<usize> = (0..N).collect();
    idx.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let top3 = &idx[..3];

    // The query IS row 2, so the nearest vectors are rows 1..3.
    assert_eq!(top3[0], 2, "the query's own row must be nearest");
    println!(
        "knn over mmap ok: top-3 rows {:?}, zero-copy: {}",
        top3,
        vectors.is_imported(),
    );

    drop(vectors);
    drop(mapped);
    let _ = std::fs::remove_file(&path);
    Ok(())
}

fn main() -> Result<(), QuantaError> {
    // Software lane: no GPU, no surface, no render. Pure headless compute.
    let gpu = init_cpu();

    let mut data = VecAdd {
        a: vec![1.0; 1024],
        b: vec![2.0; 1024],
        result: vec![0.0; 1024],
    };

    vector_add(&gpu, &mut data, 1024)?.wait()?;

    assert_eq!(data.result[0], 3.0);
    println!("headless compute ok: 1.0 + 2.0 = {}", data.result[0]);

    // The Thiaba shape end-to-end: knn over an mmap'd [N, DIM] region
    // with zero host copies (software = pointer passthrough).
    #[cfg(unix)]
    knn_over_mmap(&gpu)?;

    Ok(())
}

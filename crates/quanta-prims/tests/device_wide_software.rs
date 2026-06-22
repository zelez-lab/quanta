//! Device-wide reductions on the SOFTWARE lane.
//!
//! The other prims test suites use `quanta::init()` (real GPU — Metal on
//! the dev Mac), so they never exercised the CPU software backend. This
//! suite pins `device_reduce_*` on `quanta::init_cpu()` so a regression in
//! the CPU executor (cooperative subgroup reductions / barrier
//! segmentation) can't slip through unnoticed.

#![cfg(feature = "gpu")]

use quanta_prims::{
    device_reduce_add_f32, device_reduce_add_f32_field, device_reduce_add_i32,
    device_reduce_add_i32_field, device_reduce_add_u32, device_reduce_add_u32_field,
    device_reduce_max_f32, device_reduce_max_i32, device_reduce_max_u32,
    device_reduce_max_u32_field, device_reduce_min_f32, device_reduce_min_i32,
    device_reduce_min_u32, device_reduce_min_u32_field,
};

fn gpu() -> quanta::Gpu {
    quanta::init_cpu()
}

const SIZES: [usize; 4] = [5, 256, 257, 1000];

#[test]
fn reduce_add_f32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 0.5).collect();
        let want: f32 = data.iter().sum();
        let got = device_reduce_add_f32(&g, &data).unwrap();
        assert!(
            (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
            "add f32 n={n}: {got} vs {want}"
        );
    }
}

#[test]
fn reduce_min_max_f32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) % 101) as f32 - 50.0).collect();
        let want_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let want_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            device_reduce_min_f32(&g, &data).unwrap(),
            want_min,
            "min n={n}"
        );
        assert_eq!(
            device_reduce_max_f32(&g, &data).unwrap(),
            want_max,
            "max n={n}"
        );
    }
}

#[test]
fn reduce_add_u32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<u32> = (0..n).map(|i| (i % 13) as u32).collect();
        let want: u32 = data.iter().sum();
        assert_eq!(
            device_reduce_add_u32(&g, &data).unwrap(),
            want,
            "add u32 n={n}"
        );
    }
}

#[test]
fn reduce_min_max_u32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % 257) as u32).collect();
        let want_min = *data.iter().min().unwrap();
        let want_max = *data.iter().max().unwrap();
        assert_eq!(
            device_reduce_min_u32(&g, &data).unwrap(),
            want_min,
            "min u32 n={n}"
        );
        assert_eq!(
            device_reduce_max_u32(&g, &data).unwrap(),
            want_max,
            "max u32 n={n}"
        );
    }
}

#[test]
fn reduce_add_i32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<i32> = (0..n).map(|i| (i % 11) as i32 - 5).collect();
        let want: i32 = data.iter().sum();
        assert_eq!(
            device_reduce_add_i32(&g, &data).unwrap(),
            want,
            "add i32 n={n}"
        );
    }
}

#[test]
fn reduce_min_max_i32_software() {
    let g = gpu();
    for &n in &SIZES {
        let data: Vec<i32> = (0..n).map(|i| ((i * 19 + 1) % 211) as i32 - 100).collect();
        let want_min = *data.iter().min().unwrap();
        let want_max = *data.iter().max().unwrap();
        assert_eq!(
            device_reduce_min_i32(&g, &data).unwrap(),
            want_min,
            "min i32 n={n}"
        );
        assert_eq!(
            device_reduce_max_i32(&g, &data).unwrap(),
            want_max,
            "max i32 n={n}"
        );
    }
}

/// The device-resident `_field` variant (no host download of the data) must
/// agree with the host-slice variant. Covers the exact-multiple (256) and
/// padding-tail (257, 1000, 5) cases.
#[test]
fn reduce_field_matches_slice_software() {
    let g = gpu();
    for &n in &SIZES {
        // f32 add
        let f: Vec<f32> = (0..n).map(|i| (i % 7) as f32 + 0.5).collect();
        let field = g.field::<f32>(n).unwrap();
        field.write(&f).unwrap();
        let slice = device_reduce_add_f32(&g, &f).unwrap();
        let dev = device_reduce_add_f32_field(&g, &field, n).unwrap();
        assert!(
            (slice - dev).abs() <= 1e-3 * (1.0 + slice.abs()),
            "add f32 field vs slice n={n}: {dev} vs {slice}"
        );

        // u32 add / min / max
        let u: Vec<u32> = (0..n).map(|i| ((i * 17 + 3) % 257) as u32).collect();
        let uf = g.field::<u32>(n).unwrap();
        uf.write(&u).unwrap();
        assert_eq!(
            device_reduce_add_u32_field(&g, &uf, n).unwrap(),
            device_reduce_add_u32(&g, &u).unwrap(),
            "add u32 n={n}"
        );
        assert_eq!(
            device_reduce_min_u32_field(&g, &uf, n).unwrap(),
            device_reduce_min_u32(&g, &u).unwrap(),
            "min u32 n={n}"
        );
        assert_eq!(
            device_reduce_max_u32_field(&g, &uf, n).unwrap(),
            device_reduce_max_u32(&g, &u).unwrap(),
            "max u32 n={n}"
        );

        // i32 add
        let s: Vec<i32> = (0..n).map(|i| (i % 11) as i32 - 5).collect();
        let sf = g.field::<i32>(n).unwrap();
        sf.write(&s).unwrap();
        assert_eq!(
            device_reduce_add_i32_field(&g, &sf, n).unwrap(),
            device_reduce_add_i32(&g, &s).unwrap(),
            "add i32 n={n}"
        );
    }
}

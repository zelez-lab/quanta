//! Raspberry Pi V3D hardware-Vulkan probe — validates the two V3D
//! driver-limitation fixes end-to-end on a real device:
//!
//! 1. **Subgroup-free reduce** (GAP 1): `supports_subgroups()` must
//!    report false on V3D / true on lavapipe, and the
//!    `device_reduce_*` wrappers must produce correct results either
//!    way (tree kernels on V3D, warp kernels elsewhere).
//! 2. **Folded 1D dispatch** (GAP 2): elementwise kernels with
//!    n > 65535 workgroups (LocalSize [1,1,1]) must write *every*
//!    output — the Vulkan driver folds the grid into 2D and the
//!    SPIR-V `QuarkId` linearizes it back.
//!
//! Run (Pi, V3D hardware):
//!   cargo run --release --example pi_v3d_probe -p quanta-array \
//!       --features vulkan
//! Force lavapipe:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json <same command>
//!
//! Prints one line per check and a final PASS / FAIL summary.

use quanta_array::Array;

fn main() {
    // Real device (V3D / lavapipe / Metal); software fallback so the
    // probe also smoke-runs on machines without a GPU backend.
    let gpu = quanta::init().unwrap_or_else(|_| quanta::init_cpu());
    println!("device: {}", gpu.name());
    println!("supports_subgroups: {}", gpu.supports_subgroups());

    let mut failures = 0usize;
    let mut check = |name: &str, ok: bool, detail: String| {
        println!(
            "{:40} {}  {}",
            name,
            if ok { "OK " } else { "FAIL" },
            detail
        );
        if !ok {
            failures += 1;
        }
    };

    // ── GAP 1: device-wide reduce (tree fallback on V3D) ────────────
    let n = 200_000usize;
    let f: Vec<f32> = (0..n).map(|i| ((i % 4001) as f32) * 0.25 - 500.0).collect();
    let want_sum: f32 = f.iter().sum();
    match quanta_prims::device_reduce_add_f32(&gpu, &f) {
        Ok(got) => {
            let ok = (got - want_sum).abs() <= 1e-3 * (1.0 + want_sum.abs());
            check("reduce_add_f32 (200k)", ok, format!("{got} vs {want_sum}"));
        }
        Err(e) => check("reduce_add_f32 (200k)", false, format!("{e:?}")),
    }

    let ints: Vec<i32> = (0..n)
        .map(|i| (((i as u64 * 2654435761) % 4_000_000) as i64 - 2_000_000) as i32)
        .collect();
    match quanta_prims::device_reduce_min_i32(&gpu, &ints) {
        Ok(got) => {
            let want = *ints.iter().min().unwrap();
            check(
                "reduce_min_i32 (200k)",
                got == want,
                format!("{got} vs {want}"),
            );
        }
        Err(e) => check("reduce_min_i32 (200k)", false, format!("{e:?}")),
    }

    // Full-range u32 max — also pins the unsigned-compare fix
    // (values above 2^31 compared signed would pick the wrong max).
    let us: Vec<u32> = (0..n)
        .map(|i| ((i as u64 * 2654435761) % 4_200_000_000) as u32)
        .collect();
    match quanta_prims::device_reduce_max_u32(&gpu, &us) {
        Ok(got) => {
            let want = *us.iter().max().unwrap();
            check(
                "reduce_max_u32 (200k, full-range)",
                got == want,
                format!("{got} vs {want}"),
            );
        }
        Err(e) => check("reduce_max_u32 (200k, full-range)", false, format!("{e:?}")),
    }

    // ── GAP 2: folded 1D dispatch, n > 65535 groups ─────────────────
    // 401_408 = the full-batch CNN activation shape [64, 8, 28, 28].
    let big_n = 401_408usize;
    let data: Vec<f32> = (0..big_n).map(|i| ((i % 1013) as f32) - 500.0).collect();
    match Array::from_slice(&gpu, &data, &[big_n])
        .and_then(|a| a.abs())
        .and_then(|r| r.to_vec())
    {
        Ok(out) => {
            let bad = out
                .iter()
                .zip(&data)
                .position(|(g, x)| (g - x.abs()).abs() > f32::EPSILON);
            check(
                "abs on 401408 elems (folded dispatch)",
                out.len() == big_n && bad.is_none(),
                match bad {
                    Some(i) => format!("first mismatch at {i}: {} vs {}", out[i], data[i].abs()),
                    None => format!("{} elements all correct", out.len()),
                },
            );
        }
        Err(e) => check(
            "abs on 401408 elems (folded dispatch)",
            false,
            format!("{e:?}"),
        ),
    }

    // Larger still: 700_000 (multiple full rows + remainder).
    let big2 = 700_000usize;
    let d2: Vec<f32> = (0..big2).map(|i| ((i % 89) as f32) * 0.5 - 22.0).collect();
    match (
        Array::from_slice(&gpu, &d2, &[big2]),
        Array::from_slice(&gpu, &d2, &[big2]),
    ) {
        (Ok(a), Ok(b)) => match a.mul(&b).and_then(|r| r.to_vec()) {
            Ok(out) => {
                let bad = out
                    .iter()
                    .zip(&d2)
                    .position(|(g, x)| (g - x * x).abs() > 1e-5 * (1.0 + (x * x).abs()));
                check(
                    "mul on 700000 elems (folded dispatch)",
                    out.len() == big2 && bad.is_none(),
                    match bad {
                        Some(i) => format!("first mismatch at {i}"),
                        None => format!("{} elements all correct", out.len()),
                    },
                );
            }
            Err(e) => check(
                "mul on 700000 elems (folded dispatch)",
                false,
                format!("{e:?}"),
            ),
        },
        _ => check(
            "mul on 700000 elems (folded dispatch)",
            false,
            "alloc failed".into(),
        ),
    }

    println!();
    if failures == 0 {
        println!("PASS — all V3D probes green");
    } else {
        println!("FAIL — {failures} probe(s) failed");
        std::process::exit(1);
    }
}

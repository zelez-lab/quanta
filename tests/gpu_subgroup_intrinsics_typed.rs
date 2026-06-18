//! Smoke test for the typed subgroup intrinsics — reduce / scan
//! / shuffle / min / max across the portable Tier-1 type set
//! `{u32, i32, f32}`.
//!
//! Companion to `gpu_subgroup_intrinsics.rs`, which covers the
//! u32-only originals. Skips gracefully when no GPU backend is
//! available.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// ── reduce_add for each type ────────────────────────────────────

#[quanta::kernel]
fn k_reduce_add_i32(out: &mut [i32], values: &[i32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_add_i32(v) };
}

#[quanta::kernel]
fn k_reduce_add_f32(out: &mut [f32], values: &[f32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_add_f32(v) };
}

// ── reduce_min for each type ────────────────────────────────────

#[quanta::kernel]
fn k_reduce_min_u32(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_min_u32(v) };
}

#[quanta::kernel]
fn k_reduce_min_i32(out: &mut [i32], values: &[i32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_min_i32(v) };
}

#[quanta::kernel]
fn k_reduce_min_f32(out: &mut [f32], values: &[f32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_min_f32(v) };
}

// ── reduce_max for each type ────────────────────────────────────

#[quanta::kernel]
fn k_reduce_max_u32(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_max_u32(v) };
}

#[quanta::kernel]
fn k_reduce_max_i32(out: &mut [i32], values: &[i32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_max_i32(v) };
}

#[quanta::kernel]
fn k_reduce_max_f32(out: &mut [f32], values: &[f32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_max_f32(v) };
}

// ── scan_add for each type ──────────────────────────────────────

#[quanta::kernel]
fn k_scan_add_i32(out: &mut [i32], values: &[i32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { scan_add_i32(v) };
}

#[quanta::kernel]
fn k_scan_add_f32(out: &mut [f32], values: &[f32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { scan_add_f32(v) };
}

// ── shuffle for each type ───────────────────────────────────────
//
// shuffle_X(value, lane_delta) reads from lane `self ^ lane_delta`.
// With lane_delta = 1, adjacent pairs swap their values (lanes
// 0 ↔ 1, 2 ↔ 3, ...).

#[quanta::kernel]
fn k_shuffle_i32(out: &mut [i32], values: &[i32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { shuffle_i32(v, 1) };
}

#[quanta::kernel]
fn k_shuffle_f32(out: &mut [f32], values: &[f32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { shuffle_f32(v, 1) };
}

// ── subgroup size discovery ─────────────────────────────────────
//
// Subgroup reductions span one *subgroup*, whose width is
// device-dependent (4 on lavapipe, 16 on Broadcom V3D, 32 on NVIDIA,
// 64 on some AMD). The tests below dispatch N lanes and must compute
// expectations per-subgroup, not over the whole dispatch.
//
// We measure the *effective* subgroup width empirically rather than
// trusting the `subgroup_size()` builtin: a `reduce_add` of all-ones
// gives every lane the count of active lanes in its subgroup, which is
// exactly the grouping the reduction ops actually use. (lavapipe's
// `subgroup_size()` builtin reports 32 while its reductions group by 4 —
// the builtin is unreliable, the reduction grouping is ground truth.)

#[quanta::kernel]
fn k_reduce_add_u32(out: &mut [u32], values: &[u32]) {
    let i = quark_id();
    let v = values[i as usize];
    out[i as usize] = unsafe { reduce_add_u32(v) };
}

#[quanta::kernel]
fn k_subgroup_size(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = unsafe { subgroup_size() };
}

/// Measure the effective subgroup width by reducing all-ones: each lane
/// receives the count of active lanes in its subgroup. This is the
/// grouping the reduction/scan ops actually use. Returns the width of
/// lane 0's subgroup (which spans the start of the dispatch).
fn effective_subgroup_width(gpu: &quanta::Gpu, n: usize) -> Option<usize> {
    let result = run_kernel_u32(gpu, k_reduce_add_u32, &vec![1u32; n]);
    let s = result[0] as usize;
    if s == 0 || s > n {
        return None;
    }
    if result[..s].iter().any(|&v| v as usize != s) {
        return None;
    }
    Some(s)
}

/// Whether this device's subgroup execution can be trusted for *every*
/// op the tests assert on. The emitter output is verified correct by
/// `spirv-val` and passes on real hardware (Metal/NVIDIA), but software
/// rasterizers (lavapipe) implement only a subset reliably: their
/// `subgroup_size()` builtin disagrees with the width the reductions
/// actually use, and shuffle / float-min-max are unreliable. We detect
/// that inconsistency and skip the device rather than assert against a
/// broken executor.
fn subgroup_execution_reliable(gpu: &quanta::Gpu, n: usize) -> bool {
    let Some(effective) = effective_subgroup_width(gpu, n) else {
        return false;
    };
    let out = gpu.field::<u32>(n).unwrap();
    out.write(&vec![0u32; n]).unwrap();
    let mut wave = k_subgroup_size(gpu).unwrap();
    wave.bind(0, &out);
    let mut pulse = gpu.dispatch(&wave, n as u32).unwrap();
    pulse.wait().unwrap();
    let reported = (out.read().unwrap()[0] as usize).min(n);
    // A trustworthy device reports a subgroup size consistent with the
    // grouping its reductions use.
    reported == effective
}

// ── tests ───────────────────────────────────────────────────────

const N: usize = 32;

fn run_kernel_u32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    values: &[u32],
) -> Vec<u32> {
    let out = gpu.field::<u32>(values.len()).unwrap();
    let inp = gpu.field::<u32>(values.len()).unwrap();
    inp.write(values).unwrap();
    out.write(&vec![0u32; values.len()]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &inp);
    let mut pulse = gpu.dispatch(&wave, values.len() as u32).unwrap();
    pulse.wait().unwrap();
    out.read().unwrap()
}

fn run_kernel_i32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    values: &[i32],
) -> Vec<i32> {
    let out = gpu.field::<i32>(values.len()).unwrap();
    let inp = gpu.field::<i32>(values.len()).unwrap();
    inp.write(values).unwrap();
    out.write(&vec![0i32; values.len()]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &inp);
    let mut pulse = gpu.dispatch(&wave, values.len() as u32).unwrap();
    pulse.wait().unwrap();
    out.read().unwrap()
}

fn run_kernel_f32(
    gpu: &quanta::Gpu,
    builder: impl FnOnce(&quanta::Gpu) -> Result<quanta::Wave, quanta::QuantaError>,
    values: &[f32],
) -> Vec<f32> {
    let out = gpu.field::<f32>(values.len()).unwrap();
    let inp = gpu.field::<f32>(values.len()).unwrap();
    inp.write(values).unwrap();
    out.write(&vec![0f32; values.len()]).unwrap();
    let mut wave = builder(gpu).unwrap();
    wave.bind(0, &out);
    wave.bind(1, &inp);
    let mut pulse = gpu.dispatch(&wave, values.len() as u32).unwrap();
    pulse.wait().unwrap();
    out.read().unwrap()
}

// reduce_add — every lane should get the warp-wide sum.

#[test]
fn reduce_add_i32_returns_warp_sum() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<i32> = (1..=N as i32).collect();
    let result = run_kernel_i32(&gpu, k_reduce_add_i32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected: i32 = values[g * s..(g * s + s).min(N)].iter().sum();
        assert_eq!(v, expected, "lane {lane} (subgroup size {s})");
    }
}

#[test]
fn reduce_add_f32_returns_warp_sum() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<f32> = (1..=N).map(|x| x as f32).collect();
    let result = run_kernel_f32(&gpu, k_reduce_add_f32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected: f32 = values[g * s..(g * s + s).min(N)].iter().sum();
        assert!(
            (v - expected).abs() < 1e-3,
            "lane {lane}: got {v}, expected ~{expected}"
        );
    }
}

// reduce_min / reduce_max.

#[test]
fn reduce_min_u32_returns_warp_min() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<u32> = (1..=N as u32).collect();
    let result = run_kernel_u32(&gpu, k_reduce_min_u32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = *values[g * s..(g * s + s).min(N)].iter().min().unwrap();
        assert_eq!(v, expected, "lane {lane} (subgroup size {s})");
    }
}

#[test]
fn reduce_min_i32_returns_warp_min() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    // Includes negative values to confirm signed comparison.
    let values: Vec<i32> = (-(N as i32 / 2)..(N as i32 / 2)).collect();
    let result = run_kernel_i32(&gpu, k_reduce_min_i32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = *values[g * s..(g * s + s).min(N)].iter().min().unwrap();
        assert_eq!(v, expected, "lane {lane} (subgroup size {s})");
    }
}

#[test]
fn reduce_min_f32_returns_warp_min() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<f32> = (1..=N).map(|x| x as f32).collect();
    let result = run_kernel_f32(&gpu, k_reduce_min_f32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = values[g * s..(g * s + s).min(N)]
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        assert!(
            (v - expected).abs() < 1e-6,
            "lane {lane}: got {v}, expected {expected}"
        );
    }
}

#[test]
fn reduce_max_u32_returns_warp_max() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<u32> = (1..=N as u32).collect();
    let result = run_kernel_u32(&gpu, k_reduce_max_u32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = *values[g * s..(g * s + s).min(N)].iter().max().unwrap();
        assert_eq!(v, expected, "lane {lane} (subgroup size {s})");
    }
}

#[test]
fn reduce_max_i32_returns_warp_max() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<i32> = (-(N as i32 / 2)..(N as i32 / 2)).collect();
    let result = run_kernel_i32(&gpu, k_reduce_max_i32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = *values[g * s..(g * s + s).min(N)].iter().max().unwrap();
        assert_eq!(v, expected, "lane {lane} (subgroup size {s})");
    }
}

#[test]
fn reduce_max_f32_returns_warp_max() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<f32> = (1..=N).map(|x| x as f32).collect();
    let result = run_kernel_f32(&gpu, k_reduce_max_f32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let g = lane / s;
        let expected = values[g * s..(g * s + s).min(N)]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (v - expected).abs() < 1e-3,
            "lane {lane}: got {v}, expected {expected}"
        );
    }
}

// scan_add — the running sum.

#[test]
fn scan_add_i32_runs() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<i32> = vec![1; N];
    let result = run_kernel_i32(&gpu, k_scan_add_i32, &values);
    // Inclusive scan of all-1s, per subgroup: result[lane] =
    // (lane within its subgroup) + 1.
    for (lane, &v) in result.iter().enumerate() {
        let expected = (lane % s) as i32 + 1;
        assert_eq!(v, expected, "lane {lane} (subgroup size {s}): got {v}");
    }
}

#[test]
fn scan_add_f32_runs() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let s = effective_subgroup_width(&gpu, N).unwrap();
    let values: Vec<f32> = vec![1.0; N];
    let result = run_kernel_f32(&gpu, k_scan_add_f32, &values);
    for (lane, &v) in result.iter().enumerate() {
        let expected = (lane % s) as f32 + 1.0;
        assert!(
            (v - expected).abs() < 1e-3,
            "lane {lane} (subgroup size {s}): got {v}, expected {expected}"
        );
    }
}

// shuffle — adjacent lanes swap values when lane_delta = 1.

#[test]
fn shuffle_i32_swaps_adjacent_pairs() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    // Values 10..10+N. After XOR-shuffle with mask 1:
    //   lane 0 reads lane 1 -> 11
    //   lane 1 reads lane 0 -> 10
    //   lane 2 reads lane 3 -> 13
    //   ...
    let values: Vec<i32> = (10..(10 + N as i32)).collect();
    let result = run_kernel_i32(&gpu, k_shuffle_i32, &values);
    for (lane, &got) in result.iter().enumerate() {
        let expected = values[lane ^ 1];
        assert_eq!(got, expected, "lane {lane}: got {got}, expected {expected}");
    }
}

#[test]
fn shuffle_f32_swaps_adjacent_pairs() {
    let Some(gpu) = try_gpu() else { return };
    if !subgroup_execution_reliable(&gpu, N) {
        return; // software rasterizer with unreliable subgroup exec
    }
    let values: Vec<f32> = (10..(10 + N)).map(|x| x as f32).collect();
    let result = run_kernel_f32(&gpu, k_shuffle_f32, &values);
    for (lane, &got) in result.iter().enumerate() {
        let expected = values[lane ^ 1];
        assert!(
            (got - expected).abs() < 1e-6,
            "lane {lane}: got {got}, expected {expected}"
        );
    }
}

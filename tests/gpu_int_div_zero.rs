//! Integer division/remainder by zero yields ZERO on every lane.
//!
//! The CPU reference (`driver/cpu/eval.rs`) defines x/0 = 0 and
//! x%0 = 0 for every int width; the GPU emitters guard the divide by
//! substituting the divisor before dividing (Metal hardware returns ~0
//! for u32 x/0, SPIR-V calls it undefined behavior, WGSL's own rule is
//! x/0 == x). These tests prove the contract through real dispatches:
//! the `#[quanta::kernel]` path exercises the AOT emitters (metallib
//! on Metal, SPIR-V on Vulkan), and the `wave_jit` path (jit feature)
//! exercises the runtime emitters on the same device.

fn try_gpu() -> Option<quanta::Gpu> {
    quanta::init().ok()
}

// --- AOT path: kernels compiled at build time by quanta-compiler ---

#[quanta::kernel]
fn div_rem_u32(a: &[u32], b: &[u32], q: &mut [u32], r: &mut [u32]) {
    let i = quark_id();
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
}

#[quanta::kernel]
fn div_rem_i32(a: &[i32], b: &[i32], q: &mut [i32], r: &mut [i32]) {
    let i = quark_id();
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
}

#[test]
fn aot_u32_div_rem_by_zero_is_zero() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    let a_data: Vec<u32> = vec![7, 42, 0, u32::MAX, 100, 1, 0xDEAD_BEEF, 13];
    let b_data: Vec<u32> = vec![3, 0, 0, 0, 7, 1, 0, 5];
    let n = a_data.len();

    let a = gpu.field::<u32>(n).unwrap();
    let b = gpu.field::<u32>(n).unwrap();
    let q = gpu.field::<u32>(n).unwrap();
    let r = gpu.field::<u32>(n).unwrap();
    a.write(&a_data).unwrap();
    b.write(&b_data).unwrap();

    let mut wave = div_rem_u32(&gpu).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &q);
    wave.bind(3, &r);
    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    p.wait().unwrap();

    let got_q = q.read().unwrap();
    let got_r = r.read().unwrap();
    for i in 0..n {
        let want_q = a_data[i].checked_div(b_data[i]).unwrap_or(0);
        let want_r = a_data[i].checked_rem(b_data[i]).unwrap_or(0);
        assert_eq!(
            got_q[i], want_q,
            "u32 {} / {} must be {want_q}, got {}",
            a_data[i], b_data[i], got_q[i]
        );
        assert_eq!(
            got_r[i], want_r,
            "u32 {} % {} must be {want_r}, got {}",
            a_data[i], b_data[i], got_r[i]
        );
    }
}

#[test]
fn aot_i32_div_rem_by_zero_is_zero() {
    let Some(gpu) = try_gpu() else {
        eprintln!("skipping: no GPU available");
        return;
    };

    // Negative dividends/divisors mixed with zero divisors; the signed
    // MIN/−1 case stays out (its semantics are excluded, matching the
    // op-matrix).
    let a_data: Vec<i32> = vec![-7, 42, -100, i32::MIN, 99, -1, 0, 13];
    let b_data: Vec<i32> = vec![3, 0, 0, 0, -7, 1, 0, -5];
    let n = a_data.len();

    let a = gpu.field::<i32>(n).unwrap();
    let b = gpu.field::<i32>(n).unwrap();
    let q = gpu.field::<i32>(n).unwrap();
    let r = gpu.field::<i32>(n).unwrap();
    a.write(&a_data).unwrap();
    b.write(&b_data).unwrap();

    let mut wave = div_rem_i32(&gpu).unwrap();
    wave.bind(0, &a);
    wave.bind(1, &b);
    wave.bind(2, &q);
    wave.bind(3, &r);
    let mut p = gpu.dispatch(&wave, n as u32).unwrap();
    p.wait().unwrap();

    let got_q = q.read().unwrap();
    let got_r = r.read().unwrap();
    for i in 0..n {
        let want_q = if b_data[i] == 0 {
            0
        } else {
            a_data[i].wrapping_div(b_data[i])
        };
        let want_r = if b_data[i] == 0 {
            0
        } else {
            a_data[i].wrapping_rem(b_data[i])
        };
        assert_eq!(
            got_q[i], want_q,
            "i32 {} / {} must be {want_q}, got {}",
            a_data[i], b_data[i], got_q[i]
        );
        assert_eq!(
            got_r[i], want_r,
            "i32 {} % {} must be {want_r}, got {}",
            a_data[i], b_data[i], got_r[i]
        );
    }
}

// --- JIT path: KernelDef IR built at runtime, dispatched via wave_jit ---

#[cfg(feature = "jit")]
mod jit_path {
    use super::try_gpu;
    use quanta::kernel::{BinOp, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

    /// `q[i] = a[i] / b[i]; r[i] = a[i] % b[i]` over `ty` — same bind
    /// layout as the AOT kernels above.
    fn div_rem_def(ty: ScalarType, name: &str) -> KernelDef {
        let read = |name: &str, slot: u32| KernelParam::FieldRead {
            name: name.into(),
            slot,
            scalar_type: ty,
        };
        let write = |name: &str, slot: u32| KernelParam::FieldWrite {
            name: name.into(),
            slot,
            scalar_type: ty,
        };
        KernelDef {
            name: name.into(),
            params: vec![read("a", 0), read("b", 1), write("q", 2), write("r", 3)],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty,
                },
                KernelOp::Load {
                    dst: Reg(2),
                    field: 1,
                    index: Reg(0),
                    ty,
                },
                KernelOp::BinOp {
                    dst: Reg(3),
                    a: Reg(1),
                    b: Reg(2),
                    op: BinOp::Div,
                    ty,
                },
                KernelOp::Store {
                    field: 2,
                    index: Reg(0),
                    src: Reg(3),
                    ty,
                },
                KernelOp::BinOp {
                    dst: Reg(4),
                    a: Reg(1),
                    b: Reg(2),
                    op: BinOp::Rem,
                    ty,
                },
                KernelOp::Store {
                    field: 3,
                    index: Reg(0),
                    src: Reg(4),
                    ty,
                },
            ],
            body_source: None,
            next_reg: 5,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    #[test]
    fn jit_u32_div_rem_by_zero_is_zero() {
        let Some(gpu) = try_gpu() else {
            eprintln!("skipping: no GPU available");
            return;
        };

        let a_data: Vec<u32> = vec![7, 42, 0, u32::MAX, 100, 1, 0xDEAD_BEEF, 13];
        let b_data: Vec<u32> = vec![3, 0, 0, 0, 7, 1, 0, 5];
        let n = a_data.len();

        let a = gpu.field::<u32>(n).unwrap();
        let b = gpu.field::<u32>(n).unwrap();
        let q = gpu.field::<u32>(n).unwrap();
        let r = gpu.field::<u32>(n).unwrap();
        a.write(&a_data).unwrap();
        b.write(&b_data).unwrap();

        let bytes = quanta_ir::serialize_kernel(&div_rem_def(ScalarType::U32, "jit_div_rem_u32"));
        let mut wave = gpu.wave_jit(&bytes).unwrap();
        wave.bind(0, &a);
        wave.bind(1, &b);
        wave.bind(2, &q);
        wave.bind(3, &r);
        let mut p = gpu.dispatch(&wave, n as u32).unwrap();
        p.wait().unwrap();

        let got_q = q.read().unwrap();
        let got_r = r.read().unwrap();
        for i in 0..n {
            let want_q = a_data[i].checked_div(b_data[i]).unwrap_or(0);
            let want_r = a_data[i].checked_rem(b_data[i]).unwrap_or(0);
            assert_eq!(
                got_q[i], want_q,
                "jit u32 {} / {} must be {want_q}, got {}",
                a_data[i], b_data[i], got_q[i]
            );
            assert_eq!(
                got_r[i], want_r,
                "jit u32 {} % {} must be {want_r}, got {}",
                a_data[i], b_data[i], got_r[i]
            );
        }
    }

    #[test]
    fn jit_i32_div_rem_by_zero_is_zero() {
        let Some(gpu) = try_gpu() else {
            eprintln!("skipping: no GPU available");
            return;
        };

        let a_data: Vec<i32> = vec![-7, 42, -100, i32::MIN, 99, -1, 0, 13];
        let b_data: Vec<i32> = vec![3, 0, 0, 0, -7, 1, 0, -5];
        let n = a_data.len();

        let a = gpu.field::<i32>(n).unwrap();
        let b = gpu.field::<i32>(n).unwrap();
        let q = gpu.field::<i32>(n).unwrap();
        let r = gpu.field::<i32>(n).unwrap();
        a.write(&a_data).unwrap();
        b.write(&b_data).unwrap();

        let bytes = quanta_ir::serialize_kernel(&div_rem_def(ScalarType::I32, "jit_div_rem_i32"));
        let mut wave = gpu.wave_jit(&bytes).unwrap();
        wave.bind(0, &a);
        wave.bind(1, &b);
        wave.bind(2, &q);
        wave.bind(3, &r);
        let mut p = gpu.dispatch(&wave, n as u32).unwrap();
        p.wait().unwrap();

        let got_q = q.read().unwrap();
        let got_r = r.read().unwrap();
        for i in 0..n {
            let want_q = if b_data[i] == 0 {
                0
            } else {
                a_data[i].wrapping_div(b_data[i])
            };
            let want_r = if b_data[i] == 0 {
                0
            } else {
                a_data[i].wrapping_rem(b_data[i])
            };
            assert_eq!(
                got_q[i], want_q,
                "jit i32 {} / {} must be {want_q}, got {}",
                a_data[i], b_data[i], got_q[i]
            );
            assert_eq!(
                got_r[i], want_r,
                "jit i32 {} % {} must be {want_r}, got {}",
                a_data[i], b_data[i], got_r[i]
            );
        }
    }
}

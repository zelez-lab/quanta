//! CPU-vs-GPU differential probe for the CNN forward-pass op chain.
//!
//! Runs every op the MNIST CNN forward pass uses — conv2d (im2col), relu,
//! maxpool2d, flatten, matmul+bias, and the log_softmax decomposition
//! (max_axis_last, broadcast_to+contiguous, sub, exp, sum_axis, log,
//! gather_rows) — on BOTH `quanta::init_cpu()` and `quanta::init()` (the
//! first real device: Vulkan on the Pi, Metal on a Mac), and compares
//! element-wise. Prints the first diverging op, or PASS.
//!
//! Sizes are chosen so every intermediate has well over 64 elements: the
//! Vulkan JIT used to hard-code `Wave.workgroup_size = [64,1,1]` while the
//! quanta-array kernels compile with LocalSize 1, so `dispatch(n)` launched
//! only ⌈n/64⌉ of n threads and left 63/64 of each output zeroed. Anything
//! bigger than one workgroup catches that class of bug.
//!
//! Run (Pi, Vulkan):
//!   QUANTA_NO_DOWNLOAD=1 cargo run --release --example vulkan_cpu_diff \
//!       -p quanta-autograd --features quanta-autograd/vulkan
//! Force lavapipe:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json <same command>

use quanta_array::Array;
use quanta_autograd::Tape;

/// Deterministic pseudo-random values in [-1, 1].
fn vals(count: usize, phase: f32) -> Vec<f32> {
    (0..count)
        .map(|i| ((i as f32) * 0.7311 + phase).sin())
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

struct Probe {
    cpu: quanta::Gpu,
    dev: quanta::Gpu,
    failures: usize,
}

impl Probe {
    /// Run `op` on both backends and compare the flattened f32 output.
    fn check<F>(&mut self, name: &str, tol: f32, op: F)
    where
        F: Fn(&quanta::Gpu) -> Vec<f32>,
    {
        let a = op(&self.cpu);
        let b = op(&self.dev);
        let d = max_abs_diff(&a, &b);
        let ok = d <= tol;
        if !ok {
            self.failures += 1;
        }
        println!(
            "{:<28} n={:<7} max|Δ|={:<12.6e} {}",
            name,
            a.len(),
            d,
            if ok { "OK" } else { "DIVERGES" }
        );
    }
}

fn main() {
    let cpu = quanta::init_cpu();
    let dev = quanta::init().expect("no GPU device (Vulkan/Metal)");
    println!("cpu backend vs first real device\n");
    let mut p = Probe {
        cpu,
        dev,
        failures: 0,
    };

    // Shapes: x [N,1,H,W] with N·H·W ≫ 64; conv 8 filters 3×3 pad 1;
    // pool 2×2/2; logits [N,10].
    let (n, h, w, cout) = (4usize, 12usize, 12usize, 8usize);
    let x_data = vals(n * h * w, 0.0);
    let wc_data = vals(cout * 9, 1.0);
    let flat = cout * (h / 2) * (w / 2);
    let wl_data = vals(flat * 10, 2.0);
    let bl_data = vals(10, 3.0);
    let labels_data: Vec<u32> = (0..n as u32).map(|i| (i * 3) % 10).collect();

    // ── Array-level single ops ──────────────────────────────────────────
    p.check("relu (maximum w/ 0)", 0.0, |g| {
        let a = Array::from_slice(g, &x_data, &[n * h * w]).unwrap();
        let z = Array::zeros(g, &[n * h * w]).unwrap();
        a.maximum(&z).unwrap().to_vec().unwrap()
    });
    p.check("exp", 1e-6, |g| {
        let a = Array::from_slice(g, &x_data, &[n * h * w]).unwrap();
        a.exp().unwrap().to_vec().unwrap()
    });
    p.check("log(|x|+1)", 1e-6, |g| {
        let a = Array::from_slice(g, &x_data, &[n * h * w]).unwrap();
        let one = Array::full(g, 1.0f32, &[n * h * w]).unwrap();
        a.abs()
            .unwrap()
            .add(&one)
            .unwrap()
            .log()
            .unwrap()
            .to_vec()
            .unwrap()
    });
    p.check("sub (broadcast [N,1])", 0.0, |g| {
        let a = Array::from_slice(g, &wl_data[..n * 10], &[n, 10]).unwrap();
        let b = Array::from_slice(g, &bl_data[..n], &[n, 1]).unwrap();
        a.sub(&b.broadcast_to(&[n, 10]).unwrap())
            .unwrap()
            .to_vec()
            .unwrap()
    });
    p.check("broadcast_to+contiguous", 0.0, |g| {
        let a = Array::from_slice(g, &bl_data[..n], &[n, 1]).unwrap();
        a.broadcast_to(&[n, 40])
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec()
            .unwrap()
    });
    p.check("sum_axis(1)", 1e-5, |g| {
        let a = Array::from_slice(g, &x_data[..n * 36], &[n, 36]).unwrap();
        a.sum_axis(1).unwrap().to_vec().unwrap()
    });
    p.check("max_axis_last", 0.0, |g| {
        let a = Array::from_slice(g, &x_data[..n * 36], &[n, 36]).unwrap();
        a.max_axis_last().unwrap().to_vec().unwrap()
    });
    p.check("argmax_last (as f32)", 0.0, |g| {
        let a = Array::from_slice(g, &x_data[..n * 36], &[n, 36]).unwrap();
        a.argmax_last()
            .unwrap()
            .to_vec()
            .unwrap()
            .iter()
            .map(|&u| u as f32)
            .collect()
    });
    p.check("gather_rows", 0.0, |g| {
        let a = Array::from_slice(g, &wl_data[..n * 10], &[n, 10]).unwrap();
        let idx = Array::from_slice(g, &labels_data, &[n]).unwrap();
        a.gather_rows(&idx).unwrap().to_vec().unwrap()
    });
    p.check("im2col", 0.0, |g| {
        let a = Array::from_slice(g, &x_data, &[n, 1, h, w]).unwrap();
        a.im2col(3, 3, 1, 1).unwrap().to_vec().unwrap()
    });
    // Tolerance: the branchless running max is `acc·(1−gt) + v·gt`, whose
    // rounding differs by a ULP across backends.
    p.check("maxpool2d", 1e-6, |g| {
        let a = Array::from_slice(g, &x_data, &[n, 1, h, w]).unwrap();
        a.maxpool2d(2, 2, 2, 0).unwrap().0.to_vec().unwrap()
    });
    p.check("avgpool2d", 1e-6, |g| {
        let a = Array::from_slice(g, &x_data, &[n, 1, h, w]).unwrap();
        a.avgpool2d(2, 2, 2, 0).unwrap().to_vec().unwrap()
    });
    p.check("matmul", 1e-4, |g| {
        let a = Array::from_slice(g, &x_data[..n * 36], &[n, 36]).unwrap();
        let b = Array::from_slice(g, &wl_data[..360], &[36, 10]).unwrap();
        a.matmul(&b).unwrap().to_vec().unwrap()
    });

    // ── Autograd composites ─────────────────────────────────────────────
    p.check("log_softmax", 1e-5, |g| {
        let tape = Tape::<f32>::new();
        let a = tape.var(Array::from_slice(g, &wl_data[..n * 10], &[n, 10]).unwrap());
        a.log_softmax().unwrap().value().to_vec().unwrap()
    });
    p.check("CNN fwd + cross_entropy", 1e-4, |g| {
        let tape = Tape::<f32>::new();
        let xv = tape.var(Array::from_slice(g, &x_data, &[n, 1, h, w]).unwrap());
        let wcv = tape.var(Array::from_slice(g, &wc_data, &[cout, 1, 3, 3]).unwrap());
        let wlv = tape.var(Array::from_slice(g, &wl_data, &[flat, 10]).unwrap());
        let blv = tape.var(Array::from_slice(g, &bl_data, &[1, 10]).unwrap());
        let labels = Array::from_slice(g, &labels_data, &[n]).unwrap();
        let logits = xv
            .conv2d(&wcv, 1, 1)
            .unwrap()
            .relu()
            .unwrap()
            .maxpool2d(2, 2, 2, 0)
            .unwrap()
            .flatten()
            .unwrap()
            .matmul(&wlv)
            .unwrap()
            .add(&blv)
            .unwrap();
        let loss = logits.cross_entropy(&labels).unwrap();
        loss.value().to_vec().unwrap()
    });

    println!();
    if p.failures == 0 {
        println!("PASS — all ops agree between CPU and device backends");
    } else {
        println!("FAIL — {} op(s) diverge", p.failures);
        std::process::exit(1);
    }
}

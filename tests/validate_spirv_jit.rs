#![cfg(feature = "compute")]
//! JIT SPIR-V validation via spirv-val.
//!
//! The wave_jit SPIR-V emitter's output for CONTROL-FLOW kernels never
//! passed through a validator: the op-matrix and differential kernels
//! are mostly straight-line, so the wasm-bool/`%bool` seam — a Cmp
//! result spilled through a `Copy`, a `Branch` condition loaded back
//! from a register slot — shipped invalid modules for years. Real
//! drivers (Iris Xe, Apple) tolerated them; lavapipe's LLVM segfaulted
//! compiling them, which is how the seam surfaced. These are the three
//! quanta-bench kernels — the reproducers, covering both shapes — put
//! through the exact emitter path `wave_jit` uses, then `spirv-val`.
//!
//! Mirrors `validate_spirv.rs` (the AOT twin's validator): self-skips
//! with a notice when spirv-val is absent.

use std::io::Write;
use std::process::{Command, Stdio};

const SPIRV_VAL: &str = "/opt/homebrew/bin/spirv-val";

// The `for` loop shape: the loop-exit condition round-trips through a
// demoted register slot, so the Branch arm must materialize `%bool`
// from the loaded uint (`!= 0`), never branch on the raw value.
#[quanta::kernel(jit)]
fn heavy_compute(input: &[f32], output: &mut [f32]) {
    let i = quark_id();
    let mut x = input[i];
    for _ in 0..1000 {
        x = (x.sin() * x.cos()) + (x.abs() + 1.0f32).sqrt();
    }
    output[i] = x;
}

// Straight-line control: the always-valid baseline.
#[quanta::kernel(jit)]
fn add_one(data: &mut [f32]) {
    let i = quark_id();
    data[i] = data[i] + 1.0;
}

// The `while` + `&&` shape: a Cmp result (`%bool`) flows through a
// `Copy` whose IR label says U32 (wasm typing) into a uint slot — the
// Copy arm must bridge from the register's ACTUAL type (OpSelect 0/1),
// not trust the label.
#[quanta::kernel(jit)]
fn mandelbrot(output: &mut [u32], width: u32, height: u32) {
    let idx = quark_id();
    let px = idx % width;
    let py = idx / width;
    let x0 = (px as f32 / width as f32) * 3.5f32 - 2.5f32;
    let y0 = (py as f32 / height as f32) * 2.0f32 - 1.0f32;
    let (mut x, mut y) = (0.0f32, 0.0f32);
    let mut iter = 0u32;
    while x * x + y * y <= 4.0f32 && iter < 1000u32 {
        let tmp = x * x - y * y + x0;
        y = 2.0f32 * x * y + y0;
        x = tmp;
        iter += 1u32;
    }
    output[idx] = iter;
}

// `subgroup_size()` reads the `SubgroupSize` builtin: a scalar `Input`
// variable that must carry the `GroupNonUniform` capability and sit on
// the entry-point interface. It was a constant 32 for years — valid
// SPIR-V, wrong number — so this pins the module SHAPE, and the
// `subgroup_size_is_loaded_not_constant` test pins the semantics.
#[quanta::kernel(jit)]
fn report_subgroup_size(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = unsafe { subgroup_size() };
}

fn spirv_val(label: &str, words: &[u8]) {
    if !std::path::Path::new(SPIRV_VAL).exists() {
        eprintln!("skipping [{label}]: {SPIRV_VAL} not installed");
        return;
    }
    let mut child = Command::new(SPIRV_VAL)
        .args(["--target-env", "vulkan1.3", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn spirv-val");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(words)
        .expect("write spirv-val stdin");
    let out = child.wait_with_output().expect("spirv-val run");
    assert!(
        out.status.success(),
        "[{label}] invalid JIT SPIR-V:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn emit(label: &str, def: &[u8]) -> Vec<u8> {
    let kernel = quanta_ir::deserialize_kernel(def).expect("deserialize KernelDef");
    quanta_ir::emit_spirv::emit(&kernel)
        .unwrap_or_else(|e| panic!("[{label}] JIT SPIR-V emission failed: {e}"))
}

#[test]
fn jit_spirv_validates_loop_kernels() {
    for (label, def) in [
        ("heavy_compute", HEAVY_COMPUTE_DEF),
        ("add_one", ADD_ONE_DEF),
        ("mandelbrot", MANDELBROT_DEF),
        ("report_subgroup_size", REPORT_SUBGROUP_SIZE_DEF),
    ] {
        let words = emit(label, def);
        spirv_val(label, &words);
    }
}

/// The `SubgroupSize` read must be an `OpLoad` from a `BuiltIn 36`
/// `Input` variable, under the `GroupNonUniform` capability — never a
/// literal. Checked on the raw words so it holds with or without
/// spirv-val installed.
#[test]
fn subgroup_size_is_loaded_not_constant() {
    let bytes = emit("report_subgroup_size", REPORT_SUBGROUP_SIZE_DEF);
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    const OP_CAPABILITY: u32 = 17;
    const OP_DECORATE: u32 = 71;
    const OP_VARIABLE: u32 = 59;
    const CAPABILITY_GROUP_NON_UNIFORM: u32 = 61;
    const DECORATION_BUILTIN: u32 = 11;
    const BUILTIN_SUBGROUP_SIZE: u32 = 36;
    const STORAGE_CLASS_INPUT: u32 = 1;
    let (mut has_cap, mut builtin_var) = (false, None);
    let mut i = 5; // past the header
    while i < words.len() {
        let (wc, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        let ops = &words[i + 1..i + wc];
        match op {
            OP_CAPABILITY if ops == [CAPABILITY_GROUP_NON_UNIFORM] => has_cap = true,
            OP_DECORATE
                if ops.len() == 3
                    && ops[1] == DECORATION_BUILTIN
                    && ops[2] == BUILTIN_SUBGROUP_SIZE =>
            {
                builtin_var = Some(ops[0]);
            }
            _ => {}
        }
        i += wc;
    }
    assert!(has_cap, "GroupNonUniform capability missing");
    let var = builtin_var.expect("no variable decorated BuiltIn SubgroupSize");
    // The decorated id must be an `Input`-class OpVariable.
    let mut i = 5;
    let mut is_input_var = false;
    while i < words.len() {
        let (wc, op) = ((words[i] >> 16) as usize, words[i] & 0xffff);
        if op == OP_VARIABLE && words[i + 2] == var && words[i + 3] == STORAGE_CLASS_INPUT {
            is_input_var = true;
        }
        i += wc;
    }
    assert!(
        is_input_var,
        "BuiltIn SubgroupSize id {var} is not an Input OpVariable"
    );
}

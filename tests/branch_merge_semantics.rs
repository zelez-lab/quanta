//! Branch-merge mutable-register semantics — the CPU backend as oracle.
//!
//! The KernelOp contract is mutable-register semantics: a register written
//! inside a `Branch` arm (backed by a dominating entry `Const` init) must be
//! observed after the merge with whichever value the *taken* path produced.
//! The SPIR-V emitters used to model registers as pure SSA renames, so
//! `let idx = if c { i } else { 0 }` silently yielded the value of whichever
//! arm was *emitted last* for every thread on the Vulkan lane — it passed
//! validation and computed wrong results.
//!
//! The CPU interpreter implements mutable registers correctly, so this test
//! pins the IR-level semantics every backend (including the fixed SPIR-V
//! variable-demotion model) must match. The same select shape is pinned
//! structurally for SPIR-V in
//! `crates/quanta-ir/tests/emit_spirv_int_signedness.rs`
//! (`branch_arm_write_read_after_merge_is_valid_spirv`).
//!
//! Run: cargo test --test branch_merge_semantics --features software

#![cfg(feature = "software")]

/// `idx = if i < n { i } else { 999 }` — the two arms differ, and both are
/// exercised across the dispatch. A merge that keeps the last-emitted arm
/// would return 999 (or `i`) for EVERY element.
#[quanta::kernel(workgroup = [4])]
fn select_shape(out: &mut [u32], n: u32) {
    let i = quark_id();
    let idx: u32 = if i < n { i } else { 999u32 };
    out[i as usize] = idx;
}

#[test]
fn select_shape_keeps_taken_arm_value_per_element() {
    let gpu = quanta::init_cpu();
    let total = 8usize;
    let n = 5u32;

    let out = gpu.field::<u32>(total).unwrap();
    out.write(&vec![u32::MAX; total]).unwrap();

    let mut wave = select_shape(&gpu).unwrap();
    wave.bind(0, &out);
    wave.set_value(1, n);
    gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();

    let got = out.read().unwrap();
    let want: Vec<u32> = (0..total as u32)
        .map(|i| if i < n { i } else { 999 })
        .collect();
    assert_eq!(
        got, want,
        "post-merge read must observe the taken arm's value per element \
         (then-arm for i<{n}, else-arm otherwise)"
    );
}

/// Loop-carried accumulator read after the loop: the sibling shape (a
/// register written in a Loop body, read past the merge). The accumulator
/// must reflect all iterations, not the pre-loop init.
#[quanta::kernel(workgroup = [4])]
fn loop_carried_sum(out: &mut [u32], k: u32) {
    let i = quark_id();
    let mut acc: u32 = i;
    let mut j: u32 = 0u32;
    while j < k {
        acc = acc + 2u32;
        j = j + 1u32;
    }
    out[i as usize] = acc;
}

#[test]
fn loop_carried_register_read_after_loop() {
    let gpu = quanta::init_cpu();
    let total = 8usize;
    let k = 6u32;

    let out = gpu.field::<u32>(total).unwrap();
    out.write(&vec![0; total]).unwrap();

    let mut wave = loop_carried_sum(&gpu).unwrap();
    wave.bind(0, &out);
    wave.set_value(1, k);
    gpu.dispatch(&wave, total as u32).unwrap().wait().unwrap();

    let got = out.read().unwrap();
    let want: Vec<u32> = (0..total as u32).map(|i| i + 2 * k).collect();
    assert_eq!(got, want, "loop-carried accumulator lost past the merge");
}

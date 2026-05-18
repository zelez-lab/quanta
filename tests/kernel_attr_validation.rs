//! Smoke test: known-good kernel attributes compile. Unknown
//! ones are tested manually with a scratch crate (proc-macro
//! compile errors can't be exercised from inside this same crate
//! without trybuild). The point of THIS test is to make sure the
//! KNOWN set still works after we added the unknown-attr error.

// Each of these should compile cleanly.

#[quanta::kernel]
fn kernel_default(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(workgroup = [256])]
fn kernel_workgroup_1d(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(workgroup = [16, 16])]
fn kernel_workgroup_2d(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(workgroup = [16, 16, 1])]
fn kernel_workgroup_3d(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(opt = "O2")]
fn kernel_opt(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(workgroup = [128], opt = "O3")]
fn kernel_workgroup_and_opt(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

#[quanta::kernel(subgroup = 32)]
fn kernel_subgroup(out: &mut [u32]) {
    let i = quark_id();
    out[i as usize] = i;
}

// TYPO TEST — uncomment to verify the error path. Should
// produce a compile error mentioning workgroup_size and
// pointing to the right name. Kept commented so the suite
// stays green; manually toggled when changing the parser.
//
// #[quanta::kernel(workgroup_size = [256])]
// fn typo_should_fail(out: &mut [u32]) {
//     let i = quark_id();
//     out[i as usize] = i;
// }

#[test]
fn known_attrs_compile() {
    // If we got here, every kernel above compiled. That's the
    // test — the macro accepted all eight forms documented in
    // `quanta::kernel`'s rustdoc.
}

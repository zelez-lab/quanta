//! Cross-crate `#[quanta::device]` import test.
//!
//! This crate is *separate* from `quanta-rand` — it exercises the
//! "downstream user" path. Two flavors, both end-to-end bit-exact:
//!
//! 1. **Explicit `import_devices!`**: user writes one line at file
//!    scope listing the imports, then calls the device fn by bare
//!    name in the kernel body.
//! 2. **Auto-discovery**: user just writes `quanta_rand::foo(...)`
//!    inside the kernel body. The kernel macro detects the
//!    qualified call, rewrites it to bare-name, and emits the
//!    matching `_src!()` invocation automatically.

#[derive(quanta_compute_dsl::Fields)]
pub struct ImportTestData {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

// Flavor 1: explicit `import_devices!` + bare call site.
quanta_compute_dsl::import_devices!(quanta_rand::philox4x32_10_first_u32_kernel);

#[quanta_compute_dsl::kernel(crate = quanta_core)]
pub fn import_test_kernel(d: &ImportTestData) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}

// Flavor 2: auto-discovery — qualified path in body, no import line.
#[quanta_compute_dsl::kernel(crate = quanta_core)]
pub fn auto_discover_test_kernel(d: &ImportTestData) {
    let id = quark_id();
    let r: u32 =
        quanta_rand::philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}

// ── Cross-crate device-fn body resolution probe ───────────────────
//
// `block_reduce_add_u32_kernel` is a `#[quanta::device]` from
// `quanta-prims`. Its body uses several GPU intrinsics
// (`reduce_add_u32`, `subgroup_size`, `proton_id`, `shared_store_u32`,
// `barrier`, `shared_load_u32`). When `_src!()` expands the device-fn
// body at this downstream crate's compile site, those bare-name calls
// must resolve to *something* the Rust toolchain can typecheck —
// otherwise `cargo check` fails with `E0425: cannot find function`.
// The `__device_host_stubs` glob injected by the `_src!` macro covers
// every intrinsic so any device-fn body parses and resolves on host.
//
// A single `import_devices!` at file scope is enough to exercise the
// macro expansion; we don't need to dispatch the device fn to verify
// host-side resolution.
quanta_compute_dsl::import_devices!(quanta_prims::block_reduce_add_u32_kernel);

#[cfg(test)]
mod tests {
    use super::*;
    use quanta_rand::philox4x32::philox4x32_10_first_u32;

    fn run(
        kernel: impl Fn(
            &quanta::Gpu,
            &mut ImportTestData,
            u32,
        ) -> Result<quanta::Pulse, quanta::QuantaError>,
    ) -> Vec<u32> {
        let gpu = quanta::init_cpu();
        let count: usize = 32;
        let seed_lo: u32 = 0xDEAD_BEEF;
        let seed_hi: u32 = 0xCAFE_BABE;

        let mut data = ImportTestData {
            out: vec![0u32; count],
            seed_lo,
            seed_hi,
        };
        kernel(&gpu, &mut data, count as u32)
            .expect("dispatch")
            .wait()
            .expect("wait");
        data.out
    }

    fn host_expected() -> Vec<u32> {
        let seed_lo: u32 = 0xDEAD_BEEF;
        let seed_hi: u32 = 0xCAFE_BABE;
        (0..32u32)
            .map(|id| philox4x32_10_first_u32(id, 0, 0, 0, seed_lo, seed_hi))
            .collect()
    }

    #[test]
    fn cross_crate_device_fn_round_trip_via_import_devices() {
        let out = run(import_test_kernel);
        assert_eq!(
            out,
            host_expected(),
            "explicit-import variant must match host reference"
        );
    }

    #[test]
    fn cross_crate_device_fn_round_trip_via_auto_discovery() {
        let out = run(auto_discover_test_kernel);
        assert_eq!(
            out,
            host_expected(),
            "auto-discovery variant must match host reference"
        );
    }
}

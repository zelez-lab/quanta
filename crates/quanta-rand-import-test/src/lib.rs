//! Cross-crate `#[quanta::device]` import test.
//!
//! This crate is *separate* from `quanta-rand` — it exercises the
//! "downstream user" path: invoke the auto-generated `_src!()`
//! macro at file scope to splice quanta-rand's device-fn source
//! into the current crate's macro process, then call the fn
//! unqualified from inside a `#[quanta::kernel]`.

// Splice quanta-rand's Philox4x32 device-fn source into this
// crate's macro process. After this line, a `#[quanta::kernel]`
// can call `philox4x32_10_first_u32_kernel(...)` as if the
// function were defined locally.
quanta_rand::philox4x32_10_first_u32_kernel_src!();

#[derive(quanta::Fields)]
pub struct ImportTestData {
    pub out: Vec<u32>,
    pub seed_lo: u32,
    pub seed_hi: u32,
}

#[quanta::kernel]
pub fn import_test_kernel(d: &ImportTestData) {
    let id = quark_id();
    let r: u32 = philox4x32_10_first_u32_kernel(id, 0u32, 0u32, 0u32, d.seed_lo, d.seed_hi);
    d.out[id as usize] = r;
}

#[cfg(test)]
mod tests {
    use super::*;
    use quanta_rand::philox4x32::philox4x32_10_first_u32;

    #[test]
    fn cross_crate_device_fn_round_trip() {
        let gpu = quanta::init_cpu();
        let count: usize = 32;
        let seed_lo: u32 = 0xDEAD_BEEF;
        let seed_hi: u32 = 0xCAFE_BABE;

        let mut data = ImportTestData {
            out: vec![0u32; count],
            seed_lo,
            seed_hi,
        };
        import_test_kernel(&gpu, &mut data, count as u32)
            .expect("dispatch")
            .wait()
            .expect("wait");

        // Host reference: the same algorithm, called via
        // quanta-rand's public host-side API.
        let expected: Vec<u32> = (0..count as u32)
            .map(|id| philox4x32_10_first_u32(id, 0, 0, 0, seed_lo, seed_hi))
            .collect();

        assert_eq!(
            data.out, expected,
            "cross-crate device-fn import must produce bit-exact host↔GPU output"
        );
    }
}

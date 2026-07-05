//! Minimal SPIR-V metadata probing — read a compute module's declared
//! `LocalSize` (workgroup size) without a full parser.
//!
//! The Vulkan driver needs the *actual* workgroup size baked into a shader
//! to convert a thread-count dispatch into a group count
//! (`Device::wave_dispatch_threads` computes
//! `groups = quarks.div_ceil(wave.workgroup_size[0])`). Guessing a default
//! (the old hard-coded `[64, 1, 1]`) silently under-dispatches any kernel
//! whose real `OpExecutionMode … LocalSize` differs: a kernel compiled with
//! LocalSize 1 but dispatched as if 64 runs only ⌈n/64⌉ of its n threads and
//! leaves the rest of the output buffer untouched. That is exactly what broke
//! every `quanta-array` JIT kernel (workgroup_size `[1,1,1]`) on Vulkan while
//! CPU / Metal — whose JIT paths carry the KernelDef's true size — stayed
//! correct.

/// Scan a SPIR-V module (as words, header included) for
/// `OpExecutionMode <entry> LocalSize x y z` and return `[x, y, z]`.
///
/// Returns `None` for malformed modules or modules that don't declare a
/// literal LocalSize (e.g. `LocalSizeId`, which our emitters never produce) —
/// callers should fall back to a conservative default.
pub(crate) fn local_size(words: &[u32]) -> Option<[u32; 3]> {
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const OP_EXECUTION_MODE: u32 = 16;
    const MODE_LOCAL_SIZE: u32 = 17;
    // Header: magic, version, generator, bound, schema.
    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return None;
    }
    let mut i = 5usize;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xFFFF;
        if word_count == 0 || i + word_count > words.len() {
            return None; // malformed stream — don't guess
        }
        // OpExecutionMode <entry-point-id> <mode> <literals…>
        if opcode == OP_EXECUTION_MODE && word_count == 6 && words[i + 2] == MODE_LOCAL_SIZE {
            return Some([words[i + 3], words[i + 4], words[i + 5]]);
        }
        i += word_count;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::local_size;

    /// Hand-assemble a minimal module: header + OpExecutionMode LocalSize.
    fn module_with_local_size(x: u32, y: u32, z: u32) -> alloc::vec::Vec<u32> {
        let mut w = alloc::vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
        // OpCapability Shader (2 words, opcode 17)
        w.extend_from_slice(&[(2 << 16) | 17, 1]);
        // OpExecutionMode %1 LocalSize x y z (6 words, opcode 16, mode 17)
        w.extend_from_slice(&[(6 << 16) | 16, 1, 17, x, y, z]);
        w
    }

    #[test]
    fn finds_local_size() {
        assert_eq!(
            local_size(&module_with_local_size(1, 1, 1)),
            Some([1, 1, 1])
        );
        assert_eq!(
            local_size(&module_with_local_size(64, 1, 1)),
            Some([64, 1, 1])
        );
        assert_eq!(
            local_size(&module_with_local_size(16, 16, 1)),
            Some([16, 16, 1])
        );
    }

    #[test]
    fn rejects_bad_magic_and_missing_mode() {
        assert_eq!(local_size(&[0xdead_beef, 0, 0, 0, 0]), None);
        // Header only — no instructions.
        assert_eq!(local_size(&[0x0723_0203, 0x0001_0300, 0, 10, 0]), None);
        // Zero word count must not loop forever.
        assert_eq!(local_size(&[0x0723_0203, 0, 0, 0, 0, 0]), None);
    }

    /// Round-trip through the real JIT emitter: the KernelDef's
    /// workgroup_size must be recoverable from the emitted module. This pins
    /// the contract the Vulkan `wave_impl` now relies on.
    #[cfg(feature = "jit")]
    #[test]
    fn roundtrips_emitted_kernels() {
        use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType};
        for wg in [[1u32, 1, 1], [64, 1, 1], [16, 16, 1], [256, 1, 1]] {
            let def = KernelDef {
                name: "wg_probe".into(),
                params: alloc::vec![KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 0,
                    scalar_type: ScalarType::U32,
                }],
                body: alloc::vec![
                    KernelOp::QuarkId { dst: Reg(0) },
                    KernelOp::Store {
                        field: 0,
                        index: Reg(0),
                        src: Reg(0),
                        ty: ScalarType::U32,
                    },
                ],
                body_source: None,
                next_reg: 1,
                opt_level: 0,
                device_sources: alloc::vec![],
                device_functions: alloc::vec![],
                workgroup_size: wg,
                subgroup_size: None,
                dynamic_shared_bytes: 0,
            };
            let spirv = quanta_ir::emit_spirv::emit(&def).expect("emit");
            let words: alloc::vec::Vec<u32> = spirv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(local_size(&words), Some(wg), "workgroup {:?}", wg);
        }
    }
}

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

/// The Vulkan descriptor type a shader binding resolves to. Buffers are
/// `StorageBuffer`; `&Texture2D` sampled slots are `SampledImage` (bound as a
/// combined image+sampler on the render path); `&mut Texture2D` storage slots
/// are `StorageImage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DescriptorKind {
    StorageBuffer,
    StorageImage,
    SampledImage,
}

/// Reflect the descriptor kind of every `(set = 0)` binding in a compute
/// module by walking the type/variable/decoration stream. The Vulkan pipeline
/// builds its descriptor-set layout from this so image slots get the right
/// descriptor type — the storage-buffer-only fallback stays correct for the
/// common (buffer-only) case, where this returns only `StorageBuffer` entries.
///
/// Result is ordered by binding number. Bindings without a resolvable kind
/// (or in a set other than 0, which our emitters never produce) are omitted.
/// Works identically for AOT and JIT SPIR-V — both share the emitter's
/// OpVariable/OpTypePointer/OpTypeImage encoding.
pub(crate) fn binding_kinds(words: &[u32]) -> alloc::vec::Vec<(u32, DescriptorKind)> {
    use alloc::collections::BTreeMap;
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const OP_DECORATE: u32 = 71;
    const OP_TYPE_IMAGE: u32 = 25;
    const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const DECORATION_BINDING: u32 = 33;
    const DECORATION_DESCRIPTOR_SET: u32 = 34;
    const STORAGE_CLASS_UNIFORM_CONSTANT: u32 = 0;

    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return alloc::vec::Vec::new();
    }

    // Per-result-id facts gathered in a single pass. SPIR-V requires types to
    // be declared before use, and decorations precede the function body, so a
    // forward scan resolves every reference by the time a variable needs it.
    #[derive(Clone, Copy)]
    enum TypeKind {
        Image,
        SampledImage,
        Other,
    }
    let mut type_kind: BTreeMap<u32, TypeKind> = BTreeMap::new();
    // pointer id → pointee type id
    let mut pointer_pointee: BTreeMap<u32, u32> = BTreeMap::new();
    // variable id → (pointer type id, storage class)
    let mut variables: alloc::vec::Vec<(u32, u32, u32)> = alloc::vec::Vec::new();
    // id → binding number, id → descriptor set
    let mut binding_of: BTreeMap<u32, u32> = BTreeMap::new();
    let mut set_of: BTreeMap<u32, u32> = BTreeMap::new();

    let mut i = 5usize;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xFFFF;
        if word_count == 0 || i + word_count > words.len() {
            return alloc::vec::Vec::new(); // malformed — don't guess
        }
        match opcode {
            OP_TYPE_IMAGE if word_count >= 2 => {
                type_kind.insert(words[i + 1], TypeKind::Image);
            }
            OP_TYPE_SAMPLED_IMAGE if word_count >= 2 => {
                type_kind.insert(words[i + 1], TypeKind::SampledImage);
            }
            OP_TYPE_POINTER if word_count >= 4 => {
                // OpTypePointer <result> <storage-class> <pointee-type>
                pointer_pointee.insert(words[i + 1], words[i + 3]);
                type_kind.insert(words[i + 1], TypeKind::Other);
            }
            OP_VARIABLE if word_count >= 4 => {
                // OpVariable <result-type(ptr)> <result-id> <storage-class>
                // stored as (ptr_type, var_id, storage_class).
                variables.push((words[i + 1], words[i + 2], words[i + 3]));
            }
            OP_DECORATE if word_count >= 4 => {
                let target = words[i + 1];
                match words[i + 2] {
                    DECORATION_BINDING => {
                        binding_of.insert(target, words[i + 3]);
                    }
                    DECORATION_DESCRIPTOR_SET => {
                        set_of.insert(target, words[i + 3]);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        i += word_count;
    }

    let mut out: BTreeMap<u32, DescriptorKind> = BTreeMap::new();
    for (ptr_ty, var_id, storage_class) in variables {
        // Only descriptor-backed variables carry a Binding decoration.
        let Some(&binding) = binding_of.get(&var_id) else {
            continue;
        };
        // Our emitters only ever use descriptor set 0; ignore anything else.
        if set_of.get(&var_id).copied().unwrap_or(0) != 0 {
            continue;
        }
        let pointee = pointer_pointee.get(&ptr_ty).copied();
        let kind = match pointee.and_then(|p| type_kind.get(&p).copied()) {
            Some(TypeKind::Image) => {
                // An image in UniformConstant storage is either a sampled or a
                // storage image; the emitter marks storage images `sampled=2`
                // and puts sampled images behind OpTypeSampledImage, so a bare
                // OpTypeImage variable is always a storage image.
                if storage_class == STORAGE_CLASS_UNIFORM_CONSTANT {
                    DescriptorKind::StorageImage
                } else {
                    continue;
                }
            }
            Some(TypeKind::SampledImage) => DescriptorKind::SampledImage,
            _ => DescriptorKind::StorageBuffer,
        };
        out.insert(binding, kind);
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::{DescriptorKind, binding_kinds, local_size};

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

    /// Reflect a mixed kernel: a storage buffer (slot 0), a storage image
    /// (`&mut Texture2D`, slot 1) and a sampled image (`&Texture2D`, slot 2)
    /// must resolve to the matching descriptor kinds so the Vulkan pipeline
    /// builds a correct layout for AOT and JIT SPIR-V alike.
    #[cfg(feature = "jit")]
    #[test]
    fn reflects_binding_kinds() {
        use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType};
        let def = KernelDef {
            name: "mixed".into(),
            params: alloc::vec![
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 0,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::Texture2DWrite {
                    name: "dst".into(),
                    slot: 1,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::Texture2DRead {
                    name: "src".into(),
                    slot: 2,
                    scalar_type: ScalarType::F32,
                },
            ],
            body: alloc::vec![
                KernelOp::QuarkId { dst: Reg(0) },
                // Load from the sampled image and write to the storage image so
                // both variables survive any dead-code elimination.
                KernelOp::TextureLoad2D {
                    dst: Reg(1),
                    texture: 2,
                    x: Reg(0),
                    y: Reg(0),
                    ty: ScalarType::F32,
                },
                KernelOp::TextureWrite2D {
                    texture: 1,
                    x: Reg(0),
                    y: Reg(0),
                    value: Reg(1),
                    ty: ScalarType::F32,
                },
                KernelOp::Store {
                    field: 0,
                    index: Reg(0),
                    src: Reg(1),
                    ty: ScalarType::F32,
                },
            ],
            body_source: None,
            next_reg: 2,
            opt_level: 0,
            device_sources: alloc::vec![],
            device_functions: alloc::vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let spirv = quanta_ir::emit_spirv::emit(&def).expect("emit");
        let words: alloc::vec::Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let kinds = binding_kinds(&words);
        assert_eq!(
            kinds,
            alloc::vec![
                (0, DescriptorKind::StorageBuffer),
                (1, DescriptorKind::StorageImage),
                (2, DescriptorKind::SampledImage),
            ],
            "reflected descriptor kinds"
        );
    }

    /// A buffer-only module reflects to storage buffers only — the layout
    /// fallback path must stay unchanged for the common case.
    #[cfg(feature = "jit")]
    #[test]
    fn reflects_buffer_only_kernel() {
        use quanta_ir::{KernelDef, KernelOp, KernelParam, Reg, ScalarType};
        let def = KernelDef {
            name: "buffers".into(),
            params: alloc::vec![
                KernelParam::FieldRead {
                    name: "a".into(),
                    slot: 0,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::FieldWrite {
                    name: "out".into(),
                    slot: 1,
                    scalar_type: ScalarType::F32,
                },
            ],
            body: alloc::vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 0,
                    index: Reg(0),
                    ty: ScalarType::F32,
                },
                KernelOp::Store {
                    field: 1,
                    index: Reg(0),
                    src: Reg(1),
                    ty: ScalarType::F32,
                },
            ],
            body_source: None,
            next_reg: 2,
            opt_level: 0,
            device_sources: alloc::vec![],
            device_functions: alloc::vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        };
        let spirv = quanta_ir::emit_spirv::emit(&def).expect("emit");
        let words: alloc::vec::Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let kinds = binding_kinds(&words);
        assert_eq!(
            kinds,
            alloc::vec![
                (0, DescriptorKind::StorageBuffer),
                (1, DescriptorKind::StorageBuffer),
            ],
        );
    }
}

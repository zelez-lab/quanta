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
// Consumed only by the Vulkan compute wave path; the render-only build
// compiles this module for `fragment_output_count` alone.
#[cfg_attr(not(all(feature = "vulkan", feature = "compute")), allow(dead_code))]
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
#[cfg_attr(not(all(feature = "vulkan", feature = "compute")), allow(dead_code))]
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
#[cfg_attr(not(all(feature = "vulkan", feature = "compute")), allow(dead_code))]
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

/// Count a fragment module's color outputs: `OpVariable`s in the
/// `Output` storage class that carry a `Location` decoration and appear
/// in the fragment entry point's interface. That is exactly the set of
/// `location = N` fragment outputs (`out_color` at `Location(0)`, plus
/// any further MRT outputs) — built-in outputs like `FragDepth` are
/// `BuiltIn`-decorated, never `Location`-decorated, so they are excluded.
///
/// Returns `None` for a malformed module or one with no `Fragment`
/// entry point (e.g. a vertex module) — callers should then skip the
/// check rather than assume zero. Works for AOT and JIT SPIR-V alike;
/// the emitter uses the same OpEntryPoint / Output / Location encoding.
// Consumed only by the render-path pipeline-creation check; a
// compute-only build compiles this module for the wave-size / binding
// reflectors instead.
#[cfg_attr(not(feature = "render"), allow(dead_code))]
pub(crate) fn fragment_output_count(words: &[u32]) -> Option<usize> {
    use alloc::collections::BTreeSet;
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const OP_ENTRY_POINT: u32 = 15;
    const OP_DECORATE: u32 = 71;
    const OP_VARIABLE: u32 = 59;
    const EXECUTION_MODEL_FRAGMENT: u32 = 4;
    const STORAGE_CLASS_OUTPUT: u32 = 3;
    const DECORATION_LOCATION: u32 = 30;

    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return None;
    }

    // Interface ids of the fragment entry point (empty until we find it).
    let mut fragment_interface: Option<BTreeSet<u32>> = None;
    // variable id → storage class (only Output ones matter).
    let mut output_vars: BTreeSet<u32> = BTreeSet::new();
    // ids carrying a Location decoration.
    let mut located: BTreeSet<u32> = BTreeSet::new();

    let mut i = 5usize;
    while i < words.len() {
        let word_count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xFFFF;
        if word_count == 0 || i + word_count > words.len() {
            return None; // malformed — don't guess
        }
        match opcode {
            OP_ENTRY_POINT if word_count >= 3 && words[i + 1] == EXECUTION_MODEL_FRAGMENT => {
                // OpEntryPoint <model> <entry-id> <name-literal…> <iface…>.
                // The name is a null-terminated UTF-8 literal packed into
                // words; skip it to reach the interface id list.
                let mut j = i + 3;
                let end = i + word_count;
                // Advance past the name: each word holds 4 bytes; the
                // string ends at the word containing a 0 byte.
                while j < end {
                    let w = words[j];
                    j += 1;
                    if w.to_le_bytes().contains(&0) {
                        break;
                    }
                }
                let mut iface = BTreeSet::new();
                while j < end {
                    iface.insert(words[j]);
                    j += 1;
                }
                fragment_interface = Some(iface);
            }
            // OpVariable <result-type> <result-id> <storage-class>
            OP_VARIABLE if word_count >= 4 && words[i + 3] == STORAGE_CLASS_OUTPUT => {
                output_vars.insert(words[i + 2]);
            }
            OP_DECORATE if word_count >= 4 && words[i + 2] == DECORATION_LOCATION => {
                located.insert(words[i + 1]);
            }
            _ => {}
        }
        i += word_count;
    }

    let interface = fragment_interface?;
    // Count Output variables that are Location-decorated and belong to
    // the fragment entry's interface.
    let count = output_vars
        .iter()
        .filter(|v| located.contains(v) && interface.contains(v))
        .count();
    Some(count)
}

#[cfg(test)]
mod tests {
    use super::{fragment_output_count, local_size};
    // `binding_kinds` / `DescriptorKind` are exercised only by the
    // jit-gated round-trip tests below.
    #[cfg(feature = "jit")]
    use super::{DescriptorKind, binding_kinds};

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

    /// Hand-assemble a minimal Fragment module with `outputs`
    /// Location-decorated `Output` variables in the entry interface,
    /// plus optionally one extra `Output` variable that is *not*
    /// Location-decorated (a built-in like FragDepth) to prove it is
    /// excluded. `wrong_model` emits a Vertex entry point instead, to
    /// prove non-fragment modules return `None`.
    fn fragment_module(
        outputs: u32,
        with_builtin: bool,
        wrong_model: bool,
    ) -> alloc::vec::Vec<u32> {
        const OP_ENTRY_POINT: u32 = 15;
        const OP_DECORATE: u32 = 71;
        const OP_VARIABLE: u32 = 59;
        const EXECUTION_MODEL_VERTEX: u32 = 0;
        const EXECUTION_MODEL_FRAGMENT: u32 = 4;
        const STORAGE_CLASS_OUTPUT: u32 = 3;
        const DECORATION_LOCATION: u32 = 30;
        const DECORATION_BUILT_IN: u32 = 11;

        // Header: magic, version 1.3, generator, bound, schema.
        let mut w = alloc::vec![0x0723_0203, 0x0001_0300, 0, 100, 0];
        // OpCapability Shader.
        w.extend_from_slice(&[(2 << 16) | 17, 1]);

        // Ids: %1 = entry function, output vars start at %10, builtin %20,
        // ptr type %2 (unused value, just an operand for OpVariable).
        let entry = 1u32;
        let ptr_ty = 2u32;
        let mut output_ids: alloc::vec::Vec<u32> = (0..outputs).map(|k| 10 + k).collect();
        let builtin_id = 20u32;
        if with_builtin {
            output_ids.push(builtin_id);
        }

        // OpEntryPoint <model> %entry "main" <iface…>.
        // "main" packs into two words: 'm','a','i','n', then 0 terminator.
        let name_w0 = u32::from_le_bytes([b'm', b'a', b'i', b'n']);
        let name_w1 = 0u32;
        let model = if wrong_model {
            EXECUTION_MODEL_VERTEX
        } else {
            EXECUTION_MODEL_FRAGMENT
        };
        let mut ep = alloc::vec![model, entry, name_w0, name_w1];
        ep.extend_from_slice(&output_ids);
        let ep_wc = (ep.len() + 1) as u32;
        w.push((ep_wc << 16) | OP_ENTRY_POINT);
        w.extend_from_slice(&ep);

        // Decorations: each real output gets Location k; the builtin gets
        // BuiltIn (not Location), so it must be excluded from the count.
        for (k, id) in output_ids.iter().enumerate() {
            if with_builtin && *id == builtin_id {
                w.extend_from_slice(&[(4 << 16) | OP_DECORATE, *id, DECORATION_BUILT_IN, 22]);
            } else {
                w.extend_from_slice(&[(4 << 16) | OP_DECORATE, *id, DECORATION_LOCATION, k as u32]);
            }
        }

        // OpVariable <ptr-ty> <id> Output for every output id.
        for id in &output_ids {
            w.extend_from_slice(&[(4 << 16) | OP_VARIABLE, ptr_ty, *id, STORAGE_CLASS_OUTPUT]);
        }
        w
    }

    #[test]
    fn counts_single_fragment_output() {
        // The DSL always emits exactly one Location(0) output.
        assert_eq!(
            fragment_output_count(&fragment_module(1, false, false)),
            Some(1)
        );
    }

    #[test]
    fn counts_multiple_fragment_outputs() {
        assert_eq!(
            fragment_output_count(&fragment_module(3, false, false)),
            Some(3)
        );
    }

    #[test]
    fn excludes_builtin_outputs() {
        // One located output + one BuiltIn output → count is 1.
        assert_eq!(
            fragment_output_count(&fragment_module(1, true, false)),
            Some(1)
        );
    }

    #[test]
    fn non_fragment_module_returns_none() {
        // A vertex entry point must not be counted as a fragment.
        assert_eq!(
            fragment_output_count(&fragment_module(1, false, true)),
            None
        );
    }

    #[test]
    fn rejects_bad_magic_fragment() {
        assert_eq!(fragment_output_count(&[0xdead_beef, 0, 0, 0, 0]), None);
        // Zero word count must not loop forever.
        assert_eq!(fragment_output_count(&[0x0723_0203, 0, 0, 0, 0, 0]), None);
    }
}

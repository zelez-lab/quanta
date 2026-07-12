//! KernelDef → Vulkan SPIR-V binary.
//!
//! Walks KernelOps and produces valid Vulkan SPIR-V binary (Shader capability,
//! GLCompute execution model, StorageBuffer storage class). This replaces the
//! LLVM spirv64 backend which emits OpenCL-style SPIR-V that Vulkan rejects.
//!
//! The output is a `Vec<u8>` ready for `vkCreateShaderModule`.

mod constants;
mod device_fn;
mod emitter;
mod expr;
mod expr_atom;
mod kernel;
mod kernel_params;
mod narrow_storage;
mod ops;
mod ops_ext;
mod ops_flow;
mod ops_helpers;
mod shader;
mod shader_frag;
mod tokenizer;
mod types;

use emitter::SpvEmitter;

/// Outcome of a shader-emission attempt run in real (non-passthrough) mode.
///
/// A body the translator cannot handle interns ids into the type / constant /
/// name sections before it fails; those ids survive a rollback of the function
/// section and the id counter, so rebuilding the passthrough over the same
/// emitter re-issues them and the module fails spirv-val ("Id defined more than
/// once"). The fix is structural: on `NeedsPassthrough` the caller discards the
/// poisoned emitter and rebuilds the module from a FRESH one in passthrough
/// mode, where the interface setup is identical and consistency is by
/// construction.
pub(crate) enum ShaderEmit {
    /// The body translated; the emitter holds a complete real module.
    Real,
    /// The body defeated the translator (the warning has already been logged);
    /// rebuild the module in passthrough mode on a fresh emitter.
    NeedsPassthrough,
}

/// Emit Vulkan SPIR-V binary from a KernelDef.
///
/// Returns the SPIR-V module as bytes, ready for `vkCreateShaderModule`.
pub fn emit(kernel: &quanta_ir::KernelDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    e.emit_kernel(kernel)?;
    Ok(e.finalize())
}

/// Emit SPIR-V for a vertex shader from a [`ShaderDef`].
pub fn emit_vertex(shader: &quanta_ir::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    match e.emit_vertex_shader(shader, false)? {
        ShaderEmit::Real => Ok(e.finalize()),
        // Discard the poisoned emitter; rebuild the passthrough on a clean one.
        ShaderEmit::NeedsPassthrough => {
            let mut e = SpvEmitter::new();
            e.emit_vertex_shader(shader, true)?;
            Ok(e.finalize())
        }
    }
}

/// Emit SPIR-V for a fragment shader from a [`ShaderDef`].
pub fn emit_fragment(shader: &quanta_ir::ShaderDef) -> Result<Vec<u8>, String> {
    let mut e = SpvEmitter::new();
    match e.emit_fragment_shader(shader, false)? {
        ShaderEmit::Real => Ok(e.finalize()),
        // Discard the poisoned emitter; rebuild the passthrough on a clean one.
        ShaderEmit::NeedsPassthrough => {
            let mut e = SpvEmitter::new();
            e.emit_fragment_shader(shader, true)?;
            Ok(e.finalize())
        }
    }
}

#[cfg(test)]
mod tests {
    //! Regression: 64-bit integer ops must be emitted at 64-bit width.
    //!
    //! This emitter used to collapse `ScalarType::U64`/`I64` to `%uint`, so
    //! the `fill_uniform_f64` pack — `((hi as u64) << 32) | lo`, `>> 11`,
    //! u64→f64 — shifted the high word out to zero at 32-bit width: every
    //! f64 draw came back ~0 on Vulkan (silent miscompile; the module was
    //! spirv-val-valid). Pins the JIT-emitter twin in
    //! `crates/quanta-ir/tests/emit_spirv_u64_width.rs`.

    use quanta_ir::{BinOp, ConstValue, KernelDef, KernelOp, KernelParam, Reg, ScalarType};

    fn assert_spirv_val(spirv: &[u8]) {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut child = match Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                eprintln!("spirv-val not found on PATH; skipping validation");
                return;
            }
        };
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(spirv)
            .expect("write spirv to spirv-val");
        let out = child.wait_with_output().expect("spirv-val run");
        assert!(
            out.status.success(),
            "spirv-val rejected the module:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// Decode the module into `(opcode, operands)` instructions.
    fn decode(spirv: &[u8]) -> Vec<(u16, Vec<u32>)> {
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut out = Vec::new();
        let mut i = 5; // skip header
        while i < words.len() {
            let word = words[i];
            let opcode = (word & 0xFFFF) as u16;
            let count = (word >> 16) as usize;
            assert!(count > 0, "malformed SPIR-V instruction at word {i}");
            out.push((opcode, words[i + 1..i + count].to_vec()));
            i += count;
        }
        out
    }

    const OP_CAPABILITY: u16 = 17;
    const OP_TYPE_INT: u16 = 21;
    const OP_TYPE_IMAGE: u16 = 25;
    const OP_IMAGE_READ: u16 = 98;
    const OP_IMAGE_WRITE: u16 = 99;
    const OP_IMAGE_FETCH: u16 = 95;
    const OP_SHIFT_RIGHT_LOGICAL: u16 = 194;
    const OP_SHIFT_LEFT_LOGICAL: u16 = 196;
    const OP_BITWISE_OR: u16 = 197;
    const CAPABILITY_INT64: u32 = 11;

    /// `texture_write_2d(tex, x, y, v)` into a `&mut Texture2D<f32>` storage
    /// image — the pure write path. Coords/x-value coercion must be valid or
    /// the OpCompositeConstruct fails; the image format word must be R32f.
    fn write_storage_kernel() -> KernelDef {
        KernelDef {
            name: "write_storage".into(),
            params: vec![
                KernelParam::Texture2DWrite {
                    name: "tex".into(),
                    slot: 0,
                    scalar_type: ScalarType::F32,
                },
                KernelParam::FieldRead {
                    name: "values".into(),
                    slot: 1,
                    scalar_type: ScalarType::F32,
                },
            ],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::Load {
                    dst: Reg(1),
                    field: 1,
                    index: Reg(0),
                    ty: ScalarType::F32,
                },
                KernelOp::TextureWrite2D {
                    texture: 0,
                    x: Reg(0),
                    y: Reg(0),
                    value: Reg(1),
                    ty: ScalarType::F32,
                },
            ],
            body_source: None,
            next_reg: 2,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    /// Read-modify-write the same storage texture: the load against a write
    /// slot must emit OpImageRead (a storage image cannot be OpImageFetch'd).
    fn rmw_storage_kernel() -> KernelDef {
        KernelDef {
            name: "rmw_storage".into(),
            params: vec![KernelParam::Texture2DWrite {
                name: "tex".into(),
                slot: 0,
                scalar_type: ScalarType::F32,
            }],
            body: vec![
                KernelOp::QuarkId { dst: Reg(0) },
                KernelOp::TextureLoad2D {
                    dst: Reg(1),
                    texture: 0,
                    x: Reg(0),
                    y: Reg(0),
                    ty: ScalarType::F32,
                },
                KernelOp::TextureWrite2D {
                    texture: 0,
                    x: Reg(0),
                    y: Reg(0),
                    value: Reg(1),
                    ty: ScalarType::F32,
                },
            ],
            body_source: None,
            next_reg: 2,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    #[test]
    fn storage_texture_write_is_valid_and_r32f() {
        let spirv = super::emit(&write_storage_kernel()).expect("emit");
        assert_spirv_val(&spirv);
        let instrs = decode(&spirv);
        // OpTypeImage %result %SampledType Dim Depth Arrayed MS Sampled Format
        let img = instrs
            .iter()
            .find(|(op, _)| *op == OP_TYPE_IMAGE)
            .map(|(_, args)| args.clone())
            .expect("module must declare an OpTypeImage");
        // args[0]=result, [1]=sampled_type, [2]=Dim, .., [6]=Sampled, [7]=Format
        assert_eq!(args_sampled(&img), 2, "storage image must be sampled=2");
        assert_eq!(
            args_format(&img),
            3,
            "Texture2D<f32> storage image must be R32f (3), not Rgba32f (1)"
        );
        assert!(
            instrs.iter().any(|(op, _)| *op == OP_IMAGE_WRITE),
            "write kernel must emit OpImageWrite"
        );
    }

    #[test]
    fn storage_texture_load_emits_image_read() {
        let spirv = super::emit(&rmw_storage_kernel()).expect("emit");
        assert_spirv_val(&spirv);
        let instrs = decode(&spirv);
        assert!(
            instrs.iter().any(|(op, _)| *op == OP_IMAGE_READ),
            "load against a storage slot must emit OpImageRead"
        );
        assert!(
            !instrs.iter().any(|(op, _)| *op == OP_IMAGE_FETCH),
            "a storage image must not be OpImageFetch'd"
        );
        assert!(
            instrs.iter().any(|(op, _)| *op == OP_IMAGE_WRITE),
            "rmw kernel must still emit OpImageWrite"
        );
    }

    // OpTypeImage operand accessors (result id is args[0]).
    fn args_sampled(args: &[u32]) -> u32 {
        args[6]
    }
    fn args_format(args: &[u32]) -> u32 {
        args[7]
    }

    /// The `fill_uniform_f64` pack shape:
    /// `out[i] = ((((hi as u64) << 32) | (lo as u64)) >> 11) as f64 * 2^-53`.
    fn u64_pack_kernel() -> KernelDef {
        let body = vec![
            KernelOp::QuarkId { dst: Reg(0) },
            KernelOp::Const {
                dst: Reg(1),
                value: ConstValue::U32(0x9E37_79B9),
            },
            KernelOp::Const {
                dst: Reg(2),
                value: ConstValue::U32(0x7F4A_7C15),
            },
            KernelOp::Cast {
                dst: Reg(3),
                src: Reg(1),
                from: ScalarType::U32,
                to: ScalarType::U64,
            },
            KernelOp::Cast {
                dst: Reg(4),
                src: Reg(2),
                from: ScalarType::U32,
                to: ScalarType::U64,
            },
            KernelOp::Const {
                dst: Reg(5),
                value: ConstValue::U64(32),
            },
            KernelOp::BinOp {
                dst: Reg(6),
                a: Reg(3),
                b: Reg(5),
                op: BinOp::Shl,
                ty: ScalarType::U64,
            },
            KernelOp::BinOp {
                dst: Reg(7),
                a: Reg(6),
                b: Reg(4),
                op: BinOp::BitOr,
                ty: ScalarType::U64,
            },
            KernelOp::Const {
                dst: Reg(8),
                value: ConstValue::U64(11),
            },
            KernelOp::BinOp {
                dst: Reg(9),
                a: Reg(7),
                b: Reg(8),
                op: BinOp::Shr,
                ty: ScalarType::U64,
            },
            KernelOp::Cast {
                dst: Reg(10),
                src: Reg(9),
                from: ScalarType::U64,
                to: ScalarType::F64,
            },
            KernelOp::Const {
                dst: Reg(11),
                value: ConstValue::F64(1.0 / 9_007_199_254_740_992.0),
            },
            KernelOp::BinOp {
                dst: Reg(12),
                a: Reg(10),
                b: Reg(11),
                op: BinOp::Mul,
                ty: ScalarType::F64,
            },
            KernelOp::Store {
                field: 0,
                index: Reg(0),
                src: Reg(12),
                ty: ScalarType::F64,
            },
        ];
        KernelDef {
            name: "u64_pack_to_f64".into(),
            params: vec![KernelParam::FieldWrite {
                name: "o".into(),
                slot: 0,
                scalar_type: ScalarType::F64,
            }],
            body,
            body_source: None,
            next_reg: 13,
            opt_level: 0,
            device_sources: vec![],
            device_functions: vec![],
            workgroup_size: [1, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    #[test]
    fn u64_pack_shape_is_valid_and_64_bit_wide() {
        let spirv = super::emit(&u64_pack_kernel()).expect("emit");
        assert_spirv_val(&spirv);

        let instrs = decode(&spirv);
        assert!(
            instrs
                .iter()
                .any(|(op, args)| *op == OP_CAPABILITY && args == &[CAPABILITY_INT64]),
            "module must declare OpCapability Int64"
        );
        let ulong: Vec<u32> = instrs
            .iter()
            .filter(|(op, args)| *op == OP_TYPE_INT && args.len() == 3 && args[1] == 64)
            .map(|(_, args)| args[0])
            .collect();
        assert!(!ulong.is_empty(), "module must declare a 64-bit int type");
        for (name, opcode) in [
            ("OpShiftLeftLogical", OP_SHIFT_LEFT_LOGICAL),
            ("OpShiftRightLogical", OP_SHIFT_RIGHT_LOGICAL),
            ("OpBitwiseOr", OP_BITWISE_OR),
        ] {
            let hits: Vec<&Vec<u32>> = instrs
                .iter()
                .filter(|(op, _)| *op == opcode)
                .map(|(_, args)| args)
                .collect();
            assert!(!hits.is_empty(), "expected at least one {name}");
            for args in hits {
                assert!(
                    ulong.contains(&args[0]),
                    "{name} must produce the 64-bit int type, got type id {} \
                     (the 32-bit collapse shifts the high word out to zero)",
                    args[0]
                );
            }
        }
    }
}

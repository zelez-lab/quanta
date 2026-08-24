//! Dynamic shared memory resolution (step 048).
//!
//! `SharedDeclDyn` declares a workgroup-shared array whose length is
//! not in the kernel source; the size arrives at **wave creation**
//! (`Gpu::wave_jit_shared`) and this pass late-binds it: the dynamic
//! declaration becomes an ordinary sized `SharedDecl`, so every
//! emitter, executor and dispatch path downstream works unchanged.
//!
//! JIT-only by construction: the AOT route ships pre-emitted backend
//! artifacts with no `KernelDesc` alongside, so there is nowhere for a
//! late size to act — the kernel macro enforces `jit` on any kernel
//! that declares `#[quanta::shared(dyn)]`, and the validator keeps
//! refusing an unresolved `SharedDeclDyn` (a dyn-shared kernel
//! dispatched without a size).

use crate::types::{KernelDef, KernelOp, ScalarType};

/// Bind `bytes` of workgroup memory to the kernel's dynamic shared
/// declaration, rewriting it into a sized `SharedDecl`.
///
/// Rules, each a hard error:
/// - exactly one `SharedDeclDyn` (the CUDA model: one extern block);
/// - `bytes` non-zero and a multiple of the element width;
/// - the element type is one of the 4-byte scalars the shared
///   load/store surface speaks (`f32` / `u32` / `i32`).
pub fn resolve_dynamic_shared(def: &mut KernelDef, bytes: u32) -> Result<(), String> {
    let dyn_count = def
        .body
        .iter()
        .filter(|op| matches!(op, KernelOp::SharedDeclDyn { .. }))
        .count();
    match dyn_count {
        0 => {
            return Err(
                "kernel declares no dynamic shared memory (no `#[quanta::shared(dyn)]`)"
                    .to_string(),
            );
        }
        1 => {}
        n => {
            return Err(format!(
                "kernel declares {n} dynamic shared arrays; exactly one is supported \
                 (one size binds one block — split sized `#[quanta::shared]` arrays off it)"
            ));
        }
    }
    if bytes == 0 {
        return Err("dynamic shared size must be non-zero".to_string());
    }
    for op in &mut def.body {
        if let KernelOp::SharedDeclDyn { id, ty } = *op {
            let elem = match ty {
                ScalarType::F32 | ScalarType::U32 | ScalarType::I32 => 4u32,
                other => {
                    return Err(format!(
                        "dynamic shared element type {other:?} is not supported \
                         (the shared load/store surface speaks f32 / u32 / i32)"
                    ));
                }
            };
            if !bytes.is_multiple_of(elem) {
                return Err(format!(
                    "dynamic shared size {bytes} is not a multiple of the {elem}-byte element"
                ));
            }
            *op = KernelOp::SharedDecl {
                id,
                ty,
                count: bytes / elem,
            };
        }
    }
    def.dynamic_shared_bytes = bytes;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn def_with(body: Vec<KernelOp>) -> KernelDef {
        KernelDef {
            name: String::from("t"),
            params: Vec::new(),
            body,
            body_source: None,
            next_reg: 0,
            opt_level: 3,
            device_sources: Vec::new(),
            device_functions: Vec::new(),
            workgroup_size: [64, 1, 1],
            subgroup_size: None,
            dynamic_shared_bytes: 0,
        }
    }

    #[test]
    fn resolves_one_dyn_decl() {
        let mut def = def_with(vec![KernelOp::SharedDeclDyn {
            id: 0,
            ty: ScalarType::F32,
        }]);
        resolve_dynamic_shared(&mut def, 256).unwrap();
        assert_eq!(def.dynamic_shared_bytes, 256);
        assert!(matches!(
            def.body[0],
            KernelOp::SharedDecl {
                id: 0,
                ty: ScalarType::F32,
                count: 64
            }
        ));
    }

    #[test]
    fn rejects_zero_none_many_and_misaligned() {
        let dyn_op = KernelOp::SharedDeclDyn {
            id: 0,
            ty: ScalarType::F32,
        };
        assert!(resolve_dynamic_shared(&mut def_with(vec![]), 64).is_err());
        assert!(resolve_dynamic_shared(&mut def_with(vec![dyn_op.clone()]), 0).is_err());
        assert!(resolve_dynamic_shared(&mut def_with(vec![dyn_op.clone()]), 6).is_err());
        assert!(
            resolve_dynamic_shared(
                &mut def_with(vec![
                    dyn_op.clone(),
                    KernelOp::SharedDeclDyn {
                        id: 1,
                        ty: ScalarType::U32
                    }
                ]),
                64
            )
            .is_err()
        );
    }
}
